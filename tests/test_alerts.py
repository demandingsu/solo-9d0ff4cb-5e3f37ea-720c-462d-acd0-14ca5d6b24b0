"""Regional heavy-rainfall alert API: rules, edge cases and snapshot safety.

Every test gets unique station ids (session data dir is shared) so published
data from one test can never leak into another.
"""
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from app import alerts, services
from app.errors import SnapshotReadError
from app.storage import file_manager
from tests.conftest import make_csv

DAY = "2026-11-12"


@pytest.fixture
def stations(request):
    tag = re.sub(r"[^A-Za-z0-9]", "", request.node.name).upper()[:44]
    return f"A{tag}A", f"A{tag}B"


def _upload_minutes(client, station, minute_values):
    rows = [(f"2026-11-12T08:{minute:02d}Z", rain) for minute, rain in minute_values]
    r = client.post(
        f"/stations/{station}/days/{DAY}/batches",
        files={"file": ("x.csv", make_csv(rows, station), "text/csv")},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _upload_hhmm(client, station, start_min, stop_min, rain=1.0, base_hour=8):
    base = datetime(2026, 11, 12, base_hour, 0, tzinfo=timezone.utc)
    rows = [
        ((base + timedelta(minutes=m)).strftime("%Y-%m-%dT%H:%MZ"), rain)
        for m in range(start_min, stop_min)
    ]
    r = client.post(
        f"/stations/{station}/days/{DAY}/batches",
        files={"file": ("x.csv", make_csv(rows, station), "text/csv")},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _heavy_rows(start, stop, per_minute=1.0):
    return [(m, per_minute) for m in range(start, stop)]


def _alert(client, s1, s2, **overrides):
    payload = {
        "day": DAY,
        "station_ids": [s1, s2],
        "period_start": "2026-11-12T08:00Z",
        "period_end": "2026-11-12T09:00Z",
        "threshold_mm": 20.0,
    }
    payload.update(overrides)
    return client.post("/alerts/regional-rainfall", json=payload)


# --------------------------------------------------------------------- happy
def test_alert_triggers_on_earliest_window_with_two_stations(client, stations):
    s1, s2 = stations
    # s1 reaches 20 mm/30min from window 08:05 (minutes 05..34).
    _upload_minutes(client, s1, _heavy_rows(5, 40))
    # s2 qualifies from window 08:00 and keeps reporting through 08:39.
    _upload_minutes(client, s2, _heavy_rows(0, 40))

    r = _alert(client, s1, s2)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["triggered"] is True
    assert body["window_minutes"] == 30
    assert body["windows_evaluated"] == 31  # 60-minute period -> 31 windows
    alert = body["alert"]
    # Earliest window where BOTH stations qualify is 08:05, not s2's 08:00.
    assert alert["window_start"] == "2026-11-12T08:05:00Z"
    assert alert["window_end"] == "2026-11-12T08:35:00Z"
    assert alert["participating_station_count"] == 2
    by_station = {p["station_id"]: p for p in alert["participating_stations"]}
    assert set(by_station) == {s1, s2}
    for participant in by_station.values():
        assert participant["rain_total_mm"] == pytest.approx(30.0)
        assert len(participant["minutes"]) == 30
        assert participant["minutes"][0]["obs_time"] == alert["window_start"]
        assert participant["minutes"][-1]["obs_time"] == "2026-11-12T08:34:00Z"


def test_alert_not_triggered_when_only_one_station_reaches_threshold(client, stations):
    s1, s2 = stations
    _upload_minutes(client, s1, _heavy_rows(0, 40))
    _upload_minutes(client, s2, [(m, 0.1) for m in range(0, 40)])

    r = _alert(client, s1, s2)
    assert r.status_code == 200
    body = r.json()
    assert body["triggered"] is False
    assert body["alert"] is None
    assert body["windows_evaluated"] == 31
    assert "no 30-minute window" in body["reason"]


def test_alert_needs_two_stations_in_the_same_window_not_peaks_combined(client, stations):
    s1, s2 = stations
    # s1 qualifies in early windows, s2 only in late windows: each window has
    # at most one qualifying station, so no alert may fire.
    _upload_minutes(client, s1, _heavy_rows(0, 30))
    _upload_minutes(client, s2, _heavy_rows(30, 60))

    r = _alert(client, s1, s2)
    body = r.json()
    assert r.status_code == 200
    assert body["triggered"] is False


def test_alert_uses_exact_threshold_equality(client, stations):
    s1, s2 = stations
    # 30 minutes * 0.5 = 15.0 mm exactly; 40 minutes keeps every window whole.
    _upload_minutes(client, s1, [(m, 0.5) for m in range(0, 40)])
    _upload_minutes(client, s2, [(m, 0.5) for m in range(0, 40)])

    r = _alert(
        client, s1, s2,
        threshold_mm=15.0,
        period_start="2026-11-12T08:00Z",
        period_end="2026-11-12T08:40Z",
    )
    assert r.status_code == 200
    body = r.json()
    assert body["triggered"] is True
    assert body["alert"]["window_start"] == "2026-11-12T08:00:00Z"

    r = _alert(
        client, s1, s2,
        threshold_mm=15.0001,
        period_start="2026-11-12T08:00Z",
        period_end="2026-11-12T08:40Z",
    )
    assert r.json()["triggered"] is False


# ------------------------------------------------- completeness and conflicts
def test_missing_minute_makes_station_unusable_in_covering_windows(client, stations):
    s1, s2 = stations
    _upload_minutes(client, s1, _heavy_rows(0, 40))
    # s2 skips minute 08:07 and its totals stay below the threshold anyway, so
    # no window can ever have two qualifying stations.
    rows = [(m, 0.5) for m in range(0, 40) if m != 7]
    _upload_minutes(client, s2, rows)

    r = _alert(client, s1, s2, threshold_mm=20.0)
    body = r.json()
    assert r.status_code == 200
    assert body["triggered"] is False


def test_station_statuses_report_missing_and_conflict_minutes(client, stations):
    s1, s2 = stations
    _upload_minutes(client, s1, _heavy_rows(0, 30))
    # s2: one missing minute (08:03) and one conflicting minute (08:05).
    _upload_minutes(client, s2, [(m, 1.0) for m in range(0, 30) if m != 3])
    _upload_minutes(client, s2, [(5, 2.5)])

    r = _alert(
        client, s1, s2,
        threshold_mm=1.0,
        period_start="2026-11-12T08:00Z",
        period_end="2026-11-12T08:30Z",
    )
    assert r.status_code == 200
    body = r.json()
    # Only one window exists; s2 is unusable in it -> no two-station alert.
    assert body["triggered"] is False
    statuses = {s["station_id"]: s for s in body["station_statuses"]}
    assert statuses[s1]["usable"] is True
    assert statuses[s2]["usable"] is False
    assert statuses[s2]["missing_minutes"] == ["2026-11-12T08:03:00Z"]
    assert statuses[s2]["conflict_minutes"] == ["2026-11-12T08:05:00Z"]


def test_same_minute_same_value_duplicate_counts_once(client, stations):
    s1, s2 = stations
    # Two batches; s1 minute 08:01 appears twice with identical value.
    _upload_minutes(client, s1, _heavy_rows(0, 40))
    _upload_minutes(client, s1, [(1, 1.0)])
    _upload_minutes(client, s2, _heavy_rows(0, 40))

    r = _alert(
        client, s1, s2,
        threshold_mm=30.0,
        period_start="2026-11-12T08:00Z",
        period_end="2026-11-12T08:40Z",
    )
    body = r.json()
    assert r.status_code == 200
    assert body["triggered"] is True
    p1 = next(p for p in body["alert"]["participating_stations"] if p["station_id"] == s1)
    # 30 distinct minutes, the duplicate did not double-count minute 08:01.
    assert p1["rain_total_mm"] == pytest.approx(30.0)
    assert len(p1["minutes"]) == 30


def test_same_minute_conflicting_values_block_that_station(client, stations):
    s1, s2 = stations
    _upload_minutes(client, s1, _heavy_rows(0, 30))
    _upload_minutes(client, s2, _heavy_rows(0, 30))
    # Conflicting value for s2 at minute 08:05.
    _upload_minutes(client, s2, [(5, 2.5)])

    r = _alert(client, s1, s2, threshold_mm=20.0)
    body = r.json()
    assert r.status_code == 200
    # s2 is unusable for every window covering 08:05; only s1 can qualify.
    assert body["triggered"] is False


def test_pure_series_conflict_and_duplicate_rules():
    def rec(m, v):
        return {
            "obs_time": datetime(2026, 11, 12, 8, m, tzinfo=timezone.utc),
            "rain_mm": v,
        }

    series = alerts.series_from_records(
        "X", [rec(0, 1.0), rec(0, 1.0), rec(1, 2.0), rec(1, 2.5), rec(2, 0.5)]
    )
    assert series.values == {
        datetime(2026, 11, 12, 8, 0, tzinfo=timezone.utc): 1.0,
        datetime(2026, 11, 12, 8, 2, tzinfo=timezone.utc): 0.5,
    }
    assert series.conflicts == {
        datetime(2026, 11, 12, 8, 1, tzinfo=timezone.utc)
    }


# ------------------------------------------------------------- request rules
def test_station_count_must_be_2_to_5(client, stations):
    s1, _ = stations
    r = _alert(client, s1, s1, station_ids=[s1])
    assert r.status_code == 422
    r = _alert(client, s1, s1, station_ids=[f"S{i:03d}" for i in range(6)])
    assert r.status_code == 422


def test_duplicate_station_ids_rejected(client, stations):
    s1, _ = stations
    r = _alert(client, s1, s1, station_ids=[s1, s1])
    assert r.status_code == 422


def test_period_cannot_exceed_120_minutes(client, stations):
    s1, s2 = stations
    r = _alert(
        client, s1, s2,
        period_start="2026-11-12T08:00Z",
        period_end="2026-11-12T10:01Z",
    )
    assert r.status_code == 422
    assert "120" in r.text


def test_period_shorter_than_window_is_non_triggered_not_an_error(client, stations):
    s1, s2 = stations
    _upload_minutes(client, s1, _heavy_rows(0, 20))
    _upload_minutes(client, s2, _heavy_rows(0, 20))
    r = _alert(
        client, s1, s2,
        period_start="2026-11-12T08:00Z",
        period_end="2026-11-12T08:20Z",
    )
    assert r.status_code == 200
    body = r.json()
    assert body["triggered"] is False
    assert body["windows_evaluated"] == 0
    assert "shorter than the 30-minute window" in body["reason"]


def test_period_timestamps_must_stay_in_requested_utc_day(client, stations):
    s1, s2 = stations
    r = _alert(
        client, s1, s2,
        period_start="2026-11-12T23:50Z",
        period_end="2026-11-13T00:10Z",
    )
    assert r.status_code == 422


def test_period_timestamps_must_be_minute_aligned(client, stations):
    s1, s2 = stations
    r = _alert(
        client, s1, s2,
        period_start="2026-11-12T08:00:30Z",
        period_end="2026-11-12T09:00Z",
    )
    assert r.status_code == 422


def test_negative_threshold_rejected(client, stations):
    s1, s2 = stations
    r = _alert(client, s1, s2, threshold_mm=-1)
    assert r.status_code == 422


def test_unknown_station_is_simply_unusable_not_an_error(client, stations):
    s1, s2 = stations
    _upload_minutes(client, s1, _heavy_rows(0, 30))
    r = _alert(client, s1, s2)
    assert r.status_code == 200
    body = r.json()
    assert body["triggered"] is False
    assert body["batch_ids_by_station"][s2] == []


def test_120_minute_period_boundary_accepted(client, stations):
    s1, s2 = stations
    # Full 120 minutes of 1.0 mm/min observations for both stations.
    _upload_hhmm(client, s1, 0, 120)
    _upload_hhmm(client, s2, 0, 120)
    r = _alert(
        client, s1, s2,
        period_start="2026-11-12T08:00Z",
        period_end="2026-11-12T10:00Z",
        threshold_mm=30.0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["period_minutes"] == 120
    assert body["windows_evaluated"] == 91
    assert body["triggered"] is True


def test_three_stations_two_qualifying_is_enough(client, stations):
    s1, s2 = stations
    s3 = f"{s1}C"
    # Two complete heavy stations; the third station stays below threshold.
    _upload_minutes(client, s1, _heavy_rows(0, 40))
    _upload_minutes(client, s2, _heavy_rows(0, 40))
    _upload_minutes(client, s3, [(m, 0.1) for m in range(0, 40)])

    r = _alert(client, s1, s2, station_ids=[s1, s2, s3], threshold_mm=20.0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["triggered"] is True
    participants = {p["station_id"] for p in body["alert"]["participating_stations"]}
    assert participants == {s1, s2}
    assert body["alert"]["participating_station_count"] == 2


def test_invalid_day_shape_rejected(client, stations):
    s1, s2 = stations
    r = _alert(client, s1, s2, day="20261112")
    assert r.status_code == 422


def test_threshold_must_be_finite(client, stations):
    s1, s2 = stations
    r = _alert(client, s1, s2, threshold_mm="not-a-number")
    assert r.status_code == 422


# ----------------------------------------------------- snapshot consistency
def _call_alert(station_ids):
    return alerts.determine_regional_alert(
        day=DAY,
        station_ids=station_ids,
        period_start="2026-11-12T08:00Z",
        period_end="2026-11-12T09:00Z",
        threshold_mm=1.0,
    )


def test_alert_reads_one_snapshot_while_merge_and_upload_happen(client, stations):
    s1, s2 = stations
    for _ in range(3):
        services.upload_csv(s1, DAY, make_csv([("2026-11-12T08:01Z", 2.0)], s1))
        services.upload_csv(s2, DAY, make_csv([("2026-11-12T08:01Z", 2.0)], s2))

    original_read = file_manager.read_table_checked
    inside = threading.Event()
    release = threading.Event()
    pinned_during_read: list[str] = []

    def slow_read(refs):
        if not inside.is_set():
            inside.set()
            release.wait(timeout=15)
        # Capture every pinned file across both per-station reads.
        pinned_during_read.extend(r.batch_id for r in refs)
        return original_read(refs)

    file_manager.read_table_checked = slow_read
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            alert_future = pool.submit(_call_alert, [s1, s2])
            assert inside.wait(timeout=10)

            # While the alert read is blocked: a merge supersedes s1's files,
            # an extra upload lands for s2, and a reap pass runs. The pinned
            # snapshot files must survive all three.
            merge_future = pool.submit(services.merge_station_day, s1, DAY)
            upload_future = pool.submit(
                services.upload_csv,
                s2,
                DAY,
                make_csv([("2026-11-12T08:02Z", 5.0)], s2),
            )
            merge_future.result(timeout=20)
            upload_future.result(timeout=20)
            reaped_during = set(file_manager.reap_once())
            assert reaped_during.isdisjoint(pinned_during_read)
            release.set()
            result = alert_future.result(timeout=10)
    finally:
        file_manager.read_table_checked = original_read

    # The alert evaluated the pinned pre-merge snapshot: 3 batches per station,
    # only minute 08:01 present (the late 08:02 upload is not part of it).
    assert result["triggered"] is False
    assert len(result["batch_ids_by_station"][s1]) == 3
    assert len(result["batch_ids_by_station"][s2]) == 3
    assert set(pinned_during_read) == {
        bid for bids in result["batch_ids_by_station"].values() for bid in bids
    }

    # After the alert closes its snapshot the reaper may remove the old files.
    reaped_after = set(file_manager.reap_once())
    assert set(result["batch_ids_by_station"][s1]).issubset(reaped_after)

    # A follow-up alert sees the post-merge published set.
    follow_up = _call_alert([s1, s2])
    assert len(follow_up["batch_ids_by_station"][s1]) == 1
    assert len(follow_up["batch_ids_by_station"][s2]) == 4


def test_read_failure_aborts_without_partial_conclusion(client, stations, monkeypatch):
    s1, s2 = stations
    _upload_minutes(client, s1, _heavy_rows(0, 30))
    _upload_minutes(client, s2, _heavy_rows(0, 30))

    def failing_read(refs):
        if refs and any(s2 in r.path.parts for r in refs):
            raise SnapshotReadError("injected disk failure")
        return file_manager.read_table(refs)

    monkeypatch.setattr(file_manager, "read_table_checked", failing_read)
    r = _alert(client, s1, s2)
    assert r.status_code == 500
    assert "SnapshotReadError" in r.text


def test_upload_runs_in_parallel_with_alert_service(client, stations):
    """End-to-end: the alert never blocks a concurrent upload."""
    s1, s2 = stations
    _upload_minutes(client, s1, _heavy_rows(0, 60))
    _upload_minutes(client, s2, _heavy_rows(0, 60))

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_alert = pool.submit(
            alerts.determine_regional_alert,
            DAY,
            [s1, s2],
            "2026-11-12T08:00Z",
            "2026-11-12T09:00Z",
            30.0,
        )
        f_upload = pool.submit(
            services.upload_csv,
            s1,
            DAY,
            make_csv([("2026-11-12T08:45Z", 0.7)], s1),
        )
        alert_result = f_alert.result(timeout=20)
        upload_result = f_upload.result(timeout=20)

    assert alert_result["triggered"] is True
    assert upload_result["status"] == "published"
    # The late upload is visible to subsequent queries.
    body = services.query_station_day(s1, DAY)
    assert body["row_count"] == 61
