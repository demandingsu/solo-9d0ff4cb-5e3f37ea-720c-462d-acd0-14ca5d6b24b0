"""Regional heavy-rainfall alert determination over published minute data.

Deterministic sliding-window rules (see :func:`evaluate_alert`):

* a fixed 30-minute window slides minute by minute over the requested period
  (at most 120 minutes), every window fully inside the period;
* for a station to participate in a window, the station must have a complete,
  conflict-free observation for *every* minute of that window:
    - duplicate records at the same minute with the same value count once;
    - two records at the same minute with different values are a conflict and
      make the station unavailable in every window covering that minute;
    - a missing minute makes the station unavailable in every covering window;
* an alert fires only when **at least two** such usable stations each
  accumulate ``threshold_mm`` or more inside the *same* window. Peak values
  are never combined across different windows;
* the earliest qualifying window is returned together with the participating
  stations' rainfall. If none qualifies, a non-triggered result is returned.

All cross-station data is read from one multi-station published snapshot
(:meth:`FileManager.open_multi_snapshot`): uploads and merges keep running in
parallel, but the files selected here are pinned until the judgment ends and
any read failure aborts the whole request instead of yielding a partial
conclusion.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Sequence, Set, Tuple

from .errors import AlertValidationError
from .storage import FileManager, file_manager

WINDOW_MINUTES = 30
MAX_PERIOD_MINUTES = 120
MIN_STATIONS_IN_WINDOW = 2
MIN_REQUEST_STATIONS = 2
MAX_REQUEST_STATIONS = 5
# Floating-point tolerance so exact-threshold accumulation (e.g. 30 x 0.5 mm)
# is not defeated by binary rounding noise.
_THRESHOLD_EPS = 1e-9


# ------------------------------------------------------------- request model
@dataclass(frozen=True)
class AlertRequest:
    day: str
    station_ids: List[str]
    period_start: datetime
    period_end: datetime
    threshold_mm: float


# ------------------------------------------------------------------ parsing
def _parse_minute_iso(raw: str, field: str, day: str) -> datetime:
    if not isinstance(raw, str):
        raise AlertValidationError(f"{field} must be a UTC ISO-8601 string")
    text = raw.strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AlertValidationError(
            f"{field} {raw!r} is not a valid ISO-8601 timestamp"
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    if dt.second != 0 or dt.microsecond != 0:
        raise AlertValidationError(f"{field} must be minute-aligned (seconds = 0)")
    if dt.date().isoformat() != day:
        raise AlertValidationError(
            f"{field} {raw!r} is not within UTC day {day}"
        )
    return dt


def build_alert_request(
    day: str,
    station_ids: Sequence[str],
    period_start: str,
    period_end: str,
    threshold_mm: float,
) -> AlertRequest:
    """Validate and normalize an alert request (also used by the HTTP layer)."""
    from .errors import CsvValidationError
    from .ingestion import validate_day, validate_station_id

    try:
        validate_day(day)
    except CsvValidationError as exc:
        raise AlertValidationError(str(exc)) from exc

    stations = list(station_ids)
    if not (MIN_REQUEST_STATIONS <= len(stations) <= MAX_REQUEST_STATIONS):
        raise AlertValidationError(
            f"station_ids must contain {MIN_REQUEST_STATIONS}-"
            f"{MAX_REQUEST_STATIONS} entries, got {len(stations)}"
        )
    normalized: List[str] = []
    seen: Set[str] = set()
    for sid in stations:
        try:
            validate_station_id(sid)
        except CsvValidationError as exc:
            raise AlertValidationError(str(exc)) from exc
        if sid in seen:
            raise AlertValidationError(f"station_ids contains duplicate {sid!r}")
        seen.add(sid)
        normalized.append(sid)

    if not isinstance(threshold_mm, (int, float)):
        raise AlertValidationError("threshold_mm must be a number")
    threshold = float(threshold_mm)
    if threshold != threshold or threshold in (float("inf"), float("-inf")):
        raise AlertValidationError("threshold_mm must be finite")
    if threshold < 0:
        raise AlertValidationError("threshold_mm must be non-negative")

    start = _parse_minute_iso(period_start, "period_start", day)
    end = _parse_minute_iso(period_end, "period_end", day)
    if not (start < end):
        raise AlertValidationError("period_start must be strictly before period_end")
    span = int((end - start).total_seconds() // 60)
    if span > MAX_PERIOD_MINUTES:
        raise AlertValidationError(
            f"period spans {span} minutes; maximum is {MAX_PERIOD_MINUTES}"
        )
    return AlertRequest(
        day=day,
        station_ids=normalized,
        period_start=start,
        period_end=end,
        threshold_mm=threshold,
    )


# ------------------------------------------------------------ pure evaluation
@dataclass
class StationSeries:
    """Minute-keyed view of one station inside the snapshot."""

    station_id: str
    values: Dict[datetime, float]
    conflicts: Set[datetime]

    def usable_in(self, window_start: datetime) -> bool:
        """True when every minute of [start, start+30min) is present and clean."""
        for offset in range(WINDOW_MINUTES):
            minute = window_start + timedelta(minutes=offset)
            if minute in self.conflicts or minute not in self.values:
                return False
        return True

    def window_sum(self, window_start: datetime) -> float:
        return sum(
            self.values[window_start + timedelta(minutes=offset)]
            for offset in range(WINDOW_MINUTES)
        )

    def window_minutes(
        self, window_start: datetime
    ) -> List[Tuple[datetime, float]]:
        return [
            (
                window_start + timedelta(minutes=offset),
                self.values[window_start + timedelta(minutes=offset)],
            )
            for offset in range(WINDOW_MINUTES)
        ]


def series_from_table(station_id: str, table) -> StationSeries:
    """Build minute state straight from the canonical Parquet table.

    Same-minute records with identical values count once; same-minute records
    with differing values mark that minute as a conflict.
    """
    times = table.column("obs_time").to_pylist()
    rains = table.column("rain_mm").to_pylist()
    values: Dict[datetime, float] = {}
    conflicts: Set[datetime] = set()
    for ts, rain in zip(times, rains):
        minute: datetime = ts.replace(second=0, microsecond=0)
        value = float(rain)
        if minute in conflicts:
            continue
        if minute in values:
            if values[minute] != value:
                conflicts.add(minute)
                values.pop(minute, None)
            # identical duplicate: counted once, nothing to do
        else:
            values[minute] = value
    return StationSeries(station_id=station_id, values=values, conflicts=conflicts)


def series_from_records(station_id: str, records: Sequence[dict]) -> StationSeries:
    """Minute state from dict records whose ``obs_time`` is a UTC datetime.

    Same-minute records with identical values count once; same-minute records
    with differing values mark that minute as a conflict.
    """
    values: Dict[datetime, float] = {}
    conflicts: Set[datetime] = set()
    for rec in records:
        ts: datetime = rec["obs_time"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        minute = ts.replace(second=0, microsecond=0)
        rain = float(rec["rain_mm"])
        if minute in conflicts:
            continue
        if minute in values:
            if values[minute] != rain:
                conflicts.add(minute)
                values.pop(minute, None)
        else:
            values[minute] = rain
    return StationSeries(station_id=station_id, values=values, conflicts=conflicts)


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def evaluate_alert(
    request: AlertRequest,
    series_by_station: Dict[str, StationSeries],
    snapshot_token: str,
    batch_ids_by_station: Dict[str, List[str]],
) -> dict:
    """Slide a 30-minute window minute by minute and find the earliest alert.

    ``series_by_station`` must already contain one entry per requested station
    (empty series for stations without published data).
    """
    period_minutes = int(
        (request.period_end - request.period_start).total_seconds() // 60
    )
    window_count = max(0, period_minutes - WINDOW_MINUTES + 1)

    common = {
        "day": request.day,
        "station_ids": request.station_ids,
        "window_minutes": WINDOW_MINUTES,
        "threshold_mm": request.threshold_mm,
        "period_start": _iso_z(request.period_start),
        "period_end": _iso_z(request.period_end),
        "period_minutes": period_minutes,
        "windows_evaluated": window_count,
        "snapshot_token": snapshot_token,
        "batch_ids_by_station": batch_ids_by_station,
    }

    if window_count == 0:
        return {
            **common,
            "triggered": False,
            "alert": None,
            "station_statuses": None,
            "reason": (
                f"period of {period_minutes} minute(s) is shorter than the "
                f"{WINDOW_MINUTES}-minute window; no window could be evaluated"
            ),
        }

    last_statuses: List[dict] = []
    for offset in range(window_count):
        window_start = request.period_start + timedelta(minutes=offset)
        window_end = window_start + timedelta(minutes=WINDOW_MINUTES)

        participants: List[dict] = []
        station_statuses: List[dict] = []
        for sid in request.station_ids:
            series = series_by_station[sid]
            if not series.usable_in(window_start):
                conflict_minutes = sorted(
                    m for m in series.conflicts if window_start <= m < window_end
                )
                present = {
                    m for m in series.values if window_start <= m < window_end
                }
                missing_minutes = sorted(
                    window_start + timedelta(minutes=i)
                    for i in range(WINDOW_MINUTES)
                    if (window_start + timedelta(minutes=i)) not in present
                    and (window_start + timedelta(minutes=i)) not in conflict_minutes
                )
                station_statuses.append(
                    {
                        "station_id": sid,
                        "usable": False,
                        "conflict_minutes": [_iso_z(m) for m in conflict_minutes],
                        "missing_minutes": [_iso_z(m) for m in missing_minutes],
                    }
                )
                continue

            total = series.window_sum(window_start)
            qualifies = total + _THRESHOLD_EPS >= request.threshold_mm
            if qualifies:
                participants.append(
                    {
                        "station_id": sid,
                        "rain_total_mm": round(total, 6),
                        "minutes": [
                            {"obs_time": _iso_z(m), "rain_mm": rain}
                            for m, rain in series.window_minutes(window_start)
                        ],
                    }
                )
            station_statuses.append(
                {
                    "station_id": sid,
                    "usable": True,
                    "rain_total_mm": round(total, 6),
                    "qualifies": qualifies,
                }
            )

        if len(participants) >= MIN_STATIONS_IN_WINDOW:
            participants.sort(key=lambda p: p["station_id"])
            return {
                **common,
                "triggered": True,
                "alert": {
                    "window_start": _iso_z(window_start),
                    "window_end": _iso_z(window_end),
                    "participating_station_count": len(participants),
                    "participating_stations": participants,
                },
                "station_statuses": station_statuses,
            }
        last_statuses = station_statuses

    return {
        **common,
        "triggered": False,
        "alert": None,
        # Diagnostics for the final evaluated window: which stations were
        # unusable (and why) versus usable but below the threshold.
        "station_statuses": last_statuses,
        "reason": (
            "no 30-minute window within the period had at least two stations "
            "with complete observations each reaching the threshold"
        ),
    }


# ------------------------------------------------------------------- service
def determine_regional_alert(
    day: str,
    station_ids: Sequence[str],
    period_start: str,
    period_end: str,
    threshold_mm: float,
    fm: FileManager = file_manager,
) -> dict:
    """Orchestrate the multi-station snapshot read and sliding-window judgment."""
    request = build_alert_request(
        day, station_ids, period_start, period_end, threshold_mm
    )
    targets = [(sid, request.day) for sid in request.station_ids]

    # One published snapshot for every station/day; files are pinned until the
    # judgment is fully done, so the reaper cannot remove them mid-read and a
    # merge finishing concurrently cannot change what is evaluated.
    grouped, token = fm.open_multi_snapshot(targets)
    all_refs = [ref for refs in grouped.values() for ref in refs]
    try:
        series_by_station: Dict[str, StationSeries] = {}
        batch_ids_by_station: Dict[str, List[str]] = {}
        for sid in request.station_ids:
            refs = grouped[(sid, request.day)]
            batch_ids_by_station[sid] = [r.batch_id for r in refs]
            if refs:
                # Any read failure raises SnapshotReadError: no partial answer.
                table = fm.read_table_checked(refs)
                series_by_station[sid] = series_from_table(sid, table)
            else:
                series_by_station[sid] = StationSeries(
                    station_id=sid, values={}, conflicts=set()
                )

        return evaluate_alert(
            request, series_by_station, token, batch_ids_by_station
        )
    finally:
        fm.close_snapshot(all_refs)
