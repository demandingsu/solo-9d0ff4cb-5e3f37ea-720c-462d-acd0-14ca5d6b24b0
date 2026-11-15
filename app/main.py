"""FastAPI entrypoint: upload / merge / query API for rain-gauge batches."""
from __future__ import annotations

import contextlib
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Path as PathParam, Query, UploadFile
from pydantic import BaseModel, Field, field_validator

from . import alerts, db, services
from .config import settings
from .errors import StorageError
from .storage import file_manager


# ----------------------------------------------------------------- schemas
class BatchResponse(BaseModel):
    batch_id: str
    station_id: str
    day: str
    row_count: int
    size_bytes: int
    content_sha256: str
    status: str
    is_merged: bool


class MergeResponse(BaseModel):
    merged_batch_id: str
    station_id: str
    day: str
    source_batch_ids: List[str]
    source_count: int
    row_count: int
    size_bytes: int
    content_sha256: str
    status: str


class RainRecord(BaseModel):
    station_id: str
    obs_time: str
    rain_mm: float
    source_batch_id: str


class QueryResponse(BaseModel):
    station_id: str
    day: str
    snapshot_token: str
    file_count: int
    batch_ids: List[str]
    row_count: int
    truncated: bool
    records: List[RainRecord]


class BatchInfo(BaseModel):
    batch_id: str
    row_count: int
    size_bytes: int
    status: str
    is_merged: bool
    source_batches: Optional[List[str]]
    file_exists: bool
    created_at: str


class ListBatchesResponse(BaseModel):
    station_id: str
    day: str
    batches: List[BatchInfo]


# ------------------------------------------------------------- alert schemas
class RegionalAlertRequest(BaseModel):
    day: str = Field(..., description="UTC day, YYYY-MM-DD")
    station_ids: List[str] = Field(
        ...,
        min_length=alerts.MIN_REQUEST_STATIONS,
        max_length=alerts.MAX_REQUEST_STATIONS,
        description="2-5 station ids; all data must belong to the same UTC day",
    )
    period_start: str = Field(
        ..., description="UTC ISO-8601 minute-aligned start, e.g. 2026-11-12T08:00Z"
    )
    period_end: str = Field(
        ...,
        description=(
            "UTC ISO-8601 minute-aligned exclusive end; the period may span at "
            "most 120 minutes"
        ),
    )
    threshold_mm: float = Field(
        ..., ge=0, description="30-minute accumulated rainfall threshold in mm"
    )

    @field_validator("station_ids")
    @classmethod
    def _no_duplicate_stations(cls, value: List[str]) -> List[str]:
        if len(set(value)) != len(value):
            raise ValueError("station_ids must not contain duplicates")
        return value


class AlertMinute(BaseModel):
    obs_time: str
    rain_mm: float


class AlertParticipant(BaseModel):
    station_id: str
    rain_total_mm: float
    minutes: List[AlertMinute]


class AlertWindow(BaseModel):
    window_start: str
    window_end: str
    participating_station_count: int
    participating_stations: List[AlertParticipant]


class RegionalAlertResponse(BaseModel):
    triggered: bool
    alert: Optional[AlertWindow]
    reason: Optional[str] = None
    day: str
    station_ids: List[str]
    window_minutes: int
    threshold_mm: float
    period_start: str
    period_end: str
    period_minutes: int
    windows_evaluated: int
    snapshot_token: str
    batch_ids_by_station: dict
    station_statuses: Optional[List[dict]] = None


# ---------------------------------------------------------------- app setup
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.parquet_root.mkdir(parents=True, exist_ok=True)
    settings.tmp_dir.mkdir(parents=True, exist_ok=True)
    db.init_db()
    file_manager.startup_sweep()
    file_manager.start_reaper()
    yield
    file_manager.stop_reaper()


app = FastAPI(
    title="Rain Gauge Batch Storage API",
    version="1.0.0",
    description=(
        "Upload per-station/per-day rain CSV batches, compact them into one "
        "Parquet file, and query published files with snapshot consistency."
    ),
    lifespan=lifespan,
)


@app.exception_handler(StorageError)
async def storage_error_handler(request, exc: StorageError):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=exc.status_code,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


STATION_PATH = PathParam(..., pattern=r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")
DAY_PATH = PathParam(..., pattern=r"^\d{4}-\d{2}-\d{2}$")


# ----------------------------------------------------------------- endpoints
@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok", "data_dir": str(settings.data_dir)}


@app.post(
    "/stations/{station_id}/days/{day}/batches",
    response_model=BatchResponse,
    status_code=201,
    tags=["upload"],
    summary="Upload one station/day CSV as an independent batch",
)
async def upload_batch(
    station_id: str = STATION_PATH,
    day: str = DAY_PATH,
    file: UploadFile = File(..., description="CSV with station, UTC time, rain_mm"),
) -> BatchResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds {settings.max_upload_bytes} bytes",
        )
    result = services.upload_csv(station_id, day, content)
    return BatchResponse(**result)


@app.get(
    "/stations/{station_id}/days/{day}/records",
    response_model=QueryResponse,
    tags=["query"],
    summary="Query a station/day by actually reading published Parquet files",
)
def query_records(
    station_id: str = STATION_PATH,
    day: str = DAY_PATH,
    limit: Optional[int] = Query(None, ge=0, description="optional row cap"),
) -> QueryResponse:
    result = services.query_station_day(station_id, day, limit=limit)
    return QueryResponse(**result)


@app.post(
    "/stations/{station_id}/days/{day}/merge",
    response_model=MergeResponse,
    tags=["merge"],
    summary="Compact published small batches into one Parquet file",
)
def merge_day(
    station_id: str = STATION_PATH,
    day: str = DAY_PATH,
) -> MergeResponse:
    result = services.merge_station_day(station_id, day)
    return MergeResponse(**result)


@app.get(
    "/stations/{station_id}/days/{day}/batches",
    response_model=ListBatchesResponse,
    tags=["query"],
    summary="List all batches (published and superseded) for a station/day",
)
def get_batches(
    station_id: str = STATION_PATH,
    day: str = DAY_PATH,
) -> ListBatchesResponse:
    return ListBatchesResponse(**services.list_batches(station_id, day))


@app.post("/admin/reap", tags=["ops"], summary="Force reaping of unreferenced superseded files")
def force_reap() -> dict:
    return services.force_reap()


@app.post(
    "/alerts/regional-rainfall",
    response_model=RegionalAlertResponse,
    tags=["alert"],
    summary=(
        "Find the earliest 30-minute window where at least two stations each "
        "have complete observations reaching the accumulated-rain threshold"
    ),
)
def regional_rainfall_alert(payload: RegionalAlertRequest) -> RegionalAlertResponse:
    # Semantic validation (day-relative times, <=120 min span, station format)
    # lives in the service layer; AlertValidationError maps to HTTP 422.
    result = alerts.determine_regional_alert(
        day=payload.day,
        station_ids=payload.station_ids,
        period_start=payload.period_start,
        period_end=payload.period_end,
        threshold_mm=payload.threshold_mm,
    )
    return RegionalAlertResponse(**result)
