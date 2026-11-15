"""Domain exceptions mapped to HTTP responses in :mod:`app.main`."""
from __future__ import annotations


class StorageError(Exception):
    """Base class for expected storage-layer failures."""

    status_code = 400


class CsvValidationError(StorageError):
    status_code = 400


class AlertValidationError(StorageError):
    """Invalid regional-rainfall alert request parameters."""

    status_code = 422


class MergeConflictError(StorageError):
    """Another merge for the same station/day is already running."""

    status_code = 409


class MergeFailedError(StorageError):
    """A merge was attempted but could not be completed atomically."""

    status_code = 409


class NotFoundError(StorageError):
    status_code = 404


class SnapshotReadError(StorageError):
    """A pinned snapshot file could not be read; no partial result is returned."""

    status_code = 500
