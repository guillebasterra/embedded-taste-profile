"""Exceptions shared inside the ingest package. Messages never contain secrets."""

from __future__ import annotations


class IngestError(Exception):
    """Base class: the CLI prints the message and exits 1."""


class AuthError(IngestError):
    """Missing, rejected or unrefreshable credentials."""


class ApiError(IngestError):
    def __init__(self, status: int, message: str = ""):
        self.status = status
        super().__init__(f"Pinterest API error {status}" + (f": {message}" if message else ""))


class RateLimitStop(IngestError):
    """429 that is not worth waiting out in-process (daily cap, or retries exhausted)."""

    def __init__(self, retry_after: float | None = None):
        self.retry_after = retry_after
        super().__init__("Pinterest rate limit reached")
