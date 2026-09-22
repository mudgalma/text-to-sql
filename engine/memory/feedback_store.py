"""Load and retrieve verified, exact-query SQL corrections from feedback CSV data."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


class FeedbackStoreError(ValueError):
    """Raised when a supplied feedback CSV is malformed or unsafe to load."""


@dataclass(frozen=True)
class FeedbackEntry:
    """One verified query correction available to the pipeline."""

    query: str
    corrected_sql: str


class FeedbackStore:
    """Read optional feedback and retrieve exact or word-overlap matches deterministically."""

    _REQUIRED_COLUMNS = frozenset({"query", "corrected_sql"})
    _TOKEN = re.compile(r"[a-z0-9]+")
    _MAX_FILE_BYTES = 1_000_000
    _MAX_QUERY_CHARS = 2_000

    def __init__(self, path: str | Path | None) -> None:
        self._entries = self._load(path) if path is not None and Path(path).is_file() else []

    def retrieve_similar(self, query: str, k: int = 2) -> list[FeedbackEntry]:
        """Return up to ``k`` deterministic word-overlap matches from verified feedback."""

        if k <= 0 or not self._entries:
            return []
        query_words = self._words(query)
        scored = [
            (len(query_words & self._words(entry.query)), index, entry)
            for index, entry in enumerate(self._entries)
        ]
        return [entry for score, _, entry in sorted(scored, key=lambda item: (-item[0], item[1])) if score > 0][:k]

    def get_exact_correction(self, query: str) -> str | None:
        """Return a correction only for the same normalized query, never a merely similar one."""

        normalized = self._normalize(query)
        return next(
            (entry.corrected_sql for entry in self._entries if self._normalize(entry.query) == normalized),
            None,
        )

    def _load(self, path_value: str | Path) -> list[FeedbackEntry]:
        """Validate and load a small UTF-8 feedback CSV file."""

        path = Path(path_value).expanduser().resolve()
        if path.suffix.lower() != ".csv":
            raise FeedbackStoreError("Feedback data must be a CSV file.")
        if path.stat().st_size > self._MAX_FILE_BYTES:
            raise FeedbackStoreError("Feedback CSV exceeds the 1 MB size limit.")
        try:
            with path.open(encoding="utf-8-sig", newline="") as source:
                reader = csv.DictReader(source)
                fieldnames = set(reader.fieldnames or [])
                if not self._REQUIRED_COLUMNS.issubset(fieldnames):
                    raise FeedbackStoreError("Feedback CSV requires query and corrected_sql columns.")
                entries = [
                    FeedbackEntry(row["query"].strip(), row["corrected_sql"].strip())
                    for row in reader
                    if row.get("query", "").strip() and row.get("corrected_sql", "").strip()
                ]
        except (csv.Error, OSError, UnicodeDecodeError) as error:
            raise FeedbackStoreError("Feedback CSV could not be loaded.") from error
        return entries

    @classmethod
    def _normalize(cls, value: str) -> str:
        """Normalize text for an exact business-query comparison."""

        if not isinstance(value, str) or len(value) > cls._MAX_QUERY_CHARS:
            return ""
        return " ".join(cls._TOKEN.findall(value.lower()))

    @classmethod
    def _words(cls, value: str) -> set[str]:
        """Return bounded normalized terms for feedback retrieval."""

        if not isinstance(value, str) or len(value) > cls._MAX_QUERY_CHARS:
            return set()
        return set(cls._TOKEN.findall(value.lower()))
