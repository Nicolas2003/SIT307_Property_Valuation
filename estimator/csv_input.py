from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import IO, Any

from .features import FEATURE_NAMES, TARGET
from .validation import Issue, Report, parse_all


@dataclass
class Upload:
    features: dict[str, Any] = field(default_factory=dict)
    actual_price: float | None = None
    report: Report = field(default_factory=Report)
    unknown_columns: list[str] = field(default_factory=list)
    recognised: int = 0

    @property
    def ok(self) -> bool:
        return self.report.ok


def read_csv(source: IO[bytes] | IO[str] | str | bytes) -> Upload:
    text = _as_text(source)
    rows = list(csv.DictReader(io.StringIO(text)))

    if not rows:
        return Upload(report=Report(errors=[Issue("file", "no data row found below the header")]))
    if len(rows) > 1:
        return Upload(
            report=Report(
                errors=[
                    Issue(
                        "file",
                        f"expected one data row, found {len(rows)}. "
                        "This app estimates one property at a time.",
                    )
                ]
            )
        )

    row = {}
    for column, value in rows[0].items():
        if column:
            row[column.strip()] = value
    known = set(FEATURE_NAMES)

    raw = {}
    for name in FEATURE_NAMES:
        if name in row:
            raw[name] = row[name]
    features, report = parse_all(raw)

    actual, actual_error = _read_target(row)
    if actual_error:
        report.warnings.append(actual_error)

    unknown = sorted(set(row) - known - {TARGET})
    return Upload(
        features=features,
        actual_price=actual,
        report=report,
        unknown_columns=unknown,
        recognised=len(raw),
    )


def _read_target(row: dict[str, str]) -> tuple[float | None, Issue | None]:
    text = (row.get(TARGET) or "").strip()
    if not text:
        return None, None
    try:
        return float(text), None
    except ValueError:
        return None, Issue(TARGET, f"`{text}` is not a number; actual price not shown")


def _as_text(source: Any) -> str:
    if isinstance(source, bytes):
        return source.decode("utf-8-sig")
    if isinstance(source, str):
        return source
    data = source.read()
    return data.decode("utf-8-sig") if isinstance(data, bytes) else data
