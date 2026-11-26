"""Temporary CSV slot: the one file training reads its data from.

Training never reads the user's original CSV. The CSV wizard page detects
the chosen file's layout (``detect``) and writes only the columns the user
included into this slot (``stage_columns``), always comma-separated with
``.`` decimals. Editing the original file after staging therefore never
changes a run, and the slot's columns *are* the features trained on (see
``feature_names``).

The slot holds only the value columns; presence flags for missing cells are
derived later (see core.common.fit_and_export), never read from the file.
"""

import csv
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
SLOT_PATH = _REPO_ROOT / ".tmp" / "training_data.csv"

# Delimiters the wizard recognises, in order of preference.
_DELIMITERS = (",", ";", "\t", "|")
_DELIMITER_NAMES = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}
_ENCODINGS = ("utf-8-sig", "cp1250")
_SAMPLE_BYTES = 64 * 1024
# A cell like "15,2" or "-0,75": a decimal comma, not a thousands separator.
_DECIMAL_COMMA = re.compile(r"^\s*[-+]?\d+,\d+\s*$")


class CsvDetectError(Exception):
    """The file's layout couldn't be worked out; the message is for the user."""


@dataclass
class CsvInfo:
    delimiter: str
    decimal: str
    encoding: str
    columns: list[str]
    # Column name -> whether it has at least one numeric value.
    numeric: dict[str, bool] = field(default_factory=dict)
    n_rows: int = 0

    @property
    def delimiter_name(self) -> str:
        return _DELIMITER_NAMES.get(self.delimiter, repr(self.delimiter))


def _read_sample(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()[:_SAMPLE_BYTES]
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            # A multi-byte character cut off at the sample's end isn't a
            # real decoding failure.
            try:
                return raw[:-4].decode(encoding), encoding
            except UnicodeDecodeError:
                continue
    raise CsvDetectError("The file is not readable text (unknown encoding).")


def _detect_delimiter(lines: list[str]) -> str:
    header = lines[0]
    if not any(d in header for d in _DELIMITERS):
        return ","  # a single column; any delimiter works
    try:
        delimiter = csv.Sniffer().sniff("\n".join(lines), delimiters="".join(_DELIMITERS)).delimiter
        if delimiter in header:
            return delimiter
    except csv.Error:
        pass
    # Fallback: the candidate that splits the header and the first rows into
    # the same number (> 1) of fields.
    rows = lines[:20]
    for delimiter in _DELIMITERS:
        counts = {len(next(csv.reader([line], delimiter=delimiter))) for line in rows}
        if len(counts) == 1 and counts.pop() > 1:
            return delimiter
    raise CsvDetectError(
        "Could not work out the delimiter: the rows don't split into the same "
        "number of columns with a comma, semicolon, tab or pipe."
    )


def _detect_decimal(lines: list[str], delimiter: str) -> str:
    if delimiter == ",":
        return "."
    for line in lines[1:50]:
        for cell in next(csv.reader([line], delimiter=delimiter), []):
            if _DECIMAL_COMMA.match(cell):
                return ","
    return "."


def _read_frame(path, delimiter: str, decimal: str, encoding: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=delimiter, decimal=decimal, encoding=encoding)
    # A trailing delimiter on every line gives an empty, unnamed last column.
    empty_unnamed = [
        name for name in frame.columns
        if str(name).startswith("Unnamed:") and frame[name].isna().all()
    ]
    return frame.drop(columns=empty_unnamed)


def detect(path) -> CsvInfo:
    """Work out the delimiter, decimal mark, encoding and columns of the CSV
    at ``path``. Raises CsvDetectError with a user-facing message.
    """
    path = Path(path)
    try:
        sample, encoding = _read_sample(path)
    except OSError as exc:
        raise CsvDetectError(f"Could not read the file: {exc}") from exc

    lines = [line for line in sample.splitlines() if line.strip()]
    if not lines:
        raise CsvDetectError("The file is empty.")
    if len(lines) < 2:
        raise CsvDetectError("The file has a header but no data rows.")
    # A sample shorter than the file may end mid-row.
    if path.stat().st_size > _SAMPLE_BYTES and len(lines) > 2:
        lines = lines[:-1]

    delimiter = _detect_delimiter(lines)
    decimal = _detect_decimal(lines, delimiter)

    try:
        frame = _read_frame(path, delimiter, decimal, encoding)
    except pd.errors.ParserError as exc:
        raise CsvDetectError(
            f"Rows have different numbers of columns ({_DELIMITER_NAMES[delimiter]}-"
            f"separated): {exc}"
        ) from exc
    except (pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as exc:
        raise CsvDetectError(f"Could not parse the file: {exc}") from exc

    if frame.empty:
        raise CsvDetectError("The file has a header but no data rows.")
    columns = [str(name) for name in frame.columns]
    if len(set(columns)) != len(columns):
        raise CsvDetectError("Some columns share the same name.")
    numeric = {
        str(name): bool(pd.to_numeric(frame[name], errors="coerce").notna().any())
        for name in frame.columns
    }
    if not any(numeric.values()):
        raise CsvDetectError("No column contains numeric values.")
    return CsvInfo(delimiter, decimal, encoding, columns, numeric, len(frame))


def stage_columns(source_path, info: CsvInfo, columns) -> Path:
    """Write only ``columns`` of ``source_path`` (parsed as ``info`` says)
    into the slot as a plain comma-separated CSV, replacing what was there.
    """
    frame = _read_frame(source_path, info.delimiter, info.decimal, info.encoding)
    frame.columns = [str(name) for name in frame.columns]
    SLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame[list(columns)].to_csv(SLOT_PATH, index=False)
    return SLOT_PATH


def stage(source_path) -> Path:
    """Copy ``source_path`` into the slot unchanged, replacing what was there."""
    SLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, SLOT_PATH)
    return SLOT_PATH


def is_staged() -> bool:
    return SLOT_PATH.is_file()


def clear() -> None:
    SLOT_PATH.unlink(missing_ok=True)


def staged_columns() -> list[str] | None:
    """The slot's column names, or None when nothing is staged."""
    if not is_staged():
        return None
    return [str(name) for name in pd.read_csv(SLOT_PATH, nrows=0).columns]


@dataclass
class StagedSummary:
    """What the algorithm notices need to know about the staged data."""

    columns: list[str]
    # Rows with every column numeric: the smallest presence pattern's rows.
    complete_rows: int
    # Per column, over its numeric cells (NaN for a column without any).
    minimum: list[float]
    maximum: list[float]


def staged_summary() -> StagedSummary | None:
    """Column names, complete-row count and per-column min/max of the staged
    CSV, or None when nothing is staged.
    """
    if not is_staged():
        return None
    frame = pd.read_csv(SLOT_PATH).apply(pd.to_numeric, errors="coerce")
    return StagedSummary(
        columns=[str(name) for name in frame.columns],
        complete_rows=int(frame.notna().all(axis=1).sum()),
        minimum=frame.min().astype("float64").tolist(),
        maximum=frame.max().astype("float64").tolist(),
    )


NOT_STAGED_MESSAGE = "No dataset staged - choose a CSV on the Data page first."


class NotStagedError(FileNotFoundError):
    """Training was started without a staged CSV; the message is for the user."""

    def __init__(self):
        super().__init__(NOT_STAGED_MESSAGE)


def feature_names() -> list[str]:
    """The features to train on: every column of the staged CSV (the ones
    the CSV wizard included). Raises NotStagedError when nothing is staged.
    """
    columns = staged_columns()
    if not columns:
        raise NotStagedError()
    return columns


def load_features(features) -> np.ndarray:
    """Read the ``features`` columns of the staged CSV as a float array.

    Empty and non-numeric cells become NaN; no other cleaning is done here.
    Raises NotStagedError when nothing is staged.
    """
    if not is_staged():
        raise NotStagedError()
    frame = pd.read_csv(SLOT_PATH)
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError(
            f"The staged CSV has no column(s) {missing}; "
            f"its columns are {list(frame.columns)}"
        )
    return (
        frame[list(features)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype="float64")
    )
