"""Algorithm limitation notices shown in the GUI.

Two kinds:

- static limitations (``AlgorithmEntry.limitations`` in core.registry):
  what an algorithm can't do whatever the data or settings;
- checks (this module): what goes wrong with the staged data and the
  chosen params, worked out before training. A blocking notice means the
  run would fail or produce a model that can't work, so the GUI won't start
  it; the others are warnings.
"""

from dataclasses import dataclass
from math import comb
from typing import Optional

import numpy as np

from converter.to_tflite import CALIBRATION_MARGIN, MAX_FLAG_ERROR, MAX_INPUT_RESOLUTION
from core.common import MIN_PATTERN_ROWS
from core.temp_csv import StagedSummary

# Above this many feature subsets (= detectors fitted) a run is flagged as slow.
MANY_PATTERNS = 15

@dataclass(frozen=True)
class Notice:
    text: str
    blocking: bool = False


def _train_rows(params: dict, data: StagedSummary) -> int:
    """Training rows of the smallest presence pattern (all columns present)."""
    return int(data.complete_rows * (1 - params.get("TEST_SIZE", 0.2)))


def _int8_raw_input(params: dict, data: StagedSummary) -> list[Notice]:
    """Predict the int8 input resolution of a raw-input export, the way
    converter.to_tflite calibrates it: one scale covering 0, every feature's
    range widened by CALIBRATION_MARGIN and, with presence flags, the flag
    value 1.
    """
    if params.get("QUANTIZATION") != "int8" or params.get("INPUT_MODE") != "raw":
        return []
    lo, hi = np.array(data.minimum), np.array(data.maximum)
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return []
    spread = np.where(hi - lo == 0, 1.0, hi - lo)
    low = min(0.0, float((lo - CALIBRATION_MARGIN * spread).min()))
    high = max(0.0, float((hi + CALIBRATION_MARGIN * spread).max()))
    presence = len(data.columns) > 1
    if presence:
        high = max(high, 1.0)
    scale = (high - low) / 255
    notices = []
    coarse = [name for name, s in zip(data.columns, spread) if scale / s > MAX_INPUT_RESOLUTION]
    if coarse:
        notices.append(Notice(
            f"int8 with raw input will be coarse for {', '.join(coarse)}: one int8 "
            f"step of about {scale:.3g} has to cover 0 and every feature's range, "
            f"which is more than {MAX_INPUT_RESOLUTION:.0%} of this feature's spread. "
            f"Choose model input 'Normalized' or quantization 'None (float32)'."
        ))
    if presence and abs(np.round(1 / scale) * scale - 1) > MAX_FLAG_ERROR:
        notices.append(Notice(
            "int8 with raw input can't represent the presence flag 1 accurately "
            "next to these raw values. Choose model input 'Normalized' or "
            "quantization 'None (float32)'."
        ))
    return notices


def common_checks(params: dict, data: Optional[StagedSummary],
                  min_columns: int) -> list[Notice]:
    """Checks that apply to every algorithm."""
    notices = []
    if data is None:
        return notices
    n_columns = len(data.columns)
    if n_columns < min_columns:
        notices.append(Notice(
            f"Needs at least {min_columns} feature columns; the staged CSV has "
            f"{n_columns}. Include more columns in the CSV wizard.",
            blocking=True,
        ))
        return notices
    train_rows = _train_rows(params, data)
    if train_rows < MIN_PATTERN_ROWS:
        notices.append(Notice(
            f"Only about {train_rows} training rows have every feature present "
            f"(at least {MIN_PATTERN_ROWS} are needed). Include fewer columns, "
            f"or use a dataset with fewer missing values.",
            blocking=True,
        ))
    n_patterns = sum(comb(n_columns, size) for size in range(min_columns, n_columns + 1))
    if n_patterns > MANY_PATTERNS:
        notices.append(Notice(
            f"{n_columns} features give {n_patterns} feature subsets: one detector "
            f"is fitted per subset (so the device may leave any feature out), "
            f"which makes training slow."
        ))
    notices += _int8_raw_input(params, data)
    return notices


def _neighbors(params, data, fails_when_too_many):
    """k-nearest-neighbor detectors: k has to stay well below the rows."""
    if data is None:
        return []
    k, rows = params.get("N_NEIGHBORS", 0), _train_rows(params, data)
    if k >= rows:
        if fails_when_too_many:
            return [Notice(
                f"Number of neighbors ({k}) must be smaller than the training rows "
                f"of every feature subset (about {rows}).", blocking=True,
            )]
        return [Notice(
            f"Number of neighbors ({k}) is not smaller than the training rows "
            f"(about {rows}); it is cut to the rows - 1, where the scores lose "
            f"their meaning.",
        )]
    if k > 0.1 * rows:
        return [Notice(
            f"Number of neighbors ({k}) is more than 10% of the training rows "
            f"(about {rows}): the neighborhoods stop being local."
        )]
    return []


def knn(params, data):
    return _neighbors(params, data, fails_when_too_many=True)


def lof(params, data):
    return _neighbors(params, data, fails_when_too_many=False)


def cof(params, data):
    return _neighbors(params, data, fails_when_too_many=False)


def sod(params, data):
    notices = []
    if params.get("REF_SET", 0) >= params.get("N_NEIGHBORS", 0):
        notices.append(Notice(
            f"Reference set size ({params.get('REF_SET')}) must be smaller than "
            f"the number of neighbors ({params.get('N_NEIGHBORS')}).", blocking=True,
        ))
    return notices + _neighbors(params, data, fails_when_too_many=True)


def ocsvm(params, data):
    kernel = params.get("KERNEL")
    if kernel and kernel != "rbf":
        return [Notice(
            f"The '{kernel}' kernel separates a half-space instead of enclosing "
            f"the data: values on one side of the data are never flagged. Use "
            f"'rbf' to flag both too-low and too-high values."
        )]
    return []


def pca(params, data):
    components = params.get("N_COMPONENTS")
    if data is not None and components is not None and components > 1:
        return [Notice(
            f"Feature subsets with fewer than {components} columns keep all of "
            f"their columns ('Components kept' is capped per subset)."
        )]
    return []


def parallel(params, data):
    if params.get("N_JOBS", 1) > 1:
        return [Notice(
            "More than one parallel worker gives no speed-up here: one-column "
            "feature subsets always use one worker, and only the quick scoring "
            "of the training rows is shared between workers."
        )]
    return []


def deepsvdd(params, data, hidden_neurons):
    if params.get("USE_AE") and data is not None and len(data.columns) < min(hidden_neurons):
        return [Notice(
            f"The autoencoder variant needs at least {min(hidden_neurons)} feature "
            f"columns (PyOD requires as many features as neurons in the smallest "
            f"hidden layer, {list(hidden_neurons)}); the staged CSV has "
            f"{len(data.columns)}. Turn it off.",
            blocking=True,
        )]
    return []
