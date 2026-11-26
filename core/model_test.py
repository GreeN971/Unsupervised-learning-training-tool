"""Tests an already trained model: scores typed readings the way Edge
Impulse does, and measures with PR-AUC whether the model tells readings
outside the user's normal ranges from those inside.

Only reads a .tflite and the <name>_params.json next to it; training and
export are not involved. Scores come from the .tflite itself, so int8
rounding is included, as on the device.
"""

import json
import math
import re
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from converter.to_tflite import CALIBRATION_MARGIN, _interpreter, _run_tflite, presence_inputs

# Readings generated inside and outside the normal ranges each.
N_READINGS = 1000
# With more features than this the corners of the normal box (2^n) are skipped.
_MAX_CORNER_FEATURES = 8
# How far past a limit that lies outside the model's valid range to test, as
# a share of the feature's training range.
_BEYOND_VALID = 0.25
# Below this share of anomalies found, the rule is reported as not useful.
LOW_RULE_SHARE = 0.5

_NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


class ModelTestError(Exception):
    """The model or the input can't be tested; the message is for the user."""


@dataclass
class LoadedModel:
    path: Path
    tflite: bytes
    features: list[str]
    quantization: str
    normalized: bool  # takes values scaled to [0, 1]
    presence: bool  # takes value/flag pairs
    minimum: np.ndarray  # training range per feature
    maximum: np.ndarray
    valid_low: np.ndarray
    valid_high: np.ndarray
    trained_beyond: bool  # trained over the whole valid range
    threshold: float | None
    min_features_sent: int


def params_path(path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}_params.json")


def load(path) -> LoadedModel:
    path = Path(path)
    try:
        tflite = path.read_bytes()
    except OSError as exc:
        raise ModelTestError(f"The model can't be opened: {path} ({exc.strerror or exc})") from exc
    side = params_path(path)
    try:
        params = json.loads(side.read_text(encoding="utf-8"))
        features = [str(name) for name in params["feature_names"]]
        low = np.array(params["input_min"], dtype="float64")
        high = np.array(params["input_max"], dtype="float64")
        if not features or low.shape != (len(features),) or high.shape != low.shape:
            raise ValueError("feature names and ranges don't match")
    except FileNotFoundError as exc:
        raise ModelTestError(
            f"{side.name} is missing next to the model. It holds the feature "
            f"order and ranges, so the model can't be tested without it."
        ) from exc
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ModelTestError(f"{side.name} can't be read: {exc}") from exc

    try:
        interpreter = _interpreter(tflite)
        interpreter.allocate_tensors()
        inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
        n_inputs, n_outputs = int(inputs[0]["shape"][-1]), int(outputs[0]["shape"][-1])
        is_int8 = inputs[0]["dtype"] == np.int8
    except (ValueError, RuntimeError) as exc:
        raise ModelTestError(f"{path.name} is not a usable TFLite model: {exc}") from exc
    n = len(features)
    # From the model itself: models exported before the presence flags take
    # one plain value per feature.
    if len(inputs) != 1 or n_outputs != 1 or n_inputs not in (n, 2 * n):
        raise ModelTestError(
            f"{path.name} takes {n_inputs} inputs, which doesn't fit the {n} "
            f"feature(s) in {side.name}. They belong to different models."
        )
    valid = params.get("valid_input_range")
    if valid is not None:
        valid_low, valid_high = np.array(valid, dtype="float64").T
    else:
        margin = CALIBRATION_MARGIN * (high - low)
        valid_low, valid_high = low - margin, high + margin
    return LoadedModel(
        path=path, tflite=tflite, features=features,
        quantization="int8" if is_int8 else "float32",
        # Models from before the setting existed take normalized values.
        normalized=params.get("input_mode", "normalized") != "raw",
        presence=n_inputs == 2 * n,
        minimum=low, maximum=high, valid_low=valid_low, valid_high=valid_high,
        trained_beyond=valid is not None,
        threshold=params.get("threshold"),
        min_features_sent=int(params.get("min_features_sent", 1)),
    )


def parse_reading(text: str, model: LoadedModel) -> np.ndarray:
    """One reading typed like in Edge Impulse ("32,69"): a value per feature
    in order, an empty place for a missing one (NaN).
    """
    expected = ",".join(model.features)
    cells = [cell.strip() for cell in text.split(",")]
    n = len(model.features)
    if not any(cells):
        raise ModelTestError(f"Type a reading: {expected}")
    if len(cells) != n:
        raise ModelTestError(
            f"Expected {n} value{'s' if n > 1 else ''} ({expected}), got {len(cells)}. "
            f"Separate features with a comma and use a point for decimals."
        )
    values = np.full(n, np.nan)
    for i, cell in enumerate(cells):
        if not cell:
            continue
        if not _NUMBER.fullmatch(cell) or not math.isfinite(float(cell)):
            raise ModelTestError(f"'{cell}' ({model.features[i]}) is not a number.")
        values[i] = float(cell)
    sent = int((~np.isnan(values)).sum())
    if sent < n and not model.presence:
        raise ModelTestError(f"This model can't leave a feature out; it needs {expected}.")
    if sent < model.min_features_sent:
        raise ModelTestError(
            f"This model needs at least {model.min_features_sent} features in a reading."
        )
    return values


def encode(model: LoadedModel, rows: np.ndarray) -> np.ndarray:
    """What the network takes for raw feature ``rows`` (NaN = missing)."""
    x = np.atleast_2d(np.asarray(rows, dtype="float64"))
    if model.normalized:
        spread = model.maximum - model.minimum
        x = (x - model.minimum) / np.where(spread == 0, 1.0, spread)
    if model.presence:
        return presence_inputs(x, range(len(model.features)))
    return x.astype("float32")


def score(model: LoadedModel, rows: np.ndarray) -> np.ndarray:
    return _run_tflite(model.tflite, encode(model, rows)).ravel().astype("float64")


def edge_impulse_text(model: LoadedModel, values: np.ndarray) -> str:
    """What to type into Edge Impulse for this reading: the network's own
    inputs, which include the presence flags of a multi-feature model.
    """
    return ",".join(np.format_float_positional(v, trim="-") for v in encode(model, values)[0])


@dataclass
class RangeResult:
    pr_auc: float
    n_normal: int
    n_anomalies: int
    rule: float
    found_by_rule: int
    threshold: float | None
    found_by_threshold: int = 0
    flagged_by_threshold: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def baseline(self) -> float:
        return self.n_anomalies / (self.n_anomalies + self.n_normal)

    @property
    def low_rule(self) -> bool:
        return self.found_by_rule < LOW_RULE_SHARE * self.n_anomalies


def _ranges(model, limits):
    """Per feature the normal range and the segments outside it."""
    normal, outside, warnings = [], [], []
    for i, name in enumerate(model.features):
        low, high = limits.get(name, (None, None))
        lo = float(model.minimum[i] if low is None else low)
        hi = float(model.maximum[i] if high is None else high)
        if not lo < hi and (low is not None or high is not None):
            if low is not None and high is not None:
                raise ModelTestError(f"{name}: 'from' must be lower than 'to'.")
            raise ModelTestError(
                f"{name}: the limit leaves no normal range (trained on "
                f"{model.minimum[i]:.6g} .. {model.maximum[i]:.6g}). Set the other side too."
            )
        span = float(model.maximum[i] - model.minimum[i]) or 1.0
        bottom, top = float(model.valid_low[i]), float(model.valid_high[i])
        segments, beyond = [], False
        if low is not None:
            edge = bottom if lo > bottom else lo - _BEYOND_VALID * span
            beyond |= lo <= bottom
            segments.append((edge, lo))
        if high is not None:
            edge = top if hi < top else hi + _BEYOND_VALID * span
            beyond |= hi >= top
            segments.append((hi, edge))
        if beyond:
            warnings.append(
                f"{name}: a limit lies outside {bottom:.6g} .. {top:.6g}, the range the "
                f"model is valid for. Scores there are not reliable."
            )
        elif segments and not model.trained_beyond:
            warnings.append(
                f"{name}: this older model was only trained within "
                f"{model.minimum[i]:.6g} .. {model.maximum[i]:.6g}; scores outside are not reliable."
            )
        normal.append((lo, hi))
        outside.append(segments)
    if not any(outside):
        raise ModelTestError("Set at least one limit, otherwise nothing counts as an anomaly.")
    return normal, outside, warnings


def _even(segments, count):
    """``count`` values spread evenly over each segment, limits excluded."""
    parts = np.array_split(np.arange(count), len(segments))
    return np.concatenate([
        lo + (hi - lo) * (np.arange(len(part)) + 0.5) / len(part)
        for (lo, hi), part in zip(segments, parts)
    ])


def generate(model: LoadedModel, limits: dict, seed=0):
    """Readings inside the normal ranges and readings with at least one
    value outside, N_READINGS of each, all features sent. ``limits`` maps a
    feature to its (from, to), either of which may be None.
    """
    normal, outside, warnings = _ranges(model, limits)
    n = len(model.features)
    if n == 1:
        inside = np.linspace(*normal[0], N_READINGS)[:, None]
        return inside, _even(outside[0], N_READINGS)[:, None], warnings

    rng = np.random.default_rng(seed)
    low, high = np.array(normal).T
    inside = rng.uniform(low, high, (N_READINGS, n))
    if n <= _MAX_CORNER_FEATURES:
        corners = np.array(list(product(*normal)))
        inside[:len(corners)] = corners
    anomalies = rng.uniform(low, high, (N_READINGS, n))
    limited = [i for i in range(n) if outside[i]]
    for row in anomalies:
        for i in rng.choice(limited, rng.integers(1, len(limited) + 1), replace=False):
            lo, hi = outside[i][rng.integers(len(outside[i]))]
            row[i] = lo + (hi - lo) * rng.uniform(0.0005, 0.9995)  # off the limit itself
    return inside, anomalies, warnings


def summarize(normal_scores, anomaly_scores, threshold) -> RangeResult:
    normal_scores = np.asarray(normal_scores, dtype="float64")
    anomaly_scores = np.asarray(anomaly_scores, dtype="float64")
    labels = np.r_[np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))]
    # Rounded up as displayed: no tested normal reading may exceed it.
    rule = math.ceil(float(normal_scores.max()) * 1e4) / 1e4
    result = RangeResult(
        pr_auc=float(average_precision_score(labels, np.r_[normal_scores, anomaly_scores])),
        n_normal=len(normal_scores), n_anomalies=len(anomaly_scores),
        rule=rule, found_by_rule=int((anomaly_scores > rule).sum()), threshold=threshold,
    )
    if threshold is not None:
        result.found_by_threshold = int((anomaly_scores > threshold).sum())
        result.flagged_by_threshold = int((normal_scores > threshold).sum())
    return result


def test_ranges(model: LoadedModel, limits: dict) -> RangeResult:
    inside, anomalies, warnings = generate(model, limits)
    result = summarize(score(model, inside), score(model, anomalies), model.threshold)
    result.warnings = warnings
    return result
