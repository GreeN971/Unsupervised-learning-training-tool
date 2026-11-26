"""Shared training/export logic for every detector.

Each core/algorithms/<name>.py module declares how to build its detector
(a factory taking the number of feature columns) and a TrainingSpec saying
how that detector family has to be fed. This module turns the staged CSV
into training targets for the surrogate network and hands off to
converter.to_tflite.

Per presence pattern (subset of feature columns the device sends; a single
pattern with one feature column):

1. the detector is fitted in its own feature space (TrainingSpec.scaling,
   TrainingSpec.dequantize) on the training rows that have the pattern's
   columns;
2. its raw scores are taken for the training rows (PyOD's in-sample
   ``decision_scores_``, each point left out of its own neighborhood), the
   test rows, and synthetic coverage rows spread over and slightly beyond
   the training range;
3. a ScoreCalibrator maps the raw scores so the contamination quantile of
   the training scores lands at THRESHOLD (0.5): the device flags a sample
   as anomalous when the network's output exceeds 0.5, whatever the
   algorithm or the pattern.
"""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from converter.to_tflite import CALIBRATION_MARGIN, THRESHOLD, Pattern, to_tflite
from core import temp_csv

# Fewest training rows a presence pattern needs (all of its features
# present) for its detector to be fitted meaningfully.
MIN_PATTERN_ROWS = 50

# Synthetic coverage rows per pattern: as many as the pattern's training
# rows, capped. They extend COVERAGE_MARGIN of each column's range beyond
# its training min and max - the same margin the int8 input range is
# calibrated on (converter.to_tflite.CALIBRATION_MARGIN), i.e. the input
# range the exported model is valid for.
MAX_SYNTHETIC_ROWS = 5000
COVERAGE_MARGIN = CALIBRATION_MARGIN

# Training-score quantile mapped to 1.0. Higher scores keep rising above 1.
UPPER_QUANTILE = 0.995

# Highest calibrated score: what the detector's highest raw score over the
# model's valid input range maps to. Detectors that measure distances can
# score far-out values a hundred times higher than the threshold; without
# this bound those values would use up the int8 output's 256 levels and
# dominate the loss, at the expense of the scores around THRESHOLD.
TOP_SCORE = 2.0


class TrainingCancelled(Exception):
    """Raised when a run is interrupted via its cancel_event.

    Only checked at cooperative checkpoints (before starting, between
    presence patterns, and at each epoch boundary of the surrogate network's
    training loop) - it can't preempt a detector's own blocking .fit() call,
    so on slow classical detectors (e.g. OCSVM/KNN on a large dataset)
    cancellation takes effect only once that step returns.
    """


@dataclass(frozen=True)
class TrainingSpec:
    """How a detector family has to be fed (see the table in each
    core/algorithms module's docstring).

    ``scaling``: "robust" fits a median/IQR scaler per column on the
    training rows and gives the detector scaled features, for detectors
    that measure distances or project across features (otherwise the
    feature with the largest units dominates). "none" for detectors that
    are invariant to per-feature scaling or standardize internally.

    ``dequantize``: add a deterministic jitter of +-half the column's
    resolution (smallest gap between distinct values) to the detector's fit
    data. Neighbor-based detectors need it on rounded sensor data, where
    many identical values give zero neighbor distances and degenerate
    densities.

    ``min_pattern_columns``: fewest feature columns a presence pattern needs
    for the detector to be defined on it (SOD selects a subspace, which a
    single column doesn't have). Smaller patterns are not trained, so the
    device has to send at least that many features.
    """

    scaling: str = "none"
    dequantize: bool = False
    min_pattern_columns: int = 1


def _check_cancelled(cancel_event, output_path):
    if cancel_event is not None and cancel_event.is_set():
        raise TrainingCancelled(f"Training cancelled for {output_path.stem}")


class ScoreCalibrator:
    """Piecewise-linear, increasing map of raw detector scores:

    - ``lo`` (min training score) -> 0
    - ``thr`` (training-score quantile 1 - contamination) -> THRESHOLD (0.5)
    - ``hi`` (training-score quantile UPPER_QUANTILE) -> 1
    - ``top`` (highest score over the training and synthetic coverage rows)
      -> ``top_score``, at most TOP_SCORE

    Above ``hi`` scores go on rising at the same slope, or at the flatter one
    that reaches TOP_SCORE at ``top``: more extreme values stay
    distinguishable over the whole valid input range, so a rule taken from a
    normal range at or beyond the data's edge (core.model_test) still works.
    Only scores below ``lo`` and above ``top`` are clipped.
    """

    def __init__(self, contamination):
        self.contamination = float(contamination)

    def fit(self, raw_scores, label, coverage_scores=()):
        """``raw_scores`` are the training rows' scores and
        ``coverage_scores`` the synthetic rows' (only used for ``top``).
        Raises ValueError when the training scores are non-finite or too tied
        to place the threshold (``label`` names the feature subset).
        """
        raw_scores = np.asarray(raw_scores, dtype="float64")
        lo = float(raw_scores.min()) if raw_scores.size else np.nan
        thr = float(np.quantile(raw_scores, 1 - self.contamination)) if raw_scores.size else np.nan
        if not np.all(np.isfinite(raw_scores)) or not thr > lo:
            raise ValueError(
                f"The detector gives constant, too tied or invalid anomaly "
                f"scores on {label}, so it can't be trained on that feature "
                f"subset"
            )
        hi = float(np.quantile(raw_scores, UPPER_QUANTILE))
        if not hi > thr:
            hi = float(raw_scores.max())
        if not hi > thr:
            hi = thr + (thr - lo)
        top = max(float(raw_scores.max()), float(np.max(coverage_scores, initial=lo)))
        same_slope = 1.0 + (1.0 - THRESHOLD) * (top - hi) / (hi - thr)
        self.lo, self.thr, self.hi = lo, thr, hi
        self.top = max(top, hi)
        self.top_score = min(TOP_SCORE, max(1.0, same_slope))
        return self

    def transform(self, raw_scores):
        raw, calibrated = [self.lo, self.thr, self.hi], [0.0, THRESHOLD, 1.0]
        if self.top > self.hi:
            raw.append(self.top)
            calibrated.append(self.top_score)
        return np.interp(raw_scores, raw, calibrated).astype("float32")

    def to_dict(self):
        return {"lo": self.lo, "thr": self.thr, "hi": self.hi,
                "top": self.top, "top_score": self.top_score,
                "contamination": self.contamination}


def _resolution(column):
    """Smallest gap between distinct values of ``column`` (0 if constant)."""
    values = np.unique(column)
    if len(values) < 2:
        return 0.0
    gaps = np.diff(values)
    gaps = gaps[gaps > 1e-9 * (values[-1] - values[0])]
    return float(gaps.min()) if len(gaps) else 0.0


def _dequantize(x, rng):
    resolution = np.array([_resolution(x[:, c]) for c in range(x.shape[1])])
    return x + rng.uniform(-0.5, 0.5, x.shape) * resolution


def _decision_scores(detector, x):
    """Raw scores of new rows ``x``."""
    return np.asarray(detector.decision_function(x), dtype="float64").ravel()


def _synthetic_rows(x, rng):
    """Uniform rows over each column's training range, widened by
    COVERAGE_MARGIN on both sides: they teach the surrogate the detector's
    behavior in the sparse tails and beyond the data, where the training
    rows alone say little and anomalies actually occur.
    """
    lo, hi = x.min(axis=0), x.max(axis=0)
    margin = COVERAGE_MARGIN * (hi - lo)
    n_rows = min(len(x), MAX_SYNTHETIC_ROWS)
    return rng.uniform(lo - margin, hi + margin, (n_rows, x.shape[1]))


def _pattern_targets(make_detector, spec, x_train, x_test, contamination,
                     random_state, label):
    """Fit one detector on ``x_train`` (complete rows of one pattern's
    columns) and return the calibrated targets: ``(synthetic_rows,
    train_scores, test_scores, synthetic_scores, calibrator)``.
    """
    rng = np.random.default_rng(random_state)
    scaler = RobustScaler().fit(x_train) if spec.scaling == "robust" else None

    def detector_space(x):
        return scaler.transform(x) if scaler is not None else x

    fit_x = detector_space(x_train)
    if spec.dequantize:
        fit_x = _dequantize(fit_x, rng)
    detector = make_detector(x_train.shape[1])
    detector.fit(fit_x)

    synthetic = _synthetic_rows(x_train, rng)
    train_raw = np.asarray(detector.decision_scores_, dtype="float64").ravel()
    test_raw = _decision_scores(detector, detector_space(x_test))
    synthetic_raw = _decision_scores(detector, detector_space(synthetic))

    if not (np.all(np.isfinite(test_raw)) and np.all(np.isfinite(synthetic_raw))):
        raise ValueError(f"The detector gives invalid anomaly scores on {label}")
    calibrator = ScoreCalibrator(contamination).fit(train_raw, label, synthetic_raw)
    return (synthetic, calibrator.transform(train_raw), calibrator.transform(test_raw),
            calibrator.transform(synthetic_raw), calibrator)


def _patterns(make_detector, spec, x_train, x_test, feature_names, p,
              cancel_event, output_path):
    """Surrogate training data for every presence pattern.

    A pattern is a non-empty subset S of the feature columns: the features
    the device sends. For each S a fresh detector is fitted on just those
    columns, over the rows that have all of them (NaN = missing), and its
    scores are calibrated on their own - so a score means "how unusual for
    what was sent", whatever the number of features sent. With one feature
    column there is exactly one pattern. Patterns with fewer than
    ``spec.min_pattern_columns`` columns are skipped.
    """
    present_train = ~np.isnan(x_train)
    present_test = ~np.isnan(x_test)
    n_features = x_train.shape[1]
    if n_features < spec.min_pattern_columns:
        raise ValueError(
            f"This algorithm needs at least {spec.min_pattern_columns} feature "
            f"columns, the staged CSV has {n_features}"
        )
    patterns = []
    for size in range(spec.min_pattern_columns, n_features + 1):
        for cols in combinations(range(n_features), size):
            _check_cancelled(cancel_event, output_path)
            label = "+".join(feature_names[c] for c in cols)
            train_rows = np.flatnonzero(present_train[:, cols].all(axis=1))
            test_rows = np.flatnonzero(present_test[:, cols].all(axis=1))
            if len(train_rows) < MIN_PATTERN_ROWS or len(test_rows) == 0:
                raise ValueError(
                    f"Pattern {label} has only {len(train_rows)} training and "
                    f"{len(test_rows)} test rows with all of its values present "
                    f"(needs at least {MIN_PATTERN_ROWS} and 1)"
                )
            cols = list(cols)
            synthetic, train_scores, test_scores, synthetic_scores, calibrator = (
                _pattern_targets(
                    make_detector, spec,
                    x_train[np.ix_(train_rows, cols)], x_test[np.ix_(test_rows, cols)],
                    p["CONTAMINATION"], p["RANDOM_STATE"], label,
                )
            )
            # Full-width synthetic rows: the pattern's columns, NaN elsewhere.
            synthetic_full = np.full((len(synthetic), n_features), np.nan)
            synthetic_full[:, cols] = synthetic
            patterns.append(Pattern(
                present=tuple(cols),
                name=label,
                x_train=np.vstack([x_train[train_rows], synthetic_full]),
                y_train=np.concatenate([train_scores, synthetic_scores]),
                x_test=x_test[test_rows],
                y_test=test_scores,
                calibration=calibrator.to_dict(),
            ))
    return patterns


def fit_and_export(make_detector, spec, p, output_path,
                   progress_callback=None, cancel_event=None):
    """Fit the detector on the staged dataset and export
    .tflite/.keras/<name>_params.json and the Edge Impulse files to
    ``output_path`` (see converter.to_tflite.export).

    ``make_detector(n_columns)`` returns a fresh, unfitted detector for that
    many feature columns; ``spec`` is its TrainingSpec. Data comes from the
    temporary CSV slot (core.temp_csv) - every staged column is a feature.
    Empty or non-numeric cells count as missing; rows missing every feature
    are dropped. With one feature the network takes just its value. With
    several it takes value/flag pairs and is trained on every presence
    pattern (see ``_patterns`` and converter.to_tflite.presence_inputs), so
    the device can leave any feature out.

    ``cancel_event``, if given, is a threading.Event checked cooperatively;
    see TrainingCancelled.
    """
    output_path = Path(output_path)
    _check_cancelled(cancel_event, output_path)

    feature_names = temp_csv.feature_names()
    features = temp_csv.load_features(feature_names)
    features = features[~np.isnan(features).all(axis=1)]
    x_train, x_test = train_test_split(
        features, test_size=p["TEST_SIZE"], random_state=p["RANDOM_STATE"]
    )

    patterns = _patterns(make_detector, spec, x_train, x_test, feature_names, p,
                         cancel_event, output_path)
    _check_cancelled(cancel_event, output_path)

    to_tflite(
        name=output_path.stem,
        patterns=patterns,
        x_reference=x_train,
        feature_names=feature_names,
        layers=p["LAYERS"],
        learning_rate=p["LEARNING_RATE"],
        loss=p["LOSS"],
        optimizer=p["SURROGATE_OPTIMIZER"],
        architecture=p["ARCHITECTURE"],
        architecture_params=p["ARCHITECTURE_PARAMS"],
        quantization=p["QUANTIZATION"],
        input_mode=p["INPUT_MODE"],
        epochs=p["EPOCHS"],
        batch_size=p["BATCH_SIZE"],
        verbose=p["VERBOSE"],
        random_state=p["RANDOM_STATE"],
        output_dir=output_path.parent,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
    _check_cancelled(cancel_event, output_path)
    return output_path


def base_default_params(config_module) -> dict:
    """The params shared by every detector's config."""
    return {
        "TEST_SIZE": config_module.TEST_SIZE,
        "RANDOM_STATE": config_module.RANDOM_STATE,
        "LAYERS": config_module.LAYERS,
        "LEARNING_RATE": config_module.LEARNING_RATE,
        "LOSS": "mse",
        "SURROGATE_OPTIMIZER": "adam",
        "ARCHITECTURE": "mlp",  # set from the GUI's Topology page
        "ARCHITECTURE_PARAMS": None,  # likewise; None = build from LAYERS
        "QUANTIZATION": "int8",
        "INPUT_MODE": "raw",
        "EPOCHS": config_module.EPOCHS,
        "BATCH_SIZE": config_module.BATCH_SIZE,
        "VERBOSE": config_module.VERBOSE,
    }


def repo_root_on_syspath(file: str, levels_up: int) -> Path:
    """Ensure the repo root is importable regardless of how the caller was
    launched.
    """
    import sys

    root = Path(file).resolve().parents[levels_up]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
