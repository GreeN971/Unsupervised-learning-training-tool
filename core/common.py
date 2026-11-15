"""Shared training/export logic for the PyOD detectors.

Every algorithms/<name>/<name>.ipynb notebook follows the same shape: fit a
detector, min-max scale its anomaly scores, and hand off to
converter.to_tflite. This factors that shape out so each core/algorithms/
<name>.py module only has to declare its detector construction and
parameter schema. DeepSVDD is the one exception (its own converter and
surrogate architecture), handled separately in core/algorithms/deepsvdd.py.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from converter.to_tflite import to_tflite


class TrainingCancelled(Exception):
    """Raised when a run is interrupted via its cancel_event.

    Only checked at cooperative checkpoints (before starting, and at each
    epoch boundary of the surrogate network's training loop) - it can't
    preempt a detector's own blocking .fit() call, so on slow classical
    detectors (e.g. OCSVM/KNN on a large dataset) cancellation takes effect
    only once that step returns.
    """


def _check_cancelled(cancel_event, output_path):
    if cancel_event is not None and cancel_event.is_set():
        raise TrainingCancelled(f"Training cancelled for {output_path.stem}")


def fit_and_export(detector, config_module, p, output_path,
                    progress_callback=None, cancel_event=None):
    """Fit a detector on the configured dataset, scale its scores, and
    export .tflite/.keras/<name>_params.json to ``output_path``.

    ``cancel_event``, if given, is a threading.Event checked cooperatively;
    see TrainingCancelled.
    """
    output_path = Path(output_path)
    _check_cancelled(cancel_event, output_path)

    features = pd.read_csv(config_module.CSV_PATH)[config_module.FEATURES].to_numpy()
    x_train, x_test = train_test_split(
        features, test_size=p["TEST_SIZE"], random_state=p["RANDOM_STATE"]
    )
    detector.fit(x_train)
    _check_cancelled(cancel_event, output_path)

    raw_scores = detector.decision_function(x_train).reshape(-1, 1)
    score_scaler = MinMaxScaler(feature_range=p["SCORE_RANGE"]).fit(raw_scores)
    train_scores = score_scaler.transform(raw_scores).ravel()
    test_scores = score_scaler.transform(
        detector.decision_function(x_test).reshape(-1, 1)
    ).ravel()

    to_tflite(
        name=output_path.stem,
        x_train=x_train,
        y_train=train_scores,
        x_test=x_test,
        y_test=test_scores,
        layers=p["LAYERS"],
        learning_rate=p["LEARNING_RATE"],
        loss=p["LOSS"],
        optimizer=p["SURROGATE_OPTIMIZER"],
        architecture=p["ARCHITECTURE"],
        architecture_params=p["ARCHITECTURE_PARAMS"],
        quantization=p["QUANTIZATION"],
        flag_fraction=p["CONTAMINATION"],
        epochs=p["EPOCHS"],
        batch_size=p["BATCH_SIZE"],
        verbose=p["VERBOSE"],
        feature_names=config_module.FEATURES,
        output_dir=output_path.parent,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
    _check_cancelled(cancel_event, output_path)
    return output_path


def base_default_params(config_module) -> dict:
    """The params shared by every detector's config.py."""
    return {
        "TEST_SIZE": config_module.TEST_SIZE,
        "RANDOM_STATE": config_module.RANDOM_STATE,
        "SCORE_RANGE": config_module.SCORE_RANGE,
        "LAYERS": config_module.LAYERS,
        "LEARNING_RATE": config_module.LEARNING_RATE,
        "LOSS": "mse",  # not present in config.py - the notebooks' fixed loss
        "SURROGATE_OPTIMIZER": "adam",  # likewise the notebooks' fixed optimizer
        "ARCHITECTURE": "mlp",  # set from the GUI's Topology page
        "ARCHITECTURE_PARAMS": None,  # likewise; None = build from LAYERS
        "QUANTIZATION": "int8",
        "EPOCHS": config_module.EPOCHS,
        "BATCH_SIZE": config_module.BATCH_SIZE,
        "VERBOSE": config_module.VERBOSE,
    }


def repo_root_on_syspath(file: str, levels_up: int) -> Path:
    """Ensure the repo root is importable regardless of how the caller was
    launched (mirrors the sys.path handling in algorithms/*/config.py).
    """
    import sys

    root = Path(file).resolve().parents[levels_up]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
