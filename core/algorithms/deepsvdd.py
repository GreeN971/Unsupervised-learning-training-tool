"""DeepSVDD training + .tflite export, extracted from
algorithms/deepsvdd/deepsvdd.ipynb.

Handled separately from the other detectors: PyOD's DeepSVDD is PyTorch-
backed, and its surrogate export (converter/to_tflite_for_deepsvdd.py) uses
a hypersphere-distance architecture instead of the plain regression head
the other detectors use.

Note: the original notebook references config.LATENT_DIM, which
algorithms/deepsvdd/config.py never actually defines - a pre-existing bug
in that notebook (it would raise AttributeError if run as-is). LATENT_DIM
is set here explicitly (32, matching the "e.g. 32 dimensions" comment in
the converter) rather than carried over from the broken config.
"""

from pathlib import Path

import pandas as pd
from pyod.models.deep_svdd import DeepSVDD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from core.common import TrainingCancelled, repo_root_on_syspath

repo_root_on_syspath(__file__, 2)

from algorithms.deepsvdd import config as _config
from converter.to_tflite_for_deepsvdd import to_tflite_deep_svdd

DEFAULT_PARAMS = {
    "CONTAMINATION": _config.CONTAMINATION,
    "USE_AE": _config.USE_AE,
    "HIDDEN_NEURONS": _config.HIDDEN_NEURONS,
    "HIDDEN_ACTIVATION": _config.HIDDEN_ACTIVATION,
    "OUTPUT_ACTIVATION": _config.OUTPUT_ACTIVATION,
    "OPTIMIZER": _config.OPTIMIZER,
    "DSVDD_EPOCHS": _config.DSVDD_EPOCHS,
    "DSVDD_BATCH_SIZE": _config.DSVDD_BATCH_SIZE,
    "DROPOUT_RATE": _config.DROPOUT_RATE,
    "L2_REGULARIZER": _config.L2_REGULARIZER,
    "TEST_SIZE": _config.TEST_SIZE,
    "RANDOM_STATE": _config.RANDOM_STATE,
    "SCORE_RANGE": _config.SCORE_RANGE,
    "LAYERS": _config.LAYERS,
    "LATENT_DIM": 32,  # not present in config.py - see module docstring
    "LEARNING_RATE": _config.LEARNING_RATE,
    "LOSS": "mse",
    "SURROGATE_OPTIMIZER": "adam",
    "QUANTIZATION": "int8",
    "EPOCHS": _config.EPOCHS,
    "BATCH_SIZE": _config.BATCH_SIZE,
    "VERBOSE": _config.VERBOSE,
}


def run(params: dict, output_path: Path, progress_callback=None,
        cancel_event=None) -> Path:
    """Train DeepSVDD on the configured dataset and export .tflite."""
    p = {**DEFAULT_PARAMS, **params}
    output_path = Path(output_path)

    if cancel_event is not None and cancel_event.is_set():
        raise TrainingCancelled(f"Training cancelled for {output_path.stem}")

    detector = DeepSVDD(
        n_features=len(_config.FEATURES),
        contamination=p["CONTAMINATION"],
        use_ae=p["USE_AE"],
        hidden_neurons=p["HIDDEN_NEURONS"],
        hidden_activation=p["HIDDEN_ACTIVATION"],
        output_activation=p["OUTPUT_ACTIVATION"],
        optimizer=p["OPTIMIZER"],
        epochs=p["DSVDD_EPOCHS"],
        batch_size=p["DSVDD_BATCH_SIZE"],
        dropout_rate=p["DROPOUT_RATE"],
        l2_regularizer=p["L2_REGULARIZER"],
        random_state=p["RANDOM_STATE"],
        verbose=p["VERBOSE"],
    )

    features = pd.read_csv(_config.CSV_PATH)[_config.FEATURES].to_numpy()
    x_train, x_test = train_test_split(
        features, test_size=p["TEST_SIZE"], random_state=p["RANDOM_STATE"]
    )
    detector.fit(x_train)
    if cancel_event is not None and cancel_event.is_set():
        raise TrainingCancelled(f"Training cancelled for {output_path.stem}")

    raw_scores = detector.decision_function(x_train).reshape(-1, 1)
    score_scaler = MinMaxScaler(feature_range=p["SCORE_RANGE"]).fit(raw_scores)
    train_scores = score_scaler.transform(raw_scores).ravel()
    test_scores = score_scaler.transform(
        detector.decision_function(x_test).reshape(-1, 1)
    ).ravel()

    to_tflite_deep_svdd(
        name=output_path.stem,
        x_train=x_train,
        y_train=train_scores,
        x_test=x_test,
        y_test=test_scores,
        layers=p["LAYERS"],
        latent_dim=p["LATENT_DIM"],
        learning_rate=p["LEARNING_RATE"],
        loss=p["LOSS"],
        optimizer=p["SURROGATE_OPTIMIZER"],
        quantization=p["QUANTIZATION"],
        flag_fraction=p.get("CONTAMINATION", 0.1),
        epochs=p["EPOCHS"],
        batch_size=p["BATCH_SIZE"],
        verbose=p["VERBOSE"],
        feature_names=_config.FEATURES,
        output_dir=output_path.parent,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
    if cancel_event is not None and cancel_event.is_set():
        raise TrainingCancelled(f"Training cancelled for {output_path.stem}")
    return output_path
