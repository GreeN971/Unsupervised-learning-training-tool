"""DeepSVDD training + .tflite export.

Training (see core.common.TrainingSpec): PyOD's DeepSVDD standardizes its
inputs itself (preprocessing=True), so no extra scaling. It is PyTorch-
backed and can't be exported directly; like every other detector it is
reproduced by the surrogate network of the Topology page, one DeepSVDD per
presence pattern (each built for that pattern's number of columns).
"""

from pathlib import Path

from core.common import TrainingSpec, base_default_params, fit_and_export, repo_root_on_syspath

repo_root_on_syspath(__file__, 2)

from pyod.models.deep_svdd import DeepSVDD

from core.configs import deepsvdd as _config

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
    **base_default_params(_config),
}

TRAINING = TrainingSpec()


def run(params: dict, output_path: Path, progress_callback=None,
        cancel_event=None) -> Path:
    """Train DeepSVDD on the staged dataset and export .tflite."""
    p = {**DEFAULT_PARAMS, **params}

    def make_detector(n_columns):
        return DeepSVDD(
            n_features=n_columns,
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

    return fit_and_export(make_detector, TRAINING, p, output_path,
                          progress_callback, cancel_event=cancel_event)
