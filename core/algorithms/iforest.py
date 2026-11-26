"""Isolation Forest training + .tflite export.

Training (see core.common.TrainingSpec): Isolation trees split one feature
at a time on value order, so scaling the features changes nothing.
"""

from pathlib import Path

from core.common import TrainingSpec, base_default_params, fit_and_export, repo_root_on_syspath

repo_root_on_syspath(__file__, 2)

from pyod.models.iforest import IForest

from core.configs import iforest as _config

DEFAULT_PARAMS = {
    "CONTAMINATION": _config.CONTAMINATION,
    "N_ESTIMATORS": _config.N_ESTIMATORS,
    "MAX_SAMPLES": _config.MAX_SAMPLES,
    "MAX_FEATURES": _config.MAX_FEATURES,
    "BOOTSTRAP": _config.BOOTSTRAP,
    **base_default_params(_config),
}

TRAINING = TrainingSpec()


def run(params: dict, output_path: Path, progress_callback=None,
        cancel_event=None) -> Path:
    """Train Isolation Forest on the staged dataset and export .tflite."""
    p = {**DEFAULT_PARAMS, **params}

    def make_detector(n_columns):
        return IForest(
            contamination=p["CONTAMINATION"],
            n_estimators=p["N_ESTIMATORS"],
            max_samples=p["MAX_SAMPLES"],
            max_features=p["MAX_FEATURES"],
            bootstrap=p["BOOTSTRAP"],
            random_state=p["RANDOM_STATE"],
        )

    return fit_and_export(make_detector, TRAINING, p, output_path,
                          progress_callback, cancel_event=cancel_event)
