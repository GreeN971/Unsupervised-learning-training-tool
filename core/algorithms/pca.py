"""PCA training + .tflite export.

Training (see core.common.TrainingSpec): Standardizes internally
(STANDARDIZATION), so no extra scaling.
"""

from pathlib import Path

from core.common import TrainingSpec, base_default_params, fit_and_export, repo_root_on_syspath

repo_root_on_syspath(__file__, 2)

from pyod.models.pca import PCA

from core.configs import pca as _config

DEFAULT_PARAMS = {
    "CONTAMINATION": _config.CONTAMINATION,
    "N_COMPONENTS": _config.N_COMPONENTS,
    "WHITEN": _config.WHITEN,
    "WEIGHTED": _config.WEIGHTED,
    "STANDARDIZATION": _config.STANDARDIZATION,
    **base_default_params(_config),
}

TRAINING = TrainingSpec()


def run(params: dict, output_path: Path, progress_callback=None,
        cancel_event=None) -> Path:
    """Train PCA on the staged dataset and export .tflite."""
    p = {**DEFAULT_PARAMS, **params}

    def make_detector(n_columns):
        return PCA(
            contamination=p["CONTAMINATION"],
            # A pattern can have fewer columns than components asked for;
            # it then keeps all of its columns.
            n_components=(None if p["N_COMPONENTS"] is None
                          else min(p["N_COMPONENTS"], n_columns)),
            whiten=p["WHITEN"],
            weighted=p["WEIGHTED"],
            standardization=p["STANDARDIZATION"],
            random_state=p["RANDOM_STATE"],
        )

    return fit_and_export(make_detector, TRAINING, p, output_path,
                          progress_callback, cancel_event=cancel_event)
