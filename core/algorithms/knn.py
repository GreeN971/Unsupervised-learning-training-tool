"""KNN training + .tflite export.

Training (see core.common.TrainingSpec): Distance-based: robust-scaled
features, and dequantized fit data so rounded sensor values don't give zero
neighbor distances.
"""

from pathlib import Path

from core.common import TrainingSpec, base_default_params, fit_and_export, repo_root_on_syspath

repo_root_on_syspath(__file__, 2)

from pyod.models.knn import KNN

from core.configs import knn as _config

DEFAULT_PARAMS = {
    "CONTAMINATION": _config.CONTAMINATION,
    "N_NEIGHBORS": _config.N_NEIGHBORS,
    "METHOD": _config.METHOD,
    **base_default_params(_config),
}

TRAINING = TrainingSpec(scaling="robust", dequantize=True)


def run(params: dict, output_path: Path, progress_callback=None,
        cancel_event=None) -> Path:
    """Train KNN on the staged dataset and export .tflite."""
    p = {**DEFAULT_PARAMS, **params}

    def make_detector(n_columns):
        return KNN(
            contamination=p["CONTAMINATION"],
            n_neighbors=p["N_NEIGHBORS"],
            method=p["METHOD"],
        )

    return fit_and_export(make_detector, TRAINING, p, output_path,
                          progress_callback, cancel_event=cancel_event)
