"""PCA training + .tflite export, extracted from algorithms/pca/pca.ipynb."""

from pathlib import Path

from core.common import base_default_params, fit_and_export, repo_root_on_syspath

repo_root_on_syspath(__file__, 2)

from pyod.models.pca import PCA

from algorithms.pca import config as _config

DEFAULT_PARAMS = {
    "CONTAMINATION": _config.CONTAMINATION,
    "N_COMPONENTS": _config.N_COMPONENTS,
    "WHITEN": _config.WHITEN,
    "WEIGHTED": _config.WEIGHTED,
    "STANDARDIZATION": _config.STANDARDIZATION,
    **base_default_params(_config),
}


def run(params: dict, output_path: Path, progress_callback=None,
        cancel_event=None) -> Path:
    """Train PCA on the configured dataset and export .tflite."""
    p = {**DEFAULT_PARAMS, **params}
    detector = PCA(
        contamination=p["CONTAMINATION"],
        n_components=p["N_COMPONENTS"],
        whiten=p["WHITEN"],
        weighted=p["WEIGHTED"],
        standardization=p["STANDARDIZATION"],
        random_state=p["RANDOM_STATE"],
    )
    return fit_and_export(detector, _config, p, output_path, progress_callback,
                           cancel_event=cancel_event)
