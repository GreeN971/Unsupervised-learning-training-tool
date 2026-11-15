"""CBLOF training + .tflite export, extracted from algorithms/cblof/cblof.ipynb."""

from pathlib import Path

from core.common import base_default_params, fit_and_export, repo_root_on_syspath

repo_root_on_syspath(__file__, 2)

from pyod.models.cblof import CBLOF

from algorithms.cblof import config as _config

DEFAULT_PARAMS = {
    "CONTAMINATION": _config.CONTAMINATION,
    "N_CLUSTERS": _config.N_CLUSTERS,
    "ALPHA": _config.ALPHA,
    "BETA": _config.BETA,
    "USE_WEIGHTS": _config.USE_WEIGHTS,
    **base_default_params(_config),
}


def run(params: dict, output_path: Path, progress_callback=None,
        cancel_event=None) -> Path:
    """Train CBLOF on the configured dataset and export .tflite."""
    p = {**DEFAULT_PARAMS, **params}
    detector = CBLOF(
        contamination=p["CONTAMINATION"],
        n_clusters=p["N_CLUSTERS"],
        alpha=p["ALPHA"],
        beta=p["BETA"],
        use_weights=p["USE_WEIGHTS"],
        random_state=p["RANDOM_STATE"],
    )
    return fit_and_export(detector, _config, p, output_path, progress_callback,
                           cancel_event=cancel_event)
