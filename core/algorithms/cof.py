"""COF training + .tflite export, extracted from algorithms/cof/cof.ipynb."""

from pathlib import Path

from core.common import base_default_params, fit_and_export, repo_root_on_syspath

repo_root_on_syspath(__file__, 2)

from pyod.models.cof import COF

from algorithms.cof import config as _config

DEFAULT_PARAMS = {
    "CONTAMINATION": _config.CONTAMINATION,
    "N_NEIGHBORS": _config.N_NEIGHBORS,
    "METHOD": _config.METHOD,
    **base_default_params(_config),
}


def run(params: dict, output_path: Path, progress_callback=None,
        cancel_event=None) -> Path:
    """Train COF on the configured dataset and export .tflite."""
    p = {**DEFAULT_PARAMS, **params}
    detector = COF(
        contamination=p["CONTAMINATION"],
        n_neighbors=p["N_NEIGHBORS"],
        method=p["METHOD"],
    )
    return fit_and_export(detector, _config, p, output_path, progress_callback,
                           cancel_event=cancel_event)
