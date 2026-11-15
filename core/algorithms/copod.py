"""COPOD training + .tflite export, extracted from algorithms/copod/copod.ipynb."""

from pathlib import Path

from core.common import base_default_params, fit_and_export, repo_root_on_syspath

repo_root_on_syspath(__file__, 2)

from pyod.models.copod import COPOD

from algorithms.copod import config as _config

DEFAULT_PARAMS = {
    "CONTAMINATION": _config.CONTAMINATION,
    "N_JOBS": _config.N_JOBS,
    **base_default_params(_config),
}


def run(params: dict, output_path: Path, progress_callback=None,
        cancel_event=None) -> Path:
    """Train COPOD on the configured dataset and export .tflite."""
    p = {**DEFAULT_PARAMS, **params}
    detector = COPOD(
        contamination=p["CONTAMINATION"],
        n_jobs=p["N_JOBS"],
    )
    return fit_and_export(detector, _config, p, output_path, progress_callback,
                           cancel_event=cancel_event)
