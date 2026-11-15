"""LODA training + .tflite export, extracted from algorithms/loda/loda.ipynb."""

from pathlib import Path

from core.common import base_default_params, fit_and_export, repo_root_on_syspath

repo_root_on_syspath(__file__, 2)

from pyod.models.loda import LODA

from algorithms.loda import config as _config

DEFAULT_PARAMS = {
    "CONTAMINATION": _config.CONTAMINATION,
    "N_BINS": _config.N_BINS,
    "N_RANDOM_CUTS": _config.N_RANDOM_CUTS,
    **base_default_params(_config),
}


def run(params: dict, output_path: Path, progress_callback=None,
        cancel_event=None) -> Path:
    """Train LODA on the configured dataset and export .tflite."""
    p = {**DEFAULT_PARAMS, **params}
    detector = LODA(
        contamination=p["CONTAMINATION"],
        n_bins=p["N_BINS"],
        n_random_cuts=p["N_RANDOM_CUTS"],
    )
    return fit_and_export(detector, _config, p, output_path, progress_callback,
                           cancel_event=cancel_event)
