"""OCSVM training + .tflite export, extracted from algorithms/ocsvm/ocsvm.ipynb."""

from pathlib import Path

from core.common import base_default_params, fit_and_export, repo_root_on_syspath

repo_root_on_syspath(__file__, 2)

from pyod.models.ocsvm import OCSVM

from algorithms.ocsvm import config as _config

DEFAULT_PARAMS = {
    "CONTAMINATION": _config.CONTAMINATION,
    "KERNEL": _config.KERNEL,
    "NU": _config.NU,
    "GAMMA": _config.GAMMA,
    "DEGREE": _config.DEGREE,
    **base_default_params(_config),
}


def run(params: dict, output_path: Path, progress_callback=None,
        cancel_event=None) -> Path:
    """Train OCSVM on the configured dataset and export .tflite."""
    p = {**DEFAULT_PARAMS, **params}
    detector = OCSVM(
        contamination=p["CONTAMINATION"],
        kernel=p["KERNEL"],
        nu=p["NU"],
        gamma=p["GAMMA"],
        degree=p["DEGREE"],
    )
    return fit_and_export(detector, _config, p, output_path, progress_callback,
                           cancel_event=cancel_event)
