"""OCSVM training + .tflite export.

Training (see core.common.TrainingSpec): Kernel distances (gamma="auto" is
1/n_features, which assumes unit-scale features): robust-scaled features.
"""

from pathlib import Path

from core.common import TrainingSpec, base_default_params, fit_and_export, repo_root_on_syspath

repo_root_on_syspath(__file__, 2)

from pyod.models.ocsvm import OCSVM

from core.configs import ocsvm as _config

DEFAULT_PARAMS = {
    "CONTAMINATION": _config.CONTAMINATION,
    "KERNEL": _config.KERNEL,
    "NU": _config.NU,
    "GAMMA": _config.GAMMA,
    "DEGREE": _config.DEGREE,
    **base_default_params(_config),
}

TRAINING = TrainingSpec(scaling="robust")


def run(params: dict, output_path: Path, progress_callback=None,
        cancel_event=None) -> Path:
    """Train OCSVM on the staged dataset and export .tflite."""
    p = {**DEFAULT_PARAMS, **params}

    def make_detector(n_columns):
        return OCSVM(
            contamination=p["CONTAMINATION"],
            kernel=p["KERNEL"],
            nu=p["NU"],
            gamma=p["GAMMA"],
            degree=p["DEGREE"],
        )

    return fit_and_export(make_detector, TRAINING, p, output_path,
                          progress_callback, cancel_event=cancel_event)
