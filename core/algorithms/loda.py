"""LODA training + .tflite export.

Training (see core.common.TrainingSpec): Random projections mix the
features, so they are robust-scaled first (otherwise the feature with the
largest units dominates every projection).
"""

from pathlib import Path

from core.common import TrainingSpec, base_default_params, fit_and_export, repo_root_on_syspath

repo_root_on_syspath(__file__, 2)

from pyod.models.loda import LODA

from core.configs import loda as _config

DEFAULT_PARAMS = {
    "CONTAMINATION": _config.CONTAMINATION,
    "N_BINS": _config.N_BINS,
    "N_RANDOM_CUTS": _config.N_RANDOM_CUTS,
    **base_default_params(_config),
}

TRAINING = TrainingSpec(scaling="robust")


def run(params: dict, output_path: Path, progress_callback=None,
        cancel_event=None) -> Path:
    """Train LODA on the staged dataset and export .tflite."""
    p = {**DEFAULT_PARAMS, **params}

    def make_detector(n_columns):
        return LODA(
            contamination=p["CONTAMINATION"],
            n_bins=p["N_BINS"],
            n_random_cuts=p["N_RANDOM_CUTS"],
            random_state=p["RANDOM_STATE"],
        )

    return fit_and_export(make_detector, TRAINING, p, output_path,
                          progress_callback, cancel_event=cancel_event)
