"""LOF training + .tflite export.

Training (see core.common.TrainingSpec): Density ratio of k-neighborhoods:
robust-scaled features, and dequantized fit data so duplicate values don't
give degenerate (near-infinite) densities.
"""

from pathlib import Path

from core.common import TrainingSpec, base_default_params, fit_and_export, repo_root_on_syspath

repo_root_on_syspath(__file__, 2)

from pyod.models.lof import LOF

from core.configs import lof as _config

DEFAULT_PARAMS = {
    "CONTAMINATION": _config.CONTAMINATION,
    "N_NEIGHBORS": _config.N_NEIGHBORS,
    **base_default_params(_config),
}

TRAINING = TrainingSpec(scaling="robust", dequantize=True)


def run(params: dict, output_path: Path, progress_callback=None,
        cancel_event=None) -> Path:
    """Train LOF on the staged dataset and export .tflite."""
    p = {**DEFAULT_PARAMS, **params}

    def make_detector(n_columns):
        return LOF(
            contamination=p["CONTAMINATION"],
            n_neighbors=p["N_NEIGHBORS"],
        )

    return fit_and_export(make_detector, TRAINING, p, output_path,
                          progress_callback, cancel_event=cancel_event)
