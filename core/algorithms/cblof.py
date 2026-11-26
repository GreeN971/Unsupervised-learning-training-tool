"""CBLOF training + .tflite export.

Training (see core.common.TrainingSpec): k-means cluster distances: robust-
scaled features.
"""

from pathlib import Path

from core.common import TrainingSpec, base_default_params, fit_and_export, repo_root_on_syspath

repo_root_on_syspath(__file__, 2)

from pyod.models.cblof import CBLOF

from core.configs import cblof as _config

DEFAULT_PARAMS = {
    "CONTAMINATION": _config.CONTAMINATION,
    "N_CLUSTERS": _config.N_CLUSTERS,
    "ALPHA": _config.ALPHA,
    "BETA": _config.BETA,
    "USE_WEIGHTS": _config.USE_WEIGHTS,
    **base_default_params(_config),
}

TRAINING = TrainingSpec(scaling="robust")


def run(params: dict, output_path: Path, progress_callback=None,
        cancel_event=None) -> Path:
    """Train CBLOF on the staged dataset and export .tflite."""
    p = {**DEFAULT_PARAMS, **params}

    def make_detector(n_columns):
        return CBLOF(
            contamination=p["CONTAMINATION"],
            n_clusters=p["N_CLUSTERS"],
            alpha=p["ALPHA"],
            beta=p["BETA"],
            use_weights=p["USE_WEIGHTS"],
            random_state=p["RANDOM_STATE"],
        )

    return fit_and_export(make_detector, TRAINING, p, output_path,
                          progress_callback, cancel_event=cancel_event)
