"""ECOD training + .tflite export.

Training (see core.common.TrainingSpec): Per-feature empirical-CDF tails:
invariant to feature scaling. Uses core.novelty.InductiveECOD, because PyOD's
ECOD ranks a scored batch's values against each other.
"""

from pathlib import Path

from core.common import TrainingSpec, base_default_params, fit_and_export, repo_root_on_syspath

repo_root_on_syspath(__file__, 2)

from core.novelty import InductiveECOD as ECOD

from core.configs import ecod as _config

DEFAULT_PARAMS = {
    "CONTAMINATION": _config.CONTAMINATION,
    "N_JOBS": _config.N_JOBS,
    **base_default_params(_config),
}

TRAINING = TrainingSpec()


def run(params: dict, output_path: Path, progress_callback=None,
        cancel_event=None) -> Path:
    """Train ECOD on the staged dataset and export .tflite."""
    p = {**DEFAULT_PARAMS, **params}

    def make_detector(n_columns):
        return ECOD(
            contamination=p["CONTAMINATION"],
            # PyOD refuses n_jobs > 1 on a single column.
            n_jobs=p["N_JOBS"] if n_columns > 1 else 1,
        )

    return fit_and_export(make_detector, TRAINING, p, output_path,
                          progress_callback, cancel_event=cancel_event)
