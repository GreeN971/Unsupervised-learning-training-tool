"""SOD training + .tflite export.

Training (see core.common.TrainingSpec): Shared-neighbor subspaces: robust-
scaled features and dequantized fit data. Uses core.novelty.InductiveSOD,
because PyOD's SOD scores a batch only against itself. A single feature has
no subspace to select (every score would be 0), so only feature subsets of
at least two columns are trained.
"""

from pathlib import Path

from core.common import TrainingSpec, base_default_params, fit_and_export, repo_root_on_syspath

repo_root_on_syspath(__file__, 2)

from core.novelty import InductiveSOD as SOD

from core.configs import sod as _config

DEFAULT_PARAMS = {
    "CONTAMINATION": _config.CONTAMINATION,
    "N_NEIGHBORS": _config.N_NEIGHBORS,
    "REF_SET": _config.REF_SET,
    "ALPHA": _config.ALPHA,
    **base_default_params(_config),
}

TRAINING = TrainingSpec(scaling="robust", dequantize=True, min_pattern_columns=2)


def run(params: dict, output_path: Path, progress_callback=None,
        cancel_event=None) -> Path:
    """Train SOD on the staged dataset and export .tflite."""
    p = {**DEFAULT_PARAMS, **params}

    def make_detector(n_columns):
        return SOD(
            contamination=p["CONTAMINATION"],
            n_neighbors=p["N_NEIGHBORS"],
            ref_set=p["REF_SET"],
            alpha=p["ALPHA"],
        )

    return fit_and_export(make_detector, TRAINING, p, output_path,
                          progress_callback, cancel_event=cancel_event)
