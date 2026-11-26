"""In-memory wizard state, shared across the picker -> configure -> save ->
run pages. This is a single-window native app with one user at a time, so a
plain module-level dict is enough - no need for per-session storage.
"""


def _default_topology() -> dict:
    return {
        "type": "sequential",
        # Chosen architecture per model type.
        "architecture": {"sequential": "mlp", "functional": "residual"},
        # MLP layer-widths text ("" = the algorithm's default).
        "layers": {"mlp": ""},
        # Residual settings; see converter.to_tflite._build_residual_mlp.
        # S = width, B = n_blocks, D = block_depth, h = expansion_factor.
        "residual": {"width": 16, "n_blocks": 2, "block_depth": 2, "expansion_factor": 1.0},
    }


STATE: dict = {
    # Chosen data file from the first (Data) step. Not cleared by reset(),
    # which runs when the Algorithm step loads.
    "data_path": None,
    # Columns included in the CSV wizard (staged into core.temp_csv) and
    # the file they were chosen for; likewise kept by reset().
    "features": None,
    "features_path": None,
    "algorithm_key": None,
    "params": {},
    "topology": _default_topology(),
    "output_path": None,
}


def reset() -> None:
    STATE["algorithm_key"] = None
    STATE["params"] = {}
    STATE["topology"] = _default_topology()
    STATE["output_path"] = None
