from nicegui import ui

from app import theme
from app.pages.configure import _add_spinner, _coerce  # same number-field look
from app.state import STATE
from core.registry import REGISTRY

# Architectures offered per model type; inner keys match
# converter.to_tflite.ARCHITECTURES.
ARCHITECTURES = {
    "sequential": {"mlp": "MLP"},
    "functional": {"residual": "Residual"},
}


# Residual settings fields: (key, label, type, step, min, max, description).
RESIDUAL_FIELDS = [
    ("width", "Width S", "int", 1, 1, 512,
     "Width S of the residual stream: a linear Dense lifts the input to S, "
     "and every block starts and ends at S, so each skip is an identity "
     "mapping."),
    ("n_blocks", "Number of blocks B", "int", 1, 1, 16,
     "Number of residual blocks B. Each block's output is added to its "
     "input (identity skip), followed by ReLU."),
    ("block_depth", "Dense layers per block D", "int", 1, 2, 8,
     "Dense layers D inside each block: D - 1 inner Dense-ReLU layers of "
     "round(h x S) units, then a linear Dense back to S. D = 2 is the classic "
     "Dense-ReLU-Dense block."),
    ("expansion_factor", "Expansion factor h", "float", 0.25, 0.25, 4,
     "Width of each block's inner Dense layers relative to S "
     "(h x S units); 1 = same as S. h x S must be a whole number."),
]

RESIDUAL_INVALID_MESSAGE = (
    "Fill in Width S, Number of blocks B, Dense layers per block D and "
    "Expansion factor h within their ranges, with h x S a whole number."
)


def residual_settings(residual: dict) -> dict:
    """Validated builder kwargs for _build_residual_mlp. Raises ValueError
    on an empty or out-of-range field, or when h x S is not a whole number
    (the inner width the MAC formula assumes must be buildable).
    """
    for key, _label, _kind, _step, lo, hi, _desc in RESIDUAL_FIELDS:
        value = residual.get(key)
        if value is None or not lo <= value <= hi:
            raise ValueError(f"{key} must be between {lo} and {hi}")
    if not float(residual["expansion_factor"] * residual["width"]).is_integer():
        raise ValueError("expansion_factor x width must be a whole number")
    return {
        "width": int(residual["width"]),
        "n_blocks": int(residual["n_blocks"]),
        "block_depth": int(residual["block_depth"]),
        "expansion_factor": float(residual["expansion_factor"]),
    }


def parse_layers(text: str) -> list[int]:
    """"16, 8," -> [16, 8]. Raises ValueError unless every entry is a
    positive whole number; empty entries (e.g. a trailing comma) are skipped.
    """
    units = [int(part) for part in text.split(",") if part.strip()]
    if any(u <= 0 for u in units):
        raise ValueError("layer widths must be positive")
    return units


# Size estimation, in bytes. float32 and int8 models are built differently, so
# each architecture has one estimate per format, with its own constants.

# float32 constants.
FLOAT32_PARAM_BYTES = 4  # F: weights and biases are both 32-bit floats
FLOAT32_LAYER_OVERHEAD = 450  # Clayer per Dense layer
FLOAT32_FIXED_OVERHEAD = 650  # Cfixed
# Cadd per ADD op (a residual skip; its ReLU is fused into the ADD, so no
# separate layer). ADD has no weights - only its operator, output tensor and
# name records. Measured: a float32 residual block costs ~1041 B beyond its
# weights+biases (TF 2.21), minus its two Dense layers at 450 each.
FLOAT32_ADD_OVERHEAD = 140

# int8 constants, measured on models from converter.to_tflite.convert_model
# (TF 2.21). Only the weights shrink to 1 B: biases stay int32, and the
# per-channel quantization of Dense layers stores extra data for every unit.
INT8_WEIGHT_BYTES = 1  # FW
INT8_BIAS_BYTES = 4  # FB
INT8_UNIT_OVERHEAD = 24  # Cq: scale (4 B) + zero point (8 B), for weights and bias
INT8_LAYER_OVERHEAD = 509  # C'layer per Dense layer
INT8_ADD_OVERHEAD = 160  # C'add per ADD op
INT8_FIXED_OVERHEAD = 724  # C'fixed


def _residual_parameters(width: int, n_blocks: int, block_depth: int,
                         expansion_factor: float, n_features: int) -> tuple[int, int]:
    """(weights, biases) of converter.to_tflite._build_residual_mlp.

    Structure counted (S = width, B = blocks, D = block depth,
    H = round(h * S)):
      stem   Dense(n_features -> S)              weights n_features*S, biases S
      block  widths S -> H x (D-1) -> S          weights 2*S*H + (D-2)*H*H,
                                                 biases (D-1)*H + S
             ADD (+ fused ReLU)                  no weights
      head   Dense(S -> 1)                       weights S,            biases 1

    E.g. S=16, B=2, D=2, h=1, one feature: weights 16 + 2*2*16*16 + 16 = 1056,
    biases 16 + 2*(16 + 16) + 1 = 81.
    """
    # As converter.to_tflite._hidden_units (not imported: it pulls in TensorFlow).
    hidden = max(1, round(expansion_factor * width))
    block_widths = [width] + [hidden] * (block_depth - 1) + [width]
    block_weights = sum(a * b for a, b in zip(block_widths, block_widths[1:]))
    block_biases = sum(block_widths[1:])
    weights = n_features * width + n_blocks * block_weights + width
    biases = width + n_blocks * block_biases + 1
    return weights, biases


def _residual_dense_layers(n_blocks: int, block_depth: int) -> int:
    """Dense layers of the residual MLP: stem + B*D in the blocks + head."""
    return n_blocks * block_depth + 2


def estimate_mlp_float32_bytes(layers: list[int]) -> int:
    """Estimated float32 MLP size, for hidden widths ``layers`` plus the
    Dense(1) output: F * (weights + biases) + dense_layers * Clayer + Cfixed.

    E.g. [32, 16, 8]: weights 32*16 + 16*8 + 8*1, biases 32 + 16 + 8 + 1,
    dense_layers 4.
    """
    widths = layers + [1]
    weights = sum(a * b for a, b in zip(widths, widths[1:]))
    biases = sum(widths)
    dense_layers = len(widths)  # hidden layers + the Dense(1) output
    return (
        FLOAT32_PARAM_BYTES * (weights + biases)
        + dense_layers * FLOAT32_LAYER_OVERHEAD
        + FLOAT32_FIXED_OVERHEAD
    )


def estimate_mlp_int8_bytes(layers: list[int], n_features: int) -> int:
    """Estimated int8 MLP size, for hidden widths ``layers`` plus the Dense(1)
    output: FW * weights + (FB + Cq) * biases + dense_layers * C'layer + C'fixed.

    Weights include the input to the first layer, n_features * layers[0].
    E.g. [32, 16, 8], one feature: weights 1*32 + 32*16 + 16*8 + 8*1 = 680,
    biases 32 + 16 + 8 + 1 = 57, dense_layers 4.
    """
    widths = layers + [1]
    weights = n_features * widths[0] + sum(a * b for a, b in zip(widths, widths[1:]))
    biases = sum(widths)
    dense_layers = len(widths)  # hidden layers + the Dense(1) output
    return (
        INT8_WEIGHT_BYTES * weights
        + (INT8_BIAS_BYTES + INT8_UNIT_OVERHEAD) * biases
        + dense_layers * INT8_LAYER_OVERHEAD
        + INT8_FIXED_OVERHEAD
    )


def estimate_residual_float32_bytes(width: int, n_blocks: int, block_depth: int,
                                    expansion_factor: float, n_features: int) -> int:
    """Estimated float32 residual MLP size:
    F * (weights + biases) + dense_layers * Clayer + adds * Cadd + Cfixed,
    where dense_layers = B*D + 2 and adds = B.
    """
    weights, biases = _residual_parameters(
        width, n_blocks, block_depth, expansion_factor, n_features
    )
    dense_layers = _residual_dense_layers(n_blocks, block_depth)
    return (
        FLOAT32_PARAM_BYTES * (weights + biases)
        + dense_layers * FLOAT32_LAYER_OVERHEAD
        + n_blocks * FLOAT32_ADD_OVERHEAD
        + FLOAT32_FIXED_OVERHEAD
    )


def estimate_residual_int8_bytes(width: int, n_blocks: int, block_depth: int,
                                 expansion_factor: float, n_features: int) -> int:
    """Estimated int8 residual MLP size:
    FW * weights + (FB + Cq) * biases + dense_layers * C'layer + adds * C'add
    + C'fixed, where dense_layers = B*D + 2 and adds = B.
    """
    weights, biases = _residual_parameters(
        width, n_blocks, block_depth, expansion_factor, n_features
    )
    dense_layers = _residual_dense_layers(n_blocks, block_depth)
    return (
        INT8_WEIGHT_BYTES * weights
        + (INT8_BIAS_BYTES + INT8_UNIT_OVERHEAD) * biases
        + dense_layers * INT8_LAYER_OVERHEAD
        + n_blocks * INT8_ADD_OVERHEAD
        + INT8_FIXED_OVERHEAD
    )


def size_estimate_text(topology: dict, quantization: str, n_features: int) -> str:
    """The size label's text, for Sequential MLP and Functional Residual, as
    the float32 or int8 model the QUANTIZATION config value will produce.
    """
    if n_features is None:  # nothing staged yet
        return "Size estimation in KiB: -"
    architecture = topology["architecture"][topology["type"]]
    is_int8 = quantization == "int8"
    try:
        if architecture == "residual":
            settings = residual_settings(topology["residual"])
            if is_int8:
                size = estimate_residual_int8_bytes(**settings, n_features=n_features)
            else:
                size = estimate_residual_float32_bytes(**settings, n_features=n_features)
        else:
            layers = parse_layers(topology["layers"][architecture])
            if not layers:
                return "Size estimation in KiB: -"
            if is_int8:
                size = estimate_mlp_int8_bytes(layers, n_features)
            else:
                size = estimate_mlp_float32_bytes(layers)
    except ValueError:
        return "Size estimation in KiB: -"
    return f"Size estimation in KiB: {size / 1024:.2f}"


# MAC (multiply-accumulate operations) per inference of one sample, after the
# thesis formulas. A Dense layer from a to b units costs a*b MAC; a residual
# skip (ADD) costs S_skip = S additions per block; biases and activations
# cost 0 MAC.
MAC_OUTPUTS = 1  # V_out: the network's single anomaly-score output
MAC_DISCLAIMER = (
    "B is being set to 1. If input will be more than one it will be "
    "calculated manually"
)
# Residual block layers by role (Vzorec 1): one expanding layer S -> E*S and
# one shrinking layer E*S -> S; the rest (N_mid = D - N_exp - N_shr) keep E*S.
N_EXP = 1
N_SHR = 1


def mac_mlp(layers: list[int], n_features: int, n_outputs: int = MAC_OUTPUTS) -> int:
    """MAC of the MLP (Equation 5):
    V_in * Lay_1 + V_out * Lay_n + sum_{i=1}^{n-1} Lay_i * Lay_{i+1},
    where Lay are the hidden widths ``layers``, n = len(layers),
    V_in = ``n_features`` and V_out = ``n_outputs``.

    E.g. [32, 16, 8], one feature: 1*32 + 1*8 + (32*16 + 16*8) = 680.
    """
    hidden = sum(a * b for a, b in zip(layers, layers[1:]))
    return n_features * layers[0] + n_outputs * layers[-1] + hidden


def mac_residual(width: int, n_blocks: int, block_depth: int,
                 expansion_factor: float, n_features: int,
                 n_outputs: int = MAC_OUTPUTS) -> int:
    """MAC of the residual network (Vzorec 1):
    V_in * S + S * V_out + B * ([(N_exp + N_shr) * E + N_mid * E^2] * S^2 + S_skip),
    with S = ``width``, B = ``n_blocks``, D = ``block_depth``,
    E = ``expansion_factor``, N_exp = N_shr = 1, N_mid = D - 2 and
    S_skip = S: the skip adds the block's input to its output element-wise,
    two vectors of width S, so S additions per block.

    E * S is the inner width H; residual_settings and the builder both
    require it to be a whole number, so per block this is exactly
    (N_exp + N_shr) * S * H + N_mid * H^2 + S_skip, the formula above.

    E.g. S=16, B=2, D=3, E=3, one feature (Obrazek 4): H = 48,
    1*16 + 16*1 + 2 * [(1 + 1)*16*48 + 1*48*48 + 16] = 7744.
    """
    n_mid = block_depth - N_EXP - N_SHR
    if n_mid < 0:
        raise ValueError(f"block_depth must be at least {N_EXP + N_SHR}")
    # As converter.to_tflite._hidden_units (not imported: it pulls in TensorFlow).
    if not float(expansion_factor * width).is_integer():
        raise ValueError("expansion_factor x width must be a whole number")
    hidden = int(expansion_factor * width)
    skip = width  # S_skip: one addition per element of the width-S stream
    block = (N_EXP + N_SHR) * width * hidden + n_mid * hidden * hidden + skip
    return n_features * width + width * n_outputs + n_blocks * block


def mac_text(topology: dict, n_features: int) -> str:
    """The MAC label's text, for Sequential MLP and Functional Residual."""
    if n_features is None:  # nothing staged yet
        return "MAC: -"
    architecture = topology["architecture"][topology["type"]]
    try:
        if architecture == "residual":
            mac = mac_residual(**residual_settings(topology["residual"]),
                               n_features=n_features)
        else:
            layers = parse_layers(topology["layers"][architecture])
            if not layers:
                return "MAC: -"
            mac = mac_mlp(layers, n_features)
    except ValueError:
        return "MAC: -"
    return f"MAC: {mac}"


def topology_params(topology: dict) -> dict:
    """The surrogate-network params this page contributes to a run.

    Residual passes S, B, D and h (width, n_blocks, block_depth,
    expansion_factor) as ARCHITECTURE_PARAMS.
    MLP passes its layer widths as LAYERS; an empty box leaves LAYERS out so
    the algorithm's default applies.
    """
    architecture = topology["architecture"][topology["type"]]
    params = {"ARCHITECTURE": architecture}
    if architecture == "residual":
        params["ARCHITECTURE_PARAMS"] = residual_settings(topology["residual"])
        return params
    layers = parse_layers(topology["layers"][architecture])
    if layers:
        params["LAYERS"] = layers
    return params


def build(key: str) -> None:
    entry = REGISTRY[key]
    topology = STATE["topology"]
    # Inputs of the network for the columns the CSV wizard staged.
    n_features = entry.n_inputs()
    # Chosen on the previous (Configure) page, so it can't change while here.
    quantization = STATE["params"].get("QUANTIZATION", "int8")

    def size_text() -> str:
        return size_estimate_text(topology, quantization, n_features)

    @ui.refreshable
    def type_params() -> None:
        # Each topology type gets its own parameter set: an architecture
        # (remembered per type) and that architecture's own settings.
        model_type = topology["type"]
        architecture = topology["architecture"][model_type]
        ui.select(
            ARCHITECTURES[model_type], label="Architecture",
            value=architecture, on_change=on_architecture_change,
        ).classes("w-80").props("outlined")
        if architecture == "residual":
            residual = topology["residual"]
            for key_, label, kind, step, lo, hi, description in RESIDUAL_FIELDS:
                with ui.row().classes("items-center gap-2"):
                    with ui.element("div").classes("relative"):
                        field = ui.number(
                            label=label, value=residual[key_], min=lo, max=hi,
                            step=step, format="%.0f" if kind == "int" else None,
                        ).classes("w-72 neu-number").props("outlined")
                        _add_spinner(field, step, lo, hi)
                    field.on_value_change(
                        lambda e, k=key_, t=kind: on_residual_change(k, _coerce(t, e.value))
                    )
                    with ui.icon("info").classes("neu-info"):
                        ui.tooltip(description).classes("max-w-xs")
        else:
            ui.input(
                placeholder="Write comma behind each number",
                value=topology["layers"][architecture],
            ).classes("w-80").props("outlined").on_value_change(
                lambda e: on_layers_change(architecture, e.value)
            )

    def update_estimates() -> None:
        size_label.set_text(size_text())
        mac_label.set_text(mac_text(topology, n_features))

    def on_layers_change(architecture: str, value: str) -> None:
        topology["layers"][architecture] = value
        update_estimates()

    def on_residual_change(field_key: str, value) -> None:
        topology["residual"][field_key] = value
        update_estimates()

    def on_type_change(e) -> None:
        topology["type"] = e.value
        type_params.refresh()
        update_estimates()

    def on_architecture_change(e) -> None:
        topology["architecture"][topology["type"]] = e.value
        type_params.refresh()
        update_estimates()

    def go_next() -> None:
        try:
            topology_params(topology)
        except ValueError:
            if topology["architecture"][topology["type"]] == "residual":
                message = RESIDUAL_INVALID_MESSAGE
            else:
                message = "Layers must be positive whole numbers separated by commas."
            ui.notify(message, type="warning")
            return
        ui.navigate.to(f"/save/{key}")

    with theme.panel(3):
        # Absolute against .neu-panel (position:relative); bottom-3 keeps it
        # inside the panel's 2.5rem bottom padding, clear of the buttons row.
        size_label = ui.label(size_text()).classes(
            "absolute bottom-3 right-5 font-bold"
        )
        # The MAC count mirrors it on the left.
        with ui.row().classes("absolute bottom-3 left-5 items-center gap-1"):
            mac_label = ui.label(mac_text(topology, n_features)).classes("font-bold")
            with ui.icon("info").classes("neu-info"):
                ui.tooltip(MAC_DISCLAIMER).classes("max-w-xs")
        ui.label(f"Topology for {entry.display_name}").classes("text-h5")

        ui.select(
            {"functional": "Functional", "sequential": "Sequential"},
            label="Model type", value=topology["type"], on_change=on_type_change,
        ).classes("w-80").props("outlined")

        type_params()

        with ui.row().classes("q-mt-md gap-2"):
            ui.button(
                "Back", on_click=lambda: ui.navigate.to(f"/configure/{key}"), color=None
            ).classes("neu-ghost")
            ui.button("Continue", on_click=go_next, color=None).classes("neu-primary")
