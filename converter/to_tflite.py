"""Train a small Keras network to reproduce an algorithm's calibrated anomaly
scores and compress it to a .tflite model for microcontroller / Edge Impulse
deployment.

Generic and reusable: core.common passes, per presence pattern, the raw
feature rows and their calibrated detector scores (0.5 = anomaly threshold).
The network is trained on features normalized to [0, 1] (well conditioned).
With INPUT_MODE "raw" (the default) that normalization is then folded into
the first Dense layer's weights, so the exported model takes raw sensor
values - what Edge Impulse and the device actually have - with the exact
same outputs and no extra layers, parameters or MAC operations. With
"normalized" the device has to apply (x - input_min) / (input_max -
input_min) from <name>_params.json itself.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / ".tflite"

# Network output above which a sample counts as anomalous, for every
# algorithm and presence pattern (see core.common.ScoreCalibrator).
THRESHOLD = 0.5

INPUT_MODES = ("raw", "normalized")


@dataclass
class Pattern:
    """Surrogate training data of one presence pattern (see
    core.common._patterns). The ``x_*`` arrays are full-width raw feature
    rows (NaN where missing, and outside ``present``); the ``y_*`` arrays
    are the calibrated detector scores.
    """

    present: tuple
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    calibration: dict = field(default_factory=dict)


OPTIMIZERS = {
    "adam": tf.keras.optimizers.Adam,
    "sgd": tf.keras.optimizers.SGD,
    "rmsprop": tf.keras.optimizers.RMSprop,
}


def n_model_inputs(n_features):
    """Network inputs for ``n_features`` feature columns: just the value
    with one column, value/flag pairs with several (see ``presence_inputs``).
    """
    return 2 * n_features if n_features > 1 else n_features


def input_layout(feature_names):
    """Names of the network inputs, in order, for several feature columns:
    ``[f1, f1_present, f2, f2_present, ...]``.
    """
    return [name for feature in feature_names for name in (feature, f"{feature}_present")]


def presence_inputs(x, present):
    """Network input rows for several feature columns: interleaved
    value/flag pairs ``[v1, f1, v2, f2, ...]`` (see ``input_layout``).

    ``x`` holds the features as the network takes them (normalized during
    training; raw or normalized on export, see INPUT_MODES), NaN where
    missing. ``present`` are the column indices the device sends. A
    feature's flag is 1 when it is in ``present`` and not NaN; otherwise its
    value and its flag are both 0, so the flag is what tells "missing" from
    "value 0". This is the layout the microcontroller has to reproduce.
    """
    n_rows, n_features = x.shape
    flags = np.zeros((n_rows, n_features), dtype="float32")
    flags[:, list(present)] = 1.0
    flags[np.isnan(x)] = 0.0
    values = np.where(flags == 1.0, np.nan_to_num(x), 0.0)
    rows = np.empty((n_rows, 2 * n_features), dtype="float32")
    rows[:, 0::2] = values
    rows[:, 1::2] = flags
    return rows


def network_inputs(rows, present, scaler, presence, space):
    """Network input rows for full-width raw feature ``rows`` (NaN where
    missing) in ``space``: "normalized" (min-max to [0, 1], what the network
    is trained on) or "raw" (what a raw-input export takes). With
    ``presence`` (several feature columns) they are value/flag pairs.
    """
    x = scaler.transform(rows) if space == "normalized" else np.asarray(rows, dtype="float64")
    if presence:
        return presence_inputs(x, present)
    return x.astype("float32")


def _stack_patterns(patterns, scaler, presence, space):
    """Stack the inputs and targets of every presence pattern into one
    training and one test set, in input ``space`` (see ``network_inputs``).

    Returns the stacked ``x_train, y_train, x_test, y_test`` plus each
    row's pattern number (its position in ``patterns``) for both sets.
    """
    parts = {key: [] for key in ("x_train", "y_train", "id_train", "x_test", "y_test", "id_test")}
    for number, pattern in enumerate(patterns):
        parts["x_train"].append(
            network_inputs(pattern.x_train, pattern.present, scaler, presence, space))
        parts["y_train"].append(pattern.y_train)
        parts["id_train"].append(np.full(len(pattern.x_train), number))
        parts["x_test"].append(
            network_inputs(pattern.x_test, pattern.present, scaler, presence, space))
        parts["y_test"].append(pattern.y_test)
        parts["id_test"].append(np.full(len(pattern.x_test), number))
    return (
        np.vstack(parts["x_train"]), np.concatenate(parts["y_train"]).astype("float32"),
        np.vstack(parts["x_test"]), np.concatenate(parts["y_test"]).astype("float32"),
        np.concatenate(parts["id_train"]), np.concatenate(parts["id_test"]),
    )


def _build_mlp(n_features, layers):
    """Plain stack of Dense-ReLU layers, one per entry in ``layers``."""
    return tf.keras.Sequential(
        [tf.keras.Input(shape=(n_features,))]
        + [tf.keras.layers.Dense(units, activation="relu") for units in layers]
        + [tf.keras.layers.Dense(1)]
    )


def _hidden_units(width, expansion_factor):
    """Units of a residual block's inner Dense layers: E * S. Raises
    ValueError unless E * S is a whole number of at least 1, so the network
    always has exactly the inner width the thesis MAC formula assumes.
    """
    hidden = expansion_factor * width
    if not float(hidden).is_integer() or hidden < 1:
        raise ValueError(
            f"expansion_factor * width must be a whole number >= 1, "
            f"got {expansion_factor} * {width} = {hidden}"
        )
    return int(hidden)


def _build_residual_mlp(n_features, width, n_blocks, block_depth=2, expansion_factor=1.0):
    """Residual MLP after Gorishniy et al. (2021), Eq. 2, generalized to
    ``block_depth`` Dense layers per block.

    Hyperparameters (thesis notation):
      S = ``width``        width of the residual stream
      B = ``n_blocks``     number of residual blocks
      D = ``block_depth``  Dense layers inside each block
      h = ``expansion_factor`` expansion factor of the inner layers (Sandler et al. 2018)

    Structure (the layers of Vzorec 1, N_exp = N_shr = 1, N_mid = D - 2):
      stem   Dense(S), linear: lifts the input to the stream width
      block  Dense(h * S, ReLU)            N_exp: expands S -> h*S
             (D - 2) x Dense(h * S, ReLU)  N_mid: keep h*S
             Dense(S), linear              N_shr: shrinks h*S -> S
             added to the block's input (S_skip = S additions), then ReLU
      head   Dense(1)

    h * S must be a whole number (see ``_hidden_units``).

    D = 2 is the classic Dense-ReLU-Dense block. D must be at least 2: with
    one layer the block has no inner layer to expand, so h would have no
    effect and the thesis MAC formula (N_mid = D - 2) would not apply. Every
    block starts and ends at width S, so all skips are identity mappings (no
    projections).
    """
    if block_depth < 2:
        raise ValueError(f"block_depth must be at least 2, got {block_depth}")
    hidden = _hidden_units(width, expansion_factor)
    inputs = tf.keras.Input(shape=(n_features,))
    x = tf.keras.layers.Dense(width)(inputs)
    for _ in range(n_blocks):
        h = x
        for _ in range(block_depth - 1):
            h = tf.keras.layers.Dense(hidden, activation="relu")(h)
        h = tf.keras.layers.Dense(width)(h)
        x = tf.keras.layers.ReLU()(tf.keras.layers.Add()([h, x]))
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs, outputs)


ARCHITECTURES = {
    "mlp": _build_mlp,
    "residual": _build_residual_mlp,
}


class _ProgressCallback(tf.keras.callbacks.Callback):
    """Reports 1-indexed epoch progress via a plain callable, and stops
    training early (at the next epoch boundary) if cancel_event is set.
    """

    def __init__(self, progress_callback, total_epochs, cancel_event=None):
        super().__init__()
        self._progress_callback = progress_callback
        self._total_epochs = total_epochs
        self._cancel_event = cancel_event

    def on_epoch_end(self, epoch, logs=None):
        if self._progress_callback is not None:
            self._progress_callback(epoch + 1, self._total_epochs)
        if self._cancel_event is not None and self._cancel_event.is_set():
            self.model.stop_training = True


# No float16: its weights are expanded by a float16 DEQUANTIZE op, which
# TensorFlow Lite Micro does not support, so the model fails to allocate on
# the microcontroller (and Edge Impulse cannot profile it).
QUANTIZATIONS = ("none", "int8")

# Rows of the training inputs the int8 converter is calibrated on, on top
# of the extreme rows (see _extreme_rows).
CALIBRATION_ROWS = 1000

# Share of each feature's training range the calibrated input range extends
# beyond its min and max. core.common uses the same margin for its synthetic
# coverage rows, so the network is trained over the whole calibrated range:
# the "valid_input_range" in <name>_params.json. Wide enough that the
# detectors' scores have reached "clearly anomalous" at its edges, so an
# int8 input saturating there still reads as an anomaly.
CALIBRATION_MARGIN = 0.5


def valid_input_range(scaler):
    """Per-feature (low, high) raw values the exported model is trained and
    calibrated on: the training min/max widened by CALIBRATION_MARGIN.
    """
    margin = CALIBRATION_MARGIN * scaler.data_range_
    return scaler.data_min_ - margin, scaler.data_max_ + margin


def _extreme_rows(scaler, n_features, presence, space):
    """Network inputs at each feature's training min and max widened by
    CALIBRATION_MARGIN (all features present), plus, with presence flags,
    the all-missing row - so the int8 input range covers everything the
    device legitimately sends.
    """
    low, high = valid_input_range(scaler)
    rows = [low, high]
    if presence:
        rows.append(np.full(n_features, np.nan))
    return network_inputs(np.array(rows), range(n_features), scaler, presence, space)


def _calibration_rows(x_train, extremes):
    """Rows the int8 converter measures activation ranges on: ``extremes``
    followed by up to ``CALIBRATION_ROWS`` training input rows picked at
    random (fixed seed). Also saved as <name>_representative.npy for Edge
    Impulse.
    """
    rng = np.random.default_rng(0)
    picked = rng.choice(len(x_train), min(CALIBRATION_ROWS, len(x_train)), replace=False)
    return np.vstack([extremes, x_train[picked]]).astype("float32")


def convert_model(model, calibration, quantization="int8"):
    """Convert a trained Keras model to .tflite bytes with post-training
    quantization (PTQ).

    ``quantization`` is one of ``QUANTIZATIONS``:

    - "none": plain float32.
    - "int8": full-integer int8 (weights, activations, input and output),
      small and fast on a microcontroller. Weight ranges are read from the
      weights themselves; activation ranges are calibrated on the
      ``calibration`` rows (see ``_calibration_rows``). Values outside the
      calibrated range saturate rather than overflow.
    """
    if quantization not in QUANTIZATIONS:
        raise ValueError(
            f"Unknown quantization {quantization!r}, expected one of {QUANTIZATIONS}"
        )
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantization == "int8":
        def _representative_dataset():
            for row in calibration:
                yield [row.reshape(1, -1)]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    return converter.convert()


def _first_dense(model):
    """The Dense layer the inputs feed: the MLP's first hidden layer or the
    residual network's stem (both are applied directly to the Input).
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            return layer
    raise ValueError("The model has no Dense layer to fold the input scaling into")


def fold_input_scaling(model, scaler, presence):
    """Rewrite the first Dense layer's weights in place so the model takes
    raw feature values instead of min-max normalized ones, with identical
    outputs. No layer, parameter or MAC is added.

    MinMaxScaler normalizes as ``x_norm = x * scale + offset``. For plain
    inputs ``W x_norm + b = (W * scale) x + (b + offset W)``. For value/flag
    pairs the offset goes onto the flag's weight instead of the bias:
    ``W_v (x * scale + offset) + W_f = (W_v * scale) x + (W_f + W_v offset)``
    when present (flag 1), and both terms stay 0 when missing (value 0,
    flag 0) - the missing-feature convention is unchanged.
    """
    dense = _first_dense(model)
    kernel, bias = dense.get_weights()
    scale = scaler.scale_.astype("float64")[:, None]
    offset = scaler.min_.astype("float64")[:, None]
    kernel = kernel.astype("float64")
    if presence:
        values, flags = kernel[0::2], kernel[1::2]
        folded = kernel.copy()
        folded[0::2] = values * scale
        folded[1::2] = flags + values * offset
        folded_bias = bias
    else:
        folded = kernel * scale
        folded_bias = bias + (offset * kernel).sum(axis=0)
    dense.set_weights([folded.astype("float32"), np.asarray(folded_bias, dtype="float32")])


def _interpreter(tflite_model):
    """A TFLite interpreter on the built-in reference kernels only.

    The default interpreter also applies the XNNPACK delegate, which fails
    to prepare some small int8 models ("failed to create XNNPACK runtime")
    that are perfectly valid. The microcontroller never uses XNNPACK, so the
    built-in kernels are also the closer match to the device.
    """
    return tf.lite.Interpreter(
        model_content=tflite_model,
        experimental_op_resolver_type=(
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )


def _run_tflite(tflite_model, x):
    """Run ``tflite_model`` on every row of ``x`` (float32) and return its
    float outputs, quantizing the input and dequantizing the output the way
    the device has to when the model has int8 I/O.
    """
    interpreter = _interpreter(tflite_model)
    input_details = interpreter.get_input_details()[0]
    interpreter.resize_tensor_input(input_details["index"], x.shape)
    interpreter.allocate_tensors()
    output_details = interpreter.get_output_details()[0]

    if input_details["dtype"] == np.int8:
        scale, zero_point = input_details["quantization"]
        x = np.clip(np.round(x / scale + zero_point), -128, 127).astype(np.int8)
    interpreter.set_tensor(input_details["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(output_details["index"]).astype("float32")
    if output_details["dtype"] == np.int8:
        scale, zero_point = output_details["quantization"]
        y = (y - zero_point) * scale
    return y.reshape(len(x), -1)


def _flag_agreement(a, b):
    return float(np.mean((a > THRESHOLD) == (b > THRESHOLD)))


def _drift(reference, exported, targets):
    """How far the exported model's outputs are from the Keras model's
    (``test_*``) and from the detector's calibrated scores (``detector_*``),
    as mean/max absolute differences and how often both sides agree on
    flagging a sample (output > THRESHOLD).
    """
    error = np.abs(exported - reference)
    return {
        "test_mean_abs_error": float(error.mean()),
        "test_max_abs_error": float(error.max()),
        "test_flag_agreement": _flag_agreement(exported, reference),
        "detector_mean_abs_error": float(np.abs(exported - targets).mean()),
        "detector_flag_agreement": _flag_agreement(exported, targets),
    }


# Largest int8 input step, as a share of a feature's training range, before
# the report warns (fewer than ~100 levels across the feature).
MAX_INPUT_RESOLUTION = 0.01
# Largest error of the presence flag value 1 after int8 input quantization.
MAX_FLAG_ERROR = 0.05


def _int8_input_checks(input_scale, scaler, feature_names, presence, input_mode):
    """Per-feature int8 input resolution (one input step as a share of the
    feature's training range) and warnings when it is too coarse. The input
    tensor has a single scale, so in raw mode features with very different
    ranges (or a 0/1 flag next to large raw values) share it badly.
    """
    if input_mode == "raw":
        ranges = np.where(scaler.data_range_ == 0, 1.0, scaler.data_range_)
    else:
        ranges = np.ones(len(feature_names))
    resolution = {name: float(input_scale / r) for name, r in zip(feature_names, ranges)}
    warnings = []
    coarse = [name for name, value in resolution.items() if value > MAX_INPUT_RESOLUTION]
    if coarse:
        warnings.append(
            f"int8 input resolution is coarse for {', '.join(coarse)} (one input "
            f"step is more than {MAX_INPUT_RESOLUTION:.0%} of the feature's range): "
            f"the single int8 input scale has to cover 0 and every feature's whole "
            f"raw range, which is too wide for this feature's spread. Use input "
            f"mode 'normalized' or quantization 'none' (float32)."
        )
    if presence:
        flag_error = abs(np.round(1.0 / input_scale) * input_scale - 1.0)
        if flag_error > MAX_FLAG_ERROR:
            warnings.append(
                f"The presence flag value 1 is quantized with a {flag_error:.0%} "
                f"error in the int8 input. Use input mode 'normalized' or "
                f"quantization 'none' (float32)."
            )
    return resolution, warnings


def quantization_report(model, tflite_model, quantization, x_test, y_test,
                        pattern_ids, pattern_names, scaler, feature_names,
                        presence, input_mode):
    """Describe the exported model for the device and measure, on ``x_test``
    (in the exported model's own input format), how far its outputs drift
    from the Keras model and from the detector's calibrated scores
    ``y_test``; see ``_drift``. With several patterns the drift is also given
    per pattern under "patterns". For int8 it also holds the input/output
    scale and zero point the device needs to quantize features and
    dequantize the score, the per-feature input resolution, and warnings.
    """
    reference = model.predict(x_test, verbose=0).reshape(len(x_test), -1)
    exported = _run_tflite(tflite_model, x_test)
    targets = y_test.reshape(-1, 1)

    report = {
        "type": quantization,
        "size_bytes": len(tflite_model),
        **_drift(reference, exported, targets),
        "warnings": [],
    }
    if len(pattern_names) > 1:
        report["patterns"] = {
            name: _drift(reference[pattern_ids == number], exported[pattern_ids == number],
                         targets[pattern_ids == number])
            for number, name in enumerate(pattern_names)
        }
    if quantization == "int8":
        interpreter = _interpreter(tflite_model)
        (input_scale, input_zero_point) = interpreter.get_input_details()[0]["quantization"]
        (output_scale, output_zero_point) = interpreter.get_output_details()[0]["quantization"]
        resolution, warnings = _int8_input_checks(
            input_scale, scaler, feature_names, presence, input_mode)
        report.update(
            input_scale=float(input_scale),
            input_zero_point=int(input_zero_point),
            output_scale=float(output_scale),
            output_zero_point=int(output_zero_point),
            input_resolution=resolution,
            warnings=warnings,
        )
    return report


# Test-score quantiles the example rows in <name>_params.json are taken at.
_EXAMPLE_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def _examples(tflite_model, x_test, y_test):
    """A few test rows in the exported model's input format, spread over the
    detector's scores, with the score the .tflite gives each: what to type
    into Edge Impulse (or run on the device) to check the deployment.
    """
    order = np.argsort(y_test, kind="stable")
    picks = list(dict.fromkeys(int(order[round(q * (len(order) - 1))])
                               for q in _EXAMPLE_QUANTILES))  # by score, no repeats
    rows = np.round(x_test[picks].astype("float64"), 6).astype("float32")
    scores = _run_tflite(tflite_model, rows).ravel()
    return [
        {
            "input": [float(v) for v in row],
            "expected_score": round(float(score), 4),
            "anomaly": bool(score > THRESHOLD),
            "detector_score": round(float(y_test[i]), 4),
        }
        for row, score, i in zip(rows, scores, picks)
    ]


def _edge_impulse_text(name, params):
    """Instructions for uploading <name>.tflite to Edge Impulse as a
    "Bring your own model" and checking it.
    """
    q = params["quantization"]
    layout = params["input_layout"]
    raw = params["input_mode"] == "raw"
    presence = "missing_value" in params
    lines = [
        f"Edge Impulse - Bring your own model: {name}.tflite",
        "=" * 60,
        "",
        f"Model: {'int8 (full integer)' if q['type'] == 'int8' else 'float32'}, "
        f"{q['size_bytes']} bytes, input shape (1, {len(layout)}), output shape (1, 1).",
        "",
        "Upload (Edge Impulse Studio, new project -> upload your model /",
        "\"Bring your own model\"; menu names can differ between Studio versions):",
        f"  - Model file: {name}.tflite",
        "  - Model input: other (raw features, not an image or audio)",
        "  - Model output: regression (one value: the anomaly score)",
        f"  - Representative features, if asked: {name}_representative.npy",
    ]
    if q["type"] == "int8":
        lines += [
            "  - The model is already int8-quantized. Send float features; Edge",
            "    Impulse quantizes them with the model's own input parameters",
            f"    (scale {q['input_scale']:.6g}, zero point {q['input_zero_point']}) and",
            f"    dequantizes the score (scale {q['output_scale']:.6g}, zero point "
            f"{q['output_zero_point']}).",
        ]
    lines += ["", f"Input - {len(layout)} values, in this order:"]
    lines += [f"  {i + 1:>2}. {entry}" for i, entry in enumerate(layout)]
    lines.append("")
    if presence:
        lines += [
            "Each feature is a value/flag pair: flag 1 = the value was measured,",
            "and a missing feature is sent as value 0 with flag 0.",
        ]
    if raw:
        lines.append("Values are raw, as measured - the normalization is built into the model.")
    else:
        lines.append("Values must be normalized first: x' = (x - min) / (max - min), with")
        lines += [
            f"  {feature}: min {lo:.6g}, max {hi:.6g}"
            for feature, lo, hi in zip(params["feature_names"], params["input_min"],
                                       params["input_max"])
        ]
    if params.get("min_features_sent", 1) > 1:
        lines.append(
            f"Send at least {params['min_features_sent']} features (flag 1); "
            f"with fewer the score is not trained and not meaningful."
        )
    lines.append(
        "Trained for measured values within (outside, the score is not reliable)"
        + (":" if raw else " - before normalization:")
    )
    lines += [
        f"  {feature}: {low:.6g} .. {high:.6g}"
        for feature, (low, high) in zip(params["feature_names"], params["valid_input_range"])
    ]
    lines += [
        "",
        f"Output: anomaly score (0 = most normal, can exceed 1); anomalous when score > "
        f"{params['threshold']}.",
        "",
        "Check - these inputs should give these scores (+- a few hundredths):",
    ]
    lines += [
        f"  [{', '.join(f'{v:.6g}' for v in example['input'])}] -> "
        f"{example['expected_score']:.3f} ({'anomaly' if example['anomaly'] else 'normal'})"
        for example in params["examples"]
    ]
    for warning in q.get("warnings", []):
        lines += ["", f"WARNING: {warning}"]
    return "\n".join(lines) + "\n"


def export(model, name, out_dir, calibration, x_test, y_test, test_ids, patterns,
           scaler, feature_names, presence, quantization, input_mode, fold_error=None):
    """Write ``<name>.keras``, ``<name>.tflite``, ``<name>_params.json``,
    ``<name>_representative.npy`` and ``<name>_edge_impulse.txt`` to
    ``out_dir``. ``calibration`` and ``x_test`` are in the exported model's
    input format (raw values when ``input_mode`` is "raw").

    The params file holds the input mode and layout, the feature min/max
    (normalized mode needs them on the device), the threshold, each
    pattern's score calibration, the ``quantization_report`` under
    "quantization" and example rows with their expected scores.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / f"{name}.keras")

    tflite_model = convert_model(model, calibration, quantization)
    (out_dir / f"{name}.tflite").write_bytes(tflite_model)
    np.save(out_dir / f"{name}_representative.npy", calibration)

    pattern_names = [pattern.name for pattern in patterns]
    params = {
        "input_mode": input_mode,
        "feature_names": list(feature_names),
        "input_layout": input_layout(feature_names) if presence else list(feature_names),
        "input_min": scaler.data_min_.tolist(),
        "input_max": scaler.data_max_.tolist(),
        "valid_input_range": [list(pair) for pair in zip(*(r.tolist() for r in valid_input_range(scaler)))],
        "threshold": THRESHOLD,
        "score_calibration": {pattern.name: pattern.calibration for pattern in patterns},
    }
    if presence:
        params["missing_value"] = 0.0
        # Fewest features a trained pattern has (see core.common.TrainingSpec).
        params["min_features_sent"] = min(len(pattern.present) for pattern in patterns)
    report = quantization_report(
        model, tflite_model, quantization, x_test, y_test, test_ids, pattern_names,
        scaler, feature_names, presence, input_mode,
    )
    if fold_error is not None:
        report["fold_max_abs_error"] = fold_error
    params["quantization"] = report
    params["examples"] = _examples(tflite_model, x_test, y_test)
    (out_dir / f"{name}_params.json").write_text(json.dumps(params, indent=2))
    (out_dir / f"{name}_edge_impulse.txt").write_text(_edge_impulse_text(name, params))


def to_tflite(name, patterns, x_reference, feature_names,
              layers, learning_rate, epochs, batch_size, verbose,
              output_dir=None, progress_callback=None, cancel_event=None,
              loss="mse", optimizer="adam", architecture="mlp",
              quantization="int8", architecture_params=None, input_mode="raw",
              random_state=None):
    """Fit a regressor on the calibrated anomaly scores and export it; see
    ``export`` for the files written.

    ``patterns`` are the presence patterns' data (see ``Pattern``; one with
    a single feature column). ``x_reference`` are the real training rows
    (NaN where missing); their per-column min/max define the normalization.
    With one feature column the network takes the value; with several,
    value/flag pairs (``presence_inputs``).

    The network is always trained on normalized inputs. ``input_mode`` (see
    ``INPUT_MODES``) picks what the exported model takes: "raw" folds the
    normalization into the first Dense layer (``fold_input_scaling``),
    "normalized" leaves it to the device.

    ``loss`` is "mse": the network regresses the calibrated scores. ``optimizer`` is a
    key of ``OPTIMIZERS`` ("adam", "sgd", "rmsprop"). ``architecture`` is a
    key of ``ARCHITECTURES`` ("mlp", "residual"). The builder is called with
    ``architecture_params`` as keyword arguments when given (residual needs
    ``width``, ``n_blocks`` and optionally ``block_depth`` and
    ``expansion_factor``; see ``_build_residual_mlp``), otherwise with
    ``layers`` as the hidden widths (mlp). ``quantization`` is "none" or
    "int8" (default); see ``convert_model``.

    ``random_state``, if given, seeds the network's initialization and
    shuffling, so a run is reproducible.

    ``output_dir`` defaults to ``<repo root>/.tflite`` when not given.
    ``progress_callback``, if given, is called as
    ``progress_callback(epoch, total_epochs)`` after each training epoch.
    ``cancel_event``, if given and set, stops training at the next epoch
    boundary and skips writing any output files, returning None instead of
    the model.
    """
    if input_mode not in INPUT_MODES:
        raise ValueError(f"Unknown input mode {input_mode!r}, expected one of {INPUT_MODES}")
    n_features = x_reference.shape[1]
    presence = n_features > 1

    # MinMaxScaler ignores NaN when fitting, so each column's range comes
    # from its observed values only.
    scaler = MinMaxScaler().fit(x_reference)
    x_train, y_train, x_test, y_test, _, test_ids = _stack_patterns(
        patterns, scaler, presence, "normalized")

    if random_state is not None:
        tf.keras.utils.set_random_seed(int(random_state))
    builder_settings = (
        architecture_params if architecture_params is not None else {"layers": layers}
    )
    model = ARCHITECTURES[architecture](x_train.shape[1], **builder_settings)
    model.compile(
        optimizer=OPTIMIZERS[optimizer](learning_rate=learning_rate),
        loss=loss,
    )

    callbacks = []
    if progress_callback is not None or cancel_event is not None:
        callbacks.append(_ProgressCallback(progress_callback, epochs, cancel_event))

    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks,
    )

    if cancel_event is not None and cancel_event.is_set():
        return None

    fold_error = None
    if input_mode == "raw":
        x_train_raw, _, x_test_raw, _, _, _ = _stack_patterns(patterns, scaler, presence, "raw")
        probe = slice(0, min(len(x_test), 512))
        before = model.predict(x_test[probe], verbose=0)
        fold_input_scaling(model, scaler, presence)
        after = model.predict(x_test_raw[probe], verbose=0)
        fold_error = float(np.abs(after - before).max())
        if fold_error > 1e-3 * max(1.0, float(np.abs(before).max())):
            raise RuntimeError(
                f"Folding the input normalization changed the outputs by up to "
                f"{fold_error:.3g}; export with input mode 'normalized' instead"
            )
        x_train, x_test = x_train_raw, x_test_raw

    calibration = _calibration_rows(
        x_train, _extreme_rows(scaler, n_features, presence, input_mode))
    out_dir = Path(output_dir) if output_dir is not None else _DEFAULT_OUTPUT_DIR
    export(model, name, out_dir, calibration, x_test, y_test, test_ids, patterns,
           scaler, feature_names, presence, quantization, input_mode, fold_error)

    return model
