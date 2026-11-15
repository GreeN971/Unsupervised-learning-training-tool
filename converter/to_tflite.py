"""Train a small Keras network to reproduce an algorithm's anomaly scores and
compress it to a quantized .tflite model for microcontroller deployment.

Generic and reusable: any algorithm passes its raw features (x) and anomaly
scores (y) plus its config-driven hyperparameters. The features are normalized
to [0, 1] before training, and the normalization parameters are saved so the
microcontroller can feed the network identical data.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / ".tflite"

OPTIMIZERS = {
    "adam": tf.keras.optimizers.Adam,
    "sgd": tf.keras.optimizers.SGD,
    "rmsprop": tf.keras.optimizers.RMSprop,
}


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

# Rows of the (normalized) training set the int8 converter is calibrated on.
CALIBRATION_ROWS = 1000


def _calibration_rows(x_train):
    """Rows the int8 converter measures activation ranges on: an all-zeros
    and an all-ones row, so the input range covers the full [0, 1] that
    MinMaxScaler maps the training data to, followed by up to
    ``CALIBRATION_ROWS`` training rows picked at random (fixed seed).
    """
    n_features = x_train.shape[1]
    rng = np.random.default_rng(0)
    picked = rng.choice(len(x_train), min(CALIBRATION_ROWS, len(x_train)), replace=False)
    return np.vstack([
        np.zeros((1, n_features), dtype="float32"),
        np.ones((1, n_features), dtype="float32"),
        x_train[picked],
    ])


def convert_model(model, x_train, quantization="int8"):
    """Convert a trained Keras model to .tflite bytes with post-training
    quantization (PTQ).

    ``quantization`` is one of ``QUANTIZATIONS``:

    - "none": plain float32.
    - "int8": full-integer int8 (weights, activations, input and output),
      small and fast on a microcontroller. Weight ranges are read from the
      weights themselves; activation ranges are calibrated on
      ``_calibration_rows(x_train)``. Values outside the calibrated range
      saturate rather than overflow.
    """
    if quantization not in QUANTIZATIONS:
        raise ValueError(
            f"Unknown quantization {quantization!r}, expected one of {QUANTIZATIONS}"
        )
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantization == "int8":
        calibration = _calibration_rows(x_train)

        def _representative_dataset():
            for row in calibration:
                yield [row.reshape(1, -1)]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    return converter.convert()


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


def quantization_report(model, tflite_model, quantization, x_test, flag_fraction=0.1):
    """Describe the exported model for the device and measure how far its
    outputs drift from the Keras model on ``x_test``.

    The drift is the mean/max absolute output difference, and how often
    both models agree on flagging a sample as anomalous, using the Keras
    model's top ``flag_fraction`` of scores as the threshold. For int8 it
    also holds the input/output scale and zero point the device needs to
    quantize features and dequantize the score.
    """
    reference = model.predict(x_test, verbose=0).reshape(len(x_test), -1)
    exported = _run_tflite(tflite_model, x_test)
    error = np.abs(exported - reference)
    threshold = np.quantile(reference, 1 - flag_fraction)

    report = {
        "type": quantization,
        "size_bytes": len(tflite_model),
        "test_mean_abs_error": float(error.mean()),
        "test_max_abs_error": float(error.max()),
        "test_flag_agreement": float(np.mean((exported > threshold) == (reference > threshold))),
        "flag_fraction": flag_fraction,
    }
    if quantization == "int8":
        interpreter = _interpreter(tflite_model)
        (input_scale, input_zero_point) = interpreter.get_input_details()[0]["quantization"]
        (output_scale, output_zero_point) = interpreter.get_output_details()[0]["quantization"]
        report.update(
            input_scale=float(input_scale),
            input_zero_point=int(input_zero_point),
            output_scale=float(output_scale),
            output_zero_point=int(output_zero_point),
        )
    return report


def export(model, name, out_dir, x_train, x_test, input_scaler,
           quantization="int8", feature_names=None, flag_fraction=0.1):
    """Write ``<name>.keras``, ``<name>.tflite`` and ``<name>_params.json``
    to ``out_dir``. ``x_train``/``x_test`` are already normalized by
    ``input_scaler``.

    The params file holds the input feature names, the min/max used to
    normalize them (so the device can reproduce the scaling), and the
    ``quantization_report`` under "quantization".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / f"{name}.keras")

    tflite_model = convert_model(model, x_train, quantization)
    (out_dir / f"{name}.tflite").write_bytes(tflite_model)

    if feature_names is None:
        feature_names = [f"feature_{i + 1}" for i in range(x_train.shape[1])]
    params = {
        "feature_names": list(feature_names),
        "input_min": input_scaler.data_min_.tolist(),
        "input_max": input_scaler.data_max_.tolist(),
        "quantization": quantization_report(
            model, tflite_model, quantization, x_test, flag_fraction
        ),
    }
    (out_dir / f"{name}_params.json").write_text(json.dumps(params, indent=2))


def to_tflite(name, x_train, y_train, x_test, y_test,
              layers, learning_rate, epochs, batch_size, verbose,
              feature_names=None, output_dir=None, progress_callback=None,
              cancel_event=None, loss="mse", optimizer="adam",
              architecture="mlp", quantization="int8", architecture_params=None,
              flag_fraction=0.1):
    """Fit a regressor on the anomaly scores and export .keras + .tflite.

    Also writes ``<name>_params.json`` with the input feature names, the
    min/max used to normalize them (so the same scaling can be reproduced on
    the device) and the quantization report; see ``export``.

    ``loss`` is any Keras loss identifier (e.g. "mse",
    "binary_crossentropy", "categorical_crossentropy"). ``optimizer`` is a
    key of ``OPTIMIZERS`` ("adam", "sgd", "rmsprop"). ``architecture`` is a
    key of ``ARCHITECTURES`` ("mlp", "residual"). The builder is called with
    ``architecture_params`` as keyword arguments when given (residual needs
    ``width``, ``n_blocks`` and optionally ``block_depth`` and
    ``expansion_factor``; see ``_build_residual_mlp``), otherwise with
    ``layers`` as the hidden widths (mlp). ``quantization`` is "none" or
    "int8" (default); see ``convert_model``. ``flag_fraction``
    is the share of test samples counted as anomalous when comparing the
    .tflite to the Keras model; see ``quantization_report``.

    ``output_dir`` defaults to ``<repo root>/.tflite`` (the original notebook
    behavior) when not given. ``progress_callback``, if given, is called as
    ``progress_callback(epoch, total_epochs)`` after each training epoch.
    ``cancel_event``, if given and set, stops training at the next epoch
    boundary and skips writing any output files, returning None instead of
    the model.
    """
    # Normalize the input features so the MCU feeds the network identical data.
    input_scaler = MinMaxScaler().fit(x_train)
    x_train = input_scaler.transform(x_train).astype("float32")
    x_test = input_scaler.transform(x_test).astype("float32")

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

    out_dir = Path(output_dir) if output_dir is not None else _DEFAULT_OUTPUT_DIR
    export(model, name, out_dir, x_train, x_test, input_scaler,
           quantization, feature_names, flag_fraction)

    return model
