"""Test model page: scores an already trained .tflite like Edge Impulse does
after an upload, and measures with PR-AUC whether it detects readings
outside the user's normal ranges (core.model_test). Outside the wizard's
steps; works on any remembered or opened model (core.model_paths).
"""

import os

from nicegui import app, ui
from nicegui import run as nicegui_run

from app import theme
from core import model_paths, model_test

_LOW_RULE = (
    "The rule finds less than half of the anomalies: a reading inside your ranges "
    "already scores about as high as those outside. Usually the limits lie at or "
    "beyond the edge of the training data, where scores may stop rising - in "
    "models trained while scores were capped at 1, and with Isolation Forest, ECOD, "
    "COPOD and LODA. A "
    "rare value inside the range can cause it too: a side or a feature without limits "
    "counts as normal up to its rarest training value. Try narrower limits, both sides, "
    "and limits for every feature."
)


def _label(path: str) -> str:
    missing = "" if os.path.isfile(path) else "  (missing)"
    return f"{os.path.basename(path)} - {os.path.dirname(path)}{missing}"


def _warn(text: str, kind: str = "warn") -> None:
    icon = {"warn": "warning", "block": "block"}[kind]
    with ui.element("div").classes(f"neu-notice neu-notice-{kind}"):
        ui.icon(icon)
        ui.label(text)


def build(model_path: str = "") -> None:
    """``model_path`` is the model to open; the last used one if empty."""
    paths, list_warning = model_paths.load()
    state = {"path": model_path or (paths[0] if paths else None), "limits": {}, "reading": "",
             "features": None, "rule": None}

    def select(path: str) -> None:
        if not path or path == state["path"]:
            return
        warning = model_paths.remember(path)
        if warning:
            ui.notify(warning, type="warning", multi_line=True)
        state.update(path=path, limits={}, reading="", rule=None)
        model_row.refresh()
        body.refresh()

    async def choose_other() -> None:
        import webview  # only importable once nicegui[native] is installed

        result = await app.native.main_window.create_file_dialog(
            dialog_type=webview.FileDialog.OPEN,
            directory=model_paths.last_folder(),
            file_types=("TFLite model (*.tflite)",),
        )
        if result:
            chosen = result[0] if isinstance(result, (list, tuple)) else result
            if not chosen.lower().endswith(".tflite"):
                ui.notify("Only .tflite files are allowed.", type="warning")
                return
            select(model_paths.normalize(chosen))

    def remove() -> None:
        warning = model_paths.forget(state["path"])
        if warning:
            ui.notify(warning, type="warning", multi_line=True)
        # Nothing takes the removed model's place until the user picks it.
        state.update(path=None, limits={}, reading="", rule=None)
        model_row.refresh()
        body.refresh()

    @ui.refreshable
    def model_row() -> None:
        options = {p: _label(p) for p in model_paths.load()[0]}
        if state["path"] and state["path"] not in options:
            options[state["path"]] = _label(state["path"])  # list not writable
        with ui.row().classes("w-full items-center gap-3 no-wrap"):
            ui.select(
                options, label="Model", value=state["path"],
                on_change=lambda e: select(e.value),
            ).classes("grow min-w-0").props("outlined")
            ui.button("Choose other...", on_click=choose_other, color=None).classes("shrink-0")

    def load_again():
        """The model as it is on disk now; None (with a message) if it can't
        be tested, or if it no longer has the features shown.
        """
        try:
            model = model_test.load(state["path"])
        except model_test.ModelTestError as exc:
            ui.notify(str(exc), type="warning", multi_line=True)
            body.refresh()
            return None
        if model.features != state["features"]:
            ui.notify("The model was retrained with other features.", type="warning")
            state.update(limits={}, reading="", rule=None)
            body.refresh()
            return None
        return model

    @ui.refreshable
    def body() -> None:
        if not state["path"]:
            ui.label("Choose a trained .tflite model.").classes("neu-status")
            return
        try:
            model = model_test.load(state["path"])
        except model_test.ModelTestError as exc:
            _warn(str(exc), "block")
            ui.button("Remove from list", on_click=remove, color=None).classes("neu-ghost")
            return
        state["features"] = model.features

        ui.label(state["path"]).classes("neu-path neu-set")
        ui.label(
            f"{model.quantization}, {'normalized' if model.normalized else 'raw'} input. "
            f"Inputs: {','.join(model.features)}"
        ).classes("neu-status")

        ui.label("Normal range - anything outside is an anomaly").classes("neu-notices-title")
        for i, name in enumerate(model.features):
            limits = state["limits"].setdefault(name, [None, None])
            with ui.row().classes("w-full items-center gap-3"):
                with ui.column().classes("gap-0 w-44"):
                    ui.label(name).classes("neu-column-name")
                    ui.label(
                        f"trained on {model.minimum[i]:.6g} .. {model.maximum[i]:.6g}"
                    ).classes("neu-status")
                for side, label in enumerate(("from", "to")):
                    ui.number(
                        label=label, value=limits[side],
                        on_change=lambda e, pair=limits, s=side: pair.__setitem__(s, e.value),
                    ).classes("w-36 neu-number").props("outlined clearable")

        async def test() -> None:
            current = load_again()
            if current is None:
                return
            limits = {name: tuple(pair) for name, pair in state["limits"].items()
                      if any(v is not None for v in pair)}
            test_button.disable()
            try:
                result = await nicegui_run.io_bound(model_test.test_ranges, current, limits)
            except (model_test.ModelTestError, ValueError, RuntimeError) as exc:
                state["rule"] = None
                result_box.clear()
                with result_box:
                    _warn(str(exc), "block")
                return
            finally:
                test_button.enable()
            state["rule"] = result.rule
            result_box.clear()
            with result_box:
                _show_result(result)

        test_button = ui.button("Test ranges", on_click=test, color=None).classes("neu-primary")
        result_box = ui.column().classes("w-full gap-2")

        ui.label("Try one reading").classes("neu-notices-title")
        with ui.row().classes("w-full items-center gap-3 no-wrap"):
            reading = ui.input(
                label=",".join(model.features), value=state["reading"],
                on_change=lambda e: state.__setitem__("reading", e.value),
            ).classes("grow").props("outlined")
            score_button = ui.button("Score", color=None).classes("shrink-0")
        reading_box = ui.column().classes("w-full gap-1")

        async def score_one() -> None:
            current = load_again()
            if current is None:
                return
            reading_box.clear()
            try:
                values = model_test.parse_reading(state["reading"] or "", current)
                value = float((await nicegui_run.io_bound(model_test.score, current, values))[0])
            except (model_test.ModelTestError, ValueError, RuntimeError) as exc:
                with reading_box:
                    _warn(str(exc), "block")
                return
            verdicts = []
            if current.threshold is not None:
                verdicts.append(f"{'anomaly' if value > current.threshold else 'normal'} "
                                f"by {current.threshold:g}")
            if state["rule"] is not None:
                verdicts.append(f"{'anomaly' if value > state['rule'] else 'normal'} "
                                f"by the rule ({state['rule']:.4f})")
            with reading_box:
                ui.label(f"{value:.4f}").classes("neu-readout")
                if verdicts:
                    ui.label(", ".join(verdicts)).classes("neu-status")
                ui.label(
                    f"In Edge Impulse type: {model_test.edge_impulse_text(current, values)}"
                ).classes("neu-status")

        score_button.on_click(score_one)
        reading.on("keydown.enter", score_one)

    with theme.panel(None, wide=True):
        ui.label("Test model").classes("text-h5")
        if list_warning:
            _warn(list_warning)
        with ui.column().classes("w-full gap-4"):
            model_row()
            with ui.column().classes("w-full gap-3"):
                body()
        ui.button("Back to start", on_click=lambda: ui.navigate.to("/"), color=None).classes(
            "neu-ghost"
        )


def _show_result(result: model_test.RangeResult) -> None:
    ui.label(f"PR-AUC {result.pr_auc:.3f}").classes("neu-readout")
    ui.label(
        f"Random guessing gives {result.baseline:.2f}. Tested on {result.n_normal} readings "
        f"inside your ranges and {result.n_anomalies} outside."
    ).classes("neu-status")
    if result.threshold is not None:
        ui.label(
            f"Score > {result.threshold:g}: found {result.found_by_threshold} of "
            f"{result.n_anomalies} anomalies, flagged {result.flagged_by_threshold} of "
            f"{result.n_normal} normal"
        ).classes("neu-status")
    ui.label(
        f"Rule for this model - anomaly when score > {result.rule:.4f}: found "
        f"{result.found_by_rule} of {result.n_anomalies} anomalies, flagged 0 of "
        f"{result.n_normal} normal"
    ).classes("neu-status" if result.low_rule else "neu-status neu-ok")
    if result.low_rule:
        _warn(_LOW_RULE)
    for warning in result.warnings:
        _warn(warning)
