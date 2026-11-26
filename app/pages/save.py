import os

from nicegui import app, ui

from app import theme
from app.state import STATE
from core import model_paths
from core.registry import REGISTRY


def build(key: str) -> None:
    entry = REGISTRY[key]

    async def confirm_overwrite(path: str) -> bool:
        with ui.dialog() as dialog, ui.card():
            ui.label(f"{path} already exists. Replace it?")
            with ui.row().classes("gap-2"):
                ui.button("Cancel", on_click=lambda: dialog.submit(False), color=None).classes(
                    "neu-ghost")
                ui.button("Replace", on_click=lambda: dialog.submit(True), color=None).classes(
                    "neu-danger")
        return bool(await dialog)

    async def choose_path() -> None:
        import webview  # only importable once nicegui[native] is installed

        result = await app.native.main_window.create_file_dialog(
            dialog_type=webview.FileDialog.SAVE,
            directory=model_paths.last_folder(),
            save_filename=f"{key}.tflite",
            file_types=("TFLite model (*.tflite)",),
        )
        if result:
            chosen = result[0] if isinstance(result, (list, tuple)) else result
            # The file training will really write (always .tflite).
            path = str(model_paths.model_file(chosen))
            if path != model_paths.normalize(chosen) and os.path.exists(path):
                # The system dialog only asked about the name as typed.
                if not await confirm_overwrite(path):
                    return
            model_paths.remember_folder(path)
            STATE["output_path"] = path
            path_label.text = path
            path_label.classes(add="neu-set")

    def start() -> None:
        if not STATE["output_path"]:
            ui.notify("Choose a save location first.", type="warning")
            return
        ui.navigate.to(f"/run/{key}")

    with theme.panel(4):
        ui.label(f"Save location for {entry.display_name}").classes("text-h5")

        path_label = ui.label(
            STATE["output_path"] or "No file chosen yet"
        ).classes("neu-path" + (" neu-set" if STATE["output_path"] else ""))

        ui.button("Choose save location...", on_click=choose_path, color=None).classes(
            "q-mt-sm"
        )
        with ui.row().classes("q-mt-md gap-2"):
            ui.button(
                "Back", on_click=lambda: ui.navigate.to(f"/topology/{key}"), color=None
            ).classes("neu-ghost")
            ui.button("Start training", on_click=start, color=None).classes("neu-primary")
