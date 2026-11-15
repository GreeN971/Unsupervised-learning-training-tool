from nicegui import app, ui

from app import theme
from app.state import STATE
from core.registry import REGISTRY


def build(key: str) -> None:
    entry = REGISTRY[key]

    async def choose_path() -> None:
        import webview  # only importable once nicegui[native] is installed

        result = await app.native.main_window.create_file_dialog(
            dialog_type=webview.FileDialog.SAVE,
            save_filename=f"{key}.tflite",
            file_types=("TFLite model (*.tflite)",),
        )
        if result:
            chosen = result[0] if isinstance(result, (list, tuple)) else result
            STATE["output_path"] = chosen
            path_label.text = chosen
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
