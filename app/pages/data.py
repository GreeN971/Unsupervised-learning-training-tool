from nicegui import app, ui

from app import theme
from app.state import STATE


def build() -> None:
    async def choose_file() -> None:
        import webview  # only importable once nicegui[native] is installed

        result = await app.native.main_window.create_file_dialog(
            dialog_type=webview.FileDialog.OPEN,
            file_types=("CSV files (*.csv)",),
        )
        if result:
            chosen = result[0] if isinstance(result, (list, tuple)) else result
            if not chosen.lower().endswith(".csv"):
                ui.notify("Only .csv files are allowed.", type="warning")
                return
            STATE["data_path"] = chosen
            path_label.text = chosen
            path_label.classes(add="neu-set")

    with theme.panel(0):
        ui.label("Choose a data file").classes("text-h5")
        ui.label("Only .csv files are supported.").classes("text-caption").style(
            "color: var(--ng-text-muted)"
        )

        path_label = ui.label(
            STATE["data_path"] or "No file chosen yet"
        ).classes("neu-path" + (" neu-set" if STATE["data_path"] else ""))

        ui.button("Choose file...", on_click=choose_file, color=None).classes("q-mt-sm")

        ui.button(
            "Continue",
            on_click=lambda: ui.navigate.to("/algorithm"),
            color=None,
        ).classes("neu-primary q-mt-sm")
