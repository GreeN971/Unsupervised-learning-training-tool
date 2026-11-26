from nicegui import ui

from app import theme
from app.pages import notices
from app.state import reset
from core import temp_csv
from core.registry import REGISTRY


def build() -> None:
    if not temp_csv.is_staged():
        # Reached without the CSV wizard having staged the training data.
        ui.navigate.to("/")
        return
    reset()

    options = {key: entry.display_name for key, entry in REGISTRY.items()}
    selection = {"key": next(iter(options))}

    @ui.refreshable
    def notices_view() -> None:
        # With the algorithm's default params; the Configure page re-checks
        # with the ones chosen there.
        notices.render(REGISTRY[selection["key"]])

    def on_select(e) -> None:
        selection["key"] = e.value
        notices_view.refresh()

    def go_next() -> None:
        blocking = notices.first_blocking(REGISTRY[selection["key"]].notices())
        if blocking:
            ui.notify(blocking, type="warning", multi_line=True)
            return
        ui.navigate.to(f"/configure/{selection['key']}")

    with theme.panel(1):
        ui.label("Choose an algorithm").classes("text-h5")

        ui.select(options, value=selection["key"], label="Algorithm").classes(
            "w-72"
        ).props("outlined").on_value_change(on_select)

        with ui.column().classes("w-full"):
            notices_view()

        with ui.row().classes("q-mt-sm gap-2"):
            ui.button(
                "Back", on_click=lambda: ui.navigate.to("/csv"), color=None
            ).classes("neu-ghost")
            ui.button("Continue", on_click=go_next, color=None).classes("neu-primary")
