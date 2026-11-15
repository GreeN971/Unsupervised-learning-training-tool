from nicegui import ui

from app import theme
from app.state import reset
from core.registry import REGISTRY


def build() -> None:
    reset()

    options = {key: entry.display_name for key, entry in REGISTRY.items()}
    selection = {"key": next(iter(options))}

    with theme.panel(1):
        ui.label("Choose an algorithm").classes("text-h5")

        ui.select(options, value=selection["key"], label="Algorithm").classes(
            "w-72"
        ).props("outlined").on_value_change(lambda e: selection.__setitem__("key", e.value))

        with ui.row().classes("q-mt-sm gap-2"):
            ui.button(
                "Back", on_click=lambda: ui.navigate.to("/"), color=None
            ).classes("neu-ghost")
            ui.button(
                "Continue",
                on_click=lambda: ui.navigate.to(f"/configure/{selection['key']}"),
                color=None,
            ).classes("neu-primary")
