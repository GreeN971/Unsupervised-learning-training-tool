"""The algorithm notices block shared by the Algorithm and Configure pages:
problems with the staged data and chosen params (core.notices - blocking
ones stop the wizard), then the algorithm's standing limitations.
"""

from nicegui import ui

_ICONS = {"block": "block", "warn": "warning", "info": "info"}


def _row(text: str, kind: str) -> None:
    with ui.element("div").classes(f"neu-notice neu-notice-{kind}"):
        ui.icon(_ICONS[kind])
        ui.label(text)


def render(entry, params: dict | None = None) -> list:
    """Draw ``entry``'s notices for ``params`` (defaults filled in) and the
    staged CSV. Returns the notices found, so the caller can stop on a
    blocking one.
    """
    found = entry.notices(params)
    with ui.element("div").classes("neu-notices"):
        if found:
            ui.label("Check before training").classes("neu-notices-title")
            for notice in found:
                _row(notice.text, "block" if notice.blocking else "warn")
        if entry.limitations:
            ui.label(f"Limitations of {entry.display_name}").classes("neu-notices-title")
            for text in entry.limitations:
                _row(text, "info")
    return found


def first_blocking(found: list) -> str | None:
    """Text of the first blocking notice in ``found``, or None."""
    return next((notice.text for notice in found if notice.blocking), None)
