from nicegui import ui

from app import theme
from app.state import STATE
from core.registry import REGISTRY


def _bump(field: ui.number, delta: float, lo: float | None, hi: float | None) -> None:
    value = (field.value or 0) + delta
    if lo is not None:
        value = max(value, lo)
    if hi is not None:
        value = min(value, hi)
    field.value = round(value, 6)


def _add_spinner(field: ui.number, step: float, lo: float | None, hi: float | None) -> None:
    with ui.column().classes("neu-spin"):
        ui.icon("keyboard_arrow_up").on(
            "click", lambda: _bump(field, step, lo, hi)
        )
        ui.icon("keyboard_arrow_down").on(
            "click", lambda: _bump(field, -step, lo, hi)
        )


def _coerce(param_type: str, value):
    """ui.number always reports floats (64 -> 64.0); int params must stay
    ints, since Keras/sklearn reject float batch sizes, epochs, etc.
    """
    if param_type == "int" and value is not None:
        return int(round(value))
    return value


def _param_field(p, values: dict):
    """The input for param ``p``, kept in sync with ``values``, and its info
    icon. Returns the input.
    """
    if p.type == "int":
        step = 1
        with ui.element("div").classes("relative"):
            field = ui.number(
                label=p.label, value=values[p.key],
                min=p.min, max=p.max, step=step, format="%.0f",
            ).classes("w-64 neu-number").props("outlined")
            _add_spinner(field, step, p.min, p.max)
    elif p.type == "float":
        step = 0.01
        with ui.element("div").classes("relative"):
            field = ui.number(
                label=p.label, value=values[p.key],
                min=p.min, max=p.max, step=step,
            ).classes("w-64 neu-number").props("outlined")
            _add_spinner(field, step, p.min, p.max)
    elif p.type == "bool":
        field = ui.switch(p.label, value=values[p.key])
    elif p.type == "checkbox":
        field = ui.checkbox(p.label, value=values[p.key])
    elif p.type == "choice":
        field = ui.select(
            p.choices, label=p.label, value=values[p.key],
        ).classes("w-64").props("outlined")
    else:  # pragma: no cover - registry-defined types only
        return None

    field.on_value_change(
        lambda e, k=p.key, t=p.type: values.__setitem__(k, _coerce(t, e.value))
    )

    with ui.icon("info").classes("neu-info"):
        with ui.tooltip().classes("max-w-xs"):
            ui.label(p.description)
            if p.warning:
                ui.label(p.warning).classes("font-bold mt-1")
    return field


def _dependent_field(p, parent, values: dict) -> None:
    """``p`` beside the row of its ``parent`` input, outside the row's width so
    the parent stays aligned with the other params. Enabled only while the
    parent has the value in ``p.enabled_when``.
    """
    parent_key, enabling_value = p.enabled_when
    with ui.row().classes("absolute left-full ml-2 items-center gap-2 no-wrap"):
        field = _param_field(p, values)
    field.classes("w-36")  # wraps a long label to fit the panel
    field.set_enabled(values[parent_key] == enabling_value)
    parent.on_value_change(lambda e: field.set_enabled(e.value == enabling_value))


def build(key: str) -> None:
    entry = REGISTRY[key]
    STATE["algorithm_key"] = key
    values = {p.key: STATE["params"].get(p.key, p.default) for p in entry.params}
    STATE["params"] = values

    def go_next() -> None:
        STATE["params"] = values
        ui.navigate.to(f"/topology/{key}")

    with theme.panel(2, wide=True):  # room for the fields beside Quantization
        ui.label(f"Configure {entry.display_name}").classes("text-h5")

        with ui.column().classes("gap-3 w-full max-w-md items-center"):
            for p in entry.params:
                if p.enabled_when:
                    continue  # drawn in the row of the param it depends on
                with ui.row().classes("items-center gap-2 relative"):
                    field = _param_field(p, values)
                    for dependent in entry.params:
                        if dependent.enabled_when and dependent.enabled_when[0] == p.key:
                            _dependent_field(dependent, field, values)

        with ui.row().classes("q-mt-md gap-2"):
            ui.button("Back", on_click=lambda: ui.navigate.to("/algorithm"), color=None).classes("neu-ghost")
            ui.button("Continue", on_click=go_next, color=None).classes("neu-primary")
