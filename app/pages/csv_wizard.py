"""CSV wizard: between the Data and Algorithm steps, but not one of the
wizard's steps itself (no stepper).

Two sub-tabs. "Detect" works out the chosen file's delimiter and columns
(core.temp_csv.detect) as soon as the page opens and moves on to "Columns"
by itself when that succeeds. "Columns" lets the user include or exclude
each column; Continue writes only the included ones into the temporary CSV
slot, which training and the Topology estimates then read their features
from.
"""

from nicegui import run as nicegui_run
from nicegui import ui

from app import theme
from app.state import STATE
from core import temp_csv

# More included columns than this gets a note: multi-column training fits
# one detector per presence pattern, 2^n - 1 of them (core.common).
_MANY_COLUMNS = 6


def build() -> None:
    data_path = STATE["data_path"]
    if not data_path:
        ui.navigate.to("/")
        return

    detected: dict = {"info": None}
    include: dict[str, bool] = {}

    with theme.panel(None):
        ui.label("CSV wizard").classes("text-h5")
        ui.label(data_path).classes("neu-path neu-set")

        with ui.tabs().classes("neu-tabs") as tabs:
            detect_tab = ui.tab("Detect")
            columns_tab = ui.tab("Columns").props("disable")

        with ui.tab_panels(tabs, value=detect_tab, animated=False).classes(
            "w-full neu-tab-panels"
        ):
            with ui.tab_panel(detect_tab).classes("items-center gap-4"):
                detect_status = ui.label("Detecting columns and delimiter...").classes(
                    "neu-status"
                )
                detect_spinner = ui.spinner(size="lg")
                detect_back = ui.button(
                    "Back", on_click=lambda: ui.navigate.to("/"), color=None
                ).classes("neu-ghost")
                detect_back.set_visibility(False)

            with ui.tab_panel(columns_tab).classes("items-center gap-4"):
                summary_label = ui.label().classes("neu-status")
                columns_box = ui.column().classes("neu-columns gap-0")
                count_label = ui.label().classes("neu-status")
                with ui.row().classes("q-mt-sm gap-2"):
                    ui.button(
                        "Back", on_click=lambda: ui.navigate.to("/"), color=None
                    ).classes("neu-ghost")
                    continue_button = ui.button("Continue", color=None).classes(
                        "neu-primary"
                    )

    def update_count() -> None:
        n = sum(include.values())
        text = f"{n} of {len(include)} columns included"
        if n > _MANY_COLUMNS:
            text += (
                f" - training fits {2 ** n - 1} feature combinations, "
                "so it may take a long time"
            )
        count_label.text = text

    def show_columns(info: temp_csv.CsvInfo) -> None:
        previous = (
            STATE["features"] if STATE["features_path"] == data_path else None
        )
        summary_label.text = (
            f"{info.delimiter_name.capitalize()}-separated, "
            f"{len(info.columns)} columns, {info.n_rows} rows"
        )
        with columns_box:
            for name in info.columns:
                numeric = info.numeric[name]
                include[name] = numeric and (
                    name in previous if previous is not None else True
                )
                with ui.row().classes("w-full items-center justify-between no-wrap py-1"):
                    with ui.column().classes("gap-0"):
                        ui.label(name).classes("neu-column-name")
                        if not numeric:
                            ui.label("non-numeric").classes("neu-status")
                    switch = ui.switch(
                        "Include", value=include[name],
                        on_change=lambda e, n=name: (
                            include.__setitem__(n, e.value), update_count()
                        ),
                    )
                    switch.set_enabled(numeric)
        update_count()

    async def run_detection() -> None:
        try:
            info = await nicegui_run.io_bound(temp_csv.detect, data_path)
        except temp_csv.CsvDetectError as exc:
            detect_spinner.set_visibility(False)
            detect_status.text = f"Could not read the CSV: {exc}"
            detect_status.classes(add="neu-error")
            detect_back.set_visibility(True)
            return
        detected["info"] = info
        detect_spinner.set_visibility(False)
        detect_status.text = "Columns and delimiter detected."
        detect_status.classes(add="neu-ok")
        show_columns(info)
        columns_tab.props(remove="disable")
        tabs.value = columns_tab

    async def go_next() -> None:
        included = [name for name in detected["info"].columns if include.get(name)]
        if not included:
            ui.notify("Include at least one column.", type="warning")
            return
        continue_button.disable()
        try:
            await nicegui_run.io_bound(
                temp_csv.stage_columns, data_path, detected["info"], included
            )
        except (OSError, ValueError) as exc:
            continue_button.enable()
            ui.notify(f"Could not write the training data: {exc}", type="negative")
            return
        STATE["features"] = included
        STATE["features_path"] = data_path
        ui.navigate.to("/algorithm")

    continue_button.on_click(go_next)
    ui.timer(0.05, run_detection, once=True)
