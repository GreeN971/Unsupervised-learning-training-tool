import threading
import time
from pathlib import Path
from urllib.parse import quote

from nicegui import run as nicegui_run
from nicegui import ui

from app import theme
from app.pages.topology import topology_params
from app.state import STATE
from core import model_paths
from core.common import TrainingCancelled
from core.registry import REGISTRY


def _result(title: str, value: str) -> None:
    """One prominent line of the finished run: a small title and its value."""
    with ui.element("div").classes("neu-result"):
        ui.label(title).classes("neu-notices-title")
        ui.label(value).classes("neu-result-value")


def _size_text(path: Path) -> str:
    size = path.stat().st_size
    return f"{size / 1024:.2f} KiB ({size} B)"


def build(key: str) -> None:
    if not STATE["output_path"] or STATE["algorithm_key"] != key:
        # Reached directly (e.g. stale navigation) without going through the
        # save page first - there's no path to write to yet.
        with theme.panel(5):
            ui.label("No save location chosen - start over.")
            ui.button(
                "Back to start", on_click=lambda: ui.navigate.to("/"), color=None
            ).classes("neu-primary")
        return

    entry = REGISTRY[key]
    # Validated on the Topology page's Continue, so this won't raise here.
    params = {**STATE["params"], **topology_params(STATE["topology"])}
    output_path = Path(STATE["output_path"])

    with theme.panel(5):
        ui.label(f"Running {entry.display_name}").classes("text-h5")

        status = {
            "epoch": 0,
            "total_epochs": params.get("EPOCHS", 1) or 1,
            "done": False,
            "error": None,
            "cancelled": False,
            "list_warning": None,
        }
        cancel_event = threading.Event()
        start_time = time.monotonic()

        def progress_callback(epoch: int, total_epochs: int) -> None:
            status["epoch"] = epoch
            status["total_epochs"] = total_epochs

        elapsed_label = ui.label("00:00").classes("neu-readout")
        progress_bar = ui.linear_progress(
            value=0, show_value=False, color=None
        ).classes("w-full max-w-md")
        percent_label = ui.label("0%").classes("neu-status")
        status_label = ui.label().classes("neu-status")

        def request_cancel() -> None:
            cancel_event.set()
            interrupt_button.disable()
            status_label.text = (
                "Cancelling... (takes effect once the current step finishes)"
            )

        interrupt_button = ui.button(
            "Interrupt", on_click=request_cancel, color=None
        ).classes("neu-danger q-mt-md")

        def train() -> None:
            entry.run(params, output_path, progress_callback, cancel_event)
            # Only reached when the model was written. Done here, not in
            # tick(), so it's remembered even if the page has been left.
            status["list_warning"] = model_paths.remember(output_path)

        async def start_training() -> None:
            try:
                await nicegui_run.io_bound(train)
            except TrainingCancelled:
                status["cancelled"] = True
            except Exception as exc:  # surfaced to the user, not swallowed
                status["error"] = str(exc)
            finally:
                status["done"] = True

        def tick() -> None:
            elapsed = time.monotonic() - start_time
            minutes, seconds = divmod(int(elapsed), 60)
            elapsed_label.text = f"{minutes:02d}:{seconds:02d}"
            fraction = status["epoch"] / status["total_epochs"]
            progress_bar.value = fraction
            percent_label.text = f"{fraction * 100:.0f}%"
            if status["done"]:
                ticker.cancel()
                interrupt_button.set_visibility(False)
                if status["cancelled"]:
                    # Same algorithm, same params the user had set - just no
                    # longer running.
                    ui.navigate.to(f"/configure/{key}")
                elif status["error"]:
                    status_label.text = f"Failed: {status['error']}"
                    status_label.classes(add="neu-error")
                    ui.button(
                        "Back to start", on_click=lambda: ui.navigate.to("/"), color=None
                    ).classes("neu-primary")
                else:
                    status_label.set_visibility(False)
                    _result("Saved to", str(output_path))
                    _result("Size of the .tflite file", _size_text(output_path))
                    if status["list_warning"]:
                        ui.label(status["list_warning"]).classes("neu-status neu-error")
                    with ui.row().classes("gap-4 justify-center"):
                        ui.button(
                            "Back to start", on_click=lambda: ui.navigate.to("/"), color=None
                        ).classes("neu-primary")
                        # Names the model, so this one opens even if the list
                        # of saved models couldn't be written.
                        ui.button(
                            "Test a trained model", color=None,
                            on_click=lambda: ui.navigate.to(
                                f"/test?model={quote(str(output_path), safe='')}"),
                        ).classes("neu-primary")

        ticker = ui.timer(0.5, tick)
        ui.timer(0.05, start_training, once=True)
