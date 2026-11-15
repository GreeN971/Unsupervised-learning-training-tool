import json
import threading
import time
from pathlib import Path

from nicegui import run as nicegui_run
from nicegui import ui

from app import theme
from app.pages.topology import topology_params
from app.state import STATE
from core.common import TrainingCancelled
from core.registry import REGISTRY


_QUANTIZATION_NAMES = {"none": "float32", "int8": "int8"}


def _quantization_report(output_path: Path) -> str | None:
    """One-line summary of the "quantization" section the converter writes
    to <name>_params.json next to the .tflite, or None if it's missing.
    """
    params_path = output_path.with_name(f"{output_path.stem}_params.json")
    try:
        q = json.loads(params_path.read_text())["quantization"]
    except (OSError, ValueError, KeyError):
        return None
    return (
        f"{_QUANTIZATION_NAMES.get(q['type'], q['type'])}, {q['size_bytes']} B - "
        f"test error vs. unquantized: mean {q['test_mean_abs_error']:.4f}, "
        f"max {q['test_max_abs_error']:.4f}; anomaly flags agree on "
        f"{q['test_flag_agreement'] * 100:.1f}% of test samples"
    )


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

        async def start_training() -> None:
            try:
                await nicegui_run.io_bound(
                    entry.run, params, output_path, progress_callback, cancel_event
                )
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
                    status_label.text = f"Saved to {output_path}"
                    status_label.classes(add="neu-ok")
                    report = _quantization_report(output_path)
                    if report:
                        ui.label(report).classes("neu-status")
                    ui.button(
                        "Back to start", on_click=lambda: ui.navigate.to("/"), color=None
                    ).classes("neu-primary")

        ticker = ui.timer(0.5, tick)
        ui.timer(0.05, start_training, once=True)
