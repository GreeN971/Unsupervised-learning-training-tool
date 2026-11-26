"""NiceGUI entry point: a native window wrapping the algorithm picker,
config form, topology, save-location, and run/progress pages.

Launched by bootstrap.py using the app's own venv - do not run this
directly outside that venv, it depends on nicegui[native] + the ML stack.
"""

import multiprocessing

from nicegui import app, ui

# pywebview defaults to a private (in-memory) browser profile, so localStorage
# - which the theme toggle uses to remember light/dark across launches - would
# be wiped every time the app restarts. This must run in both the main process
# and the native window-opener child (see the comment below on why main.py is
# re-executed in that child), so it's set here, unconditionally, before either
# process reaches ui.run()/webview.start().
app.native.start_args["private_mode"] = False

# native=True makes NiceGUI spawn a second process (multiprocessing "spawn")
# whose only job is to open the OS webview window pointed at this process's
# already-running server - ui.run() itself is a no-op there (NiceGUI checks
# multiprocessing.current_process().name != 'MainProcess' and returns early).
# But "spawn" re-executes this module from scratch in that child to rebuild
# import state, so any top-level import here still runs there too. If that
# import chain pulls in tensorflow (as app.pages -> core.registry does), the
# window-opener process ends up with two separate bundled LLVM instances
# loaded at once - TensorFlow's (XLA) and QtWebEngine's (Chromium) - which
# corrupts LLVM's process-global PassRegistry and crashes the native window.
# So the app's pages (and therefore the ML stack) are only imported/
# registered in the real main process, never in the window-opener child.
if multiprocessing.current_process().name == "MainProcess":
    from app import theme
    from app.pages import configure, csv_wizard, data, picker, save, topology
    from app.pages import run as run_page
    from app.pages import test as test_page
    from core import temp_csv

    theme.install()
    # Nothing staged in an earlier session is ever trained on by accident.
    temp_csv.clear()

    @ui.page("/")
    def index_page() -> None:
        data.build()

    @ui.page("/csv")
    def csv_wizard_page() -> None:
        csv_wizard.build()

    @ui.page("/algorithm")
    def algorithm_page() -> None:
        picker.build()

    @ui.page("/configure/{key}")
    def configure_page(key: str) -> None:
        configure.build(key)

    @ui.page("/topology/{key}")
    def topology_page(key: str) -> None:
        topology.build(key)

    @ui.page("/save/{key}")
    def save_page(key: str) -> None:
        save.build(key)

    @ui.page("/run/{key}")
    def run_page_route(key: str) -> None:
        run_page.build(key)

    @ui.page("/test")
    def test_page_route(model: str = "") -> None:
        test_page.build(model)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        native=True,
        window_size=(900, 700),
        title="Anomaly Detector → TFLite",
        reload=False,
    )
