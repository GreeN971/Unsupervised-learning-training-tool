"""First-run setup: find a compatible Python, create/reuse .venv, install
requirements.txt, then hand off to the app.

Run this via run.sh (Linux/macOS) or run.bat (Windows) - they just locate
*some* Python 3 to start this script; all the version-aware logic lives here.
"""

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VENV_DIR = REPO_ROOT / ".venv"
REQUIREMENTS = REPO_ROOT / "requirements.txt"
APP_ENTRY = REPO_ROOT / "app" / "main.py"
INSTALL_MARKER = VENV_DIR / ".requirements_hash"

# TensorFlow (the tightest constraint in requirements.txt) only ships wheels
# for these CPython minors as of this writing - newest first.
SUPPORTED_MINORS = (13, 12, 11, 10)


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _requirements_hash() -> str:
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()


def _requirements_already_installed() -> bool:
    """True only if a previous run *fully completed* installing the exact
    requirements.txt currently on disk - not just if .venv/ exists, since an
    interrupted install (e.g. cancelled download) leaves a venv behind with
    nothing usable installed in it.
    """
    if not INSTALL_MARKER.exists():
        return False
    return INSTALL_MARKER.read_text().strip() == _requirements_hash()


def find_compatible_python() -> Path:
    """Return a Python 3.10-3.13 interpreter, preferring the newest available."""
    if sys.version_info[:2] in {(3, m) for m in SUPPORTED_MINORS}:
        return Path(sys.executable)

    candidates: list[list[str]] = []
    if os.name == "nt":
        for minor in SUPPORTED_MINORS:
            candidates.append(["py", f"-3.{minor}"])
    else:
        for minor in SUPPORTED_MINORS:
            exe = shutil.which(f"python3.{minor}")
            if exe:
                candidates.append([exe])

    for cmd in candidates:
        try:
            result = subprocess.run(
                [*cmd, "-c", "import sys; print(sys.executable)"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return Path(result.stdout.strip())
        except OSError:
            continue

    print(
        "\nNo compatible Python interpreter found.\n"
        "This app needs Python 3.10, 3.11, 3.12, or 3.13 - TensorFlow does not "
        "yet publish wheels for newer releases. Please install one of those "
        "versions (they can coexist with whatever Python you already have) "
        "and run this launcher again.\n",
        file=sys.stderr,
    )
    sys.exit(1)


def create_venv(python_exe: Path) -> None:
    print(f"Creating virtual environment at {VENV_DIR} using {python_exe} ...")
    subprocess.run([str(python_exe), "-m", "venv", str(VENV_DIR)], check=True)


def install_requirements() -> None:
    venv_python = _venv_python(VENV_DIR)
    print("Upgrading pip ...")
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True
    )

    print(f"Installing dependencies from {REQUIREMENTS.name} (this can take a "
          "while the first time - tensorflow and torch are large) ...")
    process = subprocess.Popen(
        [str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    return_code = process.wait()
    if return_code != 0:
        print(
            "\nDependency installation did not finish (interrupted or "
            "failed). Run this launcher again to resume - it will pick up "
            "where pip left off using its download cache.",
            file=sys.stderr,
        )
        sys.exit(return_code)

    INSTALL_MARKER.write_text(_requirements_hash())


def launch_app() -> None:
    venv_python = _venv_python(VENV_DIR)
    if not APP_ENTRY.exists():
        print(
            f"\nEnvironment ready at {VENV_DIR}, but {APP_ENTRY} doesn't exist "
            "yet - the app UI hasn't been built yet."
        )
        return
    # Run as "-m app.main" (not a bare script path) so the app's absolute
    # imports (app.pages, core.registry, ...) resolve - that requires the
    # repo root to be on sys.path, which "-m" gives us via cwd.
    os.chdir(REPO_ROOT)
    os.execv(str(venv_python), [str(venv_python), "-m", "app.main"])


def main() -> None:
    if not REQUIREMENTS.exists():
        print(f"Missing {REQUIREMENTS}", file=sys.stderr)
        sys.exit(1)

    if _requirements_already_installed():
        print(f"Reusing existing virtual environment at {VENV_DIR}.")
    else:
        if not _venv_python(VENV_DIR).exists():
            python_exe = find_compatible_python()
            create_venv(python_exe)
        install_requirements()

    launch_app()


if __name__ == "__main__":
    main()
