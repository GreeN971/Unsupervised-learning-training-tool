"""Remembers where trained models are saved, so the Test model page finds
them wherever the user put them - never by looking in the app's own folder.

Kept in .app_state/models.json rather than app.state.STATE or .tmp/, so it
survives a wizard reset, a new dataset and an app restart. A list that can't
be read or written never stops training: the functions return a warning for
the user instead of raising.
"""

import json
import os
import threading
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
LIST_PATH = _REPO_ROOT / ".app_state" / "models.json"
MAX_MODELS = 20

# Training finishes on a worker thread while the GUI may be editing the list.
_lock = threading.Lock()


def normalize(path) -> str:
    """Absolute path as stored in the list (symlinks are left alone)."""
    return os.path.abspath(os.path.expanduser(str(path)))


def model_file(chosen) -> Path:
    """The .tflite file a run really writes for the path chosen in the save
    dialog: ``.tflite`` is appended unless it is already there, so a name
    like "room.v2" is neither cut to "room" nor left without an extension.
    """
    chosen = Path(normalize(chosen))
    if chosen.suffix.lower() == ".tflite":
        return chosen.with_suffix(".tflite")
    return chosen.with_name(chosen.name + ".tflite")


def _read() -> tuple[dict, str | None]:
    empty = {"models": [], "last_folder": ""}
    try:
        data = json.loads(LIST_PATH.read_text(encoding="utf-8"))
        models, folder = data["models"], data.get("last_folder", "")
        if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
            raise ValueError("not a list of paths")
        return {"models": models, "last_folder": folder if isinstance(folder, str) else ""}, None
    except FileNotFoundError:
        return empty, None
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return empty, f"The list of saved models could not be read ({exc}); starting an empty one."


def _write(data: dict) -> str | None:
    try:
        LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Written aside and swapped in, so a crash can't leave half a list.
        temporary = LIST_PATH.with_suffix(".json.new")
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, LIST_PATH)
    except OSError as exc:
        return f"The model's location could not be remembered ({exc})."
    return None


def _update(change) -> str | None:
    with _lock:
        data, read_warning = _read()
        if read_warning and LIST_PATH.is_file():
            try:  # keep the damaged file rather than overwrite it
                os.replace(LIST_PATH, LIST_PATH.with_suffix(".json.damaged"))
            except OSError:
                pass
        change(data)
        return _write(data) or read_warning


def load() -> tuple[list[str], str | None]:
    """Remembered models, most recent first, and a warning or None."""
    data, warning = _read()
    return data["models"], warning


def remember(path) -> str | None:
    """Put ``path`` first in the list. Returns a warning or None."""
    path = normalize(path)

    def change(data):
        data["models"] = [path] + [m for m in data["models"] if m != path]
        del data["models"][MAX_MODELS:]
        data["last_folder"] = os.path.dirname(path)

    return _update(change)


def forget(path) -> str | None:
    """Take ``path`` off the list (the file itself is not touched)."""
    path = normalize(path)
    return _update(lambda data: data.update(
        models=[m for m in data["models"] if m != path]))


def remember_folder(path) -> str | None:
    """Start the next file dialog in the folder of ``path``."""
    folder = os.path.dirname(normalize(path))
    return _update(lambda data: data.update(last_folder=folder))


def last_folder() -> str:
    """Folder the file dialogs start in, or "" (the system's default)."""
    folder = _read()[0]["last_folder"]
    return folder if folder and os.path.isdir(folder) else ""
