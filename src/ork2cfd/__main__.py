"""Module entrypoint: launches the desktop GUI by default or runs the CLI."""

from __future__ import annotations

import sys

from .cli import main


def _entrypoint() -> None:
    if len(sys.argv) <= 1:
        from .gui.app import launch_gui

        launch_gui()
    else:
        main()


if __name__ == "__main__":
    _entrypoint()
