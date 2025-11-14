"""
Scaffold for a future GUI control panel.

Planned features:
- Start / Stop buttons;
- Toggles: Farm, Collect, Auto-Flee;
- Live log view and basic stats (kills / boxes / time / state).

This is intentionally decoupled from the core bot logic for now.
Later it can be implemented using PyQt / PySide or another toolkit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BotStatsViewModel:
    """Simple view-model for exposing bot statistics to the UI."""

    kills: int = 0
    boxes: int = 0
    elapsed_sec: float = 0.0
    state: str = "INIT"


class ControlPanel:
    """
    Placeholder class for the main control panel UI.

    A concrete implementation (e.g. PyQt/PySide) should:
    - render start/stop buttons and checkboxes;
    - subscribe to bot stats and log events;
    - emit user intents back to the core/FSM layer.
    """

    def __init__(self) -> None:
        self.stats = BotStatsViewModel()

    # TODO: wire to a specific GUI framework (PyQt/PySide) and bot control API.
    def show(self) -> None:
        """Show the panel (to be implemented by a concrete UI stack)."""
        raise NotImplementedError("UI toolkit integration is not implemented yet.")

