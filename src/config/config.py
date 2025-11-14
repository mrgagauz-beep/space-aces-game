"""
Configuration loading and validation utilities.

This module provides a simple interface to access ROI, HSV thresholds,
key bindings and safety settings from JSON profiles.
"""

from pathlib import Path

import ujson as json


class Config:
    """
    Thin wrapper around the profile JSON.

    Provides convenience accessors for ROI, HSV, key bindings and safety
    thresholds. All read-only for now.
    """

    def __init__(self, path: str | Path = "profiles/default.json"):
        self.path = Path(path)
        self.data = self._load()

    def _load(self) -> dict:
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # --- ROI helpers ---
    def roi_main(self):
        """Return MAIN screen ROI as (x, y, w, h)."""
        return tuple(self.data["ROI"]["MAIN"])

    def roi_minimap(self):
        """Return MINIMAP ROI as (x, y, w, h)."""
        return tuple(self.data["ROI"]["MINIMAP"])

    # --- collections ---
    def hsv(self) -> dict:
        """Return HSV thresholds dictionary."""
        return self.data.get("HSV", {})

    def keys(self) -> dict:
        """Return key bindings dictionary."""
        return self.data.get("keys", {})

    def safety(self) -> dict:
        """Return safety thresholds dictionary."""
        return self.data.get("safety", {})
