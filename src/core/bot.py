import json
import logging
import time
from pathlib import Path

from controls.input import is_exit_pressed, move_and_click, press_key
from vision.capture import capture_screen
from vision.detector import find_boxes, find_enemies


class Bot:
    def __init__(self, profile_path: str | Path | None = None):
        # Resolve profile path relative to project root by default
        if profile_path is None:
            profile_path = (
                Path(__file__).resolve().parent.parent.parent
                / "profiles"
                / "default.json"
            )
        else:
            profile_path = Path(profile_path)

        with profile_path.open("r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.state = "FARMING"

        # Configure logging: file + console
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            filename="bot.log",
            filemode="w",
        )

        self.logger = logging.getLogger(__name__)

        # Add console handler if not already present
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s %(levelname)s %(message)s"
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        self.logger.info("Bot initialized with state: %s", self.state)
        self.logger.info("Loaded profile from: %s", profile_path)

    def run(self):
        self.logger.info("Bot main loop started")

        while True:
            # Take screenshot of ROI
            frame = capture_screen(self.config["roi"])

            # Detect enemies and boxes
            enemies = find_enemies(frame, self.config)
            boxes = find_boxes(frame, self.config)

            self.logger.info(
                "Detected %d enemies, %d boxes", len(enemies), len(boxes)
            )

            if self.state == "FARMING":
                self._handle_farming(enemies, boxes)
            elif self.state == "FLEEING":
                self._handle_fleeing(enemies)
            elif self.state == "SAFE":
                self._handle_safe()

            # Check for exit hotkey
            if is_exit_pressed():
                self.logger.info("Exit key pressed, stopping bot")
                break

            time.sleep(0.1)

        self.logger.info("Bot main loop stopped")

    def _handle_farming(self, enemies, boxes):
        # Engage first enemy if any
        if enemies:
            x, y = enemies[0]
            self.logger.info("Engaging enemy at (%d, %d)", x, y)
            move_and_click(x, y)
            shoot_key = self.config.get("shoot_key")
            if shoot_key:
                press_key(shoot_key)

        # Otherwise collect first box if any
        elif boxes:
            bx, by = boxes[0]
            self.logger.info("Collecting box at (%d, %d)", bx, by)
            move_and_click(bx, by)
            loot_key = self.config.get("loot_key")
            if loot_key:
                press_key(loot_key)

        # Simple danger heuristic: too many enemies -> flee
        if len(enemies) > 5:
            self.state = "FLEEING"
            self.logger.info(
                "Too many enemies detected (%d). Switching state to FLEEING",
                len(enemies),
            )

    def _handle_fleeing(self, enemies):
        flee_key = self.config.get("flee_key")
        if flee_key:
            self.logger.info("Fleeing using key: %s", flee_key)
            press_key(flee_key)
        else:
            self.logger.info("Fleeing... (no flee_key configured)")

        time.sleep(1.0)

        # If no enemies remain, consider safe
        if len(enemies) == 0:
            self.state = "SAFE"
            self.logger.info("No enemies detected. Switching state to SAFE")

    def _handle_safe(self):
        self.logger.info("Safe mode - no immediate threats")
        time.sleep(2.0)
        self.state = "FARMING"
        self.logger.info("Switching state back to FARMING")


