"""
Finite State Machine (FSM) for Space Aces Bot.

States: FARMING, FLEEING, SAFE (COOLDOWN can be added later).
Coordinating loop that ties vision, navigation, input and config together.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import logging

from config.config import Config
from controls.input import (
    emergency_stop,
    press_key,
)
from nav.navigation import approach_enemy, go_to_minimap_point
from utils.logger import setup_logger
from vision import capture, minimap, screen


class BotState:
    """Simple namespace for FSM state names."""

    FARMING = "FARMING"
    FLEEING = "FLEEING"
    SAFE = "SAFE"


@dataclass
class BotStats:
    """Runtime counters for simple telemetry."""

    kills: int = 0
    boxes: int = 0
    deaths: int = 0


class Bot:
    """
    High-level bot controller with simple FSM.

    The bot operates in three main states:
    - FARMING: hunting mobs and collecting crates;
    - FLEEING: emergency macro and retreat;
    - SAFE: short recovery before returning to FARMING.
    """

    def __init__(self, cfg: Config | None = None):
        setup_logger()
        self.cfg = cfg or Config()
        self.logger = logging.getLogger(__name__)

        self.state: str = BotState.FARMING
        self.stats: BotStats = BotStats()

        self.logger.info("FSM Bot initialized in state=%s", self.state)

    # --------- sensing ---------
    def _sense(self) -> Dict:
        """
        Collect observations from both main screen and mini-map.

        Returns
        -------
        dict
            Includes frames and detected entities.
        """
        frame_main = capture.grab_main(self.cfg)
        frame_mm = capture.grab_minimap(self.cfg)

        enemies_mm = minimap.find_enemies(frame_mm, self.cfg)
        me_mm = minimap.find_player_crosshair(frame_mm)

        mobs_main = screen.find_mobs_main(frame_main)
        crates = screen.find_crates_main(frame_main, self.cfg)

        # TODO: hook actual HP/shield parsing from UI.
        safety_cfg = self.cfg.safety()
        hp_pct = 100
        shield_pct = 100

        obs = {
            "frame_main": frame_main,
            "frame_minimap": frame_mm,
            "enemies_mm": enemies_mm,
            "me_mm": me_mm,
            "mobs_main": mobs_main,
            "crates": crates,
            "hp_pct": hp_pct,
            "shield_pct": shield_pct,
            "safety": safety_cfg,
        }

        self.logger.debug(
            "Sense: enemies_mm=%d crates=%d mobs_main=%d me=%s hp=%d shield=%d",
            len(enemies_mm),
            len(crates),
            len(mobs_main),
            me_mm,
            hp_pct,
            shield_pct,
        )

        return obs

    # --------- actions ---------
    def _hp_low(self, obs: Dict) -> bool:
        hp = obs["hp_pct"]
        shield = obs["shield_pct"]
        safety = obs["safety"]
        return hp <= safety.get("hp_flee_pct", 35) or shield <= safety.get(
            "shield_flee_pct", 25
        )

    def _act_farming(self, obs: Dict) -> None:
        """
        FARMING behavior:
        - Prefer crates on main screen;
        - Otherwise, move towards enemies on mini-map and shoot;
        - Switch to FLEEING if HP/shield low.
        """
        crates: List[Tuple[int, int, int, int]] = obs["crates"]
        enemies_mm: List[Tuple[int, int]] = obs["enemies_mm"]
        me_mm = obs["me_mm"]

        if crates:
            x, y, w, h = crates[0]
            cx, cy = x + w // 2, y + h // 2
            self.logger.info("FARMING: collecting crate at (%d,%d)", cx, cy)
            # Click directly in main screen absolute coords
            from controls.input import move_click_abs

            # ROI MAIN gives top-left corner of main area
            mx, my, _, _ = self.cfg.roi_main()
            move_click_abs(mx + cx, my + cy)
            self.stats.boxes += 1

        elif enemies_mm and me_mm is not None:
            enemy = enemies_mm[0]
            target = approach_enemy(self.cfg.roi_minimap(), me_mm, enemy, factor=0.8)

            self.logger.info("FARMING: approaching enemy %s via %s", enemy, target)

            def poll_frame():
                return capture.grab_main(self.cfg)

            # NOTE: we use main grab for now; later can switch to full-screen.
            go_to_minimap_point(self.cfg.roi_minimap(), target, poll_frame)

            shoot_key = self.cfg.keys().get("shoot")
            if shoot_key:
                press_key(shoot_key)

        if self._hp_low(obs):
            self.logger.info("FARMING: HP low, switching to FLEEING")
            self.state = BotState.FLEEING

    def _act_fleeing(self, obs: Dict) -> None:
        """
        FLEEING behavior: execute a simple survival macro.

        Sequence (if bound in profile):
        - switch to config2;
        - afterburner;
        - cloak;
        - repair.
        """
        keys = self.cfg.keys()

        for key_name in ("config2", "afterburner", "cloak", "repair"):
            k = keys.get(key_name)
            if not k:
                continue
            self.logger.info("FLEEING: macro %s (%s)", key_name, k)
            press_key(k)
            time.sleep(0.3)

        # TODO: navigate to nearest portal using templates/vision.
        self.logger.info("FLEEING: macro done, entering SAFE")
        self.state = BotState.SAFE

    def _act_safe(self, obs: Dict) -> None:
        """
        SAFE behavior: short cooldown / recovery before returning to FARMING.
        """
        self.logger.info("SAFE: cooldown start")
        time.sleep(3.0)
        self.logger.info("SAFE: returning to FARMING")
        self.state = BotState.FARMING

    # --------- main loop ---------
    def run(self) -> None:
        """
        Main FSM loop.

        Continues until emergency_stop() returns True.
        """
        self.logger.info("FSM run loop started")
        try:
            while not emergency_stop():
                obs = self._sense()

                if self.state == BotState.FARMING:
                    self._act_farming(obs)
                elif self.state == BotState.FLEEING:
                    self._act_fleeing(obs)
                elif self.state == BotState.SAFE:
                    self._act_safe(obs)
                else:
                    self.logger.warning("Unknown state: %s", self.state)
                    self.state = BotState.FARMING

                time.sleep(0.2)
        except Exception as exc:  # pragma: no cover - safety net
            self.logger.exception("FSM crashed: %s", exc)
        finally:
            self.logger.info(
                "FSM stopped. Stats: kills=%d boxes=%d deaths=%d",
                self.stats.kills,
                self.stats.boxes,
                self.stats.deaths,
            )
