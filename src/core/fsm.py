"""
Finite State Machine (FSM) for Space Aces Bot.

States: FARMING, FLEEING, SAFE (COOLDOWN can be added later).
Coordinating loop that ties vision, navigation, input and config together.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import logging
import csv
from datetime import datetime
from pathlib import Path

from config.config import Config
from controls.input import (
    emergency_stop,
    press_key,
    human_pause,
    move_click_abs,
)
from nav.navigation import compute_approach_point, go_to_minimap_point
from utils.logger import setup_logger
from vision import capture, minimap, screen
from vision.screen import find_mob_labels_main, mob_label_box_to_click_point


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
        self._trace_path = Path("logs") / "trace.csv"
        self._trace_initialized = False

        # Target tracking on the minimap
        self.current_target_mm: Optional[Tuple[int, int]] = None
        self.current_stage: Optional[str] = None  # "approach" / "engage" / None
        # Last known player position on minimap (for smoothing crosshair detection)
        self.last_player_mm: Optional[Tuple[int, int]] = None

    def _choose_ammo_for_map(self, map_name: str | None = None) -> Optional[str]:
        """
        Choose ammo key based on current map.

        For now this is a simple wrapper around the generic shoot key.
        """
        keys = self.cfg.keys()
        return keys.get("shoot")

    def _pick_closest_enemy_mm(
        self, enemies_mm: List[Tuple[int, int]], player_mm: Optional[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        """
        Выбирает ближайшего врага на миникарте относительно позиции игрока.

        enemies_mm: список (x, y)
        player_mm: (px, py)
        """
        if not enemies_mm or player_mm is None:
            return None

        px, py = player_mm
        best: Optional[Tuple[int, int]] = None
        best_d2: Optional[float] = None

        for ex, ey in enemies_mm:
            dx = ex - px
            dy = ey - py
            d2 = dx * dx + dy * dy
            if best is None or best_d2 is None or d2 < best_d2:
                best = (ex, ey)
                best_d2 = d2

        return best

    # --------- trace logging ---------
    def logger_stat(self, obs: Dict, action: str, outcome: str = "") -> None:
        """
        Append a single RL-friendly trace row to CSV.

        Columns: timestamp, state, enemies_mm, crates_main, mobs_main,
        action, hp, shield, outcome.
        """
        self._trace_path.parent.mkdir(exist_ok=True)

        is_new = not self._trace_path.exists()
        with self._trace_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(
                    [
                        "timestamp",
                        "state",
                        "enemies_mm",
                        "crates_main",
                        "mobs_main",
                        "action",
                        "hp",
                        "shield",
                        "outcome",
                    ]
                )

            ts = datetime.utcnow().isoformat()
            enemies_mm = len(obs.get("enemies_mm", []))
            crates_main = len(obs.get("crates", []))
            mobs_main = len(obs.get("mobs_main", []))
            hp = obs.get("hp_pct", 0)
            shield = obs.get("shield_pct", 0)

            writer.writerow(
                [
                    ts,
                    self.state,
                    enemies_mm,
                    crates_main,
                    mobs_main,
                    action,
                    hp,
                    shield,
                    outcome,
                ]
            )

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
        me_mm = minimap.find_player_crosshair(frame_mm, self.cfg)
        if me_mm is not None:
            self.last_player_mm = me_mm
        else:
            me_mm = self.last_player_mm

        mobs_main = screen.find_mobs_main(frame_main, self.cfg)
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

        # Fresh minimap observations for decision making
        mm_frame = capture.grab_minimap(self.cfg)
        enemies_mm: List[Tuple[int, int]] = minimap.find_enemies(
            mm_frame, self.cfg, map_name=None
        )
        player_mm_raw = minimap.find_player_crosshair(mm_frame, self.cfg)
        if player_mm_raw is not None:
            self.last_player_mm = player_mm_raw
        player_mm = self.last_player_mm

        # Reset target if there are no enemies
        if not enemies_mm:
            if self.current_target_mm is not None or self.current_stage is not None:
                self.logger.debug("FARMING: no enemies on minimap, reset target")
            self.current_target_mm = None
            self.current_stage = None

        if crates:
            x, y, w, h = crates[0]
            cx, cy = x + w // 2, y + h // 2
            self.logger.info("FARMING: collecting crate at (%d,%d)", cx, cy)
            # Click directly in main screen absolute coords
            mx, my, _, _ = self.cfg.roi_main()
            move_click_abs(mx + cx, my + cy)
            self.stats.boxes += 1
            self.logger_stat(obs, action="collect_crate", outcome="ok")

        # Target acquisition on minimap
        if self.current_target_mm is None:
            target = self._pick_closest_enemy_mm(enemies_mm, player_mm)
            if target is not None:
                self.current_target_mm = target
                self.current_stage = "approach"
                self.logger.info("FARMING: picked target_mm=%s", target)

        # Approach stage: fly towards the selected enemy
        if self.current_stage == "approach" and self.current_target_mm is not None:
            if player_mm is None:
                self.logger.debug(
                    "FARMING: no player position on minimap, cannot approach"
                )
            else:
                enemy = self.current_target_mm
                click_xy = compute_approach_point(player_mm, enemy, factor=0.8)
                self.logger.info(
                    "FARMING: approach target_mm=%s via %s", enemy, click_xy
                )

                success = go_to_minimap_point(
                    self.cfg.roi_minimap(),
                    click_xy,
                    self.cfg,
                    grab_minimap_fn=lambda: capture.grab_minimap(self.cfg),
                    find_player_fn=minimap.find_player_crosshair,
                    timeout=8.0,
                )
                self.logger.info(
                    "FARMING: approach target_mm=%s success=%s",
                    enemy,
                    success,
                )
                self.current_stage = "engage"
                return

        # Engage stage: lock and shoot using labels on main screen
        if self.current_stage == "engage" and self.current_target_mm is not None:
            main_bgr = capture.grab_main(self.cfg)
            labels = find_mob_labels_main(main_bgr, self.cfg)
            if labels:
                h, w = main_bgr.shape[:2]
                cx_screen = w // 2
                cy_screen = h // 2

                def _center(box):
                    x, y, bw, bh = box
                    return x + bw // 2, y + bh // 2

                best_box = None
                best_d2 = None
                for box in labels:
                    cx, cy = _center(box)
                    dx = cx - cx_screen
                    dy = cy - cy_screen
                    d2 = dx * dx + dy * dy
                    if best_box is None or best_d2 is None or d2 < best_d2:
                        best_box = box
                        best_d2 = d2

                if best_box is not None:
                    click_x, click_y = mob_label_box_to_click_point(
                        best_box, self.cfg
                    )
                    move_click_abs(click_x, click_y)

                    ammo_key = self._choose_ammo_for_map()
                    if ammo_key:
                        press_key(ammo_key)
                        self.logger.info(
                            "FARMING: lock target via label at (%d,%d), ammo=%s",
                            click_x,
                            click_y,
                            ammo_key,
                        )
                        self.logger_stat(
                            obs,
                            action="lock_via_label",
                            outcome=f"ammo:{ammo_key}",
                        )
                    else:
                        self.logger.info(
                            "FARMING: lock target via label at (%d,%d), no ammo key",
                            click_x,
                            click_y,
                        )
                    # After engage, reset target to allow new decisions
                    self.current_target_mm = None
                    self.current_stage = None
                    return

            # Fallback: simple shoot if no labels are visible
            shoot_key = self.cfg.keys().get("shoot")
            if shoot_key:
                press_key(shoot_key)
            self.logger_stat(obs, action="engage_enemy", outcome="shot")
        elif not crates and not enemies_mm:
            # No crates and no enemies: allow idle-style human pause.
            human_pause("idle")

        if self._hp_low(obs):
            self.logger.info("FARMING: HP low, switching to FLEEING")
            self.logger_stat(obs, action="hp_low", outcome="to_fleeing")
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
        self.logger_stat(obs, action="flee_macro", outcome="to_safe")
        self.state = BotState.SAFE

    def _act_safe(self, obs: Dict) -> None:
        """
        SAFE behavior: short cooldown / recovery before returning to FARMING.
        """
        self.logger.info("SAFE: cooldown start")
        time.sleep(3.0)
        self.logger.info("SAFE: returning to FARMING")
        self.logger_stat(obs, action="cooldown", outcome="to_farming")
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
