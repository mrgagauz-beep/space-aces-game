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
import random

from config.config import Config
from controls.input import (
    emergency_stop,
    press_key,
    human_pause,
    move_click_abs,
    left_click_world,
    double_left_click_world,
    fire_weapon_for_map,
)
from nav.navigation import compute_approach_point, go_to_minimap_point
from utils.coords import screen_from_main_vision_point
from utils.logger import setup_logger
from vision import capture, minimap, screen
from vision.screen import MainEnemy, MainTargetKind, detect_main_targets_main


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
        # Target tracking on the main screen (click point in MAIN-local coords)
        self.current_target_main: Optional[Tuple[float, float]] = None
        # Last known player position on minimap (for smoothing crosshair detection)
        self.last_player_mm: Optional[Tuple[int, int]] = None

        # Simple cooldowns and flags for actions (seconds)
        self._last_fire_ts: float = 0.0
        self._last_world_click_ts: float = 0.0
        self.fire_started_for_target: bool = False
        self.engage_click_attempts: int = 0

        farming_cfg = self.cfg.farming() if hasattr(self.cfg, "farming") else {}
        # Maximum number of ENGAGE click-burst attempts per target.
        self.max_engage_click_attempts: int = int(
            farming_cfg.get("engage_max_click_attempts", 3)
        )
        # Minimal interval between click bursts on MAIN (milliseconds).
        self.farm_world_click_cooldown_ms: int = int(
            farming_cfg.get("world_click_cooldown_ms", 300)
        )
        # Radius (in MAIN pixels) within which we keep the current target point.
        self.farm_main_target_keep_radius_px: float = float(
            farming_cfg.get("main_target_keep_radius_px", 80.0)
        )
        # How many ticks to wait after last visible MAIN target
        # before falling back to minimap-driven search/approach.
        self.farming_main_focus_timeout_ticks: int = (
            self.cfg.farming_main_focus_timeout_ticks()
            if hasattr(self.cfg, "farming_main_focus_timeout_ticks")
            else int(farming_cfg.get("main_focus_timeout_ticks", 15))
        )
        self.farming_no_main_ticks: int = 0

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

    def _resolve_enemy_click_point(
        self, bbox: Tuple[int, int, int, int]
    ) -> Tuple[float, float]:
        """
        Return a human-like click point inside enemy bbox on MAIN.

        Adds padding from bbox edges and a small random jitter around
        the padded center to avoid pixel-perfect repetition.
        """
        x, y, w, h = bbox
        x_min = float(x)
        y_min = float(y)
        x_max = x_min + float(w)
        y_max = y_min + float(h)

        pad_x = max(min((x_max - x_min) * 0.2, 15.0), 3.0)
        pad_y = max(min((y_max - y_min) * 0.2, 15.0), 3.0)

        left = x_min + pad_x
        right = x_max - pad_x
        top = y_min + pad_y
        bottom = y_max - pad_y

        cx = (left + right) * 0.5
        cy = (top + bottom) * 0.5

        jitter_px = 2.0
        cx += random.uniform(-jitter_px, jitter_px)
        cy += random.uniform(-jitter_px, jitter_px)

        cx = max(left, min(right, cx))
        cy = max(top, min(bottom, cy))

        return cx, cy

    def _nearest_mob_to_point(
        self,
        mobs_main: List[Tuple[int, int, int, int]],
        pt: Tuple[float, float],
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """
        Find mob bbox whose center is closest to given MAIN-local point.

        Returns (bbox, distance_pixels). If no mobs, returns (None, float("inf")).
        """
        if not mobs_main:
            return None, float("inf")

        px, py = pt
        best_box: Optional[Tuple[int, int, int, int]] = None
        best_d2: Optional[float] = None

        for box in mobs_main:
            if not isinstance(box, (tuple, list)) or len(box) < 4:
                continue
            x, y, bw, bh = box[:4]
            cx = x + bw * 0.5
            cy = y + bh * 0.5
            dx = cx - px
            dy = cy - py
            d2 = dx * dx + dy * dy
            if best_box is None or best_d2 is None or d2 < best_d2:
                best_box = (x, y, bw, bh)
                best_d2 = d2

        if best_box is None or best_d2 is None:
            return None, float("inf")

        return best_box, best_d2 ** 0.5

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
        targets_main_all = detect_main_targets_main(frame_main, self.cfg, crates)
        enemies_main: List[MainEnemy] = []
        for t in targets_main_all:
            if t.kind != MainTargetKind.ENEMY:
                continue
            enemies_main.append(
                MainEnemy(
                    bbox_name=t.bbox_name,
                    bbox_ship=t.bbox_ship,
                    text=t.text,
                )
            )

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
            "enemies_main": enemies_main,
            "crates": crates,
            "hp_pct": hp_pct,
            "shield_pct": shield_pct,
            "safety": safety_cfg,
            "targets_main": targets_main_all,
            "targets_mm": enemies_mm,
        }

        self.logger.debug(
            "Sense: enemies_mm=%d crates=%d mobs_main=%d enemies_main=%d me=%s hp=%d shield=%d",
            len(enemies_mm),
            len(crates),
            len(mobs_main),
            len(enemies_main),
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
        frame_main = obs["frame_main"]
        enemies_mm: List[Tuple[int, int]] = obs["enemies_mm"]
        player_mm = obs["me_mm"]
        mobs_main: List[Tuple[int, int, int, int]] = obs["mobs_main"]
        enemies_main: List[MainEnemy] = obs.get("enemies_main", [])
        targets_main = obs.get("targets_main", enemies_main)
        targets_mm = obs.get("targets_mm", enemies_mm)

        has_main_targets = bool(enemies_main or crates)

        # Reset target if there are no enemies
        if not enemies_mm:
            if self.current_target_mm is not None or self.current_stage is not None:
                self.logger.debug("FARMING: no enemies on minimap, reset target")
            self.current_target_mm = None
            self.current_stage = None
            self.fire_started_for_target = False
            self.current_target_main = None
            self.engage_click_attempts = 0

        # Track how long MAIN has been empty to decide when to use minimap.
        if has_main_targets or (
            self.current_stage == "engage" and self.current_target_main is not None
        ):
            self.farming_no_main_ticks = 0
        else:
            self.farming_no_main_ticks += 1

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
                self.fire_started_for_target = False
                self.logger.info("FARMING: picked target_mm=%s", target)

        # Decide whether we should use minimap for search/approach this tick.
        use_minimap = (
            not has_main_targets
            and self.farming_no_main_ticks
            >= max(self.farming_main_focus_timeout_ticks, 0)
        )

        # If we already see mobs on MAIN while in approach stage,
        # switch to ENGAGE and stop further minimap clicks.
        if self.current_stage == "approach" and enemies_main:
            self.logger.info("FARMING: mobs visible on MAIN, switching to ENGAGE")
            self.current_stage = "engage"

        # Approach stage: fly towards the selected enemy using minimap only.
        if (
            use_minimap
            and self.current_stage == "approach"
            and self.current_target_mm is not None
        ):
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

                # For now we use a single blocking approach helper; no attack here.
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

        # Engage stage: lock and shoot using main-screen enemies only (no minimap clicks).
        if self.current_stage == "engage" and self.current_target_mm is not None:
            enemies_main: List[MainEnemy] = obs.get("enemies_main", [])
            if not enemies_main:
                # Target died or left the screen (no name labels).
                self.logger.info("FARMING: target lost on MAIN (no enemies), resetting target")
                self.current_target_mm = None
                self.current_stage = None
                self.fire_started_for_target = False
                self.current_target_main = None
                self.engage_click_attempts = 0
                return

            h, w = frame_main.shape[:2]
            cx_screen = w // 2
            cy_screen = h // 2

            best_enemy: Optional[MainEnemy] = None
            best_d2: Optional[float] = None
            for enemy in enemies_main:
                x1, y1, x2, y2 = enemy.bbox_ship
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                dx = cx - cx_screen
                dy = cy - cy_screen
                d2 = dx * dx + dy * dy
                if best_enemy is None or best_d2 is None or d2 < best_d2:
                    best_enemy = enemy
                    best_d2 = d2

            if best_enemy is None:
                self.logger.debug("FARMING: no suitable enemies_main to engage")
                return

            now = time.monotonic()
            cooldown_s = max(self.farm_world_click_cooldown_ms, 0) / 1000.0
            if now - self._last_world_click_ts < cooldown_s:
                # Respect minimal interval between click bursts.
                return

            # First ENGAGE burst for this target: pick click point and fire once.
            if self.current_target_main is None:
                # Convert ship bbox (x1, y1, x2, y2) to (x, y, w, h) for click helper.
                sx1, sy1, sx2, sy2 = best_enemy.bbox_ship
                ship_box_xywh = (
                    int(sx1),
                    int(sy1),
                    int(max(1.0, sx2 - sx1)),
                    int(max(1.0, sy2 - sy1)),
                )
                click_px, click_py = self._resolve_enemy_click_point(ship_box_xywh)
                world_x, world_y = screen_from_main_vision_point(
                    (click_px, click_py),
                    self.cfg.roi_main(),
                )

                double_left_click_world((world_x, world_y))
                self._last_world_click_ts = now
                self.current_target_main = (click_px, click_py)
                self.engage_click_attempts = 1

                self.logger.info(
                    "FARMING: ENGAGE initial burst at world=(%d,%d) attempts=%d",
                    int(world_x),
                    int(world_y),
                    self.engage_click_attempts,
                )

                # Fire weapon based on map; call only once per target.
                if not self.fire_started_for_target and now - self._last_fire_ts >= 0.25:
                    fire_weapon_for_map(map_name=None)
                    self._last_fire_ts = now
                    self.fire_started_for_target = True
                    self.logger.info(
                        "FARMING: fire started once for current target"
                    )
                return

            # Subsequent ENGAGE ticks: check if we are still close to some enemy.
            ship_boxes_xywh: List[Tuple[int, int, int, int]] = []
            for enemy in enemies_main:
                x1, y1, x2, y2 = enemy.bbox_ship
                ship_boxes_xywh.append(
                    (
                        int(x1),
                        int(y1),
                        int(max(1.0, x2 - x1)),
                        int(max(1.0, y2 - y1)),
                    )
                )

            nearest_box, dist_px = self._nearest_mob_to_point(
                ship_boxes_xywh, self.current_target_main
            )
            if nearest_box is None:
                # Nobody near the last click point: treat as lost.
                self.logger.info(
                    "FARMING: no mobs near ENGAGE target point, resetting target"
                )
                self.current_target_mm = None
                self.current_stage = None
                self.current_target_main = None
                self.fire_started_for_target = False
                self.engage_click_attempts = 0
                return

            if dist_px <= self.farm_main_target_keep_radius_px:
                # Still reasonably close to an enemy: avoid extra bursts.
                return

            if self.engage_click_attempts >= self.max_engage_click_attempts:
                self.logger.info(
                    "FARMING: max ENGAGE click attempts reached (%d), resetting target",
                    self.max_engage_click_attempts,
                )
                self.current_target_mm = None
                self.current_stage = None
                self.current_target_main = None
                self.fire_started_for_target = False
                self.engage_click_attempts = 0
                return

            # Retry burst on updated bbox.
            click_px, click_py = self._resolve_enemy_click_point(nearest_box)
            world_x, world_y = screen_from_main_vision_point(
                (click_px, click_py),
                self.cfg.roi_main(),
            )

            double_left_click_world((world_x, world_y))
            self._last_world_click_ts = now
            self.current_target_main = (click_px, click_py)
            self.engage_click_attempts += 1

            self.logger.info(
                "FARMING: ENGAGE retry burst at world=(%d,%d) attempts=%d dist_px=%.1f",
                int(world_x),
                int(world_y),
                self.engage_click_attempts,
                dist_px,
            )
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
