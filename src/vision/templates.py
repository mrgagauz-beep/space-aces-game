"""
Template loading and matching helpers.

This module loads sprite / icon templates from assets/templates
and provides thin wrappers around cv2.matchTemplate.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import logging

from utils.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


def load_templates(assets_dir: str | Path = "assets/templates") -> Dict[str, np.ndarray]:
    """
    Load all image templates from the given directory.

    Supported extensions: .png, .jpg, .jpeg.
    """
    assets_path = Path(assets_dir)
    templates: Dict[str, np.ndarray] = {}

    if not assets_path.exists():
        logger.warning("Templates directory does not exist: %s", assets_path)
        return templates

    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for path in assets_path.glob(ext):
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                logger.warning("Failed to load template: %s", path)
                continue
            name = path.stem
            templates[name] = img
            logger.info("Loaded template: %s", path)

    logger.info("Total templates loaded: %d", len(templates))
    return templates


def match_template(
    image: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.8,
) -> List[Tuple[int, int, int, int]]:
    """
    Run template matching and return bounding boxes where score >= threshold.

    Returns list of (x, y, w, h) in the image coordinate space.
    """
    if image is None or template is None:
        return []

    # Convert to BGR if template has alpha channel
    if template.shape[-1] == 4:
        template = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(img_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    h, w = tpl_gray.shape[:2]
    boxes: List[Tuple[int, int, int, int]] = []
    for pt in zip(*loc[::-1]):
        x, y = pt
        boxes.append((x, y, w, h))

    return boxes


if __name__ == "__main__":
    from config.config import Config
    from vision.capture import grab_main

    cfg = Config()
    main = grab_main(cfg)
    templates = load_templates()

    total_matches = 0
    for name, tpl in templates.items():
        boxes = match_template(main, tpl, threshold=0.85)
        logger.info("Template '%s': %d matches", name, len(boxes))
        total_matches += len(boxes)

    logger.info("Sanity templates: total matches=%d", total_matches)
