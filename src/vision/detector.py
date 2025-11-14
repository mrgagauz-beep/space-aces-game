import cv2
import numpy as np


def _get_config_value(config, key):
    """
    Extract value from config which can be a dict or an object.
    """
    if isinstance(config, dict):
        return config[key]
    return getattr(config, key)


def _find_objects_by_color(frame, lower_hsv, upper_hsv, min_area=50):
    """
    Generic helper to find object centers in frame by HSV color range.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        moments = cv2.moments(cnt)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2

        centers.append((cx, cy))

    return centers


def find_enemies(frame, config):
    """
    Find enemies on the frame using HSV color thresholds from config.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR image from OpenCV.
    config : dict or object
        Must provide enemy_color_lower_hsv and enemy_color_upper_hsv.

    Returns
    -------
    list[tuple[int, int]]
        List of (x, y) centers of detected enemies.
    """
    lower_hsv = _get_config_value(config, "enemy_color_lower_hsv")
    upper_hsv = _get_config_value(config, "enemy_color_upper_hsv")

    return _find_objects_by_color(frame, lower_hsv, upper_hsv)


def find_boxes(frame, config):
    """
    Find bonus boxes on the frame using HSV color thresholds from config.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR image from OpenCV.
    config : dict or object
        Must provide box_color_lower_hsv and box_color_upper_hsv.

    Returns
    -------
    list[tuple[int, int]]
        List of (x, y) centers of detected boxes.
    """
    lower_hsv = _get_config_value(config, "box_color_lower_hsv")
    upper_hsv = _get_config_value(config, "box_color_upper_hsv")

    return _find_objects_by_color(frame, lower_hsv, upper_hsv)

