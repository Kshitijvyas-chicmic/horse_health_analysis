import math


def angle_between(v1, v2):
    """
    Returns angle in degrees between vectors v1 and v2
    """
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

    cos_theta = dot / (mag1 * mag2)
    cos_theta = max(-1.0, min(1.0, cos_theta))  # numerical safety

    return math.degrees(math.acos(cos_theta))


def angle_from_vertical(v):
    """
    Returns signed angle from vertical axis (positive = forward lean)
    Vertical reference = (0, -1) or (0, 1) depending on image orientation
    """

    vertical = (0, -1)  # image y increases downward
    dot = v[0]*vertical[0] + v[1]*vertical[1]
    det = v[0]*vertical[1] - v[1]*vertical[0]

    return math.degrees(math.atan2(det, dot))

def anatomical_angle(image_angle):
    """
    Convert image-based angle to anatomical angle (0–90 degrees).

    This removes directionality and camera-side effects.
    """
    angle = abs(image_angle) % 180
    if angle > 90:
        angle = 180 - angle
    return angle
