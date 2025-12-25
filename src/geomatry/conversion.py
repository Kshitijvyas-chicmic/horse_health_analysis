def vertical_to_ground_angle(vertical_angle):
    """
    Convert anatomical angle measured from vertical
    to ground-based angle (device reference).

    Example:
    vertical = 33°  -> ground = 57°
    """
    return 90.0 - vertical_angle
