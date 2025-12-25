import math


def vector(p1, p2):
    """Return vector from p1 to p2."""
    return (p2[0] - p1[0], p2[1] - p1[1])


def magnitude(v):
    """Return magnitude (length) of vector."""
    return math.sqrt(v[0] ** 2 + v[1] ** 2)


def normalize(v):
    """Return unit vector."""
    mag = magnitude(v)
    if mag == 0:
        raise ValueError("Cannot normalize zero-length vector")
    return (v[0] / mag, v[1] / mag)


def pastern_vector(points):
    """
    P1 -> P2
    """
    return vector(points["P1"], points["P2"])


def hoof_wall_vector(points):
    """
    P3 -> P4
    """
    return vector(points["P3"], points["P4"])

def pastern_vector(points):
    v = vector(points["P1"], points["P2"])
    if magnitude(v) == 0:
        raise ValueError("Zero-length pastern vector")
    return v
