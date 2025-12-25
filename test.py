from src.cvat.parser import load_cvat_keypoints
from src.geomatry.vectors import (
    pastern_vector,
    hoof_wall_vector,
    normalize
)

data = load_cvat_keypoints(
    "data/annotations/cvat/person_keypoints_default.json"
)

for img, points in data.items():
    vp = pastern_vector(points)
    vh = hoof_wall_vector(points)

    vp_n = normalize(vp)
    vh_n = normalize(vh)

    print(img)
    print("  Pastern vector:", vp)
    print("  Hoof vector   :", vh)
    print("  Pastern unit :", vp_n)
    print("  Hoof unit    :", vh_n)
    break

from src.geomatry.angles import angle_between, angle_from_vertical, anatomical_angle

for img, points in data.items():
    vp = pastern_vector(points)
    vh = hoof_wall_vector(points)

    pastern_angle = angle_from_vertical(vp)
    hoof_angle = angle_from_vertical(vh)
    diff = pastern_angle - hoof_angle

    print(img)
    print(f"  Pastern angle (deg): {pastern_angle:.2f}")
    print(f"  Hoof angle    (deg): {hoof_angle:.2f}")
    print(f"  Difference    (deg): {diff:.2f}")
    print(f"  Pastern anatomical: {anatomical_angle(pastern_angle):.2f}")
    print(f"  Hoof anatomical   : {anatomical_angle(hoof_angle):.2f}")

    break
