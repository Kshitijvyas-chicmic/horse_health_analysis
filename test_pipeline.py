from src.cvat.parser import load_cvat_keypoints
from src.geomatry.vectors import pastern_vector, hoof_wall_vector
from src.geomatry.angles import angle_from_vertical, anatomical_angle
from src.geomatry.conversion import vertical_to_ground_angle
from src.config.constants import ANGLE_TOLERANCE_DEG


#class to calculate angles from cvat keypoints

ANNOTATION_FILE = "data/annotations/cvat/person_keypoints_default.json"


def run():
    data = load_cvat_keypoints(ANNOTATION_FILE)

    results = []

    for image_name, points in data.items():
        vp = pastern_vector(points)
        vh = hoof_wall_vector(points)

        pastern_img = angle_from_vertical(vp)
        hoof_img = angle_from_vertical(vh)

        pastern_anat = anatomical_angle(pastern_img)
        hoof_anat = anatomical_angle(hoof_img)

        #these are the ground based angles same as device reference
        pastern_ground = vertical_to_ground_angle(pastern_anat)
        hoof_ground = vertical_to_ground_angle(hoof_anat)


        diff = pastern_anat - hoof_anat

        results.append({
            "image": image_name,
            "pastern_angle": round(pastern_anat, 2),
            "hoof_angle": round(hoof_anat, 2),
            "difference": round(diff, 2),
            "pastern_ground": round(pastern_ground, 2),
            "pastern_min": round(pastern_ground - ANGLE_TOLERANCE_DEG, 2),
            "pastern_max": round(pastern_ground + ANGLE_TOLERANCE_DEG, 2),
            "hoof_ground": round(hoof_ground, 2),
            "hoof_min": round(hoof_ground - ANGLE_TOLERANCE_DEG, 2),
            "hoof_max": round(hoof_ground + ANGLE_TOLERANCE_DEG, 2),
            "difference_ground": round(pastern_ground - hoof_ground, 2)

        })

    return results


if __name__ == "__main__":
    for r in run():
        print(r)
