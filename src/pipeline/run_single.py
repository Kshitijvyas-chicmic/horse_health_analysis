from src.cvat.parser import load_cvat_keypoints
from src.geomatry.vectors import pastern_vector, hoof_wall_vector
from src.geomatry.angles import angle_from_vertical, anatomical_angle



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

        diff = pastern_anat - hoof_anat

        results.append({
            "image": image_name,
            "pastern_angle": round(pastern_anat, 2),
            "hoof_angle": round(hoof_anat, 2),
            "difference": round(diff, 2)
        })

    return results


if __name__ == "__main__":
    for r in run():
        print(r)
