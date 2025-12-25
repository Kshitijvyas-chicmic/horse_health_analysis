import json
from pathlib import Path


KEYPOINT_NAMES = ["P1", "P2", "P3", "P4"]


def load_cvat_keypoints(json_path):
    """
    Load CVAT COCO keypoints JSON and return a clean dict:
    
    {
        image_name: {
            "P1": (x, y),
            "P2": (x, y),
            "P3": (x, y),
            "P4": (x, y)
        }
    }
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    # Build image_id -> image_name mapping
    image_id_to_name = {
        img["id"]: img["file_name"]
        for img in data.get("images", [])
    }

    results = {}

    for ann in data.get("annotations", []):
        image_id = ann["image_id"]
        keypoints = ann.get("keypoints", [])

        if len(keypoints) != 4 * 3:
            raise ValueError(
                f"Expected 4 keypoints (12 values), got {len(keypoints)} "
                f"for image_id {image_id}"
            )

        image_name = image_id_to_name.get(image_id)
        if image_name is None:
            continue

        points = {}
        for i, kp_name in enumerate(KEYPOINT_NAMES):
            x = keypoints[i * 3]
            y = keypoints[i * 3 + 1]
            points[kp_name] = (float(x), float(y))

        results[image_name] = points

    return results
