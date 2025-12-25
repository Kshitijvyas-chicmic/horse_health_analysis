#class to fix bbox from keypoints, increase bbox size and remove noise and save it to a new json file

import json
import sys
from pathlib import Path

# TOP_MARGIN = 0.15
# BOTTOM_MARGIN = 0.10
# SIDE_MARGIN = 0.25

TOP_MARGIN = 0.08
BOTTOM_MARGIN = 0.05
SIDE_MARGIN = 0.12

def fix_coco_bboxes(coco_path, out_path=None):
    with open(coco_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    fixed = 0
    skipped = 0

    for ann in coco["annotations"]:
        kpts = ann.get("keypoints", [])
        if not kpts or ann.get("num_keypoints", 0) == 0:
            skipped += 1
            continue

        xs, ys = [], []
        for i in range(0, len(kpts), 3):
            x, y, v = kpts[i:i+3]
            if v > 0:
                xs.append(x)
                ys.append(y)

        if len(xs) < 2:
            skipped += 1
            continue

        #img = images[ann["image_id"]]
        img = images.get(ann["image_id"])
        if img is None:
            skipped += 1
            continue

        img_w, img_h = img["width"], img["height"]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        H = y_max - y_min
        if H <= 1:
            skipped += 1
            continue

        # Expand bbox
        # x_min -= SIDE_MARGIN * H
        # x_max += SIDE_MARGIN * H
        # y_min -= TOP_MARGIN * H
        # y_max += BOTTOM_MARGIN * H

        W = x_max - x_min
        H = y_max - y_min

        x_min -= SIDE_MARGIN * W
        x_max += SIDE_MARGIN * W
        y_min -= TOP_MARGIN * H
        y_max += BOTTOM_MARGIN * H

        # Clamp to image
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_w - 1, x_max)
        y_max = min(img_h - 1, y_max)

        w = x_max - x_min
        h = y_max - y_min
        
        if h < 0.25 * img_h:
            skipped += 1
            continue

        if w <= 1 or h <= 1:
            skipped += 1
            continue

        ann["bbox"] = [
            round(x_min, 2),
            round(y_min, 2),
            round(w, 2),
            round(h, 2)
        ]
        ann["area"] = round(w * h, 2)

        fixed += 1

    print(f"✔ Fixed bboxes: {fixed}")
    print(f"⚠ Skipped: {skipped}")

    if out_path is None:
        out_path = Path(coco_path).with_name(
            Path(coco_path).stem + "_fixed.json"
        )

    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"💾 Saved to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_bbox_from_keypoints.py input.json [output.json]")
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    fix_coco_bboxes(inp, out)
