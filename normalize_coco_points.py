import json
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python fix_coco_area.py <path_to_coco_json>")
    sys.exit(1)

json_path = Path(sys.argv[1])

if not json_path.exists():
    print(f"❌ File not found: {json_path}")
    sys.exit(1)

with open(json_path, "r") as f:
    data = json.load(f)

fixed = 0

for ann in data.get("annotations", []):
    if "area" not in ann:
        _, _, w, h = ann["bbox"]
        ann["area"] = float(w * h)
        fixed += 1
    if "iscrowd" not in ann:
        ann["iscrowd"] = 0

with open(json_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"✅ Updated {fixed} annotations with area + iscrowd")
print(f"📄 File saved: {json_path.resolve()}")
