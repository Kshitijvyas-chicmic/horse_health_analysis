import json
import os
import sys
import argparse

def validate_integrity(train_json, val_json):
    print(f"🔍 Validating integrity for {train_json} and {val_json}...")
    
    if not os.path.exists(train_json) or not os.path.exists(val_json):
        print("❌ Error: One or both JSON files are missing.")
        return False

    with open(train_json, 'r') as f:
        train_data = json.load(f)
    with open(val_json, 'r') as f:
        val_data = json.load(f)

    train_ids = set(img['id'] for img in train_data['images'])
    val_ids = set(img['id'] for img in val_data['images'])

    # 1. Leakage Check
    overlap = train_ids.intersection(val_ids)
    if overlap:
        print(f"❌ CRITICAL: Data Leakage detected! {len(overlap)} images overlap between sets.")
        return False
    print("✅ No data leakage detected.")

    # 2. Keypoint sanity check
    for name, data in [("Train", train_data), ("Val", val_data)]:
        missing_kpts = 0
        for ann in data['annotations']:
            if ann.get('num_keypoints', 0) == 0:
                missing_kpts += 1
        if missing_kpts > 0:
            print(f"⚠️ Warning: {name} set has {missing_kpts} annotations with zero keypoints.")
    
    # 3. Scale/Count Check
    if len(train_ids) < 10 or len(val_ids) < 2:
        print("❌ Error: Dataset size is too small for meaningful training.")
        return False
        
    print("✅ Integrity check passed.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate overlap and sanity of split datasets")
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--val', type=str, required=True)
    args = parser.parse_args()

    if validate_integrity(args.train, args.val):
        sys.exit(0)
    else:
        sys.exit(1)
