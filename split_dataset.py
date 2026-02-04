import json
import random
import os
import copy
import argparse

def split_dataset(all_data_path, train_path, val_path, split_ratio=0.85, seed=42):
    print(f"Loading {all_data_path}...")
    if not os.path.exists(all_data_path):
        print(f"Error: Source file {all_data_path} does not exist.")
        return

    with open(all_data_path, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']
    categories = data.get('categories', [])
    info = data.get('info', {})

    # Map image_id to annotations
    img_to_anns = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    # Shuffle images
    print(f"Using random seed: {seed}")
    random.seed(seed)
    random.shuffle(images)
    
    # Calculate split index
    total_images = len(images)
    split_idx = int(total_images * split_ratio)
    
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    def build_dataset(img_list):
        dataset = {
            "info": info,
            "categories": categories,
            "images": img_list,
            "annotations": []
        }
        for img in img_list:
            img_id = img['id']
            if img_id in img_to_anns:
                dataset['annotations'].extend(copy.deepcopy(img_to_anns[img_id]))
        return dataset

    print(f"Total Images: {total_images}")
    print(f"Training Set: {len(train_images)} images ({len(train_images)/total_images:.1%})")
    print(f"Validation Set: {len(val_images)} images ({len(val_images)/total_images:.1%})")

    train_data = build_dataset(train_images)
    val_data = build_dataset(val_images)

    # Safety Check
    train_ids = set(img['id'] for img in train_data['images'])
    val_ids = set(img['id'] for img in val_data['images'])
    overlap = train_ids.intersection(val_ids)
    
    if len(overlap) > 0:
        print(f"CRITICAL ERROR: Found {len(overlap)} images in both sets! Aborting.")
        return

    print("-" * 30)
    print(f"Saving to {train_path}...")
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=4)
        
    print(f"Saving to {val_path}...")
    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=4)

    print("Done. Split is clean and secured.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split COCO dataset into Train/Val")
    parser.add_argument('--input', type=str, required=True, help="Path to master JSON file")
    parser.add_argument('--train_out', type=str, default="train.json", help="Output path for training JSON")
    parser.add_argument('--val_out', type=str, default="val.json", help="Output path for validation JSON")
    parser.add_argument('--ratio', type=float, default=0.85, help="Ratio of training images (default 0.85)")
    args = parser.parse_args()
    
    split_dataset(args.input, args.train_out, args.val_out, args.ratio)
