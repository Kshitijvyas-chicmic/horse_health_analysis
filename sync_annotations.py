import json
import os

def sync_annotations(source_path, target_path):
    print(f"Starting sync from {source_path} to {target_path}...")
    
    if not os.path.exists(source_path):
        print(f"Error: Source file {source_path} does not exist.")
        return
    if not os.path.exists(target_path):
        print(f"Error: Target file {target_path} does not exist.")
        return

    with open(source_path, 'r') as f:
        source_data = json.load(f)
    with open(target_path, 'r') as f:
        target_data = json.load(f)

    # Map image_id to annotation in source
    # We confirmed each image has at most one annotation in refined data
    refined_annos = {ann['image_id']: ann for ann in source_data['annotations']}
    
    # First, collect all valid image IDs in this target file
    valid_image_ids = {img['id'] for img in target_data['images']}
    
    # Filter annotations: Keep only those with a valid image_id in THIS file
    original_count = len(target_data['annotations'])
    target_data['annotations'] = [
        ann for ann in target_data['annotations'] 
        if ann['image_id'] in valid_image_ids
    ]
    removed_count = original_count - len(target_data['annotations'])
    
    match_count = 0
    # Now sync refined data for the remaining valid annotations
    for ann in target_data['annotations']:
        img_id = ann['image_id']
        if img_id in refined_annos:
            source_ann = refined_annos[img_id]
            ann['area'] = source_ann.get('area', ann['area'])
            ann['bbox'] = source_ann.get('bbox', ann['bbox'])
            ann['keypoints'] = source_ann.get('keypoints', ann['keypoints'])
            ann['num_keypoints'] = source_ann.get('num_keypoints', ann['num_keypoints'])
            ann['segmentation'] = source_ann.get('segmentation', ann['segmentation'])
            if 'attributes' in source_ann:
                ann['attributes'] = source_ann['attributes']
            match_count += 1

    # Save output
    with open(target_path, 'w') as f:
        json.dump(target_data, f, indent=4)
    
    print(f"Finished {target_path}:")
    print(f"  - Removed {removed_count} orphan/duplicate annotations (not in images section).")
    print(f"  - Synchronized {match_count} annotations from refined source.")

if __name__ == "__main__":
    REFINED_PATH = "/home/chetan/AI_First/horse_health_analysis/horse_health_analysis/data/annotations/cvat/person_keypoints_default_refined.json"
    VAL_PATH = "/home/chetan/AI_First/horse_health_analysis/horse_health_analysis/data/annotations/val.json"
    TRAIN_PATH = "/home/chetan/AI_First/horse_health_analysis/horse_health_analysis/data/annotations/train.json"
    
    sync_annotations(REFINED_PATH, VAL_PATH)
    sync_annotations(REFINED_PATH, TRAIN_PATH)
