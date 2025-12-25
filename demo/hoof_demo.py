#from mmpose.apis import init_model, inference_top_down_pose_model, vis_pose_result
import cv2
from mmpose.mmpose.apis import init_model, inference_top_down_pose_model, vis_pose_result

pose_config = 'custom_configs/rtmpose_hoof_4kp.py'
pose_checkpoint = 'work_dirs/rtmpose_hoof_4kp/epoch_10.pth'
device = 'cuda:0'

# Load model
model = init_model(pose_config, pose_checkpoint, device=device)

# Image path
image_path = 'data/images/val/hoof_95.jpg'
image = cv2.imread(image_path)

# Full-image bbox [x1, y1, x2, y2, score]
bbox = [0, 0, image.shape[1], image.shape[0], 1.0]

# Run inference
pose_results, _ = inference_top_down_pose_model(model, image, [bbox], format='xyxy')


# Visualize & save
out_file = 'outputs/hoof_95_result.jpg'
vis_pose_result(model, image, pose_results, radius=5, thickness=2, out_file=out_file)

print(f'Result saved at {out_file}')
