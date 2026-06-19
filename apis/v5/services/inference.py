import base64
import httpx
import logging
import threading
from apis.logic import HPAPredictor

_frontal_mmpose = None
_frontal_mmpose_lock = threading.Lock()

def get_frontal_mmpose():
    global _frontal_mmpose
    with _frontal_mmpose_lock:
        if _frontal_mmpose is None:
            try:
                import torch
                from mmpose.apis import MMPoseInferencer
                device_id = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                logging.info(f"Loading Frontal MMPose on {device_id}...")
                _frontal_mmpose = MMPoseInferencer(pose2d='animal', device=device_id)
                logging.info("Frontal MMPose ready!")
            except Exception as e:
                logging.warning(f"Failed to load Frontal MMPose (Fallback geometry will be used): {e}")
                # Set to False to prevent retrying load on every request if it completely fails
                _frontal_mmpose = False
                
    return _frontal_mmpose if _frontal_mmpose is not False else None

async def get_image_bytes(image_input: str) -> bytes:
    """Fetches image bytes from either a URL or a Base64 string."""
    if image_input.startswith("http"):
        async with httpx.AsyncClient() as client:
            resp = await client.get(image_input, timeout=10.0)
            resp.raise_for_status()
            return resp.content
    if "," in image_input:
        image_input = image_input.split(",")[1]
    return base64.b64decode(image_input)


def run_leg_inference(predictor: HPAPredictor, image_bytes: bytes) -> dict:
    """
    MMPose inference for V4. Images are pre-cutout on mobile; never use rembg.
    """
    return predictor.predict(image_bytes, remove_bg=False)


def process_frontal_leg_symmetry(image_bytes_original: bytes, image_bytes_processed: bytes) -> str:
    """
    Runs leg_symmetry_v4 logic on paired frontal images and returns an uploaded S3 URL.
    """
    import tempfile
    import os
    from pathlib import Path
    from leg_symmetry_v4 import process_image
    from apis.v5.services.upload import upload_image_to_s3
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_original = Path(tmp_dir) / "original.jpg"
        with open(tmp_original, "wb") as f:
            f.write(image_bytes_original)
            
        tmp_processed = Path(tmp_dir) / "processed.png"
        with open(tmp_processed, "wb") as f:
            f.write(image_bytes_processed)
            
        max_retries = 3
        inferencer = get_frontal_mmpose()
        
        for attempt in range(max_retries):
            try:
                # v4's process_image() saves to a timestamped results/ subdir
                # and returns the full Path to the analyzed file (or None on failure).
                output_path = process_image(
                    str(tmp_original), str(tmp_processed),
                    do_debug=False, inferencer=inferencer
                )
                
                if output_path and Path(output_path).exists():
                    with open(output_path, "rb") as f:
                        analyzed_bytes = f.read()
                    
                    url = upload_image_to_s3(analyzed_bytes, file_extension="jpg")
                    if url:
                        return url
                    else:
                        logging.warning(f"Attempt {attempt + 1}: S3 upload failed to return URL.")
                else:
                    logging.warning(f"Attempt {attempt + 1}: Frontal analysis failed to generate output image.")
            except Exception as e:
                logging.error(f"Attempt {attempt + 1}: Error during frontal symmetry analysis: {e}", exc_info=True)
                
        logging.error("All 3 attempts to process the frontal image failed. Gracefully falling back to the original image.")
        # Fallback: Upload the original unanalyzed image so the user doesn't see a broken gray box
        fallback_url = upload_image_to_s3(image_bytes_original, file_extension="jpg")
        return fallback_url if fallback_url else ""
