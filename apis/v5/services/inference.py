import base64
import httpx
from apis.logic import HPAPredictor


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
    Runs leg_symmetry_v3 logic on paired frontal images and returns an uploaded S3 URL.
    """
    import tempfile
    import os
    from pathlib import Path
    from leg_symmetry_v3 import process_image
    from apis.v5.services.upload import upload_image_to_s3
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_original = Path(tmp_dir) / "original.jpg"
        with open(tmp_original, "wb") as f:
            f.write(image_bytes_original)
            
        tmp_processed = Path(tmp_dir) / "processed.png"
        with open(tmp_processed, "wb") as f:
            f.write(image_bytes_processed)
            
        # process_image generates several outputs; we want the '_analyzed.jpg'
        try:
            process_image(str(tmp_original), str(tmp_processed), do_debug=False)
            output_path = Path(tmp_dir) / "original_analyzed.jpg"
            
            if output_path.exists():
                with open(output_path, "rb") as f:
                    analyzed_bytes = f.read()
                return upload_image_to_s3(analyzed_bytes, file_extension="jpg")
            else:
                print(f"❌ Frontal symmetry analysis failed to generate output.")
                return ""
        except Exception as e:
            print(f"❌ Error during frontal symmetry analysis: {e}")
            return ""
