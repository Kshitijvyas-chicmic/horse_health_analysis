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
