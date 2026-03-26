import base64
import httpx
from apis.logic import HPAPredictor
from apis.yolo_predictor import YOLOPredictor

async def get_image_bytes(image_input: str) -> bytes:
    """
    Fetches image bytes from either a URL or a Base64 string.
    """
    if image_input.startswith("http"):
        async with httpx.AsyncClient() as client:
            resp = await client.get(image_input, timeout=10.0)
            resp.raise_for_status()
            return resp.content
    else:
        # Assume Base64
        # Strip data:image/jpeg;base64, prefix if present
        if "," in image_input:
            image_input = image_input.split(",")[1]
        return base64.b64decode(image_input)

def run_leg_inference(predictor: HPAPredictor, image_bytes: bytes) -> dict:
    """
    Runs MMPose inference on a single leg and returns raw results.
    """
    return predictor.predict(image_bytes)


def run_yolo_inference(yolo_predictor: YOLOPredictor, image_bytes: bytes) -> dict:
    """
    Runs YOLO Medium inference on a single leg and returns raw results.
    Same return-key contract as run_leg_inference:
      success, pastern_angle, hoof_angle, hpa_dev, model_confidence, error, image_base64
    """
    return yolo_predictor.predict(image_bytes)
