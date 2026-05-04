import cv2
import numpy as np
import logging

try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logging.warning("rembg not installed. Background removal will be skipped.")

def remove_background(image: np.ndarray) -> np.ndarray:
    """
    Removes the background from an image using rembg.
    
    Args:
        image (np.ndarray): Input image in BGR format (OpenCV default).
        
    Returns:
        np.ndarray: Image with background removed (set to black), in BGR format.
    """
    if not REMBG_AVAILABLE:
        return image

    try:
        # rembg works best with RGB or RGBA
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # remove() returns an RGBA image (with transparency)
        output_rgba = remove(image_rgb)
        
        # Convert to numpy array if it's not already
        output_rgba = np.array(output_rgba)
        
        # Separate the alpha channel
        alpha = output_rgba[:, :, 3]
        
        # Create a black background image
        black_bg = np.zeros_like(output_rgba[:, :, :3])
        
        # Mask the original RGB image using the alpha channel
        # We want the subject on a black background for the model
        mask = alpha > 0
        black_bg[mask] = output_rgba[:, :, :3][mask]
        
        # Convert back to BGR for OpenCV compatibility
        output_bgr = cv2.cvtColor(black_bg, cv2.COLOR_RGB2BGR)
        
        return output_bgr
        
    except Exception as e:
        logging.error(f"Error during background removal: {e}")
        return image
