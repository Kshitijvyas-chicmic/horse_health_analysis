import cv2
import numpy as np
import logging
import os

try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logging.warning("rembg not installed. Background removal will be skipped.")

# Global session to avoid reloading model on every request
_REMBG_SESSION = None

def get_rembg_session():
    """
    Initializes a rembg session with CPU-optimized settings.
    """
    global _REMBG_SESSION
    if _REMBG_SESSION is None and REMBG_AVAILABLE:
        try:
            # We use 'u2netp' (U2-Net Portrait) which is a smaller, faster version of u2net.
            # It uses much less RAM and CPU while still being excellent for leg/hoof isolation.
            # Also limit onnxruntime to use only 1-2 threads to prevent CPU spikes.
            import onnxruntime as ort
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 2
            sess_opts.inter_op_num_threads = 2
            
            # Use the session for faster processing and limit CPU usage
            _REMBG_SESSION = new_session("u2netp")
            logging.info("✅ Rembg session initialized (model: u2netp)")
        except Exception as e:
            logging.error(f"Failed to initialize rembg session: {e}")
    return _REMBG_SESSION

def remove_background(image: np.ndarray) -> np.ndarray:
    """
    Removes the background from an image using rembg.
    """
    if not REMBG_AVAILABLE:
        return image

    session = get_rembg_session()
    if session is None:
        return image

    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use the session for faster processing
        output_rgba = remove(image_rgb, session=session)
        output_rgba = np.array(output_rgba)
        
        alpha = output_rgba[:, :, 3]
        black_bg = np.zeros_like(output_rgba[:, :, :3])
        mask = alpha > 0
        black_bg[mask] = output_rgba[:, :, :3][mask]
        
        return cv2.cvtColor(black_bg, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.error(f"Error during background removal: {e}")
        return image
