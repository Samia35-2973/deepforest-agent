import base64
import io
import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def load_image_as_np_array(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from a file path as a NumPy array.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        RGB image as numpy array, or None if not found
        
    Raises:
        FileNotFoundError: If image file is not found at any expected path
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found at any expected path: {image_path}"
        )

    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)


def encode_image_to_base64_url(image_array: np.ndarray, format: str = 'PNG', 
                              quality: int = 80) -> Optional[str]:
    """
    Encode a NumPy image array to a base64 data URL.
    
    Args:
        image_array: Image as numpy array
        format: Output format ('PNG' or 'JPEG')
        quality: JPEG quality (only used for JPEG format)
        
    Returns:
        Base64 encoded data URL string, or None if encoding fails
    """
    if image_array is None:
        return None
    
    try:
        pil_image = Image.fromarray(image_array)
        if pil_image.mode == 'RGBA':
            background = Image.new("RGB", pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()[3])
            pil_image = background
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        byte_arr = io.BytesIO()
        if format.lower() == 'jpeg':
            pil_image.save(byte_arr, format='JPEG', quality=quality)
        elif format.lower() == 'png':
            pil_image.save(byte_arr, format='PNG')
        else:
            raise ValueError(f"Unsupported format: {format}. Choose 'jpeg' or 'png'.")

        encoded_string = base64.b64encode(byte_arr.getvalue()).decode('utf-8')
        return f"data:image/{format.lower()};base64,{encoded_string}"
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None


def convert_rgb_to_bgr(image_array: np.ndarray) -> np.ndarray:
    """
    Convert an RGB NumPy image array to BGR format for OpenCV compatibility.
    
    Args:
        image_array: RGB image as numpy array
        
    Returns:
        BGR image as numpy array
    """
    if (image_array.ndim == 3 and image_array.shape[2] == 3 and 
        image_array.dtype == np.uint8):
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    return image_array


def convert_bgr_to_rgb(image_array: np.ndarray) -> np.ndarray:
    """
    Convert a BGR NumPy image array to RGB format.
    
    Args:
        image_array: BGR image as numpy array
        
    Returns:
        RGB image as numpy array
    """
    if (image_array.ndim == 3 and image_array.shape[2] == 3 and 
        image_array.dtype == np.uint8):
        return cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    return image_array