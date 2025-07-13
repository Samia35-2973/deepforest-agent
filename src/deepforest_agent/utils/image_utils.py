import base64
import io
import os
from typing import Dict, Any, List, Literal, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import tempfile
import rasterio

from deepforest_agent.conf.config import Config

def load_image_as_np_array(image_path: str) -> np.ndarray:
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


def load_pil_image_from_path(image_path: str) -> Optional[Image.Image]:
    """
    Load PIL Image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image object, or None if loading fails
        
    Raises:
        FileNotFoundError: If image file is not found
        Exception: If image cannot be loaded or converted
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        print(f"Error loading PIL image from {image_path}: {e}")
        return None


def create_temp_image_file(image_array: np.ndarray, suffix: str = ".png") -> str:
    """
    Create a temporary image file from numpy array.
    
    Args:
        image_array: Image as numpy array
        suffix: File extension (default: ".png")
        
    Returns:
        Path to temporary file
        
    Raises:
        Exception: If temporary file creation fails
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            temp_file_path = tmp_file.name
            
        pil_image = Image.fromarray(image_array)
        pil_image.save(temp_file_path, format='PNG')
        
        print(f"Created temporary image file: {temp_file_path}")
        return temp_file_path
        
    except Exception as e:
        print(f"Error creating temporary image file: {e}")
        raise e


def cleanup_temp_file(file_path: str) -> bool:
    """
    Clean up temporary file.
    
    Args:
        file_path: Path to file to remove
        
    Returns:
        True if successful, False otherwise
    """
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"Cleaned up temporary file: {file_path}")
            return True
        except OSError as e:
            print(f"Error cleaning up temporary file {file_path}: {e}")
            return False
    return False


def validate_image_path(image_path: str) -> bool:
    """
    Validate if image path exists and is a valid image file.
    
    Args:
        image_path: Path to validate
        
    Returns:
        True if valid image path, False otherwise
    """
    if not image_path or not os.path.exists(image_path):
        return False
    
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_image_info(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Get basic information about an image file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with image info or None if error
    """
    try:
        with Image.open(image_path) as img:
            return {
                "size": img.size,
                "mode": img.mode,
                "format": img.format,
                "file_size_bytes": os.path.getsize(image_path)
            }
    except Exception as e:
        print(f"Error getting image info for {image_path}: {e}")
        return None


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


def convert_pil_image_to_bytes(image: Image.Image) -> bytes:
    """
    Convert a PIL Image to bytes in PNG format.

    Args:
        image: PIL Image object

    Returns:
        Image bytes in PNG format
    """
    img_byte_arr = io.BytesIO()

    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()

    return img_bytes


def encode_pil_image_to_base64_url(image: Image.Image) -> str:
    """
    Encode a PIL Image directly to a base64 data URL.

    Args:
        image: PIL Image object

    Returns:
        Base64 encoded PNG data URL string
    """
    img_bytes = convert_pil_image_to_bytes(image)
    img_str = base64.b64encode(img_bytes).decode()
    data_url = f"data:image/png;base64,{img_str}"
    return data_url


def decode_base64_to_pil_image(base64_data: str) -> Image.Image:
    """
    Decode base64 data to a PIL Image.

    Handles both data URL format and raw base64 strings.

    Args:
        base64_data: Base64 encoded image data, either as data URL
                    (data:image/png;base64,iVBORw0...) or raw base64 string

    Returns:
        PIL Image object

    Raises:
        ValueError: If base64 data is invalid or cannot be decoded
    """
    try:
        if base64_data.startswith('data:image'):
            # Extract base64 part after the comma
            base64_string = base64_data.split(',')[1]
        else:
            # Raw base64 data
            base64_string = base64_data

        image_bytes = base64.b64decode(base64_string)
        pil_image = Image.open(io.BytesIO(image_bytes))

        return pil_image

    except Exception as e:
        raise ValueError(f"Failed to decode base64 data to PIL Image: {e}")


def decode_base64_url_to_np_array(image_url: str) -> Optional[np.ndarray]:
    """
    Decode a base64 data URL to a NumPy array.

    Args:
        image_url: Base64 data URL (data:image/png;base64,iVBORw0...)

    Returns:
        RGB image as numpy array, or None if decoding fails
    """
    if not image_url.startswith('data:image'):
        print(f"Invalid data URL format: {image_url[:50]}...")
        return None

    try:
        pil_image = decode_base64_to_pil_image(image_url)

        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        return np.array(pil_image)

    except ValueError as e:
        print(f"Error extracting image from data URL: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error processing image URL: {e}")
        return None


def convert_rgb_to_bgr(image_array: np.ndarray) -> np.ndarray:
    """
    Convert an RGB NumPy image array to BGR format.

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

def check_image_resolution_for_deepforest(image_path: str, max_resolution_cm: float = 10.0) -> Dict[str, Any]:
    """
    Resolution check for DeepForest suitability.
    
    For GeoTIFF files: Check if pixel resolution is <= 10cm
    For other formats: Allow processing with warning
    
    Args:
        image_path: Path to the image file
        max_resolution_cm: Maximum required resolution in cm/pixel (default: 10.0)
        
    Returns:
        Dict containing:
        - is_suitable: bool - Whether resolution is suitable for DeepForest
        - resolution_cm: float or None - Actual resolution in cm/pixel
        - resolution_info: str - Resolution info
        - is_georeferenced: bool - Whether image is a GeoTIFF
        - warning: str or None - Warning message if any
    """
    try:
        with rasterio.open(image_path) as src:
            if src.crs is None:
                return _non_geotiff_result(image_path, "No coordinate system found")
            if src.crs.is_geographic:
                return _non_geotiff_result(image_path, "Geographic coordinates detected")
            transform = src.transform
            if transform.is_identity:
                return _non_geotiff_result(image_path, "No spatial transformation found")
            
            # Calculate pixel size
            pixel_width = abs(transform.a)
            pixel_height = abs(transform.e)
            pixel_size = max(pixel_width, pixel_height)
            
            # Convert to centimeters based on CRS units
            crs_units = src.crs.to_dict().get('units', '').lower()
            
            if crs_units in ['m', 'metre', 'meter']:
                resolution_cm = pixel_size * 100
            elif 'foot' in crs_units or crs_units == 'ft':
                resolution_cm = pixel_size * 30.48
            else:
                return {
                    "is_suitable": True,
                    "resolution_cm": None,
                    "resolution_info": f"Unknown units '{crs_units}' - proceeding optimistically",
                    "is_georeferenced": True,
                    "warning": f"Cannot determine pixel size units: {crs_units}"
                }

            is_suitable = resolution_cm <= max_resolution_cm
            
            return {
                "is_suitable": is_suitable,
                "resolution_cm": resolution_cm,
                "resolution_info": f"{resolution_cm:.1f} cm/pixel ({'suitable' if is_suitable else 'insufficient'} for DeepForest)",
                "is_georeferenced": True,
                "warning": None if is_suitable else f"Resolution {resolution_cm:.1f} cm/pixel exceeds {max_resolution_cm} cm/pixel threshold"
            }
            
    except rasterio.RasterioIOError:
        return _non_geotiff_result(image_path, "Not a GeoTIFF file")
    except Exception as e:
        return _non_geotiff_result(image_path, f"Error reading file: {str(e)}")


def _non_geotiff_result(image_path: str, reason: str) -> Dict[str, Any]:
    """
    Helper function for non-GeoTIFF images to allow processing with warning.
    
    Args:
        image_path: Path to the image file
        reason: Reason why it's not treated as GeoTIFF
        
    Returns:
        Dict with suitable=True but warning about using GeoTIFF
    """
    file_ext = os.path.splitext(image_path)[1].lower()
    
    return {
        "is_suitable": True,
        "resolution_cm": None,
        "resolution_info": f"Non-geospatial image ({file_ext}) - proceeding without resolution check",
        "is_georeferenced": False,
        "warning": f"For optimal DeepForest results, use GeoTIFF images with ≤10 cm/pixel resolution. Current: {reason.lower()}"
    }

def determine_patch_size(image_file_path: str, image_dimensions: Optional[Tuple[int, int]] = None) -> int:
    """
    Determine patch size based on image file type and dimensions for OOM fallback strategy.
    
    Args:
        image_file_path: Path to the image file
        image_dimensions: Optional tuple of (width, height) if known
        
    Returns:
        int: Patch size optimized for image type and size
    """
    # Get image dimensions if not provided
    if image_dimensions is None:
        try:
            with Image.open(image_file_path) as img:
                width, height = img.size
        except Exception:
            return Config.DEEPFOREST_DEFAULTS["patch_size"]
    else:
        width, height = image_dimensions
    
    # Determine maximum dimension
    max_dimension = max(width, height)
    
    # For large dimensions, use larger patch sizes to handle OOM
    if max_dimension > 7500:
        return 2000
    else:
        return 1500

def get_image_dimensions_fast(image_path: str) -> Optional[Tuple[int, int]]:
    """
    Get image dimensions quickly without loading full image into memory.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (width, height) or None if cannot determine
    """
    try:
        # Try with PIL first
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        try:
            # Fallback to rasterio for GeoTIFF files
            with rasterio.open(image_path) as src:
                return (src.width, src.height)
        except Exception:
            return None