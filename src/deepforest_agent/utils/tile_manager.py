import numpy as np
from typing import Tuple, Dict, List, Any, Optional
from PIL import Image
import rasterio as rio
from rasterio.windows import Window
from deepforest import preprocess
try:
    import slidingwindow
    SLIDINGWINDOW_AVAILABLE = True
except ImportError:
    SLIDINGWINDOW_AVAILABLE = False
    print("Warning: slidingwindow not available, falling back to deepforest preprocess")

from deepforest_agent.conf.config import Config


def tile_image_for_analysis(
    image: Image.Image,
    patch_size: int = Config.DEEPFOREST_DEFAULTS["patch_size"],
    patch_overlap: float = Config.DEEPFOREST_DEFAULTS["patch_overlap"],
    image_file_path: Optional[str] = None,
) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
    """
    Tile am Image for visual analysis.
    
    Args:
        image (Image.Image): PIL Image to tile
        patch_size (int): Size of each tile in pixels (default: 400)
        patch_overlap (float): Overlap between tiles as fraction 0-1 (default: 0.05)
        image_file_path (Optional[str]): Path to raster file for memory-efficient dimension reading
        
    Returns:
        Tuple containing:
            - List[Image.Image]: List of PIL Image tiles
            - List[Dict[str, Any]]: List of tile metadata with coordinates
            
    Raises:
        ValueError: If patch_overlap > 1 or image is too small for patch_size
        Exception: If tiling process fails
    """
    try:
        # Use slidingwindow for all image types if available
        if SLIDINGWINDOW_AVAILABLE:
            height = width = None
            method = "unknown"
            
            if image_file_path:
                try:
                    # Get raster shape without keeping file open
                    with rio.open(image_file_path) as src:
                        height = src.shape[0]  
                        width = src.shape[1]
                        method = "slidingwindow_raster"
                    print(f"Using raster dimensions: {width}x{height} from file path")
                except Exception as raster_error:
                    print(f"Raster reading failed: {raster_error}, using PIL image dimensions")
                    height = width = None
            
            # If raster reading failed or no file path, get dimensions from PIL image
            if height is None or width is None:
                width, height = image.size
                method = "slidingwindow_pil"
                print(f"Using PIL dimensions: {width}x{height} from image object")
            
            try:
                # Generate windows using slidingwindow for any image type
                windows = slidingwindow.generateForSize(
                    height=height,
                    width=width,
                    dimOrder=slidingwindow.DimOrder.ChannelHeightWidth,
                    maxWindowSize=patch_size,
                    overlapPercent=patch_overlap
                )
                
                print(f"Generated {len(windows)} tiles using slidingwindow with method: {method}")
                
                tiles = []
                tile_metadata = []
                
                for i, window in enumerate(windows):
                    x = window.x
                    y = window.y
                    w = window.w
                    h = window.h
                    
                    # Extract actual image data for this tile
                    if method == "slidingwindow_raster" and image_file_path:
                        try:
                            with rio.open(image_file_path) as src:
                                window_data = src.read(window=Window(x, y, w, h))
                                if window_data.ndim == 3:
                                    window_data = window_data.transpose(1, 2, 0)

                                if window_data.dtype != np.uint8:
                                    if window_data.max() <= 1.0:
                                        window_data = (window_data * 255).astype(np.uint8)
                                    else:
                                        window_data = window_data.astype(np.uint8)

                                tile_pil = Image.fromarray(window_data)
                                print(f"Tile {i}: Read raster data {window_data.shape} -> PIL {tile_pil.size}")
                                
                        except Exception as raster_read_error:
                            print(f"Failed to read raster tile {i}: {raster_read_error}")
                            tile_pil = image.crop((x, y, x + w, y + h))
                            print(f"Tile {i}: Fallback PIL crop -> {tile_pil.size}")
                    else:
                        tile_pil = image.crop((x, y, x + w, y + h))
                        print(f"Tile {i}: PIL crop ({x},{y},{x+w},{y+h}) -> {tile_pil.size}")
                    
                    tiles.append(tile_pil)
                    
                    # Create tile metadata with tile info
                    metadata = {
                        "tile_index": i,
                        "window_coords": {
                            "x": x,
                            "y": y, 
                            "width": w,
                            "height": h
                        },
                        "tile_size": tile_pil.size,
                        "original_image_size": (width, height),
                        "method": method,
                        "actual_crop_bounds": (x, y, x + w, y + h)
                    }
                    tile_metadata.append(metadata)
                
                print(f"Successfully created {len(tiles)} tiles using slidingwindow method")
                return tiles, tile_metadata
                
            except Exception as slidingwindow_error:
                print(f"Slidingwindow method failed: {slidingwindow_error}, falling back to deepforest preprocess")
        
        # Fallback to deepforest preprocess method only if slidingwindow failed
        print(f"Using PIL-based tiling for image with size {image.size}")
        
        numpy_image = np.array(image)
        
        if numpy_image.shape[2] == 4:
            numpy_image = numpy_image[:, :, :3]
        elif numpy_image.shape[2] != 3:
            raise ValueError(f"Image must have 3 channels (RGB), got {numpy_image.shape[2]}")

        numpy_image = numpy_image.transpose(2, 0, 1)
        numpy_image = numpy_image / 255.0
        numpy_image = numpy_image.astype(np.float32)
        
        print(f"Tiling image with shape {numpy_image.shape} using patch_size={patch_size}, patch_overlap={patch_overlap}")

        windows = preprocess.compute_windows(numpy_image, patch_size, patch_overlap)
        
        print(f"Generated {len(windows)} tiles for analysis using deepforest preprocess")

        tiles = []
        tile_metadata = []
        
        for i, window in enumerate(windows):
            tile_array = numpy_image[window.indices()]
            tile_array = tile_array.transpose(1, 2, 0)
            if tile_array.dtype != np.uint8:
                tile_array = (tile_array * 255).astype(np.uint8) if tile_array.max() <= 1.0 else tile_array.astype(np.uint8)

            tile_pil = Image.fromarray(tile_array)
            tiles.append(tile_pil)

            x, y, w, h = window.getRect()
            print(f"DeepForest tile {i}: array shape {tile_array.shape} -> PIL {tile_pil.size}")
            
            # Create tile metadata
            metadata = {
                "tile_index": i,
                "window_coords": {
                    "x": x,
                    "y": y, 
                    "width": w,
                    "height": h
                },
                "tile_size": tile_pil.size,
                "original_image_size": image.size,
                "method": "deepforest_preprocess"
            }
            tile_metadata.append(metadata)

        if not tiles:
            raise Exception("No tiles were created - check image dimensions and parameters")
        
        # Check for empty or invalid tiles
        valid_tiles = []
        valid_metadata = []
        for i, tile in enumerate(tiles):
            if tile.size[0] > 0 and tile.size[1] > 0:
                valid_tiles.append(tile)
                valid_metadata.append(tile_metadata[i])
            else:
                print(f"Warning: Tile {i} has invalid size {tile.size}, skipping")
        
        if not valid_tiles:
            raise Exception("No valid tiles were created")
        
        if len(valid_tiles) != len(tiles):
            print(f"Filtered {len(tiles)} -> {len(valid_tiles)} valid tiles")
            tiles = valid_tiles
            tile_metadata = valid_metadata
        
        print(f"Successfully created {len(tiles)} tiles for multi-image analysis using fallback method")
        return tiles, tile_metadata
        
    except Exception as e:
        print(f"Error during image tiling: {e}")
        raise e