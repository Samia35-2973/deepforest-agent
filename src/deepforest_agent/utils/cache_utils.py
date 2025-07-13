import hashlib
import json
import time
import tempfile
import os
from typing import Dict, Any, Optional, List
from PIL import Image
import pickle
import gzip
import base64

from deepforest_agent.conf.config import Config
from deepforest_agent.utils.image_utils import convert_pil_image_to_bytes

class ToolCallCache:
    """
    Cache utility with data handling and efficient image storage.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the tool call cache with data handling.
        
        Args:
            cache_dir: Directory to store cached images. If None, uses system temp directory.
        """
        self.cache_data = {}

        if cache_dir is None:
            self.cache_dir = os.path.join(tempfile.gettempdir(), "deepforest_cache")
        else:
            self.cache_dir = cache_dir
            
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Cache directory: {self.cache_dir}")
    
    def _normalize_arguments(self, arguments: Dict[str, Any]) -> str:
        """
        Normalize tool arguments to create a consistent cache key.
        
        Args:
            arguments: Tool arguments to normalize
            
        Returns:
            Normalized JSON string of arguments sorted by key
        """
        normalized_args = Config.DEEPFOREST_DEFAULTS.copy()
        normalized_args.update(arguments)
        if "model_names" in arguments:
            normalized_args["model_names"] = arguments["model_names"]
        
        print(f"Cache normalization: {arguments} -> {normalized_args}")
        return json.dumps(normalized_args, sort_keys=True, separators=(',', ':'))
    
    def _create_cache_key(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Create a unique cache key from tool name and arguments.
        
        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool
            
        Returns:
            MD5 hash that uniquely identifies this tool call
        """
        cache_input = f"{tool_name}:{self._normalize_arguments(arguments)}"
        return hashlib.md5(cache_input.encode('utf-8')).hexdigest()
    
    def _store_image(self, image: Image.Image, cache_key: str) -> str:
        """
        Store PIL Image while preserving original characteristics.
        
        Args:
            image: PIL Image to store
            cache_key: Unique identifier for this cache entry
            
        Returns:
            File path where the image was stored
        """
        if image is None:
            return None

        image_filename = f"cached_image_{cache_key}.pkl.gz"
        image_path = os.path.join(self.cache_dir, image_filename)
        
        try:
            # Pickle for exact PIL Image preservation, compressed with gzip
            with gzip.open(image_path, 'wb') as f:
                pickle.dump(image, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
            print(f"Image cached to {image_path} ({file_size_mb:.2f} MB)")
            
            return image_path
            
        except Exception as e:
            print(f"Error storing image efficiently: {e}")
            return self._fallback_image_storage(image)
    
    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load PIL Image from storage.
        
        Args:
            image_path: File path where image was stored
            
        Returns:
            Reconstructed PIL Image, or None if loading fails
        """
        if not image_path or not os.path.exists(image_path):
            return None
            
        try:
            with gzip.open(image_path, 'rb') as f:
                image = pickle.load(f)
            
            print(f"Image loaded from cache: {image_path}")
            return image
            
        except Exception as e:
            print(f"Error loading cached image: {e}")
            return None
    
    def _fallback_image_storage(self, image: Image.Image) -> str:
        """
        Fallback method for image storage when storage fails.
        
        Args:
            image: PIL Image to store

        Returns:
            Base64 encoded string of the image
        """
        img_bytes = convert_pil_image_to_bytes(image)
        
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def get_cached_result(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result with data handling.
        
        Args:
            tool_name: Name of the tool being called
            arguments: Arguments for the tool call
            
        Returns:
            Dictionary containing all cached data or None if not found
        """
        cache_key = self._create_cache_key(tool_name, arguments)
        
        if cache_key not in self.cache_data:
            print(f"Cache MISS: No cached result for {tool_name} with key {cache_key}")
            return None
        
        cached_entry = self.cache_data[cache_key]
        cached_result = {}
        
        if "detection_summary" in cached_entry["result"]:
            cached_result["detection_summary"] = cached_entry["result"]["detection_summary"]
            print(f"Cache: Retrieved detection_summary: {cached_result['detection_summary']}")

        if "detections_list" in cached_entry["result"]:
            cached_result["detections_list"] = cached_entry["result"]["detections_list"]
            print(f"Cache: Retrieved {len(cached_result['detections_list'])} detections")

        if "total_detections" in cached_entry["result"]:
            cached_result["total_detections"] = cached_entry["result"]["total_detections"]

        if "status" in cached_entry["result"]:
            cached_result["status"] = cached_entry["result"]["status"]

        if "annotated_image_path" in cached_entry["result"]:
            cached_result["annotated_image"] = self._load_image(
                cached_entry["result"]["annotated_image_path"]
            )
            if cached_result["annotated_image"]:
                print(f"Cache: Retrieved annotated image ({cached_result['annotated_image'].size})")

        cached_result["cache_info"] = {
            "cached_at": cached_entry["timestamp"],
            "cache_hit": True,
            "cache_key": cache_key,
            "tool_name": tool_name,
            "arguments": arguments
        }
        
        print(f"Successfully retrieved all data for {tool_name}")
        return cached_result
    
    def store_result(self, tool_name: str, arguments: Dict[str, Any], result: Dict[str, Any]) -> str:
        """
        Store tool call result with data handling.
        
        Args:
            tool_name: Name of the tool that was executed
            arguments: Arguments that were passed to the tool
            result: Result dictionary containing:
                - detection_summary (str): Text summary of what was detected
                - detections_list (List): List of detection objects
                - total_detections (int): Count of detections
                - status (str): Success/error status
                - annotated_image (PIL.Image, optional): Image with annotations
                
        Returns:
            Cache key that was used to store this result
        """
        cache_key = self._create_cache_key(tool_name, arguments)

        storable_result = {}

        if "detection_summary" in result:
            storable_result["detection_summary"] = result["detection_summary"]
            print(f"Detection_summary = {result['detection_summary']}")
        else:
            print("No detection_summary found in result to cache")

        if "detections_list" in result:
            storable_result["detections_list"] = result["detections_list"]
            print(f"Detections_list with {len(result['detections_list'])} items")
        else:
            print("No detections_list found in result to cache")
            storable_result["detections_list"] = []

        if "total_detections" in result:
            storable_result["total_detections"] = result["total_detections"]
        else:
            storable_result["total_detections"] = len(storable_result["detections_list"])

        if "status" in result:
            storable_result["status"] = result["status"]
        else:
            storable_result["status"] = "unknown"

        if "annotated_image" in result and result["annotated_image"] is not None:
            image_path = self._store_image(result["annotated_image"], cache_key)
            if image_path:
                storable_result["annotated_image_path"] = image_path
                print(f"Annotated_image stored efficiently")
        else:
            print("No annotated_image to store")

        self.cache_data[cache_key] = {
            "tool_name": tool_name,
            "arguments": arguments.copy(),
            "result": storable_result,
            "timestamp": time.time(),
            "cache_key": cache_key
        }
        
        print(f"Successfully cached all data for {tool_name} with key {cache_key}")
        return cache_key
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about cached data.
        
        Returns:
            Dictionary with comprehensive cache statistics
        """
        total_images = 0
        total_detections = 0
        cache_size_mb = 0
        
        for entry in self.cache_data.values():
            result = entry["result"]

            if "annotated_image_path" in result:
                total_images += 1
                # Calculate file size if image exists
                if os.path.exists(result["annotated_image_path"]):
                    cache_size_mb += os.path.getsize(result["annotated_image_path"]) / (1024 * 1024)
            
            # Count total detections across all cached results
            total_detections += result.get("total_detections", 0)
        
        return {
            "total_entries": len(self.cache_data),
            "total_images_cached": total_images,
            "total_detections_cached": total_detections,
            "cache_size_mb": round(cache_size_mb, 2),
            "cache_directory": self.cache_dir,
            "tools_cached": set(entry["tool_name"] for entry in self.cache_data.values())
        }
    
    def cleanup_cache_files(self):
        """
        Clean up cached image files from disk.

        Returns:
            The total number of files that were successfully removed.
        """
        files_removed = 0
        for entry in self.cache_data.values():
            if "annotated_image_path" in entry["result"]:
                image_path = entry["result"]["annotated_image_path"]
                if os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                        files_removed += 1
                    except Exception as e:
                        print(f"Error removing cached image {image_path}: {e}")
        
        print(f"Cleaned up {files_removed} cached image files")
        return files_removed

tool_call_cache = ToolCallCache()