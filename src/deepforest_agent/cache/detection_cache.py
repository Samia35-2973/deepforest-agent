import numpy as np
from typing import Tuple, List
from deepforest_agent.utils.parameters_manager import DetectionParameters

class CacheManager:
    """
    Intelligent cache management for detection results.
    
    This class encapsulates all caching logic including validation, updates,
    and retrieval operations, providing a clean interface for cache operations.
    """

    def __init__(self):
        """Initialize cache with structured prediction storage."""
        self.cached_predictions = {
            "image_data": None,
            "predictions_json_str": None,
            "annotated_image_array": None,
            "summary_text": None,
            "models_detected": set(),
            "last_alive_dead_trees_requested": False,
            "current_image_hash": None,
            "detection_parameters": {},
            "detection_parameters_dict": {}
        }

    def should_run_detection(self, image_hash: str, params: DetectionParameters) -> Tuple[bool, str]:
        """
        Determine whether to run new detection based on cache state.
        
        Args:
            image_hash: Hash of the current image
            params: Detection parameters for the request
            
        Returns:
            Tuple of (should_run: bool, reason: str)
        """
        # Check for image changes
        if image_hash != self.cached_predictions["current_image_hash"]:
            return True, "New image detected (hash changed)"
        
        # Check for new models
        requested_models = set(params.model_names)
        already_detected = self.cached_predictions["models_detected"]
        new_models = requested_models - already_detected
        
        if new_models:
            return True, f"New models requested: {list(new_models)}"
        
        # Check for parameter changes
        cached_params_dict = self.cached_predictions.get("detection_parameters_dict", {})
        requested_params_dict = params.to_dict()
        
        for model in requested_models:
            if model not in cached_params_dict:
                return True, f"No cached parameters for model: {model}"
            
            cached_params = cached_params_dict[model]
            
            for param_name, requested_value in requested_params_dict.items():
                cached_value = cached_params.get(param_name)
                if cached_value != requested_value:
                    return True, f"Parameter '{param_name}' changed for model '{model}': {cached_value} → {requested_value}"
        
        return False, f"All models {list(requested_models)} already detected with identical parameters"

    def update_cache(self, image_hash: str, params: DetectionParameters, 
                    summary_text: str, annotated_image_array: np.ndarray, 
                    json_output: str) -> None:
        """
        Update cache with new detection results.
        
        Args:
            image_hash: Hash of the processed image
            params: Detection parameters used
            summary_text: Human-readable summary of results
            annotated_image_array: Image with bounding boxes
            json_output: JSON string of detection data
        """
        self.cached_predictions.update({
            "summary_text": summary_text,
            "annotated_image_array": annotated_image_array,
            "predictions_json_str": json_output,
            "current_image_hash": image_hash,
            "models_detected": self.cached_predictions["models_detected"].union(set(params.model_names)),
        })
        
        # Update parameter tracking for each model
        params_dict = params.to_dict()
        for model in params.model_names:
            self.cached_predictions["detection_parameters_dict"][model] = params_dict.copy()
        
        print(f"CACHE UPDATE: Models {params.model_names} cached with parameters: {params_dict}")

    def get_detection_summary(self, requested_models: List[str]) -> str:
        """
        Generate summary of cached detections for requested models.
        
        Args:
            requested_models: List of model names being requested
            
        Returns:
            Human-readable summary of available cached results
        """
        if not self.cached_predictions["summary_text"]:
            return "No previous detections available."
        
        cached_models = self.cached_predictions["models_detected"]
        available_models = set(requested_models).intersection(cached_models)
        
        if not available_models:
            return "No cached detections for the requested models."
        
        summary_parts = [
            f"Using cached detection results for: {', '.join(sorted(available_models))}",
            self.cached_predictions["summary_text"] or "Detection completed successfully."
        ]
        
        return "\n".join(summary_parts)

    def clear_cache_for_new_image(self, new_image_hash: str) -> None:
        """
        Clear cache when a new image is detected.
        
        Args:
            new_image_hash: Hash of the new image
        """
        print("New image detected, clearing detection cache")
        self.cached_predictions.update({
            "current_image_hash": new_image_hash,
            "models_detected": set(),
            "detection_parameters": {},
            "detection_parameters_dict": {},
            "summary_text": None,
            "predictions_json_str": None,
            "annotated_image_array": None
        })
