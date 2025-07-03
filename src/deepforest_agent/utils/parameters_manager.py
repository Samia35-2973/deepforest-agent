import numpy as np
from typing import List
from dataclasses import dataclass, field

@dataclass
class DetectionParameters:
    """
    Centralized management of detection parameters with defaults and validation.
    
    This class uses simple class variables that can be accessed both from the class
    (DetectionParameters.patch_size) and from instances (params.patch_size).
    Each default is defined exactly once, eliminating repetition.
    """
    patch_size: int = 400
    patch_overlap: float = 0.05
    iou_threshold: float = 0.15
    thresh: float = 0.001
    alive_dead_trees: bool = False
    model_names: List[str] = field(default_factory=lambda: ["bird", "tree", "livestock"])

    @classmethod
    def get_default_model_names(cls) -> List[str]:
        """
        Get the default model names for class-level access.
        
        This helper method access the default value at the class level 
        for tool declarations.
        
        Returns:
            List of default model names
        """
        return ["bird", "tree", "livestock"]

    @classmethod
    def from_arguments(cls, arguments: dict) -> 'DetectionParameters':
        """
        Create DetectionParameters from user arguments with intelligent defaults.
        
        This method extracts parameters from user input and applies our class defaults
        for any missing values, ensuring consistent behavior across the system.
        
        Args:
            arguments: Dictionary of user-provided parameters from tool calls
            
        Returns:
            DetectionParameters instance with normalized values
        """
        return cls(
            patch_size=arguments.get('patch_size', cls.patch_size),
            patch_overlap=arguments.get('patch_overlap', cls.patch_overlap),
            iou_threshold=arguments.get('iou_threshold', cls.iou_threshold),
            thresh=arguments.get('thresh', cls.thresh),
            alive_dead_trees=arguments.get('alive_dead_trees', cls.alive_dead_trees),
            model_names=arguments.get('model_names', cls.get_default_model_names())
        )

    def to_dict(self) -> dict:
        """
        Convert to dictionary for caching and comparison operations.
        
        This method creates a dictionary representation suitable for comparing
        parameters across different requests to determine cache validity.
        
        Returns:
            Dictionary with parameter names as keys and current values
        """
        return {
            'patch_size': self.patch_size,
            'patch_overlap': self.patch_overlap,
            'iou_threshold': self.iou_threshold,
            'thresh': self.thresh,
            'alive_dead_trees': self.alive_dead_trees
        }

    def to_deepforest_args(self, image_data: np.ndarray) -> dict:
        """
        Convert to arguments suitable for DeepForest predictor.
        
        This method transforms our normalized parameter structure into the
        specific argument format expected by the DeepForest predict_objects method.
        
        Args:
            image_data: Image data as numpy array to be processed
            
        Returns:
            Dictionary of arguments for DeepForest predict_objects method
        """
        return {
            "image_data_array": image_data,
            "model_names": self.model_names,
            "patch_size": self.patch_size,
            "patch_overlap": self.patch_overlap,
            "iou_threshold": self.iou_threshold,
            "thresh": self.thresh,
            "alive_dead_trees": self.alive_dead_trees
        }