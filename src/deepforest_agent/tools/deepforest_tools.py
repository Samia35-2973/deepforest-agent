import json
import os
import tempfile
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import shape

from deepforest import main
from deepforest.model import CropModel
from deepforest_agent.conf.config import Config
from deepforest_agent.utils.image_utils import convert_rgb_to_bgr, convert_bgr_to_rgb


class DeepForestPredictor:
    """Predictor class for DeepForest object detection models."""

    def __init__(self):
        """Initialize the DeepForest predictor."""
        pass

    def _generate_detection_summary(self, predictions_df: pd.DataFrame, 
                                   alive_dead_trees: bool = False) -> str:
        """
        Generate human-readable summary of detection results.
        
        Args:
            predictions_df: DataFrame containing detection results
            alive_dead_trees: Whether alive/dead tree classification was used
            
        Returns:
            Human-readable summary string
        """
        if predictions_df.empty:
            return "No objects detected by DeepForest with the requested models."
        
        detection_summary_parts = []
        counts = predictions_df['label'].value_counts()
        
        for label, count in counts.items():
            detection_summary_parts.append(f"{count} {label.replace('_', ' ')}s")
        
        detection_summary = f"DeepForest detected: {', '.join(detection_summary_parts)}."

        if alive_dead_trees and ("alive_tree" in counts or "dead_tree" in counts):
            alive_count = counts.get('alive_tree', 0)
            dead_count = counts.get('dead_tree', 0)
            detection_summary += (
                f" Specifically, {alive_count} alive trees and "
                f"{dead_count} dead trees."
            )
        
        return detection_summary

    @staticmethod
    def _plot_boxes(image_array: np.ndarray, predictions: pd.DataFrame, 
                   colors: dict, thickness: int = 2) -> np.ndarray:
        """
        Plot bounding boxes on image.
        
        Args:
            image_array: Input image as numpy array
            predictions: DataFrame with detection results
            colors: Color mapping for different labels
            thickness: Line thickness for bounding boxes
            
        Returns:
            Image array with drawn bounding boxes
        """
        image = image_array.copy()
        if (image.ndim == 3 and image.shape[2] == 3 and 
            image.dtype == np.uint8):
            image = convert_rgb_to_bgr(image)

        for _, row in predictions.iterrows():
            xmin, ymin = int(row['xmin']), int(row['ymin'])
            xmax, ymax = int(row['xmax']), int(row['ymax'])
            label = str(row['label'])
            color = colors.get(label.lower(), (200, 200, 200))
            
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
            
            text_x = xmin
            text_y = ymin - 10 if ymin - 10 > 10 else ymin + 15
            cv2.putText(image, label, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        
        if (image_array.ndim == 3 and image_array.shape[2] == 3 and 
            image_array.dtype == np.uint8):
            image = convert_bgr_to_rgb(image)
        return image

    def predict_objects(
        self,
        image_data_array: np.ndarray,
        model_names: Optional[List[str]] = None,
        patch_size: int = 400,
        patch_overlap: float = 0.05,
        iou_threshold: float = 0.15,
        thresh: float = 0.001,
        alive_dead_trees: bool = False
    ) -> Tuple[str, Optional[np.ndarray], str]:
        """
        Predict objects using DeepForest models with intelligent method selection.
        
        This function uses predict_tile method of DeepForest models
        
        Args:
            image_data_array: Input image as numpy array
            model_names: List of model names to use for prediction
            patch_size: Size of patches for tiled prediction
            patch_overlap: Overlap ratio between patches
            iou_threshold: IoU threshold for non-maximum suppression
            thresh: Confidence threshold for detections
            alive_dead_trees: Whether to classify trees as alive/dead
            
        Returns:
            Tuple containing:
            - detection_summary: Human-readable summary of detections
            - annotated_image_array: Image with bounding boxes drawn
            - json_output: JSON string with detection data
        """

        all_predictions_df = pd.DataFrame(
            columns=["xmin", "ymin", "xmax", "ymax", "score", "label", "model_type"]
        )
        
        model_instances = {}
        for model_name_key in model_names:
            model_path = Config.DEEPFOREST_MODELS.get(model_name_key)
            if model_path is None:
                print(f"Warning: Model '{model_name_key}' not found in "
                      f"Config.DEEPFOREST_MODELS. Skipping.")
                continue
            
            try:
                model = main.deepforest()
                model.load_model(model_name=model_path)
                model_instances[model_name_key] = model
            except Exception as e:
                print(f"Error loading DeepForest model '{model_name_key}' "
                      f"from path '{model_path}': {e}. Skipping this model.")
                continue
        
        temp_file_path = None

        # Process each model
        for model_type, model in model_instances.items():
            current_predictions = pd.DataFrame()
            try:
                if model_type == "tree" and alive_dead_trees:
                    with tempfile.NamedTemporaryFile(suffix=".png", 
                                                   delete=False) as tmp_file:
                        temp_file_path = tmp_file.name
                        pil_image = Image.fromarray(image_data_array)
                        pil_image.save(temp_file_path, format='PNG')
                    
                    print(f"Saved NumPy array to temporary file: "
                          f"{temp_file_path} for CropModel prediction.")
                    
                    crop_model_instance = CropModel(num_classes=2)
                    current_predictions = model.predict_tile(
                        raster_path=temp_file_path,
                        patch_size=patch_size,
                        patch_overlap=patch_overlap,
                        crop_model=crop_model_instance,
                        iou_threshold=iou_threshold,
                        thresh=thresh
                    )
                else:
                    current_predictions = model.predict_tile(
                        image=image_data_array,
                        patch_size=patch_size,
                        patch_overlap=patch_overlap,
                        iou_threshold=iou_threshold,
                        thresh=thresh
                    )

                if not current_predictions.empty:
                    current_predictions['model_type'] = model_type
                    if 'label' in current_predictions.columns:
                        current_predictions['label'] = (
                            current_predictions['label'].apply(
                                lambda x: str(x).lower()
                            )
                        )

                    # Handle alive/dead tree classification results
                    if (alive_dead_trees and 'cropmodel_label' in 
                        current_predictions.columns and model_type == "tree"):
                        current_predictions['label'] = (
                            current_predictions.apply(
                                lambda row: (
                                    'alive_tree' if row['cropmodel_label'] == 0 
                                    else 'dead_tree' if row['cropmodel_label'] == 1 
                                    else row['label']
                                ),
                                axis=1
                            )
                        )
                        current_predictions = current_predictions.drop(
                            columns=['cropmodel_label', 'cropmodel_score'], 
                            errors='ignore'
                        )
                    
                    all_predictions_df = pd.concat(
                        [all_predictions_df, current_predictions], 
                        ignore_index=True
                    )

            except Exception as e:
                print(f"Error during DeepForest prediction for model "
                      f"'{model_type}': {e}")
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        print(f"Cleaned up temporary file: {temp_file_path}")
                    except OSError as e:
                        print(f"Error cleaning up temporary file "
                              f"{temp_file_path}: {e}")
                temp_file_path = None

        # Generate detection summary
        detection_summary = self._generate_detection_summary(
            all_predictions_df, alive_dead_trees
        )

        # Create annotated image with bounding boxes
        annotated_image_array = None
        if image_data_array.ndim == 2:
            annotated_image_array = cv2.cvtColor(
                image_data_array, cv2.COLOR_GRAY2RGB
            )
        elif (image_data_array.ndim == 3 and 
                image_data_array.shape[2] == 4):
            annotated_image_array = cv2.cvtColor(
                image_data_array, cv2.COLOR_RGBA2RGB
            )
        else:
            annotated_image_array = image_data_array.copy()

        if annotated_image_array.dtype != np.uint8:
            annotated_image_array = annotated_image_array.astype(np.uint8)

        annotated_image_array = self._plot_boxes(
            annotated_image_array, all_predictions_df, Config.COLORS
        )
            
        json_output_df = all_predictions_df.copy()
        
        essential_columns = ['xmin', 'ymin', 'xmax', 'ymax', 'score', 'label']
        json_output_df = json_output_df[
            [col for col in essential_columns if col in json_output_df.columns]
        ]
        
        json_output = (
            json_output_df.to_json(orient='records', default_handler=str)
            if not json_output_df.empty else json.dumps([])
        )

        return detection_summary, annotated_image_array, json_output