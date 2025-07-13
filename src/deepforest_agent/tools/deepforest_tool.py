import json
import os
import tempfile
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import shape

from deepforest import main
from deepforest.model import CropModel
from deepforest_agent.conf.config import Config
from deepforest_agent.utils.image_utils import convert_rgb_to_bgr, convert_bgr_to_rgb, load_image_as_np_array, create_temp_image_file, cleanup_temp_file


class DeepForestPredictor:
    """Predictor class for DeepForest object detection models."""

    def __init__(self):
        """Initialize the DeepForest predictor."""
        pass

    def _generate_detection_summary(self, predictions_df: pd.DataFrame,
                                   alive_dead_trees: bool = False) -> str:
        """
        Generate summary of detection results.

        Args:
            predictions_df: DataFrame containing detection results
            alive_dead_trees: Whether alive/dead tree classification was used

        Returns:
            DeepForest Detection Summary String
        """
        if predictions_df.empty:
            return "No objects detected by DeepForest with the requested models."

        detection_summary_parts = []
        counts = predictions_df['label'].value_counts()

        if 'classification_label' in predictions_df.columns:
            non_tree_df = predictions_df[predictions_df['label'] != 'tree']
            if not non_tree_df.empty:
                non_tree_counts = non_tree_df['label'].value_counts()
                for label, count in non_tree_counts.items():
                    label_str = str(label).replace('_', ' ')
                    if count == 1:
                        detection_summary_parts.append(f"{count} {label_str}")
                    else:
                        detection_summary_parts.append(f"{count} {label_str}s")

            tree_df = predictions_df[predictions_df['label'] == 'tree']
            if not tree_df.empty:
                total_trees = len(tree_df)
                classification_counts = tree_df['classification_label'].value_counts()
                
                classification_parts = []
                for class_label, count in classification_counts.items():
                    class_str = str(class_label).replace('_', ' ')
                    classification_parts.append(f"{count} are classified as {class_str}")
                
                if total_trees == 1:
                    detection_summary_parts.append(f"from {total_trees} tree, {' and '.join(classification_parts)}")
                else:
                    detection_summary_parts.append(f"from {total_trees} trees, {' and '.join(classification_parts)}")
        else:
            for label, count in counts.items():
                label_str = str(label).replace('_', ' ')
                if count == 1:
                    detection_summary_parts.append(f"{count} {label_str}")
                else:
                    detection_summary_parts.append(f"{count} {label_str}s")

        detection_summary = f"DeepForest detected: {', '.join(detection_summary_parts)}."

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
        image = convert_rgb_to_bgr(image)

        for _, row in predictions.iterrows():
            xmin, ymin = int(row['xmin']), int(row['ymin'])
            xmax, ymax = int(row['xmax']), int(row['ymax'])

            if 'classification_label' in row and pd.notna(row['classification_label']):
                label = str(row['classification_label'])
            else:
                label = str(row['label'])
            color = colors.get(label.lower(), (200, 200, 200))

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)

            text_x = xmin
            text_y = ymin - 10 if ymin - 10 > 10 else ymin + 15
            cv2.putText(image, label, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        image = convert_bgr_to_rgb(image)

        return image

    def predict_objects(
        self,
        image_data_array: Optional[np.ndarray] = None,
        image_file_path: Optional[str] = None,
        model_names: Optional[List[str]] = None,
        patch_size: int = Config.DEEPFOREST_DEFAULTS["patch_size"],
        patch_overlap: float = Config.DEEPFOREST_DEFAULTS["patch_overlap"],
        iou_threshold: float = Config.DEEPFOREST_DEFAULTS["iou_threshold"],
        thresh: float = Config.DEEPFOREST_DEFAULTS["thresh"],
        alive_dead_trees: bool = Config.DEEPFOREST_DEFAULTS["alive_dead_trees"]
    ) -> Tuple[str, Optional[np.ndarray], List[Dict[str, Any]]]:
        """
        Predict objects using DeepForest models with predict_tile method of DeepForest models

        Args:
            image_data_array: Input image as numpy array (optional if image_file_path not provided)
            image_file_path: Path to image file
            model_names: List of model names to use for prediction
            patch_size: Size of patches for tiled prediction
            patch_overlap: Patch overlap among windows
            iou_threshold: Minimum IoU overlap among predictions between windows to be suppressed
            thresh: Score threshold used to filter bboxes after soft-NMS is performed
            alive_dead_trees: Whether to classify trees as alive/dead

        Returns:
            Tuple containing:
            - detection_summary: Human-readable summary of detections
            - annotated_image_array: Image with bounding boxes drawn
            - detections_list: List of detection data
        """

        if model_names is None:
            model_names = ["tree", "bird", "livestock"]

        if image_file_path is None and image_data_array is None:
            raise ValueError("Either image_data_array or image_file_path must be provided")

        temp_file_path = None
        use_provided_path = image_file_path is not None
        
        if not use_provided_path:
            if image_data_array is not None:
                temp_file_path = create_temp_image_file(image_data_array, suffix=".png")
                working_file_path = temp_file_path
                working_array = image_data_array
            else:
                raise ValueError("image_data_array cannot be None when use_provided_path is False")
        else:
            working_file_path = image_file_path
            working_array = load_image_as_np_array(image_file_path)

        all_predictions_df = pd.DataFrame({
            "xmin": pd.Series(dtype=int),
            "ymin": pd.Series(dtype=int), 
            "xmax": pd.Series(dtype=int),
            "ymax": pd.Series(dtype=int),
            "score": pd.Series(dtype=float),
            "label": pd.Series(dtype=str),
            "model_type": pd.Series(dtype=str)
        })

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
                    crop_model_instance = CropModel(num_classes=2)
                    current_predictions = model.predict_tile(
                        raster_path=working_file_path,
                        patch_size=patch_size,
                        patch_overlap=patch_overlap,
                        crop_model=crop_model_instance,
                        iou_threshold=iou_threshold,
                        thresh=thresh
                    )
                else:
                    current_predictions = model.predict_tile(
                        raster_path=working_file_path,
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
                        current_predictions['classification_label'] = (
                            current_predictions.apply(
                                lambda row: (
                                    'alive_tree' if row['cropmodel_label'] == 0
                                    else 'dead_tree' if row['cropmodel_label'] == 1
                                    else row['label']
                                ),
                                axis=1
                            )
                        )
                        if 'cropmodel_score' in current_predictions.columns:
                            current_predictions['classification_score'] = current_predictions['cropmodel_score']
                            current_predictions = current_predictions.drop(columns=['cropmodel_score'], errors='ignore')
                        
                        current_predictions = current_predictions.drop(
                            columns=['cropmodel_label'],
                            errors='ignore'
                        )

                    all_predictions_df = pd.concat(
                        [all_predictions_df, current_predictions],
                        ignore_index=True
                    )

            except Exception as e:
                print(f"Error during DeepForest prediction for model "
                      f"'{model_type}': {e}")
        if temp_file_path:
            cleanup_temp_file(temp_file_path)

        # Generate detection summary
        detection_summary = self._generate_detection_summary(
            all_predictions_df, alive_dead_trees
        )

        # Create annotated image with bounding boxes
        annotated_image_array = None
        if working_array.ndim == 2:
            annotated_image_array = cv2.cvtColor(
                working_array, cv2.COLOR_GRAY2RGB
            )
        elif (working_array.ndim == 3 and
                working_array.shape[2] == 4):
            annotated_image_array = cv2.cvtColor(
                working_array, cv2.COLOR_RGBA2RGB
            )
        else:
            annotated_image_array = working_array.copy()

        if annotated_image_array.dtype != np.uint8:
            annotated_image_array = annotated_image_array.astype(np.uint8)

        annotated_image_array = self._plot_boxes(
            annotated_image_array, all_predictions_df, Config.COLORS
        )

        output_df = all_predictions_df.copy()

        essential_columns = ['xmin', 'ymin', 'xmax', 'ymax', 'score', 'label']
        if 'classification_label' in output_df.columns:
            essential_columns.append('classification_label')
        if 'classification_score' in output_df.columns:
            essential_columns.append('classification_score')

        output_df = output_df[
            [col for col in essential_columns if col in output_df.columns]
        ]
        detections_list = []
        if not output_df.empty:
            for _, row in output_df.iterrows():
                record = {
                    "xmin": int(row['xmin']),
                    "ymin": int(row['ymin']),
                    "xmax": int(row['xmax']),
                    "ymax": int(row['ymax']),
                    "score": float(row['score']),
                    "label": str(row['label'])
                }
                if 'classification_label' in row:
                    record["classification_label"] = str(row['classification_label'])
                if 'classification_score' in row:
                    try:
                        record["classification_score"] = float(row['classification_score'])
                    except (ValueError, TypeError):
                        pass

                detections_list.append(record)

        return detection_summary, annotated_image_array, detections_list