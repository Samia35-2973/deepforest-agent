import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from deepforest_agent.conf.config import Config
from deepforest_agent.tools.deepforest_tools import DeepForestPredictor
from deepforest_agent.utils.image_utils import load_image_as_np_array

TEST_IMAGE_PATH_SMALL = "data/AWPE Pigeon Lake 2020 DJI_0005.JPG"
TEST_IMAGE_PATH_LARGE = "data/OSBS_029.tif"

deepforest_predictor = DeepForestPredictor()


def display_image_for_test(image_array: np.ndarray, title: str = "Test Image"):
    """
    Display an image using matplotlib for visual inspection during testing.
    
    Args:
        image_array: Image as numpy array
        title: Title for the plot
    """
    plt.imshow(image_array)
    plt.axis('off')
    plt.title(title)
    plt.show()


def test_deepforest_predict_objects_basic_detection_bird():
    """Test basic bird detection with default parameters on a small image."""
    image_array = load_image_as_np_array(TEST_IMAGE_PATH_SMALL)
    if image_array is None:
        return

    summary, annotated_image, raw_predictions_json = (
        deepforest_predictor.predict_objects(
            image_data_array=image_array,
            model_names=["bird"]
        )
    )

    assert "DeepForest detected" in summary
    assert ("bird" in summary or "No objects detected" in summary)
    assert annotated_image is not None
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape[:2] == image_array.shape[:2]

    predictions = json.loads(raw_predictions_json)
    assert isinstance(predictions, list)
    if predictions:
        bird_labels_found = any(
            p["label"] == "bird" for p in predictions if 'label' in p
        )
        assert bird_labels_found

    display_image_for_test(annotated_image, "Bird Detection Test")


def test_deepforest_predict_objects_basic_detection_tree():
    """Test basic tree detection with default parameters on a small image."""
    image_array = load_image_as_np_array(TEST_IMAGE_PATH_SMALL)
    if image_array is None:
        return

    summary, annotated_image, raw_predictions_json = (
        deepforest_predictor.predict_objects(
            image_data_array=image_array,
            model_names=["tree"]
        )
    )

    assert "DeepForest detected" in summary
    assert "tree" in summary or "No objects detected" in summary
    assert annotated_image is not None
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape[:2] == image_array.shape[:2]

    predictions = json.loads(raw_predictions_json)
    assert isinstance(predictions, list)
    if predictions:
        tree_labels_found = any(
            p["label"] == "tree" for p in predictions if 'label' in p
        )
        assert tree_labels_found

    display_image_for_test(annotated_image, "Tree Detection Test")


def test_deepforest_predict_objects_multiple_models():
    """Test detection using multiple models simultaneously."""
    image_array = load_image_as_np_array(TEST_IMAGE_PATH_SMALL)
    if image_array is None:
        return

    summary, annotated_image, raw_predictions_json = (
        deepforest_predictor.predict_objects(
            image_data_array=image_array,
            model_names=["bird", "tree", "livestock"]
        )
    )

    assert "DeepForest detected" in summary
    assert annotated_image is not None
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape[:2] == image_array.shape[:2]

    predictions = json.loads(raw_predictions_json)
    assert isinstance(predictions, list)
    if predictions:
        labels = {p['label'] for p in predictions if 'label' in p}
        assert "bird" in labels or "tree" in labels or "livestock" in labels

    display_image_for_test(annotated_image, "Multiple Models Test")


def test_deepforest_predict_objects_large_image_processing():
    """Test processing of large images using tiled prediction."""
    image_array = load_image_as_np_array(TEST_IMAGE_PATH_LARGE)
    if image_array is None:
        return

    summary, annotated_image, raw_predictions_json = (
        deepforest_predictor.predict_objects(
            image_data_array=image_array,
            model_names=["tree"],
            patch_size=400,
            patch_overlap=0.05,
            iou_threshold=0.15,
            thresh=0.001
        )
    )

    assert "DeepForest detected" in summary
    assert annotated_image is not None
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape[:2] == image_array.shape[:2]

    predictions = json.loads(raw_predictions_json)
    assert isinstance(predictions, list)
    if predictions:
        assert any(p['label'] == 'tree' for p in predictions if 'label' in p)

    display_image_for_test(annotated_image, "Large Image Processing Test")


def test_deepforest_predict_objects_custom_patch_size():
    """Test detection with custom patch size parameter."""
    image_array = load_image_as_np_array(TEST_IMAGE_PATH_SMALL)
    if image_array is None:
        return

    summary, annotated_image, raw_predictions_json = (
        deepforest_predictor.predict_objects(
            image_data_array=image_array,
            model_names=["tree"],
            patch_size=800,
            patch_overlap=0.05,
            iou_threshold=0.15,
            thresh=0.001
        )
    )

    assert "DeepForest detected" in summary or "No objects detected" in summary
    assert annotated_image is not None
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape[:2] == image_array.shape[:2]

    predictions = json.loads(raw_predictions_json)
    assert isinstance(predictions, list)
    if predictions:
        assert any(p['label'] == 'tree' for p in predictions if 'label' in p)

    display_image_for_test(annotated_image, "Custom Patch Size Test")


def test_deepforest_predict_objects_multiple_custom_parameters():
    """Test detection with multiple custom parameters."""
    image_array = load_image_as_np_array(TEST_IMAGE_PATH_SMALL)
    if image_array is None:
        return

    summary, annotated_image, raw_predictions_json = (
        deepforest_predictor.predict_objects(
            image_data_array=image_array,
            model_names=["tree"],
            patch_size=600,
            patch_overlap=0.1,
            iou_threshold=0.3,
            thresh=0.01
        )
    )

    assert "DeepForest detected" in summary or "No objects detected" in summary
    assert annotated_image is not None
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape[:2] == image_array.shape[:2]

    predictions = json.loads(raw_predictions_json)
    assert isinstance(predictions, list)
    if predictions:
        assert any(p['label'] == 'tree' for p in predictions if 'label' in p)

    display_image_for_test(annotated_image, "Multiple Custom Parameters Test")


def test_deepforest_predict_objects_alive_dead_trees():
    """Test alive/dead tree classification detection."""
    image_array = load_image_as_np_array(TEST_IMAGE_PATH_SMALL)
    if image_array is None:
        return

    summary, annotated_image, raw_predictions_json = (
        deepforest_predictor.predict_objects(
            image_data_array=image_array,
            model_names=["tree"],
            alive_dead_trees=True
        )
    )

    assert "DeepForest detected" in summary
    assert annotated_image is not None
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape[:2] == image_array.shape[:2]

    print(summary)

    predictions = json.loads(raw_predictions_json)
    assert isinstance(predictions, list)
    if predictions:
        labels = {p['label'] for p in predictions if 'label' in p}
        assert ('alive_tree' in labels or 'dead_tree' in labels or 
                'tree' in labels)

        assert (("alive trees" in summary or "dead trees" in summary) or 
                ("No objects detected" in summary))

    display_image_for_test(annotated_image, "Alive/Dead Tree Detection Test")


def test_deepforest_predict_objects_no_detections():
    """Test the function gracefully handles cases with no detections."""
    blank_image = np.zeros((100, 100, 3), dtype=np.uint8)

    summary, annotated_image, raw_predictions_json = (
        deepforest_predictor.predict_objects(
            image_data_array=blank_image,
            model_names=["tree"],
            thresh=1.0
        )
    )

    assert "No objects detected by DeepForest" in summary
    assert annotated_image is not None
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape[:2] == blank_image.shape[:2]

    predictions = json.loads(raw_predictions_json)
    assert isinstance(predictions, list)
    assert len(predictions) == 0

    display_image_for_test(annotated_image, "No Detections Test")


def test_deepforest_predict_objects_custom_thresholds():
    """Test detection with custom threshold parameters."""
    image_array = load_image_as_np_array(TEST_IMAGE_PATH_SMALL)
    if image_array is None:
        return

    summary, annotated_image, raw_predictions_json = (
        deepforest_predictor.predict_objects(
            image_data_array=image_array,
            model_names=["tree"],
            thresh=0.9,
            iou_threshold=0.5
        )
    )

    assert ("DeepForest detected" in summary or 
            "No objects detected" in summary)
    assert annotated_image is not None
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape[:2] == image_array.shape[:2]

    predictions = json.loads(raw_predictions_json)
    assert isinstance(predictions, list)
    if predictions:
        assert any(p['label'] == 'tree' for p in predictions if 'label' in p)

    display_image_for_test(annotated_image, "Custom Thresholds Test")


def test_deepforest_predict_objects_unsupported_model_name():
    """Test behavior with an unsupported model name."""
    image_array = load_image_as_np_array(TEST_IMAGE_PATH_SMALL)
    if image_array is None:
        return

    summary, annotated_image, raw_predictions_json = (
        deepforest_predictor.predict_objects(
            image_data_array=image_array,
            model_names=["tree", "nonexistent_model"]
        )
    )
    
    assert ("DeepForest detected" in summary or 
            "No objects detected" in summary)
    assert annotated_image is not None
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape[:2] == image_array.shape[:2]

    predictions = json.loads(raw_predictions_json)
    assert isinstance(predictions, list)
    if predictions:
        assert all(p['label'] == 'tree' for p in predictions if 'label' in p)

    display_image_for_test(annotated_image, "Unsupported Model Test")


def test_plot_boxes_basic():
    """Test _plot_boxes with some sample bounding box data."""
    img = np.zeros((100, 100, 3), dtype=np.uint8) + 255
    predictions = pd.DataFrame([
        {'xmin': 10, 'ymin': 10, 'xmax': 30, 'ymax': 30, 
         'label': 'bird', 'score': 0.9},
        {'xmin': 50, 'ymin': 50, 'xmax': 70, 'ymax': 70, 
         'label': 'tree', 'score': 0.8}
    ])

    annotated_img = DeepForestPredictor._plot_boxes(
        img, predictions, Config.COLORS
    )
    assert annotated_img.shape == img.shape
    assert not np.array_equal(annotated_img, img)

    display_image_for_test(annotated_img, "Plot Boxes Basic Test")


def test_plot_boxes_empty_predictions():
    """Test _plot_boxes with empty predictions DataFrame."""
    img = np.zeros((100, 100, 3), dtype=np.uint8) + 255
    predictions = pd.DataFrame(
        columns=['xmin', 'ymin', 'xmax', 'ymax', 'label', 'score']
    )

    annotated_img = DeepForestPredictor._plot_boxes(
        img, predictions, Config.COLORS
    )
    assert np.array_equal(annotated_img, img)

    display_image_for_test(annotated_img, "Empty Predictions Test")


def test_deepforest_predict_objects_default_parameters():
    """Test that default parameters work correctly with tiled prediction."""
    image_array = load_image_as_np_array(TEST_IMAGE_PATH_SMALL)
    if image_array is None:
        return

    summary, annotated_image, raw_predictions_json = (
        deepforest_predictor.predict_objects(
            image_data_array=image_array,
            model_names=["tree"]
        )
    )

    assert ("DeepForest detected" in summary or "No objects detected" in summary)
    assert annotated_image is not None
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape[:2] == image_array.shape[:2]

    predictions = json.loads(raw_predictions_json)
    assert isinstance(predictions, list)

    print("Default parameters test completed successfully")
    display_image_for_test(annotated_image, "Default Parameters Test")


def test_generate_detection_summary():
    """Test the _generate_detection_summary method directly."""
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    summary = deepforest_predictor._generate_detection_summary(empty_df)
    assert "No objects detected" in summary
    
    # Test with basic detections
    predictions_df = pd.DataFrame([
        {'label': 'tree', 'score': 0.9},
        {'label': 'tree', 'score': 0.8},
        {'label': 'bird', 'score': 0.7}
    ])
    summary = deepforest_predictor._generate_detection_summary(predictions_df)
    assert "DeepForest detected" in summary
    assert "2 trees" in summary
    assert "1 bird" in summary
    
    # Test with alive/dead tree classification
    alive_dead_df = pd.DataFrame([
        {'label': 'alive_tree', 'score': 0.9},
        {'label': 'dead_tree', 'score': 0.8},
        {'label': 'alive_tree', 'score': 0.7}
    ])
    summary = deepforest_predictor._generate_detection_summary(
        alive_dead_df, alive_dead_trees=True
    )
    assert "DeepForest detected" in summary
    assert "2 alive trees" in summary
    assert "1 dead tree" in summary
    
    print("Detection summary tests completed successfully")