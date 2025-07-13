import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from deepforest_agent.conf.config import Config
from deepforest_agent.tools.deepforest_tool import DeepForestPredictor
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

    summary, annotated_image, detections_list = (
        deepforest_predictor.predict_objects(
            image_data_array=image_array,
            model_names=["bird"]
        )
    )

    assert "DeepForest detected" in summary or "No objects detected" in summary
    assert ("bird" in summary or "No objects detected" in summary)
    assert annotated_image is not None
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape[:2] == image_array.shape[:2]

    assert isinstance(detections_list, list)
    if detections_list:
        bird_labels_found = any(
            detection["label"] == "bird" for detection in detections_list if 'label' in detection
        )
        assert bird_labels_found

    display_image_for_test(annotated_image, "Bird Detection Test")


def test_deepforest_predict_objects_basic_detection_tree():
    """Test basic tree detection with default parameters on a small image."""
    image_array = load_image_as_np_array(TEST_IMAGE_PATH_SMALL)
    if image_array is None:
        return

    summary, annotated_image, detections_list = (
        deepforest_predictor.predict_objects(
            image_data_array=image_array,
            model_names=["tree"]
        )
    )

    assert "DeepForest detected" in summary or "No objects detected" in summary
    assert "tree" in summary or "No objects detected" in summary
    assert annotated_image is not None
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape[:2] == image_array.shape[:2]

    assert isinstance(detections_list, list)
    if detections_list:
        tree_labels_found = any(
            detection["label"] == "tree" for detection in detections_list if 'label' in detection
        )
        assert tree_labels_found

    display_image_for_test(annotated_image, "Tree Detection Test")


def test_deepforest_predict_objects_multiple_models():
    """Test detection using multiple models simultaneously."""
    image_array = load_image_as_np_array(TEST_IMAGE_PATH_SMALL)
    if image_array is None:
        return

    summary, annotated_image, detections_list = (
        deepforest_predictor.predict_objects(
            image_data_array=image_array,
            model_names=["bird", "tree", "livestock"]
        )
    )

    assert "DeepForest detected" in summary or "No objects detected" in summary
    assert annotated_image is not None
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape[:2] == image_array.shape[:2]

    assert isinstance(detections_list, list)
    if detections_list:
        labels = {detection['label'] for detection in detections_list if 'label' in detection}
        assert "bird" in labels or "tree" in labels or "livestock" in labels

    display_image_for_test(annotated_image, "Multiple Models Test")


def test_deepforest_predict_objects_large_image_processing():
    """Test processing of large images using tiled prediction."""
    summary, annotated_image, detections_list = (
        deepforest_predictor.predict_objects(
            image_file_path=TEST_IMAGE_PATH_LARGE,
            model_names=["tree"],
            patch_size=Config.DEEPFOREST_DEFAULTS["patch_size"],
            patch_overlap=Config.DEEPFOREST_DEFAULTS["patch_overlap"],
            iou_threshold=Config.DEEPFOREST_DEFAULTS["iou_threshold"],
            thresh=Config.DEEPFOREST_DEFAULTS["thresh"]
        )
    )

    assert "DeepForest detected" in summary or "No objects detected" in summary
    assert annotated_image is not None
    assert isinstance(annotated_image, np.ndarray)

    assert isinstance(detections_list, list)
    if detections_list:
        assert any(detection['label'] == 'tree' for detection in detections_list if 'label' in detection)

    display_image_for_test(annotated_image, "Large Image Processing Test")


def test_deepforest_predict_objects_custom_patch_size():
    """Test detection with custom patch size parameter."""
    image_array = load_image_as_np_array(TEST_IMAGE_PATH_SMALL)
    if image_array is None:
        return

    summary, annotated_image, detections_list = (
        deepforest_predictor.predict_objects(
            image_data_array=image_array,
            model_names=["tree"],
            patch_size=800,
            patch_overlap=Config.DEEPFOREST_DEFAULTS["patch_overlap"],
            iou_threshold=Config.DEEPFOREST_DEFAULTS["iou_threshold"],
            thresh=Config.DEEPFOREST_DEFAULTS["thresh"]
        )
    )

    assert "DeepForest detected" in summary or "No objects detected" in summary
    assert annotated_image is not None
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape[:2] == image_array.shape[:2]

    assert isinstance(detections_list, list)
    if detections_list:
        assert any(detection['label'] == 'tree' for detection in detections_list if 'label' in detection)

    display_image_for_test(annotated_image, "Custom Patch Size Test")


def test_deepforest_predict_objects_multiple_custom_parameters():
    """Test detection with multiple custom parameters."""
    image_array = load_image_as_np_array(TEST_IMAGE_PATH_SMALL)
    if image_array is None:
        return

    summary, annotated_image, detections_list = (
        deepforest_predictor.predict_objects(
            image_data_array=image_array,
            model_names=["tree"],
            patch_size=600,
            patch_overlap=0.1,
            iou_threshold=0.3,
            thresh=0.3
        )
    )

    assert "DeepForest detected" in summary or "No objects detected" in summary
    assert annotated_image is not None
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape[:2] == image_array.shape[:2]

    assert isinstance(detections_list, list)
    if detections_list:
        assert any(detection['label'] == 'tree' for detection in detections_list if 'label' in detection)

    display_image_for_test(annotated_image, "Multiple Custom Parameters Test")


def test_deepforest_predict_objects_alive_dead_trees():
    """Test alive/dead tree classification detection."""
    summary, annotated_image, detections_list = (
        deepforest_predictor.predict_objects(
            image_file_path=TEST_IMAGE_PATH_LARGE,
            model_names=["tree"],
            alive_dead_trees=True
        )
    )

    assert "DeepForest detected" in summary or "No objects detected" in summary
    assert annotated_image is not None
    assert isinstance(annotated_image, np.ndarray)

    print(summary)

    assert isinstance(detections_list, list)
    if detections_list:
        tree_detections = [d for d in detections_list if d.get('label') == 'tree']
        assert len(tree_detections) > 0, "Expected at least one tree detection"
        
        # Check for classification_label field in tree detections
        classification_labels = {d.get('classification_label') for d in tree_detections 
                               if 'classification_label' in d}
        assert ('alive_tree' in classification_labels or 'dead_tree' in classification_labels), \
               f"Expected alive_tree or dead_tree in classification labels, got: {classification_labels}"
        
        # Check that summary mentions classification results
        assert (("alive" in summary and "tree" in summary) or 
                ("dead" in summary and "tree" in summary) or 
                ("No objects detected" in summary)), \
               f"Summary should mention alive/dead classification: {summary}"

    display_image_for_test(annotated_image, "Alive/Dead Tree Detection Test")


def test_deepforest_predict_objects_no_detections():
    """Test the function gracefully handles cases with no detections."""
    blank_image = np.zeros((100, 100, 3), dtype=np.uint8)

    summary, annotated_image, detections_list = (
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

    assert isinstance(detections_list, list)
    assert len(detections_list) == 0

    display_image_for_test(annotated_image, "No Detections Test")


def test_deepforest_predict_objects_custom_thresholds():
    """Test detection with custom threshold parameters."""
    image_array = load_image_as_np_array(TEST_IMAGE_PATH_SMALL)
    if image_array is None:
        return

    summary, annotated_image, detections_list = (
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

    assert isinstance(detections_list, list)
    if detections_list:
        assert any(detection['label'] == 'tree' for detection in detections_list if 'label' in detection)

    display_image_for_test(annotated_image, "Custom Thresholds Test")


def test_deepforest_predict_objects_unsupported_model_name():
    """Test behavior with an unsupported model name."""
    image_array = load_image_as_np_array(TEST_IMAGE_PATH_SMALL)
    if image_array is None:
        return

    summary, annotated_image, detections_list = (
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

    assert isinstance(detections_list, list)
    if detections_list:
        labels = {detection['label'] for detection in detections_list if 'label' in detection}
        assert 'tree' in labels
        assert 'nonexistent_model' not in labels

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

    predictions = pd.DataFrame({
        "xmin": pd.Series(dtype=float),
        "ymin": pd.Series(dtype=float), 
        "xmax": pd.Series(dtype=float),
        "ymax": pd.Series(dtype=float),
        "label": pd.Series(dtype=str),
        "score": pd.Series(dtype=float)
    })

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

    summary, annotated_image, detections_list = (
        deepforest_predictor.predict_objects(
            image_data_array=image_array,
            model_names=["tree"]
        )
    )

    assert ("DeepForest detected" in summary or "No objects detected" in summary)
    assert annotated_image is not None
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape[:2] == image_array.shape[:2]

    assert isinstance(detections_list, list)

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
    
    print("Detection summary tests completed successfully")


def test_detections_list_structure():
    """Test that detections_list has the correct structure."""
    image_array = load_image_as_np_array(TEST_IMAGE_PATH_SMALL)
    if image_array is None:
        return

    summary, annotated_image, detections_list = (
        deepforest_predictor.predict_objects(
            image_data_array=image_array,
            model_names=["tree"]
        )
    )

    assert isinstance(detections_list, list)
    
    if detections_list:
        for detection in detections_list:
            assert isinstance(detection, dict)
            assert 'xmin' in detection
            assert 'ymin' in detection
            assert 'xmax' in detection
            assert 'ymax' in detection
            assert 'score' in detection
            assert 'label' in detection
            
            assert isinstance(detection['xmin'], int)
            assert isinstance(detection['ymin'], int)
            assert isinstance(detection['xmax'], int)
            assert isinstance(detection['ymax'], int)
            assert isinstance(detection['score'], float)
            assert isinstance(detection['label'], str)
    
    print("Detections list structure test completed successfully")


def test_error_handling_invalid_model():
    """Test error handling when all models are invalid."""
    image_array = load_image_as_np_array(TEST_IMAGE_PATH_SMALL)
    if image_array is None:
        return

    summary, annotated_image, detections_list = (
        deepforest_predictor.predict_objects(
            image_data_array=image_array,
            model_names=["invalid_model_1", "invalid_model_2"]
        )
    )

    assert "No objects detected" in summary
    assert annotated_image is not None
    assert isinstance(annotated_image, np.ndarray)
    assert isinstance(detections_list, list)
    assert len(detections_list) == 0
    
    print("Error handling test completed successfully")


def test_input_validation():
    """Test input validation for the predict_objects method."""
    # Test with neither image_data_array nor image_file_path provided
    try:
        deepforest_predictor.predict_objects(
            image_data_array=None,
            image_file_path=None,
            model_names=["tree"]
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Either image_data_array or image_file_path must be provided" in str(e)
    
    print("Input validation test completed successfully")