import json
from json import JSONDecoder
import re
from typing import Dict, Any, Optional, Union, List
import numpy as np
from PIL import Image

from deepforest_agent.tools.deepforest_tool import DeepForestPredictor
from deepforest_agent.utils.state_manager import session_state_manager
from deepforest_agent.utils.image_utils import validate_image_path
from deepforest_agent.conf.config import Config

deepforest_predictor = DeepForestPredictor()

def run_deepforest_object_detection(
    session_id: str,
    model_names: List[str] = ["tree", "bird", "livestock"],
    patch_size: int = Config.DEEPFOREST_DEFAULTS["patch_size"],
    patch_overlap: float = Config.DEEPFOREST_DEFAULTS["patch_overlap"],
    iou_threshold: float = Config.DEEPFOREST_DEFAULTS["iou_threshold"],
    thresh: float = Config.DEEPFOREST_DEFAULTS["thresh"],
    alive_dead_trees: bool = Config.DEEPFOREST_DEFAULTS["alive_dead_trees"]
) -> Dict[str, Any]:
    """
    Run DeepForest object detection on the globally stored image.

    Args:
        session_id (str): Unique session identifier for this user
        model_names: List of model names to use ("tree", "bird", "livestock")
        patch_size: Patch size for each window in pixels (not geographic units). The size for the crops used to cut the input image/raster into smaller pieces.
        patch_overlap: Patch overlap among windows. The horizontal and vertical overlap among patches (must be between 0-1).
        iou_threshold: Minimum IoU overlap among predictions between windows to be suppressed.
        thresh: Score threshold used to filter bboxes after soft-NMS is performed.
        alive_dead_trees: Whether to classify trees as alive/dead

    Returns:
        Dictionary with detection_summary and detections_list
    """
    # Validate session exists
    if not session_state_manager.session_exists(session_id):
        return {
            "detection_summary": f"Session {session_id} not found.",
            "detections_list": [],
            "status": "error"
        }

    image_file_path = session_state_manager.get(session_id, "image_file_path")
    current_image = session_state_manager.get(session_id, "current_image")
    
    if image_file_path is None and current_image is None:
        return {
            "detection_summary": f"No image available for detection in session {session_id}.",
            "detections_list": [],
            "status": "error"
        }

    if image_file_path and not validate_image_path(image_file_path):
        print(f"Warning: Invalid image file path {image_file_path}, falling back to PIL image")
        image_file_path = None
    
    try:
        if image_file_path:
            print(f"DeepForest: Processing image from file path: {image_file_path}")
            detection_summary, annotated_image, detections_list = deepforest_predictor.predict_objects(
                image_file_path=image_file_path,
                model_names=model_names,
                patch_size=patch_size,
                patch_overlap=patch_overlap,
                iou_threshold=iou_threshold,
                thresh=thresh,
                alive_dead_trees=alive_dead_trees
            )
        else:
            print(f"DeepForest: Processing PIL image (size: {current_image.size})")
            image_array = np.array(current_image)
            detection_summary, annotated_image, detections_list = deepforest_predictor.predict_objects(
                image_data_array=image_array,
                model_names=model_names,
                patch_size=patch_size,
                patch_overlap=patch_overlap,
                iou_threshold=iou_threshold,
                thresh=thresh,
                alive_dead_trees=alive_dead_trees
            )
        
        if annotated_image is not None:
            session_state_manager.set(session_id, "annotated_image", Image.fromarray(annotated_image))
        
        result = {
            "detection_summary": detection_summary,
            "detections_list": detections_list,
            "total_detections": len(detections_list),
            "status": "success"
        }
        
        return result
        
    except Exception as e:
        error_msg = f"Error during image detection in session {session_id}: {str(e)}"
        print(f"DeepForest Detection Error: {error_msg}")
        return {
            "detection_summary": error_msg,
            "detections_list": [],
            "total_detections": 0,
            "status": "error"
        }

def extract_all_tool_calls(text: str) -> List[Dict[str, Any]]:
    """
    Extract all tool call information from model output text.
    
    Args:
        text: The model's output text that may contain multiple tool calls
        
    Returns:
        List of dictionaries with tool call info (empty list if none found)
    """
    tool_calls = []

    # Method 1: Wrapped in XML
    xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    xml_matches = re.findall(xml_pattern, text, re.DOTALL)
    
    for match in xml_matches:
        try:
            result = json.loads(match.strip())
            if isinstance(result, dict) and "name" in result and "arguments" in result:
                print(f"Found valid XML tool call: {result}")
                tool_calls.append(result)
        except json.JSONDecodeError as e:
            print(f"Failed to parse XML tool call JSON: {e}")
            continue
    
    # Method 2: If no XML format found, try raw JSON format
    if not tool_calls:
        decoder = JSONDecoder()
        brace_start = 0
        
        while True:
            match = text.find('{', brace_start)
            if match == -1:
                break
            try:
                result, index = decoder.raw_decode(text[match:])
                if isinstance(result, dict) and "name" in result and "arguments" in result:
                    print(f"Found valid raw JSON tool call: {result}")
                    tool_calls.append(result)
                    brace_start = match + index
                else:
                    brace_start = match + 1
            except ValueError:
                brace_start = match + 1
    
    print(f"Total tool calls extracted: {len(tool_calls)}")
    return tool_calls

def handle_tool_call(tool_name: str, tool_arguments: Dict[str, Any], session_id: str) -> Union[str, Dict[str, Any]]:
    """
    Handle tool call execution from tool name and arguments.
    
    Args:
        tool_name (str): The name of the tool to be executed.
        tool_arguments (Dict[str, Any]): A dictionary of arguments for the tool.
        session_id: Unique session identifier for this user
        
    Returns:
        Either error message (str) or tool execution result (dict)
    """
    print(f"Tool Call Detected:")
    print(f"Tool Name: {tool_name}")
    print(f"Arguments: {tool_arguments}")
    
    if tool_name == "run_deepforest_object_detection":
        try:
            result = run_deepforest_object_detection(session_id=session_id, **tool_arguments)
            
            return result
            
        except Exception as e:
            error_msg = f"Error executing {tool_name} in session {session_id}: {str(e)}"
            print(f"Tool Execution Failed: {error_msg}")
            
            return error_msg
    else:
        error_msg = f"Unknown tool: {tool_name}"
        print(f"Unknown Tool: {error_msg}")
        
        return error_msg
