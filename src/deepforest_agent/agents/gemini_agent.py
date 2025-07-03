import google.generativeai as genai
from dotenv import load_dotenv
import json
import base64
import numpy as np
from typing import Tuple, Optional, List
from deepforest_agent.utils.parameters_manager import DetectionParameters
from deepforest_agent.utils.file_manager import FileManager
from deepforest_agent.utils.api_error_handler import APIErrorHandler
from deepforest_agent.cache.detection_cache import CacheManager

import openai
from deepforest_agent.conf.config import Config
from deepforest_agent.tools.deepforest_tools import DeepForestPredictor
from deepforest_agent.utils.image_utils import encode_image_to_base64_url, load_image_as_np_array

load_dotenv()
genai.configure(api_key=Config.GOOGLE_API_KEY)

class GeminiAgent:
    """
    Intelligent ecological analysis agent combining conversational AI with computer vision.
    
    This refactored version eliminates code repetition through specialized helper classes
    while maintaining all original functionality and improving maintainability.
    """

    def __init__(self):
        """
        Initialize the GeminiAgent with OpenAI client configuration and prediction caching.
        
        The initialization sets up the hybrid AI system with intelligent caching capabilities
        for optimized performance during multi-turn conversations about ecological data.
        """
        self.client = openai.OpenAI(
            api_key=Config.GOOGLE_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            max_retries=5,
            timeout=60.0
        )
        self.model_name = "gemini-2.0-flash"
        self.messages = []
        self.deepforest_predictor = DeepForestPredictor()
        
        self.cache_manager = CacheManager()

    def _set_system_message(self) -> str:
        """
        Generate comprehensive system message defining the AI assistant's capabilities.
        
        Returns:
            Detailed system prompt for ecological analysis assistant
        """
        return """You are Computer Vision Ecological Analysis Assistant, a specialized AI agent designed to assist with ecological data analysis and object detection tasks. You are tasked to respond to the latest user prompt. You have to find the original image and use your image understanding skills to analyze the image and provide ecological insights that are relevant to the user's question. You can also use the DeepForest tool to perform object detection on the image, which will provide you with detailed information about detected objects, including their coordinates, confidence scores, and labels that can also help with spatial analysis of the image. The tool will also display the annotated image to the user, so whenever necessary for precise detection, run the DeepForest tool. You can combine both the computer vision analysis of the image and the precise detection data to provide comprehensive ecological insights and answer the user's original question. Use spatial analysis using the coordinates provided in the detection data and visualizing the image to answer the user question better. You can also look for the answer in the conversation history to answer the user's question. Give the response in three parts consisting of a direct answer to the user's question, a detailed explanation of your reasoning with supporting evidence from your vision analysis and detection data, and you might need to include the full detection data in an specific format if the user requests it. If the user asks for a specific format, you can format the detection data accordingly and present it in a markdown. But if user doesn't ask for detection data you don't need to provide it. Last part of your response should be a summary.
    """

    def _get_deepforest_tool_declaration(self) -> dict:
        """
        Generate OpenAI function declaration for DeepForest integration.
        
        Returns:
            Complete tool declaration with parameter specifications using class variables
        """
        return {
            "type": "function",
            "function": {
                "name": "deepforest_predict_objects",
                "description": (
                    "Performs object detection using DeepForest with full parameter control. "
                    "The function selects predict_tile detection method"
                    "ALL PARAMETERS ARE SUPPORTED - you can use any patch_size, overlap, threshold values requested by the user. "
                    "The image_data_array is automatically provided from the uploaded image - you never need to ask for it. "
                    "IMPORTANT: Only call this if the requested detection hasn't been performed yet for the current image."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_data_array": {
                            "type": "string", 
                            "description": "Base64 encoded image data (automatically provided from uploaded image)."
                        },
                        "model_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": f"List of DeepForest model names to use (e.g., [\"bird\", \"tree\"]). Defaults to {DetectionParameters.get_default_model_names()} if not specified."
                        },
                        "patch_size": {
                            "type": "integer",
                            "description": f"Patch size for detection windows. Any value is supported (default: {DetectionParameters.patch_size}). Larger values process bigger areas at once. Setting this will automatically use predict_tile method.",
                            "default": DetectionParameters.patch_size
                        },
                        "patch_overlap": {
                            "type": "number",
                            "description": f"Overlap ratio between patches (0.0-1.0). Higher values improve detection at patch boundaries. (default: {DetectionParameters.patch_overlap})",
                            "default": DetectionParameters.patch_overlap
                        },
                        "iou_threshold": {
                            "type": "number",
                            "description": f"IoU threshold for non-maximum suppression (0.0-1.0). Lower values remove more overlapping detections. (default: {DetectionParameters.iou_threshold})",
                            "default": DetectionParameters.iou_threshold
                        },
                        "thresh": {
                            "type": "number",
                            "description": f"Confidence threshold for detections (0.0-1.0). Lower values include more detections. (default: {DetectionParameters.thresh})",
                            "default": DetectionParameters.thresh
                        },
                        "alive_dead_trees": {
                            "type": "boolean",
                            "description": f"Enable alive/dead tree classification. Forces use of predict_tile method. (default: {DetectionParameters.alive_dead_trees})",
                            "default": DetectionParameters.alive_dead_trees
                        }
                    },
                    "required": ["image_data_array", "model_names"] 
                }
            }
        }

    def _handle_tool_call(self, message, image_data: np.ndarray, image_path: str) -> dict:
        """
        Orchestrate DeepForest tool execution with intelligent caching integration.
        
        This method bridges the AI conversation system with the specialized computer
        vision tools, managing the complex workflow of parameter extraction, cache
        validation, tool execution, and result integration.
        
        Args:
            message: OpenAI message containing tool call information
            image_data: Preprocessed image as numpy array
            image_path: Original image file path for cache management
            
        Returns:
            Formatted tool response for integration into conversation flow
        """
        tool_call = message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        
        # Extract file metadata using centralized manager
        image_hash = FileManager.validate_and_extract_info(image_path)
        
        # Create normalized parameters using centralized management
        params = DetectionParameters.from_arguments(arguments)
        
        # Use cache manager for decision logic
        should_run, reason = self.cache_manager.should_run_detection(image_hash, params)
        
        if should_run:
            print(f"Running detection: {reason}")
            
            deepforest_args = params.to_deepforest_args(image_data)
            summary_text, annotated_image_array, json_output = self.deepforest_predictor.predict_objects(**deepforest_args)
            
            print(f"Detection Summary: {summary_text}")
            print(f"JSON Output: {json_output}")

            # Update cache with results
            self.cache_manager.update_cache(image_hash, params, summary_text, annotated_image_array, json_output)
            
        else:
            print(f"Using cached results: {reason}")
            summary_text = self.cache_manager.get_detection_summary(params.model_names)
            json_output = self.cache_manager.cached_predictions["predictions_json_str"]
        
        response = {
            "role": "tool",
            "content": json.dumps({
                "summary": summary_text,
                "detections_json": json_output if json_output else "[]"
            }),
            "name": tool_call.function.name,
            "tool_call_id": tool_call.id
        }
        return response

    def _add_detection_context_to_messages(self, openai_messages: List[dict]) -> List[dict]:
        """
        Add FULL detection data context to conversation.
        """
        # Check if we have detection data available
        if (self.cache_manager.cached_predictions["predictions_json_str"] and 
            self.cache_manager.cached_predictions["predictions_json_str"] != "[]"):
            
            json_data = self.cache_manager.cached_predictions["predictions_json_str"]
            summary_text = self.cache_manager.cached_predictions["summary_text"]
            
            context_message = {
                "role": "system", 
                "content": (
                    f"FULL DETECTION DATA AVAILABLE:\n"
                    f"Summary: {summary_text}\n"
                    f"Complete JSON Detection Data: {json_data}\n"
                    f"You have access to ALL detection coordinates, confidence scores, and labels. "
                    f"You can reformat this complete data into any format the user requests "
                    f"(JSON, tables, CSV, etc.). Use this FULL dataset for any analysis."
                )
            }
            openai_messages.append(context_message)
        
        return openai_messages

    def _convert_gradio_to_openai_messages(self, gradio_history: List[dict], 
                                     current_prompt: str, image_path: str) -> List[dict]:
        """
        Convert Gradio conversation format to OpenAI-compatible message format.
        """
        openai_messages = []
        
        # Add detection context if models have been used
        if self.cache_manager.cached_predictions["models_detected"]:
            context_message = (
                f"DETECTION CONTEXT: For the current image, these object types have already been detected: "
                f"{', '.join(sorted(self.cache_manager.cached_predictions['models_detected']))}. "
                f"Consider using cached results if the user is asking about these objects."
            )
            openai_messages.append({"role": "system", "content": context_message})

        openai_messages = self._add_detection_context_to_messages(openai_messages)

        system_content = self._set_system_message()
        openai_messages.append({"role": "system", "content": system_content})
        
        # Convert conversation history and handle image for the latest user message
        for i, message in enumerate(gradio_history):
            if message["role"] == "user":
                content = message["content"]
                content = content.replace("\n[Image uploaded]", "")
                
                # If this is the last user message and we have an image, include it
                if i == len(gradio_history) - 1 and image_path:
                    try:
                        image_array = load_image_as_np_array(image_path)
                        data_url = encode_image_to_base64_url(image_array)
                        
                        if data_url:
                            content_blocks = [
                                {"type": "text", "text": content},
                                {"type": "image_url", "image_url": {"url": data_url, "detail": "auto"}}
                            ]
                            openai_messages.append({"role": "user", "content": content_blocks})
                        else:
                            print(f"Warning: Could not encode image {image_path} to base64")
                            openai_messages.append({"role": "user", "content": content})
                    except Exception as e:
                        print(f"Error encoding image file to base64: {e}")
                        openai_messages.append({"role": "user", "content": content})
                else:
                    openai_messages.append({"role": "user", "content": content})
                    
            elif message["role"] == "assistant":
                openai_messages.append({"role": "assistant", "content": message["content"]})
        
        return openai_messages

    def model_response(self, gradio_history: List[dict], user_prompt: str, 
                      image_path: str) -> Tuple[str, Optional[np.ndarray]]:
        """
        Generate comprehensive AI response with tool integration and caching.
        
        This is the main orchestration method that handles the complete workflow
        of conversation processing, tool integration, and response generation
        for ecological analysis tasks.
        
        Args:
            gradio_history: Previous conversation messages
            user_prompt: Current user input
            image_path: Path to uploaded image file
            
        Returns:
            Tuple of (response_text: str, annotated_image: Optional[np.ndarray])
        """
        # Handle image changes with centralized cache management
        current_hash = FileManager.validate_and_extract_info(image_path)
        if current_hash != self.cache_manager.cached_predictions["current_image_hash"]:
            self.cache_manager.clear_cache_for_new_image(current_hash)
        print(f"Gradio history: {gradio_history}")
        # Convert conversation format
        self.messages = self._convert_gradio_to_openai_messages(gradio_history, user_prompt, image_path)
        print(f"image path: {image_path}")
        # Prepare tools and response variables
        tools = [self._get_deepforest_tool_declaration()]
        annotated_image = None
        reply = ""
        print(f"Converted OpenAI messages: {self.messages}")
        # Execute primary API call
        def primary_api_call():
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                tools=tools,
                tool_choice="auto",
            )

        result = APIErrorHandler.handle_api_call(primary_api_call, "primary chat completion")
        
        # Handle API errors
        if isinstance(result, tuple):
            return result
        
        print(f"Primary API call result: {result}")

        response = result

        # Process tool calls if present
        if response.choices[0].finish_reason == "tool_calls":
            message = response.choices[0].message
            tool_response = self._handle_tool_call(message, load_image_as_np_array(image_path), image_path)
            
            self.messages.append(message)
            self.messages.append(tool_response)

            follow_up_prompt = (
                "The DeepForest tool has completed its analysis and you now have access to detailed detection data. "
                "Use both your computer vision analysis of the original image AND the detection data to provide "
                "comprehensive ecological insights and spatial analysis to answer the user's original question."
                "You can also look for the answer in the conversation history to answer the user's question."
                "Use spatial analysis using the coordinates provided in the detection data and visualizing"
                "the image to answer the user question better."
            )

            simulated_user_msg = [{"type": "text", "text": follow_up_prompt}]
            self.messages.append({"role": "user", "content": simulated_user_msg})
            print(f"Messages after tool call: {self.messages}")
            # Single follow-up call without annotated image attachment
            def reasoning_api_call():
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages
                )

            reasoning_result = APIErrorHandler.handle_api_call(reasoning_api_call, "reasoning response")
            print(f"Reasoning API call result: {reasoning_result}")
            if isinstance(reasoning_result, tuple):
                reply = "Analysis completed, but there was an error generating the explanation."
            else:
                reply = reasoning_result.choices[0].message.content or "Model returned no explanation."
            
            # Return annotated image if available
            annotated_image = self.cache_manager.cached_predictions.get("annotated_image_array")

        else:
            reply = response.choices[0].message.content or "Model returned no content."

        return reply, annotated_image