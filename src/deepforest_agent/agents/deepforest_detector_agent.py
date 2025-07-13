from typing import Dict, List, Any, Optional
import json
import re
import time

from deepforest_agent.utils.cache_utils import tool_call_cache
from deepforest_agent.models.smollm3_3b import SmolLM3ModelManager
from deepforest_agent.tools.tool_handler import handle_tool_call, extract_all_tool_calls
from deepforest_agent.conf.config import Config
from deepforest_agent.prompts.prompt_templates import create_detector_system_prompt_with_reasoning, get_deepforest_tool_schema
from deepforest_agent.utils.state_manager import session_state_manager
from deepforest_agent.utils.logging_utils import multi_agent_logger
from deepforest_agent.utils.parsing_utils import parse_deepforest_agent_response_with_reasoning
from deepforest_agent.utils.rtree_spatial_utils import DetectionSpatialAnalyzer
from deepforest_agent.utils.detection_narrative_generator import DetectionNarrativeGenerator



class DeepForestDetectorAgent:
    """
    DeepForest detector agent responsible for executing object detection.
    Uses SmolLM3-3B model for tool calling.
    """
    
    def __init__(self):
        """Initialize the DeepForest Detector Agent."""
        self.agent_config = Config.AGENT_CONFIGS["deepforest_detector"]
        self.model_manager = SmolLM3ModelManager(Config.AGENT_MODELS["deepforest_detector"])

    def _filter_models_based_on_visual(self, visual_objects: List[str], original_models: List[str]):
        """
        Filter original model names based on visual agent's detected objects.
        Remove models that weren't visually detected.
        
        Args:
            visual_objects (List[str]): Objects detected by visual agent
            original_models (List[str]): Original model list from tool call
        """
        pass
    
    def execute_detection_with_context(
        self, 
        user_message: str, 
        session_id: str,
        visual_objects_detected: List[str],
        memory_context: str
    ) -> Dict[str, Any]:
        """
        Execute DeepForest detection with R-tree spatial analysis and narrative generation.
        
        Args:
            user_message (str): User's query
            session_id (str): Unique session identifier for this user
            visual_objects_detected (List[str]): Objects detected by visual agent
            memory_context (str): Context from memory agent
            
        Returns:
            Dictionary with detection results, R-tree analysis, and narrative
        """
        # Validate session exists
        if not session_state_manager.session_exists(session_id):
            return {
                "detection_summary": f"Session {session_id} not found.",
                "detections_list": [],
                "total_detections": 0,
                "status": "error",
                "error": f"Session {session_id} not found",
                "detection_narrative": "No detection narrative available due to session error."
            }

        try:
            tool_generation_start = time.perf_counter()

            system_prompt = create_detector_system_prompt_with_reasoning(
                user_message, memory_context, visual_objects_detected
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]

            deepforest_tool_schema = get_deepforest_tool_schema()

            response = self.model_manager.generate_response(
                messages=messages,
                max_new_tokens=self.agent_config["max_new_tokens"],
                temperature=self.agent_config["temperature"],
                top_p=self.agent_config["top_p"],
                tools=[deepforest_tool_schema]
            )

            tool_generation_time = time.perf_counter() - tool_generation_start

            print(f"Session {session_id} - Detector Raw Response: {response}")

            multi_agent_logger.log_agent_execution(
                session_id=session_id,
                agent_name="detector",
                agent_input=f"User: {user_message}",
                agent_output=response,
                execution_time=tool_generation_time
            )

            parsed_response = self._parse_response_with_reasoning(response)
            
            if "error" in parsed_response:
                multi_agent_logger.log_error(
                    session_id=session_id,
                    error_type="tool_call_parsing_error",
                    error_message=parsed_response["error"]
                )

                return {
                    "detection_summary": f"Tool call parsing failed: {parsed_response['error']}",
                    "detections_list": [],
                    "total_detections": 0,
                    "status": "error",
                    "error": parsed_response["error"],
                    "detection_narrative": "No detection narrative available due to parsing error."
                }
            
            reasoning = parsed_response["reasoning"]
            tool_calls = parsed_response["tool_calls"]

            print(f"Session {session_id} - Reasoning: {reasoning}")
            print(f"Session {session_id} - Found {len(tool_calls)} tool calls")

            all_results = []
            combined_detection_summary = []
            combined_detections_list = []
            total_detections = 0
            
            for i, tool_call in enumerate(tool_calls):
                print(f"Session {session_id} - Executing tool call {i+1}/{len(tool_calls)}")
                
                tool_name = tool_call["name"]
                tool_arguments = tool_call["arguments"]

                cached_result = tool_call_cache.get_cached_result(tool_name, tool_arguments)
                
                if cached_result:
                    print(f"Session {session_id} - Tool call {i+1}: Using cached results")

                    if cached_result.get("annotated_image"):
                        session_state_manager.set(session_id, "annotated_image", cached_result["annotated_image"])
                    
                    cache_key = cached_result["cache_info"]["cache_key"]
                    session_state_manager.add_tool_call_to_history(
                        session_id, tool_name, tool_arguments, cache_key
                    )

                    multi_agent_logger.log_tool_call(
                        session_id=session_id,
                        tool_name=tool_name,
                        tool_arguments=tool_arguments,
                        tool_result=cached_result,
                        execution_time=0.0,
                        cache_hit=True,
                        reasoning=f"Tool call {i+1}: {reasoning}"
                    )

                    tool_result = {
                        "tool_call_number": i + 1,
                        "tool_name": tool_name,
                        "tool_arguments": tool_arguments,
                        "cache_key": cache_key,
                        "detection_summary": cached_result["detection_summary"],
                        "detections_list": cached_result.get("detections_list", []),
                        "total_detections": len(cached_result.get("detections_list", [])),
                        "status": "success",
                        "cache_hit": True
                    }
                    
                    all_results.append(tool_result)
                    combined_detection_summary.append(cached_result["detection_summary"])
                    combined_detections_list.extend(cached_result.get("detections_list", []))
                    total_detections += len(cached_result.get("detections_list", []))
                    
                else:
                    print(f"Session {session_id} - Tool call {i+1}: Cache MISS, executing tool")

                    tool_execution_start = time.perf_counter()
                    execution_result = handle_tool_call(tool_name, tool_arguments, session_id)

                    tool_execution_time = time.perf_counter() - tool_execution_start
                    
                    if isinstance(execution_result, dict) and "detection_summary" in execution_result:
                        cache_result = {
                            "detection_summary": execution_result["detection_summary"],
                            "detections_list": execution_result.get("detections_list", []),
                            "total_detections": execution_result.get("total_detections", 0),
                            "status": "success"
                        }
                    
                        annotated_image = session_state_manager.get(session_id, "annotated_image")
                        if annotated_image:
                            cache_result["annotated_image"] = annotated_image
                        
                        cache_key = tool_call_cache.store_result(tool_name, tool_arguments, cache_result)
                        
                        session_state_manager.add_tool_call_to_history(
                            session_id, tool_name, tool_arguments, cache_key
                        )

                        multi_agent_logger.log_tool_call(
                            session_id=session_id,
                            tool_name=tool_name,
                            tool_arguments=tool_arguments,
                            tool_result=execution_result,
                            execution_time=tool_execution_time,
                            cache_hit=False,
                            reasoning=f"Tool call {i+1}: {reasoning}"
                        )

                        tool_result = {
                            "tool_call_number": i + 1,
                            "tool_name": tool_name,
                            "tool_arguments": tool_arguments,
                            "cache_key": cache_key,
                            "detection_summary": execution_result["detection_summary"],
                            "detections_list": execution_result.get("detections_list", []),
                            "total_detections": execution_result.get("total_detections", 0),
                            "status": "success",
                            "cache_hit": False
                        }
                        
                        all_results.append(tool_result)
                        combined_detection_summary.append(execution_result["detection_summary"])
                        combined_detections_list.extend(execution_result.get("detections_list", []))
                        total_detections += execution_result.get("total_detections", 0)
                        
                    else:
                        error_msg = str(execution_result) if isinstance(execution_result, str) else "Unknown tool execution error"
                        print(f"Session {session_id} - Tool call {i+1} execution failed: {error_msg}")
                        
                        multi_agent_logger.log_error(
                            session_id=session_id,
                            error_type="tool_execution_error",
                            error_message=f"Tool call {i+1} execution failed after {tool_execution_time:.2f}s: {error_msg}"
                        )

                        tool_result = {
                            "tool_call_number": i + 1,
                            "tool_name": tool_name,
                            "tool_arguments": tool_arguments,
                            "detection_summary": f"Tool call {i+1} failed: {error_msg}",
                            "detections_list": [],
                            "total_detections": 0,
                            "status": "error",
                            "error": error_msg,
                            "cache_hit": False
                        }
                        all_results.append(tool_result)

            final_detection_summary = " | ".join(combined_detection_summary) if combined_detection_summary else "No successful detections"
            
            # Generate comprehensive R-tree based narrative
            detection_narrative = self._generate_spatial_narrative(
                combined_detections_list, session_id
            )
            
            # Log the detection narrative
            multi_agent_logger.log_agent_execution(
                session_id=session_id,
                agent_name="detection_narrative",
                agent_input=f"Detection narrative for {len(combined_detections_list)} detections",
                agent_output=detection_narrative,
                execution_time=0.0
            )
            
            result = {
                "detection_summary": final_detection_summary,
                "detections_list": combined_detections_list,
                "total_detections": total_detections,
                "status": "success",
                "reasoning": reasoning,
                "visual_objects_input": visual_objects_detected,
                "tool_calls_executed": len(tool_calls),
                "tool_results": all_results,
                "detection_narrative": detection_narrative,
                "raw_tool_response": response
            }
            
            print(f"Session {session_id} - Executed {len(tool_calls)} tool calls successfully")
            print(f"Session {session_id} - Generated detection narrative ({len(detection_narrative)} characters)")
            return result
                    
        except Exception as e:
            error_msg = f"Error in detector agent for session {session_id}: {str(e)}"
            print(f"Detector Agent Error: {error_msg}")
            
            multi_agent_logger.log_error(
                session_id=session_id,
                error_type="detector_agent_exception",
                error_message=error_msg
            )

            return {
                "detection_summary": f"Detection agent error: {error_msg}",
                "detections_list": [],
                "total_detections": 0,
                "status": "error",
                "error": error_msg,
                "visual_objects_input": visual_objects_detected,
                "detection_narrative": f"Detection narrative generation failed due to error: {error_msg}"
            }

    def _generate_spatial_narrative(self, detections_list: List[Dict[str, Any]], session_id: str) -> str:
        """
        Generate comprehensive spatial narrative using R-tree analysis.
        
        Args:
            detections_list: Combined list of all detections
            session_id: Session identifier for getting image dimensions
            
        Returns:
            Comprehensive detection narrative
        """
        if not detections_list:
            return "No detections available for spatial narrative generation."
        
        try:
            # Get image dimensions
            current_image = session_state_manager.get(session_id, "current_image")
            if current_image:
                image_width, image_height = current_image.size
            else:
                # Default dimensions if image not available
                image_width, image_height = 1920, 1080
            
            # Generate narrative using DetectionNarrativeGenerator
            narrative_generator = DetectionNarrativeGenerator(image_width, image_height)
            comprehensive_narrative = narrative_generator.generate_comprehensive_narrative(detections_list)
            
            print(f"Session {session_id} - Generated comprehensive spatial narrative")
            
            return comprehensive_narrative
            
        except Exception as e:
            error_msg = f"Error generating spatial narrative: {str(e)}"
            print(f"Session {session_id} - {error_msg}")
            
            # Just return the detection summary itself
            total_count = len(detections_list)
            label_counts = {}
            classification_counts = {}
            
            for detection in detections_list:
                base_label = detection.get('label', 'unknown')
                label_counts[base_label] = label_counts.get(base_label, 0) + 1
                
                # Handle tree classifications
                if base_label == 'tree':
                    classification_label = detection.get('classification_label')
                    classification_score = detection.get('classification_score')
                    
                    # Only count valid classifications (not NaN)
                    if (classification_label and 
                        classification_score is not None and
                        str(classification_label).lower() != 'nan' and
                        str(classification_score).lower() != 'nan'):
                        
                        classification_counts[classification_label] = classification_counts.get(classification_label, 0) + 1
            
            # Build simple summary
            object_parts = []
            for label, count in label_counts.items():
                if label == 'tree' and classification_counts:
                    # Special handling for trees with classifications
                    total_trees = count
                    tree_part = f"{total_trees} trees are detected"
                    
                    if classification_counts:
                        classification_parts = []
                        for class_label, class_count in classification_counts.items():
                            class_name = class_label.replace('_', ' ')
                            classification_parts.append(f"{class_count} {class_name}s")
                        
                        tree_part += f". These {total_trees} trees are classified as {' and '.join(classification_parts)}"
                    
                    object_parts.append(tree_part)
                else:
                    label_name = label.replace('_', ' ')
                    object_parts.append(f"{count} {label_name}{'s' if count != 1 else ''}")
            
            fallback_summary = f"DeepForest detected {total_count} objects: {', '.join(object_parts)}."
            
            return fallback_summary

    def _parse_response_with_reasoning(self, response: str) -> Dict[str, Any]:
        """
        Parse model response to extract reasoning and multiple tool calls.
        
        Args:
            response (str): Raw response from the model
            
        Returns:
            Dictionary containing either:
            - {"reasoning": str, "tool_call": dict} on success
            - {"error": str} on parsing failure
        """
        return parse_deepforest_agent_response_with_reasoning(response)