import time
import json
import torch
import gc
from typing import Dict, List, Any, Optional, Generator

from deepforest_agent.agents.memory_agent import MemoryAgent
from deepforest_agent.agents.deepforest_detector_agent import DeepForestDetectorAgent
from deepforest_agent.agents.visual_analysis_agent import VisualAnalysisAgent
from deepforest_agent.agents.ecology_analysis_agent import EcologyAnalysisAgent
from deepforest_agent.utils.state_manager import session_state_manager
from deepforest_agent.utils.cache_utils import tool_call_cache
from deepforest_agent.utils.image_utils import check_image_resolution_for_deepforest
from deepforest_agent.utils.logging_utils import multi_agent_logger
from deepforest_agent.utils.detection_narrative_generator import DetectionNarrativeGenerator


class AgentOrchestrator:
    """
    Orchestrates the multi-agent workflow with memory context + visual contexts + DeepForest detection context + ecological synthesis.
    """
    
    def __init__(self):
        """Initialize the Agent Orchestrator."""
        self.memory_agent = MemoryAgent()
        self.detector_agent = DeepForestDetectorAgent()
        self.visual_agent = VisualAnalysisAgent()
        self.ecology_agent = EcologyAnalysisAgent()

        self.execution_stats = {
            "total_runs": 0,
            "successful_runs": 0,
            "average_execution_time": 0.0,
            "memory_direct_answers": 0,
            "deepforest_skipped": 0
        }

    def _log_gpu_memory(self, session_id: str, stage: str, agent_name: str):
        """
        Log current GPU memory usage.
        
        Args:
            session_id (str): Unique identifier for the user session being processed
            stage (str): Workflow stage identifier (e.g., "before", "after", "cleanup")
            agent_name (str): Name of the agent being monitored (e.g., "Visual Analysis", 
                            "DeepForest Detection", "Memory Agent")
        """
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / 1024**3
            cached_gb = torch.cuda.memory_reserved() / 1024**3
            
            multi_agent_logger.log_agent_execution(
                session_id=session_id,
                agent_name=f"gpu_memory_{stage}",
                agent_input=f"{agent_name} - {stage}",
                agent_output=f"GPU Memory - Allocated: {allocated_gb:.2f} GB, Cached: {cached_gb:.2f} GB",
                execution_time=0.0
            )
            print(f"Session {session_id} - {agent_name} {stage}: GPU Memory - Allocated: {allocated_gb:.2f} GB, Cached: {cached_gb:.2f} GB")

    def cleanup_all_agents(self):
        """Cleanup models to manage memory."""
        print("Orchestrator cleanup:")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            print(f"Final GPU memory after orchestrator cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    def _aggressive_gpu_cleanup(self, session_id: str, stage: str):
        """
        Perform aggressive GPU memory cleanup.
        
        Args:
            session_id (str): Unique identifier for the user session
            stage (str): Workflow stage identifier for logging context
        """
        if torch.cuda.is_available():
            for i in range(3):
                gc.collect()
                torch.cuda.empty_cache()
            
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
            except:
                pass
            
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            
            print(f"Session {session_id} - {stage} aggressive cleanup: {allocated:.2f} GB allocated, {cached:.2f} GB cached")

    def _format_detection_data_for_monitor(self, detection_narrative: str, detections_list: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Format detection data for monitor display.
        
        Args:
            detection_narrative: Generated detection context from DeepForest Data
            detections_list: Full DeepForest detection data
            
        Returns:
            Formatted detection data for monitor
        """
        monitor_parts = []
        
        if detections_list:
            monitor_parts.append("=== DEEPFOREST DETECTIONS ===")
            monitor_parts.append(json.dumps(detections_list, indent=2))
            monitor_parts.append("")
        
        if detection_narrative:
            monitor_parts.append("=== DETECTION NARRATIVE ===")
            monitor_parts.append(detection_narrative)

        return "\n".join(monitor_parts) if monitor_parts else "No detection data available"

    def _get_cached_detection_narrative(self, tool_cache_id: str) -> Optional[str]:
        """
        Retrieve detection narrative using tool cache ID from the tool_call_cache.
        
        Args:
            tool_cache_id: Tool cache identifier
            
        Returns:
            Detection context from DeepForest Data if found, None otherwise
        """
        try:
            print(f"Looking up cached detection narrative for tool_cache_id: {tool_cache_id}")
            
            # Handle multiple cache IDs
            cache_ids = [id.strip() for id in tool_cache_id.split(",")] if tool_cache_id else []
            all_narratives = []
            
            for cache_id in cache_ids:
                if cache_id in tool_call_cache.cache_data:
                    cached_entry = tool_call_cache.cache_data[cache_id]
                    cached_result = cached_entry.get("result", {})
                    tool_name = cached_entry.get("tool_name", "unknown")
                    tool_arguments = cached_entry.get("arguments", {})
                    
                    # Get all possible arguments including defaults from Config
                    from deepforest_agent.conf.config import Config
                    all_arguments = Config.DEEPFOREST_DEFAULTS.copy()
                    all_arguments.update(tool_arguments)
                    
                    # Format tool call info with all arguments
                    args_str = ", ".join([f"{k}={v}" for k, v in all_arguments.items()])
                    
                    # Check if we have detections_list to generate narrative from
                    detections_list = cached_result.get("detections_list", [])
                    
                    if detections_list:
                        print(f"Found {len(detections_list)} cached detections for cache ID {cache_id}")
                        
                        # Get image dimensions for narrative generation
                        try:
                            session_keys = list(session_state_manager._sessions.keys())
                            if session_keys:
                                current_image = session_state_manager.get(session_keys[0], "current_image")
                                if current_image:
                                    image_width, image_height = current_image.size
                                else:
                                    image_width, image_height = 0, 0
                            else:
                                image_width, image_height = 0, 0
                        except:
                            image_width, image_height = 0, 0
                        
                        # Generate fresh narrative from cached detection data
                        narrative_generator = DetectionNarrativeGenerator(image_width, image_height)
                        cached_detection_narrative = narrative_generator.generate_comprehensive_narrative(detections_list)
                        
                        # Format with proper tool cache ID structure
                        formatted_narrative = f"**TOOL CACHE ID:** {cache_id}\nDeepForest tool run with arguments ({args_str}) and got the below narratives:\nDETECTION NARRATIVE:\n{cached_detection_narrative}"
                        all_narratives.append(formatted_narrative)
                    else:
                        detection_summary = cached_result.get("detection_summary", "")
                        if detection_summary:
                            formatted_summary = f"**TOOL CACHE ID:** {cache_id}\nDeepForest tool run with arguments ({args_str}) and got the below narratives:\nDETECTION NARRATIVE:\n{detection_summary}"
                            all_narratives.append(formatted_summary)
            
            if all_narratives:
                print(f"Generated {len(all_narratives)} cached detection narratives")
                return "\n\n".join(all_narratives)
            
            print(f"No cached data found for tool_cache_id(s): {tool_cache_id}")
            return None
            
        except Exception as e:
            print(f"Error retrieving cached detection narrative for {tool_cache_id}: {e}")
            return None
    
    def process_user_message_streaming(
        self, 
        user_message: str, 
        conversation_history: List[Dict[str, Any]],
        session_id: str
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Orchestrate the multi-agent workflow with memory context and detection narrative flow.
        
        Args:
            user_message: Current user message/query to be processed
            conversation_history: Full conversation history 
            session_id: Unique session identifier for this user's workflow
            
        Yields:
            Dict[str, Any]: Progress updates during processing
        """
        start_time = time.perf_counter()
        self.execution_stats["total_runs"] += 1

        print(f"Session {session_id} - Query: {user_message}")
        print(f"Session {session_id} - Conversation history length: {len(conversation_history)}")
        
        agent_results = {}
        execution_summary = {
            "agents_executed": [],
            "execution_order": [],
            "timings": {},
            "status": "in_progress",
            "session_id": session_id,
            "workflow_type": "memory_narrative_flow",
            "memory_provided_direct_answer": False,
            "deepforest_executed": False
        }

        memory_context = ""
        visual_context = ""
        detection_narrative = ""
        memory_tool_cache_id = None
        current_tool_cache_id = None
        
        try:
            if not session_state_manager.session_exists(session_id):
                raise ValueError(f"Session {session_id} not found")
            
            session_state_manager.set_processing_state(session_id, True)
            session_state_manager.reset_cancellation(session_id)
            
            yield {
                "stage": "memory", 
                "message": "Analyzing conversation memory and context...", 
                "type": "progress"
            }
            
            if session_state_manager.is_cancelled(session_id):
                raise Exception("Processing cancelled by user")
            
            print(f"\nSTEP 1: Memory Agent Processing (Session {session_id})")
            self._log_gpu_memory(session_id, "before", "Memory Agent")
            memory_start = time.perf_counter()
            
            memory_result = self.memory_agent.process_conversation_history_structured(
                conversation_history=conversation_history,
                latest_message=user_message,
                session_id=session_id
            )
            
            memory_time = time.perf_counter() - memory_start
            self._log_gpu_memory(session_id, "after", "Memory Agent")
            self._aggressive_gpu_cleanup(session_id, "after_memory_agent")
            execution_summary["timings"]["memory_agent"] = memory_time
            execution_summary["agents_executed"].append("memory")
            execution_summary["execution_order"].append("memory")
            agent_results["memory"] = memory_result
            
            # Extract memory context and tool cache ID
            memory_context = memory_result.get("relevant_context", "No memory context available")
            tool_cache_id = memory_result.get("tool_cache_id")
            
            print(f"Session {session_id} - Memory Agent: Completed in {memory_time:.2f}s")
            print(f"Session {session_id} - Memory Has Answer: {memory_result['answer_present']}")
            print(f"Session {session_id} - Tool Cache ID: {tool_cache_id}")
            
            if memory_result["answer_present"]:
                print(f"Session {session_id} - Memory has direct answer - using cached data for synthesis")
                
                self.execution_stats["memory_direct_answers"] += 1
                execution_summary["memory_provided_direct_answer"] = True
                
                # Get cached detection narrative if available
                cached_detection_narrative = ""
                if tool_cache_id:
                    cached_detection_narrative = self._get_cached_detection_narrative(tool_cache_id) or ""

                yield {
                    "stage": "ecology", 
                    "message": "Using memory context and cached detection narrative for synthesis...", 
                    "type": "progress"
                }
                
                if session_state_manager.is_cancelled(session_id):
                    raise Exception("Processing cancelled by user")
                
                print(f"\nSTEP 2 (MEMORY PATH): Ecology Agent with Memory Context (Session {session_id})")
                self._log_gpu_memory(session_id, "before", "Ecology Agent (Memory Path)")
                ecology_start = time.perf_counter()
                
                # Prepare comprehensive context
                comprehensive_context = self._prepare_comprehensive_context(
                    memory_context=memory_context,
                    visual_context="",
                    detection_narrative=cached_detection_narrative,
                    tool_cache_id=tool_cache_id
                )
                
                final_response = ""
                for token_result in self.ecology_agent.synthesize_analysis_streaming(
                    user_message=user_message,
                    memory_context=comprehensive_context,
                    cached_json=None,
                    current_json=None,
                    session_id=session_id
                ):

                    if session_state_manager.is_cancelled(session_id):
                        raise Exception("Processing cancelled by user")
                        
                    final_response += token_result["token"]

                    yield {
                        "stage": "ecology_streaming",
                        "message": final_response,
                        "type": "streaming",
                        "is_complete": token_result["is_complete"]
                    }

                    if token_result["is_complete"]:
                        ecology_time = time.perf_counter() - ecology_start
                        self._log_gpu_memory(session_id, "after", "Ecology Agent (Memory Path)")
                        execution_summary["timings"]["ecology_agent"] = ecology_time
                        execution_summary["agents_executed"].append("ecology")
                        execution_summary["execution_order"].append("ecology")
                        agent_results["ecology"] = {"final_response": final_response}
                        print(f"Session {session_id} - Ecology (Memory Path): Completed in {ecology_time:.2f}s")
                        break

                total_time = time.perf_counter() - start_time
                execution_summary["timings"]["total"] = total_time
                execution_summary["status"] = "completed_via_memory"

                detection_data_monitor = self._format_detection_data_for_monitor(
                    detection_narrative=cached_detection_narrative
                )

                yield {
                    "stage": "complete",
                    "message": final_response,
                    "type": "final",
                    "detection_data": detection_data_monitor,
                    "agent_results": agent_results,
                    "execution_summary": execution_summary,
                    "execution_time": total_time,
                    "status": "success"
                }
                return
            else:
                for result in self._execute_full_pipeline_with_narrative_flow(
                    user_message=user_message,
                    conversation_history=conversation_history,
                    session_id=session_id,
                    memory_context=memory_context,
                    memory_tool_cache_id=memory_result.get("tool_cache_id"),
                    start_time=start_time
                ):
                    yield result
                    if result["type"] == "final":
                        return
        
        except Exception as e:
            error_msg = f"Orchestrator error (Session {session_id}): {str(e)}"
            print(f"ORCHESTRATOR ERROR: {error_msg}")

            try:
                self._aggressive_gpu_cleanup(session_id, "emergency")
            except Exception as cleanup_error:
                print(f"Emergency cleanup error: {cleanup_error}")

            partial_time = time.perf_counter() - start_time
            execution_summary["timings"]["total"] = partial_time
            execution_summary["status"] = "error"
            execution_summary["error"] = error_msg

            fallback_response = self._create_fallback_response(
                user_message=user_message,
                agent_results=agent_results,
                error=error_msg,
                session_id=session_id
            )

            yield {
                "stage": "error",
                "message": fallback_response,
                "type": "final",
                "detection_data": "Error occurred - no detection data available",
                "agent_results": agent_results,
                "execution_summary": execution_summary,
                "execution_time": partial_time,
                "status": "error",
                "error": error_msg
            }
        
        finally:
            session_state_manager.set_processing_state(session_id, False)

    def _execute_full_pipeline_with_narrative_flow(
        self,
        user_message: str,
        conversation_history: List[Dict[str, Any]],
        session_id: str,
        memory_context: str,
        memory_tool_cache_id: Optional[str],
        start_time: float
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Execute the complete pipeline using memory context, visual contexts, and detection narratives.

        Args:
            user_message: Current user query
            conversation_history: Complete conversation context
            session_id: Unique session identifier
            memory_context: Context from memory agent
            memory_tool_cache_id (Optional[str]): Cache identifier from memory agent
            start_time: Start time for total execution calculation
            
        Yields:
            Dict[str, Any]: Progress updates during processing containing:
                - stage (str): Current workflow stage ("visual_analysis", "detector", etc.)
                - message (str): Human-readable progress message
                - type (str): Update type ("progress", "streaming", "final")
                - Additional stage-specific data (detection_data, agent_results, etc.)
        """
        agent_results = {}
        execution_summary = {
            "agents_executed": [],
            "execution_order": [],
            "timings": {},
            "status": "in_progress",
            "session_id": session_id,
            "workflow_type": "Full Pipeline with Narrative Flow",
            "memory_provided_direct_answer": False,
            "deepforest_executed": False
        }
        
        visual_context = ""
        detection_narrative = ""
        
        yield {"stage": "visual_analysis", "message": "Analyzing image with unified full/tiled approach...", "type": "progress"}
        
        if session_state_manager.is_cancelled(session_id):
            raise Exception("Processing cancelled by user")
        
        print(f"\nSTEP 1: Visual Analysis (Session {session_id})")
        self._log_gpu_memory(session_id, "before", "Visual Analysis")
        visual_start = time.perf_counter()
        
        # Unified visual analysis
        visual_analysis_result = self.visual_agent.analyze_full_image(
            user_message=user_message,
            session_id=session_id
        )
        
        visual_time = time.perf_counter() - visual_start
        self._log_gpu_memory(session_id, "after", "Visual Analysis")
        self._aggressive_gpu_cleanup(session_id, "after_visual_analysis")
        execution_summary["timings"]["visual_analysis"] = visual_time
        execution_summary["agents_executed"].append("visual_analysis")
        execution_summary["execution_order"].append("visual_analysis")
        agent_results["visual_analysis"] = visual_analysis_result
        
        # Extract visual context
        visual_context = visual_analysis_result.get("visual_analysis", "No visual analysis available")
        
        print(f"Session {session_id} - Visual Analysis: {visual_analysis_result.get('status')}")
        print(f"Session {session_id} - Analysis Type: {visual_analysis_result.get('analysis_type')}")

        yield {"stage": "resolution_check", "message": "Checking image resolution for DeepForest suitability...", "type": "progress"}
        
        if session_state_manager.is_cancelled(session_id):
            raise Exception("Processing cancelled by user")
        
        print(f"\nSTEP 2: Resolution Check (Session {session_id})")
        resolution_start = time.perf_counter()
        
        image_file_path = session_state_manager.get(session_id, "image_file_path")
        resolution_result = None
        
        if image_file_path:
            resolution_result = check_image_resolution_for_deepforest(image_file_path)
            resolution_time = time.perf_counter() - resolution_start
            
            multi_agent_logger.log_resolution_check(
                session_id=session_id,
                image_file_path=image_file_path,
                resolution_result=resolution_result,
                execution_time=resolution_time
            )
        else:
            resolution_result = {
                "is_suitable": True,
                "resolution_info": "No file path available for resolution check",
                "error": None
            }
            resolution_time = time.perf_counter() - resolution_start
        
        execution_summary["timings"]["resolution_check"] = resolution_time
        execution_summary["agents_executed"].append("resolution_check")
        execution_summary["execution_order"].append("resolution_check")
        agent_results["resolution_check"] = resolution_result

        # Determine if DeepForest should run
        detection_result = None
        image_quality_good = visual_analysis_result.get("image_quality_for_deepforest", "No").lower() == "yes"
        resolution_suitable = resolution_result.get("is_suitable", True)
        
        if resolution_suitable and image_quality_good:
            yield {"stage": "detector", "message": "Quality and resolution good - executing DeepForest detection with narrative generation...", "type": "progress"}
            
            if session_state_manager.is_cancelled(session_id):
                raise Exception("Processing cancelled by user")
            
            print(f"\nSTEP 3: DeepForest Detection with R-tree and Narrative (Session {session_id})")
            self._log_gpu_memory(session_id, "before", "DeepForest Detection")
            detector_start = time.perf_counter()
            
            visual_objects = visual_analysis_result.get("deepforest_objects_present", [])
            
            try:
                detection_result = self.detector_agent.execute_detection_with_context(
                    user_message=user_message,
                    session_id=session_id,
                    visual_objects_detected=visual_objects,
                    memory_context=memory_context
                )
                
                detector_time = time.perf_counter() - detector_start
                self._log_gpu_memory(session_id, "after", "DeepForest Detection")
                self._aggressive_gpu_cleanup(session_id, "after_deepforest_detection")
                execution_summary["timings"]["detector_agent"] = detector_time
                execution_summary["agents_executed"].append("detector")
                execution_summary["execution_order"].append("detector")
                execution_summary["deepforest_executed"] = True
                agent_results["detector"] = detection_result

                # Extract detection narrative and tool cache ID from current run
                current_detection_narrative = detection_result.get("detection_narrative", "No detection narrative available")

                # Combine cached narratives from memory with current detection narrative
                combined_narratives = []

                # Add cached narratives from memory's tool cache IDs (if any)
                if memory_tool_cache_id:
                    cached_narrative = self._get_cached_detection_narrative(memory_tool_cache_id)
                    if cached_narrative:
                        combined_narratives.append(cached_narrative)

                # Add current detection narratives for ALL tool results
                tool_results = detection_result.get("tool_results", [])
                if tool_results:
                    for tool_result in tool_results:
                        cache_key = tool_result.get("cache_key")
                        tool_arguments = tool_result.get("tool_arguments", {})
                        
                        if cache_key and tool_arguments:
                            # Get all possible arguments including defaults from Config
                            from deepforest_agent.conf.config import Config
                            all_arguments = Config.DEEPFOREST_DEFAULTS.copy()
                            all_arguments.update(tool_arguments)
                            
                            # Format tool call info with all arguments
                            args_str = ", ".join([f"{k}={v}" for k, v in all_arguments.items()])
                            
                            formatted_current = f"**TOOL CACHE ID:** {cache_key}\nDeepForest tool run with arguments ({args_str}) and got the below narratives:\nDETECTION NARRATIVE:\n{current_detection_narrative}"
                            combined_narratives.append(formatted_current)

                # If no tool results but we have narrative, add it without formatting
                if not tool_results and current_detection_narrative and current_detection_narrative != "No detection narrative available":
                    combined_narratives.append(current_detection_narrative)

                # Combine all narratives
                detection_narrative = "\n\n".join(combined_narratives) if combined_narratives else "No detection narrative available"
                
                print(f"Session {session_id} - DeepForest Detection completed with narrative")
                
            except Exception as detector_error:
                print(f"Session {session_id} - DeepForest Detection FAILED: {detector_error}")
                detection_result = None
                detection_narrative = f"DeepForest detection failed: {str(detector_error)}"
        else:
            skip_reasons = []
            if not resolution_suitable:
                skip_reasons.append("insufficient resolution")
            if not image_quality_good:
                skip_reasons.append("poor image quality")
            
            print(f"Session {session_id} - Skipping DeepForest detection: {', '.join(skip_reasons)}")
            execution_summary["deepforest_executed"] = False
            execution_summary["deepforest_skip_reason"] = ", ".join(skip_reasons)
            detection_narrative = f"DeepForest detection was skipped due to: {', '.join(skip_reasons)}"

        yield {"stage": "ecology", "message": "Synthesizing ecological insights from all contexts...", "type": "progress"}
        
        if session_state_manager.is_cancelled(session_id):
            raise Exception("Processing cancelled by user")

        print(f"\nSTEP 4: Ecology Analysis with Comprehensive Context (Session {session_id})")
        self._log_gpu_memory(session_id, "before", "Ecology Analysis")
        ecology_start = time.perf_counter()

        # Prepare comprehensive context for ecology agent
        comprehensive_context = self._prepare_comprehensive_context(
            memory_context=memory_context,
            visual_context=visual_context,
            detection_narrative=detection_narrative,
            tool_cache_id=memory_tool_cache_id
        )

        final_response = ""
        try:
            for token_result in self.ecology_agent.synthesize_analysis_streaming(
                user_message=user_message,
                memory_context=comprehensive_context,
                cached_json=None,
                current_json=None,
                session_id=session_id
            ):
                if session_state_manager.is_cancelled(session_id):
                    raise Exception("Processing cancelled by user")
                    
                final_response += token_result["token"]

                yield {
                    "stage": "ecology_streaming",
                    "message": final_response,
                    "type": "streaming",
                    "is_complete": token_result["is_complete"]
                }

                if token_result["is_complete"]:
                    break
        
        except Exception as ecology_error:
            print(f"Session {session_id} - Ecology streaming error: {ecology_error}")
            if not final_response:
                final_response = f"Ecology analysis failed: {str(ecology_error)}"

        finally:
            ecology_time = time.perf_counter() - ecology_start
            self._log_gpu_memory(session_id, "after", "Ecology Analysis")
            self._aggressive_gpu_cleanup(session_id, "after_ecology_analysis")
            execution_summary["timings"]["ecology_agent"] = ecology_time
            execution_summary["agents_executed"].append("ecology")
            execution_summary["execution_order"].append("ecology")
            agent_results["ecology"] = {"final_response": final_response}

        # Store context data for memory agent's next turn
        current_turn = len(session_state_manager.get(session_id, "conversation_history", [])) // 2 + 1
        all_tool_cache_ids = []
        if memory_tool_cache_id:
            all_tool_cache_ids.extend([id.strip() for id in memory_tool_cache_id.split(",")])

        # Add all current tool cache IDs
        tool_results = detection_result.get("tool_results", []) if detection_result else []
        for tool_result in tool_results:
            cache_key = tool_result.get("cache_key")
            if cache_key:
                all_tool_cache_ids.append(cache_key)

        combined_tool_cache_id = ", ".join(all_tool_cache_ids) if all_tool_cache_ids else None
        self.memory_agent.store_turn_context(
            session_id=session_id,
            turn_number=current_turn,
            visual_context=visual_context,
            detection_narrative=detection_narrative,
            tool_cache_id=combined_tool_cache_id
        )
        
        # Final result
        total_time = time.perf_counter() - start_time
        execution_summary["timings"]["total"] = total_time
        execution_summary["status"] = "completed_narrative_flow"

        detection_data_monitor = self._format_detection_data_for_monitor(
            detection_narrative=detection_narrative,
            detections_list=detection_result.get("detections_list", []) if detection_result else None
        )

        print(f"Session {session_id} - NARRATIVE FLOW WORKFLOW COMPLETED")
        
        yield {
            "stage": "complete",
            "message": final_response,
            "type": "final",
            "detection_data": detection_data_monitor,
            "agent_results": agent_results,
            "execution_summary": execution_summary,
            "execution_time": total_time,
            "status": "success"
        }

    def _prepare_comprehensive_context(
        self,
        memory_context: str,
        visual_context: str,
        detection_narrative: str,
        tool_cache_id: Optional[str]
    ) -> str:
        """
        Prepare comprehensive context combining all data sources with better formatting.
        
        Args:
            memory_context: Context from memory agent
            visual_context: Visual analysis context
            detection_narrative: R-tree based detection narrative  
            tool_cache_id: Tool cache reference if available
            
        Returns:
            Combined context string for ecology agent
        """
        context_parts = []
        
        # Memory context section
        if memory_context and memory_context != "No memory context available":
            context_parts.append("--- START OF MEMORY CONTEXT ---")
            context_parts.append(memory_context)
            context_parts.append("--- END OF MEMORY CONTEXT ---")
            context_parts.append("")

        # Tool cache reference
        if tool_cache_id:
            context_parts.append(f"**TOOL CACHE ID:** {tool_cache_id}")
            context_parts.append("")
        
        # Detection narrative section
        if detection_narrative and detection_narrative not in ["No detection analysis available", ""]:
            context_parts.append("--- START OF DETECTION ANALYSIS ---")
            context_parts.append(detection_narrative)
            context_parts.append("--- END OF DETECTION ANALYSIS ---")
            context_parts.append("")

         # Visual context section  
        if visual_context and visual_context != "No visual analysis available":
            context_parts.append("--- START OF VISUAL ANALYSIS ---")
            context_parts.append(visual_context)
            context_parts.append("There may be information that are not clear or accurate in this visual analysis. So make sure to mention that this analysis is provided by a visual analysis agent and it may not be very accurate as there is no confidence score associated with it. You can only provide this analysis seperately in a different section and inform the user that you are not very confident about this analysis.")
            context_parts.append("--- END OF VISUAL ANALYSIS ---")
            context_parts.append("")
        
        # If we have very little context, provide a meaningful message
        if not context_parts or len("".join(context_parts)) < 50:
            return "No comprehensive context available for this query. Please provide more information or try a different approach."
        
        result_context = "\n".join(context_parts)

        print(f"Prepared comprehensive context ({len(result_context)} characters)")
        print(f"Context preview: {result_context[:200]}...")
        
        return result_context

    def _create_fallback_response(
        self,
        user_message: str,
        agent_results: Dict[str, Any],
        error: str,
        session_id: str
    ) -> str:
        """Create a fallback response when the orchestrator encounters errors."""
        response_parts = []
        response_parts.append(f"I encountered some processing issues but can provide analysis based on available data:")
        response_parts.append("")

        memory_result = agent_results.get("memory", {})
        if memory_result and memory_result.get("relevant_context"):
            response_parts.append(f"**Memory Context**: {memory_result['relevant_context']}")
            response_parts.append("")

        visual_result = agent_results.get("visual_analysis", {})
        if visual_result and visual_result.get("visual_analysis"):
            response_parts.append(f"**Visual Analysis**: {visual_result['visual_analysis']}")
            response_parts.append("")

        detector_result = agent_results.get("detector", {})
        if detector_result and detector_result.get("detection_narrative"):
            response_parts.append(f"**Detection Results**: {detector_result['detection_narrative']}")
            response_parts.append("")
        
        response_parts.append(f"Note: Workflow was interrupted ({error}). Please try your query again for full results.")
        
        return "\n".join(response_parts)