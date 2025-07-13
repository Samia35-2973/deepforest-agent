from typing import Dict, List, Any, Optional
import re
import time
import json

from deepforest_agent.models.smollm3_3b import SmolLM3ModelManager
from deepforest_agent.conf.config import Config
from deepforest_agent.prompts.prompt_templates import format_memory_prompt
from deepforest_agent.utils.state_manager import session_state_manager
from deepforest_agent.utils.logging_utils import multi_agent_logger
from deepforest_agent.utils.parsing_utils import parse_memory_agent_response
from deepforest_agent.utils.cache_utils import tool_call_cache
from deepforest_agent.conf.config import Config

class MemoryAgent:
    """
    Memory agent responsible for analyzing conversation history in new format.
    Uses SmolLM3-3B model for getting relevant context
    """
    
    def __init__(self):
        """Initialize the Memory Agent with model manager and configuration."""
        self.agent_config = Config.AGENT_CONFIGS["memory"]
        self.model_manager = SmolLM3ModelManager(Config.AGENT_MODELS["memory"])
    
    def _filter_conversation_history(self, conversation_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter conversation history to include user and assistant messages.
        
        Args:
            conversation_history: Full conversation history
                
        Returns:
            Filtered history with only user/assistant messages
        """
        filtered_history = []
        
        for message in conversation_history:
            if message.get("role") in ["user", "assistant"]:
                content = message.get("content", "")
                if isinstance(content, list):
                    text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                    content = " ".join(text_parts)
                elif isinstance(content, str):
                    content = content
                else:
                    content = str(content)
                
                filtered_history.append({
                    "role": message["role"],
                    "content": content
                })

        return filtered_history

    def _get_conversation_history_context(self, session_id: str) -> str:
        """
        Get formatted conversation history with turn-based structure.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Formatted conversation history with turn structure
        """
        conversation_history = session_state_manager.get(session_id, "conversation_history", [])
        
        print(f"Session {session_id} - Conversation length: {len(conversation_history)}")

        if not conversation_history:
            return "No previous conversation history available."
        
        # Build turn-based history
        formatted_history = []
        turn_number = 1
        
        # Process conversation in pairs (user -> assistant)
        i = 0
        while i < len(conversation_history):
            if i + 1 < len(conversation_history):
                user_msg = conversation_history[i]
                assistant_msg = conversation_history[i + 1]
                
                if user_msg.get("role") == "user" and assistant_msg.get("role") == "assistant":
                    # Extract user query
                    user_content = user_msg.get("content", "")
                    if isinstance(user_content, list):
                        text_parts = [item.get("text", "") for item in user_content if item.get("type") == "text"]
                        user_query = " ".join(text_parts)
                    else:
                        user_query = str(user_content)
                    
                    # Get stored context data for this turn
                    visual_context = session_state_manager.get(session_id, f"turn_{turn_number}_visual_context", "No visual analysis available")
                    detection_narrative = session_state_manager.get(session_id, f"turn_{turn_number}_detection_narrative", "No detection narrative available") 
                    tool_cache_id = session_state_manager.get(session_id, f"turn_{turn_number}_tool_cache_id", "No tool cache ID")
                    tool_call_info = "No tool call information available"
                    if tool_cache_id:
                        try:
                            if tool_cache_id in tool_call_cache.cache_data:
                                cached_entry = tool_call_cache.cache_data[tool_cache_id]
                                tool_name = cached_entry.get("tool_name", "unknown")
                                stored_arguments = cached_entry.get("arguments", {})

                                all_arguments = Config.DEEPFOREST_DEFAULTS.copy()
                                all_arguments.update(stored_arguments)
                                
                                # Format tool call info with all arguments
                                args_str = ", ".join([f"{k}={v}" for k, v in all_arguments.items()])
                                tool_call_info = f"Tool: {tool_name} called with arguments: {args_str}"
                        except Exception as e:
                            tool_call_info = f"Error retrieving tool call info: {str(e)}"

                    turn_text = f"--- Turn {turn_number}: ---\n"
                    turn_text += f"Turn {turn_number} User query: {user_query}\n"
                    turn_text += f"Turn {turn_number} Visual analysis full image or per tile: {visual_context}\n"
                    turn_text += f"Turn {turn_number} Tool cache ID: {tool_cache_id}\n"
                    turn_text += f"Turn {turn_number} Tool call details: {tool_call_info}\n"
                    turn_text += f"Turn {turn_number} Detection Data Analysis: {detection_narrative}\n"
                    turn_text += f"--- Turn {turn_number} Completed ---\n"
                    
                    formatted_history.append(turn_text)
                    turn_number += 1
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        
        if not formatted_history:
            return "No complete conversation turns available."
        
        print(f"Formatted {len(formatted_history)} conversation turns")
        return "\n\n".join(formatted_history)

    def process_conversation_history_structured(
        self, 
        conversation_history: List[Dict[str, Any]], 
        latest_message: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Process conversation history and extract relevant context with structured output.
        
        Args:
            conversation_history: Full conversation history 
            latest_message: Current user message requiring context analysis
            session_id: Unique session identifier for this user
            
        Returns:
            Dict with structured output including tool_cache_id and relevant context
        """
        if not session_state_manager.session_exists(session_id):
            return {
                "answer_present": False,
                "direct_answer": "NO",
                "tool_cache_id": None,
                "relevant_context": f"Session {session_id} not found. Current query: {latest_message}",
                "raw_response": f"Session {session_id} not found"
            }

        filtered_history = self._filter_conversation_history(conversation_history)
        conversation_context = self._get_conversation_history_context(session_id)
        
        memory_prompt = format_memory_prompt(filtered_history, latest_message, conversation_context)
        print(f"Memory Agent Prompt:\n{memory_prompt}\n")
        
        messages = [
            {"role": "system", "content": memory_prompt},
            {"role": "user", "content": latest_message}
        ]
        
        memory_execution_start = time.perf_counter()

        try:
            response = self.model_manager.generate_response(
                messages=messages,
                max_new_tokens=self.agent_config["max_new_tokens"],
                temperature=self.agent_config["temperature"],
                top_p=self.agent_config["top_p"]
            )

            memory_execution_time = time.perf_counter() - memory_execution_start
            
            print(f"Session {session_id} - Memory Agent: Raw response received")
            print(f"Raw Response: {response}")
            
            parsed_result = parse_memory_agent_response(response)

            multi_agent_logger.log_agent_execution(
                session_id=session_id,
                agent_name="memory",
                agent_input=f"Latest message: {latest_message}",
                agent_output=response,
                execution_time=memory_execution_time
            )
            
            print(f"Session {session_id} - Memory Agent: Analysis completed")
            print(f"Has Answer: {parsed_result['answer_present']}")
            
            return parsed_result
            
        except Exception as e:
            memory_execution_time = time.perf_counter() - memory_execution_start
            error_msg = f"Error processing conversation history in session {session_id}: {str(e)}"
            print(f"Session {session_id} - Memory Agent Error: {e}")

            multi_agent_logger.log_error(
                session_id=session_id,
                error_type="memory_agent_error",
                error_message=f"Memory agent failed after {memory_execution_time:.2f}s: {str(e)}"
            )

            return {
                "answer_present": False,
                "direct_answer": "NO",
                "tool_cache_id": None,
                "relevant_context": f"{error_msg}. Current query: {latest_message}",
                "raw_response": str(e)
            }

    def store_turn_context(self, session_id: str, turn_number: int, visual_context: str, 
                          detection_narrative: str, tool_cache_id: Optional[str]) -> None:
        """
        Store context data for a specific conversation turn.
        
        Args:
            session_id: Session identifier
            turn_number: Turn number in conversation
            visual_context: Visual analysis context
            detection_narrative: Detection narrative
            tool_cache_id: Tool cache identifier
        """
        session_state_manager.set(session_id, f"turn_{turn_number}_visual_context", visual_context)
        session_state_manager.set(session_id, f"turn_{turn_number}_detection_narrative", detection_narrative)
        session_state_manager.set(session_id, f"turn_{turn_number}_tool_cache_id", tool_cache_id or "No tool cache ID")
        
        print(f"Session {session_id} - Stored context for turn {turn_number}")