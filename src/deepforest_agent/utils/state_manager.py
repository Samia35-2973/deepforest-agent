import threading
import uuid
import time
from typing import Optional, Any, Dict, List

from deepforest_agent.utils.cache_utils import tool_call_cache


class SessionStateManager:
    """
    Session-based state manager with thread ID for the DeepForest Agent.
    
    This class manages state for multiple concurrent users with each user
    having their own session containing current image, conversation
    history, and session information.
    
    Attributes:
        _lock (threading.Lock): Thread synchronization lock
        _sessions (Dict[str, Dict[str, Any]]): Dictionary mapping session_ids to session state
        _cleanup_interval (int): Time in seconds after which inactive sessions are cleaned up
    """
    
    def __init__(self, cleanup_interval: int = 3600) -> None:
        """
        Initialize the session state manager.

        Args:
            cleanup_interval (int): Time in seconds after which inactive sessions 
                                  are eligible for cleanup (default: 1 hour)
        """
        self._lock = threading.Lock()
        self._sessions = {}
        self._cleanup_interval = cleanup_interval

    def create_session(self, image: Any = None) -> str:
        """
        Create a new session with initial image.
        
        Args:
            image (Any, optional): Initial image for the session
            
        Returns:
            str: Unique session ID
        """
        session_id = str(uuid.uuid4())[:12]
        
        with self._lock:
            self._sessions[session_id] = {
                "current_image": image,
                "conversation_history": [],
                "annotated_image": None,
                "thread_id": session_id,
                "first_message": True,
                "created_at": time.time(),
                "last_accessed": time.time(),
                "is_cancelled": False,
                "is_processing": False,
                "tool_call_history": [],
                "visual_analysis_history": []
            }
        
        return session_id
    
    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """
        Get complete state for a specific session.
        
        Args:
            session_id (str): The session ID to retrieve
            
        Returns:
            Dict[str, Any]: Copy of session state dictionary
            
        Raises:
            KeyError: If session_id doesn't exist
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session {session_id} not found")
            
            self._sessions[session_id]["last_accessed"] = time.time()
            
            # Return a copy to prevent external modification
            return self._sessions[session_id].copy()
    
    def get(self, session_id: str, key: str, default: Any = None) -> Any:
        """
        Get a value from session state.
        
        Args:
            session_id (str): The session ID
            key (str): The state key to retrieve
            default (Any, optional): Default value if key not found.
        
        Returns:
            Any: The value associated with the key, or default if not found

        Raises:
            KeyError: If session_id doesn't exist
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session {session_id} not found")
            
            self._sessions[session_id]["last_accessed"] = time.time()

            return self._sessions[session_id].get(key, default)
    
    def set(self, session_id: str, key: str, value: Any) -> None:
        """
        Set a value in session state.
        
        Args:
            session_id (str): The session ID
            key (str): The state key to set
            value (Any): The value to store

        Raises:
            KeyError: If session_id doesn't exist
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session {session_id} not found")
            
            self._sessions[session_id][key] = value
            self._sessions[session_id]["last_accessed"] = time.time()
    
    def update(self, session_id: str, updates: Dict[str, Any]) -> None:
        """
        Update multiple values in session state.
        
        Args:
            session_id (str): The session ID
            updates (Dict[str, Any]): Dictionary of key-value pairs to update

        Raises:
            KeyError: If session_id doesn't exist
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session {session_id} not found")
            
            self._sessions[session_id].update(updates)
            self._sessions[session_id]["last_accessed"] = time.time()

    def set_processing_state(self, session_id: str, is_processing: bool) -> None:
        """
        Set processing state for a session.
        
        Args:
            session_id (str): The session ID
            is_processing (bool): Whether processing is active
        """
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["is_processing"] = is_processing
                self._sessions[session_id]["last_accessed"] = time.time()

    def cancel_session(self, session_id: str) -> None:
        """
        Cancel processing for a session.
        
        Args:
            session_id (str): The session ID to cancel
        """
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["is_cancelled"] = True
                self._sessions[session_id]["is_processing"] = False
                self._sessions[session_id]["last_accessed"] = time.time()

    def is_cancelled(self, session_id: str) -> bool:
        """
        Check if session is cancelled.
        
        Args:
            session_id (str): The session ID to check
            
        Returns:
            bool: True if cancelled
        """
        with self._lock:
            if session_id not in self._sessions:
                return True
            return self._sessions[session_id].get("is_cancelled", False)

    def reset_cancellation(self, session_id: str) -> None:
        """
        Reset cancellation flag for a session.
        
        Args:
            session_id (str): The session ID to reset
        """
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["is_cancelled"] = False
                self._sessions[session_id]["last_accessed"] = time.time()

    def add_tool_call_to_history(self, session_id: str, tool_name: str, arguments: Dict[str, Any], cache_key: str) -> None:
        """
        Add a tool call to the session's tool call history.
        
        Args:
            session_id (str): The session ID
            tool_name (str): Name of the tool that was called
            arguments (Dict[str, Any]): Arguments passed to the tool
            cache_key (str): Cache key used for this tool call
            
        Raises:
            KeyError: If session_id doesn't exist
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session {session_id} not found")
            
            tool_call_entry = {
                "tool_name": tool_name,
                "arguments": arguments.copy(),
                "cache_key": cache_key,
                "timestamp": time.time(),
                "call_number": len(self._sessions[session_id]["tool_call_history"]) + 1
            }
            
            self._sessions[session_id]["tool_call_history"].append(tool_call_entry)
            self._sessions[session_id]["last_accessed"] = time.time()

    def get_tool_call_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get the tool call history for a specific session.
        
        Args:
            session_id (str): The session ID
            
        Returns:
            List[Dict[str, Any]]: List of tool calls made in this session
            
        Raises:
            KeyError: If session_id doesn't exist
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session {session_id} not found")
            
            self._sessions[session_id]["last_accessed"] = time.time()
            return self._sessions[session_id]["tool_call_history"].copy()

    def add_visual_analysis_to_history(self, session_id: str, visual_analysis: str, additional_objects: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add a visual analysis response to the session's history.
        
        Args:
            session_id (str): The session ID
            visual_analysis (str): Visual analysis text from visual agent
            additional_objects (Optional[List[Dict[str, Any]]]): Additional objects detected by visual agent
            
        Raises:
            KeyError: If session_id doesn't exist
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session {session_id} not found")
            
            visual_entry = {
                "visual_analysis": visual_analysis,
                "additional_objects": additional_objects or [],
                "timestamp": time.time(),
                "turn_number": len(self._sessions[session_id]["visual_analysis_history"]) + 1
            }
            
            self._sessions[session_id]["visual_analysis_history"].append(visual_entry)
            self._sessions[session_id]["last_accessed"] = time.time()

    def get_visual_analysis_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all visual analysis responses from previous turns.
        
        Args:
            session_id (str): The session ID
            
        Returns:
            List[Dict[str, Any]]: List of visual analysis entries with text and additional objects
            
        Raises:
            KeyError: If session_id doesn't exist
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session {session_id} not found")
            
            self._sessions[session_id]["last_accessed"] = time.time()

            return self._sessions[session_id]["visual_analysis_history"].copy()

    def get_formatted_tool_call_history(self, session_id: str) -> str:
        """
        Get formatted tool call history for memory agent context.
        
        Args:
            session_id (str): The session ID
            
        Returns:
            str: Formatted tool call history string
            
        Raises:
            KeyError: If session_id doesn't exist
        """
        try:
            tool_calls = self.get_tool_call_history(session_id)
            if not tool_calls:
                return "No previous tool calls in this session."
            
            formatted_history = []
            for tool_call in tool_calls:
                call_info = f"Tool Call #{tool_call.get('call_number', 'N/A')}: "
                call_info += f"{tool_call.get('tool_name', 'unknown')} "
                call_info += f"with args {tool_call.get('arguments', {})}"
                formatted_history.append(call_info)
            
            return "\n".join(formatted_history)
        except KeyError:
            return f"Session {session_id} not found - no tool call history available."

    def store_conversation_turn_context(
        self, 
        session_id: str, 
        turn_number: int,
        user_query: str,
        visual_context: str,
        detection_narrative: str,
        tool_cache_id: Optional[str],
        ecology_response: str
    ) -> None:
        """
        Store complete turn context for memory agent.
        
        Args:
            session_id (str): The session ID
            turn_number (int): Sequential number of this conversation turn (1-indexed)
            user_query (str): The original user question of the current turn
            visual_context (str): Complete visual analysis output from the visual agent
            detection_narrative (str): Comprehensive spatial analysis narrative generated
                                     from DeepForest detection results
            tool_cache_id (Optional[str]): Cache identifier for DeepForest tool execution
                                         results
            ecology_response (str): Final synthesized ecological analysis response
        """
        turn_data = {
            "user_query": user_query,
            "visual_context": visual_context, 
            "detection_narrative": detection_narrative,
            "tool_cache_id": tool_cache_id or "No tool cache ID",
            "ecology_response": ecology_response,
            "timestamp": time.time()
        }
        
        self.set(session_id, f"conversation_turn_{turn_number}", turn_data)
        
        # Update turn counter
        current_turns = self.get(session_id, "total_turns", 0)
        self.set(session_id, "total_turns", max(current_turns, turn_number))

    def get_cache_stats_for_session(self, session_id: str) -> Dict[str, Any]:
        """
        Get cache statistics specific to this session.
        
        Args:
            session_id (str): The session ID
            
        Returns:
            Dict[str, Any]: Cache statistics for this session
            
        Raises:
            KeyError: If session_id doesn't exist
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session {session_id} not found")
            
            session_tool_calls = self._sessions[session_id]["tool_call_history"]
            
            return {
                "session_id": session_id,
                "total_tool_calls": len(session_tool_calls),
                "tool_calls": session_tool_calls,
                "global_cache_stats": tool_call_cache.get_cache_stats()
            }

    def clear_session_cache_data(self, session_id: str) -> None:
        """
        Clear tool call history for a specific session.
        
        Note: This only clears the session's record of tool calls,
        not the global cache itself.
        
        Args:
            session_id (str): The session ID
            
        Raises:
            KeyError: If session_id doesn't exist
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session {session_id} not found")
            
            self._sessions[session_id]["tool_call_history"] = []
            self._sessions[session_id]["last_accessed"] = time.time()

    def clear_conversation(self, session_id: str) -> None:
        """
        Clear conversation-specific state for a session.

        current_image and thread_id are preserved so that users can 
        start a new conversation without re-uploading the image.
        
        Args:
            session_id (str): The session ID to clear
            
        Raises:
            KeyError: If session_id doesn't exist
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session {session_id} not found")
            
            self._sessions[session_id].update({
                "conversation_history": [],
                "annotated_image": None,
                "first_message": True,
                "last_accessed": time.time(),
                "is_cancelled": True,
                "is_processing": False,
                "tool_call_history": [],
                "visual_analysis_history": []
            })

    def reset_for_new_image(self, session_id: str, image: Any) -> None:
        """
        Reset session state for new image upload.
        
        Args:
            session_id (str): The session ID
            image (Any): The new image object (typically PIL Image)
            
        Raises:
            KeyError: If session_id doesn't exist
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session {session_id} not found")
            
            self._sessions[session_id].update({
                "current_image": image,
                "conversation_history": [],
                "annotated_image": None,
                "first_message": True,
                "last_accessed": time.time(),
                "tool_call_history": [],
                "visual_analysis_history": []
            })

    def add_to_conversation(self, session_id: str, message: Dict[str, Any]) -> None:
        """
        Add a message to conversation history for a specific session.
        
        Args:
            session_id (str): The session ID
            message (Dict[str, Any]): Message dictionary with role and content
            
        Raises:
            KeyError: If session_id doesn't exist
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session {session_id} not found")
            
            self._sessions[session_id]["conversation_history"].append(message)
            self._sessions[session_id]["last_accessed"] = time.time()

    def get_conversation_length(self, session_id: str) -> int:
        """
        Get the length of conversation history for a session.
        
        Args:
            session_id (str): The session ID
            
        Returns:
            int: Number of messages in conversation history
            
        Raises:
            KeyError: If session_id doesn't exist
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session {session_id} not found")
            
            self._sessions[session_id]["last_accessed"] = time.time()
            return len(self._sessions[session_id]["conversation_history"])
    
    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists.
        
        Args:
            session_id (str): The session ID to check
            
        Returns:
            bool: True if session exists, False otherwise
        """
        with self._lock:
            return session_id in self._sessions
    
    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all active sessions.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping session_ids to session info
        """
        with self._lock:
            session_info = {}
            for session_id, session_data in self._sessions.items():
                session_info[session_id] = {
                    "thread_id": session_data.get("thread_id"),
                    "created_at": session_data.get("created_at"),
                    "last_accessed": session_data.get("last_accessed"),
                    "conversation_length": len(session_data.get("conversation_history", [])),
                    "has_image": session_data.get("current_image") is not None,
                    "has_annotated_image": session_data.get("annotated_image") is not None,
                    "tool_calls_count": len(session_data.get("tool_call_history", []))
                }
            return session_info
    
    def cleanup_inactive_sessions(self) -> int:
        """
        Remove sessions that haven't been accessed recently.
        
        Returns:
            int: Number of sessions cleaned up
        """
        current_time = time.time()
        cleaned_count = 0
        
        with self._lock:
            inactive_sessions = []
            for session_id, session_data in self._sessions.items():
                last_accessed = session_data.get("last_accessed", 0)
                if current_time - last_accessed > self._cleanup_interval:
                    inactive_sessions.append(session_id)
            
            for session_id in inactive_sessions:
                del self._sessions[session_id]
                cleaned_count += 1
        
        return cleaned_count
    
    def delete_session(self, session_id: str) -> bool:
        """
        Manually delete a specific session.
        
        Args:
            session_id (str): The session ID to delete
            
        Returns:
            bool: True if session was deleted, False if it didn't exist
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False


# Global session manager instance
session_state_manager = SessionStateManager()