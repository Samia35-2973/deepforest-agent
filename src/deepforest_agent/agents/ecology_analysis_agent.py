import json
from typing import Dict, List, Any, Optional, Generator

from deepforest_agent.models.llama32_3b_instruct import Llama32ModelManager
from deepforest_agent.conf.config import Config
from deepforest_agent.prompts.prompt_templates import create_ecology_synthesis_prompt
from deepforest_agent.utils.state_manager import session_state_manager


class EcologyAnalysisAgent:
    """
    Ecology analysis agent responsible for combining all data into comprehensive ecological insights.
    Uses Llama-3.2-3B-Instruct model for detailed structured response generation with analysis.
    """
    
    def __init__(self):
        """Initialize the Ecology Analysis Agent."""
        self.agent_config = Config.AGENT_CONFIGS["ecology_analysis"]
        self.model_manager = Llama32ModelManager(Config.AGENT_MODELS["ecology_analysis"])

    def synthesize_analysis_streaming(
        self,
        user_message: str,
        memory_context: str,
        cached_json: Optional[Dict[str, Any]] = None,
        current_json: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Synthesize all agent outputs with streaming text generation.
        
        Args:
            user_message (str): The user's original query for the analysis.
            memory_context (str): The context and conversation history provided
                by a memory agent.
            cached_json (Optional[Dict[str, Any]]): A dictionary of previously
                cached JSON data, if available. Defaults to None.
            current_json (Optional[Dict[str, Any]]): A dictionary of new JSON data
                from the current analysis step. Defaults to None.
            session_id (Optional[str]): A unique session identifier for tracking
                and logging. Defaults to None.
            
        Yields:
            Dict[str, Any]: Dictionary containing:
                - token: Generated text token
                - is_complete: Whether generation is finished
        """
        if session_id and not session_state_manager.session_exists(session_id):
            yield {
                "token": f"Session {session_id} not found. Unable to synthesize analysis.",
                "is_complete": True
            }
            return

        try:
            synthesis_prompt = create_ecology_synthesis_prompt(
                user_message=user_message,
                comprehensive_context=memory_context,
                cached_json=cached_json,
                current_json=current_json
            )
            print(f"Ecology Synthesis Prompt:\n{synthesis_prompt}\n")
            
            messages = [
                {"role": "system", "content": synthesis_prompt},
                {"role": "user", "content": user_message}
            ]

            print(f"Session {session_id} - Ecology Agent: Starting streaming synthesis")

            # Stream the response token by token
            for token_data in self.model_manager.generate_response_streaming(
                messages=messages,
                max_new_tokens=self.agent_config["max_new_tokens"],
                temperature=self.agent_config["temperature"],
                top_p=self.agent_config["top_p"]
            ):
                yield token_data
                
                if token_data["is_complete"]:
                    print(f"Session {session_id} - Ecology Agent: Streaming synthesis completed")
                    break
            
        except Exception as e:
            error_msg = f"Error in ecology synthesis for session {session_id}: {str(e)}"
            print(f"Ecology Analysis Error: {error_msg}")
            
            # Yield error_msg response as single token
            yield {
                "token": error_msg,
                "is_complete": True
            }