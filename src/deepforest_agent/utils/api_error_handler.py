import openai

class APIErrorHandler:
    """
    Standardized error handling for API communications.
    """

    @staticmethod
    def handle_api_call(api_call_func, error_context: str = "API call"):
        """
        Execute API call with standardized error handling.
        
        Args:
            api_call_func: Function that makes the API call
            error_context: Description of the operation for error messages
            
        Returns:
            API response or tuple of (error_message, None) on failure
        """
        try:
            return api_call_func()
        except openai.BadRequestError as e:
            error_msg = f"An error occurred with the AI model. Please try again or rephrase your request."
            print(f"OpenAI BadRequestError during {error_context}: {e.response}")
            return error_msg, None
        except Exception as e:
            error_msg = f"An unexpected error occurred. Please try again."
            print(f"Unexpected error during {error_context}: {e}")
            return error_msg, None