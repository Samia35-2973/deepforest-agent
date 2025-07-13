import gc
from typing import Tuple, Dict, Any, Optional, List, Generator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer
from threading import Thread

from deepforest_agent.conf.config import Config

class SmolLM3ModelManager:
    """
    Manages SmolLM3-3B model instances
    
    Attributes:
        model_id (str): HuggingFace model identifier
        load_count (int): Number of times model has been loaded
    """
    
    def __init__(self, model_id: str = Config.AGENT_MODELS["deepforest_detector"]):
        """
        Initialize the SmolLM3 model manager.
        
        Args:
            model_id (str, optional): HuggingFace model identifier. 
                                    Defaults to "HuggingFaceTB/SmolLM3-3B".
        """
        self.model_id = model_id
        self.load_count = 0
    
    def generate_response(
        self, 
        messages: List[Dict[str, str]],
        max_new_tokens: int = Config.AGENT_CONFIGS["deepforest_detector"]["max_new_tokens"],
        temperature: float = Config.AGENT_CONFIGS["deepforest_detector"]["temperature"],
        top_p: float = Config.AGENT_CONFIGS["deepforest_detector"]["top_p"],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate text response
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            tools (Optional[List[Dict[str, Any]]]): List of tools
            
        Raises:
            Exception: If generation fails due to model issues, memory, or other errors
        """
        print(f"Loading SmolLM3 for inference #{self.load_count + 1}")

        model, tokenizer = self._load_model()
        self.load_count += 1

        try:
            if tools:
                text = tokenizer.apply_chat_template(
                    messages,
                    xml_tools=tools,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
            
        except Exception as e:
            print(f"Error during SmolLM3 text generation: {e}")
            raise e
            
        finally:
            print(f"Releasing SmolLM3 GPU memory after inference")
            if 'model' in locals():
                if hasattr(model, 'cpu'):
                    model.cpu()
                del model
            if 'tokenizer' in locals():
                del tokenizer
            if 'model_inputs' in locals():
                del model_inputs
            if 'generated_ids' in locals():
                del generated_ids
            
            # Multiple garbage collection passes
            for _ in range(3):
                gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect() 
                torch.cuda.synchronize()
                try:
                    torch.cuda.memory._record_memory_history(enabled=None)
                except:
                    pass
                print(f"GPU memory after aggressive cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f} GB cached")
    
    def generate_response_streaming(
        self, 
        messages: List[Dict[str, str]],
        max_new_tokens: int = Config.AGENT_CONFIGS["deepforest_detector"]["max_new_tokens"],
        temperature: float = Config.AGENT_CONFIGS["deepforest_detector"]["temperature"],
        top_p: float = Config.AGENT_CONFIGS["deepforest_detector"]["top_p"]
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate text response with streaming (token by token)
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            
        Yields:
            Dict[str, Any]: Dictionary containing:
                - token: The generated token/text chunk
                - is_complete: Whether generation is finished
                
        Raises:
            Exception: If generation fails due to model issues, memory, or other errors
        """
        print(f"Loading SmolLM3 for streaming inference #{self.load_count + 1}")

        model, tokenizer = self._load_model()
        self.load_count += 1
        
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            streamer = TextIteratorStreamer(
                tokenizer, 
                timeout=60.0, 
                skip_prompt=True, 
                skip_special_tokens=True
            )

            generation_kwargs = {
                "input_ids": model_inputs.input_ids,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id,
                "streamer": streamer
            }

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            for new_text in streamer:
                yield {"token": new_text, "is_complete": False}

            thread.join()
            yield {"token": "", "is_complete": True}
            
        except Exception as e:
            print(f"Error during SmolLM3 streaming generation: {e}")
            yield {"token": f"[Error: {str(e)}]", "is_complete": True}
            
        finally:
            print(f"Releasing SmolLM3 GPU memory after inference")
            if 'model' in locals():
                if hasattr(model, 'cpu'):
                    model.cpu()
                del model
            if 'tokenizer' in locals():
                del tokenizer
            if 'model_inputs' in locals():
                del model_inputs
            if 'generated_ids' in locals():
                del generated_ids
            
            # Multiple garbage collection passes
            for _ in range(3):
                gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect() 
                torch.cuda.synchronize()
                try:
                    torch.cuda.memory._record_memory_history(enabled=None)
                except:
                    pass
                print(f"GPU memory after aggressive cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f} GB cached")

    def _load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Private method for model and tokenizer loading.
        
        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: Loaded model and tokenizer
            
        Raises:
            Exception: If model loading fails due to network, memory, or other issues
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            return model, tokenizer
            
        except Exception as e:
            print(f"Error loading SmolLM3 model: {e}")
            raise e