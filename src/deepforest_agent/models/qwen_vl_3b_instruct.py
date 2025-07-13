import gc
from typing import Tuple, Dict, Any, Optional, List, Union
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
from qwen_vl_utils import process_vision_info

from deepforest_agent.conf.config import Config


class QwenVL3BModelManager:
    """Manages Qwen2.5-VL-3B model instances for visual analysis tasks.

    Attributes:
        model_id (str): HuggingFace model identifier
        load_count (int): Number of times model has been loaded
    """
    
    def __init__(self, model_id: str = Config.AGENT_MODELS["visual_analysis"]):
        """
        Initialize the Qwen2.5-VL-3B model manager.
        
        Args:
            model_id (str, optional): HuggingFace model identifier. 
                                    Defaults to "Qwen/Qwen2.5-VL-3B-Instruct".
        """
        self.model_id = model_id
        self.load_count = 0

    def _load_model(self) -> Tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
        """
        Private method for model loading implementation.
        
        Returns:
            Tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]: 
                Loaded model and processor instances
                
        Raises:
            Exception: If model or processor loading fails
        """
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )

            processor = AutoProcessor.from_pretrained(
                self.model_id, 
                use_fast=True
            )
            
            return model, processor
            
        except Exception as e:
            print(f"Error loading Qwen VL model: {e}")
            raise e
    
    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = Config.AGENT_CONFIGS["visual_analysis"]["max_new_tokens"],
        temperature: float = Config.AGENT_CONFIGS["visual_analysis"]["temperature"]
    ) -> str:
        """
        Generate multimodal response.
        
        Args:
            messages: List of messages with text and images
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            str: Generated response text based on the input messages
            
        Raises:
            Exception: If text generation fails for any reason
        """
        print(f"Loading Qwen VL for inference #{self.load_count + 1}")

        model, processor = self._load_model()
        self.load_count += 1

        try:
            # Process vision info using qwen_vl_utils
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Use process_vision_info for proper image handling
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            response = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return response
            
        except Exception as e:
            print(f"Error during Qwen VL generation: {e}")
            raise e
    
        finally:
            print(f"Releasing Qwen VL GPU memory after inference")
            if 'model' in locals():
                if hasattr(model, 'cpu'):
                    model.cpu()
                del model
            if 'processor' in locals():
                del processor
            if 'inputs' in locals():
                del inputs
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
                print(f"GPU memory after VLM cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f} GB cached")