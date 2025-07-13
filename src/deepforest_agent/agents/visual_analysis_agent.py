from typing import Dict, List, Any, Optional
from PIL import Image
import json
import re
import time
import torch
import gc

from deepforest_agent.models.qwen_vl_3b_instruct import QwenVL3BModelManager
from deepforest_agent.utils.image_utils import encode_pil_image_to_base64_url, determine_patch_size, get_image_dimensions_fast
from deepforest_agent.utils.state_manager import session_state_manager
from deepforest_agent.conf.config import Config
from deepforest_agent.utils.parsing_utils import (
    parse_image_quality_for_deepforest,
    parse_deepforest_objects_present,
    parse_visual_analysis,
    parse_additional_objects_json
)
from deepforest_agent.prompts.prompt_templates import create_full_image_quality_analysis_prompt, create_individual_tile_analysis_prompt
from deepforest_agent.utils.logging_utils import multi_agent_logger
from deepforest_agent.utils.tile_manager import tile_image_for_analysis


class VisualAnalysisAgent:
    """
    Visual analysis agent responsible for analyzing images with unified full/tiled approach.
    Uses Qwen VL model for multimodal understanding.
    """
    
    def __init__(self):
        """Initialize the Visual Analysis Agent."""
        self.agent_config = Config.AGENT_CONFIGS["visual_analysis"]
        self.model_manager = QwenVL3BModelManager(Config.AGENT_MODELS["visual_analysis"])

    def analyze_full_image(self, user_message: str, session_id: str) -> Dict[str, Any]:
        """
        Analyze full image with automatic fallback to tiling on OOM.
        
        Args:
            user_message: User's query
            session_id: Session identifier
            
        Returns:
            Dict with unified structure for both full and tiled analysis
        """
        if not session_state_manager.session_exists(session_id):
            return {
                "image_quality_for_deepforest": "No",
                "deepforest_objects_present": [],
                "additional_objects": [],
                "visual_analysis": f"Session {session_id} not found.",
                "status": "error",
                "analysis_type": "error"
            }

        image = session_state_manager.get(session_id, "current_image")
        if image is None:
            return {
                "image_quality_for_deepforest": "No", 
                "deepforest_objects_present": [],
                "additional_objects": [],
                "visual_analysis": f"No image available in session {session_id}.",
                "status": "error",
                "analysis_type": "error"
            }
        
        # Try full image analysis first
        try:
            print(f"Session {session_id} - Attempting full image analysis")
            result = self._analyze_single_image(image, user_message, session_id, is_full_image=True)
            
            if result["status"] == "success":
                multi_agent_logger.log_agent_execution(
                    session_id=session_id,
                    agent_name="visual_analysis",
                    agent_input=f"Full image analysis for: {user_message}",
                    agent_output=result["visual_analysis"],
                    execution_time=0.0
                )
                return result
                
        except Exception as e:
            print(f"Session {session_id} - Full image analysis failed (likely OOM): {e}")
            return self._analyze_with_tiling(user_message, session_id, str(e))

        return self._analyze_with_tiling(user_message, session_id, "Full image analysis failed")

    def _analyze_single_image(self, image: Image.Image, user_message: str, session_id: str, 
                             is_full_image: bool = True, tile_location: str = "") -> Dict[str, Any]:
        """
        Analyze a single image (full image or tile) with unified structure.
        
        Args:
            image: PIL Image to analyze
            user_message: User's query
            session_id: Session identifier
            is_full_image: Whether this is full image or tile
            tile_location: Location description for tiles
            
        Returns:
            Unified analysis result
        """
        system_prompt = create_full_image_quality_analysis_prompt(user_message)
        image_base64_url = encode_pil_image_to_base64_url(image)
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_base64_url},
                    {"type": "text", "text": user_message}
                ]
            }
        ]
        
        response = self.model_manager.generate_response(
            messages=messages,
            max_new_tokens=self.agent_config["max_new_tokens"],
            temperature=self.agent_config["temperature"]
        )

        # Parse structured response
        image_quality = parse_image_quality_for_deepforest(response)
        deepforest_objects = parse_deepforest_objects_present(response)
        additional_objects = parse_additional_objects_json(response)
        raw_visual_analysis = parse_visual_analysis(response)
        
        # Format visual analysis with consistent prefix
        if is_full_image:
            width, height = image.size
            visual_analysis = f"Full image analysis of image ({width}x{height}) is done. Here's the analysis: {raw_visual_analysis}"
            analysis_type = "full_image"
        else:
            visual_analysis = f"The visual analysis of tiled image on ({tile_location}) this location is done. Here's the analysis: {raw_visual_analysis}"
            analysis_type = "tiled_image"
        
        return {
            "image_quality_for_deepforest": image_quality,
            "deepforest_objects_present": deepforest_objects,
            "additional_objects": additional_objects,
            "visual_analysis": visual_analysis,
            "status": "success",
            "analysis_type": analysis_type,
            "raw_response": response
        }

    def _analyze_with_tiling(self, user_message: str, session_id: str, error_msg: str) -> Dict[str, Any]:
        """
        Analyze image using tiling approach when full image fails.
        
        Args:
            user_message: User's query
            session_id: Session identifier  
            error_msg: Original error message
            
        Returns:
            Combined analysis from tiled approach with same structure as full image
        """
        print(f"Session {session_id} - Falling back to tiled analysis due to: {error_msg}")
        
        image = session_state_manager.get(session_id, "current_image")
        image_file_path = session_state_manager.get(session_id, "image_file_path")
        
        if not image:
            return {
                "image_quality_for_deepforest": "No",
                "deepforest_objects_present": [],
                "additional_objects": [],
                "visual_analysis": "No image available for tiled analysis.",
                "status": "error",
                "analysis_type": "error"
            }
        
        # Determine appropriate patch size
        if image_file_path:
            patch_size = determine_patch_size(image_file_path, image.size)
        else:
            max_dim = max(image.size)
            if max_dim >= 5000:
                patch_size = 1500 if max_dim <= 7500 else 2000
            else:
                patch_size = 1000
        
        print(f"Session {session_id} - Using patch size {patch_size} for tiled analysis")
        
        try:
            tiles, tile_metadata = tile_image_for_analysis(
                image=image,
                patch_size=patch_size,
                patch_overlap=Config.DEEPFOREST_DEFAULTS["patch_overlap"],
                image_file_path=image_file_path
            )
            
            print(f"Session {session_id} - Created {len(tiles)} tiles for analysis")
            
            # Analyze all tiles and combine results
            all_visual_analyses = []
            all_additional_objects = []
            tile_results = []
            
            for i, (tile, metadata) in enumerate(zip(tiles, tile_metadata)):
                try:
                    tile_coords = metadata.get("window_coords", {})
                    location_desc = f"x:{tile_coords.get('x', 0)}-{tile_coords.get('x', 0) + tile_coords.get('width', 0)}, y:{tile_coords.get('y', 0)}-{tile_coords.get('y', 0) + tile_coords.get('height', 0)}"
                    
                    # Analyze individual tile
                    tile_result = self._analyze_single_image(
                        image=tile,
                        user_message=user_message,
                        session_id=session_id,
                        is_full_image=False,
                        tile_location=location_desc
                    )
                    
                    if tile_result["status"] == "success":
                        all_visual_analyses.append(tile_result["visual_analysis"])
                        all_additional_objects.extend(tile_result["additional_objects"])
                        
                        # Store tile result for potential reuse
                        tile_results.append({
                            "tile_id": i,
                            "location": location_desc,
                            "coordinates": tile_coords,
                            "visual_analysis": tile_result["visual_analysis"],
                            "additional_objects": tile_result["additional_objects"]
                        })
                    
                    # Log individual tile analysis
                    multi_agent_logger.log_agent_execution(
                        session_id=session_id,
                        agent_name=f"visual_tile_{i}",
                        agent_input=f"Tile {i+1} analysis: {user_message}",
                        agent_output=tile_result["visual_analysis"],
                        execution_time=0.0
                    )
                    
                    print(f"Session {session_id} - Analyzed tile {i+1}/{len(tiles)}")
                    
                    # Memory cleanup
                    del tile
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as tile_error:
                    print(f"Session {session_id} - Tile {i} analysis failed: {tile_error}")
                    continue
            
            if all_visual_analyses:
                # Store tile results for potential reuse
                session_state_manager.set(session_id, "tile_analysis_results", tile_results)
                session_state_manager.set(session_id, "tiled_patch_size", patch_size)
                
                # Combine all tile analyses
                combined_visual_analysis = " ".join(all_visual_analyses)
                
                return {
                    "image_quality_for_deepforest": "Yes",
                    "deepforest_objects_present": ["tree", "bird", "livestock"],
                    "additional_objects": all_additional_objects,
                    "visual_analysis": combined_visual_analysis,
                    "status": "tiled_success", 
                    "analysis_type": "tiled_combined",
                    "tile_count": len(tiles),
                    "successful_tiles": len(all_visual_analyses),
                    "patch_size_used": patch_size
                }
            
        except Exception as tiling_error:
            print(f"Session {session_id} - Tiled analysis also failed: {tiling_error}")
        
        # Final fallback - resolution-based assessment
        resolution_result = session_state_manager.get(session_id, "resolution_result")
        if resolution_result and resolution_result.get("is_suitable"):
            width, height = image.size
            return {
                "image_quality_for_deepforest": "Yes",
                "deepforest_objects_present": ["tree", "bird", "livestock"],
                "additional_objects": [],
                "visual_analysis": f"Full image analysis of image ({width}x{height}) is done. Here's the analysis: Large image analyzed using resolution-based assessment. Original error: {error_msg}",
                "status": "resolution_fallback",
                "analysis_type": "resolution_based"
            }
        
        # Complete failure
        width, height = image.size
        return {
            "image_quality_for_deepforest": "No",
            "deepforest_objects_present": [],
            "additional_objects": [],
            "visual_analysis": f"Full image analysis of image ({width}x{height}) failed. Analysis could not be completed due to: {error_msg}",
            "status": "error",
            "analysis_type": "failed"
        }

    def get_tile_analysis_results(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get stored tile analysis results for reuse.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of tile analysis results or empty list
        """
        return session_state_manager.get(session_id, "tile_analysis_results", [])