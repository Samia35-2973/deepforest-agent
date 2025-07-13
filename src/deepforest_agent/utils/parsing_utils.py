import json
import re
from typing import Dict, List, Any, Optional


def parse_image_quality_for_deepforest(response: str) -> str:
    """
    Parse IMAGE_QUALITY_FOR_DEEPFOREST from response.
    
    Args:
        response: Model response text
        
    Returns:
        "Yes" or "No"
    """
    quality_match = re.search(r'(?:\*\*)?IMAGE_QUALITY_FOR_DEEPFOREST[:\*\s]+\[?(YES|NO|Yes|No|yes|no)\]?', response, re.IGNORECASE)
    if quality_match:
        quality_value = quality_match.group(1).upper()
        return "Yes" if quality_value == "YES" else "No"
    return "No"

def parse_deepforest_objects_present(response: str) -> List[str]:
    """
    Parse DEEPFOREST_OBJECTS_PRESENT from response.
    
    Args:
        response: Model response text
        
    Returns:
        List of objects present
    """
    objects_match = re.search(r'(?:\*\*)?DEEPFOREST_OBJECTS_PRESENT[:\*\s]+(\[.*?\])', response, re.DOTALL)
    if objects_match:
        try:
            objects_str = objects_match.group(1)
            objects_str = re.sub(r'[`\'"]', '"', objects_str)
            objects_list = json.loads(objects_str)
            
            allowed_objects = ["bird", "tree", "livestock"]
            validated_objects = [obj for obj in objects_list if obj in allowed_objects]
            return validated_objects
        except json.JSONDecodeError:
            objects_str = objects_match.group(1)
            manual_objects = re.findall(r'"(bird|tree|livestock)"', objects_str)
            return list(set(manual_objects))
    return []


def parse_additional_objects_json(response: str) -> List[Dict[str, Any]]:
    """
    Parse ADDITIONAL_OBJECTS_JSON from response.
    
    Args:
        response: Model response text
        
    Returns:
        List of additional objects with coordinates
    """
    additional_match = re.search(r'(?:\*\*)?ADDITIONAL_OBJECTS_JSON[:\*\s]+(.*?)(?=\n(?:\*\*)?(?:VISUAL_ANALYSIS|IMAGE_QUALITY|DEEPFOREST_OBJECTS)|$)', response, re.DOTALL)
    if additional_match:
        try:
            additional_str = additional_match.group(1).strip()
            if additional_str.startswith('```json'):
                additional_str = additional_str[7:]
            if additional_str.startswith('```'):
                additional_str = additional_str[3:]
            if additional_str.endswith('```'):
                additional_str = additional_str[:-3]
            
            additional_str = additional_str.strip()
            
            if additional_str.startswith('[') and additional_str.endswith(']'):
                additional_objects = json.loads(additional_str)
                if isinstance(additional_objects, list):
                    return additional_objects
            else:
                additional_objects = []
                for line in additional_str.split('\n'):
                    line = line.strip().rstrip(',')
                    if line and line.startswith('{') and line.endswith('}'):
                        try:
                            obj = json.loads(line)
                            additional_objects.append(obj)
                        except json.JSONDecodeError:
                            continue
                return additional_objects
                
        except Exception as e:
            print(f"Error parsing additional objects JSON: {e}")
    return []


def parse_visual_analysis(response: str) -> str:
    """
    Parse VISUAL_ANALYSIS from response.
    
    Args:
        response: Model response text
        
    Returns:
        Visual analysis text
    """
    analysis_match = re.search(r'(?:\*\*)?VISUAL_ANALYSIS[:\*\s]+(.*?)(?=\n(?:\*\*)?(?:IMAGE_QUALITY|DEEPFOREST_OBJECTS|ADDITIONAL_OBJECTS)|$)', response, re.IGNORECASE | re.DOTALL)
    if analysis_match:
        return analysis_match.group(1).strip()
    else:
        fallback_match = re.search(r'(?:\*\*)?VISUAL_ANALYSIS[:\*\s]+(.*)', response, re.IGNORECASE | re.DOTALL)
        if fallback_match:
            return fallback_match.group(1).strip()
    return response


def parse_deepforest_agent_response_with_reasoning(response: str) -> Dict[str, Any]:
    """
    Parse DeepForest detector agent response with reasoning.
    
    Args:
        response: Model response text
        
    Returns:
        Dictionary with reasoning and tool calls
    """
    from deepforest_agent.tools.tool_handler import extract_all_tool_calls
    
    try:
        tool_calls = extract_all_tool_calls(response)
        
        if not tool_calls:
            return {"error": "No valid tool calls found in response"}
        
        reasoning_text = ""
        first_json_match = re.search(r'\{[^}]*"name"[^}]*"arguments"[^}]*\}', response)
        
        if first_json_match:
            reasoning_text = response[:first_json_match.start()].strip()
            reasoning_text = re.sub(r'^(REASONING:|Reasoning:|Analysis:|\*\*REASONING:\*\*)', '', reasoning_text).strip()
            
        if not reasoning_text:
            reasoning_text = "Tool calls generated based on analysis"
        
        return {
            "reasoning": reasoning_text,
            "tool_calls": tool_calls
        }
        
    except Exception as e:
        return {"error": f"Unexpected error parsing response: {str(e)}"}

def parse_memory_agent_response(response: str) -> Dict[str, Any]:
    """
    Parse memory agent structured response format with new TOOL_CACHE_ID field.
    
    Args:
        response: Model response text
        
    Returns:
        Dictionary with answer_present, direct_answer, tool_cache_id, and relevant_context
    """
    try:
        # Parse ANSWER_PRESENT
        answer_present_match = re.search(r'(?:\*\*)?ANSWER_PRESENT:(?:\*\*)?\s*\[?(YES|NO)\]?', response, re.IGNORECASE)
        answer_present = False
        if answer_present_match:
            answer_present = answer_present_match.group(1).upper() == "YES"
        
        # Parse TOOL_CACHE_ID
        tool_cache_id_match = re.search(r'(?:\*\*)?TOOL_CACHE_ID:(?:\*\*)?\s*(.*?)(?=\n(?:\*\*)?(?:RELEVANT_CONTEXT|$))', response, re.IGNORECASE | re.DOTALL)
        tool_cache_id = None

        if tool_cache_id_match:
            tool_cache_id_text = tool_cache_id_match.group(1).strip()
            
            # Extract all cache IDs using multiple patterns
            cache_ids = []
            
            # Pattern 1: IDs within brackets [id1, id2, ...]
            bracket_pattern = r'\[([^\[\]]*)\]'
            bracket_matches = re.findall(bracket_pattern, tool_cache_id_text)
            for bracket_content in bracket_matches:
                if bracket_content.strip():  # Skip empty brackets
                    # Extract hex IDs from bracket content
                    hex_ids = re.findall(r'([a-fA-F0-9]{8,})', bracket_content)
                    cache_ids.extend(hex_ids)
            
            # Pattern 2: Direct hex IDs (not in brackets)
            # Remove bracketed content first, then find remaining hex IDs
            text_without_brackets = re.sub(r'\[[^\[\]]*\]', '', tool_cache_id_text)
            direct_hex_ids = re.findall(r'([a-fA-F0-9]{8,})', text_without_brackets)
            cache_ids.extend(direct_hex_ids)
            
            # Pattern 3: Standalone hex IDs on separate lines (check the whole response)
            standalone_pattern = r'^([a-fA-F0-9]{8,})$'
            standalone_matches = re.findall(standalone_pattern, response, re.MULTILINE)
            cache_ids.extend(standalone_matches)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_cache_ids = []
            for cache_id in cache_ids:
                if cache_id not in seen:
                    seen.add(cache_id)
                    unique_cache_ids.append(cache_id)
            
            if unique_cache_ids:
                tool_cache_id = ", ".join(unique_cache_ids) if len(unique_cache_ids) > 1 else unique_cache_ids[0]
            elif tool_cache_id_text and tool_cache_id_text.lower() not in ["", "empty", "none", "no tool cache id"]:
                tool_cache_id = tool_cache_id_text
        
        # Parse RELEVANT_CONTEXT
        context_match = re.search(
            r'(?:\*\*)?RELEVANT_CONTEXT:(?:\*\*)?\s*(.*?)(?=\n\*\*[A-Z_]+:|\Z)',
            response,
            re.IGNORECASE | re.DOTALL
        )

        relevant_context = ""
        if context_match:
            relevant_context = context_match.group(1).strip()
        elif not answer_present:
            relevant_context = response
        
        return {
            "answer_present": answer_present,
            "direct_answer": "YES" if answer_present else "NO",
            "tool_cache_id": tool_cache_id,
            "relevant_context": relevant_context,
            "raw_response": response
        }
        
    except Exception as e:
        print(f"Error parsing memory response: {e}")
        return {
            "answer_present": False,
            "direct_answer": "NO",
            "tool_cache_id": None,
            "relevant_context": response,
            "raw_response": response
        }