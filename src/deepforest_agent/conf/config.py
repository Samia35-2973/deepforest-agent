import os

class Config:
    """
    Configuration class defining DeepForest model paths, visualization colors, and agent models.
    """

    DEEPFOREST_MODELS = {
        "bird": "weecology/deepforest-bird",
        "tree": "weecology/deepforest-tree",
        "livestock": "weecology/deepforest-livestock"
    }

    DEEPFOREST_DEFAULTS = {
        "patch_size": 400,
        "patch_overlap": 0.05,
        "iou_threshold": 0.15,
        "thresh": 0.55,
        "alive_dead_trees": False
    }

    COLORS = {
        "bird": (0, 0, 255),      # Red (BGR)
        "tree": (0, 255, 0),      # Green (BGR)
        "livestock": (255, 0, 0), # Blue (BGR)
        "alive_tree": (255, 255, 0), # Cyan (BGR)
        "dead_tree": (0, 165, 255) # Orange (BGR)
    }

    AGENT_MODELS = {
        "memory": "HuggingFaceTB/SmolLM3-3B",
        "deepforest_detector": "HuggingFaceTB/SmolLM3-3B",
        "visual_analysis": "Qwen/Qwen2.5-VL-3B-Instruct",
        "ecology_analysis": "meta-llama/Llama-3.2-3B-Instruct"
    }

    # Agent-specific generation parameters
    AGENT_CONFIGS = {
        "memory": {
            "max_new_tokens": 16000,
            "temperature": 0.6,
            "top_p": 0.95
        },
        "deepforest_detector": {
            "max_new_tokens": 16000, 
            "temperature": 0.6,
            "top_p": 0.95
        },
        "visual_analysis": {
            "max_new_tokens": 5000,
            "temperature": 0.1
        },
        "ecology_analysis": {
            "max_new_tokens": 16000,
            "temperature": 0.6,
            "top_p": 0.95
        }
    }

    NO_ALBUMENTATIONS = os.getenv("NO_ALBUMENTATIONS", "")