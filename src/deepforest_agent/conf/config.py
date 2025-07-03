import os

class Config:
    DEEPFOREST_MODELS = {
        "bird": "weecology/deepforest-bird",
        "tree": "weecology/deepforest-tree",
        "livestock": "weecology/deepforest-livestock"
    }

    COLORS = {
        "bird": (0, 0, 255),      # Red (BGR)
        "tree": (0, 255, 0),      # Green (BGR)
        "livestock": (255, 0, 0), # Blue (BGR)
        "alive_tree": (255, 255, 0), # Cyan (BGR)
        "dead_tree": (0, 165, 255) # Orange (BGR)
    }
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    NO_ALBUMENTATIONS_UPDATE = os.getenv("NO_ALBUMENTATIONS_UPDATE", "")