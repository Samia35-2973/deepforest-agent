import os
import hashlib
from typing import Tuple, Optional

class FileManager:
    """
    Centralized file operations handler to eliminate repetitive file handling code.
    
    This class consolidates all file-related operations including existence checking,
    and hash computation, for file metadata operations throughout the agent.
    """

    @staticmethod
    def validate_and_extract_info(image_path: str) -> Tuple[Optional[str], str]:
        """
        Validate file existence and extract metadata in a single operation.
        
        This method combines existence checking, and hash computation to 
        eliminate repetitive file validation patterns.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            image_hash: MD5 hash of the file content, or None if file doesn't exist
        """
        if not image_path or not os.path.exists(image_path):
            return None, '.unknown'
        
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                image_hash = hashlib.md5(image_data).hexdigest()
            
            return image_hash
        except (OSError, IOError) as e:
            print(f"Error accessing file {image_path}: {e}")
            return None