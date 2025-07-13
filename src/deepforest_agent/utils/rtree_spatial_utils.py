import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from rtree import index
import pandas as pd


class DetectionSpatialAnalyzer:
    """
    Spatial analyzer using R-tree for DeepForest detection results.
    """
    
    def __init__(self, image_width: int, image_height: int):
        """
        Initialize spatial analyzer with image dimensions.
        
        Args:
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
        """
        self.image_width = image_width
        self.image_height = image_height
        self.spatial_index = index.Index()
        self.detections = []
    
    def add_detections(self, detections_list: List[Dict[str, Any]]) -> None:
        """
        Add detections to R-tree spatial index.
        
        Args:
            detections_list: List of detection dictionaries with coordinates
        """
        for i, detection in enumerate(detections_list):
            xmin = detection.get('xmin', 0)
            ymin = detection.get('ymin', 0)
            xmax = detection.get('xmax', 0)
            ymax = detection.get('ymax', 0)

            # Validate box ordering - swap if necessary
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            if ymin > ymax:
                ymin, ymax = ymax, ymin
            
            # Clamp to image bounds
            xmin = max(0, min(xmin, self.image_width))
            ymin = max(0, min(ymin, self.image_height))
            xmax = max(0, min(xmax, self.image_width))
            ymax = max(0, min(ymax, self.image_height))
            
            # Skip invalid boxes (zero area after validation)
            if xmin >= xmax or ymin >= ymax:
                continue
            
            # Add to R-tree index
            self.spatial_index.insert(i, (xmin, ymin, xmax, ymax))
            
            # Store detection with spatial info
            detection_copy = detection.copy()
            detection_copy['detection_id'] = i
            detection_copy['centroid_x'] = (xmin + xmax) / 2
            detection_copy['centroid_y'] = (ymin + ymax) / 2
            detection_copy['area'] = (xmax - xmin) * (ymax - ymin)
            self.detections.append(detection_copy)
    
    def get_grid_analysis(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze detections using 3x3 grid system.
        
        Returns:
            Dictionary with analysis for each grid cell
        """
        grid_width = self.image_width / 3
        grid_height = self.image_height / 3
        
        grid_names = {
            (0, 0): "Top-Left (Northwest)", (1, 0): "Top-Center (North)", (2, 0): "Top-Right (Northeast)",
            (0, 1): "Middle-Left (West)", (1, 1): "Center", (2, 1): "Middle-Right (East)",
            (0, 2): "Bottom-Left (Southwest)", (1, 2): "Bottom-Center (South)", (2, 2): "Bottom-Right (Southeast)"
        }
        
        grid_analysis = {}
        
        for (grid_x, grid_y), grid_name in grid_names.items():
            # Define grid bounds
            x_min = grid_x * grid_width
            y_min = grid_y * grid_height
            x_max = (grid_x + 1) * grid_width
            y_max = (grid_y + 1) * grid_height
            
            # Query R-tree for intersecting detections
            intersecting_ids = list(self.spatial_index.intersection((x_min, y_min, x_max, y_max)))
            grid_detections = [self.detections[i] for i in intersecting_ids]
            
            # Analyze by confidence categories
            confidence_analysis = self._analyze_confidence_categories(grid_detections)
            
            grid_analysis[grid_name] = {
                "total_detections": len(grid_detections),
                "confidence_analysis": confidence_analysis,
                "bounds": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}
            }
        
        return grid_analysis
    
    def _analyze_confidence_categories(self, detections: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze detections by confidence categories.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Analysis by confidence categories (Low, Medium, High)
        """
        categories = {
            "Detections with Low Confidence Score (0.0-0.3)": {"detections": [], "range": (0.0, 0.3)},
            "Detections with Medium Confidence Score (0.3-0.7)": {"detections": [], "range": (0.3, 0.7)},
            "Detections with High Confidence Score (0.7-1.0)": {"detections": [], "range": (0.7, 1.0)}
        }
        
        for detection in detections:
            score = detection.get('score', 0.0)
            if score < 0.3:
                categories["Detections with Low Confidence Score (0.0-0.3)"]["detections"].append(detection)
            elif score < 0.7:
                categories["Detections with Medium Confidence Score (0.3-0.7)"]["detections"].append(detection)
            else:
                categories["Detections with High Confidence Score (0.7-1.0)"]["detections"].append(detection)
        
        # Calculate statistics for each category
        analysis = {}
        for category_name, category_data in categories.items():
            cat_detections = category_data["detections"]
            if cat_detections:
                areas = [d['area'] for d in cat_detections]
                analysis[category_name] = {
                    "count": len(cat_detections),
                    "avg_area": np.mean(areas),
                    "min_area": np.min(areas),
                    "max_area": np.max(areas),
                    "total_area_covered": np.sum(areas),
                    "labels": [d.get('label', 'unknown') for d in cat_detections]
                }
            else:
                analysis[category_name] = {
                    "count": 0,
                    "avg_area": 0,
                    "min_area": 0,
                    "max_area": 0,
                    "total_area_covered": 0,
                    "labels": []
                }
        
        return analysis
    
    def analyze_spatial_relationships_with_indexing(self, confidence_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Analyze spatial relationships using R-tree indexing for confidence >= 0.3 detections.
        
        Args:
            confidence_threshold: Minimum confidence score (default: 0.3)
            
        Returns:
            List of spatial relationship dictionaries with intersection and nearest neighbor data
        """
        # Filter detections by confidence threshold
        high_confidence_detections = [
            d for d in self.detections 
            if d.get('score', 0.0) >= confidence_threshold
        ]
        
        if not high_confidence_detections:
            return []
        
        relationships = []
        
        for detection in high_confidence_detections:
            # Get bounding box coordinates directly
            xmin = detection.get('xmin', 0)
            ymin = detection.get('ymin', 0) 
            xmax = detection.get('xmax', 0)
            ymax = detection.get('ymax', 0)
            detection_id = detection.get('detection_id', 0)
            
            # Get object label (handle classification labels for trees)
            if 'classification_label' in detection and detection['classification_label'] and str(detection['classification_label']).lower() != 'nan':
                object_label = detection['classification_label']
            else:
                object_label = detection.get('label', 'unknown')
            
            # Find intersecting objects using spatial index
            intersecting_ids = list(self.spatial_index.intersection((xmin, ymin, xmax, ymax)))
            
            # Remove self from intersections
            intersecting_ids = [idx for idx in intersecting_ids if idx != detection_id]
            
            # Get details of intersecting objects
            intersecting_objects = []
            for idx in intersecting_ids:
                if idx < len(self.detections):
                    intersecting_detection = self.detections[idx]
                    if intersecting_detection.get('score', 0.0) >= confidence_threshold:
                        if 'classification_label' in intersecting_detection and intersecting_detection['classification_label'] and str(intersecting_detection['classification_label']).lower() != 'nan':
                            intersecting_label = intersecting_detection['classification_label']
                        else:
                            intersecting_label = intersecting_detection.get('label', 'unknown')
                        intersecting_objects.append(intersecting_label)
            
            # Find nearest neighbor using spatial index
            nearest_ids = list(self.spatial_index.nearest((xmin, ymin, xmax, ymax), 2))  # 2 to get self + nearest
            nearest_neighbor = None
            
            for idx in nearest_ids:
                if idx != detection_id and idx < len(self.detections):
                    nearest_detection = self.detections[idx]
                    if nearest_detection.get('score', 0.0) >= confidence_threshold:
                        if 'classification_label' in nearest_detection and nearest_detection['classification_label'] and str(nearest_detection['classification_label']).lower() != 'nan':
                            nearest_label = nearest_detection['classification_label']
                        else:
                            nearest_label = nearest_detection.get('label', 'unknown')
                        nearest_neighbor = nearest_label
                        break
            
            # Determine grid region
            grid_region = self._determine_grid_region(detection)
            
            # Count intersecting objects by type
            object_counts = {}
            for obj_label in intersecting_objects:
                object_counts[obj_label] = object_counts.get(obj_label, 0) + 1
            
            relationships.append({
                'object_type': object_label,
                'object_location': f"({ymin}, {xmin})",
                'grid_region': grid_region,
                'intersecting_objects': object_counts,
                'nearest_neighbor': nearest_neighbor,
                'confidence_score': detection.get('score', 0.0),
                'total_intersections': len(intersecting_objects)
            })
        
        return relationships

    def _determine_grid_region(self, detection: Dict[str, Any]) -> str:
        """
        Determine which grid region a detection belongs to based on its centroid.
        
        Args:
            detection: Detection dictionary with coordinates
            
        Returns:
            Grid region name (e.g., "northern", "northwest", etc.)
        """
        centroid_x = detection.get('centroid_x', 0)
        centroid_y = detection.get('centroid_y', 0)
        
        grid_width = self.image_width / 3
        grid_height = self.image_height / 3
        
        # Determine grid position
        grid_x = int(centroid_x // grid_width)
        grid_y = int(centroid_y // grid_height)
        
        # Ensure within bounds
        grid_x = min(2, max(0, grid_x))
        grid_y = min(2, max(0, grid_y))
        
        grid_names = {
            (0, 0): "northwestern", (1, 0): "northern", (2, 0): "northeastern",
            (0, 1): "western", (1, 1): "central", (2, 1): "eastern", 
            (0, 2): "southwestern", (1, 2): "southern", (2, 2): "southeastern"
        }
        
        return grid_names.get((grid_x, grid_y), "central")

    def generate_spatial_narrative(self, confidence_threshold: float = 0.3) -> str:
        """
        Generate narrative description of spatial relationships using R-tree analysis.
        
        Args:
            confidence_threshold: Minimum confidence score for analysis (default: 0.3)
            
        Returns:
            Natural language narrative of spatial relationships
        """
        relationships = self.analyze_spatial_relationships_with_indexing(confidence_threshold)
        
        if not relationships:
            return f"No objects with confidence score >= {confidence_threshold} found for spatial relationship analysis."
        
        narrative_parts = []
        
        # Process each relationship and only include different object types
        for rel in relationships:
            object_type = rel['object_type']
            confidence_score = rel['confidence_score']
            grid_region = rel['grid_region']
            object_location = rel['object_location']
            
            # Only process intersecting objects that are DIFFERENT from the main object
            different_intersecting = {}
            for intersecting_type, count in rel['intersecting_objects'].items():
                if intersecting_type != object_type:  # Only different object types
                    different_intersecting[intersecting_type] = count
            
            # Generate narrative for intersecting different objects
            if different_intersecting:
                intersecting_parts = []
                for obj_label, count in different_intersecting.items():
                    if count == 1:
                        intersecting_parts.append(f"{count} {obj_label.replace('_', ' ')}")
                    else:
                        intersecting_parts.append(f"{count} {obj_label.replace('_', ' ')}s")
                
                intersecting_desc = ", ".join(intersecting_parts)
                
                narrative_parts.append(
                    f"I am about {confidence_score*100:.1f}% confident that, in {grid_region} region, "
                    f"{intersecting_desc} found overlapping around the {object_type.replace('_', ' ')} "
                    f"object at location (top, left) = {object_location}.\n"
                )
            
            # Only add nearest neighbor information if it's a DIFFERENT object type
            if rel['nearest_neighbor'] and rel['nearest_neighbor'] != object_type:
                narrative_parts.append(
                    f"I am about {confidence_score*100:.1f}% confident that, in {grid_region} region, "
                    f"around the {object_type.replace('_', ' ')} at location (top, left) = {object_location} "
                    f"the nearest neighbor is a {rel['nearest_neighbor'].replace('_', ' ')}.\n"
                )
        
        if narrative_parts:
            # Remove duplicates while preserving order
            unique_narratives = []
            seen = set()
            for part in narrative_parts:
                if part not in seen:
                    unique_narratives.append(part)
                    seen.add(part)
            
            return " ".join(unique_narratives)
        else:
            return f"Spatial analysis completed for {len(relationships)} objects with confidence >= {confidence_threshold}, but no significant spatial relationships between different object types detected."
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive detection statistics.
        
        Returns:
            Dictionary with overall statistics
        """
        if not self.detections:
            return {"total_count": 0}
        
        # Basic counts and confidence
        total_count = len(self.detections)
        scores = [d.get('score', 0.0) for d in self.detections]
        overall_confidence = np.mean(scores)
        
        # Size statistics
        areas = [d['area'] for d in self.detections]
        avg_area = np.mean(areas)
        min_area = np.min(areas)
        max_area = np.max(areas)
        total_area = np.sum(areas)
        
        # Label distribution
        labels = [d.get('label', 'unknown') for d in self.detections]
        # Handle classification labels for trees
        classified_labels = []
        for d in self.detections:
            if 'classification_label' in d and d['classification_label'] and str(d['classification_label']).lower() != 'nan':
                classified_labels.append(d['classification_label'])
            else:
                classified_labels.append(d.get('label', 'unknown'))
        
        from collections import Counter
        label_counts = Counter(classified_labels)
        
        return {
            "total_count": total_count,
            "overall_confidence": overall_confidence,
            "size_stats": {
                "avg_area": avg_area,
                "min_area": min_area,
                "max_area": max_area,
                "total_area_covered": total_area
            },
            "label_distribution": dict(label_counts),
            "confidence_distribution": {
                "low_count": len([s for s in scores if s < 0.3]),
                "medium_count": len([s for s in scores if 0.3 <= s < 0.7]),
                "high_count": len([s for s in scores if s >= 0.7])
            }
        }