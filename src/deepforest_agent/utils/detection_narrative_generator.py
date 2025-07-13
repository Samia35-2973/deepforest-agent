import numpy as np
from typing import List, Dict, Any
from collections import Counter, defaultdict

from deepforest_agent.utils.rtree_spatial_utils import DetectionSpatialAnalyzer


class DetectionNarrativeGenerator:
    """
    Generates natural language narratives from DeepForest detection results with proper classification handling.
    """
    
    def __init__(self, image_width: int, image_height: int):
        """
        Initialize narrative generator with image dimensions.
        
        Args:
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
        """
        self.image_width = image_width
        self.image_height = image_height
        self.spatial_analyzer = DetectionSpatialAnalyzer(image_width, image_height)
    
    def generate_comprehensive_narrative(self, detections_list: List[Dict[str, Any]]) -> str:
        """
        Generate comprehensive detection narrative using spatial analysis with proper classification handling.
        
        Args:
            detections_list: List of detection dictionaries from DeepForest
            
        Returns:
            Natural language narrative describing all aspects of detections
        """
        if not detections_list:
            return "No objects were detected by DeepForest in this image."
        
        # Add detections to spatial analyzer
        self.spatial_analyzer.add_detections(detections_list)
        
        # Get comprehensive statistics
        stats = self.spatial_analyzer.get_detection_statistics()
        grid_analysis = self.spatial_analyzer.get_grid_analysis()

        narrative_parts = []
        
        # 1. Overall Summary with proper classification handling
        narrative_parts.append(self._generate_overall_summary(detections_list))
        
        # 2. Confidence Analysis
        narrative_parts.append(self._generate_confidence_analysis(detections_list))
        
        # 3. Spatial Distribution Analysis
        narrative_parts.append(self._generate_spatial_distribution_narrative(grid_analysis, detections_list))
        
        # 4. Spatial Relationships Analysis using R-tree indexing
        narrative_parts.append(self._generate_spatial_relationships_narrative(detections_list))
        
        # 5. Object Coverage Analysis
        narrative_parts.append(self._generate_coverage_analysis(detections_list))
        
        return "\n\n".join(narrative_parts)
    
    def _generate_overall_summary(self, detections_list: List[Dict[str, Any]]) -> str:
        """
        Generate overall detection summary with proper classification handling.
        
        Args:
            detections_list (List[Dict[str, Any]]): List of all detection results
            
        Returns:
            str: Formatted summary section including:
                - Total detection count and average confidence
                - Base object counts (birds, trees, livestock)
                - Tree classification breakdown (alive trees, dead trees)
        """
        total_count = len(detections_list)
        
        # Calculate overall confidence
        scores = [s for s in (d.get("score") for d in detections_list) if s is not None and np.isfinite(s)]
        overall_confidence = float(np.mean(scores)) if scores else 0.0
        
        # Proper object counting with classification handling
        base_label_counts = {}  # bird, tree, livestock
        classification_counts = {}  # alive_tree, dead_tree
        
        for detection in detections_list:
            base_label = detection.get('label', 'unknown')
            base_label_counts[base_label] = base_label_counts.get(base_label, 0) + 1
            
            # Handle tree classifications
            if base_label == 'tree':
                classification_label = detection.get('classification_label')
                classification_score = detection.get('classification_score')
                
                # Only count valid classifications (not NaN or None)
                if (classification_label and 
                    classification_score is not None and
                    str(classification_label).lower() != 'nan' and
                    str(classification_score).lower() != 'nan'):
                    
                    classification_counts[classification_label] = classification_counts.get(classification_label, 0) + 1
        
        summary = f"**Overall Detection Summary**\n"
        summary += f"In the whole image, {total_count} objects were detected with an average confidence of {overall_confidence:.3f}.\n\n"
        
        # Object breakdown with proper classification display
        object_parts = []
        for label, count in base_label_counts.items():
            label_name = label.replace('_', ' ')
            
            if label == 'tree' and classification_counts:
                # Special handling for trees with classifications
                total_trees = count
                classified_trees = sum(classification_counts.values())
                
                if classified_trees > 0:
                    tree_part = f"{total_trees} trees are detected"
                    classification_parts = []
                    for class_label, class_count in classification_counts.items():
                        class_name = class_label.replace('_', ' ')
                        classification_parts.append(f"{class_count} {class_name}s")
                    
                    tree_part += f". These {total_trees} trees are classified as {' and '.join(classification_parts)}"
                    object_parts.append(tree_part)
                else:
                    object_parts.append(f"{count} {label_name}{'s' if count != 1 else ''}")
            else:
                object_parts.append(f"{count} {label_name}{'s' if count != 1 else ''}")
        
        summary += "Whole image Object breakdown: " + ", ".join(object_parts) + "."
        
        return summary
    
    def _generate_confidence_analysis(self, detections_list: List[Dict[str, Any]]) -> str:
        """
        Generate confidence-based analysis with proper classification handling.
        
        Args:
            detections_list (List[Dict[str, Any]]): List of all detection results
            
        Returns:
            str: Formatted confidence analysis section including:
                - Object counts per confidence range
                - Base object type breakdown within each range
                - Tree classification details (alive/dead) within each range
        """
        # Group by confidence ranges
        confidence_groups = {
            "Detections with High Confidence Score (0.7-1.0)": [],
            "Detections with Medium Confidence Score (0.3-0.7)": [],
            "Detections with Low Confidence Score (0.0-0.3)": []
        }
        
        for detection in detections_list:
            score = detection.get('score', 0.0)
            if score >= 0.7:
                confidence_groups["Detections with High Confidence Score (0.7-1.0)"].append(detection)
            elif score >= 0.3:
                confidence_groups["Detections with Medium Confidence Score (0.3-0.7)"].append(detection)
            else:
                confidence_groups["Detections with Low Confidence Score (0.0-0.3)"].append(detection)
        
        narrative = f"**Whole image Confidence Score Analysis**\n"
        
        for conf_range, detections in confidence_groups.items():
            if not detections:
                narrative += f"{conf_range}: No objects detected\n"
                continue
                
            count = len(detections)
            narrative += f"{conf_range}: {count} objects detected in the whole image\n"
            
            # Count by base labels and classifications
            base_counts = {}
            class_counts = {}
            
            for detection in detections:
                base_label = detection.get('label', 'unknown')
                base_counts[base_label] = base_counts.get(base_label, 0) + 1
                
                if base_label == 'tree':
                    classification_label = detection.get('classification_label')
                    if (classification_label and 
                        str(classification_label).lower() != 'nan'):
                        class_counts[classification_label] = class_counts.get(classification_label, 0) + 1
            
            # Display breakdown
            breakdown_parts = []
            for label, label_count in base_counts.items():
                if label == 'tree' and class_counts:
                    tree_part = f"{label_count} trees"
                    class_parts = []
                    for class_label, class_count in class_counts.items():
                        class_name = class_label.replace('_', ' ')
                        class_parts.append(f"{class_count} {class_name}s")
                    if class_parts:
                        tree_part += f" ({', '.join(class_parts)})"
                    breakdown_parts.append(tree_part)
                else:
                    label_name = label.replace('_', ' ')
                    breakdown_parts.append(f"{label_count} {label_name}{'s' if label_count != 1 else ''}")
            
            narrative += f"  - {', '.join(breakdown_parts)}\n"
        
        return narrative
    
    def _generate_spatial_distribution_narrative(self, grid_analysis: Dict[str, Dict[str, Any]], detections_list: List[Dict[str, Any]]) -> str:
        """
        Generate spatial distribution narrative using 9-grid analysis
        
        Args:
            grid_analysis (Dict[str, Dict[str, Any]]): Pre-computed grid analysis from spatial_analyzer
                containing detection counts and confidence analysis for each grid section
            detections_list (List[Dict[str, Any]]): Original detection list for additional processing
            
        Returns:
            str: Formatted spatial distribution section including:
                - Grid-by-grid object analysis with confidence breakdowns
                - Tree classification details within each grid section
                - Density pattern identification (dense vs sparse regions)
        """
        narrative = f"**Spatial Distribution Analysis**\n"
        narrative += f"The image is divided into nine grid sections for spatial analysis:\n\n"
        
        # Grid-by-grid analysis
        for grid_name, grid_data in grid_analysis.items():
            total_dets = grid_data['total_detections']
            conf_analysis = grid_data['confidence_analysis']
            
            if total_dets == 0:
                narrative += f"{grid_name}: No objects detected\n"
                continue
            
            narrative += f"{grid_name}: {total_dets} objects detected\n"
            
            # Per confidence category analysis
            for conf_category, conf_data in conf_analysis.items():
                if conf_data['count'] > 0:
                    # Count base labels and classifications for this grid/confidence
                    grid_detections = [d for d in detections_list 
                                    if self._detection_in_grid(d, grid_data['bounds'])]
                    
                    conf_range = self._get_confidence_range(conf_category)
                    conf_detections = [d for d in grid_detections 
                                    if conf_range[0] <= d.get('score', 0) < conf_range[1] or
                                    (conf_range[1] == 1.0 and d.get('score', 0) == 1.0)]
                    
                    base_counts, class_counts = self._count_labels_with_classification(conf_detections)
                    
                    # Display object breakdown
                    object_desc = []
                    for label, count in base_counts.items():
                        if label == 'tree' and label in class_counts:
                            tree_desc = f"{count} trees"
                            if class_counts[label]:
                                class_parts = []
                                for class_label, class_count in class_counts[label].items():
                                    class_name = class_label.replace('_', ' ')
                                    class_parts.append(f"{class_count} {class_name}s")
                                tree_desc += f" ({', '.join(class_parts)})"
                            object_desc.append(tree_desc)
                        else:
                            label_name = label.replace('_', ' ')
                            object_desc.append(f"{count} {label_name}{'s' if count != 1 else ''}")
                    
                    # Simple description
                    narrative += f"  - {conf_category}: {', '.join(object_desc)}\n"
            
            narrative += "\n"
        
        # Overall density patterns
        grid_counts = {name: data['total_detections'] for name, data in grid_analysis.items()}
        avg_count = sum(grid_counts.values()) / len(grid_counts) if grid_counts else 0
        
        dense_regions = [name for name, count in grid_counts.items() if count > avg_count]
        sparse_regions = [name for name, count in grid_counts.items() if count < avg_count]
        
        if dense_regions or sparse_regions:
            narrative += "**Density Patterns:**\n"
            if dense_regions:
                narrative += f"Dense regions: {', '.join(dense_regions)}\n"
            if sparse_regions:
                narrative += f"Sparse regions: {', '.join(sparse_regions)}\n"
        
        return narrative
    
    def _generate_coverage_analysis(self, detections_list: List[Dict[str, Any]]) -> str:
        """
        Generate object coverage analysis broken down by object type.
        
        Args:
            detections_list (List[Dict[str, Any]]): List of all detection results
            
        Returns:
            str: Formatted coverage analysis including:
                - Percentage coverage for each object type (birds, trees, livestock)
                - Tree classification coverage breakdown (alive trees vs dead trees)
                - Total area calculations relative to full image
        """
        narrative = f"**Object Coverage Analysis**\n"
        
        total_image_area = self.image_width * self.image_height
        
        # Calculate coverage by object type
        base_coverage = {}
        classification_coverage = {}
        
        for detection in detections_list:
            width = detection.get('xmax', 0) - detection.get('xmin', 0)
            height = detection.get('ymax', 0) - detection.get('ymin', 0)
            area = width * height
            
            base_label = detection.get('label', 'unknown')
            base_coverage[base_label] = base_coverage.get(base_label, 0) + area
            
            # Handle tree classifications
            if base_label == 'tree':
                classification_label = detection.get('classification_label')
                if (classification_label and 
                    str(classification_label).lower() != 'nan'):
                    classification_coverage[classification_label] = classification_coverage.get(classification_label, 0) + area
        
        # Display coverage percentages
        coverage_parts = []
        for label, area in base_coverage.items():
            coverage_percent = (area / total_image_area) * 100
            
            if label == 'tree' and classification_coverage:
                # Show tree breakdown
                tree_coverage = f"{label}s: {coverage_percent:.2f}%"
                
                class_parts = []
                for class_label, class_area in classification_coverage.items():
                    class_percent = (class_area / total_image_area) * 100
                    class_name = class_label.replace('_', ' ')
                    class_parts.append(f"{class_name}s: {class_percent:.2f}%")
                
                if class_parts:
                    tree_coverage += f" ({', '.join(class_parts)})"
                coverage_parts.append(tree_coverage)
            else:
                label_name = label.replace('_', ' ')
                coverage_parts.append(f"{label_name}s: {coverage_percent:.2f}%")
        
        narrative += ", ".join(coverage_parts) + " of the total image area."
        
        return narrative

    def _generate_spatial_relationships_narrative(self, detections_list: List[Dict[str, Any]]) -> str:
        """
        Generate spatial relationships narrative using R-tree indexing.
        
        Args:
            detections_list (List[Dict[str, Any]]): List of all detection results
            
        Returns:
            str: Formatted spatial relationships section including:
                - Count of high-confidence objects analyzed
                - R-tree based intersection and proximity analysis
                - Natural language descriptions of object relationships
                - Confidence threshold information (>= 0.3)
        """
        spatial_relationships = self.spatial_analyzer.analyze_spatial_relationships_with_indexing(confidence_threshold=0.3)
        
        if not spatial_relationships:
            return "**Spatial Relationships Analysis (Confidence ≥ 0.3)**\nNo objects with sufficient confidence found for spatial relationship analysis."
        
        narrative = f"**Spatial Relationships Analysis in the whole image (Confidence ≥ 0.3)**\n"
        
        # Generate narrative using the spatial analyzer
        spatial_narrative = self.spatial_analyzer.generate_spatial_narrative(confidence_threshold=0.3)
        narrative += spatial_narrative
        
        return narrative
    
    def _detection_in_grid(self, detection: Dict[str, Any], grid_bounds: Dict[str, float]) -> bool:
        """
        Check if detection overlaps with grid bounds.
        
        Args:
            detection (Dict[str, Any]): Detection dictionary with 'xmin', 'ymin', 'xmax', 'ymax' keys
            grid_bounds (Dict[str, float]): Grid section bounds with 'x_min', 'y_min', 'x_max', 'y_max' keys
            
        Returns:
            bool: True if detection bounding box overlaps with grid bounds, False otherwise
        """
        det_xmin = detection.get('xmin', 0)
        det_ymin = detection.get('ymin', 0)
        det_xmax = detection.get('xmax', 0)
        det_ymax = detection.get('ymax', 0)
        
        return not (det_xmax <= grid_bounds['x_min'] or det_xmin >= grid_bounds['x_max'] or
                   det_ymax <= grid_bounds['y_min'] or det_ymin >= grid_bounds['y_max'])
    
    def _get_confidence_range(self, conf_category: str) -> tuple:
        """
        Get confidence range tuple from category string.
        
        Args:
            conf_category (str): Category name containing "High", "Medium", or other confidence indicator
            
        Returns:
            tuple: (min_confidence, max_confidence) as floats
                - High: (0.7, 1.0)
                - Medium: (0.3, 0.7) 
                - Low/Other: (0.0, 0.3)
        """
        if "High" in conf_category:
            return (0.7, 1.0)
        elif "Medium" in conf_category:
            return (0.3, 0.7)
        else:
            return (0.0, 0.3)
    
    def _count_labels_with_classification(self, detections: List[Dict[str, Any]]) -> tuple:
        """
        Count base labels and classifications separately.
        
        Args:
            detections (List[Dict[str, Any]]): List of detection dictionaries
            
        Returns:
            tuple: (base_counts, class_counts) where:
                - base_counts (Dict[str, int]): Count of each base object type
                - class_counts (Dict[str, Dict[str, int]]): Nested count structure for
                    tree classifications under 'tree' key
        """
        base_counts = {}
        class_counts = {}
        
        for detection in detections:
            base_label = detection.get('label', 'unknown')
            base_counts[base_label] = base_counts.get(base_label, 0) + 1
            
            if base_label == 'tree':
                classification_label = detection.get('classification_label')
                if (classification_label and 
                    str(classification_label).lower() != 'nan'):
                    
                    if base_label not in class_counts:
                        class_counts[base_label] = {}
                    class_counts[base_label][classification_label] = class_counts[base_label].get(classification_label, 0) + 1
        
        return base_counts, class_counts