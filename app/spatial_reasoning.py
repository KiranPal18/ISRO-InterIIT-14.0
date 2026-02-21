#!/usr/bin/env python3
"""
Spatial Reasoning Module for VRSBench Grounding

This module provides:
1. Spatial term extraction from natural language queries
2. Coordinate-based spatial filtering/ranking of predicted masks
3. Enhanced query generation with explicit spatial coordinates

Key insight: The SAM3 agent correctly identifies object TYPES but often fails
to select the correct INSTANCE when spatial terms like "bottom-left" or 
"right side" are used. This module adds post-processing to fix that.
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class SpatialRegion:
    """Defines a spatial region with normalized coordinate bounds."""
    name: str
    x_range: Tuple[float, float]  # (min_x, max_x) normalized [0, 1]
    y_range: Tuple[float, float]  # (min_y, max_y) normalized [0, 1]
    priority: int = 1  # Higher = more specific

# Define spatial regions using normalized coordinates
# Image coordinate system: (0,0) = top-left, (1,1) = bottom-right
SPATIAL_REGIONS: Dict[str, SpatialRegion] = {
    # Corners (most specific)
    "top-left": SpatialRegion("top-left", (0.0, 0.33), (0.0, 0.33), priority=3),
    "top-right": SpatialRegion("top-right", (0.67, 1.0), (0.0, 0.33), priority=3),
    "bottom-left": SpatialRegion("bottom-left", (0.0, 0.33), (0.67, 1.0), priority=3),
    "bottom-right": SpatialRegion("bottom-right", (0.67, 1.0), (0.67, 1.0), priority=3),
    
    # Edges (medium specificity)
    "top": SpatialRegion("top", (0.0, 1.0), (0.0, 0.33), priority=2),
    "bottom": SpatialRegion("bottom", (0.0, 1.0), (0.67, 1.0), priority=2),
    "left": SpatialRegion("left", (0.0, 0.33), (0.0, 1.0), priority=2),
    "right": SpatialRegion("right", (0.67, 1.0), (0.0, 1.0), priority=2),
    
    # Middle regions
    "center": SpatialRegion("center", (0.33, 0.67), (0.33, 0.67), priority=2),
    "middle": SpatialRegion("middle", (0.25, 0.75), (0.25, 0.75), priority=1),
    
    # Combined regions (lower priority, used as fallback)
    "top-left-area": SpatialRegion("top-left-area", (0.0, 0.5), (0.0, 0.5), priority=1),
    "top-right-area": SpatialRegion("top-right-area", (0.5, 1.0), (0.0, 0.5), priority=1),
    "bottom-left-area": SpatialRegion("bottom-left-area", (0.0, 0.5), (0.5, 1.0), priority=1),
    "bottom-right-area": SpatialRegion("bottom-right-area", (0.5, 1.0), (0.5, 1.0), priority=1),
    
    # Middle-edge combinations
    "middle-left": SpatialRegion("middle-left", (0.0, 0.4), (0.3, 0.7), priority=2),
    "middle-right": SpatialRegion("middle-right", (0.6, 1.0), (0.3, 0.7), priority=2),
    "middle-top": SpatialRegion("middle-top", (0.3, 0.7), (0.0, 0.4), priority=2),
    "middle-bottom": SpatialRegion("middle-bottom", (0.3, 0.7), (0.6, 1.0), priority=2),
    
    # Far edges (very edge of image)
    "far-left": SpatialRegion("far-left", (0.0, 0.2), (0.0, 1.0), priority=2),
    "far-right": SpatialRegion("far-right", (0.8, 1.0), (0.0, 1.0), priority=2),
    "far-top": SpatialRegion("far-top", (0.0, 1.0), (0.0, 0.2), priority=2),
    "far-bottom": SpatialRegion("far-bottom", (0.0, 1.0), (0.8, 1.0), priority=2),
}

# Patterns for extracting spatial terms from queries
SPATIAL_PATTERNS = [
    # Corner patterns (high priority - most specific)
    (r'\b(top[\s-]?left|upper[\s-]?left|left[\s-]?top)\s*(corner|area|part|section|side)?\b', 'top-left'),
    (r'\b(top[\s-]?right|upper[\s-]?right|right[\s-]?top)\s*(corner|area|part|section|side)?\b', 'top-right'),
    (r'\b(bottom[\s-]?left|lower[\s-]?left|left[\s-]?bottom)\s*(corner|area|part|section|side)?\b', 'bottom-left'),
    (r'\b(bottom[\s-]?right|lower[\s-]?right|right[\s-]?bottom)\s*(corner|area|part|section|side)?\b', 'bottom-right'),
    
    # Corner patterns with "corner of the image"
    (r'\b(top[\s-]?left|upper[\s-]?left)\s*corner\s*(of\s+the\s+image)?\b', 'top-left'),
    (r'\b(top[\s-]?right|upper[\s-]?right)\s*corner\s*(of\s+the\s+image)?\b', 'top-right'),
    (r'\b(bottom[\s-]?left|lower[\s-]?left)\s*corner\s*(of\s+the\s+image)?\b', 'bottom-left'),
    (r'\b(bottom[\s-]?right|lower[\s-]?right)\s*corner\s*(of\s+the\s+image)?\b', 'bottom-right'),
    
    # Middle + direction
    (r'\b(middle[\s-]?left|mid[\s-]?left|left[\s-]?middle)\b', 'middle-left'),
    (r'\b(middle[\s-]?right|mid[\s-]?right|right[\s-]?middle)\b', 'middle-right'),
    (r'\b(middle[\s-]?top|mid[\s-]?top|top[\s-]?middle)\b', 'middle-top'),
    (r'\b(middle[\s-]?bottom|mid[\s-]?bottom|bottom[\s-]?middle)\b', 'middle-bottom'),
    
    # Far + direction
    (r'\bfar[\s-]?(left|right|top|bottom)\b', None),  # Handle in code
    (r'\b(very|extreme)\s+(left|right|top|bottom)\b', None),  # Handle in code
    
    # Edge patterns
    (r'\b(at|on|along)\s+the\s+(left|right|top|bottom)\s*edge\b', None),  # Handle in code
    (r'\b(left|right|top|bottom)\s*edge\s*(of\s+the\s+image)?\b', None),  # Handle in code
    
    # Single directions with context
    (r'\b(at|in|on|towards?)\s+the\s+(top|bottom|left|right)\s*(of\s+the\s+image|side|part|area)?\b', None),
    (r'\b(located|positioned|situated)\s+(at|on|in)\s+the\s+(top|bottom|left|right)\b', None),
    
    # Simple single directions at end of phrase
    (r'\bon\s+the\s+(left|right)\s*side\b', None),
    (r'\bat\s+the\s+(top|bottom)\b', None),
    
    # Row/column patterns
    (r'\b(top|bottom|first|last)\s+row\b', None),
    (r'\b(left|right|first|last)\s+column\b', None),
    
    # Center/middle
    (r'\b(center|central|middle)\s+(of\s+the\s+image|part|area|section)?\b', 'center'),
    (r'\bin\s+the\s+(center|middle)\b', 'center'),
]

# Additional patterns for relative positions
RELATIVE_PATTERNS = [
    (r'\b(?:second|2nd)\s+from\s+(?:the\s+)?(?:left|right|top|bottom)\b', 'ordinal'),
    (r'\b(?:third|3rd)\s+from\s+(?:the\s+)?(?:left|right|top|bottom)\b', 'ordinal'),
    (r'\b(?:first|1st)\s+from\s+(?:the\s+)?(?:left|right|top|bottom)\b', 'ordinal'),
    (r'\bclosest\s+to\b', 'proximity'),
    (r'\bnear(?:est)?\s+(?:to\s+)?(?:the\s+)?\b', 'proximity'),
    (r'\bfarthest\s+from\b', 'proximity_far'),
    (r'\bbetween\b', 'between'),
]

# Ordinal number mapping
ORDINAL_MAP = {
    'first': 1, '1st': 1,
    'second': 2, '2nd': 2,
    'third': 3, '3rd': 3,
    'fourth': 4, '4th': 4,
    'fifth': 5, '5th': 5,
    'last': -1,  # Special: means last item
}


@dataclass
class OrdinalConstraint:
    """Represents an ordinal spatial constraint like 'second from the right'."""
    ordinal: int  # 1-indexed position, -1 for "last"
    direction: str  # 'left', 'right', 'top', 'bottom'
    axis: str  # 'x' for left/right, 'y' for top/bottom
    ascending: bool  # True if counting from low values (left/top)


def extract_ordinal_constraint(query: str) -> Optional[OrdinalConstraint]:
    """
    Extract ordinal constraint from query like "second from the right".
    
    Returns:
        OrdinalConstraint if found, None otherwise
    """
    query_lower = query.lower()
    
    # Pattern 1: [ordinal] from the [direction]
    # e.g., "second from the right", "third from left", "1st from the bottom"
    pattern1 = r'\b(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|last)\s+from\s+(?:the\s+)?(left|right|top|bottom)\b'
    match = re.search(pattern1, query_lower)
    
    if match:
        ordinal_text = match.group(1)
        direction = match.group(2)
        
        ordinal = ORDINAL_MAP.get(ordinal_text, 1)
        
        if direction in ['left', 'right']:
            axis = 'x'
            ascending = (direction == 'left')  # Count from left = ascending x
        else:
            axis = 'y'
            ascending = (direction == 'top')  # Count from top = ascending y
        
        return OrdinalConstraint(
            ordinal=ordinal,
            direction=direction,
            axis=axis,
            ascending=ascending
        )
    
    # Pattern 2: [ordinal] [object] from the [direction]
    # e.g., "third vehicle from the left", "second car from bottom"
    pattern2 = r'\b(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|last)\s+\w+\s+from\s+(?:the\s+)?(left|right|top|bottom)\b'
    match2 = re.search(pattern2, query_lower)
    
    if match2:
        ordinal_text = match2.group(1)
        direction = match2.group(2)
        
        ordinal = ORDINAL_MAP.get(ordinal_text, 1)
        
        if direction in ['left', 'right']:
            axis = 'x'
            ascending = (direction == 'left')
        else:
            axis = 'y'
            ascending = (direction == 'top')
        
        return OrdinalConstraint(
            ordinal=ordinal,
            direction=direction,
            axis=axis,
            ascending=ascending
        )
    
    # Pattern 3: [direction] [ordinal] [object]
    # e.g., "the left second vehicle", "the bottom third car"  
    pattern3 = r'\b(left|right|top|bottom)\s+(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th)\b'
    match3 = re.search(pattern3, query_lower)
    
    if match3:
        direction = match3.group(1)
        ordinal_text = match3.group(2)
        
        ordinal = ORDINAL_MAP.get(ordinal_text, 1)
        
        if direction in ['left', 'right']:
            axis = 'x'
            ascending = (direction == 'left')
        else:
            axis = 'y'
            ascending = (direction == 'top')
        
        return OrdinalConstraint(
            ordinal=ordinal,
            direction=direction,
            axis=axis,
            ascending=ascending
        )
    
    # Pattern 4: the [ordinal]-most from [direction] or [direction]-most
    # e.g., "the second-most from right", "leftmost" 
    pattern4 = r'\b(left|right|top|bottom)[-\s]?most\b'
    match4 = re.search(pattern4, query_lower)
    
    if match4:
        direction = match4.group(1)
        
        if direction in ['left', 'right']:
            axis = 'x'
            ascending = (direction == 'left')
        else:
            axis = 'y'
            ascending = (direction == 'top')
        
        return OrdinalConstraint(
            ordinal=1,  # -most means first
            direction=direction,
            axis=axis,
            ascending=ascending
        )
    
    return None


def filter_predictions_by_region(
    predictions: List[Dict],
    target_region: Dict,
    threshold: float = 0.05
) -> List[Dict]:
    """
    Filter predictions to those within or near a target spatial region.
    
    Args:
        predictions: List of OBB predictions with 'obbox' key
        target_region: Dict with 'x_range' and 'y_range' keys
        threshold: Extra margin around the region (default 0.05 = 5%)
    
    Returns:
        Filtered list of predictions that fall within the region
    """
    if not predictions or not target_region:
        return predictions
    
    x_min, x_max = target_region.get('x_range', (0, 1))
    y_min, y_max = target_region.get('y_range', (0, 1))
    
    # Add threshold margin
    x_min = max(0, x_min - threshold)
    x_max = min(1, x_max + threshold)
    y_min = max(0, y_min - threshold)
    y_max = min(1, y_max + threshold)
    
    filtered = []
    for pred in predictions:
        obb = pred.get('obbox', [0.5, 0.5])
        cx, cy = obb[0], obb[1]
        
        if x_min <= cx <= x_max and y_min <= cy <= y_max:
            filtered.append(pred)
    
    # If filtering removes everything, return all predictions
    return filtered if filtered else predictions


def apply_ordinal_reranking(
    predictions: List[Dict],
    ordinal: OrdinalConstraint,
    target_region: Dict = None
) -> List[Dict]:
    """
    Rerank predictions based on ordinal constraint.
    
    E.g., "second from the right" selects the 2nd item when sorted by x descending.
    
    Args:
        predictions: List of OBB predictions
        ordinal: OrdinalConstraint specifying position
        target_region: Optional region to filter to first (e.g., "top row")
    """
    if not predictions or not ordinal:
        return predictions
    
    # First filter to target region if specified
    # This handles cases like "second from right in the TOP ROW"
    if target_region:
        predictions = filter_predictions_by_region(predictions, target_region)
    
    # Sort predictions by the relevant axis
    if ordinal.axis == 'x':
        key_fn = lambda p: p.get('obbox', [0.5])[0]  # x coordinate
    else:
        key_fn = lambda p: p.get('obbox', [0.5, 0.5])[1]  # y coordinate
    
    sorted_preds = sorted(predictions, key=key_fn, reverse=not ordinal.ascending)
    
    # Select the ordinal-th item
    target_idx = ordinal.ordinal - 1  # Convert to 0-indexed
    if ordinal.ordinal == -1:  # "last"
        target_idx = len(sorted_preds) - 1
    
    if 0 <= target_idx < len(sorted_preds):
        # Assign high score to the target, decreasing for others
        for i, pred in enumerate(sorted_preds):
            if i == target_idx:
                pred['spatial_score'] = 10.0  # High priority
                pred['ordinal_match'] = True
            else:
                # Score based on distance from target position
                distance = abs(i - target_idx)
                pred['spatial_score'] = max(0, 5.0 - distance)
                pred['ordinal_match'] = False
        
        # Re-sort by spatial score
        return sorted(sorted_preds, key=lambda x: -x.get('spatial_score', 0))
    
    return predictions


def extract_spatial_terms(query: str) -> List[Dict[str, Any]]:
    """
    Extract spatial terms and their regions from a natural language query.
    
    Returns:
        List of dicts with keys: 'term', 'region', 'priority', 'match'
    """
    query_lower = query.lower()
    results = []
    
    # Check compound patterns first
    for pattern, region_name in SPATIAL_PATTERNS:
        matches = list(re.finditer(pattern, query_lower))
        for match in matches:
            matched_text = match.group()
            
            if region_name:
                # Direct region mapping
                region = SPATIAL_REGIONS.get(region_name)
                if region:
                    results.append({
                        'term': matched_text,
                        'region': region,
                        'region_name': region_name,
                        'priority': region.priority,
                        'span': match.span()
                    })
            else:
                # Need to extract direction from the matched text
                dir_name = _extract_direction_from_match(matched_text)
                if dir_name:
                    region = SPATIAL_REGIONS.get(dir_name)
                    if region:
                        # Determine priority based on specificity of match
                        priority = region.priority
                        if 'far' in matched_text or 'very' in matched_text or 'extreme' in matched_text:
                            priority = region.priority + 1
                        elif 'towards' in matched_text:
                            priority = region.priority - 1
                        
                        results.append({
                            'term': matched_text,
                            'region': region,
                            'region_name': dir_name,
                            'priority': priority,
                            'span': match.span()
                        })
    
    # Deduplicate overlapping matches, keeping highest priority
    results = _deduplicate_spatial_results(results)
    
    return sorted(results, key=lambda x: -x['priority'])


def _extract_direction_from_match(text: str) -> Optional[str]:
    """Extract the direction (left, right, top, bottom) from a matched spatial phrase."""
    text = text.lower()
    
    # Check for combined directions first
    if 'far' in text or 'very' in text or 'extreme' in text:
        if 'left' in text:
            return 'far-left'
        elif 'right' in text:
            return 'far-right'
        elif 'top' in text:
            return 'far-top'
        elif 'bottom' in text:
            return 'far-bottom'
    
    # Check for row/column patterns
    if 'row' in text:
        if 'top' in text or 'first' in text:
            return 'top'
        elif 'bottom' in text or 'last' in text:
            return 'bottom'
    
    if 'column' in text:
        if 'left' in text or 'first' in text:
            return 'left'
        elif 'right' in text or 'last' in text:
            return 'right'
    
    # Single directions
    for direction in ['left', 'right', 'top', 'bottom']:
        if direction in text:
            return direction
    
    return None
    
    # Deduplicate overlapping matches, keeping highest priority
    results = _deduplicate_spatial_results(results)
    
    return sorted(results, key=lambda x: -x['priority'])


def _deduplicate_spatial_results(results: List[Dict]) -> List[Dict]:
    """Remove overlapping matches, keeping highest priority."""
    if not results:
        return results
    
    # Sort by priority descending
    results = sorted(results, key=lambda x: -x['priority'])
    
    kept = []
    used_spans = []
    
    for r in results:
        span = r['span']
        overlaps = False
        for used in used_spans:
            if not (span[1] <= used[0] or span[0] >= used[1]):
                overlaps = True
                break
        
        if not overlaps:
            kept.append(r)
            used_spans.append(span)
    
    return kept


def compute_spatial_score(cx: float, cy: float, region: SpatialRegion, 
                          prefer_extreme: bool = False, extreme_direction: str = None) -> float:
    """
    Compute how well a point (cx, cy) matches a spatial region.
    
    Args:
        cx, cy: Point coordinates (normalized 0-1)
        region: Target spatial region
        prefer_extreme: If True, prefer points at the extreme of the region (for "corner" terms)
        extreme_direction: Which direction is "extreme" (e.g., "bottom-right" means prefer high x, high y)
    
    Returns:
        Score in [0, 1] where 1 = perfect match, 0 = outside region
    """
    x_min, x_max = region.x_range
    y_min, y_max = region.y_range
    
    # Check if inside region
    in_x = x_min <= cx <= x_max
    in_y = y_min <= cy <= y_max
    
    if in_x and in_y:
        if prefer_extreme and extreme_direction:
            # For corner/edge terms, prefer points at the extreme of the region
            # e.g., "bottom-right" should prefer high x AND high y
            score = 0.5  # Base score for being in region
            
            # Compute how "extreme" this point is in each direction
            x_range = x_max - x_min if x_max > x_min else 1.0
            y_range = y_max - y_min if y_max > y_min else 1.0
            
            if 'right' in extreme_direction:
                # Prefer higher x values
                score += 0.25 * ((cx - x_min) / x_range)
            elif 'left' in extreme_direction:
                # Prefer lower x values
                score += 0.25 * ((x_max - cx) / x_range)
            else:
                # No x preference - reward being in range
                score += 0.125
            
            if 'bottom' in extreme_direction:
                # Prefer higher y values
                score += 0.25 * ((cy - y_min) / y_range)
            elif 'top' in extreme_direction:
                # Prefer lower y values
                score += 0.25 * ((y_max - cy) / y_range)
            else:
                # No y preference - reward being in range
                score += 0.125
            
            return min(1.0, score)
        else:
            # Default: score based on distance to region center
            region_cx = (x_min + x_max) / 2
            region_cy = (y_min + y_max) / 2
            dist = np.sqrt((cx - region_cx)**2 + (cy - region_cy)**2)
            max_dist = np.sqrt(0.5**2 + 0.5**2)  # Max possible distance in normalized coords
            return 1.0 - (dist / max_dist) * 0.5  # Score between 0.5 and 1.0
    else:
        # Score based on distance to region boundary
        dx = max(x_min - cx, 0, cx - x_max)
        dy = max(y_min - cy, 0, cy - y_max)
        dist = np.sqrt(dx**2 + dy**2)
        return max(0, 0.5 - dist)  # Decay outside region


def rerank_predictions_by_spatial(
    predictions: List[Dict],
    spatial_terms: List[Dict],
    alpha: float = 0.5,
    confidence_weight: float = 0.6,
    spatial_threshold: float = 0.5
) -> List[Dict]:
    """
    Rerank predictions based on spatial term matching AND SAM confidence.
    
    The key insight: If multiple OBBs are in the correct spatial region,
    prefer the one with higher SAM confidence rather than extreme position.
    
    Args:
        predictions: List of OBB predictions with 'obbox' key [cx, cy, w, h, angle]
                    and optionally 'sam_score' for SAM confidence
        spatial_terms: Output from extract_spatial_terms()
        alpha: Weight for spatial score vs confidence (0 = ignore spatial, 1 = only spatial)
        confidence_weight: Weight given to SAM confidence in final ranking (0-1)
        spatial_threshold: Minimum spatial score to be considered "in region"
    
    Returns:
        Reranked list of predictions with added 'spatial_score' and 'combined_score' keys
    """
    if not predictions or not spatial_terms:
        return predictions
    
    for pred in predictions:
        obb = pred.get('obbox', pred.get('obbs', [0.5, 0.5, 0.1, 0.1, 0]))
        if isinstance(obb, list) and len(obb) >= 2:
            cx, cy = obb[0], obb[1]
        else:
            cx, cy = 0.5, 0.5
        
        # Get SAM confidence if available
        sam_score = pred.get('sam_score', pred.get('score', 0.5))
        pred['sam_score'] = sam_score
        
        # Compute spatial score as weighted average across all spatial terms
        spatial_scores = []
        in_region_scores = []  # Track if inside region (binary)
        
        for st in spatial_terms:
            term_text = st.get('term', '').lower()
            region_name = st.get('region_name', '').lower()
            region = st['region']
            
            # Check if point is inside the region
            x_min, x_max = region.x_range
            y_min, y_max = region.y_range
            in_region = x_min <= cx <= x_max and y_min <= cy <= y_max
            in_region_scores.append(1.0 if in_region else 0.0)
            
            # Detect if this is a corner or edge term that should prefer extremes
            is_corner = 'corner' in term_text or (
                any(x in region_name for x in ['top-left', 'top-right', 'bottom-left', 'bottom-right'])
            )
            is_edge = 'edge' in term_text or 'far' in term_text or 'very' in term_text
            prefer_extreme = is_corner or is_edge
            
            # Determine extreme direction from region name
            extreme_direction = region_name if prefer_extreme else None
            
            score = compute_spatial_score(cx, cy, st['region'], 
                                         prefer_extreme=prefer_extreme,
                                         extreme_direction=extreme_direction)
            weighted_score = score * st['priority']
            spatial_scores.append(weighted_score)
        
        if spatial_scores:
            # Raw spatial score (with priority weighting)
            raw_spatial = np.mean(spatial_scores)
            # Normalize to [0, 1] range - max possible is ~3 (priority 3 * score 1.0)
            max_priority = max(st['priority'] for st in spatial_terms) if spatial_terms else 3
            pred['spatial_score'] = raw_spatial
            pred['spatial_score_normalized'] = min(1.0, raw_spatial / max_priority)
            pred['in_region'] = np.mean(in_region_scores) > 0.5
        else:
            pred['spatial_score'] = 0.5  # Neutral
            pred['spatial_score_normalized'] = 0.5
            pred['in_region'] = True  # Assume in region if no constraints
        
        # Compute combined score:
        # - If IN region: heavily weight SAM confidence (the agent found it, trust it!)
        # - If NOT in region: penalize significantly
        spatial_norm = pred['spatial_score_normalized']
        
        if pred['in_region']:
            # Trust SAM confidence for objects in the correct region
            # Spatial score only provides a small bonus for tie-breaking
            # Higher confidence_weight means more trust in SAM's detection
            # Use a more aggressive confidence weight for in-region candidates
            effective_conf_weight = confidence_weight + 0.1 * sam_score  # Boost high-confidence SAMs
            effective_conf_weight = min(0.85, effective_conf_weight)  # Cap at 85%
            pred['combined_score'] = (
                effective_conf_weight * sam_score + 
                (1 - effective_conf_weight) * spatial_norm
            )
        else:
            # Object not in target region - heavy penalty
            pred['combined_score'] = 0.2 * spatial_norm + 0.1 * sam_score
    
    # Sort by combined score descending
    return sorted(predictions, key=lambda x: -x.get('combined_score', 0))


def filter_predictions_by_spatial(
    predictions: List[Dict],
    spatial_terms: List[Dict],
    threshold: float = 0.3
) -> List[Dict]:
    """
    Filter predictions that don't match spatial constraints.
    
    Args:
        predictions: List of OBB predictions
        spatial_terms: Output from extract_spatial_terms()
        threshold: Minimum spatial score to keep
    
    Returns:
        Filtered list of predictions
    """
    if not predictions or not spatial_terms:
        return predictions
    
    # First rerank to compute spatial scores
    reranked = rerank_predictions_by_spatial(predictions, spatial_terms)
    
    # Filter by threshold
    filtered = [p for p in reranked if p.get('spatial_score', 0) >= threshold]
    
    # If filtering removes all predictions, keep the best one
    if not filtered and reranked:
        filtered = [reranked[0]]
    
    return filtered


def select_best_spatial_match(
    predictions: List[Dict],
    query: str,
    return_all: bool = False
) -> List[Dict]:
    """
    Main function: Select the best prediction(s) based on spatial reasoning.
    
    Args:
        predictions: List of OBB predictions with 'obbox' key
        query: Original natural language query
        return_all: If True, return all predictions sorted; if False, return best match
    
    Returns:
        List of selected predictions
    """
    if not predictions:
        return predictions
    
    # Extract spatial terms
    spatial_terms = extract_spatial_terms(query)
    
    if not spatial_terms:
        # No spatial constraints - return as-is
        return predictions if return_all else predictions[:1]
    
    # Rerank by spatial match
    reranked = rerank_predictions_by_spatial(predictions, spatial_terms)
    
    if return_all:
        return reranked
    
    # Return best match
    return reranked[:1] if reranked else predictions[:1]


def enhance_query_with_spatial_hints(query: str) -> str:
    """
    Enhance the query with explicit spatial coordinate hints for the agent.
    
    This adds explicit guidance about what coordinates the agent should look for.
    """
    spatial_terms = extract_spatial_terms(query)
    
    if not spatial_terms:
        return query
    
    hints = []
    for st in spatial_terms:
        region = st['region']
        term = st['term']
        
        # Convert to image coordinates description
        x_desc = f"x in [{region.x_range[0]:.1f}-{region.x_range[1]:.1f}]"
        y_desc = f"y in [{region.y_range[0]:.1f}-{region.y_range[1]:.1f}]"
        
        # Human-readable hint
        if 'left' in term:
            x_hint = "LEFT side (low x values)"
        elif 'right' in term:
            x_hint = "RIGHT side (high x values)"
        else:
            x_hint = "middle x"
        
        if 'top' in term or 'upper' in term:
            y_hint = "TOP of image (low y values)"
        elif 'bottom' in term or 'lower' in term:
            y_hint = "BOTTOM of image (high y values)"
        else:
            y_hint = "middle y"
        
        hints.append(f"'{term}' means {x_hint}, {y_hint}")
    
    if hints:
        enhanced = query + "\n\n[SPATIAL HINTS: " + "; ".join(hints) + "]"
        return enhanced
    
    return query


def get_spatial_analysis(query: str) -> Dict[str, Any]:
    """
    Get a complete spatial analysis of a query.
    
    Returns:
        Dict with:
            - 'spatial_terms': List of extracted spatial terms
            - 'target_region': Combined target region bounds
            - 'enhanced_query': Query with spatial hints
            - 'has_spatial_constraint': Boolean
            - 'ordinal_constraint': OrdinalConstraint if present
    """
    spatial_terms = extract_spatial_terms(query)
    ordinal = extract_ordinal_constraint(query)
    
    has_constraint = bool(spatial_terms) or (ordinal is not None)
    
    if not has_constraint:
        return {
            'spatial_terms': [],
            'target_region': None,
            'enhanced_query': query,
            'has_spatial_constraint': False,
            'ordinal_constraint': None
        }
    
    # Combine regions (intersection for high priority terms)
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0
    
    for st in spatial_terms:
        region = st['region']
        # Use intersection for high priority terms
        if st['priority'] >= 2:
            x_min = max(x_min, region.x_range[0])
            x_max = min(x_max, region.x_range[1])
            y_min = max(y_min, region.y_range[0])
            y_max = min(y_max, region.y_range[1])
    
    # Include ordinal info if present
    ordinal_info = None
    if ordinal:
        ordinal_info = {
            'ordinal': ordinal.ordinal,
            'direction': ordinal.direction,
            'axis': ordinal.axis,
            'ascending': ordinal.ascending
        }
    
    return {
        'spatial_terms': spatial_terms,
        'target_region': {
            'x_range': (x_min, x_max),
            'y_range': (y_min, y_max)
        },
        'enhanced_query': enhance_query_with_spatial_hints(query),
        'has_spatial_constraint': True,
        'ordinal_constraint': ordinal_info
    }


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Test queries from VRSBench failures
    test_queries = [
        "The large vehicle located at the bottom-left corner of the image.",
        "The small dark-colored vehicle positioned at the bottom-right corner of the image.",
        "The large vehicle at the top-right corner of the image.",
        "The small vehicle is towards the middle-left side of the image.",
        "The large vehicle located second from the right in the top row.",
        "The large yellow vehicle situated closest to the green area.",
        "Another small vehicle is at the very bottom of the image.",
        "The large vehicle located on the right side.",
    ]
    
    print("=" * 70)
    print("SPATIAL REASONING MODULE TESTS")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        analysis = get_spatial_analysis(query)
        
        if analysis['has_spatial_constraint']:
            print(f"  Spatial terms found:")
            for st in analysis['spatial_terms']:
                print(f"    - '{st['term']}' -> {st['region_name']} (priority: {st['priority']})")
            print(f"  Target region: x={analysis['target_region']['x_range']}, y={analysis['target_region']['y_range']}")
        else:
            print("  No spatial constraints detected")
        
        print("-" * 50)
    
    # Test reranking
    print("\n" + "=" * 70)
    print("RERANKING TEST")
    print("=" * 70)
    
    query = "The large vehicle located at the bottom-left corner of the image."
    spatial_terms = extract_spatial_terms(query)
    
    # Simulated predictions (GT at bottom-left, wrong pred at center)
    predictions = [
        {'object_id': 1, 'obbox': [0.29, 0.50, 0.04, 0.20, -11.0]},  # Center (wrong)
        {'object_id': 2, 'obbox': [0.18, 0.99, 0.05, 0.13, -86.0]},  # Bottom-left (correct)
        {'object_id': 3, 'obbox': [0.80, 0.20, 0.05, 0.10, -45.0]},  # Top-right (wrong)
    ]
    
    print(f"Query: {query}")
    print(f"Predictions before reranking:")
    for p in predictions:
        print(f"  {p}")
    
    reranked = rerank_predictions_by_spatial(predictions, spatial_terms)
    print(f"\nPredictions after reranking:")
    for p in reranked:
        print(f"  {p} (spatial_score: {p['spatial_score']:.3f})")


# =============================================================================
# Relative Spatial Reasoning (e.g., "to the right of the truck")
# =============================================================================

@dataclass
class RelativeSpatialConstraint:
    """Represents a relative spatial constraint like 'to the right of the truck'."""
    target_class: str       # The class we're looking for (e.g., "airplane")
    reference_class: str    # The reference object (e.g., "truck")
    relation: str           # Spatial relation: 'left_of', 'right_of', 'above', 'below', 'near'
    margin: float = 0.05    # Margin for spatial comparison


def extract_relative_spatial_constraint(query: str) -> Optional[RelativeSpatialConstraint]:
    """
    Extract relative spatial constraint from query.
    
    Examples:
        - "all airplanes to the right of the truck" -> target=airplane, ref=truck, relation=right_of
        - "vehicles above the building" -> target=vehicle, ref=building, relation=above
        - "cars near the swimming pool" -> target=car, ref=swimming pool, relation=near
    
    Returns:
        RelativeSpatialConstraint if found, None otherwise
    """
    query_lower = query.lower()
    
    # Patterns for relative spatial references
    # Pattern: [target] to the [direction] of [reference]
    patterns = [
        # "airplanes to the right of the truck"
        (r'(?:all\s+)?(?:the\s+)?(\w+s?)\s+(?:to\s+the\s+)?(left|right)\s+of\s+(?:the\s+)?(.+?)(?:\s+in|\s+seen|\s*$|\.)', 3),
        # "vehicles above/below the building"  
        (r'(?:all\s+)?(?:the\s+)?(\w+s?)\s+(above|below|over|under)\s+(?:the\s+)?(.+?)(?:\s+in|\s+seen|\s*$|\.)', 3),
        # "cars near/beside/next to the pool" - different structure (no direction word captured)
        (r'(?:all\s+)?(?:the\s+)?(\w+s?)\s+(near|beside|close\s+to)\s+(?:the\s+)?(.+?)(?:\s+in|\s+seen|\s*$|\.)', 3),
        # "cars next to the pool"
        (r'(?:all\s+)?(?:the\s+)?(\w+s?)\s+(next)\s+to\s+(?:the\s+)?(.+?)(?:\s+in|\s+seen|\s*$|\.)', 3),
    ]
    
    for pattern, num_groups in patterns:
        match = re.search(pattern, query_lower)
        if match:
            groups = match.groups()
            
            if len(groups) >= 3:
                target = groups[0].strip()
                direction = groups[1].strip()
                reference = groups[2].strip()
                
                # Map direction to relation
                relation_map = {
                    'left': 'left_of',
                    'right': 'right_of',
                    'above': 'above',
                    'over': 'above',
                    'below': 'below',
                    'under': 'below',
                    'near': 'near',
                    'beside': 'near',
                    'next to': 'near',
                    'close to': 'near',
                }
                
                relation = relation_map.get(direction, 'near')
                
                # Clean up reference (remove trailing words)
                reference = re.sub(r'\s+(in|seen|visible|of|the)\s*$', '', reference).strip()
                
                return RelativeSpatialConstraint(
                    target_class=target,
                    reference_class=reference,
                    relation=relation
                )
    
    return None


def filter_by_relative_position(
    target_candidates: List[Dict],
    reference_bbox: List[float],
    relation: str,
    margin: float = 0.05
) -> List[Dict]:
    """
    Filter target candidates based on their position relative to a reference bbox.
    
    Args:
        target_candidates: List of OBB predictions for target objects
        reference_bbox: [cx, cy, w, h, angle] of reference object
        relation: 'left_of', 'right_of', 'above', 'below', 'near'
        margin: Tolerance margin for spatial comparison
    
    Returns:
        Filtered list of candidates satisfying the spatial relation
    """
    if not target_candidates or not reference_bbox:
        return target_candidates
    
    ref_cx, ref_cy = reference_bbox[0], reference_bbox[1]
    ref_w, ref_h = reference_bbox[2], reference_bbox[3]
    
    filtered = []
    
    for candidate in target_candidates:
        obb = candidate.get('obbox', candidate.get('obbs', [0.5, 0.5, 0.1, 0.1, 0]))
        cand_cx, cand_cy = obb[0], obb[1]
        
        satisfies = False
        
        if relation == 'right_of':
            # Candidate center should be to the right of reference's right edge
            ref_right_edge = ref_cx + ref_w / 2
            satisfies = cand_cx > (ref_right_edge - margin)
            
        elif relation == 'left_of':
            # Candidate center should be to the left of reference's left edge
            ref_left_edge = ref_cx - ref_w / 2
            satisfies = cand_cx < (ref_left_edge + margin)
            
        elif relation == 'above':
            # Candidate center should be above reference's top edge (lower y)
            ref_top_edge = ref_cy - ref_h / 2
            satisfies = cand_cy < (ref_top_edge + margin)
            
        elif relation == 'below':
            # Candidate center should be below reference's bottom edge (higher y)
            ref_bottom_edge = ref_cy + ref_h / 2
            satisfies = cand_cy > (ref_bottom_edge - margin)
            
        elif relation == 'near':
            # Candidate should be within a certain distance of reference
            distance = np.sqrt((cand_cx - ref_cx)**2 + (cand_cy - ref_cy)**2)
            # "Near" means within 2x the reference object's size
            near_threshold = max(ref_w, ref_h) * 2 + margin
            satisfies = distance < near_threshold
        
        if satisfies:
            # Add relative position info
            candidate['relative_position'] = {
                'relation': relation,
                'reference_center': (ref_cx, ref_cy),
                'distance_x': cand_cx - ref_cx,
                'distance_y': cand_cy - ref_cy
            }
            filtered.append(candidate)
    
    return filtered


def get_relative_spatial_analysis(query: str) -> Dict[str, Any]:
    """
    Analyze query for relative spatial constraints.
    
    Returns:
        Dict with:
            - has_relative_constraint: bool
            - constraint: RelativeSpatialConstraint or None
            - target_class: str - what to find
            - reference_class: str - reference object
            - relation: str - spatial relation
    """
    constraint = extract_relative_spatial_constraint(query)
    
    if constraint:
        return {
            'has_relative_constraint': True,
            'constraint': constraint,
            'target_class': constraint.target_class,
            'reference_class': constraint.reference_class,
            'relation': constraint.relation
        }
    
    return {
        'has_relative_constraint': False,
        'constraint': None,
        'target_class': None,
        'reference_class': None,
        'relation': None
    }