#!/usr/bin/env python3
"""
VRSBench Grounding Baseline

Clean implementation for VRSBench grounding evaluation with proper output structure:
- outputs/: JSONL results with query, image_url, GT, intermediate masks, final boxes
- visualizations/<image_name>/: PNG files for all mask types

OBB Format: (center_x, center_y, width, height, angle)
- All spatial params normalized [0, 1]
- Angle in degrees, range [-90, 0) per OpenCV cv2.minAreaRect() convention
"""

import os
import re
import sys
import json
import shutil
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from pycocotools import mask as mask_utils

# Import centralized config
try:
    from config import (
        HF_HOME, TORCH_HOME, SAM3_PATH as CONFIG_SAM3_PATH,
        QWEN_FP16_MODEL_ID, SAM3_MODEL_ID, VLLM_PORT,
        SAM3_CHECKPOINT_PATH
    )
    os.environ["HF_HOME"] = HF_HOME
    os.environ["TRANSFORMERS_CACHE"] = HF_HOME
    os.environ["TORCH_HOME"] = TORCH_HOME
    SAM3_PATH = CONFIG_SAM3_PATH
except ImportError:
    # Fallback for standalone use
    os.environ.setdefault("HF_HOME", "/workspace/cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/cache/huggingface")
    os.environ.setdefault("TORCH_HOME", "/workspace/cache/torch")
    SAM3_PATH = os.environ.get("SAM3_PATH", "/workspace/sam3")
    QWEN_FP16_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    SAM3_MODEL_ID = "facebook/sam3"
    VLLM_PORT = 8001
    SAM3_CHECKPOINT_PATH = None

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")

# Enable TF32 for faster inference
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Add SAM3 to path if exists
if os.path.exists(SAM3_PATH):
    sys.path.insert(0, SAM3_PATH)

# Import Shapely for IoU computation
try:
    from shapely.geometry import Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    Polygon = None  # For type hints when shapely not available
    print("Warning: shapely not installed. OBB IoU will use approximation.")

# Import spatial reasoning module
from spatial_reasoning import (
    extract_spatial_terms,
    rerank_predictions_by_spatial,
    get_spatial_analysis,
    select_best_spatial_match,
    extract_ordinal_constraint,
    apply_ordinal_reranking,
    filter_predictions_by_region,
    OrdinalConstraint,
    # Relative spatial reasoning
    extract_relative_spatial_constraint,
    filter_by_relative_position,
    get_relative_spatial_analysis,
    RelativeSpatialConstraint
)


# =============================================================================
# Utility Functions
# =============================================================================

def detect_query_type(query: str) -> Dict[str, Any]:
    """
    Detect if query expects single or multiple objects.
    
    Returns:
        Dict with:
            - 'expects_multiple': bool - True if query asks for multiple objects
            - 'target_class': str - The object class being queried
            - 'spatial_constraint': bool - True if specific location is mentioned
    """
    query_lower = query.lower()
    
    # Patterns indicating multiple objects expected
    plural_patterns = [
        r'\ball\s+(?:the\s+)?\w+s\b',           # "all the cars", "all aircrafts"
        r'\bevery\s+\w+\b',                      # "every building"
        r'\b(?:all|every)\s+(?:of\s+)?(?:the\s+)?', # "all of the"
        r'\blocate\s+.*\b\w+s\b',               # "locate aircrafts" (plural noun)
        r'\bfind\s+.*\b\w+s\b',                 # "find vehicles"
        r'\b\w+s\s+(?:seen|visible|present)\b', # "aircrafts seen in"
        r'\bboxes\s+for\s+(?:the\s+)?\w+s\b',  # "boxes for the aircrafts"
    ]
    
    # Patterns indicating single object expected
    singular_patterns = [
        r'\bthe\s+(?:one|single|only)\s+',       # "the one", "the single"
        r'\b(?:a|an)\s+\w+\b',                   # "a car", "an airplane"
        r'\bthe\s+\w+\s+(?:at|in|on|near)\b',   # "the car at the corner"
        r'\b(?:first|second|third|last)\s+\w+\b', # ordinal
        r'\bclosest|nearest|farthest\b',         # superlative spatial
    ]
    
    # Check for plural indicators
    expects_multiple = False
    for pattern in plural_patterns:
        if re.search(pattern, query_lower):
            expects_multiple = True
            break
    
    # Override if strong singular indicators present
    if expects_multiple:
        for pattern in singular_patterns:
            if re.search(pattern, query_lower):
                expects_multiple = False
                break
    
    # Extract target class (improved heuristic)
    # Look for "for the <class>" or "<class> in/at/on"
    target_class = None
    
    # Words to exclude from being detected as target class
    exclude_words = {'the', 'a', 'an', 'in', 'at', 'on', 'of', 'to', 'for', 'and', 'or',
                     'scene', 'image', 'picture', 'photo', 'view', 'area', 'region',
                     'seen', 'visible', 'present', 'located', 'found'}
    
    class_patterns = [
        # "for the aircrafts seen in" -> captures "aircrafts"
        r'for\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:seen|visible|in\s+the\s+image)',
        # "boxes for the ground track field" -> captures compound nouns
        r'boxes\s+for\s+(?:the\s+)?(.+?)\s+(?:seen|visible|in)',
        # "locate all vehicles" -> captures after "all"
        r'(?:locate|find)\s+(?:all\s+)?(?:the\s+)?(\w+s)\s+(?:in|$)',
        # "every storage tank" -> captures after "every"  
        r'every\s+(\w+(?:\s+\w+)?)',
        # "the airplane at the corner" -> captures noun before spatial
        r'the\s+(?:\w+\s+)?(\w+)\s+(?:at|in|on|near|located)',
        # "the large building" -> captures noun after adjective
        r'the\s+\w+\s+(\w+)\s+(?:at|in|on|near|$)',
    ]
    
    for pattern in class_patterns:
        match = re.search(pattern, query_lower)
        if match:
            candidate = match.group(1).strip()
            # Filter out excluded words
            words = candidate.split()
            filtered = [w for w in words if w not in exclude_words]
            if filtered:
                target_class = ' '.join(filtered)
                break
    
    # Check for spatial constraints (absolute position)
    spatial_constraint = bool(re.search(
        r'\b(?:top|bottom|left|right|center|corner|edge|middle)\b', 
        query_lower
    ))
    
    # Check for relative spatial constraints ("to the right of the truck")
    relative_spatial = get_relative_spatial_analysis(query)
    has_relative_constraint = relative_spatial.get('has_relative_constraint', False)
    
    return {
        'expects_multiple': expects_multiple,
        'target_class': target_class,
        'spatial_constraint': spatial_constraint,
        'relative_spatial': relative_spatial,
        'has_relative_constraint': has_relative_constraint,
        'query_type': 'multi' if expects_multiple else 'single'
    }


def select_multiple_objects(
    candidates: List[Dict],
    query_info: Dict,
    confidence_threshold: float = 0.3,
    max_objects: int = 50,
    nms_threshold: float = 0.5,
    min_size: float = 0.01
) -> List[Dict]:
    """
    Select multiple objects from candidates for plural queries.
    
    Uses confidence thresholding, size filtering, and NMS to get distinct objects.
    
    Args:
        candidates: List of OBB predictions with 'obbox' and 'sam_score'
        query_info: Output from detect_query_type()
        confidence_threshold: Minimum confidence to include (default 0.3 for multi-object)
        max_objects: Maximum number of objects to return
        nms_threshold: IoU threshold for NMS deduplication
        min_size: Minimum width or height (normalized) to filter tiny detections
    
    Returns:
        List of selected predictions
    """
    if not candidates:
        return []
    
    # Sort by confidence
    sorted_candidates = sorted(candidates, key=lambda x: x.get('sam_score', 0), reverse=True)
    
    # Filter by confidence AND size
    filtered = []
    for c in sorted_candidates:
        score = c.get('sam_score', 0)
        obb = c.get('obbox', [0.5, 0.5, 0.01, 0.01, 0])
        width, height = obb[2], obb[3]
        
        # Must pass confidence threshold AND be reasonably sized
        if score >= confidence_threshold and width >= min_size and height >= min_size:
            filtered.append(c)
    
    if not filtered:
        # If nothing passes threshold, return top candidate if it's reasonably sized
        for c in sorted_candidates:
            obb = c.get('obbox', [0.5, 0.5, 0.01, 0.01, 0])
            if obb[2] >= min_size and obb[3] >= min_size:
                return [c]
        return [sorted_candidates[0]] if sorted_candidates else []
    
    # Apply Non-Maximum Suppression to remove duplicates
    selected = []
    for candidate in filtered:
        if len(selected) >= max_objects:
            break
        
        # Check IoU with already selected
        is_duplicate = False
        for sel in selected:
            iou = compute_obb_iou(candidate['obbox'], sel['obbox'])
            if iou > nms_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            selected.append(candidate)
    
    return selected


def format_grounding_response(predictions: List[Dict]) -> List[Dict]:
    """
    Format predictions into Inter-IIT grounding response format.
    
    Args:
        predictions: List of OBB predictions with 'obbox' key
    
    Returns:
        List of {"object-id": str, "obbox": [cx, cy, w, h, angle]}
    """
    response = []
    for i, pred in enumerate(predictions):
        obbox = pred.get('obbox', pred.get('obbs', [0.5, 0.5, 0.1, 0.1, 0]))
        # Ensure proper format: [cx, cy, w, h, angle]
        if len(obbox) >= 5:
            formatted_obb = [round(v, 4) for v in obbox[:5]]
        else:
            formatted_obb = obbox + [0] * (5 - len(obbox))
        
        response.append({
            "object-id": str(i + 1),
            "obbox": formatted_obb
        })
    
    return response


def enhance_query_for_sam3(query: str) -> str:
    """
    Enhance user query with SAM3-friendly prompt suggestions.
    Domain-agnostic guidance for better segmentation results.
    """
    enhanced = f"""{query}

IMPORTANT: The SAM3 segment_phrase tool works best with simple, common noun phrases. If the exact query doesn't produce results, try these strategies:
- Use simpler synonyms (e.g., "running track" instead of "ground track field", "car" instead of "automobile")
- Try color-based descriptions if visible (e.g., "red building", "blue vehicle")
- Use the most generic category (e.g., "building", "road", "field", "vehicle", "tree")
- For compound terms, try each word separately (e.g., for "swimming pool" try "pool" then "water")
- For sports facilities: try "track", "field", "court", "pitch", "stadium"
- For infrastructure: try "road", "building", "bridge", "tower"
- For vegetation: try "tree", "forest", "grass", "field"
- For water bodies: try "water", "river", "lake", "pond"

Be creative with synonyms - the model may not know specialized terminology but understands common visual concepts."""
    
    return query


def normalize_angle_opencv(angle: float, w: float, h: float) -> Tuple[float, float, float]:
    """
    Normalize angle to OpenCV cv2.minAreaRect() convention: [-90, 0).
    """
    while angle < -90:
        angle += 180
        w, h = h, w
    while angle >= 0:
        angle -= 90
        w, h = h, w
    return w, h, angle


def compute_obbs_from_masks(masks: List[str], H: int, W: int) -> List[Dict]:
    """
    Convert RLE masks to Oriented Bounding Boxes (normalized).
    
    Output format: [cx, cy, w, h, angle]
    """
    obbs = []
    
    for idx, rle_str in enumerate(masks):
        try:
            if isinstance(rle_str, str):
                rle_dict = {"counts": rle_str.encode('utf-8'), "size": [H, W]}
            else:
                rle_dict = rle_str
            mask = mask_utils.decode(rle_dict)
            
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                continue
            
            largest = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest)
            (cx, cy), (w, h), angle = rect
            
            w_norm, h_norm, angle_norm = normalize_angle_opencv(angle, w, h)
            
            obbs.append({
                "object_id": idx + 1,
                "obbox": [
                    round(cx / W, 4),
                    round(cy / H, 4),
                    round(w_norm / W, 4),
                    round(h_norm / H, 4),
                    round(angle_norm, 2)
                ],
                "raw_rect": rect,
                "mask_pixels": int(np.sum(mask))
            })
        except Exception as e:
            print(f"  Warning: Error processing mask {idx}: {e}")
            continue
    
    return obbs


def corners_to_5param(corners: List[float], img_w: int = 1, img_h: int = 1) -> Optional[List[float]]:
    """
    Convert 8-point corner format to 5-param OBB format.
    """
    if len(corners) != 8:
        return None
    
    points = np.array(corners).reshape(4, 2) * np.array([img_w, img_h])
    points = points.astype(np.float32)
    
    rect = cv2.minAreaRect(points)
    (cx, cy), (w, h), angle = rect
    
    w_norm, h_norm, angle_norm = normalize_angle_opencv(angle, w, h)
    
    return [
        round(cx / img_w, 4),
        round(cy / img_h, 4),
        round(w_norm / img_w, 4),
        round(h_norm / img_h, 4),
        round(angle_norm, 2)
    ]


def obb_to_polygon(obb: List[float]) -> Optional[Polygon]:
    """Convert 5-param OBB to Shapely Polygon."""
    if not SHAPELY_AVAILABLE or len(obb) != 5:
        return None
    
    cx, cy, w, h, angle = obb
    angle_rad = np.radians(angle)
    
    hw, hh = w / 2, h / 2
    corners = np.array([
        [-hw, -hh],
        [hw, -hh],
        [hw, hh],
        [-hw, hh]
    ])
    
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    rotated = corners @ rotation.T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    
    return Polygon(rotated)


def compute_obb_iou(obb1: List[float], obb2: List[float], return_areas: bool = False) -> float:
    """Compute IoU between two 5-param OBBs.
    
    If return_areas=True, returns (iou, intersection_area, union_area)
    """
    poly1 = obb_to_polygon(obb1)
    poly2 = obb_to_polygon(obb2)
    
    if poly1 is None or poly2 is None:
        return (0.0, 0.0, 0.0) if return_areas else 0.0
    
    if not poly1.is_valid:
        poly1 = poly1.buffer(0)
    if not poly2.is_valid:
        poly2 = poly2.buffer(0)
    
    try:
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        iou = intersection / union if union > 0 else 0.0
        if return_areas:
            return (iou, intersection, union)
        return iou
    except Exception:
        return (0.0, 0.0, 0.0) if return_areas else 0.0


def compute_grounding_score(
    pred_obbs: List[List[float]], 
    gt_obbs: List[List[float]], 
    alpha: float = 2.5  # GeoNLI uses alpha=2.5 for count penalty
) -> Dict[str, float]:
    """
    Compute grounding score: S = CP × MeanIoU
    
    GeoNLI Evaluation: CP = exp(-α × |N_pred - N_ref|) with α = 2.5
    
    Also returns cumulative intersection/union for VRSBench's cumIoU metric.
    """
    n_pred = len(pred_obbs)
    n_gt = len(gt_obbs)
    
    count_penalty = np.exp(-alpha * abs(n_pred - n_gt))
    
    if n_gt == 0:
        mean_iou = 1.0 if n_pred == 0 else 0.0
        return {
            "mean_iou": mean_iou,
            "count_penalty": count_penalty,
            "grounding_score": count_penalty * mean_iou,
            "matched_ious": [],
            "cum_intersection": 0.0,
            "cum_union": 0.0,
        }
    
    if n_pred == 0:
        return {
            "mean_iou": 0.0,
            "count_penalty": count_penalty,
            "grounding_score": 0.0,
            "matched_ious": [0.0] * n_gt,
            "cum_intersection": 0.0,
            "cum_union": 0.0,
        }
    
    # Compute IoU matrix with areas for cumulative IoU
    iou_matrix = np.zeros((n_pred, n_gt))
    intersection_matrix = np.zeros((n_pred, n_gt))
    union_matrix = np.zeros((n_pred, n_gt))
    
    for i, pred in enumerate(pred_obbs):
        for j, gt in enumerate(gt_obbs):
            iou, inter, uni = compute_obb_iou(pred, gt, return_areas=True)
            iou_matrix[i, j] = iou
            intersection_matrix[i, j] = inter
            union_matrix[i, j] = uni
    
    # Greedy matching
    matched_ious = []
    matched_intersections = []
    matched_unions = []
    used_pred = set()
    used_gt = set()
    
    flat_indices = []
    for i in range(n_pred):
        for j in range(n_gt):
            flat_indices.append((iou_matrix[i, j], i, j))
    flat_indices.sort(reverse=True)
    
    for iou, pred_idx, gt_idx in flat_indices:
        if pred_idx not in used_pred and gt_idx not in used_gt:
            matched_ious.append(iou)
            matched_intersections.append(intersection_matrix[pred_idx, gt_idx])
            matched_unions.append(union_matrix[pred_idx, gt_idx])
            used_pred.add(pred_idx)
            used_gt.add(gt_idx)
    
    for gt_idx in range(n_gt):
        if gt_idx not in used_gt:
            matched_ious.append(0.0)
            matched_intersections.append(0.0)
            # For unmatched GTs, union is the GT's own area
            gt_poly = obb_to_polygon(gt_obbs[gt_idx])
            gt_area = gt_poly.area if gt_poly and gt_poly.is_valid else 0.0
            matched_unions.append(gt_area)
    
    mean_iou = np.mean(matched_ious) if matched_ious else 0.0
    max_iou = max(matched_ious) if matched_ious else 0.0
    
    # Cumulative IoU (VRSBench uses sum(I) / sum(U))
    cum_intersection = sum(matched_intersections)
    cum_union = sum(matched_unions)
    
    # Standard VRSBench metrics: Acc@threshold
    # A sample is "accurate" if the best matched IoU exceeds the threshold
    acc_05 = 1.0 if max_iou >= 0.5 else 0.0
    acc_07 = 1.0 if max_iou >= 0.7 else 0.0
    acc_03 = 1.0 if max_iou >= 0.3 else 0.0
    acc_09 = 1.0 if max_iou >= 0.9 else 0.0
    
    return {
        "mean_iou": round(mean_iou, 4),
        "max_iou": round(max_iou, 4),
        "count_penalty": round(count_penalty, 4),
        "grounding_score": round(count_penalty * mean_iou, 4),
        "matched_ious": [round(x, 4) for x in matched_ious],
        # Cumulative IoU components (for aggregation)
        "cum_intersection": cum_intersection,
        "cum_union": cum_union,
        # VRSBench standard metrics
        "acc_0.3": acc_03,
        "acc_0.5": acc_05,
        "acc_0.7": acc_07,
        "acc_0.9": acc_09,
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def draw_obb_on_image(img: np.ndarray, obb: List[float], color: Tuple[int, int, int], 
                      label: str = None, thickness: int = 2) -> np.ndarray:
    """Draw a single OBB on image."""
    H, W = img.shape[:2]
    cx, cy, w, h, angle = obb
    
    # Convert to pixel coordinates
    rect = ((cx * W, cy * H), (w * W, h * H), angle)
    box = cv2.boxPoints(rect)
    box = np.intp(box)  # Use np.intp instead of deprecated np.int0
    
    cv2.drawContours(img, [box], 0, color, thickness)
    
    if label:
        cv2.putText(img, label, (int(cx * W) - 10, int(cy * H) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return img


def save_visualization(img_np: np.ndarray, pred_obbs: List[List[float]], 
                       gt_obbs: List[List[float]], output_path: str, 
                       title: str = ""):
    """Save comparison visualization with GT (green) and Pred (red)."""
    vis_img = img_np.copy()
    
    # Draw GT in green
    for i, gt in enumerate(gt_obbs):
        draw_obb_on_image(vis_img, gt, (0, 255, 0), f"GT{i+1}", 2)
    
    # Draw predictions in red
    for i, pred in enumerate(pred_obbs):
        draw_obb_on_image(vis_img, pred, (255, 0, 0), f"P{i+1}", 2)
    
    # Add legend
    cv2.putText(vis_img, "GT (green) | Pred (red)", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if title:
        cv2.putText(vis_img, title[:50], (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    Image.fromarray(vis_img).save(output_path)


# =============================================================================
# Intermediate Mask Logger
# =============================================================================

class IntermediateMaskLogger:
    """Logger for intermediate SAM3 masks during agent reasoning."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.intermediate_dir = os.path.join(output_dir, "intermediate_masks")
        os.makedirs(self.intermediate_dir, exist_ok=True)
        
        self.step_count = 0
        self.log_entries = []
        
    def log_sam_call(self, text_prompt: str, outputs: Dict, 
                     output_json_path: str, output_image_path: str) -> Dict:
        """Log a SAM3 call with outputs."""
        self.step_count += 1
        
        # Copy visualization
        step_image_path = None
        if output_image_path and os.path.exists(output_image_path):
            safe_prompt = text_prompt.replace(' ', '_').replace('/', '_')[:30]
            step_image_path = os.path.join(
                self.intermediate_dir, 
                f"step_{self.step_count:02d}_{safe_prompt}.png"
            )
            shutil.copy(output_image_path, step_image_path)
        
        # Extract mask info
        num_masks = len(outputs.get("pred_masks", []))
        scores = outputs.get("pred_scores", [])
        
        # Compute OBBs
        H = outputs.get("orig_img_h", 0)
        W = outputs.get("orig_img_w", 0)
        masks = outputs.get("pred_masks", [])
        
        intermediate_obbs = []
        if H > 0 and W > 0 and masks:
            obb_dicts = compute_obbs_from_masks(masks, H, W)
            for i, obb in enumerate(obb_dicts):
                if i < len(scores):
                    obb["score"] = round(scores[i], 4)
                intermediate_obbs.append(obb)
        
        entry = {
            "step": self.step_count,
            "text_prompt": text_prompt,
            "num_masks": num_masks,
            "scores": [round(s, 4) for s in scores] if scores else [],
            "obbs": [obb["obbox"] for obb in intermediate_obbs],
            "visualization_path": os.path.basename(step_image_path) if step_image_path else None,
        }
        
        self.log_entries.append(entry)
        return entry
    
    def get_summary(self) -> Dict:
        """Get summary of intermediate steps."""
        return {
            "total_steps": self.step_count,
            "prompts_tried": [e["text_prompt"] for e in self.log_entries],
            "steps": self.log_entries
        }


# =============================================================================
# VRSBench Inference Pipeline
# =============================================================================

class VRSBenchPipeline:
    """VRSBench grounding inference pipeline."""
    
    def __init__(
        self,
        output_dir: str,
        vis_dir: str,
        use_agent: bool = True,
        enhance_queries: bool = False,
        max_iterations: int = 10,
        confidence_threshold: float = 0.1,
        alpha: float = 1.0,
        quiet: bool = False,
        use_spatial_reasoning: bool = True
    ):
        self.output_dir = output_dir
        self.vis_dir = vis_dir
        self.use_agent = use_agent
        self.enhance_queries = enhance_queries
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.alpha = alpha
        self.quiet = quiet
        self.use_spatial_reasoning = use_spatial_reasoning
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        
        self.sam3_model = None
        self.sam3_processor = None
        self.mllm = None
        self.sampling_params = None
        
    def _log(self, msg: str):
        """Print only if not in quiet mode."""
        if not self.quiet:
            print(msg)
        
    def initialize(self):
        """Initialize SAM3 and MLLM models."""
        self._log("\n" + "-"*60)
        self._log("Initializing VRSBench Pipeline")
        self._log("-"*60)
        
        self._log("\n[1/2] Loading SAM3 model...")
        self._init_sam3()
        
        if self.use_agent:
            self._log("\n[2/2] Loading MLLM...")
            self._init_mllm()
        else:
            self._log("\n[2/2] Skipping MLLM (direct mode)")
        
        self._log("\n" + "-"*60)
        self._log("Initialization complete")
        self._log("-"*60)
    
    def _init_sam3(self):
        """Initialize SAM3 model."""
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import sam3 as sam3_module
        
        sam3_root = os.path.dirname(sam3_module.__file__)
        bpe_path = os.path.join(sam3_root, "..", "assets", "bpe_simple_vocab_16e6.txt.gz")
        
        # Use local checkpoint if available, otherwise download from HF
        if SAM3_CHECKPOINT_PATH and os.path.exists(SAM3_CHECKPOINT_PATH):
            self._log(f"  Loading SAM3 from local checkpoint: {SAM3_CHECKPOINT_PATH}")
            self.sam3_model = build_sam3_image_model(
                bpe_path=bpe_path,
                checkpoint_path=SAM3_CHECKPOINT_PATH,
                load_from_HF=False
            )
        else:
            self._log("  Loading SAM3 from HuggingFace...")
            self.sam3_model = build_sam3_image_model(bpe_path=bpe_path, load_from_HF=True)
        
        self.sam3_processor = Sam3Processor(
            self.sam3_model, 
            confidence_threshold=self.confidence_threshold
        )
        self._log("  ✓ SAM3 model loaded")
    
    def _init_mllm(self):
        """Initialize MLLM using vLLM server client."""
        try:
            # Try to use the shared vLLM server client
            from vllm_server import get_vllm_client, is_server_running
            
            if is_server_running():
                self.mllm = get_vllm_client(auto_start=False)
                self._log("  ✓ Connected to existing vLLM server")
            else:
                # Try to start the server
                self.mllm = get_vllm_client(auto_start=True)
                if self.mllm:
                    self._log("  ✓ vLLM server started and connected")
                else:
                    self._log("  ✗ Failed to start vLLM server, falling back to direct SAM3")
                    self.use_agent = False
                    return
                    
        except ImportError:
            self._log("  ⚠ vllm_server module not found")
            self._log("  Falling back to direct SAM3 mode (no agent reasoning)")
            self._log("  To enable agent mode, ensure vllm_server.py is available and vLLM server is running")
            self.use_agent = False
        except Exception as e:
            self._log(f"  ✗ MLLM initialization failed: {e}")
            self._log("  Falling back to direct SAM3 mode")
            self.use_agent = False
    
    def run_inference(
        self,
        image_path: str,
        query: str,
        sample_id: str,
        gt_corners: List[float] = None
    ) -> Dict:
        """Run inference with comprehensive logging."""
        # Load image
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        H, W = img_np.shape[:2]
        
        # Create visualization folder for this image
        image_name = os.path.basename(image_path).replace('.', '_')
        image_vis_dir = os.path.join(self.vis_dir, image_name)
        os.makedirs(image_vis_dir, exist_ok=True)
        
        # Save input image
        input_path = os.path.join(image_vis_dir, "input.png")
        img.save(input_path)
        
        # Initialize mask logger
        mask_logger = IntermediateMaskLogger(image_vis_dir)
        
        # Convert GT corners to 5-param format
        gt_obbs = []
        if gt_corners and len(gt_corners) == 8:
            gt_obb = corners_to_5param(gt_corners)
            if gt_obb:
                gt_obbs = [gt_obb]
        
        # Run inference
        if self.use_agent and self.mllm is not None:
            result = self._run_agent_inference(
                image_path, query, image_vis_dir, mask_logger, H, W
            )
        else:
            result = self._run_direct_inference(
                img_np, query, image_vis_dir, mask_logger, H, W
            )
        
        # Add ground truth
        result["ground_truth_obbs"] = gt_obbs
        result["ground_truth_corners"] = gt_corners
        
        # Detect query type (single vs multiple objects)
        query_info = detect_query_type(query)
        result["query_info"] = query_info
        
        # Check for relative spatial constraints (e.g., "airplanes to the right of the truck")
        if query_info.get('has_relative_constraint', False):
            result["relative_spatial"] = query_info['relative_spatial']
            
            # For relative spatial queries, we need to:
            # 1. Detect reference objects (e.g., "the truck")
            # 2. Detect target objects (e.g., "airplanes")
            # 3. Filter targets by spatial relation to reference
            
            rel_info = query_info['relative_spatial']
            reference_class = rel_info.get('reference_class', '')
            target_class = rel_info.get('target_class', '')
            relation = rel_info.get('relation', '')
            
            # Run SAM3 to detect reference objects using the processor
            if reference_class and self.sam3_processor:
                # Detect reference objects first
                from sam3.agent.client_sam3 import call_sam_service
                import tempfile
                
                ref_out_folder = os.path.join(image_vis_dir, "reference_detection")
                os.makedirs(ref_out_folder, exist_ok=True)
                
                ref_json_path = call_sam_service(
                    sam3_processor=self.sam3_processor,
                    image_path=image_path,
                    text_prompt=reference_class,
                    output_folder_path=ref_out_folder
                )
                
                ref_obbs = []
                if ref_json_path and os.path.exists(ref_json_path):
                    with open(ref_json_path, 'r') as f:
                        ref_outputs = json.load(f)
                    ref_masks = ref_outputs.get("pred_masks", [])
                    ref_scores = ref_outputs.get("pred_scores", [])
                    if ref_masks:
                        ref_obb_dicts = compute_obbs_from_masks(ref_masks, H, W)
                        for i, obb_dict in enumerate(ref_obb_dicts):
                            score = ref_scores[i] if i < len(ref_scores) else 0.5
                            if score > 0.3:  # Reference threshold
                                ref_obbs.append({"obbox": obb_dict["obbox"], "score": score})
                
                result["reference_objects"] = ref_obbs
                result["reference_class"] = reference_class
                
                # If we have reference objects, filter target candidates
                if ref_obbs:
                    intermediate_masks = result.get("intermediate_masks", {})
                    all_candidates = []
                    for step in intermediate_masks.get("steps", []):
                        step_obbs = step.get("obbs", [])
                        step_scores = step.get("scores", [])
                        for i, obb in enumerate(step_obbs):
                            score = step_scores[i] if i < len(step_scores) else 0.5
                            all_candidates.append({"obbox": obb, "sam_score": score})
                    
                    # Use the first (highest confidence) reference object as anchor
                    ref_obb = ref_obbs[0]["obbox"]
                    constraint = rel_info.get('constraint')
                    
                    if constraint and all_candidates:
                        # Filter candidates by relative position
                        filtered = filter_by_relative_position(
                            all_candidates, ref_obb, constraint.relation
                        )
                        result["pre_filter_count"] = len(all_candidates)
                        result["post_filter_count"] = len(filtered)
                        result["relative_filtered"] = True
                        
                        if filtered:
                            # Apply multi-object selection on filtered candidates
                            selected = select_multiple_objects(
                                filtered,
                                query_info,
                                confidence_threshold=max(0.3, self.confidence_threshold),
                                nms_threshold=0.5,
                                min_size=0.01
                            )
                            pred_obbs = [s["obbox"] for s in selected]
                            result["final_obbs"] = pred_obbs
                            result["num_objects_selected"] = len(selected)
        
        # Apply spatial reasoning post-processing
        pred_obbs = result.get("final_obbs", [])
        spatial_analysis = get_spatial_analysis(query)
        ordinal_constraint = spatial_analysis.get("ordinal_constraint")
        
        result["spatial_analysis"] = {
            "has_constraint": spatial_analysis["has_spatial_constraint"],
            "terms": [st["term"] for st in spatial_analysis["spatial_terms"]],
            "regions": [st["region_name"] for st in spatial_analysis["spatial_terms"]],
            "target_region": spatial_analysis["target_region"],
            "ordinal_constraint": ordinal_constraint
        }
        
        if self.use_spatial_reasoning and spatial_analysis["has_spatial_constraint"]:
            # KEY INSIGHT: Use INTERMEDIATE masks for reranking, not just final selection
            # The agent often has multiple correct candidates but picks the wrong one
            intermediate_masks = result.get("intermediate_masks", {})
            all_intermediate_obbs = []
            
            # Collect ALL OBBs from intermediate steps
            for step in intermediate_masks.get("steps", []):
                step_obbs = step.get("obbs", [])
                step_scores = step.get("scores", [])
                for i, obb in enumerate(step_obbs):
                    score = step_scores[i] if i < len(step_scores) else 0.5
                    all_intermediate_obbs.append({
                        "obbox": obb,
                        "sam_score": score,
                        "step": step.get("step", 0),
                        "prompt": step.get("text_prompt", "")
                    })
            
            # If we have intermediate OBBs, use those for spatial reranking
            if all_intermediate_obbs:
                result["original_obbs"] = pred_obbs.copy() if pred_obbs else []
                result["intermediate_obb_count"] = len(all_intermediate_obbs)
                
                # Check if we have an ordinal constraint (e.g., "second from the right")
                if ordinal_constraint:
                    # Use ordinal reranking
                    ordinal_obj = OrdinalConstraint(
                        ordinal=ordinal_constraint['ordinal'],
                        direction=ordinal_constraint['direction'],
                        axis=ordinal_constraint['axis'],
                        ascending=ordinal_constraint['ascending']
                    )
                    # Pass target_region so we first filter to the spatial area
                    # e.g., "second from right in the TOP ROW" first filters to top, then ranks
                    target_region = spatial_analysis.get("target_region")
                    reranked = apply_ordinal_reranking(all_intermediate_obbs, ordinal_obj, target_region)
                    result["ordinal_reranked"] = True
                else:
                    # Use spatial region reranking
                    spatial_terms = spatial_analysis["spatial_terms"]
                    reranked = rerank_predictions_by_spatial(all_intermediate_obbs, spatial_terms)
                    result["ordinal_reranked"] = False
                
                # Log spatial scores for top candidates
                result["spatial_scores"] = [round(p.get("spatial_score", 0), 4) for p in reranked[:5]]
                result["spatial_candidates"] = [
                    {"center": (round(p["obbox"][0], 3), round(p["obbox"][1], 3)), 
                     "spatial_score": round(p.get("spatial_score", 0), 3)}
                    for p in reranked[:3]
                ]
                
                # Handle single vs multiple object selection
                if reranked:
                    if query_info['expects_multiple']:
                        # Multi-object query: select all valid candidates
                        # Use higher threshold (0.3) for multi-object to reduce false positives
                        selected = select_multiple_objects(
                            reranked, 
                            query_info,
                            confidence_threshold=max(0.3, self.confidence_threshold),
                            nms_threshold=0.5,
                            min_size=0.01
                        )
                        pred_obbs = [s["obbox"] for s in selected]
                        result["num_objects_selected"] = len(selected)
                    else:
                        # Single-object query: use best spatial match
                        best_match = reranked[0]
                        pred_obbs = [best_match["obbox"]]
                    
                    result["final_obbs"] = pred_obbs
                    result["spatial_reranked"] = True
                    result["selected_from_intermediate"] = True
                else:
                    result["spatial_reranked"] = False
            elif pred_obbs:
                # Fallback: rerank final OBBs if no intermediate available
                result["original_obbs"] = pred_obbs.copy()
                pred_dicts = [{"obbox": obb, "idx": i} for i, obb in enumerate(pred_obbs)]
                
                if ordinal_constraint:
                    ordinal_obj = OrdinalConstraint(
                        ordinal=ordinal_constraint['ordinal'],
                        direction=ordinal_constraint['direction'],
                        axis=ordinal_constraint['axis'],
                        ascending=ordinal_constraint['ascending']
                    )
                    reranked = apply_ordinal_reranking(pred_dicts, ordinal_obj)
                else:
                    spatial_terms = spatial_analysis["spatial_terms"]
                    reranked = rerank_predictions_by_spatial(pred_dicts, spatial_terms)
                
                result["spatial_scores"] = [round(p.get("spatial_score", 0), 4) for p in reranked]
                
                if reranked:
                    if query_info['expects_multiple']:
                        # Multi-object query: select all valid candidates
                        selected = select_multiple_objects(
                            reranked, 
                            query_info,
                            confidence_threshold=max(0.3, self.confidence_threshold),
                            nms_threshold=0.5,
                            min_size=0.01
                        )
                        pred_obbs = [s["obbox"] for s in selected]
                        result["num_objects_selected"] = len(selected)
                    else:
                        # Single-object query: use best match
                        best_match = reranked[0]
                        pred_obbs = [best_match["obbox"]]
                    
                    result["final_obbs"] = pred_obbs
                    result["spatial_reranked"] = True
            else:
                result["spatial_reranked"] = False
        else:
            # No spatial constraint - handle multi-object for non-spatial queries too
            if query_info['expects_multiple'] and pred_obbs:
                # For multi-object queries without spatial constraints,
                # return all high-confidence predictions
                intermediate_masks = result.get("intermediate_masks", {})
                all_candidates = []
                for step in intermediate_masks.get("steps", []):
                    step_obbs = step.get("obbs", [])
                    step_scores = step.get("scores", [])
                    for i, obb in enumerate(step_obbs):
                        score = step_scores[i] if i < len(step_scores) else 0.5
                        all_candidates.append({"obbox": obb, "sam_score": score})
                
                if all_candidates:
                    selected = select_multiple_objects(
                        all_candidates,
                        query_info,
                        confidence_threshold=max(0.3, self.confidence_threshold),
                        nms_threshold=0.5,
                        min_size=0.01
                    )
                    pred_obbs = [s["obbox"] for s in selected]
                    result["final_obbs"] = pred_obbs
                    result["num_objects_selected"] = len(selected)
            
            result["spatial_reranked"] = False
        
        # Compute evaluation metrics (with potentially reranked predictions)
        evaluation = compute_grounding_score(pred_obbs, gt_obbs, self.alpha)
        result["evaluation"] = evaluation
        
        # Format grounding response for Inter-IIT format
        pred_dicts = [{"obbox": obb} for obb in pred_obbs]
        result["grounding_response"] = format_grounding_response(pred_dicts)
        
        # Save comparison visualization
        comparison_path = os.path.join(image_vis_dir, "comparison.png")
        save_visualization(img_np, pred_obbs, gt_obbs, comparison_path, query[:50])
        result["visualization_path"] = comparison_path
        
        return result
    
    def _run_agent_inference(
        self,
        image_path: str,
        query: str,
        vis_dir: str,
        mask_logger: IntermediateMaskLogger,
        H: int,
        W: int
    ) -> Dict:
        """Run agent inference with intermediate logging."""
        from sam3.agent.agent_core import agent_inference
        from sam3.agent.client_sam3 import call_sam_service
        
        enhanced_query = enhance_query_for_sam3(query) if self.enhance_queries else query
        
        sam_out_folder = os.path.join(vis_dir, "sam_out")
        os.makedirs(sam_out_folder, exist_ok=True)
        
        def logging_sam_service(image_path, text_prompt, output_folder_path):
            json_path = call_sam_service(
                sam3_processor=self.sam3_processor,
                image_path=image_path,
                text_prompt=text_prompt,
                output_folder_path=output_folder_path
            )
            
            if json_path and os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    outputs = json.load(f)
                outputs["orig_img_h"] = H
                outputs["orig_img_w"] = W
                img_path = outputs.get("output_image_path", None)
                mask_logger.log_sam_call(text_prompt, outputs, json_path, img_path)
            
            return json_path
        
        def send_generate_request(messages):
            """Use VLLMClient's chat_for_agent method for agent-style messages."""
            return self.mllm.chat_for_agent(messages, max_tokens=4096, temperature=0.0)
        
        try:
            # Save RGB for agent
            rgb_path = os.path.join(vis_dir, "input_rgb.png")
            Image.open(image_path).convert("RGB").save(rgb_path)
            
            messages_history, final_outputs, rendered_image = agent_inference(
                img_path=rgb_path,
                initial_text_prompt=enhanced_query,
                send_generate_request=send_generate_request,
                call_sam_service=logging_sam_service,
                max_generations=self.max_iterations,
                debug=False,
            )
            
            # Extract final masks
            masks = final_outputs.get("pred_masks", [])
            scores = final_outputs.get("pred_scores", [])
            
            final_obbs = []
            if masks:
                obb_dicts = compute_obbs_from_masks(masks, H, W)
                for i, obb in enumerate(obb_dicts):
                    if i < len(scores):
                        obb["score"] = round(scores[i], 4)
                    final_obbs.append(obb["obbox"])
            
            # Save final mask visualization
            if rendered_image is not None:
                final_vis_path = os.path.join(vis_dir, "final_mask.png")
                if isinstance(rendered_image, np.ndarray):
                    Image.fromarray(rendered_image).save(final_vis_path)
                elif isinstance(rendered_image, Image.Image):
                    rendered_image.save(final_vis_path)
            
            # Extract reasoning
            reasoning = []
            for msg in messages_history:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and "text" in p]
                        content = " ".join(text_parts)
                    reasoning.append({"role": role, "content": content[:500]})
            
            return {
                "method": "agent",
                "query": query,
                "enhanced_query": enhanced_query if self.enhance_queries else None,
                "intermediate_masks": mask_logger.get_summary(),
                "final_obbs": final_obbs,
                "final_scores": [round(s, 4) for s in scores] if scores else [],
                "reasoning": reasoning,
                "image_size": [H, W]
            }
            
        except Exception as e:
            import traceback
            return {
                "method": "agent",
                "query": query,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "intermediate_masks": mask_logger.get_summary(),
                "final_obbs": [],
                "final_scores": [],
                "reasoning": [],
                "image_size": [H, W]
            }
    
    def _run_direct_inference(
        self,
        img_np: np.ndarray,
        query: str,
        vis_dir: str,
        mask_logger: IntermediateMaskLogger,
        H: int,
        W: int
    ) -> Dict:
        """Run direct SAM3 inference without agent."""
        try:
            outputs = self.sam3_processor.process(
                image=img_np,
                text=query
            )
            
            outputs["orig_img_h"] = H
            outputs["orig_img_w"] = W
            mask_logger.log_sam_call(query, outputs, None, None)
            
            masks = outputs.get("pred_masks", [])
            scores = outputs.get("pred_scores", [])
            
            final_obbs = []
            if masks:
                obb_dicts = compute_obbs_from_masks(masks, H, W)
                for i, obb in enumerate(obb_dicts):
                    if i < len(scores):
                        obb["score"] = round(scores[i], 4)
                    final_obbs.append(obb["obbox"])
            
            return {
                "method": "direct",
                "query": query,
                "intermediate_masks": mask_logger.get_summary(),
                "final_obbs": final_obbs,
                "final_scores": [round(s, 4) for s in scores] if scores else [],
                "image_size": [H, W]
            }
            
        except Exception as e:
            import traceback
            return {
                "method": "direct",
                "query": query,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "intermediate_masks": mask_logger.get_summary(),
                "final_obbs": [],
                "final_scores": [],
                "image_size": [H, W]
            }


# =============================================================================
# Data Loading
# =============================================================================

def load_vrsbench_validation(
    cache_dir: str = None,
    images_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    quiet: bool = False
) -> List[Dict]:
    """Load VRSBench validation set for grounding task."""
    from huggingface_hub import hf_hub_download
    
    if not quiet:
        print("Loading VRSBench validation set...")
    
    json_path = hf_hub_download(
        repo_id="xiang709/VRSBench",
        filename="VRSBench_EVAL_referring.json",
        repo_type="dataset",
        cache_dir=cache_dir
    )
    if not quiet:
        print(f"  JSON path: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if not quiet:
        print(f"  Total annotations: {len(data)}")
    
    if images_dir is None:
        images_dir = os.path.join(cache_dir, "Images_val")
    
    if not os.path.exists(images_dir):
        print(f"\n⚠️  Images directory not found: {images_dir}")
        print("   Please download and extract Images_val.zip from:")
        print("   https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_val.zip")
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    samples = []
    missing_images = 0
    
    for i, item in enumerate(data):
        if max_samples and len(samples) >= max_samples:
            break
        
        image_name = item.get("image_id", "")
        if not image_name:
            continue
            
        image_path = os.path.join(images_dir, image_name)
        
        if not os.path.exists(image_path):
            missing_images += 1
            if missing_images <= 5 and not quiet:
                print(f"  Warning: Image not found: {image_path}")
            continue
        
        sample = {
            "image_path": image_path,
            "image_name": image_name,
            "image_url": f"https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_val/{image_name}",
            "sample_idx": i,
            "question_id": item.get("question_id", i),
            "query": item.get("question", ""),
            "obj_corner": item.get("obj_corner", []),
            "category": item.get("obj_cls", ""),
            "is_unique": item.get("unique", False)
        }
        
        if sample["query"] and sample["obj_corner"]:
            samples.append(sample)
    
    if missing_images > 5 and not quiet:
        print(f"  ... and {missing_images - 5} more missing images")
    
    if not quiet:
        print(f"  Loaded {len(samples)} valid samples")
    return samples


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="VRSBench Grounding Baseline")
    parser.add_argument("--output-dir", "-o", type=str, 
                        default="./outputs/grounding",
                        help="Output directory for results")
    parser.add_argument("--vis-dir", "-v", type=str,
                        default="./outputs/visualizations",
                        help="Visualization directory")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Path to Images_val folder")
    parser.add_argument("--max-samples", "-n", type=int, default=None,
                        help="Maximum samples to process")
    parser.add_argument("--use-agent", action="store_true", default=True,
                        help="Use agent pipeline")
    parser.add_argument("--no-agent", action="store_true",
                        help="Use direct SAM3 without agent")
    parser.add_argument("--enhance-queries", action="store_true", default=True,
                        help="Add SAM3-friendly hints")
    parser.add_argument("--no-enhance", action="store_true",
                        help="Disable query enhancement")
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="Max agent iterations")
    parser.add_argument("--confidence-threshold", type=float, default=0.1,
                        help="SAM3 confidence threshold")
    parser.add_argument("--alpha", type=float, default=2.5,
                        help="Count penalty weight (GeoNLI uses 2.5)")
    parser.add_argument("--resume-from", type=int, default=0,
                        help="Resume from sample index")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress all output except progress bar")
    parser.add_argument("--no-spatial", action="store_true",
                        help="Disable spatial reasoning post-processing")
    
    args = parser.parse_args()
    
    # Quiet mode - suppress all non-essential output
    QUIET = args.quiet
    use_spatial = not args.no_spatial
    
    use_agent = args.use_agent and not args.no_agent
    enhance_queries = args.enhance_queries and not args.no_enhance
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    run_vis_dir = os.path.join(args.vis_dir, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(run_vis_dir, exist_ok=True)
    
    # Save config
    config = {
        "use_agent": use_agent,
        "enhance_queries": enhance_queries,
        "use_spatial_reasoning": use_spatial,
        "max_iterations": args.max_iterations,
        "confidence_threshold": args.confidence_threshold,
        "alpha": args.alpha,
        "max_samples": args.max_samples,
        "images_dir": args.images_dir,
        "timestamp": timestamp
    }
    with open(os.path.join(run_output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Initialize pipeline
    pipeline = VRSBenchPipeline(
        output_dir=run_output_dir,
        vis_dir=run_vis_dir,
        use_agent=use_agent,
        enhance_queries=enhance_queries,
        max_iterations=args.max_iterations,
        confidence_threshold=args.confidence_threshold,
        alpha=args.alpha,
        quiet=QUIET,
        use_spatial_reasoning=use_spatial
    )
    pipeline.initialize()
    
    # Load dataset
    samples = load_vrsbench_validation(
        images_dir=args.images_dir,
        max_samples=args.max_samples,
        quiet=QUIET
    )
    
    # Results file
    results_file = os.path.join(run_output_dir, "results.jsonl")
    
    # Running metrics
    total_score = 0.0
    total_iou = 0.0
    total_max_iou = 0.0
    total_acc_03 = 0.0
    total_acc_05 = 0.0
    total_acc_07 = 0.0
    total_acc_09 = 0.0
    processed = 0
    
    # VRSBench-specific metrics: Unique vs Non-Unique breakdown
    # These match the paper's Table 3 exactly
    unique_metrics = {"count": 0, "acc_05": 0.0, "acc_07": 0.0, "cum_i": 0.0, "cum_u": 0.0}
    nonunique_metrics = {"count": 0, "acc_05": 0.0, "acc_07": 0.0, "cum_i": 0.0, "cum_u": 0.0}
    
    # Cumulative IoU (cumI/cumU) - different from mean IoU
    total_cum_i = 0.0  # Sum of intersection areas
    total_cum_u = 0.0  # Sum of union areas
    
    if not QUIET:
        print(f"\nProcessing {len(samples)} samples...")
        print(f"Results: {results_file}")
        print(f"Visualizations: {run_vis_dir}")
        print("-" * 60)
    
    # Timing for average time per query
    import time
    start_time = time.time()
    query_times = []
    
    # Create progress bar with dynamic postfix
    pbar = tqdm(samples, desc="Processing", disable=False)
    
    for idx, sample in enumerate(pbar):
        if idx < args.resume_from:
            continue
        
        try:
            query_start = time.time()
            
            image_path = sample["image_path"]
            image_name = sample["image_name"]
            query = sample["query"]
            obj_corner = sample["obj_corner"]
            question_id = sample["question_id"]
            
            sample_id = f"q{question_id}_{image_name.replace('.', '_')}"
            
            # Run inference
            result = pipeline.run_inference(
                image_path,
                query,
                sample_id,
                gt_corners=obj_corner
            )
            
            # Track query time
            query_time = time.time() - query_start
            query_times.append(query_time)
            
            # Build output record
            output_record = {
                "question_id": question_id,
                "sample_idx": sample["sample_idx"],
                "image_name": image_name,
                "image_url": sample["image_url"],
                "query": query,
                "category": sample["category"],
                "is_unique": sample["is_unique"],
                "ground_truth_corners": obj_corner,
                "ground_truth_obbs": result.get("ground_truth_obbs", []),
                "intermediate_masks": result.get("intermediate_masks", {}),
                "original_obbs": result.get("original_obbs"),  # Before spatial reranking
                "final_obbs": result.get("final_obbs", []),
                "final_scores": result.get("final_scores", []),
                "spatial_analysis": result.get("spatial_analysis", {}),
                "spatial_reranked": result.get("spatial_reranked", False),
                "spatial_scores": result.get("spatial_scores", []),
                "evaluation": result.get("evaluation", {}),
                "image_size": result.get("image_size", []),
                "method": result.get("method", ""),
                "reasoning": result.get("reasoning", [])
            }
            
            # Update metrics
            eval_metrics = result.get("evaluation", {})
            current_score = eval_metrics.get("grounding_score", 0.0)
            current_iou = eval_metrics.get("mean_iou", 0.0)
            total_score += current_score
            total_iou += current_iou
            total_acc_03 += eval_metrics.get("acc_0.3", 0.0)
            total_acc_05 += eval_metrics.get("acc_0.5", 0.0)
            total_acc_07 += eval_metrics.get("acc_0.7", 0.0)
            total_acc_09 += eval_metrics.get("acc_0.9", 0.0)
            total_max_iou += eval_metrics.get("max_iou", 0.0)
            
            # VRSBench cumulative IoU
            total_cum_i += eval_metrics.get("cum_intersection", 0.0)
            total_cum_u += eval_metrics.get("cum_union", 0.0)
            
            # VRSBench Unique vs Non-Unique breakdown
            is_unique = sample.get("is_unique", True)
            sample_acc_05 = eval_metrics.get("acc_0.5", 0.0)
            sample_acc_07 = eval_metrics.get("acc_0.7", 0.0)
            sample_cum_i = eval_metrics.get("cum_intersection", 0.0)
            sample_cum_u = eval_metrics.get("cum_union", 0.0)
            
            if is_unique:
                unique_metrics["count"] += 1
                unique_metrics["acc_05"] += sample_acc_05
                unique_metrics["acc_07"] += sample_acc_07
                unique_metrics["cum_i"] += sample_cum_i
                unique_metrics["cum_u"] += sample_cum_u
            else:
                nonunique_metrics["count"] += 1
                nonunique_metrics["acc_05"] += sample_acc_05
                nonunique_metrics["acc_07"] += sample_acc_07
                nonunique_metrics["cum_i"] += sample_cum_i
                nonunique_metrics["cum_u"] += sample_cum_u
            
            processed += 1
            
            # Update progress bar with live metrics
            avg_time = sum(query_times) / len(query_times) if query_times else 0
            eta_seconds = avg_time * (len(samples) - idx - 1)
            eta_str = f"{int(eta_seconds // 60)}m{int(eta_seconds % 60)}s"
            
            pbar.set_postfix({
                'IoU': f'{current_iou:.2f}',
                'Score': f'{current_score:.2f}',
                'AvgIoU': f'{total_iou/processed:.3f}',
                'Acc@0.5': f'{total_acc_05/processed:.1%}',
                'Acc@0.7': f'{total_acc_07/processed:.1%}',
                't/q': f'{avg_time:.1f}s',
                'ETA': eta_str
            })
            
            # Write to JSONL
            with open(results_file, "a") as f:
                f.write(json.dumps(output_record, default=str) + "\n")
                
        except Exception as e:
            if not QUIET:
                print(f"\n  Error processing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
            continue
    
    pbar.close()
    
    # Final summary with all metrics
    # Calculate VRSBench-specific aggregated metrics
    cum_iou_all = round(total_cum_i / total_cum_u, 4) if total_cum_u > 0 else 0.0
    
    # Unique metrics
    unique_acc_05 = round(unique_metrics["acc_05"] / unique_metrics["count"], 4) if unique_metrics["count"] > 0 else 0.0
    unique_acc_07 = round(unique_metrics["acc_07"] / unique_metrics["count"], 4) if unique_metrics["count"] > 0 else 0.0
    unique_cum_iou = round(unique_metrics["cum_i"] / unique_metrics["cum_u"], 4) if unique_metrics["cum_u"] > 0 else 0.0
    
    # Non-unique metrics  
    nonunique_acc_05 = round(nonunique_metrics["acc_05"] / nonunique_metrics["count"], 4) if nonunique_metrics["count"] > 0 else 0.0
    nonunique_acc_07 = round(nonunique_metrics["acc_07"] / nonunique_metrics["count"], 4) if nonunique_metrics["count"] > 0 else 0.0
    nonunique_cum_iou = round(nonunique_metrics["cum_i"] / nonunique_metrics["cum_u"], 4) if nonunique_metrics["cum_u"] > 0 else 0.0
    
    summary = {
        "total_samples": len(samples),
        "total_queries_processed": processed,
        # Primary metrics
        "average_grounding_score": round(total_score / processed, 4) if processed > 0 else 0.0,
        "average_mean_iou": round(total_iou / processed, 4) if processed > 0 else 0.0,
        "average_max_iou": round(total_max_iou / processed, 4) if processed > 0 else 0.0,
        # VRSBench cumulative IoU (sum_I / sum_U)
        "cumulative_iou": cum_iou_all,
        # VRSBench standard accuracy metrics (All)
        "acc_0.3": round(total_acc_03 / processed, 4) if processed > 0 else 0.0,
        "acc_0.5": round(total_acc_05 / processed, 4) if processed > 0 else 0.0,
        "acc_0.7": round(total_acc_07 / processed, 4) if processed > 0 else 0.0,
        "acc_0.9": round(total_acc_09 / processed, 4) if processed > 0 else 0.0,
        # VRSBench Unique object metrics (matches paper Table 3)
        "unique": {
            "count": unique_metrics["count"],
            "acc_0.5": unique_acc_05,
            "acc_0.7": unique_acc_07,
            "cum_iou": unique_cum_iou,
        },
        # VRSBench Non-Unique object metrics (matches paper Table 3)
        "non_unique": {
            "count": nonunique_metrics["count"],
            "acc_0.5": nonunique_acc_05,
            "acc_0.7": nonunique_acc_07,
            "cum_iou": nonunique_cum_iou,
        },
        # Timing statistics
        "total_time_seconds": round(time.time() - start_time, 2),
        "avg_time_per_query": round(sum(query_times) / len(query_times), 2) if query_times else 0.0,
        "min_time_per_query": round(min(query_times), 2) if query_times else 0.0,
        "max_time_per_query": round(max(query_times), 2) if query_times else 0.0,
        # Config and paths
        "config": config,
        "results_file": results_file,
        "visualizations_dir": run_vis_dir
    }
    
    with open(os.path.join(run_output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Calculate total time
    total_time = time.time() - start_time
    total_minutes = int(total_time // 60)
    total_seconds = int(total_time % 60)
    
    # Always print final summary (even in quiet mode)
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Samples: {len(samples)}")
    print(f"Processed: {processed}")
    print(f"Total Time: {total_minutes}m {total_seconds}s")
    print(f"Avg Time/Query: {summary['avg_time_per_query']:.2f}s")
    print(f"\n--- Primary Metrics ---")
    print(f"Average Grounding Score: {summary['average_grounding_score']:.4f}")
    print(f"Average Mean IoU: {summary['average_mean_iou']:.4f}")
    print(f"Average Max IoU: {summary['average_max_iou']:.4f}")
    print(f"Cumulative IoU (cumI/cumU): {summary['cumulative_iou']:.4f}")
    print(f"\n--- VRSBench Accuracy Metrics (All) ---")
    print(f"Acc@0.3: {summary['acc_0.3']:.2%}")
    print(f"Acc@0.5: {summary['acc_0.5']:.2%}")
    print(f"Acc@0.7: {summary['acc_0.7']:.2%}")
    print(f"Acc@0.9: {summary['acc_0.9']:.2%}")
    print(f"\n--- VRSBench Metrics by Category (Paper Table 3 format) ---")
    print(f"                   | Acc@0.5 | Acc@0.7 | cumIoU")
    print(f"  Unique ({summary['unique']['count']:5d}) | {summary['unique']['acc_0.5']:6.2%} | {summary['unique']['acc_0.7']:6.2%} | {summary['unique']['cum_iou']:.4f}")
    print(f"  Non-Unique ({summary['non_unique']['count']:5d}) | {summary['non_unique']['acc_0.5']:6.2%} | {summary['non_unique']['acc_0.7']:6.2%} | {summary['non_unique']['cum_iou']:.4f}")
    print(f"  All ({processed:5d}) | {summary['acc_0.5']:6.2%} | {summary['acc_0.7']:6.2%} | {summary['cumulative_iou']:.4f}")
    print(f"\nResults: {results_file}")
    print(f"Visualizations: {run_vis_dir}")
    print(f"Summary: {os.path.join(run_output_dir, 'summary.json')}")


# =============================================================================
# API-friendly wrapper functions for integration with main.py
# =============================================================================

# Global pipeline instance for reuse
_grounding_pipeline = None


def load_grounding_model(
    use_agent: bool = True,
    confidence_threshold: float = 0.1,
    use_spatial_reasoning: bool = True
) -> VRSBenchPipeline:
    """
    Load and initialize the grounding pipeline.
    
    Args:
        use_agent: Whether to use MLLM for reasoning
        confidence_threshold: Confidence threshold for SAM3
        use_spatial_reasoning: Whether to apply spatial reasoning
        
    Returns:
        Initialized VRSBenchPipeline
    """
    global _grounding_pipeline
    
    if _grounding_pipeline is not None:
        return _grounding_pipeline
    
    import tempfile
    output_dir = os.path.join(tempfile.gettempdir(), "grounding_output")
    vis_dir = os.path.join(tempfile.gettempdir(), "grounding_vis")
    
    pipeline = VRSBenchPipeline(
        output_dir=output_dir,
        vis_dir=vis_dir,
        use_agent=use_agent,
        enhance_queries=False,
        max_iterations=10,
        confidence_threshold=confidence_threshold,
        alpha=1.0,
        quiet=True,  # Suppress logging for API usage
        use_spatial_reasoning=use_spatial_reasoning
    )
    pipeline.initialize()
    
    _grounding_pipeline = pipeline
    return pipeline


def answer_grounding_query(
    image_path: str,
    query: str,
    spatial_resolution_m: float = 1.0,
    pipeline: VRSBenchPipeline = None
) -> Dict[str, Any]:
    """
    Answer a grounding query for an image.
    
    This is the main API function for grounding inference.
    
    Args:
        image_path: Path to the image file
        query: Grounding query (e.g., "the red car on the left")
        spatial_resolution_m: Spatial resolution in meters per pixel
        pipeline: Optional pre-initialized pipeline
        
    Returns:
        Dict with grounding results:
            - "bounding_boxes": List of OBB boxes [cx, cy, w, h, angle] (normalized 0-1)
            - "count": Number of objects found
            - "success": Whether inference succeeded
    """
    import uuid
    
    # Get or create pipeline
    if pipeline is None:
        pipeline = load_grounding_model()
    
    try:
        # Generate unique sample ID
        sample_id = str(uuid.uuid4())[:8]
        
        # Run inference
        result = pipeline.run_inference(
            image_path=image_path,
            query=query,
            sample_id=sample_id,
            gt_corners=None  # No ground truth for API calls
        )
        
        # Extract final OBBs - these should be normalized [cx, cy, w, h, angle]
        final_obbs = result.get("final_obbs", [])
        
        # Format response - return just the boxes as list of [cx, cy, w, h, angle]
        response = {
            "bounding_boxes": final_obbs,
            "count": len(final_obbs),
            "success": True
        }
        
        return response
        
    except Exception as e:
        return {
            "bounding_boxes": [],
            "count": 0,
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    main()