#!/usr/bin/env python3
"""
Numeric VQA Pipeline for GeoNLI/ISRO Inter-IIT

Based on the proven agent_post_process.py pipeline, extended for numeric queries:
- COUNT: "How many storage tanks are present in the scene?"
- AREA: "What is the area of the blue region in the larger swimming pool in meters square?"
- LENGTH/WIDTH: "What is the length of the runway?"

Pipeline:
1. Parse query to determine type (count, area, length, etc.)
2. Use SAM3 agent to detect/segment relevant objects (reusing agent_post_process logic)
3. Apply MLLM for complex selection (blue region, larger pool)
4. Compute numeric answer using spatial resolution metadata

Author: GeoNLI Team
"""

import os
import re
import sys
import json
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm
from pycocotools import mask as mask_utils
import time
import logging

# Import centralized config
try:
    from config import (
        get_adapter_path, BASE_MODEL_ID, HF_HOME, TORCH_HOME, 
        SAM3_PATH as CONFIG_SAM3_PATH, SAM3_CHECKPOINT_PATH
    )
    # Use config paths if available
    os.environ["HF_HOME"] = HF_HOME
    os.environ["TRANSFORMERS_CACHE"] = HF_HOME
    os.environ["TORCH_HOME"] = TORCH_HOME
    SAM3_PATH = CONFIG_SAM3_PATH
except ImportError:
    BASE_MODEL_ID = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"
    get_adapter_path = None
    SAM3_CHECKPOINT_PATH = None
    # Environment setup (fallback for standalone use)
    os.environ.setdefault("HF_HOME", "/workspace/cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/cache/huggingface")
    os.environ.setdefault("TORCH_HOME", "/workspace/cache/torch")
    SAM3_PATH = os.environ.get("SAM3_PATH", "/workspace/sam3")

# Logger setup
logger = logging.getLogger(__name__)

# Default adapter directory
ADAPTER_DIR = "numeric_sft_0.67_qwen2.54bit"

# Enable TF32 for faster inference on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Add SAM3 to path if exists
if os.path.exists(SAM3_PATH):
    sys.path.insert(0, SAM3_PATH)


# =============================================================================
# Model Loading and Inference Functions (for API compatibility)
# =============================================================================

def load_numeric_model(adapter_dir=None, load_in_4bit=True):
    """
    Load numeric VQA model with adapter weights.
    Compatible with main.py API interface.
    """
    try:
        from unsloth import FastVisionModel
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Determine correct path using config or fallback
        if adapter_dir and os.path.isdir(adapter_dir):
            model_path = adapter_dir
        elif get_adapter_path:
            model_path = get_adapter_path("numeric")
        elif os.path.isdir(ADAPTER_DIR):
            model_path = ADAPTER_DIR
        else:
            logger.warning(f"Adapter not found, loading base model.")
            model_path = BASE_MODEL_ID

        logger.info(f"Loading numeric model from: {model_path}")
        
        model, tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=load_in_4bit,
        )
        
        # CRITICAL: Enable Unsloth inference optimizations
        FastVisionModel.for_inference(model)
        model.to(device)
        
        logger.info("Numeric model loaded successfully")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading numeric model: {e}")
        raise


def extract_number(text):
    """
    Robustly extracts the first number found in text.
    Handles 'no vehicles' -> 0, 'one' -> 1, etc.
    """
    text = str(text).lower().strip()
    
    # Text-to-number mapping for common cases
    word_map = {
        'no': 0, 'zero': 0, 'none': 0,
        'one': 1, 'two': 2, 'three': 3, 
        'four': 4, 'five': 5, 'six': 6, 
        'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    
    # Check for direct word matches first
    if text in word_map:
        return word_map[text]
    
    # Heuristic: "no vehicles" or "none"
    if text.startswith("no ") or text == "none":
        return 0

    # Try to find floating point numbers first (for area/length queries)
    float_matches = re.findall(r'\d+\.?\d*', text)
    if float_matches:
        val = float(float_matches[0])
        # Return int if it's a whole number, otherwise float
        return int(val) if val == int(val) else val
    
    # Check for word numbers inside sentences
    for word, num in word_map.items():
        if f" {word} " in f" {text} ":
            return num

    return -1  # Indicator for "Could not find a number"


def answer_numeric_question(
    model,
    tokenizer,
    image,
    instruction,
    device="cuda",
    max_new_tokens=10,
    temperature=0.1,
    spatial_resolution_m=1.0
):
    """
    Answer numeric question about an image.
    Compatible with main.py API interface.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        image: PIL Image or path to image
        instruction: The question to answer
        device: Device to run on
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        spatial_resolution_m: Spatial resolution in meters per pixel (for area/length queries)
    """
    try:
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Prepare messages for numeric query
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"{instruction}\nAnswer with a single number."}
                ]
            }
        ]
        
        # Apply chat template
        input_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(
            images=[image],
            text=[input_text],
            return_tensors="pt"
        ).to(device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                temperature=temperature,
                do_sample=False
            )
        
        # Decode
        raw_pred = tokenizer.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0].strip()
        
        # Extract number
        value = extract_number(raw_pred)
        
        # Return the numeric value (or the raw prediction if extraction fails)
        if value != -1:
            return value
        else:
            # Try to return raw text if no number found
            return raw_pred
    
    except Exception as e:
        logger.error(f"Error answering numeric question: {e}")
        return -1


# =============================================================================
# Numeric Query Types
# =============================================================================

class NumericQueryType(Enum):
    """Types of numeric queries."""
    COUNT = "count"           # How many X are there?
    AREA = "area"             # What is the area of X?
    LENGTH = "length"         # What is the length of X?
    WIDTH = "width"           # What is the width of X?
    PERIMETER = "perimeter"   # What is the perimeter of X?
    DISTANCE = "distance"     # What is the distance between X and Y?
    RATIO = "ratio"           # What is the ratio of X to Y?
    UNKNOWN = "unknown"


@dataclass
class ParsedNumericQuery:
    """Parsed numeric query structure."""
    original_query: str
    query_type: NumericQueryType
    target_object: str                    # Object to measure/count
    target_attribute: Optional[str]       # e.g., "blue region", "larger"
    reference_object: Optional[str]       # For distance/ratio queries
    unit: Optional[str]                   # meters, meters square, etc.
    confidence: float = 0.9


# =============================================================================
# Query Parser
# =============================================================================

class NumericQueryParser:
    """Parse numeric queries to extract type and target."""
    
    def __init__(self, mllm_client=None):
        self.mllm_client = mllm_client
    
    def parse(self, query: str) -> ParsedNumericQuery:
        """Parse a numeric query to extract type and target."""
        # Try MLLM first if available
        if self.mllm_client is not None:
            try:
                return self._mllm_parse(query)
            except Exception as e:
                print(f"  MLLM parse failed: {e}, falling back to regex")
        
        # Fallback to regex
        return self._regex_parse(query)
    
    def _mllm_parse(self, query: str) -> ParsedNumericQuery:
        """Parse using MLLM for complex queries."""
        prompt = self._format_parse_prompt(query)
        response = self.mllm_client.generate(prompt, max_tokens=256)
        
        parsed = self._extract_json(response)
        if parsed:
            return self._build_parsed_query(query, parsed)
        
        # Fallback to regex
        return self._regex_parse(query)
    
    def _format_parse_prompt(self, query: str) -> str:
        return f"""Parse this numeric query about a satellite/aerial image.

Query: "{query}"

Extract:
1. query_type: One of [count, area, length, width, perimeter, distance, ratio]
2. target_object: The main object to measure/count (e.g., "storage tanks", "swimming pool", "runway")
3. target_attribute: Any specific attribute (e.g., "blue region", "larger", "left side") or null
4. reference_object: For distance/ratio, the second object, or null
5. unit: Expected unit (e.g., "meters", "meters square", "count") or null

Output as JSON:
{{
    "query_type": "...",
    "target_object": "...",
    "target_attribute": "..." or null,
    "reference_object": "..." or null,
    "unit": "..." or null
}}

JSON:"""
    
    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON from response."""
        patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{[^{}]*"query_type"[^{}]*\})',
            r'(\{[^{}]*\})',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    cleaned = match.strip()
                    cleaned = re.sub(r',\s*}', '}', cleaned)
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    continue
        return None
    
    def _build_parsed_query(self, query: str, parsed: dict) -> ParsedNumericQuery:
        """Build ParsedNumericQuery from parsed dict."""
        query_type_str = parsed.get("query_type", "unknown").lower()
        query_type = NumericQueryType.UNKNOWN
        for qt in NumericQueryType:
            if qt.value == query_type_str:
                query_type = qt
                break
        
        return ParsedNumericQuery(
            original_query=query,
            query_type=query_type,
            target_object=parsed.get("target_object", "object"),
            target_attribute=parsed.get("target_attribute"),
            reference_object=parsed.get("reference_object"),
            unit=parsed.get("unit"),
        )
    
    def _regex_parse(self, query: str) -> ParsedNumericQuery:
        """Fallback regex parsing - reliable for standard queries."""
        query_lower = query.lower()
        
        # Determine query type
        query_type = NumericQueryType.UNKNOWN
        if any(w in query_lower for w in ["how many", "count", "number of"]):
            query_type = NumericQueryType.COUNT
        elif "area" in query_lower or "square" in query_lower:
            query_type = NumericQueryType.AREA
        elif "length" in query_lower or "long" in query_lower:
            query_type = NumericQueryType.LENGTH
        elif "width" in query_lower or "wide" in query_lower:
            query_type = NumericQueryType.WIDTH
        elif "perimeter" in query_lower:
            query_type = NumericQueryType.PERIMETER
        elif "distance" in query_lower or "between" in query_lower:
            query_type = NumericQueryType.DISTANCE
        elif "ratio" in query_lower:
            query_type = NumericQueryType.RATIO
        
        # Extract target object
        target_object = self._extract_target_object(query_lower)
        
        # Extract target attribute (color + region, size modifiers)
        target_attribute = self._extract_target_attribute(query_lower)
        
        # Extract unit
        unit = self._extract_unit(query_lower)
        
        return ParsedNumericQuery(
            original_query=query,
            query_type=query_type,
            target_object=target_object,
            target_attribute=target_attribute,
            reference_object=None,
            unit=unit,
        )
    
    def _extract_target_object(self, query_lower: str) -> str:
        """Extract target object from query."""
        # Common RS objects (in priority order)
        objects = [
            "storage tank", "swimming pool", "ground track field", 
            "baseball diamond", "basketball court", "tennis court",
            "soccer field", "runway", "airplane", "aircraft",
            "vehicle", "ship", "harbor", "building", "road", "bridge",
            "helicopter", "train", "windmill", "dam", "pool",
            "tank", "plane", "car", "truck", "boat"
        ]
        for obj in objects:
            if obj in query_lower:
                return obj
        
        # Try to extract via pattern
        patterns = [
            r"how many (\w+(?:\s+\w+)?)",
            r"area of (?:the )?(?:\w+ )?(\w+(?:\s+\w+)?)",
            r"length of (?:the )?(\w+(?:\s+\w+)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                return match.group(1).strip()
        
        return "object"
    
    def _extract_target_attribute(self, query_lower: str) -> Optional[str]:
        """Extract target attribute from query."""
        colors = ["blue", "red", "green", "yellow", "white", "gray", "grey", 
                  "orange", "brown", "cyan", "turquoise", "black", "pink", "purple"]
        size_modifiers = ["larger", "largest", "smaller", "smallest", "bigger", "biggest"]
        
        target_attribute = None
        
        # Look for "COLOR region" pattern
        for color in colors:
            if color in query_lower and "region" in query_lower:
                target_attribute = f"{color} region"
                break
        
        # Also check for size modifiers like "larger swimming pool"
        for mod in size_modifiers:
            if mod in query_lower:
                if target_attribute:
                    target_attribute = f"{mod} {target_attribute}"
                else:
                    target_attribute = mod
                break
        
        return target_attribute
    
    def _extract_unit(self, query_lower: str) -> Optional[str]:
        """Extract expected unit from query."""
        if "meters square" in query_lower or "square meters" in query_lower or "m²" in query_lower:
            return "meters square"
        elif "meters" in query_lower:
            return "meters"
        elif "kilometers" in query_lower or "km" in query_lower:
            return "kilometers"
        return None


# =============================================================================
# Color Extraction Utilities
# =============================================================================

def extract_color_region(image: np.ndarray, mask: np.ndarray, color_name: str) -> np.ndarray:
    """
    Extract a specific color region from within a mask.
    
    For satellite imagery, we use relative color comparison instead of absolute HSV
    thresholds since colors appear differently due to atmospheric effects, sensor
    characteristics, and lighting conditions.
    
    Args:
        image: RGB image as numpy array
        mask: Binary mask (H, W) where 1 indicates the object region
        color_name: Name of the color to extract (e.g., "blue", "green")
    
    Returns:
        Binary mask of the color region within the original mask
    """
    color_name_lower = color_name.lower().strip()
    H, W = mask.shape
    
    # Get RGB values at masked pixels
    r = image[:, :, 0].astype(float)
    g = image[:, :, 1].astype(float)
    b = image[:, :, 2].astype(float)
    
    eps = 1e-5  # Avoid division by zero
    
    # Compute relative color scores for satellite imagery
    # These work better than absolute HSV thresholds
    if color_name_lower in ["blue", "cyan", "turquoise"]:
        # Blueness = (B - R) / (B + R) - higher means more blue
        # For swimming pools in satellite imagery, water typically has B > R
        blueness = (b - r) / (b + r + eps)
        
        # Get blueness values only in the mask
        masked_blueness = blueness[mask > 0]
        
        if len(masked_blueness) == 0:
            return np.zeros_like(mask)
        
        # Use adaptive thresholding: B should be ~9% higher than R for "blue" water
        # This was calibrated based on satellite imagery analysis
        threshold = 0.09
        
        # Create color mask
        color_region = (blueness > threshold) & (mask > 0)
        
        print(f"    Blue extraction: {color_region.sum()} pixels with blueness > {threshold}")
        
        return color_region.astype(np.uint8)
    
    elif color_name_lower in ["green"]:
        # Greenness: G is dominant channel
        greenness = (g - np.maximum(r, b)) / (g + np.maximum(r, b) + eps)
        threshold = 0.05
        color_region = (greenness > threshold) & (mask > 0)
        return color_region.astype(np.uint8)
    
    elif color_name_lower in ["red", "orange", "brown"]:
        # Redness: R is dominant
        redness = (r - np.maximum(g, b)) / (r + np.maximum(g, b) + eps)
        threshold = 0.05
        color_region = (redness > threshold) & (mask > 0)
        return color_region.astype(np.uint8)
    
    elif color_name_lower in ["yellow"]:
        # Yellow: R and G are high, B is low
        yellowness = ((r + g) / 2 - b) / ((r + g) / 2 + b + eps)
        threshold = 0.1
        color_region = (yellowness > threshold) & (mask > 0)
        return color_region.astype(np.uint8)
    
    elif color_name_lower in ["white"]:
        # White: High brightness, low saturation
        brightness = (r + g + b) / 3
        max_rgb = np.maximum(r, np.maximum(g, b))
        min_rgb = np.minimum(r, np.minimum(g, b))
        saturation = (max_rgb - min_rgb) / (max_rgb + eps)
        
        # High brightness (top 30% of mask) and low saturation
        masked_brightness = brightness[mask > 0]
        bright_thresh = np.percentile(masked_brightness, 70) if len(masked_brightness) > 0 else 200
        
        color_region = (brightness > bright_thresh) & (saturation < 0.3) & (mask > 0)
        return color_region.astype(np.uint8)
    
    elif color_name_lower in ["gray", "grey"]:
        # Gray: Low saturation, medium brightness
        max_rgb = np.maximum(r, np.maximum(g, b))
        min_rgb = np.minimum(r, np.minimum(g, b))
        saturation = (max_rgb - min_rgb) / (max_rgb + eps)
        
        color_region = (saturation < 0.2) & (mask > 0)
        return color_region.astype(np.uint8)
    
    elif color_name_lower in ["black"]:
        # Black: Low brightness
        brightness = (r + g + b) / 3
        masked_brightness = brightness[mask > 0]
        dark_thresh = np.percentile(masked_brightness, 30) if len(masked_brightness) > 0 else 50
        
        color_region = (brightness < dark_thresh) & (mask > 0)
        return color_region.astype(np.uint8)
    
    else:
        # Fallback to HSV-based approach for unknown colors
        print(f"  Warning: Unknown color '{color_name}', using HSV fallback")
        return _hsv_color_extraction(image, mask, color_name_lower)


def _hsv_color_extraction(image: np.ndarray, mask: np.ndarray, color_name: str) -> np.ndarray:
    """Fallback HSV-based color extraction for unknown colors."""
    color_ranges = {
        "pink": [(140, 50, 100), (170, 255, 255)],
        "purple": [(125, 50, 50), (150, 255, 255)],
    }
    
    if color_name not in color_ranges:
        return mask
    
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower, upper = color_ranges[color_name]
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)
    
    color_mask = cv2.inRange(hsv, lower, upper)
    combined = cv2.bitwise_and(color_mask, mask.astype(np.uint8) * 255)
    
    return (combined > 0).astype(np.uint8)


# =============================================================================
# Mask Computation Utilities
# =============================================================================

def decode_rle_mask(rle_str: str, H: int, W: int) -> np.ndarray:
    """Decode RLE string to binary mask."""
    if isinstance(rle_str, str):
        rle_dict = {"counts": rle_str.encode('utf-8'), "size": [H, W]}
    else:
        rle_dict = rle_str
    return mask_utils.decode(rle_dict)


def compute_mask_area_pixels(mask: np.ndarray) -> int:
    """Compute area in pixels."""
    return int(np.sum(mask > 0))


def compute_mask_area_meters(mask: np.ndarray, spatial_resolution_m: float) -> float:
    """
    Compute area in square meters.
    
    Args:
        mask: Binary mask
        spatial_resolution_m: Meters per pixel
    
    Returns:
        Area in square meters
    """
    pixel_area = compute_mask_area_pixels(mask)
    return pixel_area * (spatial_resolution_m ** 2)


def compute_mask_length_meters(mask: np.ndarray, spatial_resolution_m: float) -> float:
    """
    Compute length (major axis) in meters using oriented bounding box.
    
    Args:
        mask: Binary mask
        spatial_resolution_m: Meters per pixel
    
    Returns:
        Length in meters
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return 0.0
    
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    (_, _), (w, h), _ = rect
    
    # Length is the longer dimension
    length_pixels = max(w, h)
    return length_pixels * spatial_resolution_m


def compute_mask_width_meters(mask: np.ndarray, spatial_resolution_m: float) -> float:
    """
    Compute width (minor axis) in meters using oriented bounding box.
    
    Args:
        mask: Binary mask
        spatial_resolution_m: Meters per pixel
    
    Returns:
        Width in meters
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return 0.0
    
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    (_, _), (w, h), _ = rect
    
    # Width is the shorter dimension
    width_pixels = min(w, h)
    return width_pixels * spatial_resolution_m


def compute_mask_perimeter_meters(mask: np.ndarray, spatial_resolution_m: float) -> float:
    """
    Compute perimeter in meters.
    
    Args:
        mask: Binary mask
        spatial_resolution_m: Meters per pixel
    
    Returns:
        Perimeter in meters
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return 0.0
    
    largest = max(contours, key=cv2.contourArea)
    perimeter_pixels = cv2.arcLength(largest, closed=True)
    return perimeter_pixels * spatial_resolution_m


def compute_obbs_from_masks(masks: List[str], H: int, W: int) -> List[Dict]:
    """
    Convert RLE masks to Oriented Bounding Boxes (normalized).
    
    Output format: [cx, cy, w, h, angle]
    """
    obbs = []
    
    for idx, rle_str in enumerate(masks):
        try:
            mask = decode_rle_mask(rle_str, H, W)
            
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
            
            # Normalize angle to [-90, 0)
            if angle < -45:
                angle += 90
                w, h = h, w
            elif angle > 45:
                angle -= 90
                w, h = h, w
            
            obbs.append({
                "object_id": idx + 1,
                "obbox": [
                    round(cx / W, 4),
                    round(cy / H, 4),
                    round(w / W, 4),
                    round(h / H, 4),
                    round(angle, 2)
                ],
                "raw_rect": rect,
                "mask_pixels": int(np.sum(mask)),
                "mask": mask  # Keep mask for numeric computation
            })
        except Exception as e:
            print(f"  Warning: Error processing mask {idx}: {e}")
            continue
    
    return obbs


# =============================================================================
# Object Synonym Mapping for SAM3
# =============================================================================

# SAM3 may not recognize all object names, so we provide synonyms
# Based on testing: "circular structure", "round structure", "white circle" work well for tanks
SAM3_OBJECT_SYNONYMS = {
    "storage tank": ["circular structure", "round structure", "white circle", "tank", "oil tank"],
    "oil tank": ["circular structure", "round structure", "tank", "storage tank"],
    "fuel tank": ["circular structure", "round structure", "tank", "storage tank"],
    "tank": ["circular structure", "round structure", "storage tank"],
    "swimming pool": ["pool", "water pool", "blue rectangle"],
    "ground track field": ["track field", "running track", "track", "athletic field", "oval track"],
    "baseball diamond": ["baseball field", "baseball", "diamond"],
    "basketball court": ["basketball", "court"],
    "tennis court": ["tennis", "court"],
    "soccer field": ["soccer", "football field", "field"],
    "airplane": ["plane", "aircraft", "jet"],
    "aircraft": ["plane", "airplane", "jet"],
    "vehicle": ["car", "automobile"],
    "runway": ["airstrip", "landing strip"],
}


def get_sam3_prompts(target_object: str) -> List[str]:
    """
    Get a list of prompts to try for SAM3 detection.
    
    Returns the original prompt plus any synonyms if available.
    """
    target_lower = target_object.lower().strip()
    prompts = [target_lower]
    
    if target_lower in SAM3_OBJECT_SYNONYMS:
        prompts.extend(SAM3_OBJECT_SYNONYMS[target_lower])
    
    return prompts


# =============================================================================
# Numeric VQA Pipeline
# =============================================================================

class NumericVQAPipeline:
    """Numeric VQA inference pipeline using SAM3 agent."""
    
    def __init__(
        self,
        output_dir: str,
        vis_dir: str,
        confidence_threshold: float = 0.1,
        quiet: bool = False,
    ):
        self.output_dir = output_dir
        self.vis_dir = vis_dir
        self.confidence_threshold = confidence_threshold
        self.quiet = quiet
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        
        self.sam3_model = None
        self.sam3_processor = None
        self.mllm = None
        self.sampling_params = None
        self.query_parser = None
    
    def _log(self, msg: str):
        """Print only if not in quiet mode."""
        if not self.quiet:
            print(msg)
    
    def initialize(self):
        """Initialize SAM3 and MLLM models."""
        self._log("\n" + "-"*60)
        self._log("Initializing Numeric VQA Pipeline")
        self._log("-"*60)
        
        self._log("\n[1/3] Loading SAM3 model...")
        self._init_sam3()
        
        self._log("\n[2/3] Loading MLLM...")
        self._init_mllm()
        
        self._log("\n[3/3] Initializing query parser...")
        self.query_parser = NumericQueryParser(mllm_client=None)  # Use regex for simplicity
        self._log("  ✓ Query parser ready (regex mode)")
        
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
        """Initialize MLLM using vLLM server client for agent reasoning."""
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
                    self._log("  ✗ Failed to start vLLM server")
                    self.mllm = None
                    
        except ImportError:
            self._log("  ⚠ vllm_server module not found, trying direct vLLM...")
            # Fallback to direct vLLM (legacy behavior)
            try:
                from vllm import LLM, SamplingParams
                
                self.mllm = LLM(
                    model="Qwen/Qwen2.5-VL-7B-Instruct",
                    tensor_parallel_size=2,
                    trust_remote_code=True,
                    dtype="bfloat16",
                    max_model_len=32768,
                    gpu_memory_utilization=0.8,
                    disable_log_stats=True,
                    limit_mm_per_prompt={"image": 5},
                )
                self.sampling_params = SamplingParams(
                    max_tokens=8192,
                    temperature=0.0,
                )
                self._log("  ✓ MLLM loaded directly (Qwen2.5-VL-7B)")
                
            except Exception as e:
                self._log(f"  ✗ MLLM loading failed: {e}")
                self.mllm = None
        except Exception as e:
            self._log(f"  ✗ MLLM initialization failed: {e}")
            self.mllm = None
    
    def run_inference(
        self,
        image_path: str,
        query: str,
        spatial_resolution_m: float,
        sample_id: str,
        gt_answer: Optional[float] = None,
    ) -> Dict:
        """
        Run numeric VQA inference.
        
        Args:
            image_path: Path to the image
            query: The numeric query
            spatial_resolution_m: Spatial resolution in meters/pixel
            sample_id: Sample identifier
            gt_answer: Ground truth answer (optional)
        
        Returns:
            Dict with query results
        """
        start_time = time.time()
        
        # Load image
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        H, W = img_np.shape[:2]
        
        # Create visualization folder
        image_name = os.path.basename(image_path).replace('.', '_')
        image_vis_dir = os.path.join(self.vis_dir, sample_id)
        os.makedirs(image_vis_dir, exist_ok=True)
        
        # Save input
        img.save(os.path.join(image_vis_dir, "input.png"))
        
        # Parse query
        parsed_query = self.query_parser.parse(query)
        self._log(f"  Query type: {parsed_query.query_type.value}")
        self._log(f"  Target: {parsed_query.target_object}")
        if parsed_query.target_attribute:
            self._log(f"  Attribute: {parsed_query.target_attribute}")
        
        # Detect objects using SAM3 with synonym fallback
        from sam3.agent.client_sam3 import call_sam_service
        
        sam_out_folder = os.path.join(image_vis_dir, "sam3_output")
        os.makedirs(sam_out_folder, exist_ok=True)
        
        # Get list of prompts to try (primary + synonyms)
        sam_prompts = get_sam3_prompts(parsed_query.target_object)
        
        masks = []
        scores = []
        successful_prompt = None
        
        for sam_prompt in sam_prompts:
            self._log(f"  Trying SAM3 prompt: '{sam_prompt}'")
            
            json_path = call_sam_service(
                sam3_processor=self.sam3_processor,
                image_path=image_path,
                text_prompt=sam_prompt,
                output_folder_path=sam_out_folder
            )
            
            # Load SAM3 results
            if json_path and os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    sam_outputs = json.load(f)
                masks = sam_outputs.get("pred_masks", [])
                scores = sam_outputs.get("pred_scores", [])
            
            if masks:
                successful_prompt = sam_prompt
                self._log(f"  ✓ SAM3 detected {len(masks)} masks with prompt '{sam_prompt}'")
                break
            else:
                self._log(f"    No detections with '{sam_prompt}', trying next...")
        
        if not masks:
            self._log(f"  ✗ SAM3 detected 0 masks with all prompts: {sam_prompts}")
        
        # Compute OBBs and keep masks
        obb_dicts = compute_obbs_from_masks(masks, H, W)
        
        # Filter by confidence
        filtered_obbs = [
            obb for i, obb in enumerate(obb_dicts) 
            if i < len(scores) and scores[i] >= self.confidence_threshold
        ]
        
        if not filtered_obbs and obb_dicts:
            # If nothing passes threshold, use top result
            filtered_obbs = [obb_dicts[0]]
        
        self._log(f"  After filtering: {len(filtered_obbs)} objects")
        
        # Compute numeric answer based on query type
        computed_answer = self._compute_answer(
            parsed_query=parsed_query,
            obbs=filtered_obbs,
            masks=masks,
            image=img_np,
            H=H, W=W,
            spatial_resolution_m=spatial_resolution_m
        )
        
        # Calculate error if GT available
        error = None
        if gt_answer is not None:
            error = abs(computed_answer - gt_answer)
            rel_error = error / gt_answer if gt_answer != 0 else float('inf')
            self._log(f"  Computed: {computed_answer:.2f}, GT: {gt_answer:.2f}, Error: {error:.2f} ({rel_error*100:.1f}%)")
        else:
            self._log(f"  Computed answer: {computed_answer:.2f}")
        
        # Build result
        result = {
            "sample_id": sample_id,
            "image_path": image_path,
            "query": query,
            "parsed_query": {
                "type": parsed_query.query_type.value,
                "target_object": parsed_query.target_object,
                "target_attribute": parsed_query.target_attribute,
                "unit": parsed_query.unit,
            },
            "spatial_resolution_m": spatial_resolution_m,
            "num_masks_detected": len(masks),
            "num_masks_filtered": len(filtered_obbs),
            "computed_answer": round(computed_answer, 2),
            "ground_truth": gt_answer,
            "absolute_error": round(error, 2) if error is not None else None,
            "inference_time_s": round(time.time() - start_time, 2),
            "obbs": [obb["obbox"] for obb in filtered_obbs],
        }
        
        # Save visualization
        self._save_visualization(img_np, filtered_obbs, image_vis_dir, parsed_query)
        
        return result
    
    def _compute_answer(
        self,
        parsed_query: ParsedNumericQuery,
        obbs: List[Dict],
        masks: List[str],
        image: np.ndarray,
        H: int, W: int,
        spatial_resolution_m: float
    ) -> float:
        """
        Compute the numeric answer based on query type.
        """
        query_type = parsed_query.query_type
        
        if query_type == NumericQueryType.COUNT:
            return float(len(obbs))
        
        if not obbs:
            return 0.0
        
        # For area/length/width queries, we need to select the right object
        selected_mask = self._select_target_mask(
            obbs, image, H, W, parsed_query
        )
        
        if selected_mask is None:
            return 0.0
        
        # Handle color region extraction
        if parsed_query.target_attribute:
            # Check for color in attribute
            colors = ["blue", "red", "green", "yellow", "white", "gray", "grey", 
                      "orange", "brown", "cyan", "turquoise", "black", "pink", "purple"]
            for color in colors:
                if color in parsed_query.target_attribute.lower():
                    self._log(f"  Extracting {color} region from mask")
                    selected_mask = extract_color_region(image, selected_mask, color)
                    break
        
        # Compute based on query type
        if query_type == NumericQueryType.AREA:
            return compute_mask_area_meters(selected_mask, spatial_resolution_m)
        
        elif query_type == NumericQueryType.LENGTH:
            return compute_mask_length_meters(selected_mask, spatial_resolution_m)
        
        elif query_type == NumericQueryType.WIDTH:
            return compute_mask_width_meters(selected_mask, spatial_resolution_m)
        
        elif query_type == NumericQueryType.PERIMETER:
            return compute_mask_perimeter_meters(selected_mask, spatial_resolution_m)
        
        else:
            self._log(f"  Warning: Unsupported query type {query_type}")
            return 0.0
    
    def _select_target_mask(
        self,
        obbs: List[Dict],
        image: np.ndarray,
        H: int, W: int,
        parsed_query: ParsedNumericQuery
    ) -> Optional[np.ndarray]:
        """
        Select the target mask based on query attributes.
        
        For "larger swimming pool", select the mask with largest area.
        For single result, return that mask.
        """
        if not obbs:
            return None
        
        # If only one result, use it
        if len(obbs) == 1:
            return obbs[0].get("mask")
        
        # Check for size modifiers in attribute
        attribute = (parsed_query.target_attribute or "").lower()
        
        if "larger" in attribute or "largest" in attribute or "bigger" in attribute or "biggest" in attribute:
            # Select the mask with the most pixels
            sorted_obbs = sorted(obbs, key=lambda x: x.get("mask_pixels", 0), reverse=True)
            self._log(f"  Selecting largest mask ({sorted_obbs[0].get('mask_pixels', 0)} pixels)")
            return sorted_obbs[0].get("mask")
        
        elif "smaller" in attribute or "smallest" in attribute:
            # Select the mask with the fewest pixels
            sorted_obbs = sorted(obbs, key=lambda x: x.get("mask_pixels", 0))
            self._log(f"  Selecting smallest mask ({sorted_obbs[0].get('mask_pixels', 0)} pixels)")
            return sorted_obbs[0].get("mask")
        
        else:
            # Default to highest confidence (first in list after filtering)
            return obbs[0].get("mask")
    
    def _save_visualization(
        self,
        image: np.ndarray,
        obbs: List[Dict],
        vis_dir: str,
        parsed_query: ParsedNumericQuery
    ):
        """Save visualization of detected objects."""
        vis_img = image.copy()
        
        for i, obb in enumerate(obbs):
            # Draw OBB
            obbox = obb.get("obbox", [0.5, 0.5, 0.1, 0.1, 0])
            H, W = image.shape[:2]
            cx, cy, w, h, angle = obbox
            cx, cy, w, h = cx * W, cy * H, w * W, h * H
            
            rect = ((cx, cy), (w, h), angle)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            color = (0, 255, 0)  # Green
            cv2.drawContours(vis_img, [box], 0, color, 2)
            
            # Add label
            label = f"{i+1}: {obb.get('mask_pixels', 0)} px"
            cv2.putText(vis_img, label, (int(cx), int(cy)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save
        vis_path = os.path.join(vis_dir, "detection.png")
        Image.fromarray(vis_img).save(vis_path)


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Numeric VQA Pipeline")
    parser.add_argument("--data_dir", type=str, 
                        default="./sample_dataset_inter_iit_v1_3",
                        help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str,
                        default="./outputs/numeric",
                        help="Output directory")
    parser.add_argument("--confidence_threshold", type=float, default=0.1,
                        help="SAM3 confidence threshold")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    args = parser.parse_args()
    
    # Setup output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    vis_dir = os.path.join(run_output_dir, "vis")
    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Initialize pipeline
    pipeline = NumericVQAPipeline(
        output_dir=run_output_dir,
        vis_dir=vis_dir,
        confidence_threshold=args.confidence_threshold,
        quiet=args.quiet
    )
    pipeline.initialize()
    
    # Find sample queries
    sample_files = []
    data_dir = args.data_dir
    
    for filename in os.listdir(data_dir):
        if filename.endswith("_query.json"):
            query_path = os.path.join(data_dir, filename)
            response_path = os.path.join(data_dir, filename.replace("_query.json", "_response.json"))
            
            sample_files.append({
                "query_file": query_path,
                "response_file": response_path if os.path.exists(response_path) else None,
            })
    
    print(f"\nFound {len(sample_files)} sample files")
    
    # Process samples
    results = []
    
    for sample_info in sample_files:
        with open(sample_info["query_file"], 'r') as f:
            query_data = json.load(f)
        
        # Get image info
        image_info = query_data.get("input_image", {})
        image_id = image_info.get("image_id", "")
        image_path = os.path.join(data_dir, image_id)
        
        if not os.path.exists(image_path):
            print(f"  Image not found: {image_path}")
            continue
        
        spatial_resolution_m = image_info.get("metadata", {}).get("spatial_resolution_m", 1.0)
        
        # Get numeric query
        queries = query_data.get("queries", {})
        attr_queries = queries.get("attribute_query", {})
        numeric_query = attr_queries.get("numeric", {})
        
        if not numeric_query:
            print(f"  No numeric query in {sample_info['query_file']}")
            continue
        
        query_text = numeric_query.get("instruction", "")
        
        # Get ground truth from response file
        gt_answer = None
        if sample_info["response_file"]:
            with open(sample_info["response_file"], 'r') as f:
                response_data = json.load(f)
            gt_answer = response_data.get("queries", {}).get("attribute_query", {}).get("numeric", {}).get("response")
        
        sample_id = image_id.replace('.', '_')
        
        print(f"\n{'='*60}")
        print(f"Processing: {sample_id}")
        print(f"Query: {query_text}")
        print(f"Spatial Resolution: {spatial_resolution_m} m/px")
        if gt_answer is not None:
            print(f"Ground Truth: {gt_answer}")
        print("-"*60)
        
        # Run inference
        result = pipeline.run_inference(
            image_path=image_path,
            query=query_text,
            spatial_resolution_m=spatial_resolution_m,
            sample_id=sample_id,
            gt_answer=gt_answer
        )
        
        results.append(result)
    
    # Save results
    results_file = os.path.join(run_output_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("NUMERIC VQA COMPLETE")
    print("="*60)
    
    total_samples = len(results)
    samples_with_gt = [r for r in results if r.get("ground_truth") is not None]
    
    if samples_with_gt:
        total_error = sum(r.get("absolute_error", 0) for r in samples_with_gt)
        avg_error = total_error / len(samples_with_gt)
        print(f"Samples with GT: {len(samples_with_gt)}")
        print(f"Average Absolute Error: {avg_error:.2f}")
        
        # Per-sample breakdown
        for r in samples_with_gt:
            gt = r.get("ground_truth", 0)
            computed = r.get("computed_answer", 0)
            err = r.get("absolute_error", 0)
            rel_err = err / gt * 100 if gt != 0 else float('inf')
            print(f"  {r['sample_id']}: GT={gt}, Computed={computed}, Error={err:.2f} ({rel_err:.1f}%)")
    
    print(f"\nResults saved to: {results_file}")
    print(f"Visualizations: {vis_dir}")


if __name__ == "__main__":
    main()
