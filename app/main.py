"""
Unified Inference Engine - Main Entry Point
Handles all VQA tasks: Caption, Binary, Numeric, Semantic, and Grounding
Processes JSON input and returns formatted JSON output matching the Inter-IIT format
"""

import os
import sys
import json
import argparse
import logging
import gc
import tempfile
import torch
import requests
from PIL import Image
from io import BytesIO

# Import your specific modules
from caption import load_caption_model, generate_caption
from binary import load_binary_model, answer_binary_question
from numeric import load_numeric_model, answer_numeric_question
from semantic import load_semantic_model, answer_semantic_question

# Lazy import grounding to avoid loading heavy dependencies when not needed
grounding_module = None

def get_grounding_module():
    """Lazy load grounding module."""
    global grounding_module
    if grounding_module is None:
        from grounding import load_grounding_model, answer_grounding_query
        grounding_module = {
            'load': load_grounding_model,
            'answer': answer_grounding_query
        }
    return grounding_module

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global cache
MODEL_CACHE = {}

# Set to True ONLY if you have limited VRAM and crash when loading all 4 models
LOW_VRAM_MODE = False 

def clear_gpu_memory():
    """Helper to clear CUDA cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def load_image(image_source: str) -> Image.Image:
    """Load image from URL or local path"""
    try:
        if image_source.startswith("http"):
            # logger.info(f"Downloading image from URL: {image_source}")
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(image_source, headers=headers, stream=True)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            # logger.info(f"Loading image from local path: {image_source}")
            if not os.path.exists(image_source):
                raise FileNotFoundError(f"Image not found: {image_source}")
            image = Image.open(image_source).convert("RGB")
        return image
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise

def get_model_instance(task_type):
    """
    Loads model for specific task. 
    """
    global MODEL_CACHE
    
    # Return cached model if available
    if task_type in MODEL_CACHE:
        return MODEL_CACHE[task_type]

    # If Low VRAM mode is ON, clear existing models before loading new one
    if LOW_VRAM_MODE and MODEL_CACHE:
        logger.info("Low VRAM Mode: Unloading existing models...")
        keys = list(MODEL_CACHE.keys())
        for k in keys:
            del MODEL_CACHE[k]
        MODEL_CACHE = {}
        clear_gpu_memory()

    logger.info(f"Loading model for task: {task_type}")
    
    try:
        if task_type == 'caption':
            model, tokenizer = load_caption_model(load_in_4bit=True)
        elif task_type == 'binary':
            model, tokenizer = load_binary_model(load_in_4bit=True)
        elif task_type == 'numeric':
            model, tokenizer = load_numeric_model(load_in_4bit=True)
        elif task_type == 'semantic':
            model, tokenizer = load_semantic_model(load_in_4bit=True)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
            
        MODEL_CACHE[task_type] = (model, tokenizer)
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Critical error loading {task_type} model: {e}")
        raise

def preload_all_models(include_grounding: bool = False):
    """
    Explicitly loads all models into VRAM at startup.
    This prevents loading lag during the inference requests.
    
    Args:
        include_grounding: Whether to also preload grounding model (heavy, uses vLLM)
    """
    logger.info("--- PRELOADING ALL MODELS (High VRAM Usage) ---")
    tasks = ['caption', 'binary', 'numeric', 'semantic']
    
    for task in tasks:
        try:
            get_model_instance(task)
            logger.info(f"✔ {task.capitalize()} model ready.")
        except Exception as e:
            logger.error(f"✘ Failed to preload {task}: {e}")
            sys.exit(1) # Exit if initialization fails
    
    # Optionally preload grounding (uses vLLM server, heavy)
    if include_grounding:
        try:
            grounding = get_grounding_module()
            grounding['load']()
            logger.info("✔ Grounding model ready.")
        except Exception as e:
            logger.warning(f"⚠ Grounding model not loaded: {e}")
            
    logger.info("--- ALL MODELS LOADED ---")

def run_inference(request_data):
    """
    Main Orchestrator Function
    Returns JSON in the exact Inter-IIT format with responses added to queries.
    """
    
    # 1. Parse Image Source
    input_info = request_data.get("input_image", {})
    img_src = input_info.get("image_url") or input_info.get("image_path")
    spatial_resolution_m = input_info.get("metadata", {}).get("spatial_resolution_m", 1.0)
    
    if not img_src:
        raise ValueError("Input JSON must contain 'image_url' or 'image_path'")

    pil_image = load_image(img_src)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Process Queries
    queries = request_data.get("queries", {})
    
    # --- A. CAPTION QUERY ---
    if "caption_query" in queries:
        q_obj = queries["caption_query"]
        instruction = q_obj.get("instruction", "")
        model, tokenizer = get_model_instance("caption") 
        try:
            response = generate_caption(model, tokenizer, pil_image, instruction, device=device)
            q_obj["response"] = response
        except Exception as e:
            logger.error(f"Caption failed: {e}")
            q_obj["response"] = "Error generating caption"

    # --- B. GROUNDING QUERY ---
    if "grounding_query" in queries:
        q_obj = queries["grounding_query"]
        instruction = q_obj.get("instruction", "")
        try:
            # Grounding requires the image path, not PIL image
            if img_src.startswith("http"):
                # Save to temp file
                temp_path = os.path.join(tempfile.gettempdir(), f"grounding_temp_{os.getpid()}.jpg")
                pil_image.save(temp_path, "JPEG")
                grounding_img_path = temp_path
            else:
                grounding_img_path = img_src
            
            grounding = get_grounding_module()
            result = grounding['answer'](grounding_img_path, instruction, spatial_resolution_m)
            
            # Format response as list of objects with object-id and obbox
            # obbox format: [center_x, center_y, width, height, angle]
            if result.get("success", False):
                boxes = result.get("bounding_boxes", [])
                formatted_boxes = []
                for i, box in enumerate(boxes, start=1):
                    formatted_boxes.append({
                        "object-id": str(i),
                        "obbox": box  # Should be [cx, cy, w, h, angle] normalized
                    })
                q_obj["response"] = formatted_boxes
            else:
                q_obj["response"] = []
                
            # Cleanup temp file if created
            if img_src.startswith("http"):
                try:
                    os.remove(temp_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Grounding failed: {e}")
            q_obj["response"] = []

    # --- C. ATTRIBUTE QUERIES ---
    if "attribute_query" in queries:
        attrs = queries["attribute_query"]
        
        # 1. Binary - returns "Yes" or "No"
        if "binary" in attrs:
            q_obj = attrs["binary"]
            instruction = q_obj.get("instruction", "")
            model, tokenizer = get_model_instance("binary")
            try:
                response = answer_binary_question(model, tokenizer, pil_image, instruction, device=device)
                # Normalize to "Yes" or "No"
                if isinstance(response, str):
                    response_lower = response.lower().strip()
                    if "yes" in response_lower:
                        q_obj["response"] = "Yes"
                    elif "no" in response_lower:
                        q_obj["response"] = "No"
                    else:
                        q_obj["response"] = response.strip()
                else:
                    q_obj["response"] = str(response)
            except Exception as e:
                logger.error(f"Binary failed: {e}")
                q_obj["response"] = "Error"

        # 2. Numeric - returns a float
        if "numeric" in attrs:
            q_obj = attrs["numeric"]
            instruction = q_obj.get("instruction", "")
            model, tokenizer = get_model_instance("numeric")
            try:
                response = answer_numeric_question(
                    model, tokenizer, pil_image, instruction, 
                    device=device, spatial_resolution_m=spatial_resolution_m
                )
                # Ensure it's a float
                if isinstance(response, (int, float)):
                    q_obj["response"] = float(response)
                elif isinstance(response, str):
                    # Try to extract number from string
                    import re
                    numbers = re.findall(r'[-+]?\d*\.?\d+', response)
                    if numbers:
                        q_obj["response"] = float(numbers[0])
                    else:
                        q_obj["response"] = 0.0
                else:
                    q_obj["response"] = 0.0
            except Exception as e:
                logger.error(f"Numeric failed: {e}")
                q_obj["response"] = 0.0

        # 3. Semantic - returns a text string
        if "semantic" in attrs:
            q_obj = attrs["semantic"]
            instruction = q_obj.get("instruction", "")
            model, tokenizer = get_model_instance("semantic")
            try:
                response = answer_semantic_question(model, tokenizer, pil_image, instruction, device=device)
                q_obj["response"] = response.strip() if isinstance(response, str) else str(response)
            except Exception as e:
                logger.error(f"Semantic failed: {e}")
                q_obj["response"] = "Error"

    # Return the complete request with responses added
    return request_data

def main():
    parser = argparse.ArgumentParser(description="VRSBench Unified Inference Engine")
    parser.add_argument("--input", type=str, help="Path to input JSON file")
    parser.add_argument("--json", type=str, help="Inline JSON string")
    parser.add_argument("--output", type=str, default="output.json", help="Path to save output JSON")
    parser.add_argument("--low-vram", action="store_true", help="Enable Low VRAM mode (unload models after use)")
    
    args = parser.parse_args()

    # Set global VRAM flag
    global LOW_VRAM_MODE
    if args.low_vram:
        LOW_VRAM_MODE = True

    # ---------------------------------------------------------
    # STEP 1: LOAD MODELS ONCE (Only if NOT in Low VRAM Mode)
    # ---------------------------------------------------------
    if not LOW_VRAM_MODE:
        preload_all_models()

    # ---------------------------------------------------------
    # STEP 2: LOAD INPUT DATA
    # ---------------------------------------------------------
    try:
        if args.input:
            with open(args.input, 'r') as f:
                data = json.load(f)
        elif args.json:
            data = json.loads(args.json)
        else:
            logger.warning("No input provided.")
            return
            
    except Exception as e:
        logger.error(f"Error parsing input JSON: {e}")
        return

    # ---------------------------------------------------------
    # STEP 3: RUN INFERENCE
    # ---------------------------------------------------------
    try:
        # Since models are preloaded, this will be fast
        result_data = run_inference(data)
    except Exception as e:
        logger.error(f"Fatal error during inference: {e}")
        return

    # ---------------------------------------------------------
    # STEP 4: SAVE OUTPUT
    # ---------------------------------------------------------
    try:
        with open(args.output, 'w') as f:
            json.dump(result_data, f, indent=4)
        print(f"\nSuccess! Output saved to: {args.output}")
        print(json.dumps(result_data, indent=4))
    except Exception as e:
        logger.error(f"Error saving output: {e}")

if __name__ == "__main__":
    main()