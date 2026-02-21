"""
Configuration Module for Satellite VQA System
Centralizes all path and environment configurations for RunPod deployment
"""

import os

# =============================================================================
# Base Model Configuration
# =============================================================================
# 4-bit quantized model for LoRA adapters (caption, binary, semantic, numeric)
# Use local path if available, otherwise HuggingFace ID
_LOCAL_4BIT_PATH = "/workspace/models/qwen2.5-vl-7b-instruct-bnb-4bit"
BASE_MODEL_ID = _LOCAL_4BIT_PATH if os.path.exists(_LOCAL_4BIT_PATH) else "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"

# FP16 model for grounding and complex reasoning (via vLLM)
# Use HuggingFace ID - vLLM will cache it locally in HF_CACHE_DIR
# Note: Local path /workspace/models/qwen2.5-vl-7b-instruct-fp16 has config issues with vLLM 0.12
QWEN_FP16_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# vLLM cache directory for downloaded models
VLLM_DOWNLOAD_DIR = os.environ.get("VLLM_DOWNLOAD_DIR", "/workspace/models/hf_cache")

# SAM3 model from HuggingFace
SAM3_MODEL_ID = "facebook/sam3"

# =============================================================================
# Path Configuration
# =============================================================================
# Models directory - check environment variable first, then common paths
MODELS_DIR = os.environ.get("MODELS_DIR", None)

if MODELS_DIR is None or not os.path.exists(MODELS_DIR):
    # Check common paths in order of priority (volume-based deployment)
    _models_candidates = [
        "/workspace/models",  # RunPod persistent volume (preferred)
        "/app/models",  # Docker container path (fallback)
        os.path.join(os.path.dirname(__file__), "models"),  # Local development
        ".",  # Current directory as last resort
    ]
    for _path in _models_candidates:
        if os.path.exists(_path):
            MODELS_DIR = _path
            break
    else:
        MODELS_DIR = _models_candidates[-1]  # Default to current dir

# =============================================================================
# Adapter Paths
# =============================================================================
ADAPTER_PATHS = {
    "caption": os.path.join(MODELS_DIR, "captioning_sft_0.6507"),
    "binary": os.path.join(MODELS_DIR, "sft_binary_0.91-score_qwen2.5_4bit"),
    "semantic": os.path.join(MODELS_DIR, "sft_semantic_0.85_qwen2.57b_4bit"),
    "numeric": os.path.join(MODELS_DIR, "numeric_sft_0.67_qwen2.54bit"),
}

# Legacy adapter names (for backward compatibility)
LEGACY_ADAPTER_NAMES = {
    "caption": ["captioning_sft_0.6507", "merged_qwen_model"],
    "binary": ["sft_binary_0.91-score_qwen2.5_4bit"],
    "semantic": ["sft_semantic_0.85_qwen2.57b_4bit", "outputs_vqa_short"],
    "numeric": ["numeric_sft_0.67_qwen2.54bit"],
}

# =============================================================================
# SAM3 Configuration
# =============================================================================
# SAM3 code path (local repo)
SAM3_PATH = os.environ.get("SAM3_PATH", None)
if SAM3_PATH is None or not os.path.exists(SAM3_PATH):
    _sam3_code_candidates = [
        "/app/sam3",  # Docker container path (code is in image)
        "/workspace/sam3",  # RunPod persistent volume
        os.path.join(os.path.dirname(__file__), "sam3"),  # Local development
    ]
    for _path in _sam3_code_candidates:
        if os.path.exists(_path):
            SAM3_PATH = _path
            break
    else:
        SAM3_PATH = _sam3_code_candidates[-1]  # Default to local

# SAM3 checkpoint path (weights file)
# Set this to use local weights instead of downloading from HuggingFace
SAM3_CHECKPOINT_PATH = os.environ.get("SAM3_CHECKPOINT_PATH", None)
if SAM3_CHECKPOINT_PATH is None:
    # Check if weights exist in models directory (prioritize .pt files)
    _sam3_ckpt_candidates = [
        "/workspace/models/sam3/sam3.pt",  # Primary location (full checkpoint)
        os.path.join(MODELS_DIR, "sam3", "sam3.pt"),
        os.path.join(MODELS_DIR, "sam3.pt"),
        os.path.join(os.path.dirname(__file__), "models", "sam3", "sam3.pt"),
    ]
    for _ckpt in _sam3_ckpt_candidates:
        if os.path.exists(_ckpt):
            SAM3_CHECKPOINT_PATH = _ckpt
            break

# =============================================================================
# Cache Directories
# =============================================================================
HF_HOME = os.environ.get("HF_HOME", "/workspace/cache/huggingface")
TORCH_HOME = os.environ.get("TORCH_HOME", "/workspace/cache/torch")
TRANSFORMERS_CACHE = os.environ.get("TRANSFORMERS_CACHE", HF_HOME)

# Set environment variables for HuggingFace
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE
os.environ["TORCH_HOME"] = TORCH_HOME

# =============================================================================
# GPU Configuration
# =============================================================================
# Default to using both GPUs (for 2x A40 setup)
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# Enable TF32 for faster inference on Ampere GPUs
ENABLE_TF32 = os.environ.get("ENABLE_TF32", "1") == "1"

# =============================================================================
# vLLM Configuration (for FP16 Qwen model)
# =============================================================================
VLLM_GPU_ID = int(os.environ.get("VLLM_GPU_ID", "1"))  # GPU 1 for vLLM
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8001"))
VLLM_MEMORY_UTILIZATION = float(os.environ.get("VLLM_MEMORY_UTIL", "0.45"))

# =============================================================================
# API Configuration
# =============================================================================
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))
API_WORKERS = int(os.environ.get("API_WORKERS", "1"))

# =============================================================================
# Inference Configuration
# =============================================================================
# Load all models into memory at startup (requires sufficient VRAM)
PRELOAD_ALL_MODELS = os.environ.get("PRELOAD_ALL_MODELS", "1") == "1"

# Low VRAM mode: unload models after each inference
LOW_VRAM_MODE = os.environ.get("LOW_VRAM_MODE", "0") == "1"

# Model quantization settings
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "1") == "1"
LOAD_IN_8BIT = os.environ.get("LOAD_IN_8BIT", "0") == "1"

# =============================================================================
# Helper Functions
# =============================================================================

def get_adapter_path(task_type: str) -> str:
    """
    Get the adapter path for a given task type.
    Checks multiple possible locations for backward compatibility.
    """
    # First try the configured path
    primary_path = ADAPTER_PATHS.get(task_type)
    if primary_path and os.path.exists(primary_path):
        return primary_path
    
    # Try legacy names
    legacy_names = LEGACY_ADAPTER_NAMES.get(task_type, [])
    for name in legacy_names:
        # Check in MODELS_DIR
        path = os.path.join(MODELS_DIR, name)
        if os.path.exists(path):
            return path
        # Check in current directory
        if os.path.exists(name):
            return name
    
    # Return primary path even if not found (will use base model)
    return primary_path or BASE_MODEL_ID


def validate_paths() -> dict:
    """
    Validate all configured paths and return status.
    """
    status = {
        "models_dir": os.path.exists(MODELS_DIR),
        "sam3_path": os.path.exists(SAM3_PATH),
        "adapters": {}
    }
    
    for task, path in ADAPTER_PATHS.items():
        status["adapters"][task] = {
            "path": path,
            "exists": os.path.exists(path),
            "has_config": os.path.exists(os.path.join(path, "adapter_config.json")) if os.path.exists(path) else False
        }
    
    return status


def print_config():
    """Print current configuration for debugging."""
    print("=" * 60)
    print("Satellite VQA Configuration")
    print("=" * 60)
    print(f"MODELS_DIR: {MODELS_DIR}")
    print(f"SAM3_PATH: {SAM3_PATH}")
    print(f"HF_HOME: {HF_HOME}")
    print(f"CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}")
    print(f"\nModel IDs:")
    print(f"  BASE_MODEL_ID (4-bit): {BASE_MODEL_ID}")
    print(f"  QWEN_FP16_MODEL_ID: {QWEN_FP16_MODEL_ID}")
    print(f"  SAM3_MODEL_ID: {SAM3_MODEL_ID}")
    print(f"\nvLLM Config:")
    print(f"  GPU: {VLLM_GPU_ID}")
    print(f"  Port: {VLLM_PORT}")
    print(f"  Memory Util: {VLLM_MEMORY_UTILIZATION}")
    print("=" * 60)
    
    # Validate paths
    status = validate_paths()
    print("\nPath Validation:")
    for task, info in status["adapters"].items():
        exists = "✓" if info["exists"] else "✗"
        print(f"  {task}: {exists} {info['path']}")


if __name__ == "__main__":
    print_config()
    print(f"PRELOAD_ALL_MODELS: {PRELOAD_ALL_MODELS}")
    print(f"LOW_VRAM_MODE: {LOW_VRAM_MODE}")
    print("-" * 60)
    print("Adapter Paths:")
    for task, path in ADAPTER_PATHS.items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {task}: {path} [{exists}]")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
    print("\nValidation Status:")
    import json
    print(json.dumps(validate_paths(), indent=2))
