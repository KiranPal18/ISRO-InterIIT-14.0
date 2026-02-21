#!/usr/bin/env python3
"""
Download/Setup Model Weights for RunPod Persistent Volume

This script populates the /workspace/models directory on first run.
Run this ONCE after attaching a persistent volume to your RunPod.

Models to download:
1. Qwen2.5-VL-7B-Instruct (4-bit) - for LoRA adapters (~4-5 GB)
2. Qwen2.5-VL-7B-Instruct (FP16) - for vLLM grounding (~15 GB)
3. SAM3 model weights (facebook/sam3) (~3.4 GB)

Usage on RunPod:
    cd /app
    python download_weights.py
    
Or with custom directory:
    python download_weights.py --output-dir /workspace/models
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command with progress display."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        return False
    return True


def get_dir_size_gb(path: Path) -> float:
    """Get directory size in GB."""
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    return total / (1024**3)


def download_qwen_4bit_model(output_dir: Path):
    """Download Qwen2.5-VL-7B-Instruct 4-bit model for LoRA adapters."""
    model_id = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"
    target_dir = output_dir / "qwen2.5-vl-7b-instruct-bnb-4bit"
    
    if target_dir.exists() and any(target_dir.iterdir()):
        size = get_dir_size_gb(target_dir)
        print(f"✓ Qwen 4-bit model already exists at {target_dir} ({size:.2f} GB)")
        return True
    
    print(f"\nDownloading {model_id}...")
    print("This is the 4-bit quantized model (~4-5 GB) for LoRA adapters.\n")
    
    try:
        from huggingface_hub import snapshot_download
        
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✓ Qwen 4-bit model downloaded to {target_dir}")
        return True
        
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download
        
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        return True
        
    except Exception as e:
        print(f"ERROR downloading Qwen 4-bit model: {e}")
        return False


def download_qwen_fp16_model(output_dir: Path):
    """Download Qwen2.5-VL-7B-Instruct FP16 model for vLLM (grounding/numeric)."""
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    target_dir = output_dir / "qwen2.5-vl-7b-instruct-fp16"
    
    if target_dir.exists() and any(target_dir.iterdir()):
        size = get_dir_size_gb(target_dir)
        print(f"✓ Qwen FP16 model already exists at {target_dir} ({size:.2f} GB)")
        return True
    
    print(f"\nDownloading {model_id}...")
    print("This is the FP16 full model (~15 GB) for vLLM grounding/numeric.\n")
    
    try:
        from huggingface_hub import snapshot_download
        
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✓ Qwen FP16 model downloaded to {target_dir}")
        return True
        
    except Exception as e:
        print(f"ERROR downloading Qwen FP16 model: {e}")
        return False


def download_sam3_model(output_dir: Path):
    """Download SAM3 model weights from facebook/sam3."""
    model_id = "facebook/sam3"
    target_dir = output_dir / "sam3"
    
    # Check for either .pt or .safetensors
    checkpoint_pt = target_dir / "sam3.pt"
    checkpoint_safetensors = target_dir / "model.safetensors"
    
    if checkpoint_pt.exists() or checkpoint_safetensors.exists():
        size = get_dir_size_gb(target_dir)
        print(f"✓ SAM3 checkpoint already exists at {target_dir} ({size:.2f} GB)")
        return True
    
    print(f"\nDownloading {model_id}...")
    print("This is the SAM3 segmentation model (~3.4 GB).\n")
    print("NOTE: You may need to accept the license at https://huggingface.co/facebook/sam3")
    
    try:
        from huggingface_hub import snapshot_download
        
        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the entire repo (includes model.safetensors)
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
            
        print(f"✓ SAM3 model downloaded to {target_dir}")
        return True
        
    except Exception as e:
        print(f"ERROR downloading SAM3 model: {e}")
        print("You may need to:")
        print("  1. Login with: huggingface-cli login")
        print("  2. Accept the license at: https://huggingface.co/facebook/sam3")
        return False


def copy_lora_adapters(output_dir: Path, source_dir: Path = None):
    """Copy LoRA adapters from source to persistent volume."""
    adapters = [
        "captioning_sft_0.6507",
        "sft_binary_0.91-score_qwen2.5_4bit",
        "sft_semantic_0.85_qwen2.57b_4bit",
        "numeric_sft_0.67_qwen2.54bit",
    ]
    
    # Check if adapters already exist
    existing = sum(1 for a in adapters if (output_dir / a).exists())
    if existing == len(adapters):
        print(f"✓ All {len(adapters)} LoRA adapters already exist in {output_dir}")
        return True
    
    if source_dir is None:
        # Check common source locations
        candidates = [
            Path("/app/adapters"),
            Path("/app/models"),
            Path(__file__).parent / "models",
            Path(__file__).parent.parent,  # Project root
        ]
        for c in candidates:
            if c.exists() and any((c / a).exists() for a in adapters):
                source_dir = c
                break
    
    if source_dir is None:
        print("! LoRA adapters not found in image. You need to copy them manually:")
        for a in adapters:
            print(f"    {a}/")
        return False
    
    print(f"\nCopying LoRA adapters from {source_dir} to {output_dir}...")
    copied = 0
    for adapter in adapters:
        src = source_dir / adapter
        dst = output_dir / adapter
        if src.exists() and not dst.exists():
            print(f"  Copying {adapter}...")
            shutil.copytree(src, dst)
            copied += 1
        elif dst.exists():
            print(f"  ✓ {adapter} already exists")
    
    print(f"✓ Copied {copied} LoRA adapters")
    return True


def check_volume_setup(output_dir: Path) -> dict:
    """Check what's already set up in the volume."""
    status = {
        "qwen_4bit": (output_dir / "qwen2.5-vl-7b-instruct-bnb-4bit").exists(),
        "qwen_fp16": (output_dir / "qwen2.5-vl-7b-instruct-fp16").exists(),
        "sam3": (output_dir / "sam3").exists() and (
            (output_dir / "sam3" / "model.safetensors").exists() or
            (output_dir / "sam3" / "sam3.pt").exists()
        ),
        "caption_adapter": (output_dir / "captioning_sft_0.6507").exists(),
        "binary_adapter": (output_dir / "sft_binary_0.91-score_qwen2.5_4bit").exists(),
        "semantic_adapter": (output_dir / "sft_semantic_0.85_qwen2.57b_4bit").exists(),
        "numeric_adapter": (output_dir / "numeric_sft_0.67_qwen2.54bit").exists(),
    }
    return status


def main():
    parser = argparse.ArgumentParser(
        description="Download/setup model weights for RunPod persistent volume"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="/workspace/models",
        help="Directory to save weights (default: /workspace/models for RunPod)"
    )
    parser.add_argument(
        "--skip-fp16",
        action="store_true",
        help="Skip FP16 Qwen download (if not using vLLM for grounding)"
    )
    parser.add_argument(
        "--skip-sam3",
        action="store_true",
        help="Skip SAM3 download (for basic VQA only)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check what's installed, don't download"
    )
    parser.add_argument(
        "--adapters-source",
        type=str,
        default=None,
        help="Source directory for LoRA adapters (if copying from local)"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    
    print("="*60)
    print("  Satellite VQA - Persistent Volume Setup")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check current status
    status = check_volume_setup(output_dir)
    
    print("\nCurrent volume status:")
    print(f"  Qwen 4-bit:        {'✓' if status['qwen_4bit'] else '✗'}")
    print(f"  Qwen FP16:         {'✓' if status['qwen_fp16'] else '✗'}")
    print(f"  SAM3:              {'✓' if status['sam3'] else '✗'}")
    print(f"  Caption adapter:   {'✓' if status['caption_adapter'] else '✗'}")
    print(f"  Binary adapter:    {'✓' if status['binary_adapter'] else '✗'}")
    print(f"  Semantic adapter:  {'✓' if status['semantic_adapter'] else '✗'}")
    print(f"  Numeric adapter:   {'✓' if status['numeric_adapter'] else '✗'}")
    
    if args.check_only:
        total_size = get_dir_size_gb(output_dir)
        print(f"\nTotal volume usage: {total_size:.2f} GB")
        return
    
    # Calculate what needs to be downloaded
    to_download = []
    if not status['qwen_4bit']:
        to_download.append(("Qwen 4-bit", "~4-5 GB"))
    if not status['qwen_fp16'] and not args.skip_fp16:
        to_download.append(("Qwen FP16", "~15 GB"))
    if not status['sam3'] and not args.skip_sam3:
        to_download.append(("SAM3", "~3.4 GB"))
    
    if to_download:
        print(f"\nModels to download:")
        for name, size in to_download:
            print(f"  - {name} ({size})")
        print()
    
    # Track success
    success = True
    
    # 1. Download Qwen 4-bit model (required for LoRA adapters)
    if not status['qwen_4bit']:
        print(f"\n[1/3] Downloading Qwen2.5-VL-7B-Instruct (4-bit)...")
        if not download_qwen_4bit_model(output_dir):
            print("FATAL: Qwen 4-bit model download failed!")
            success = False
    
    # 2. Download Qwen FP16 model (for vLLM grounding/numeric)
    if not args.skip_fp16 and not status['qwen_fp16']:
        print(f"\n[2/3] Downloading Qwen2.5-VL-7B-Instruct (FP16)...")
        if not download_qwen_fp16_model(output_dir):
            print("WARNING: Qwen FP16 model download failed!")
            print("Grounding will fall back to 4-bit model.")
    
    # 3. Download SAM3 model
    if not args.skip_sam3 and not status['sam3']:
        print(f"\n[3/3] Downloading SAM3 model (facebook/sam3)...")
        if not download_sam3_model(output_dir):
            print("WARNING: SAM3 model download failed!")
            print("You may need to accept the license first.")
    
    # 4. Copy LoRA adapters if source provided
    if args.adapters_source:
        copy_lora_adapters(output_dir, Path(args.adapters_source))
    
    # Final summary
    print("\n" + "="*60)
    print("  SETUP COMPLETE")
    print("="*60)
    
    # Refresh status
    status = check_volume_setup(output_dir)
    
    # Show what's installed
    print("\nInstalled weights:")
    for item in sorted(output_dir.iterdir()):
        if item.is_dir():
            size = get_dir_size_gb(item)
            print(f"  📁 {item.name}: {size:.2f} GB")
    
    total_size = get_dir_size_gb(output_dir)
    print(f"\nTotal volume usage: {total_size:.2f} GB")
    
    # Check if all required components are present
    required_ok = all([
        status['qwen_4bit'],
        status['caption_adapter'],
        status['binary_adapter'],
        status['semantic_adapter'],
        status['numeric_adapter'],
    ])
    
    optional_ok = status['qwen_fp16'] and status['sam3']
    
    if required_ok:
        print("\n✓ All required models installed!")
        if optional_ok:
            print("✓ All optional models installed (FP16, SAM3)")
        else:
            if not status['qwen_fp16']:
                print("! Qwen FP16 not installed - grounding will use 4-bit model")
            if not status['sam3']:
                print("! SAM3 not installed - grounding/numeric segmentation disabled")
        
        print("\nVolume is ready! You can now start the API server.")
    else:
        print("\n✗ Missing required components. Check above for details.")
        
        # Check for missing adapters
        missing_adapters = []
        if not status['caption_adapter']:
            missing_adapters.append("captioning_sft_0.6507")
        if not status['binary_adapter']:
            missing_adapters.append("sft_binary_0.91-score_qwen2.5_4bit")
        if not status['semantic_adapter']:
            missing_adapters.append("sft_semantic_0.85_qwen2.57b_4bit")
        if not status['numeric_adapter']:
            missing_adapters.append("numeric_sft_0.67_qwen2.54bit")
        
        if missing_adapters:
            print("\nMissing LoRA adapters - copy them manually to:")
            for a in missing_adapters:
                print(f"  {output_dir / a}/")
        
        sys.exit(1)


if __name__ == "__main__":
    main()

