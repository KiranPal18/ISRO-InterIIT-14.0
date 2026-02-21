"""
Binary Module - Binary VQA (Yes/No Questions)
Answers binary questions about satellite images
"""

import torch
import os
from PIL import Image
from unsloth import FastVisionModel
import logging

# Import centralized config
try:
    from config import get_adapter_path, BASE_MODEL_ID
except ImportError:
    BASE_MODEL_ID = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"
    get_adapter_path = None

logger = logging.getLogger(__name__)

# Default adapter directory
ADAPTER_DIR = "sft_binary_0.91-score_qwen2.5_4bit"
FIXED_INSTRUCTION = (
    "You are a binary vision question-answering system. "
    "Answer with ONLY 'yes' or 'no'. No explanation."
)

def load_binary_model(adapter_dir=None, load_in_4bit=True):
    """
    Load binary VQA model with adapter weights
    """
    try:
        # Determine device automatically
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Determine correct path using config or fallback
        if adapter_dir and os.path.isdir(adapter_dir):
            model_path = adapter_dir
        elif get_adapter_path:
            model_path = get_adapter_path("binary")
        elif os.path.isdir(ADAPTER_DIR):
            model_path = ADAPTER_DIR
        else:
            model_path = BASE_MODEL_ID
        
        logger.info(f"Loading binary model from: {model_path} on {device}")
        
        # Load model
        model, tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=load_in_4bit,
            # use_gradient_checkpointing is generally for training, not inference
        )
        
        # Move to device
        model.to(device)
        
        # CRITICAL: Enable Unsloth inference optimizations (RoPE scaling, etc.)
        # This matches the notebook's "FastVisionModel.for_inference(model)"
        FastVisionModel.for_inference(model)
        
        logger.info("Binary model loaded successfully")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading binary model: {e}")
        raise


def answer_binary_question(
    model,
    tokenizer,
    image,
    question,
    device="cuda",
    max_new_tokens=4,
    temperature=0.0 # Inference usually defaults to deterministic for Yes/No
):
    """
    Answer binary (yes/no) question about an image.
    Logic aligned with notebook 'predict_yes_no' function.
    """
    try:
        # 1. Load/Convert image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # 2. Format the conversation (Matching Notebook Logic)
        # The notebook combines fixed instruction + question into one user text prompt
        full_prompt = f"{FIXED_INSTRUCTION} Question: {question}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": image # Passed as object here, tokenizer handles it below
                    },
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]
        
        # 3. Apply chat template to get the raw text prompt (tokenize=False)
        text_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 4. Tokenize inputs (Matches Notebook: images list + text list)
        inputs = tokenizer(
            images=[image],
            text=[text_prompt],
            return_tensors="pt"
        ).to(device)
        
        # 5. Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                # Use temperature if provided and valid, otherwise deterministic
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None
            )
        
        # 6. Decode output (Matching Notebook slicing logic)
        # output_ids[:, inputs.input_ids.shape[1]:] removes the input tokens
        response = tokenizer.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )[0]
        
        # 7. Post-process answer
        text_response = response.strip().lower()
        
        if "yes" in text_response:
            return "Yes"
        elif "no" in text_response:
            return "No"
        else:
            return text_response
            
    except Exception as e:
        logger.error(f"Error answering binary question: {e}")
        raise