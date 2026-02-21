"""
Caption Module - Image Captioning with LoRA Adapters
Generates detailed captions for satellite images
"""

import torch
import os
import math
import numpy as np
from PIL import Image
from unsloth import FastVisionModel
from transformers import AutoTokenizer, AutoModel
import logging

# Import centralized config
try:
    from config import get_adapter_path, BASE_MODEL_ID
except ImportError:
    BASE_MODEL_ID = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"
    get_adapter_path = None

logger = logging.getLogger(__name__)

# Default adapter directory
ADAPTER_DIR = "captioning_sft_0.6507"

# The specific prompt used during training (Critical for performance)
USER_PROMPT = (
    # "Generate a detailed caption describing all visible elements in the satellite image, "
    "including object types, counts, relative locations, textures, land-use patterns, and overall scene context. "
    "*WARNING:* The final evaluation metric (BERT-BLEU4) includes a *severe length penalty* "
    "that significantly reduces the score if the generated caption's length widely deviates from the optimal length. "
    "Therefore, match the caption length to the visual complexity of the image. "
    "Produce a natural human-like satellite image caption, "
    "typically *between 52 to 55 words*, but be ready to slightly adjust this range if the scene demands a much shorter or longer description to achieve the best metric score."
)


def load_caption_model(adapter_dir=None, load_in_4bit=True):
    """
    Load caption model with adapter weights
    """
    try:
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Determine correct path using config or fallback
        if adapter_dir and os.path.isdir(adapter_dir):
            model_path = adapter_dir
        elif get_adapter_path:
            model_path = get_adapter_path("caption")
        elif os.path.isdir(ADAPTER_DIR):
            model_path = ADAPTER_DIR
        else:
            logger.warning(f"Adapter not found, loading base model.")
            model_path = BASE_MODEL_ID
            
        logger.info(f"Loading caption model from: {model_path}")
        
        model, tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=load_in_4bit,
            # use_gradient_checkpointing is usually for training
        )
        
        # CRITICAL: Enable Unsloth inference optimizations
        FastVisionModel.for_inference(model)
        model.to(device)
        
        logger.info("Caption model loaded successfully")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading caption model: {e}")
        raise


def generate_caption(
    model,
    tokenizer,
    image,
    instruction=None,
    device="cuda",
    max_new_tokens=256,
    temperature=1.0,
    min_p=0.1
):
    """
    Generate caption for an image using the specific training prompt.
    """
    try:
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Use the training prompt if no specific instruction override is provided
        final_instruction = instruction if instruction else USER_PROMPT
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": final_instruction}
                ]
            }
        ]
        
        # Tokenize
        input_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(device)
        
        # Generate
        # Using parameters from the notebook (temp=1.0, min_p=0.1)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                temperature=temperature,
                min_p=min_p,
                do_sample=True # Explicitly True based on notebook logic
            )
        
        # Decode
        caption = tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return caption
    
    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        raise


# ============================================================================
# EVALUATION CLASS (Ported from Notebook)
# ============================================================================
class BERT_BLEU_Evaluator:
    """
    Evaluator class to compute BERT-based similarity scores.
    """
    def __init__(self, device=None, model_name="bert-base-uncased"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        # Exclude [CLS] and [SEP]
        return tokens[1:-1], embeddings[1:-1]

    def get_ngram_embeddings(self, tokens, embeddings, n):
        length = len(tokens)
        if length < n: return [], []
        ngrams = []
        ngram_vecs = []
        for i in range(length - n + 1):
            ngram_tokens = tuple(tokens[i : i + n])
            vec = torch.mean(embeddings[i : i + n], dim=0)
            ngrams.append(ngram_tokens)
            ngram_vecs.append(vec)
        return ngrams, ngram_vecs

    def compute_cosine_matrix(self, cand_vecs, ref_vecs):
        if not cand_vecs or not ref_vecs: return torch.tensor(0.0)
        C = torch.stack(cand_vecs)
        R = torch.stack(ref_vecs)
        C_norm = torch.nn.functional.normalize(C, p=2, dim=1)
        R_norm = torch.nn.functional.normalize(R, p=2, dim=1)
        return torch.mm(C_norm, R_norm.t())

    def calculate_score(self, candidate_text, reference_text, lp_alpha=0.5, n_gram_max=4):
        c_toks, c_embs = self.get_embeddings(candidate_text)
        r_toks, r_embs = self.get_embeddings(reference_text)
        
        L_C = len(c_toks)
        L_R = len(r_toks)
        if L_R == 0: return {"score": 0.0, "LP": 0.0}
        
        # Length Penalty
        diff = abs(L_C - L_R)
        lp = math.exp(-lp_alpha * (diff / L_R))
        
        p_n_scores = []
        for n in range(1, n_gram_max + 1):
            c_ngrams, c_vecs = self.get_ngram_embeddings(c_toks, c_embs, n)
            r_ngrams, r_vecs = self.get_ngram_embeddings(r_toks, r_embs, n)
            
            if not r_vecs or not c_vecs:
                p_n_scores.append(0.0)
                continue
            
            sim_matrix = self.compute_cosine_matrix(c_vecs, r_vecs)
            max_sims, _ = torch.max(sim_matrix, dim=0)
            p_n = torch.sum(max_sims).item() / len(r_vecs)
            p_n_scores.append(p_n)

        max_p_n = max(p_n_scores) if p_n_scores else 0.0
        final_score = lp * max_p_n
        return {"score": final_score, "LP": lp}