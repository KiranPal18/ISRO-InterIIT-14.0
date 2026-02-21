"""
Semantic Module - Semantic VQA (Descriptive Answers)
Answers semantic questions about satellite images
"""

import torch
import numpy as np
import os
from PIL import Image
from transformers import AutoTokenizer, AutoModel
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
ADAPTER_DIR = "sft_semantic_0.85_qwen2.57b_4bit" 


def load_semantic_model(adapter_dir=None, load_in_4bit=True):
    """
    Load semantic VQA model with adapter weights
    """
    try:
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Determine correct path using config or fallback
        if adapter_dir and os.path.isdir(adapter_dir):
            model_path = adapter_dir
        elif get_adapter_path:
            model_path = get_adapter_path("semantic")
        elif os.path.isdir(ADAPTER_DIR):
            model_path = ADAPTER_DIR
        else:
            logger.warning(f"Adapter not found, loading base model.")
            model_path = BASE_MODEL_ID
            
        logger.info(f"Loading semantic model from: {model_path}")
        
        model, tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=load_in_4bit,
        )
        
        # CRITICAL: Enable Unsloth inference optimizations
        FastVisionModel.for_inference(model)
        model.to(device)
        
        logger.info("Semantic model loaded successfully")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading semantic model: {e}")
        raise


def answer_semantic_question(
    model,
    tokenizer,
    image,
    instruction,
    device="cuda",
    max_new_tokens=10, # Updated to 10 to match Notebook (Brevity)
    temperature=0.1,
    min_p=0.0
):
    """
    Answer semantic question about an image.
    Logic matches the 'evaluate' loop in the notebook.
    """
    try:
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # 1. Prepare Messages (Exact match to Notebook)
        # We append the instruction constraint to the user query
        user_prompt = f"{instruction}\nAnswer with a single word or short phrase."

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer visually based questions concisely with a single word or short phrase."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        
        # 2. Tokenize
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
        
        # 3. Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                temperature=temperature,
                min_p=min_p,
                do_sample=False
            )
        
        # 4. Decode
        response = tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # 5. Clean
        response = clean_semantic_response(response)
        
        return response # Return cleaned response
    
    except Exception as e:
        logger.error(f"Error answering semantic question: {e}")
        raise


def clean_semantic_response(text):
    """
    Clean semantic response by removing common prefixes.
    Matches 'clean_prediction' from the Notebook.
    """
    text = text.strip()
    
    # Remove trailing period if it's the only punctuation
    if text.endswith('.') and text.count('.') == 1:
        text = text[:-1]
    
    # Remove common conversational fillers (Updated list from Notebook)
    fillers = [
        "The primary object is", 
        "The image shows", 
        "The answer is", 
        "It is a", 
        "It is an", 
        "This is"
    ]
    
    lower_text = text.lower()
    for filler in fillers:
        if lower_text.startswith(filler.lower()):
            # Slice the original text to preserve case of the answer
            text = text[len(filler):].strip()
            # Remove leading punctuation (like "The answer is: Tarmac" -> ": Tarmac")
            if text.startswith(":") or text.startswith("."):
                text = text[1:].strip()
            break
    
    return text


class CustomBertBleuScorer:
    """
    Compute similarity between text using BERT embeddings.
    Exact copy of the class used in the Jupyter Notebook.
    """
    
    def __init__(self, model_name="bert-base-uncased", device="cuda"):
        logger.info(f"Loading BERT scorer ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        self.model.eval()

    def _get_ngrams(self, text, n):
        """Extract n-grams from text"""
        tokens = str(text).lower().split()
        if len(tokens) < n:
            return []
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def _get_embeddings(self, phrases):
        """Get BERT embeddings for phrases"""
        if not phrases:
            return None
        inputs = self.tokenizer(
            phrases,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        last_hidden_state = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def compute_score(self, candidate, reference, N=4, alpha=0.5):
        """Compute BERT BLEU score"""
        cand_tokens = str(candidate).lower().split()
        ref_tokens = str(reference).lower().split()
        L_C = len(cand_tokens)
        L_R = len(ref_tokens)
        
        if L_R == 0:
            return 0.0
        
        # Length Penalty
        lp_val = np.exp(-alpha * (abs(L_C - L_R) / L_R))
        
        p_n_scores = []
        for n in range(1, N + 1):
            cand_ngrams = self._get_ngrams(candidate, n)
            ref_ngrams = self._get_ngrams(reference, n)
            
            if not ref_ngrams:
                p_n_scores.append(0.0)
                continue
            if not cand_ngrams:
                p_n_scores.append(0.0)
                continue

            cand_embs = self._get_embeddings(cand_ngrams)
            ref_embs = self._get_embeddings(ref_ngrams)
            cosine_sim_matrix = torch.mm(cand_embs, ref_embs.t())
            max_sims_per_ref, _ = torch.max(cosine_sim_matrix, dim=0)
            p_n = torch.mean(max_sims_per_ref).item()
            p_n_scores.append(p_n)
        
        if not p_n_scores:
            return 0.0
        
        return lp_val * max(p_n_scores)