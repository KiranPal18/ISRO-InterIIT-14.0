"""
FastAPI Web Server for Satellite VQA (Inter-IIT Format)
Serves inference requests over HTTP/HTTPS

Input/Output format matches the Inter-IIT sample dataset:
- Input: JSON with input_image and queries (caption_query, grounding_query, attribute_query)
- Output: Same JSON with responses added to each query
"""

import json
import logging
import traceback
from typing import Optional, Dict, Any, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import your inference engine
from main import run_inference, preload_all_models

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models matching Inter-IIT format
# ============================================================================

class ImageMetadata(BaseModel):
    """Image metadata"""
    width: Optional[int] = None
    height: Optional[int] = None
    spatial_resolution_m: Optional[float] = Field(default=1.0, description="Meters per pixel")

class ImageInput(BaseModel):
    """Input image specification matching Inter-IIT format"""
    image_id: str = Field(..., description="Unique identifier for image")
    image_url: Optional[str] = Field(None, description="HTTP(S) URL to image")
    image_path: Optional[str] = Field(None, description="Local path to image")
    metadata: Optional[ImageMetadata] = None

class CaptionQuery(BaseModel):
    """Caption generation request"""
    instruction: str = Field(..., description="Caption instruction")
    response: Optional[str] = None  # Will be filled by inference

class GroundingQuery(BaseModel):
    """Grounding (OBB) request"""
    instruction: str = Field(..., description="Grounding instruction")
    response: Optional[List[Dict[str, Any]]] = None  # List of {object-id, obbox}

class BinaryQuery(BaseModel):
    """Binary (Yes/No) VQA request"""
    instruction: str = Field(..., description="Yes/No question")
    response: Optional[str] = None  # "Yes" or "No"

class NumericQuery(BaseModel):
    """Numeric VQA request"""
    instruction: str = Field(..., description="Numeric question")
    response: Optional[float] = None  # Float value

class SemanticQuery(BaseModel):
    """Semantic VQA request"""
    instruction: str = Field(..., description="Semantic question")
    response: Optional[str] = None  # Text answer

class AttributeQueries(BaseModel):
    """Container for attribute queries"""
    binary: Optional[BinaryQuery] = None
    numeric: Optional[NumericQuery] = None
    semantic: Optional[SemanticQuery] = None

class Queries(BaseModel):
    """All query types"""
    caption_query: Optional[CaptionQuery] = None
    grounding_query: Optional[GroundingQuery] = None
    attribute_query: Optional[AttributeQueries] = None

class InferenceRequest(BaseModel):
    """Complete inference request matching Inter-IIT format"""
    input_image: ImageInput
    queries: Optional[Queries] = None

    class Config:
        # Allow extra fields to be passed through
        extra = "allow"

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    models_loaded: int

# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(
    title="Satellite VQA API",
    description="Multi-task VQA system for satellite imagery",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # allow all domains
    allow_credentials=True,
    allow_methods=["*"],              # allow all HTTP methods
    allow_headers=["*"],              # allow all headers
)

# Global state
MODELS_LOADED = False

# ============================================================================
# Startup & Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global MODELS_LOADED
    try:
        logger.info("Loading all models at startup...")
        preload_all_models()
        MODELS_LOADED = True
        logger.info("✓ All models loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load models: {e}")
        MODELS_LOADED = False

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API server...")

# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for RunPod monitoring
    
    Returns:
        HealthResponse: Status and model information
    """
    return HealthResponse(
        status="healthy" if MODELS_LOADED else "degraded",
        message="All systems operational" if MODELS_LOADED else "Models not loaded",
        models_loaded=4 if MODELS_LOADED else 0
    )

# ============================================================================
# Main Inference Endpoint
# ============================================================================

@app.post("/api/infer", response_model=Dict[str, Any])
async def infer(request: Request):
    """
    Run inference on satellite image with specified queries.
    
    Input format (Inter-IIT):
    ```json
    {
        "input_image": {
            "image_id": "sample1.png",
            "image_url": "https://...",
            "metadata": {"width": 512, "height": 512, "spatial_resolution_m": 1.57}
        },
        "queries": {
            "caption_query": {"instruction": "..."},
            "grounding_query": {"instruction": "..."},
            "attribute_query": {
                "binary": {"instruction": "..."},
                "numeric": {"instruction": "..."},
                "semantic": {"instruction": "..."}
            }
        }
    }
    ```
    
    Output: Same JSON with 'response' field added to each query.
    """
    try:
        if not MODELS_LOADED:
            raise HTTPException(status_code=503, detail="Models not loaded. Try again in 30 seconds.")
        
        # Get raw JSON body to preserve structure
        request_dict = await request.json()
        
        image_id = request_dict.get("input_image", {}).get("image_id", "unknown")
        logger.info(f"Inference request for image: {image_id}")
        
        # Run inference - this modifies request_dict in place
        result = run_inference(request_dict)
        
        logger.info(f"✓ Inference completed for {image_id}")
        
        return result
    
    except Exception as e:
        logger.error(f"✗ Inference failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# ============================================================================
# Batch Inference Endpoint (Optional)
# ============================================================================

@app.post("/api/infer-batch")
async def infer_batch(request: Request):
    """
    Run inference on multiple images sequentially.
    
    Input: Array of inference requests
    Output: Array of results (each with responses added)
    """
    try:
        if not MODELS_LOADED:
            raise HTTPException(status_code=503, detail="Models not loaded. Try again in 30 seconds.")
        
        requests_list = await request.json()
        
        if not isinstance(requests_list, list):
            raise HTTPException(status_code=400, detail="Expected array of requests")
        
        results = []
        for i, req in enumerate(requests_list):
            image_id = req.get("input_image", {}).get("image_id", f"item_{i}")
            logger.info(f"Processing batch item {i+1}/{len(requests_list)}: {image_id}")
            result = run_inference(req)
            results.append(result)
        
        logger.info(f"✓ Batch processing completed: {len(results)} images")
        return results
    
    except Exception as e:
        logger.error(f"✗ Batch inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch error: {str(e)}")

# ============================================================================
# Model Info Endpoint
# ============================================================================

@app.get("/api/models")
async def get_models():
    """
    Get information about loaded models
    
    Returns:
        Dict with model information
    """
    try:
        from config import ADAPTER_PATHS
        adapter_info = ADAPTER_PATHS
    except ImportError:
        adapter_info = {
            "caption": "./models/captioning_sft_0.6507",
            "binary": "./models/sft_binary_0.91-score_qwen2.5_4bit",
            "numeric": "./models/numeric_sft_0.67_qwen2.54bit",
            "semantic": "./models/sft_semantic_0.85_qwen2.57b_4bit"
        }
    
    return {
        "caption": {
            "adapter": adapter_info.get("caption", "unknown"),
            "status": "loaded" if MODELS_LOADED else "not_loaded"
        },
        "binary": {
            "adapter": adapter_info.get("binary", "unknown"),
            "status": "loaded" if MODELS_LOADED else "not_loaded"
        },
        "numeric": {
            "adapter": adapter_info.get("numeric", "unknown"),
            "status": "loaded" if MODELS_LOADED else "not_loaded"
        },
        "semantic": {
            "adapter": adapter_info.get("semantic", "unknown"),
            "status": "loaded" if MODELS_LOADED else "not_loaded"
        },
        "grounding": {
            "description": "SAM3 + vLLM for object grounding",
            "status": "available",
            "note": "Loaded on-demand per request"
        }
    }

# ============================================================================
# Documentation & Root Endpoint
# ============================================================================

@app.get("/")
async def root():
    """API documentation root"""
    return {
        "name": "Satellite VQA API",
        "version": "1.0.0",
        "description": "Multi-task VQA system for satellite imagery with grounding support",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "inference": "/api/infer",
            "batch": "/api/infer-batch",
            "models": "/api/models"
        },
        "supported_tasks": {
            "caption": "Generate descriptive captions for satellite images",
            "binary": "Answer yes/no questions about the image",
            "numeric": "Count objects or answer numeric questions",
            "semantic": "Answer descriptive semantic questions",
            "grounding": "Locate objects with oriented bounding boxes (OBB)"
        }
    }

# ============================================================================
# Run Server (for local testing)
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
