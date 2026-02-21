# ISRO InterIIT - Satellite VQA System

This repository contains the solution for the **ISRO Inter IIT 14.0** challenge, featuring a Unified Inference Engine for Satellite Visual Question Answering (VQA). The system utilizes a multi-model architecture to handle detailed image captioning, binary classification, numeric estimation, semantic reasoning, and precise visual grounding on satellite imagery.

## 🚀 Features

The system supports the following VQA tasks:
- **Caption Generation**: detailed descriptions of satellite imagery.
- **Binary Q&A**: Yes/No questions about scene content.
- **Numeric Q&A**: Counting objects and numerical estimation.
- **Semantic Q&A**: Complex reasoning about object attributes and context.
- **Visual Grounding**: Locating objects with oriented bounding boxes (OBB).

## 🏗️ Pipeline Architecture

The system implements a specialized pipeline for processing different types of queries:

### 1. Orchestrator
The central engine (`main.py`) acts as a task router. It analyzes the input JSON, loads the satellite image, and directs specific queries to their respective specialized modules.

### 2. Multi-Process Inference
To maximize performance and accuracy, we employ a hybrid model strategy:

*   **Task-Specific Qwen Models (GPU 0)**:
    *   **Caption, Binary, Numeric, Semantic**: These tasks are handled by a 4-bit quantized **Qwen2.5-VL-7B-Instruct** base model loaded with task-specific LoRA adapters. This ensures high efficiency and task specialization without excessive memory overhead.
    
*   **Grounding & Spatial Reasoning (GPU 1 + GPU 0)**:
    *   **Visual Prompting**: Queries are first processed by a full FP16 **Qwen2.5-VL-7B** model hosted on a dedicated vLLM server (port 8001). This model performs complex reasoning to identify potential object locations.
    *   **Segmentation**: The **SAM3 (Segment Anything Model 3)** is used to generate precise pixel-level masks for the identified objects.
    *   **Spatial Reasoning**: A custom `spatial_reasoning` module applies geometric constraints (e.g., "bottom right", "near the runway") to filter and rerank detections, converting masks into the final Oriented Bounding Boxes (OBB).

## ☁️ Deployment on RunPod

We have designed the system to run in a high-performance cloud environment using **RunPod**.

### Infrastructure Setup
*   **Hardware**: 2x **NVIDIA A40 GPUs** (48GB VRAM each).
    *   **GPU 0**: Runs the API server, SAM3, and 4-bit LoRA models.
    *   **GPU 1**: Dedicated to the vLLM Inference Server (FP16 Model) for heavy reasoning tasks.
*   **Container**: Custom Docker image pre-built with all dependencies (PyTorch, vLLM, SAM3, etc.).

### Execution Flow
1.  **Docker Initialization**: The container starts and executes `start.sh`.
2.  **Model Loading**:
    *   vLLM server launches on localhost:8001 (GPU 1).
    *   FastAPI server launches on 0.0.0.0:8000 (GPU 0), preloading LoRA adapters and SAM3.
3.  **API Exposure**: The inference endpoint is exposed via RunPod's public proxy or SSH tunnel.

## 📂 Project Structure

```
.
├── app/
│   ├── api.py                 # FastAPI server implementation (Port 8000)
│   ├── main.py                # Pipeline Orchestrator & CLI entry
│   ├── vllm_server.py         # vLLM Server Manager (Port 8001)
│   ├── config.py              # Configuration & Model Paths
│   ├── grounding.py           # Grounding logic (SAM3 + Spatial Reasoning)
│   ├── spatial_reasoning.py   # Geometric analysis submodule
│   ├── sam3/                  # Segment Anything Model 3 integration
│   └── ... (task modules: binary.py, caption.py, numeric.py, semantic.py)
├── README.md                  # This file
└── ...
```

## 🛠️ Installation & Setup

### 1. Environment Setup

Required dependencies are listed in `app/requirements.txt`. It is recommended to use a virtual environment or Docker container.

```bash
pip install -r app/requirements.txt
```

### 2. Download Model Weights

The system uses several large models locally. Run the setup script to download them to your workspace (default: `/workspace/models`).

```bash
python app/download_weights.py
```

Models used:
- **Qwen2.5-VL-7B-Instruct (4-bit)**: Optimized for general understanding.
- **Qwen2.5-VL-7B-Instruct (FP16)**: For complex reasoning and grounding via vLLM.
- **SAM3**: For precise segmentation and object grounding.

## 💻 Usage

### 1. API Server (Recommended)

To start the FastAPI server:

```bash
# Using the startup script (also starts SSH)
bash app/start.sh

# Or directly with uvicorn
cd app
uvicorn api:app --host 0.0.0.0 --port 8000
```

**API Endpoint:** `POST /predict`

### 2. Command Line Interface

You can run the inference engine directly on a JSON file.

```bash
python app/main.py --input path/to/input.json --output path/to/output.json
```

## 📝 Input Format

The system accepts JSON input following the Inter-IIT format:

```json
{
    "input_image": {    
        "image_id": "sample1.png",
        "image_url": "https://bit.ly/4ouV45l",
        "metadata": {
            "width": 512,
            "height": 512,
            "spatial_resolution_m": 1.57
        }
    },
    "queries": {
        "caption_query": {
            "instruction": "Generate a detailed caption..."
        },
        "grounding_query": {
            "instruction": "Locate aircrafts..."
        },
        "attribute_query": {
            "binary": { "instruction": "Is there a..." },
            "numeric": { "instruction": "How many..." },
            "semantic": { "instruction": "What is..." }
        }
    }
}
```

## ⚙️ Configuration

System path configurations and model IDs can be customized in `app/config.py`. 
For Low-VRAM environments, you can toggle `LOW_VRAM_MODE = True` in `app/main.py`.

## 📜 License

[License Information]
