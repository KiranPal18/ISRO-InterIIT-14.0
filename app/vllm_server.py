"""
vLLM Server Manager and Client for Grounding & Complex Numeric Tasks

This module provides:
1. A vLLM OpenAI-compatible server manager
2. A client wrapper to call the vLLM server
3. Shared across grounding.py and numeric.py pipelines

Architecture:
- vLLM server runs on port 8001 (internal)
- FastAPI serves on port 8000 (external)
- Both share the same Qwen2.5-VL-7B model via HTTP API
"""

import os
import sys
import time
import json
import base64
import subprocess
import threading
import logging
import requests
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VLLMServerConfig:
    """Configuration for vLLM server."""
    # Use HuggingFace model ID - will be cached locally in download_dir
    # Note: Local path has config issues with vLLM 0.12+ and transformers 4.57+
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    host: str = "0.0.0.0"
    port: int = 8001
    tensor_parallel_size: int = 1  # Use 1 GPU for vLLM, leave other for LoRA models
    gpu_memory_utilization: float = 0.50  # Can use more since GPU 1 is dedicated
    max_model_len: int = 16384  # Reduced for memory efficiency
    trust_remote_code: bool = True
    dtype: str = "bfloat16"
    download_dir: str = "/workspace/models/hf_cache"  # Cache downloaded models
    
    # Environment
    cuda_visible_devices: str = "1"  # Use GPU 1 for vLLM, GPU 0 for LoRA
    
    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}"
    
    @property
    def completions_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"
    
    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"


# Global config instance
VLLM_CONFIG = VLLMServerConfig()

# Global server process
_server_process: Optional[subprocess.Popen] = None
_server_lock = threading.Lock()


# =============================================================================
# Server Management
# =============================================================================

def is_server_running() -> bool:
    """Check if vLLM server is running and healthy."""
    try:
        response = requests.get(VLLM_CONFIG.health_url, timeout=5)
        return response.status_code == 200
    except:
        return False


def start_vllm_server(
    config: Optional[VLLMServerConfig] = None,
    wait_for_ready: bool = True,
    timeout: int = 300  # 5 minutes max wait
) -> bool:
    """
    Start vLLM OpenAI-compatible server as a subprocess.
    
    Args:
        config: Server configuration (uses global default if None)
        wait_for_ready: Wait for server to be ready before returning
        timeout: Maximum seconds to wait for server
        
    Returns:
        True if server started successfully
    """
    global _server_process
    
    if config is None:
        config = VLLM_CONFIG
    
    with _server_lock:
        # Check if already running
        if is_server_running():
            logger.info("vLLM server already running")
            return True
        
        # Kill any existing process
        if _server_process is not None:
            try:
                _server_process.terminate()
                _server_process.wait(timeout=10)
            except:
                pass
        
        logger.info(f"Starting vLLM server on port {config.port}...")
        logger.info(f"  Model: {config.model}")
        logger.info(f"  GPU: {config.cuda_visible_devices}")
        logger.info(f"  Memory utilization: {config.gpu_memory_utilization}")
        
        # Build command
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", config.model,
            "--host", config.host,
            "--port", str(config.port),
            "--tensor-parallel-size", str(config.tensor_parallel_size),
            "--gpu-memory-utilization", str(config.gpu_memory_utilization),
            "--max-model-len", str(config.max_model_len),
            "--dtype", config.dtype,
            "--trust-remote-code",
            "--disable-log-requests",
        ]
        
        # Set environment for GPU selection
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
        
        # Start server process
        try:
            _server_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Start output reader thread
            def log_output():
                for line in _server_process.stdout:
                    logger.debug(f"[vLLM] {line.rstrip()}")
            
            threading.Thread(target=log_output, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Failed to start vLLM server: {e}")
            return False
    
    # Wait for server to be ready
    if wait_for_ready:
        logger.info("Waiting for vLLM server to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if is_server_running():
                elapsed = time.time() - start_time
                logger.info(f"vLLM server ready in {elapsed:.1f}s")
                return True
            
            # Check if process died
            if _server_process.poll() is not None:
                logger.error("vLLM server process terminated unexpectedly")
                return False
            
            time.sleep(5)
        
        logger.error(f"vLLM server failed to start within {timeout}s")
        return False
    
    return True


def stop_vllm_server():
    """Stop the vLLM server."""
    global _server_process
    
    with _server_lock:
        if _server_process is not None:
            logger.info("Stopping vLLM server...")
            try:
                _server_process.terminate()
                _server_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _server_process.kill()
            _server_process = None
            logger.info("vLLM server stopped")


# =============================================================================
# Client Wrapper
# =============================================================================

class VLLMClient:
    """
    Client for calling vLLM server with vision capabilities.
    
    Provides a simple interface compatible with the existing code.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3
    ):
        self.base_url = base_url or VLLM_CONFIG.base_url
        self.completions_url = f"{self.base_url}/v1/chat/completions"
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
    
    def is_available(self) -> bool:
        """Check if server is available."""
        try:
            response = self._session.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def _encode_image(self, image: Union[str, Image.Image]) -> str:
        """Encode image to base64 data URI."""
        if isinstance(image, str):
            # File path
            with open(image, "rb") as f:
                image_bytes = f.read()
        else:
            # PIL Image
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/png;base64,{base64_image}"
    
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[Union[str, Image.Image]]] = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Send chat completion request to vLLM server.
        
        Args:
            messages: Chat messages in OpenAI format
            images: Optional list of images to include
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        # Convert images to base64 and inject into messages
        if images:
            formatted_messages = self._format_messages_with_images(messages, images)
        else:
            formatted_messages = messages
        
        payload = {
            "model": VLLM_CONFIG.model,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._session.post(
                    self.completions_url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
            except requests.exceptions.Timeout:
                last_error = "Request timed out"
                logger.warning(f"vLLM request timeout (attempt {attempt + 1})")
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.warning(f"vLLM request failed: {e} (attempt {attempt + 1})")
            except (KeyError, IndexError) as e:
                last_error = f"Invalid response format: {e}"
                logger.error(f"vLLM invalid response: {e}")
                break
            
            time.sleep(2 ** attempt)  # Exponential backoff
        
        raise RuntimeError(f"vLLM request failed after {self.max_retries} attempts: {last_error}")
    
    def _format_messages_with_images(
        self,
        messages: List[Dict[str, Any]],
        images: List[Union[str, Image.Image]]
    ) -> List[Dict[str, Any]]:
        """Format messages with image content for vision model."""
        formatted = []
        image_idx = 0
        
        for msg in messages:
            if msg["role"] == "user" and image_idx < len(images):
                # Add image to user message
                content = msg.get("content", "")
                if isinstance(content, str):
                    # Convert to multimodal format
                    image_url = self._encode_image(images[image_idx])
                    formatted.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            },
                            {
                                "type": "text",
                                "text": content
                            }
                        ]
                    })
                    image_idx += 1
                else:
                    formatted.append(msg)
            else:
                formatted.append(msg)
        
        return formatted
    
    def _process_agent_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process messages in agent format ({"type": "image", "image": path})
        and convert to OpenAI-compatible format ({"type": "image_url", "image_url": {...}}).
        
        This is used by agent_inference from SAM3 which uses a different message format.
        """
        processed_messages = []
        
        for message in messages:
            processed_message = message.copy()
            
            if message["role"] == "user" and "content" in message:
                content = message.get("content", [])
                
                # Handle list content (multimodal messages)
                if isinstance(content, list):
                    processed_content = []
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "image":
                            # Convert agent-style image to OpenAI format
                            image_path = c.get("image", "")
                            # Handle escaped paths
                            new_image_path = image_path.replace("?", "%3F")
                            
                            try:
                                base64_image = self._encode_image(new_image_path)
                                processed_content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": base64_image,
                                        "detail": "high"
                                    }
                                })
                            except FileNotFoundError:
                                logger.warning(f"Image file not found: {new_image_path}")
                                continue
                            except Exception as e:
                                logger.warning(f"Error processing image {new_image_path}: {e}")
                                continue
                        else:
                            processed_content.append(c)
                    
                    processed_message["content"] = processed_content
                # String content stays as-is
                
            processed_messages.append(processed_message)
        
        return processed_messages
    
    def chat_for_agent(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.0
    ) -> Optional[str]:
        """
        Chat completion for SAM3 agent format messages.
        
        Handles the agent-style message format with {"type": "image", "image": path}
        and converts it to OpenAI-compatible format before sending.
        
        Args:
            messages: Messages in SAM3 agent format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text, or None if inference fails
        """
        try:
            # Convert agent messages to OpenAI format
            processed_messages = self._process_agent_messages(messages)
            
            logger.info("🔍 Calling vLLM server for agent inference...")
            
            payload = {
                "model": VLLM_CONFIG.model,
                "messages": processed_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            # Retry logic
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    response = self._session.post(
                        self.completions_url,
                        json=payload,
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                    
                except requests.exceptions.Timeout:
                    last_error = "Request timed out"
                    logger.warning(f"Agent request timeout (attempt {attempt + 1})")
                except requests.exceptions.RequestException as e:
                    last_error = str(e)
                    logger.warning(f"Agent request failed: {e} (attempt {attempt + 1})")
                except (KeyError, IndexError) as e:
                    last_error = f"Invalid response format: {e}"
                    logger.error(f"Agent invalid response: {e}")
                    break
                
                time.sleep(2 ** attempt)  # Exponential backoff
            
            logger.error(f"Agent request failed after {self.max_retries} attempts: {last_error}")
            return None
            
        except Exception as e:
            logger.error(f"Agent chat failed: {e}")
            return None
    
    def generate(
        self,
        prompt: str,
        image: Optional[Union[str, Image.Image]] = None,
        max_tokens: int = 8192,
        temperature: float = 0.0
    ) -> str:
        """
        Simple generate interface (compatible with existing code).
        
        Args:
            prompt: Text prompt
            image: Optional image
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        messages = [{"role": "user", "content": prompt}]
        images = [image] if image else None
        return self.chat_completion(messages, images, max_tokens, temperature)


# =============================================================================
# Singleton Client Instance
# =============================================================================

_client_instance: Optional[VLLMClient] = None

def get_vllm_client(auto_start: bool = True) -> Optional[VLLMClient]:
    """
    Get or create the global vLLM client instance.
    
    Args:
        auto_start: Start vLLM server if not running
        
    Returns:
        VLLMClient instance, or None if server unavailable
    """
    global _client_instance
    
    if _client_instance is None:
        _client_instance = VLLMClient()
    
    if not _client_instance.is_available():
        if auto_start:
            logger.info("vLLM server not available, starting...")
            if start_vllm_server():
                return _client_instance
            else:
                logger.error("Failed to start vLLM server")
                return None
        else:
            return None
    
    return _client_instance


# =============================================================================
# Cleanup
# =============================================================================

import atexit
atexit.register(stop_vllm_server)


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="vLLM Server Manager")
    parser.add_argument("command", choices=["start", "stop", "status", "test"])
    parser.add_argument("--gpu", type=str, default="1", help="GPU to use")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    
    args = parser.parse_args()
    
    VLLM_CONFIG.cuda_visible_devices = args.gpu
    VLLM_CONFIG.port = args.port
    
    if args.command == "start":
        success = start_vllm_server()
        print(f"Server started: {success}")
        if success:
            print(f"Server URL: {VLLM_CONFIG.base_url}")
            input("Press Enter to stop server...")
            stop_vllm_server()
    
    elif args.command == "stop":
        stop_vllm_server()
        print("Server stopped")
    
    elif args.command == "status":
        running = is_server_running()
        print(f"Server running: {running}")
    
    elif args.command == "test":
        client = get_vllm_client()
        if client:
            response = client.generate("Hello, what is 2+2?")
            print(f"Response: {response}")
        else:
            print("Server not available")
