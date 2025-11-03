#!/usr/bin/env python3
"""
Ollama client for local LLM inference with JSON mode support.

Implements model fallback logic (qwen2.5 -> llama3.1 -> mistral-nemo)
with strict JSON validation and timeout handling.

Author: Senior ML Engineer
Date: 2025-10-07
"""

import json
import logging
import time
import subprocess
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""
    base_url: str = "http://localhost:11434"
    model_id: str = "qwen2.5:7b-instruct"
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    timeout_s: int = 30
    retries: int = 2
    fallback_models: List[str] = None
    
    def __post_init__(self):
        if self.fallback_models is None:
            self.fallback_models = [
                "qwen2.5:7b-instruct",
                "llama3.1:8b-instruct", 
                "mistral-nemo:12b-instruct"
            ]


class OllamaClient:
    """
    Client for Ollama local LLM inference.
    
    Supports:
    - JSON-mode generation
    - Automatic model fallback
    - Model pulling if missing
    - Timeout and retry handling
    """
    
    def __init__(self, config: OllamaConfig):
        """
        Initialize Ollama client.
        
        Args:
            config: OllamaConfig instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required. Install: pip install requests")
        
        self.base_url = config.base_url.rstrip('/')
        self.model_id = None  # Will be set by ensure_model_available
        
        # Ensure model is available
        self._ensure_model_available()
    
    def _ensure_model_available(self) -> str:
        """
        Ensure a model is available, trying fallbacks if needed.
        
        Returns:
            Available model ID
            
        Raises:
            RuntimeError if no models available
        """
        # Try preferred model first
        models_to_try = [self.config.model_id] + [
            m for m in self.config.fallback_models 
            if m != self.config.model_id
        ]
        
        for model_id in models_to_try:
            self.logger.info(f"Checking model availability: {model_id}")
            
            if self._check_model_exists(model_id):
                self.logger.info(f"✅ Model {model_id} is available")
                self.model_id = model_id
                return model_id
            
            # Try to pull the model
            self.logger.info(f"Model {model_id} not found, attempting to pull...")
            if self._pull_model(model_id):
                self.logger.info(f"✅ Successfully pulled {model_id}")
                self.model_id = model_id
                return model_id
            else:
                self.logger.warning(f"Failed to pull {model_id}, trying next fallback")
        
        raise RuntimeError(
            f"No models available. Tried: {models_to_try}. "
            f"Install Ollama and run: ollama pull {self.config.model_id}"
        )
    
    def _check_model_exists(self, model_id: str) -> bool:
        """Check if model is already pulled."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                models = [m['name'] for m in data.get('models', [])]
                return model_id in models
            
            return False
        except Exception as e:
            self.logger.debug(f"Error checking model existence: {e}")
            return False
    
    def _pull_model(self, model_id: str) -> bool:
        """
        Pull model using ollama CLI.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if successful
        """
        try:
            self.logger.info(f"Pulling model: ollama pull {model_id}")
            result = subprocess.run(
                ["ollama", "pull", model_id],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout for pull
            )
            
            if result.returncode == 0:
                return True
            else:
                self.logger.error(f"Pull failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout pulling {model_id}")
            return False
        except FileNotFoundError:
            self.logger.error("ollama CLI not found. Install from https://ollama.ai")
            return False
        except Exception as e:
            self.logger.error(f"Error pulling model: {e}")
            return False
    
    def generate_json(self, system_message: str = "", user_message: str = "", 
                     retry_count: int = 0) -> Dict[str, Any]:
        """
        Generate JSON response from LLM.
        
        Args:
            system_message: System prompt
            user_message: User prompt
            retry_count: Current retry attempt (internal)
            
        Returns:
            Parsed JSON response
            
        Raises:
            RuntimeError on failure after retries
        """
        if retry_count >= self.config.retries:
            raise RuntimeError(f"Failed after {self.config.retries} retries")
        
        # Build request payload
        messages = []
        
        if system_message:
            messages.append({
                "role": "system",
                "content": system_message
            })
        
        if user_message:
            messages.append({
                "role": "user",
                "content": user_message
            })
        
        payload = {
            "model": self.model_id,
            "messages": messages,
            "format": "json",  # Request JSON output
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens
            },
            "stream": False
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.config.timeout_s
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            if response.status_code != 200:
                self.logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                raise RuntimeError(f"API returned {response.status_code}")
            
            result = response.json()
            content = result.get('message', {}).get('content', '')
            
            if not content:
                raise ValueError("Empty response from model")
            
            # Parse JSON
            try:
                parsed = json.loads(content)
                self.logger.debug(f"JSON response parsed successfully ({latency_ms}ms)")
                return parsed
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"Invalid JSON on attempt {retry_count + 1}: {e}")
                
                # Retry with stronger instruction
                if retry_count < self.config.retries - 1:
                    user_retry = user_message + "\n\nReturn valid JSON only; no prose."
                    
                    return self.generate_json(
                        system_message=system_message,
                        user_message=user_retry,
                        retry_count=retry_count + 1
                    )
                else:
                    raise RuntimeError(f"Failed to get valid JSON after {self.config.retries} attempts")
        
        except requests.Timeout:
            self.logger.error(f"Request timeout after {self.config.timeout_s}s")
            raise RuntimeError(f"LLM request timeout")
        
        except Exception as e:
            self.logger.error(f"Error in generate_json: {e}")
            raise
    
    def health_check(self) -> tuple[bool, str]:
        """
        Check if Ollama service is healthy.
        
        Returns:
            Tuple of (is_healthy, message)
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                return True, "Ollama service is healthy"
            else:
                return False, f"Ollama service returned {response.status_code}"
        except Exception as e:
            return False, f"Ollama service unreachable: {str(e)}"


if __name__ == '__main__':
    # Test the client
    logging.basicConfig(level=logging.INFO)
    
    config = OllamaConfig()
    
    try:
        client = OllamaClient(config)
        
        # Test JSON generation
        test_prompt = {
            "system": "You are a JSON echo bot. Return the user's message as JSON with a 'message' field.",
            "user": "Hello, world!"
        }
        
        result = client.generate_json(test_prompt)
        print(f"✅ Test successful: {result}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
