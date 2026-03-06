"""
LLM-based OCR Post-Correction Package

Provides local LLM integration (Ollama) for typo/diacritic restoration
with content-preservation guardrails.
"""

from .corrector import LLMCorrector, CorrectionResult, GuardrailViolation
from .clients.ollama_client import OllamaClient, OllamaConfig
from .json_schemas import (
    CorrectionRequest,
    CorrectionResponse,
    CorrectionFlags,
    ContextInfo,
    EditOperation
)
from .prompt_schemas import (
    build_full_prompt,
    build_correction_prompt,
    get_system_message,
    validate_response_schema
)

__all__ = [
    # Main corrector
    'LLMCorrector',
    'CorrectionResult',
    'GuardrailViolation',
    
    # Ollama client
    'OllamaClient',
    'OllamaConfig',
    
    # Schemas
    'CorrectionRequest',
    'CorrectionResponse',
    'CorrectionFlags',
    'ContextInfo',
    'EditOperation',
    
    # Prompt utilities
    'build_full_prompt',
    'build_correction_prompt',
    'get_system_message',
    'validate_response_schema',
]
