"""
LLM helpers for generating modern-language wrapper text around transliteration.

Supports: Ollama (local), OpenAI-compatible API (e.g. OpenClaw), or no-op for testing.
"""

import json
import os
from typing import Optional

# Optional: ollama
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


def _ollama_generate(
    prompt: str,
    model: str = "mistral:latest",
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """Generate via Ollama local API."""
    if not REQUESTS_AVAILABLE:
        return ""
    url = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        out = r.json()
        return out.get("response", "").strip()
    except Exception as e:
        return f"[Ollama error: {e}]"  # Caller may replace with fallback if desired


def _openai_compatible_generate(
    prompt: str,
    api_base: str,
    model: str,
    api_key: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """Generate via OpenAI-compatible API (e.g. OpenClaw)."""
    if not REQUESTS_AVAILABLE:
        return ""
    url = api_base.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        out = r.json()
        choice = out.get("choices", [{}])[0]
        return (choice.get("message", {}).get("content") or "").strip()
    except Exception as e:
        return f"[API error: {e}]"  # Caller may replace with fallback if desired


def _is_error_response(text: str) -> bool:
    """True if the response looks like an error message from this module."""
    return isinstance(text, str) and text.strip().startswith("[") and "error" in text.lower()


def generate_wrapper_paragraph(
    transliteration_snippet: str,
    language: str = "English",
    style: str = "academic commentary",
    provider: str = "ollama",
    model: Optional[str] = None,
    fallback_on_error: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Ask the LLM to generate a short paragraph in the given language that could
    sit alongside transliteration (e.g. citation, line reference, or translation).
    If fallback_on_error is set and the LLM returns an error message, that string is used instead.
    """
    prompt = f"""Generate one short paragraph in {language} only, in an {style} style, that could appear in a scholarly publication next to Akkadian/cuneiform transliteration. The paragraph should reference or comment on the following transliteration excerpt, but do not repeat it verbatim. Keep it to 1-3 sentences. Output only the paragraph, no quotes or labels.

Transliteration excerpt:
{transliteration_snippet[:800]}
"""
    # Only pass through kwargs that the generate functions accept
    gen_kw = {k: kwargs[k] for k in ("temperature", "max_tokens") if k in kwargs}
    out = ""
    if provider == "ollama":
        out = _ollama_generate(prompt, model=model or "mistral:latest", **gen_kw)
    elif provider == "openai":
        out = _openai_compatible_generate(
            prompt,
            api_base=kwargs.get("api_base", "https://api.openai.com/v1"),
            model=model or "gpt-4o-mini",
            api_key=kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY"),
            **gen_kw,
        )
    elif provider == "openclaw" or kwargs.get("api_base"):
        out = _openai_compatible_generate(
            prompt,
            api_base=kwargs.get("api_base", "https://api.openclaw.ai/v1"),
            model=model or kwargs.get("model", "gpt-4o-mini"),
            api_key=kwargs.get("api_key") or os.environ.get("OPENCLAW_API_KEY", os.environ.get("OPENAI_API_KEY")),
            **gen_kw,
        )
    else:
        out = "[Set provider to 'ollama', 'openai', or 'openclaw']"
    if fallback_on_error is not None and _is_error_response(out):
        return fallback_on_error
    return out
