import os
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import random
import string
from dotenv import load_dotenv


_THIS_FILE = Path(__file__).resolve()


def _resolve_repo_root() -> Path:
    for candidate in _THIS_FILE.parents:
        if (candidate / ".env").exists():
            return candidate
    # Fallback to the outermost parent if no .env is found
    return _THIS_FILE.parents[-1]


_REPO_ROOT = _resolve_repo_root()
load_dotenv(dotenv_path=_REPO_ROOT / ".env", override=False)  # Loads .env if present
DEFAULT_MODEL = os.getenv("DEFAULT_AGENT_MODEL", "Qwen3-32b")

logger = logging.getLogger(__name__)

# Generate fixed seed block once at module load time for prefix caching
# This ensures the same seed is used across all requests in an eval run
_FIXED_SEED_LINES = [
    ''.join(random.choices(string.ascii_letters + string.digits, k=80))
    for _ in range(2)
]
_FIXED_SEED_BLOCK = (
    "<RANDOM SEED PLEASE IGNORE>\n" +
    "\n".join(_FIXED_SEED_LINES) +
    "\n</RANDOM SEED>"
)

def generate(
    prompt_text,
    system_text="You are a Diplomacy AI. Play faithfully to your personality profile. Always try to move the game forward per your objectives. Avoid repetition in your journal. Return valid JSON.",
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    retries: int = 2,
    client_options: Optional[Dict[str, Any]] = None,
):
    """
    Calls OpenAI's chat model to get Diplomacy orders using requests.
    The prompt_text should be structured JSON describing the game state.
    Retries the call up to `retries` times with a 5-second delay between attempts on failure.
    """
    client_options = client_options or {}

    api_key = client_options.get("api_key")
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY", "")

    base_url = client_options.get("base_url") or os.getenv("OPENAI_BASE_URL", "http://localhost:8001/v1")

    timeout = client_options.get("timeout", 600)

    headers: Dict[str, str] = {
        "Content-Type": "application/json"
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    extra_headers = client_options.get("headers")
    if isinstance(extra_headers, dict):
        headers.update({str(k): str(v) for k, v in extra_headers.items()})
    include_seed = True
    if include_seed:
        # Use fixed seed block generated at module load time (same across all requests in this eval run)
        system_text += '\n\n' + _FIXED_SEED_BLOCK
    
    messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": prompt_text}
        ]
    
    #print(prompt_text)
    if model_name == "google/gemini-2.0-flash-001":
        temperature = 0.0    

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        #"min_p": 0.1,
        "top_k": 100,
        "max_output_tokens": 600,
        "response_format": {"type": "json_object"},
        "alpha_frequency": 0.1
    }

    if True and model_name == 'deepseek/deepseek-r1':
        print('adding deepseek stuff')
        payload['provider'] = {
                    "order": [
                        #"DeepInfra", # llama-3.1-8b, mistral-small-3, qwen-72b, r1
                        #"Mistral" # mistral-small-3
                        #"DeepSeek", # r1
                        #"Lambda", # llama-3.1-8b
                        #"NovitaAI",  # qwen-72b, llama-3.1-8b
                        #"nebius", # qwen-72b, r1
                        #"Hyperbolic", # qwen-72b
                        #"inference.net", # llama-3.1-8b
                        #"friendly", # r1
                        #"fireworks", # r1
                        #"klusterai", # r1
                        "together", # r1
                    ],
                    "allow_fallbacks": True
                }

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            
            llm_response = response.json()["choices"][0]["message"]["content"]
            #logger.debug(f"LLM raw response: {llm_response}")
            #print(llm_response)
            return llm_response

        except requests.RequestException as e:
            logger.error(f"API request failed on attempt {attempt}: {e}")
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse response on attempt {attempt}: {e}")

        if attempt < retries:
            time.sleep(3)  # wait 5 seconds before retrying

    # Fallback after all retries have been exhausted
    #return {"journal_update": ["(Error calling LLM, fallback)"], "orders": []}
    return ''
