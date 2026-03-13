# ai_models.py
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class AIModel:
    """AI Model configuration"""
    provider: str
    model_name: str
    api_url: str
    api_key_env: str
    is_free: bool = True
    max_tokens: int = 4096
    temperature: float = 0.7
    
    def get_api_key(self, config) -> str:
        """Get API key from config"""
        return getattr(config, self.api_key_env, "")

# List of all available models
AVAILABLE_MODELS: List[AIModel] = [
    # Hermes Models
    AIModel(
        provider="hermes",
        model_name="adamo1139/Hermes-3-Llama-3.1-8B-FP8-Dynamic",
        api_url="https://hermes.ai.unturf.com/v1/chat/completions",
        api_key_env="hermes_api_key",
        is_free=True
    ),
    
    # OpenRouter Models
    AIModel(
        provider="openrouter",
        model_name="nvidia/nemotron-3-super-120b-a12b:free",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="openrouter_api_key",
        is_free=True
    ),
    AIModel(
        provider="openrouter",
        model_name="nvidia/llama-nemotron-embed-vl-1b-v2:free",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="openrouter_api_key",
        is_free=True
    ),
    AIModel(
        provider="openrouter",
        model_name="stepfun/step-3.5-flash:free",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="openrouter_api_key",
        is_free=True
    ),
    AIModel(
        provider="openrouter",
        model_name="arcee-ai/trinity-large-preview:free",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="openrouter_api_key",
        is_free=True
    ),
    AIModel(
        provider="openrouter",
        model_name="liquid/lfm-2.5-1.2b-thinking:free",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="openrouter_api_key",
        is_free=True
    ),
    AIModel(
        provider="openrouter",
        model_name="liquid/lfm-2.5-1.2b-instruct:free",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="openrouter_api_key",
        is_free=True
    ),
    AIModel(
        provider="openrouter",
        model_name="qwen/qwen3-next-80b-a3b-instruct:free",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="openrouter_api_key",
        is_free=True
    ),
    AIModel(
        provider="openrouter",
        model_name="openai/gpt-oss-120b:free",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="openrouter_api_key",
        is_free=True
    ),
    AIModel(
        provider="openrouter",
        model_name="z-ai/glm-4.5-air:free",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="openrouter_api_key",
        is_free=True
    ),
    AIModel(
        provider="openrouter",
        model_name="qwen/qwen3-coder:free",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="openrouter_api_key",
        is_free=True
    ),
    AIModel(
        provider="openrouter",
        model_name="qwen/qwen3-4b:free",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="openrouter_api_key",
        is_free=True
    ),
    AIModel(
        provider="openrouter",
        model_name="google/gemma-3-27b-it:free",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="openrouter_api_key",
        is_free=True
    ),
    AIModel(
        provider="openrouter",
        model_name="meta-llama/llama-3.3-70b-instruct:free",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="openrouter_api_key",
        is_free=True
    ),
    AIModel(
        provider="openrouter",
        model_name="nousresearch/hermes-3-llama-3.1-405b:free",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="openrouter_api_key",
        is_free=True
    ),
    
    # Groq Models (if you want to add them)
    # AIModel(
    #     provider="groq",
    #     model_name="mixtral-8x7b-32768",
    #     api_url="https://api.groq.com/openai/v1/chat/completions",
    #     api_key_env="groq_api_key",
    #     is_free=True
    # ),
    
    # Google Gemini Models (if you want to add them)
    # AIModel(
    #     provider="google",
    #     model_name="gemini-pro",
    #     api_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
    #     api_key_env="google_api_key",
    #     is_free=True
    # ),
]

def get_available_models() -> List[AIModel]:
    """Return list of available models"""
    return AVAILABLE_MODELS