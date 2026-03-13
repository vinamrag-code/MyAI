# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self):
        # SearXNG Config
        self.searxng_url = os.getenv("SEARXNG_URL", "http://140.238.166.109:8081/search")
        self.searxng_language = os.getenv("SEARXNG_LANGUAGE", "en")
        self.searxng_categories = os.getenv("SEARXNG_CATEGORIES", "general,news")
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "12"))
        
        # API Keys
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        self.hermes_api_key = os.getenv("HERMES_API_KEY", "choose-any-value")
        
        # API Endpoints
        self.hermes_api_url = os.getenv("HERMES_API_URL", "https://hermes.ai.unturf.com/v1/chat/completions")
        self.openrouter_api_url = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
        self.groq_api_url = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
        self.google_api_url = os.getenv("GOOGLE_API_URL", "https://generativelanguage.googleapis.com/v1beta/models")
        
        # Model Configuration
        self.cooldown_period = int(os.getenv("COOLDOWN_PERIOD", "300"))
        
        # Server Config
        self.port = int(os.getenv("PORT", "8000"))
        self.host = os.getenv("HOST", "0.0.0.0")