# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self):
        # Search Config (DDGS)
        self.ddgs_region = os.getenv("DDGS_REGION", "")
        self.ddgs_timelimit = os.getenv("DDGS_TIMELIMIT", "")
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