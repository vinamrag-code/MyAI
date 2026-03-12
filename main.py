"""
main.py — FastAPI server
Exposes your 3-layer fallback agent as a single REST API.
Any future project just calls  POST /chat  — done.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
import os

from agent import FallbackAgent

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────────
agent = FallbackAgent()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 AI Agent starting — 3-layer fallback ready")
    logger.info("  Layer 1 (Custom Bot) : %s", "✅ configured" if os.getenv("CUSTOM_BOT_URL") else "⚠️  CUSTOM_BOT_URL not set")
    logger.info("  Layer 2 (Serper)     : %s", "✅ configured" if os.getenv("SERPER_API_KEY") else "⚠️  SERPER_API_KEY not set")
    logger.info("  Layer 3 (Gemini)     : %s", "✅ configured" if os.getenv("GEMINI_API_KEY") else "⚠️  GEMINI_API_KEY not set")
    yield
    logger.info("🛑 Shutting down")

app = FastAPI(
    title="AI Fallback Agent",
    description="3-layer AI agent: Custom Bot → Google Search → Gemini 2.0 Flash",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend UI from /static
static_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(static_dir):
    app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="ui")

# ── Pydantic models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's question or message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation memory. Auto-generated if not provided.")
    use_memory: bool = Field(True, description="Whether to use conversation history for context")

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    layer: int = Field(description="Which layer answered: 1=Custom Bot, 2=Google Search, 3=Gemini, 0=All failed")
    layer_name: str = Field(description="Human-readable layer name")
    source_label: Optional[str] = Field(None, description="Label shown to end users when not Layer 1")

class SessionResponse(BaseModel):
    session_id: str
    cleared: bool

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "name": "AI Fallback Agent",
        "version": "1.0.0",
        "layers": {
            "1": "Custom Bot API",
            "2": "Google Serper Search",
            "3": "Gemini 2.0 Flash",
        },
        "endpoints": {
            "chat":          "POST /chat",
            "new_session":   "POST /session/new",
            "clear_session": "DELETE /session/{session_id}",
            "health":        "GET /health",
            "ui":            "GET /ui",
            "docs":          "GET /docs",
        },
    }

@app.get("/health", tags=["Info"])
def health():
    return {
        "status": "ok",
        "layers_configured": {
            "layer1_custom_bot": bool(os.getenv("CUSTOM_BOT_URL")),
            "layer2_searxng":    True  # SearXNG is self-hosted at 140.238.166.109:8081,
            "layer3_gemini":     bool(os.getenv("GEMINI_API_KEY")),
        },
    }

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    """
    Main endpoint. Send a message, get an AI answer.
    The agent automatically tries all 3 layers in order.

    Use this single endpoint in ALL your future projects.
    """
    session_id = req.session_id or ("session_" + uuid.uuid4().hex[:12])

    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    result = await agent.ask(
        message=req.message.strip(),
        session_id=session_id,
        use_memory=req.use_memory,
    )

    # Source label: only show when fallback was used
    source_label = None
    if result["layer"] == 2:
        source_label = "via Google Search"
    elif result["layer"] == 3:
        source_label = "via Gemini AI"
    elif result["layer"] == 0:
        source_label = "No source available"

    return ChatResponse(
        answer=result["answer"],
        session_id=session_id,
        layer=result["layer"],
        layer_name=result["layer_name"],
        source_label=source_label,
    )

@app.post("/session/new", tags=["Session"])
def new_session():
    """Generate a new session ID."""
    return {"session_id": "session_" + uuid.uuid4().hex[:12]}

@app.delete("/session/{session_id}", response_model=SessionResponse, tags=["Session"])
def clear_session(session_id: str):
    """Clear conversation memory for a session."""
    agent.clear_session(session_id)
    return SessionResponse(session_id=session_id, cleared=True)

@app.get("/ui", tags=["UI"])
def ui():
    """Redirect to the chat UI."""
    return FileResponse(os.path.join(static_dir, "index.html"))
