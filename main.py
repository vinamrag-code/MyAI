"""
main.py — FastAPI server for AI Chat and Web Search
====================================================
"""

import logging
import time
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import SearchAgent, FilteredAnswer, SearchResult
from ai_client_new import AIClient  # Import the new client
from config import Config

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Load config ────────────────────────────────────────────────────────────────
config = Config()

# ── Initialize components ──────────────────────────────────────────────────────
search_agent = SearchAgent()
ai_client = AIClient(config)  # No registry needed now

def safe_str(value, default=""):
    """Safely convert any value to string, handling None"""
    if value is None:
        return default
    return str(value)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 MyAI Server starting")
    logger.info(f"  SearXNG URL: {config.searxng_url}")
    
    await ai_client.initialize()
    
    yield
    
    logger.info("🛑 Shutting down")
    await ai_client.close()

app = FastAPI(
    title="MyAI",
    description="Your Personal AI Assistant with Memory + Web Search",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve static files ─────────────────────────────────────────────────────────
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Serve index.html at root ──────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the frontend HTML file"""
    static_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_path):
        return FileResponse(static_path)
    
    current_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(current_path):
        return FileResponse(current_path)
    
    return HTMLResponse(content=f"""
    <html>
        <head><title>MyAI</title></head>
        <body style="font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px;">
            <h1 style="color: #10a37f;">🤖 MyAI</h1>
            <div style="background: #f0f0f0; padding: 20px; border-radius: 10px;">
                <h2>Frontend file not found</h2>
                <p>Please create: <code>{static_path}</code></p>
                <p>API documentation: <a href="/docs">/docs</a></p>
            </div>
        </body>
    </html>
    """, status_code=200)

# ── API Routes ─────────────────────────────────────────────────────────────────

@app.get("/api/info", tags=["Info"])
def api_info():
    return {
        "name": "MyAI",
        "version": "1.0.0",
        "description": "Your Personal AI Assistant with Memory + Web Search",
        "models_available": len(ai_client.models),
        "endpoints": {
            "web_search": "GET /search?q=<query>",
            "ai_chat": "POST /api/chat",
            "health": "GET /health",
            "stats": "GET /stats",
        }
    }

@app.get("/health", tags=["Info"])
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "searxng_configured": bool(config.searxng_url),
        "models_available": len(ai_client.models)
    }

@app.get("/stats", tags=["Info"])
async def get_stats():
    """Get system statistics"""
    return {
        "timestamp": datetime.now().isoformat(),
        "ai_stats": ai_client.get_stats()
    }

@app.get("/api/models", tags=["AI Chat"])
def list_models():
    """List available AI models for the chat."""
    models = []
    for i, m in enumerate(ai_client.models):
        model_id = f"{m.provider}:{m.model_name}"
        display_name = m.model_name.split("/")[-1] if "/" in m.model_name else m.model_name
        models.append({
            "id": model_id,
            "index": i,
            "provider": m.provider,
            "model_name": m.model_name,
            "display_name": display_name,
            "is_free": m.is_free,
        })
    return {"models": models}

# ── Pydantic models ────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_results: int = Field(10, ge=1, le=50)

class SearchResultItem(BaseModel):
    title: str
    content: str
    url: str
    published: Optional[str] = ""
    source: str = ""

class SearchResponse(BaseModel):
    query: str
    total_results: int
    returned_results: int
    timestamp: str
    results: List[SearchResultItem]
    answer: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    user_id: int
    message: str
    ocr_context: Optional[str] = None
    model_id: Optional[str] = None  # e.g. "openrouter:meta-llama/llama-3.3-70b-instruct:free"

class ChatResponse(BaseModel):
    response: str
    user_id: int
    timestamp: str

class ClearConversationRequest(BaseModel):
    user_id: int

# ── Web Search Endpoints ───────────────────────────────────────────────────────

@app.get("/search", response_model=SearchResponse, tags=["Web Search"])
async def web_search(
    q: str = Query(..., min_length=1, description="Search query"),
    max_results: int = Query(10, ge=1, le=50, description="Maximum results")
):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    logger.info(f"📥 Web search request: '{q}'")
    
    try:
        results = await search_agent.search(q, max_results)
        best_answer = search_agent.get_best_answer(q, results)
        
        formatted_results = []
        for r in results[:max_results]:
            formatted_results.append(SearchResultItem(
                title=safe_str(r.title, "No title"),
                content=safe_str(r.content, "No content available")[:300] + "..." if len(safe_str(r.content)) > 300 else safe_str(r.content, "No content available"),
                url=safe_str(r.url, "#"),
                published=safe_str(r.published),
                source=safe_str(r.source, "Unknown")
            ))
        
        return SearchResponse(
            query=q,
            total_results=len(results),
            returned_results=len(formatted_results),
            timestamp=datetime.now().isoformat(),
            results=formatted_results,
            answer=best_answer.to_dict() if best_answer else None
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── AI Chat Endpoints ──────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse, tags=["AI Chat"])
async def ai_chat(request: ChatRequest):
    logger.info(f"📝 AI Chat - User {request.user_id}: {request.message[:50]}...")
    
    try:
        response = await ai_client.chat_with_memory(
            user_id=request.user_id,
            user_message=request.message,
            ocr_context=request.ocr_context,
            model_id=request.model_id
        )
        
        return ChatResponse(
            response=response,
            user_id=request.user_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"AI Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/clear")
async def clear_conversation(request: ClearConversationRequest):
    ai_client.clear_conversation(request.user_id)
    return {"status": "success", "message": "Conversation cleared"}

@app.get("/api/chat/history/{user_id}")
async def get_conversation_history(user_id: int):
    history = ai_client.get_conversation(user_id)
    return {"user_id": user_id, "history": history}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)