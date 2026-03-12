"""
agent.py — 3-Layer Fallback AI Agent
=====================================
Layer 1 → Your Custom Chatbot API        (primary — plug in your curl later)
Layer 2 → SearXNG self-hosted search     (your server: 140.238.166.109:8081)
           + Gemini synthesises results into a clean answer
Layer 3 → Gemini 2.0 Flash direct        (final safety net)

Usage:
    from agent import FallbackAgent
    agent = FallbackAgent()
    result = await agent.ask("What is the capital of France?", session_id="user_123")
"""

import os
import re
import json
import logging
import asyncio
import httpx
from typing import Optional
from urllib.parse import urlencode
from dotenv import load_dotenv

# LangChain
from langchain-google-genai import ChatGoogleGenerativeAI  # This is correct!
from langchain.memory import ConversationBufferWindowMemory

load_dotenv()
logger = logging.getLogger(__name__)

# ── Env ────────────────────────────────────────────────────────────────────────
CUSTOM_BOT_URL              = os.getenv("CUSTOM_BOT_URL", "")
CUSTOM_BOT_API_KEY          = os.getenv("CUSTOM_BOT_API_KEY", "")
CUSTOM_BOT_HEADERS          = os.getenv("CUSTOM_BOT_HEADERS", "{}")
CUSTOM_BOT_PAYLOAD_TEMPLATE = os.getenv("CUSTOM_BOT_PAYLOAD_TEMPLATE", "")

# Your SearXNG instance
SEARXNG_URL      = os.getenv("SEARXNG_URL", "http://140.238.166.109:8081/search")
SEARXNG_LANGUAGE = os.getenv("SEARXNG_LANGUAGE", "en")
SEARXNG_CATEGORIES = os.getenv("SEARXNG_CATEGORIES", "general,news")  # comma-separated

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "12"))

# ── Fallback trigger conditions ────────────────────────────────────────────────
# Layer 1 responses matching these patterns are treated as non-answers → fallback
WEAK_RESPONSE_PATTERNS = [
    r"i (don'?t|do not) know",
    r"i('m| am) not sure",
    r"i (can'?t|cannot) (answer|help|find|assist)",
    r"no (information|data|result)s? (available|found)",
    r"sorry,? i",
    r"i don'?t have (access|information|data)",
    r"outside (my|the) (scope|knowledge|training)",
    r"^\s*$",            # empty
    r"^.{0,15}$",        # extremely short (< 15 chars)
    r"\b(error|exception|failed|timed? ?out)\b",
]

def _is_weak(text: str) -> bool:
    """Return True if the response looks like a non-answer that should trigger fallback."""
    t = text.strip().lower()
    return any(re.search(p, t) for p in WEAK_RESPONSE_PATTERNS)


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 1 — Custom Chatbot API  (add your curl details in .env)
# ══════════════════════════════════════════════════════════════════════════════
class CustomBotLayer:
    """
    Calls your existing deployed chatbot.

    HOW TO CONFIGURE — map your curl command to .env:
    ─────────────────────────────────────────────────
    curl -X POST https://your-bot.com/api/chat
      -H "Authorization: Bearer TOKEN"
      -d '{"query": "hello"}'

    →  CUSTOM_BOT_URL     = https://your-bot.com/api/chat
       CUSTOM_BOT_API_KEY = TOKEN
       CUSTOM_BOT_PAYLOAD_TEMPLATE = {"query": "{message}"}

    Placeholders: {message} → user question, {session_id} → session id

    Auto-detects the answer field from common response shapes:
      { "answer"|"response"|"text"|"message"|"output"|"result"|"content"|"reply" }
    """

    async def ask(self, message: str, session_id: str = "default") -> Optional[str]:
        if not CUSTOM_BOT_URL:
            logger.info("Layer 1: CUSTOM_BOT_URL not set — skipping")
            return None

        # Build headers
        headers = {"Content-Type": "application/json"}
        if CUSTOM_BOT_API_KEY:
            headers["Authorization"] = f"Bearer {CUSTOM_BOT_API_KEY}"
        try:
            headers.update(json.loads(CUSTOM_BOT_HEADERS))
        except Exception:
            pass

        # Build payload from template or default
        if CUSTOM_BOT_PAYLOAD_TEMPLATE:
            raw = (CUSTOM_BOT_PAYLOAD_TEMPLATE
                   .replace("{message}", message)
                   .replace("{session_id}", session_id))
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"message": message}
        else:
            payload = {"message": message, "session_id": session_id}

        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                resp = await client.post(CUSTOM_BOT_URL, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            # Auto-detect answer field
            for field in ("answer","response","text","message","output","result","content","reply"):
                if field in data and isinstance(data[field], str) and data[field].strip():
                    return data[field].strip()

            if isinstance(data, str) and data.strip():
                return data.strip()

            logger.warning("Layer 1: unrecognised response shape — keys: %s", list(data.keys()))
            return None

        except httpx.TimeoutException:
            logger.warning("Layer 1: timed out after %ds", REQUEST_TIMEOUT)
            return None
        except Exception as e:
            logger.warning("Layer 1: error — %s", e)
            return None


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — SearXNG Search  +  Gemini synthesis (CORRECTED)
# ══════════════════════════════════════════════════════════════════════════════
class SearXNGLayer:
    """
    Queries your self-hosted SearXNG instance at 140.238.166.109:8081,
    then uses Gemini 2.0 Flash to synthesise the raw results into a clean answer.

    Mirrors exactly this curl:
      curl -X POST "http://140.238.166.109:8081/search" \\
        -d "q=<query>&format=json&categories=news,general&language=en"

    The results array contains: title, content, url, publishedDate
    Gemini synthesises these into a natural language answer.
    """

    def __init__(self):
        self._llm: Optional[ChatGoogleGenerativeAI] = None
        if GEMINI_API_KEY:
            self._llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=GEMINI_API_KEY,
                temperature=0.3,
                convert_system_message_to_human=True,
            )

    async def _search(self, query: str) -> list[dict]:
        """
        POST to SearXNG with form-encoded body — exactly matching your curl.
        Returns list of {title, content, url, publishedDate} dicts.
        """
        params = {
            "q":          query,
            "format":     "json",
            "categories": SEARXNG_CATEGORIES,
            "language":   SEARXNG_LANGUAGE,
        }

        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                resp = await client.post(
                    SEARXNG_URL,
                    data=params,  # Using data= instead of content= for proper form encoding
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                resp.raise_for_status()
                data = resp.json()

            results = data.get("results", [])
            # Extract only the fields we need
            clean = []
            for r in results[:8]:   # top 8 results
                entry = {
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", ""),
                    "published": r.get("publishedDate", ""),
                }
                # Only include if there's actual content
                if entry["title"] or entry["content"]:
                    clean.append(entry)
            return clean

        except httpx.TimeoutException:
            logger.warning("Layer 2 SearXNG: timed out after %ds", REQUEST_TIMEOUT)
            return []
        except Exception as e:
            logger.warning("Layer 2 SearXNG: search error — %s", e)
            return []

    def _format_results_for_llm(self, results: list[dict]) -> str:
        """Format search results into a readable block for Gemini."""
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r['title']}")
            if r.get("published"):
                # Format the published date nicely
                pub_date = r['published']
                if len(pub_date) > 10:  # If it's a full timestamp
                    pub_date = pub_date[:10]  # Just take YYYY-MM-DD
                lines.append(f"    Published: {pub_date}")
            if r.get("content"):
                # Truncate content if too long
                content = r['content']
                if len(content) > 400:
                    content = content[:400] + "..."
                lines.append(f"    {content}")
            if r.get("url"):
                lines.append(f"    Source: {r['url']}")
            lines.append("")
        return "\n".join(lines)

    async def ask(self, message: str) -> Optional[str]:
        if not SEARXNG_URL:
            logger.info("Layer 2: SEARXNG_URL not set — skipping")
            return None
        if not self._llm:
            logger.info("Layer 2: GEMINI_API_KEY not set — cannot synthesise — skipping")
            return None

        logger.info("Layer 2: searching SearXNG for: %s", message)
        results = await self._search(message)

        if not results:
            logger.warning("Layer 2: SearXNG returned no results")
            return None

        formatted = self._format_results_for_llm(results)
        logger.info("Layer 2: got %d results, synthesising with Gemini…", len(results))

        synthesis_prompt = f"""You are a helpful AI assistant. A user asked:
"{message}"

Here are live search results retrieved for this question:
─────────────────────────────────────────────
{formatted}
─────────────────────────────────────────────

Using ONLY the information in the search results above, write a clear, accurate, and helpful answer.
- Be concise but complete.
- If the results include recent dates or news, mention that.
- Do NOT say "based on the search results" or "I searched" — just answer naturally.
- If the results don't contain enough relevant info, say: "I couldn't find a reliable answer to that."

Provide a direct, factual answer to the user's question."""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self._llm.invoke, synthesis_prompt)
            answer = response.content.strip()
            return answer if answer else None
        except Exception as e:
            logger.warning("Layer 2: Gemini synthesis error — %s", e)
            return None


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 3 — Gemini 2.0 Flash direct  (final safety net)
# ══════════════════════════════════════════════════════════════════════════════
class GeminiLayer:
    """Direct Gemini 2.0 Flash call with conversation history. Always answers."""

    def __init__(self):
        self._llm: Optional[ChatGoogleGenerativeAI] = None
        if GEMINI_API_KEY:
            self._llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=GEMINI_API_KEY,
                temperature=0.7,
                convert_system_message_to_human=True,
            )

    async def ask(self, message: str, history: str = "") -> Optional[str]:
        if not self._llm:
            logger.warning("Layer 3: GEMINI_API_KEY not set — cannot answer")
            return None
        try:
            system = "You are a helpful, knowledgeable AI assistant. Answer clearly and concisely."
            prompt = f"{system}\n\n{history}\nUser: {message}\nAssistant:" if history else f"{system}\n\nUser: {message}\nAssistant:"
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self._llm.invoke, prompt)
            return response.content.strip() or None
        except Exception as e:
            logger.warning("Layer 3: error — %s", e)
            return None


# ══════════════════════════════════════════════════════════════════════════════
#  FALLBACK AGENT — orchestrates all 3 layers
# ══════════════════════════════════════════════════════════════════════════════
class FallbackAgent:
    """
    Tries Layer 1 → Layer 2 → Layer 3 in order.
    Returns the first satisfactory answer with metadata about which layer responded.
    """

    def __init__(self):
        self.layer1 = CustomBotLayer()
        self.layer2 = SearXNGLayer()
        self.layer3 = GeminiLayer()
        self._sessions: dict[str, ConversationBufferWindowMemory] = {}

    def _get_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationBufferWindowMemory(
                k=8, memory_key="history", human_prefix="User", ai_prefix="Assistant"
            )
        return self._sessions[session_id]

    def clear_session(self, session_id: str):
        self._sessions.pop(session_id, None)

    async def ask(
        self,
        message: str,
        session_id: str = "default",
        use_memory: bool = True,
    ) -> dict:
        """
        Returns:
          {
            "answer":     str,
            "layer":      1 | 2 | 3 | 0,
            "layer_name": "Custom Bot" | "SearXNG Search" | "Gemini" | "None",
            "session_id": str,
          }
        """
        memory  = self._get_memory(session_id)
        history = memory.load_memory_variables({}).get("history", "") if use_memory else ""

        answer     = None
        layer_used = None
        layer_name = None

        # ── Layer 1: Custom Chatbot ────────────────────────────────────────────
        logger.info("[%s] Trying Layer 1 (Custom Bot)…", session_id)
        l1 = await self.layer1.ask(message, session_id)
        if l1 and not _is_weak(l1):
            answer, layer_used, layer_name = l1, 1, "Custom Bot"
            logger.info("[%s] ✅ Layer 1 answered", session_id)

        # ── Layer 2: SearXNG + Gemini synthesis ────────────────────────────────
        if not answer:
            logger.info("[%s] → Falling back to Layer 2 (SearXNG Search)…", session_id)
            l2 = await self.layer2.ask(message)
            if l2 and not _is_weak(l2):
                answer, layer_used, layer_name = l2, 2, "SearXNG Search"
                logger.info("[%s] ✅ Layer 2 answered", session_id)

        # ── Layer 3: Gemini direct ─────────────────────────────────────────────
        if not answer:
            logger.info("[%s] → Falling back to Layer 3 (Gemini direct)…", session_id)
            l3 = await self.layer3.ask(message, history)
            if l3:
                answer, layer_used, layer_name = l3, 3, "Gemini"
                logger.info("[%s] ✅ Layer 3 answered", session_id)

        # ── All failed ─────────────────────────────────────────────────────────
        if not answer:
            answer = "I'm sorry, I wasn't able to find an answer to that right now. Please try again."
            layer_used, layer_name = 0, "None"
            logger.warning("[%s] ❌ All 3 layers failed", session_id)

        # Persist to session memory
        if use_memory and layer_used:
            memory.save_context({"input": message}, {"output": answer})

        return {
            "answer":     answer,
            "layer":      layer_used,
            "layer_name": layer_name,
            "session_id": session_id,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Example usage
# ══════════════════════════════════════════════════════════════════════════════
async def example():
    """Example of how to use the agent."""
    agent = FallbackAgent()
    
    # Test with your query
    result = await agent.ask("who won t20 wc 2026", session_id="test_user")
    
    print(f"Answer: {result['answer']}")
    print(f"Layer used: {result['layer_name']} (Layer {result['layer']})")
    print(f"Session ID: {result['session_id']}")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    asyncio.run(example())