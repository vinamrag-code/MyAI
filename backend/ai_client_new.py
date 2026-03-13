# ai_client_new.py
import asyncio
import json
import time
import traceback
from typing import List, Dict, Optional, Any
import aiohttp
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ai_models import AIModel, get_available_models

logger = structlog.get_logger()

class NonRetryableError(Exception):
    """Permanent failure — do NOT retry"""
    pass

class RetryableError(Exception):
    """Temporary failure — OK to retry"""
    pass

class AIClient:
    def __init__(self, config):
        self.config = config
        self.models: List[AIModel] = get_available_models()
        self.session: Optional[aiohttp.ClientSession] = None
        self.model_stats: Dict[str, Dict[str, Any]] = {}  # Track model performance
        
        # Conversation memory
        self.conversations: Dict[int, List[Dict[str, str]]] = {}
        self.max_conversation_length = 20
        self.conversation_timeout = 1800  # 30 minutes
        self.conversation_timestamps: Dict[int, float] = {}

    async def initialize(self):
        timeout = aiohttp.ClientTimeout(total=60)
        self.session = aiohttp.ClientSession(timeout=timeout)
        logger.info("AIClient initialized", models_available=len(self.models))

    async def close(self):
        if self.session:
            await self.session.close()
            logger.info("AIClient session closed")

    # ============================================================
    # CONVERSATION MEMORY
    # ============================================================

    def get_conversation(self, user_id: int) -> List[Dict[str, str]]:
        now = time.time()
        last_time = self.conversation_timestamps.get(user_id, 0)

        if now - last_time > self.conversation_timeout:
            self.conversations.pop(user_id, None)
            self.conversation_timestamps.pop(user_id, None)
            return []

        return self.conversations.get(user_id, [])

    def add_to_conversation(self, user_id: int, role: str, content: str):
        if user_id not in self.conversations:
            self.conversations[user_id] = []

        self.conversations[user_id].append({"role": role, "content": content})
        self.conversation_timestamps[user_id] = time.time()

        if len(self.conversations[user_id]) > self.max_conversation_length:
            self.conversations[user_id] = self.conversations[user_id][-self.max_conversation_length:]

    def clear_conversation(self, user_id: int):
        self.conversations.pop(user_id, None)
        self.conversation_timestamps.pop(user_id, None)

    def build_messages(self, user_id: int, new_message: str, ocr_context: Optional[str] = None) -> List[Dict[str, str]]:
        messages = []

        system_prompt = (
            "You are MyAI, a helpful AI Assistant. "
            "CRITICAL RULES:\n"
            "1. You MUST respond ONLY in English or Hindi.\n"
            "2. NEVER include any Discord links, advertisements, URLs, or promotional content.\n"
            "3. Do NOT include discord.gg, discord.com, or any other invite links.\n"
            "4. Do NOT add any promotional text like 'Join our Discord' or 'Check out our server'.\n"
            "5. If the user uses Chinese characters, Japanese Kanji, or Korean Hangul, "
            "you MUST translate their query to English, answer in English/Hindi, "
            "but your final output text must be in English or Hindi only.\n"
            "6. Just answer the user's question directly and concisely without any marketing content."
        )
        messages.append({"role": "system", "content": system_prompt})

        history = self.get_conversation(user_id)
        messages.extend(history)

        if ocr_context:
            current_message = f"[OCR Content]\n{ocr_context}\n\n[User Query]\n{new_message}"
        else:
            current_message = new_message

        messages.append({"role": "user", "content": current_message})
        return messages

    # ============================================================
    # MODEL SELECTION AND FALLBACK
    # ============================================================

    async def chat_with_memory(
        self,
        user_id: int,
        user_message: str,
        ocr_context: Optional[str] = None,
    ) -> str:
        messages = self.build_messages(user_id, user_message, ocr_context)

        response = await self.chat_completion(
            messages=messages,
            user_id=user_id
        )

        self.add_to_conversation(user_id, "user", user_message)
        self.add_to_conversation(user_id, "assistant", response)
        return response

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        user_id: Optional[int] = None,
    ) -> str:
        if not self.models:
            raise RuntimeError("No available models configured")

        total_models = len(self.models)
        logger.info("Starting model fallback chain", total_models=total_models)

        attempt_count = 0
        last_error: Optional[Exception] = None
        failed_models: List[Dict[str, str]] = []

        for model in self.models:
            attempt_count += 1

            # Check if model is in cooldown
            model_key = f"{model.provider}:{model.model_name}"
            stats = self.model_stats.get(model_key, {})
            
            cooldown_until = stats.get('cooldown_until', 0)
            if cooldown_until > time.time():
                remaining = int(cooldown_until - time.time())
                logger.info(
                    "SKIPPING model — in cooldown",
                    position=f"{attempt_count}/{total_models}",
                    provider=model.provider,
                    model=model.model_name,
                    cooldown_remaining=remaining
                )
                failed_models.append({
                    "provider": model.provider,
                    "model": model.model_name,
                    "reason": f"cooldown ({remaining}s)"
                })
                continue

            logger.info(
                "ATTEMPTING model",
                position=f"{attempt_count}/{total_models}",
                provider=model.provider,
                model=model.model_name
            )

            try:
                start_time = time.time()
                result = await self._call_api(model, messages)
                response_time = time.time() - start_time

                # Update stats on success
                await self._update_model_stats(model_key, success=True, response_time=response_time)

                logger.info(
                    "SUCCESS",
                    provider=model.provider,
                    model=model.model_name,
                    response_time=f"{response_time:.2f}s"
                )
                return result

            except NonRetryableError as e:
                logger.warning(
                    "FAILED (non-retryable)",
                    provider=model.provider,
                    model=model.model_name,
                    error=str(e)
                )
                failed_models.append({
                    "provider": model.provider,
                    "model": model.model_name,
                    "reason": f"non_retryable: {str(e)[:100]}"
                })

            except RetryableError as e:
                logger.error(
                    "FAILED (retryable, cooling down)",
                    provider=model.provider,
                    model=model.model_name,
                    error=str(e)
                )
                # Put in cooldown
                await self._cooldown_model(model_key)
                failed_models.append({
                    "provider": model.provider,
                    "model": model.model_name,
                    "reason": f"retryable: {str(e)[:100]}"
                })

            except Exception as e:
                logger.error(
                    "FAILED (unexpected)",
                    provider=model.provider,
                    model=model.model_name,
                    error=str(e),
                    traceback=traceback.format_exc()
                )
                failed_models.append({
                    "provider": model.provider,
                    "model": model.model_name,
                    "reason": f"unexpected: {str(e)[:100]}"
                })

            # Brief pause before next model
            if attempt_count < total_models:
                await asyncio.sleep(0.5)

        # All models failed
        logger.error(
            "ALL MODELS EXHAUSTED",
            attempted=attempt_count,
            failed_models=failed_models
        )
        raise RuntimeError(
            f"All {total_models} models failed. Please try again later."
        )

    # ============================================================
    # API CALLS
    # ============================================================

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        retry=retry_if_exception_type(RetryableError),
        reraise=True
    )
    async def _call_api(self, model: AIModel, messages: List[Dict[str, str]]) -> str:
        if not self.session:
            raise RuntimeError("Session not initialized")

        api_key = model.get_api_key(self.config)
        
        headers = {
            "Content-Type": "application/json"
        }

        # Set authorization header based on provider
        if model.provider == "hermes":
            headers["Authorization"] = f"Bearer {api_key}"
        elif model.provider == "openrouter":
            headers["Authorization"] = f"Bearer {api_key}"
            headers["HTTP-Referer"] = "http://localhost:8000"
            headers["X-Title"] = "MyAI"
        elif model.provider == "groq":
            headers["Authorization"] = f"Bearer {api_key}"
        elif model.provider == "google":
            # Google uses API key in URL
            pass

        payload = {
            "model": model.model_name,
            "messages": messages,
            "temperature": model.temperature,
            "max_tokens": model.max_tokens
        }

        # Special handling for Google
        if model.provider == "google":
            url = f"{model.api_url}/{model.model_name}:generateContent?key={api_key}"
            # Convert to Google's format
            google_payload = {
                "contents": [
                    {
                        "parts": [{"text": m["content"]} for m in messages if m["role"] != "system"]
                    }
                ]
            }
            if messages and messages[0]["role"] == "system":
                google_payload["system_instruction"] = {
                    "parts": [{"text": messages[0]["content"]}]
                }
            payload = google_payload
        else:
            url = model.api_url

        try:
            async with self.session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=45)
            ) as response:
                status = response.status
                raw_body = await response.text()

                logger.debug(
                    "Response received",
                    provider=model.provider,
                    model=model.model_name,
                    status=status
                )

                if status == 429:
                    raise RetryableError(f"Rate limited: {raw_body[:200]}")
                
                if status >= 500:
                    raise RetryableError(f"Server error {status}: {raw_body[:200]}")
                
                if status != 200:
                    raise NonRetryableError(f"HTTP {status}: {raw_body[:200]}")

                # Parse response based on provider
                return await self._parse_response(model.provider, raw_body)

        except asyncio.TimeoutError:
            raise RetryableError("Request timeout")
        except aiohttp.ClientError as e:
            raise RetryableError(f"Connection error: {e}")

    async def _parse_response(self, provider: str, raw_body: str) -> str:
        """Parse API response based on provider"""
        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError as e:
            raise NonRetryableError(f"Invalid JSON: {e}")

        # Check for error in response
        if "error" in data:
            error_msg = str(data["error"])
            raise NonRetryableError(f"API error: {error_msg}")

        # Parse based on provider
        if provider == "google":
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError) as e:
                raise NonRetryableError(f"Unexpected Google response format: {e}")
        else:
            # OpenAI-compatible format
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as e:
                raise NonRetryableError(f"Unexpected response format: {e}")

    # ============================================================
    # MODEL STATISTICS AND COOLDOWN
    # ============================================================

    async def _update_model_stats(self, model_key: str, success: bool, response_time: float = None):
        """Update model performance statistics"""
        if model_key not in self.model_stats:
            self.model_stats[model_key] = {
                "success_count": 0,
                "failure_count": 0,
                "avg_response_time": 0,
                "cooldown_until": 0
            }
        
        stats = self.model_stats[model_key]
        
        if success:
            stats["success_count"] += 1
            if response_time:
                # Exponential moving average
                alpha = 0.3
                if stats["avg_response_time"] == 0:
                    stats["avg_response_time"] = response_time
                else:
                    stats["avg_response_time"] = alpha * response_time + (1 - alpha) * stats["avg_response_time"]
        else:
            stats["failure_count"] += 1

    async def _cooldown_model(self, model_key: str, duration: int = 60):
        """Put a model in cooldown"""
        if model_key not in self.model_stats:
            self.model_stats[model_key] = {}
        
        cooldown_time = time.time() + duration
        self.model_stats[model_key]["cooldown_until"] = cooldown_time
        logger.info("Model in cooldown", model=model_key, duration=duration)

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        now = time.time()
        stats = []
        
        for model in self.models:
            model_key = f"{model.provider}:{model.model_name}"
            model_stat = self.model_stats.get(model_key, {})
            
            in_cooldown = model_stat.get('cooldown_until', 0) > now
            cooldown_remaining = int(model_stat.get('cooldown_until', 0) - now) if in_cooldown else 0
            
            stats.append({
                "provider": model.provider,
                "model": model.model_name,
                "is_free": model.is_free,
                "success_count": model_stat.get('success_count', 0),
                "failure_count": model_stat.get('failure_count', 0),
                "avg_response_time": f"{model_stat.get('avg_response_time', 0):.2f}s" if model_stat.get('avg_response_time') else "N/A",
                "in_cooldown": in_cooldown,
                "cooldown_remaining": cooldown_remaining
            })
        
        return {
            "total_models": len(self.models),
            "active_conversations": len(self.conversations),
            "model_stats": stats
        }