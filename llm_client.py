from __future__ import annotations

"""
Gemini LLM client with rate limiting, retry logic, and daily quota tracking.

Uses google-genai SDK (new API, not the legacy google-generativeai).
"""

import time
import logging
from datetime import date, datetime
from threading import Lock

from google import genai
from google.genai import types as genai_types

from config import (
    GEMINI_API_KEY,
    MODEL_NAME,
    RATE_LIMIT,
    TEMP_ANALYSIS,
    MAX_TOKENS_FULL,
)

logger = logging.getLogger(__name__)


class DailyQuotaTracker:
    """Thread-safe tracker for daily API request count."""

    def __init__(self, daily_limit: int, warn_threshold: int):
        self.daily_limit = daily_limit
        self.warn_threshold = warn_threshold
        self._count = 0
        self._date = date.today()
        self._lock = Lock()

    def increment(self) -> int:
        with self._lock:
            today = date.today()
            if today != self._date:
                self._count = 0
                self._date = today
            self._count += 1
            count = self._count

        if count >= self.daily_limit:
            raise RuntimeError(
                f"Daily API limit of {self.daily_limit} requests reached. "
                "Try again tomorrow or upgrade to a paid tier."
            )
        if count >= self.warn_threshold:
            logger.warning(
                "Approaching daily API limit: %d/%d requests used today.",
                count,
                self.daily_limit,
            )
        return count

    @property
    def count(self) -> int:
        with self._lock:
            return self._count


class RateLimiter:
    """Enforces a minimum delay between successive API calls."""

    def __init__(self, delay_seconds: float):
        self.delay = delay_seconds
        self._last_call: float = 0.0
        self._lock = Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self.delay:
                sleep_time = self.delay - elapsed
                logger.debug("Rate limiter sleeping %.1f seconds.", sleep_time)
                time.sleep(sleep_time)
            self._last_call = time.monotonic()


class GeminiClient:
    """
    Synchronous Gemini LLM client with:
    - Per-call rate limiting (8-second floor between requests)
    - Exponential backoff retry on 429 quota errors
    - Daily request quota tracking
    - Convenience method for analysis + compressed briefing in one call
    """

    BRIEFING_MARKER = "---BRIEFING---"

    def __init__(
        self,
        api_key: str = GEMINI_API_KEY,
        model_name: str = MODEL_NAME,
    ):
        if not api_key:
            raise ValueError(
                "Gemini API key is required. Set the GEMINI_API_KEY environment "
                "variable. Get a free key at https://aistudio.google.com/apikey"
            )
        self.model_name = model_name
        self._client = genai.Client(api_key=api_key)
        self._rate_limiter = RateLimiter(RATE_LIMIT["delay_between_requests"])
        self._quota = DailyQuotaTracker(
            daily_limit=RATE_LIMIT["daily_request_limit"],
            warn_threshold=RATE_LIMIT["daily_warn_threshold"],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        temperature: float = TEMP_ANALYSIS,
        max_tokens: int = MAX_TOKENS_FULL,
    ) -> str:
        """
        Generate a response for the given prompt.

        Parameters
        ----------
        prompt : str
            The full prompt to send to the model.
        temperature : float
            Sampling temperature (0.0–1.0).
        max_tokens : int
            Maximum output tokens.

        Returns
        -------
        str
            The model's text response.
        """
        return self._call_with_retry(prompt, temperature, max_tokens)

    def generate_with_briefing(
        self,
        prompt: str,
        temperature: float = TEMP_ANALYSIS,
        max_tokens: int = MAX_TOKENS_FULL,
    ) -> tuple[str, str]:
        """
        Generate a full analysis and a compressed briefing in a single call.

        The prompt is automatically augmented with an instruction to produce a
        600-800 word briefing after the ``---BRIEFING---`` marker.

        Parameters
        ----------
        prompt : str
            Base analysis prompt (should NOT already include the briefing instruction).
        temperature : float
            Sampling temperature.
        max_tokens : int
            Maximum output tokens (should be large enough for both sections).

        Returns
        -------
        tuple[str, str]
            (full_output, briefing)  — briefing may be empty if the model did not
            include the marker.
        """
        augmented_prompt = self._append_briefing_instruction(prompt)
        raw = self._call_with_retry(augmented_prompt, temperature, max_tokens)
        full_output, briefing = self._split_briefing(raw)
        return full_output, briefing

    @property
    def requests_today(self) -> int:
        """Number of API calls made today."""
        return self._quota.count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_retry(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Execute an API call with rate limiting and exponential backoff."""
        max_attempts = RATE_LIMIT["retry_max_attempts"]
        initial_delay = RATE_LIMIT["retry_initial_delay"]
        backoff = RATE_LIMIT["retry_backoff_factor"]

        for attempt in range(1, max_attempts + 1):
            # Enforce rate limit before every call
            self._rate_limiter.wait()
            # Check / increment daily quota
            req_num = self._quota.increment()
            logger.debug(
                "API call #%d today | attempt %d/%d | model=%s",
                req_num,
                attempt,
                max_attempts,
                self.model_name,
            )

            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                return response.text

            except Exception as exc:
                exc_str = str(exc)
                is_quota_error = "429" in exc_str or "quota" in exc_str.lower() or "RESOURCE_EXHAUSTED" in exc_str

                if is_quota_error and attempt < max_attempts:
                    wait = initial_delay * (backoff ** (attempt - 1))
                    logger.warning(
                        "Quota/rate-limit error (attempt %d/%d). "
                        "Waiting %.0f seconds before retry. Error: %s",
                        attempt,
                        max_attempts,
                        wait,
                        exc_str[:200],
                    )
                    # Decrement quota counter since the call didn't succeed
                    with self._quota._lock:
                        self._quota._count -= 1
                    time.sleep(wait)
                    continue

                # Non-quota error or last attempt — re-raise
                logger.error(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt,
                    max_attempts,
                    exc_str[:500],
                )
                raise

        # Should be unreachable
        raise RuntimeError("LLM call failed after all retry attempts.")

    @staticmethod
    def _append_briefing_instruction(prompt: str) -> str:
        instruction = (
            "\n\n"
            "---\n"
            "IMPORTANT FINAL INSTRUCTION:\n"
            "After completing your full analysis above, output the exact marker "
            "\"---BRIEFING---\" on its own line, then write a compressed briefing "
            "of 600-800 words that:\n"
            "1. Captures ALL key data points with specific numbers (no vague language).\n"
            "2. Includes every score/rating you assigned with its rationale.\n"
            "3. States your conclusions clearly and directly.\n"
            "4. Is written so that a downstream AI agent can use it as a complete "
            "substitute for your full analysis without losing critical information.\n"
            "5. Uses tight, dense prose — no filler, no repetition.\n"
            "The briefing is the single most important output: downstream agents "
            "depend on it entirely."
        )
        return prompt + instruction

    @staticmethod
    def _split_briefing(raw: str) -> tuple[str, str]:
        marker = "---BRIEFING---"
        if marker in raw:
            parts = raw.split(marker, 1)
            return parts[0].strip(), parts[1].strip()
        # Fallback: return everything as full output, empty briefing
        logger.warning(
            "Model did not include the ---BRIEFING--- marker. "
            "Briefing will be empty."
        )
        return raw.strip(), ""


# ---------------------------------------------------------------------------
# Module-level singleton (lazy-initialized)
# ---------------------------------------------------------------------------
_client_instance: GeminiClient | None = None


def get_client() -> GeminiClient:
    """Return the module-level GeminiClient singleton."""
    global _client_instance
    if _client_instance is None:
        _client_instance = GeminiClient()
    return _client_instance
