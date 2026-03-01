#!/usr/bin/env python3
"""
Agent helpers: intent detection, parameter extraction, and date parsing.

This module provides:
- ``Intent`` enum for classifying user commands
- ``InferenceParams`` / ``PendingInference`` typed dataclasses
- ``LLMHelper`` that wraps LocalLLMClient for multi-agent prompting
- ``DateParser`` minimal regex fallback for ISO dates
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from llm_client import EASY_DIR, LocalLLMClient

# ── Prompt loading ──────────────────────────────────────────────────────────

PROMPTS_DIR = EASY_DIR / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system_prompt.txt"


def _load_prompt(name: str) -> str:
    """Load a prompt from the prompts/ directory by filename."""
    path = PROMPTS_DIR / name
    if path.exists():
        return path.read_text().strip()
    raise FileNotFoundError(f"Prompt file not found: {path}")


# ── Enums & dataclasses ────────────────────────────────────────────────────

class Intent(Enum):
    """Classified user intent."""
    INFERENCE = "INFERENCE"
    ANALYSIS = "ANALYSIS"
    QUESTION = "QUESTION"
    CHAT = "CHAT"


@dataclass
class InferenceParams:
    """Structured parameters extracted from a natural-language inference request."""
    start_datetime: str | None = None
    rollout_steps: int | None = None
    missing: list[str] = field(default_factory=list)
    understood: str = ""

    @property
    def is_complete(self) -> bool:
        """True when all required information is present."""
        return self.start_datetime is not None and len(self.missing) == 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InferenceParams:
        """Build from an LLM-extracted dict, tolerating extra/missing keys."""
        return cls(
            start_datetime=d.get("start_datetime"),
            rollout_steps=d.get("rollout_steps"),
            missing=d.get("missing", []),
            understood=d.get("understood", ""),
        )


@dataclass
class PendingInference:
    """State for a multi-turn inference conversation awaiting more info."""
    original_cmd: str
    params: InferenceParams
    dry_run: bool = False
    skip_download: bool = False


# ── JSON extraction helper ──────────────────────────────────────────────────

def _extract_json_object(text: str) -> str | None:
    """Extract the first top-level JSON object from *text* using brace counting.

    Unlike a simple ``\\{[^{}]*\\}`` regex this handles nested braces such as
    ``{"missing": ["date", "year"]}``.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# ── LLM helper (agent prompting) ───────────────────────────────────────────

class LLMHelper:
    """Use LLM for natural language understanding and responses."""

    def __init__(self, llm: LocalLLMClient):
        self.llm = llm
        self._system_prompt = _load_prompt("system_prompt.txt")
        self._intent_prompt = _load_prompt("intent_agent.txt")
        self._params_prompt = _load_prompt("params_agent.txt")
        self._followup_prompt = _load_prompt("followup_agent.txt")
        self._response_prompt = _load_prompt("response_agent.txt")

    def ask(
        self,
        user_input: str,
        stream: bool = False,
        history: list[tuple[str, str]] | None = None,
    ) -> str | None:
        """Get a natural language response from the LLM."""
        if not self.llm.available:
            return None
        return self.llm.generate(
            user_input, self._system_prompt, stream=stream, history=history,
            agent="Response Agent", final_color="\033[32m",
        )

    def detect_intent(self, user_input: str) -> Intent:
        """Detect user intent using LLM with visible streaming output."""
        if not self.llm.available:
            return Intent.QUESTION

        prompt = f'{self._intent_prompt} "{user_input}"'
        response = self.llm.generate(
            prompt, "", stream=True, agent="Intent Agent", final_color="\033[37m",
        )

        if not response:
            return Intent.QUESTION

        last_line = response.strip().split("\n")[-1].strip()
        token = last_line.split()[0].upper().strip(".:,\"'()") if last_line.split() else ""
        try:
            return Intent(token)
        except ValueError:
            return Intent.QUESTION

    def extract_params(self, user_input: str, max_retries: int = 2) -> InferenceParams | None:
        """Extract structured parameters from natural language using LLM.

        Returns ``InferenceParams`` on success, ``None`` if extraction fails.
        """
        if not self.llm.available:
            return None

        for attempt in range(max_retries):
            prompt = f'{self._params_prompt}\nQuery: "{user_input}"'

            if attempt > 0:
                prompt += "\n\nIMPORTANT: Output ONLY valid JSON."

            response = self.llm.generate(
                prompt, "", stream=True, agent="Params Agent", final_color="\033[37m",
            )

            if not response:
                continue

            try:
                cleaned = response.strip()
                cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
                cleaned = re.sub(r'\s*```$', '', cleaned)

                json_str = _extract_json_object(cleaned)
                if json_str:
                    result = json.loads(json_str)
                    return InferenceParams.from_dict(result)
                else:
                    result = json.loads(cleaned)
                    return InferenceParams.from_dict(result)
            except json.JSONDecodeError:
                if attempt == max_retries - 1:
                    print(f"  [LLM JSON parse failed: {response[:80]}...]")

        return None

    def generate_followup(self, query: str, understood: str, missing: list[str]) -> str | None:
        """Generate a follow-up question for missing information."""
        if not self.llm.available:
            return None

        prompt = self._followup_prompt.format(
            query=query,
            understood=understood,
            missing=", ".join(missing),
        )
        return self.llm.generate(
            prompt, "", stream=True, agent="Follow-up Agent", final_color="\033[32m",
        )

    def generate_response(self, date: str, rollout: int) -> str | None:
        """Generate a friendly response explaining what will happen."""
        if not self.llm.available:
            return None

        prompt = self._response_prompt.format(date=date, rollout=rollout)
        return self.llm.generate(
            prompt, "", agent="Response Agent", final_color="\033[32m",
        )


# ── Date parsing fallback ──────────────────────────────────────────────────

class DateParser:
    """Minimal fallback parser for when LLM is unavailable. Only handles ISO dates."""

    @classmethod
    def parse(cls, text: str) -> tuple[str | None, str | None]:
        """Parse ISO format date only (YYYY-MM-DD). Use LLM for natural language."""
        iso_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', text)
        if iso_match:
            year, month, day = iso_match.group(1), iso_match.group(2), iso_match.group(3)
            return f"{year}-{month}-{day} 10:00", f"{year}-{month}-{day} 17:00"
        return None, None

    @classmethod
    def parse_rollout(cls, text: str) -> int | None:
        """Parse numeric rollout steps only. Use LLM for natural language."""
        match = re.search(r'(\d+)\s*(?:rollout\s*)?steps?', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r'(\d+)\s*(?:hour|hr)s?\s*ahead', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None
