#!/usr/bin/env python3
"""
Surya Easy Inference - Natural Language Interface with GPT-OSS

A terminal interface for running solar predictions and analysis
using natural language. Uses GPT-OSS on GPU with Harmony format.

Usage:
    python llm_interface.py                # Default (GPT-OSS-120B)
    python llm_interface.py --model gpt-oss-20b  # GPT-OSS 20B (single GPU)
    python llm_interface.py --no-llm       # Disable LLM (regex only)

Commands:
    Natural language queries like:
    - "What can you do?"
    - "Run prediction for october 23 2014"
    - "Analyze all channels and rank by MSE"
    - "Which channel is best for solar flares?"
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

# Constants
ROOT_DIR = Path(__file__).resolve().parent.parent
EASY_DIR = ROOT_DIR / "easy_inference"
CONFIG_PATH = EASY_DIR / "config_easy.yaml"
LOG_DIR = EASY_DIR / "logs"
SCOPE_PATH = EASY_DIR / "SCOPE.md"

CHANNELS = {
    "aia": ["aia94", "aia131", "aia171", "aia193", "aia211", "aia304", "aia335", "aia1600"],
    "hmi": ["hmi_m", "hmi_bx", "hmi_by", "hmi_bz", "hmi_v"],
}
ALL_CHANNELS = CHANNELS["aia"] + CHANNELS["hmi"]

# Channel descriptions for LLM context
CHANNEL_INFO = {
    "aia94": "94A - Hot flare plasma (~6.3 MK), best for solar flares",
    "aia131": "131A - Flare plasma + cooler regions",
    "aia171": "171A - Quiet corona (~0.6 MK), good for coronal loops",
    "aia193": "193A - Corona and hot flares, general purpose",
    "aia211": "211A - Active regions (~2 MK)",
    "aia304": "304A - Chromosphere (~50,000 K), shows prominences",
    "aia335": "335A - Active regions (~2.5 MK)",
    "aia1600": "1600A - Upper photosphere, UV continuum",
    "hmi_m": "Total magnetic field magnitude",
    "hmi_bx": "Magnetic field X component",
    "hmi_by": "Magnetic field Y component",
    "hmi_bz": "Magnetic field Z component (vertical)",
    "hmi_v": "Line-of-sight velocity (Doppler)",
}


class SessionLogger:
    """Log all commands and results to a JSON-lines file."""

    def __init__(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = LOG_DIR / f"session_{self.session_id}.log"
        self.log("session_start", {"cwd": str(ROOT_DIR)})

    def log(self, event: str, data: dict[str, Any] | None = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            **(data or {}),
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_log_path(self) -> str:
        return str(self.log_file)


AVAILABLE_MODELS = [
    {"name": "GPT-OSS-20B",  "hf_id": "openai/gpt-oss-20b",  "reasoning": "low",    "desc": "Fast, single GPU"},
    {"name": "GPT-OSS-20B",  "hf_id": "openai/gpt-oss-20b",  "reasoning": "medium", "desc": "Balanced, single GPU"},
    {"name": "GPT-OSS-20B",  "hf_id": "openai/gpt-oss-20b",  "reasoning": "high",   "desc": "Deep reasoning, single GPU"},
    {"name": "GPT-OSS-120B", "hf_id": "openai/gpt-oss-120b", "reasoning": "low",    "desc": "Fast, minimal thinking"},
    {"name": "GPT-OSS-120B", "hf_id": "openai/gpt-oss-120b", "reasoning": "medium", "desc": "Balanced"},
    {"name": "GPT-OSS-120B", "hf_id": "openai/gpt-oss-120b", "reasoning": "high",   "desc": "Deep reasoning"},
]


def _is_model_cached(hf_id: str) -> bool:
    """Check if a HuggingFace model is already downloaded locally."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / f"models--{hf_id.replace('/', '--')}"
    if not model_dir.exists():
        return False
    snapshots = model_dir / "snapshots"
    return snapshots.exists() and any(snapshots.iterdir())


def select_model() -> tuple[str, str] | None:
    """Show interactive model selection menu. Returns (hf_id, reasoning) or None."""
    print("\n  Available Modes:")
    print("  " + "-" * 70)
    print(f"  {'#':>3}  {'Model':<14} {'Reasoning':<10} {'Description':<24} {'Cached':<8}")
    print("  " + "-" * 70)

    for i, m in enumerate(AVAILABLE_MODELS, 1):
        cached = "yes" if _is_model_cached(m["hf_id"]) else ""
        default = " (default)" if i == 1 else ""
        print(f"  {i:>3}  {m['name']:<14} {m['reasoning']:<10} {m['desc']:<24} {cached:<8}{default}")

    print("  " + "-" * 70)
    print("    0 / q  = No LLM (regex mode)")
    print()

    while True:
        try:
            choice = input("  Select mode [1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if choice == "":
            sel = AVAILABLE_MODELS[0]  # default: GPT-OSS-20B low
            print(f"  -> {sel['name']} ({sel['reasoning']})")
            return sel["hf_id"], sel["reasoning"]
        if choice in ("0", "q", "Q"):
            return None
        try:
            idx = int(choice)
            if 1 <= idx <= len(AVAILABLE_MODELS):
                sel = AVAILABLE_MODELS[idx - 1]
                print(f"  -> {sel['name']}")
                return sel["hf_id"], sel["reasoning"]
            else:
                print(f"  Invalid choice. Enter 1-{len(AVAILABLE_MODELS)}, 0, or q.")
        except ValueError:
            print(f"  Invalid input. Enter a number 1-{len(AVAILABLE_MODELS)}, 0, or q.")


class LocalLLMClient:
    """Client for GPT-OSS on GPU with Harmony format parsing via openai_harmony."""

    DEFAULT_MODEL = "openai/gpt-oss-120b"

    MODEL_ALIASES = {
        "gpt-oss": "openai/gpt-oss-120b",
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "gpt-oss-20b": "openai/gpt-oss-20b",
    }

    REASONING_LEVELS = ("low", "medium", "high")

    def __init__(self, model: str | None = None, reasoning: str = "medium"):
        resolved = self.MODEL_ALIASES.get(model, model) if model else None
        self.model = resolved or self.DEFAULT_MODEL
        self.reasoning = reasoning if reasoning in self.REASONING_LEVELS else "medium"
        self.available = False
        self.error_message: str | None = None
        self._tokenizer = None
        self._model = None
        self._harmony_enc = None
        self._load_model()

    def _load_model(self):
        """Load model, tokenizer, and Harmony encoding."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading {self.model}...", end=" ", flush=True)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model,
                dtype=torch.bfloat16,
                device_map="auto",
            )
            self._model.eval()

            from openai_harmony import HarmonyEncodingName, load_harmony_encoding
            self._harmony_enc = load_harmony_encoding(
                HarmonyEncodingName.HARMONY_GPT_OSS
            )

            self.available = True
            print("OK")

        except Exception as e:
            self.error_message = str(e)
            print(f"FAILED ({e})")

    def check_connection(self) -> tuple[bool, str]:
        """Check if the model is loaded."""
        if self.available:
            return True, f"Local model loaded ({self.model})"
        return False, f"Model not loaded: {self.error_message or 'Unknown error'}"

    def _parse_harmony_output(self, new_token_ids) -> str:
        """Parse generated tokens via openai_harmony. Returns the final channel text."""
        from openai_harmony import Role
        token_list = new_token_ids.tolist()
        msgs = self._harmony_enc.parse_messages_from_completion_tokens(
            token_list, role=Role.ASSISTANT, strict=False
        )
        for m in reversed(msgs):
            if m.channel == "final":
                return "".join(c.text for c in m.content if hasattr(c, "text")).strip()
        if msgs:
            return "".join(c.text for c in msgs[-1].content if hasattr(c, "text")).strip()
        return ""

    # Sentinels injected into the text queue for channel transitions
    _THINK_START = "\x00T\x00"
    _FINAL_START = "\x00F\x00"

    @property
    def display_name(self) -> str:
        """Short display name for the model, e.g. 'GPT-OSS-120B'."""
        return self.model.split("/")[-1].upper()

    def _patch_streamer(self, streamer):
        """Monkey-patch streamer.put() to decode via StreamableParser.

        Completely bypasses TextIteratorStreamer's own decode logic.
        Uses openai_harmony StreamableParser for all token decoding.
        Only content from analysis and final channels is emitted.
        """
        from openai_harmony import Role, StreamableParser

        _parser = StreamableParser(
            self._harmony_enc, role=Role.ASSISTANT, strict=False
        )
        _seen_analysis = [False]
        _seen_final = [False]
        _skip_prompt = [True]
        _THINK_START = self._THINK_START
        _FINAL_START = self._FINAL_START

        def _put_with_harmony(value):
            if _skip_prompt[0]:
                _skip_prompt[0] = False
                return
            ids = value.flatten().tolist() if hasattr(value, "flatten") else [value]
            for tid in ids:
                _parser.process(tid)
                delta = _parser.last_content_delta
                ch = _parser.current_channel

                if ch == "analysis" and not _seen_analysis[0]:
                    streamer.text_queue.put(_THINK_START)
                    _seen_analysis[0] = True

                if ch == "final" and not _seen_final[0]:
                    streamer.text_queue.put(_FINAL_START)
                    _seen_final[0] = True

                if delta and ch in ("analysis", "final"):
                    streamer.text_queue.put(delta)

        streamer.put = _put_with_harmony

    def _build_messages(
        self,
        prompt: str,
        system: str = "",
        history: list[tuple[str, str]] | None = None,
    ) -> list[dict]:
        """Build HF chat messages list with reasoning effort in system message."""
        messages = []
        sys_content = (
            f"Reasoning: {self.reasoning}\n\n"
            "# Valid channels: analysis, commentary, final. "
            "Channel must be included for every message."
        )
        if system:
            sys_content += f"\n\n{system}"
        messages.append({"role": "system", "content": sys_content})
        if history:
            for user_msg, bot_msg in history:
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                if bot_msg:
                    messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _tokenize(self, messages: list[dict]):
        """Apply chat template and tokenize. Returns (input_ids, attention_mask)."""
        input_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(input_text, return_tensors="pt").to(self._model.device)
        return inputs["input_ids"], inputs["attention_mask"]

    def generate(
        self,
        prompt: str,
        system: str = "",
        timeout: int = 90,
        stream: bool = False,
        history: list[tuple[str, str]] | None = None,
        agent: str = "",
        final_color: str = "",
    ) -> str | None:
        """Generate a response.

        stream=False: silent internal call, returns parsed final-channel text.
        stream=True:  prints streaming output to terminal.
        history: list of (user_msg, assistant_msg) tuples for conversation context.
        agent: display name for this agent (e.g. "Intent Agent").
        final_color: ANSI color code for the final output (e.g. "\\033[32m" for green).
                     Empty string keeps it gray.
        """
        if not self.available:
            return None

        import torch
        from transformers import TextIteratorStreamer
        import threading

        try:
            messages = self._build_messages(prompt, system, history)
            input_ids, attention_mask = self._tokenize(messages)
            prompt_len = input_ids.shape[1]

            if not stream:
                with torch.no_grad():
                    outputs = self._model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=512,
                        temperature=0.1,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
                return self._parse_harmony_output(outputs[0][prompt_len:])

            # ── Streaming path ──
            streamer = TextIteratorStreamer(
                self._tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            self._patch_streamer(streamer)

            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
                streamer=streamer,
            )

            thread = threading.Thread(target=lambda: self._model.generate(**gen_kwargs))
            thread.start()

            answer_text = ""
            in_final = False
            in_thinking = False
            tag = f"[{agent}]" if agent else ""

            # Thinking line in dark gray
            print(f"\033[90m{tag} [thinking] ", end="", flush=True)

            for new_text in streamer:
                if new_text == self._THINK_START:
                    in_thinking = True
                    continue
                if new_text == self._FINAL_START:
                    if in_thinking:
                        print(" [/thinking]", end="", flush=True)
                    # Final output on next line, in chosen color or gray
                    color = final_color if final_color else "\033[90m"
                    print(f"\033[0m", flush=True)
                    print(f"{color}", end="", flush=True)
                    in_final = True
                    in_thinking = False
                    continue
                print(new_text, end="", flush=True)
                if in_final:
                    answer_text += new_text

            thread.join()
            print("\033[0m", flush=True)  # reset, single newline

            return answer_text.strip() or None

        except Exception as e:
            print(f"  [LLM Error: {e}]")
            return None

    def chat(
        self,
        message: str,
        history: list[tuple[str, str]],
        system: str = "",
    ) -> Generator[str, None, None]:
        """Streaming chat for Gradio UI. Yields only the final-channel answer."""
        if not self.available:
            yield f"Model not available: {self.error_message}"
            return

        from transformers import TextIteratorStreamer
        import threading

        try:
            messages = self._build_messages(message, system, history)
            input_ids, attention_mask = self._tokenize(messages)

            streamer = TextIteratorStreamer(
                self._tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            self._patch_streamer(streamer)

            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
                streamer=streamer,
            )

            thread = threading.Thread(target=lambda: self._model.generate(**gen_kwargs))
            thread.start()

            answer_text = ""
            in_final = False

            for new_text in streamer:
                if new_text == self._THINK_START:
                    continue
                if new_text == self._FINAL_START:
                    in_final = True
                    continue
                if in_final:
                    answer_text += new_text
                    yield answer_text

            thread.join()

        except Exception as e:
            yield f"Error: {str(e)}"


SYSTEM_PROMPT_PATH = EASY_DIR / "system_prompt.txt"


class LLMHelper:
    """Use LLM for natural language understanding and responses."""

    INTENT_DETECTION_PROMPT = """You are a command router for Surya, a solar forecasting system.

Classify the user's intent into ONE of these actions:
- INFERENCE: User wants to run the model, get predictions, generate output for a date
- ANALYSIS: User wants to analyze existing predictions, compare channels, see MSE/errors
- QUESTION: User is asking a question about the system, channels, or solar physics
- CHAT: Greetings, small talk, casual conversation, thanks, or anything not related to solar forecasting

Output ONLY one word: INFERENCE, ANALYSIS, QUESTION, or CHAT

Examples:
"get model output for october 23 2014" -> INFERENCE
"run prediction for the solar flare day" -> INFERENCE
"predict 5 hours ahead for march 2020" -> INFERENCE
"show me what happens on twelve october" -> INFERENCE
"analyze the results" -> ANALYSIS
"compare all channels" -> ANALYSIS
"which channel has lowest error" -> ANALYSIS
"rank channels by MSE" -> ANALYSIS
"what channel shows flares?" -> QUESTION
"how does the model work?" -> QUESTION
"tell me about aia94" -> QUESTION
"hi" -> CHAT
"hello" -> CHAT
"thanks" -> CHAT
"how are you" -> CHAT

User query:"""

    PARAM_EXTRACTION_PROMPT = """You are a parameter extraction assistant for Surya solar forecasting.

TASK: Extract date, time, and rollout parameters. Output JSON with your analysis.

REQUIRED INFO:
- Date with YEAR (e.g., "october 23 2014", "2020-05-20") - YEAR IS REQUIRED
- Rollout steps (optional, default: 1)

DATE RULES:
- "twelve" = 12, "twenty third" = 23
- YEAR is required - if missing, set date to null
- Default time: 10:00 (if hour not specified)

ROLLOUT RULES:
- "N steps", "N rollout steps", "N hours ahead"
- Word numbers: "five" = 5

OUTPUT FORMAT (JSON only):
{
  "start_datetime": "YYYY-MM-DD HH:MM" or null,
  "rollout_steps": N or null,
  "missing": ["list of missing required fields"],
  "understood": "brief description of what you understood"
}

EXAMPLES:
Query: "october 23 2014 with 5 steps"
{"start_datetime": "2014-10-23 10:00", "rollout_steps": 5, "missing": [], "understood": "Run prediction for Oct 23, 2014 with 5 rollout steps"}

Query: "may twenty with 1 step"
{"start_datetime": null, "rollout_steps": 1, "missing": ["year"], "understood": "May 20th with 1 rollout step, but year is missing"}

Query: "run prediction"
{"start_datetime": null, "rollout_steps": null, "missing": ["date", "year"], "understood": "Want to run prediction but no date specified"}

Query: "2020-05-20"
{"start_datetime": "2020-05-20 10:00", "rollout_steps": null, "missing": [], "understood": "May 20, 2020, will use default rollout steps"}

Now parse this query:"""

    FOLLOWUP_PROMPT = """You are a helpful assistant for Surya solar forecasting.
The user wants to run a prediction but some information is missing.

Based on what they said and what's missing, ask a friendly follow-up question to get the missing info.
Keep it conversational and brief (1-2 sentences).

User said: "{query}"
Understood: {understood}
Missing: {missing}

Generate a friendly follow-up question:"""

    RESPONSE_PROMPT = """You are a helpful assistant for Surya solar forecasting.
Generate a brief, friendly response explaining what you're about to do.

Parameters extracted:
- Date: {date}
- Rollout steps: {rollout}
- Action: Running solar prediction model

Generate a 1-2 sentence response explaining what will happen. Be specific about the date and prediction horizon.
Include something interesting about solar forecasting if relevant."""

    def __init__(self, llm: LocalLLMClient):
        self.llm = llm
        self._system_prompt = self._load_system_prompt()

    @staticmethod
    def _load_system_prompt() -> str:
        """Load system prompt from external file."""
        if SYSTEM_PROMPT_PATH.exists():
            return SYSTEM_PROMPT_PATH.read_text().strip()
        return "You are a helpful assistant for Surya, a solar forecasting system."

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

    _VALID_INTENTS = {"INFERENCE", "ANALYSIS", "QUESTION", "CHAT"}

    def detect_intent(self, user_input: str) -> str:
        """Detect user intent using LLM with visible streaming output."""
        if not self.llm.available:
            return "QUESTION"

        prompt = f"{self.INTENT_DETECTION_PROMPT} \"{user_input}\""
        response = self.llm.generate(prompt, "", timeout=15, stream=True, agent="Intent Agent", final_color="\033[37m")

        if not response:
            return "QUESTION"

        last_line = response.strip().split("\n")[-1].strip()
        token = last_line.split()[0].upper().strip(".:,\"'()") if last_line.split() else ""
        return token if token in self._VALID_INTENTS else "QUESTION"

    def extract_params(self, user_input: str, max_retries: int = 2) -> dict[str, Any] | None:
        """Extract structured parameters from natural language using LLM.

        Returns dict with keys: start_datetime, rollout_steps, missing, understood
        Returns None if extraction fails after retries.
        """
        if not self.llm.available:
            return None

        for attempt in range(max_retries):
            prompt = f"{self.PARAM_EXTRACTION_PROMPT}\nQuery: \"{user_input}\""

            if attempt > 0:
                prompt += "\n\nIMPORTANT: Output ONLY valid JSON."

            response = self.llm.generate(prompt, "", timeout=45, stream=True, agent="Params Agent")

            if not response:
                continue

            try:
                cleaned = response.strip()
                cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
                cleaned = re.sub(r'\s*```$', '', cleaned)

                # Find JSON object
                json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return result
                else:
                    return json.loads(cleaned)
            except json.JSONDecodeError:
                if attempt == max_retries - 1:
                    print(f"  [LLM JSON parse failed: {response[:80]}...]")

        return None

    def generate_followup(self, query: str, understood: str, missing: list[str]) -> str | None:
        """Generate a follow-up question for missing information."""
        if not self.llm.available:
            return None

        prompt = self.FOLLOWUP_PROMPT.format(
            query=query,
            understood=understood,
            missing=", ".join(missing)
        )
        return self.llm.generate(prompt, "", timeout=30, stream=True, agent="Follow-up Agent")

    def generate_response(self, date: str, rollout: int) -> str | None:
        """Generate a friendly response explaining what will happen."""
        if not self.llm.available:
            return None

        prompt = self.RESPONSE_PROMPT.format(
            date=date,
            rollout=rollout
        )
        return self.llm.generate(prompt, "", timeout=30, agent="Response Agent")

    def is_query_complete(self, params: dict[str, Any]) -> bool:
        """Check if extracted params have all required info."""
        if not params:
            return False
        missing = params.get("missing", [])
        has_date = params.get("start_datetime") is not None
        return has_date and len(missing) == 0


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


class SuryaInterface:
    """Main interface class with LLM support."""

    def __init__(self, use_llm: bool = True, llm_model: str | None = None, reasoning: str = "medium"):
        self.logger = SessionLogger()
        self.date_parser = DateParser()
        self.last_output_dir: str | None = None
        self._reasoning = reasoning

        # Initialize LLM
        self.llm: LocalLLMClient | None = None
        self.llm_helper: LLMHelper | None = None
        self.pending_inference: dict | None = None  # For multi-turn conversations
        self.conversation_history: list[tuple[str, str]] = []  # (user, assistant)

        if use_llm:
            self._init_llm(llm_model)

    def _init_llm(self, model: str | None):
        """Initialize local HuggingFace model."""
        self.llm = LocalLLMClient(model=model, reasoning=self._reasoning)
        if self.llm.available:
            self.llm_helper = LLMHelper(self.llm)
        else:
            self.llm = None

    def run(self):
        """Main REPL loop."""
        self.show_welcome()

        while True:
            try:
                print()
                cmd = input("[[SuryaGPT]] > ").strip()
                if not cmd:
                    continue
                print()  # blank line after input
                if cmd.lower() in ["exit", "quit", "q"]:
                    print("\nGoodbye!")
                    self.logger.log("session_end")
                    break

                self.logger.log("command", {"input": cmd})
                self.process(cmd)

            except KeyboardInterrupt:
                print("\n(Use 'exit' to quit)")
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                self.logger.log("error", {"message": str(e)})

    def process(self, cmd: str):
        """Route command to handler."""
        cmd_lower = cmd.lower()

        # Quick commands
        if cmd_lower in ["help", "?", "h"]:
            self.show_help()
            return
        if cmd_lower in ["exit", "quit", "q"]:
            return
        if cmd_lower == "status":
            self.show_status()
            return
        if cmd_lower == "examples":
            self.show_examples()
            return

        # Route to appropriate handler
        self.process_command(cmd)

    def process_command(self, cmd: str):
        """Route command using LLM intent detection."""
        cmd_lower = cmd.lower()

        # Quick commands (bypass LLM for efficiency)
        if any(kw in cmd_lower for kw in ["config", "setting"]):
            self.show_config()
            return
        if "status" in cmd_lower and len(cmd_lower) < 20:
            self.show_status()
            return
        if "help" in cmd_lower and len(cmd_lower) < 15:
            self.show_help()
            return
        if "example" in cmd_lower and len(cmd_lower) < 20:
            self.show_examples()
            return
        if cmd_lower in ["channels", "list channels"]:
            self.list_channels()
            return
        if cmd_lower in ["cancel", "nevermind", "never mind", "reset"]:
            self.pending_inference = None
            print("  OK, cancelled.")
            return

        # Check if we have a pending inference waiting for more info
        if self.pending_inference and self.llm_helper:
            # Try to complete the pending inference with this input
            if self.complete_pending_inference(cmd):
                return
            # If that didn't work, continue with normal processing

        # Use LLM to detect intent
        if self.llm_helper:
            intent = self.llm_helper.detect_intent(cmd)

            if intent == "INFERENCE":
                self.run_inference(cmd)
            elif intent == "ANALYSIS":
                self.run_analysis(cmd)
            elif intent in ("CHAT", "QUESTION"):
                response = self.llm_helper.ask(
                    cmd, stream=True, history=self.conversation_history
                )
                if response:
                    self.conversation_history.append((cmd, response))
                    # Keep history bounded to avoid blowing up context
                    if len(self.conversation_history) > 20:
                        self.conversation_history = self.conversation_history[-20:]
                    self.logger.log("llm_response", {"input": cmd, "response": response})
        else:
            # Fallback to keyword matching if LLM unavailable
            if any(kw in cmd_lower for kw in ["run", "inference", "predict", "forecast"]):
                self.run_inference(cmd)
            elif any(kw in cmd_lower for kw in ["analyze", "analyse", "compare", "mse", "error", "plot", "rank"]):
                self.run_analysis(cmd)
            else:
                self.handle_unknown(cmd)

    def run_inference_with_params(
        self,
        start_dt: str,
        end_dt: str,
        rollout_steps: int,
        dry_run: bool = False,
        skip_download: bool = False,
    ):
        """Run inference with specific parameters."""
        args = [
            sys.executable,
            str(EASY_DIR / "run_easy_inference.py"),
            "--config-path", str(CONFIG_PATH),
            "--no-prompt",
            "--start-datetime", start_dt,
            "--end-datetime", end_dt,
            "--rollout-steps", str(rollout_steps),
        ]

        if dry_run:
            args.append("--dry-run")
        if skip_download:
            args.append("--skip-download")

        print(f"\n  Start: {start_dt}")
        print(f"  End: {end_dt}")
        print(f"  Rollout steps: {rollout_steps}")
        print("\n>>> Executing inference...")
        print(f"    Command: {' '.join(args)}\n")

        self.logger.log("inference_start", {
            "start_datetime": start_dt,
            "end_datetime": end_dt,
            "rollout_steps": rollout_steps,
        })

        result = subprocess.run(args, cwd=ROOT_DIR)
        self.logger.log("inference_end", {"returncode": result.returncode})

        if result.returncode == 0:
            print("\n>>> Inference completed successfully!")
        else:
            print(f"\n>>> Inference failed with code {result.returncode}")

    def run_inference(self, cmd: str):
        """Execute inference with LLM extraction, validation, and follow-up questions."""

        # Check for dry-run / skip-download flags
        dry_run = "--dry-run" in cmd or "dry run" in cmd.lower()
        skip_download = "--skip-download" in cmd or "skip download" in cmd.lower()

        if not self.llm_helper:
            # Fallback to basic regex if no LLM
            self._run_inference_basic(cmd, dry_run, skip_download)
            return

        # Use LLM to extract and validate parameters
        params = self.llm_helper.extract_params(cmd)

        if not params:
            print("  Extraction failed, trying regex fallback...")
            self._run_inference_basic(cmd, dry_run, skip_download)
            return

        # Show what LLM understood
        understood = params.get("understood", "Processing your request")
        print(f"  > {understood}")

        # Check if we have all required info
        missing = params.get("missing", [])
        start_dt = params.get("start_datetime")
        rollout = params.get("rollout_steps")

        if missing or not start_dt:
            # Ask follow-up question
            print()
            followup = self.llm_helper.generate_followup(
                cmd,
                understood,
                missing if missing else ["complete date with year"]
            )
            if followup:
                print(f"  {followup}")
            else:
                print("  I need a complete date with year (e.g., 'october 23 2014').")
                print("  Please provide the missing information.")

            # Store pending context for next command
            self.pending_inference = {
                "original_cmd": cmd,
                "params": params,
                "dry_run": dry_run,
                "skip_download": skip_download,
            }
            return

        # Clear any pending context
        self.pending_inference = None

        # Set default rollout if not specified
        if rollout is None:
            rollout = 1
            print(f"  Using default: {rollout} rollout step")

        # Execute inference
        self._execute_inference(start_dt, rollout, dry_run, skip_download)

    def complete_pending_inference(self, additional_info: str):
        """Complete a pending inference with additional user input."""
        if not self.pending_inference:
            return False

        # Combine original command with new info
        original = self.pending_inference["original_cmd"]
        combined = f"{original} {additional_info}"

        print(f"\n  Combining with previous request: '{original}'")

        # Re-extract with combined input
        params = self.llm_helper.extract_params(combined)

        if params and params.get("start_datetime"):
            understood = params.get("understood", "Got it!")
            print(f"\n  > {understood}")

            start_dt = params.get("start_datetime")
            rollout = params.get("rollout_steps") or 1

            # Execute
            self._execute_inference(
                start_dt,
                rollout,
                self.pending_inference["dry_run"],
                self.pending_inference["skip_download"]
            )
            self.pending_inference = None
            return True

        print("\n  Still missing some information. Please provide a complete date with year.")
        return False

    def _execute_inference(self, start_dt: str, rollout: int, dry_run: bool, skip_download: bool):
        """Execute the actual inference command."""
        args = [
            sys.executable,
            str(EASY_DIR / "run_easy_inference.py"),
            "--config-path", str(CONFIG_PATH),
            "--no-prompt",
            "--start-datetime", start_dt,
            "--rollout-steps", str(rollout),
        ]

        if dry_run:
            args.append("--dry-run")
        if skip_download:
            args.append("--skip-download")

        # Compute input/output timeline (cadence = 60 min)
        try:
            dt = datetime.strptime(start_dt, "%Y-%m-%d %H:%M")
        except ValueError:
            dt = datetime.strptime(start_dt, "%Y-%m-%d %H:%M:%S")
        from datetime import timedelta

        prediction_steps = rollout + 1
        fmt = "%b %d %H:%M"
        date_str = dt.strftime("%B %d, %Y")

        # Build labels: first 2 inputs are always GT, then slides
        input_t0 = dt - timedelta(minutes=60)
        input_t1 = dt
        window = [f"GT {input_t0.strftime(fmt)}", f"GT {input_t1.strftime(fmt)}"]

        print(f"\n  Surya Inference Plan  ({date_str}, {prediction_steps} steps)")
        print(f"  " + "-" * 60)
        for step in range(1, prediction_steps + 1):
            out_t = dt + timedelta(minutes=60 * step)
            out_label = f"Pred {out_t.strftime(fmt)}"
            print(f"    Step {step}/{prediction_steps}  [{window[0]}, {window[1]}] -> {out_label}")
            window = [window[1], out_label]
        print(f"  " + "-" * 60)
        print()

        self.logger.log("inference_start", {
            "start_datetime": start_dt,
            "rollout_steps": rollout,
        })

        try:
            result = subprocess.run(args, cwd=ROOT_DIR)
            self.logger.log("inference_end", {"returncode": result.returncode})

            if result.returncode == 0:
                print("\n>>> Prediction complete!")
            else:
                print(f"\n>>> Inference failed with code {result.returncode}")
        except Exception as e:
            print(f"\n>>> Inference error: {e}")
            self.logger.log("inference_error", {"error": str(e)})

    def _run_inference_basic(self, cmd: str, dry_run: bool, skip_download: bool):
        """Basic inference without LLM (fallback mode)."""
        start_dt, _ = self.date_parser.parse(cmd)
        rollout = self.date_parser.parse_rollout(cmd) or 1

        if not start_dt:
            print("  No date found. Please use format: 'YYYY-MM-DD' (e.g., 2014-10-23)")
            return

        self._execute_inference(start_dt, rollout, dry_run, skip_download)

    def run_analysis_with_params(
        self,
        channel: str = "aia94",
        analyze_all: bool = False,
        rank: bool = False,
        mse_trend: bool = False,
        generate_plots: bool = True,
        channel_filter: str = "",
    ):
        """Run analysis with specific parameters."""
        output_dirs = sorted(
            EASY_DIR.glob("outputs_*"), key=lambda p: p.stat().st_mtime, reverse=True
        )

        prediction_nc = None
        for d in output_dirs:
            nc_file = d / "prediction.nc"
            if nc_file.exists():
                prediction_nc = nc_file
                break

        if not prediction_nc:
            print("\nNo prediction.nc found. Run inference first.")
            return

        print(f"\n>>> Analyzing: {prediction_nc}")

        args = [
            sys.executable,
            str(EASY_DIR / "analyze_predictions.py"),
            "--prediction-nc", str(prediction_nc),
        ]

        if analyze_all:
            args.append("--all-channels")
            print("    Analyzing: all channels")
        else:
            args.extend(["--channel", channel])
            print(f"    Channel: {channel}")

        if mse_trend:
            args.append("--show-mse-trend")
            print("    Generating: MSE trend plot")

        if rank:
            args.append("--rank-channels")
            print("    Generating: channel ranking")

        if not generate_plots:
            args.append("--no-plots")
        else:
            print("    Generating: comparison plots with difference maps")

        self.logger.log("analysis_start", {
            "prediction_nc": str(prediction_nc),
            "channel": channel,
            "analyze_all": analyze_all,
            "rank": rank,
            "mse_trend": mse_trend,
            "generate_plots": generate_plots,
        })

        result = subprocess.run(args, cwd=ROOT_DIR)
        self.logger.log("analysis_end", {"returncode": result.returncode})

    def run_analysis(self, cmd: str):
        """Run analysis (regex parsing)."""
        output_dirs = sorted(
            EASY_DIR.glob("outputs_*"), key=lambda p: p.stat().st_mtime, reverse=True
        )

        prediction_nc = None
        for d in output_dirs:
            nc_file = d / "prediction.nc"
            if nc_file.exists():
                prediction_nc = nc_file
                break

        if not prediction_nc:
            print("\nNo prediction.nc found. Run inference first.")
            return

        channel = "aia94"
        for ch in ALL_CHANNELS:
            if ch in cmd.lower():
                channel = ch
                break

        analyze_all = "all" in cmd.lower()

        print(f"\n>>> Analyzing: {prediction_nc}")
        print(f"    Channel: {'all' if analyze_all else channel}")

        args = [
            sys.executable,
            str(EASY_DIR / "analyze_predictions.py"),
            "--prediction-nc", str(prediction_nc),
        ]

        if analyze_all:
            args.append("--all-channels")
        else:
            args.extend(["--channel", channel])

        if "mse" in cmd.lower() or "trend" in cmd.lower():
            args.append("--show-mse-trend")

        if "rank" in cmd.lower():
            args.append("--rank-channels")

        self.logger.log("analysis_start", {"prediction_nc": str(prediction_nc), "channel": channel})
        result = subprocess.run(args, cwd=ROOT_DIR)
        self.logger.log("analysis_end", {"returncode": result.returncode})

    def show_welcome(self):
        """Show welcome message."""
        if self.llm and self.llm.available:
            llm_status = f"LLM: {self.llm.model}"
        else:
            llm_status = "LLM: OFF (regex mode)"

        print(f"""
===========================================================
              SURYA - Solar Forecasting Interface
                   366M Foundation Model
===========================================================
  {llm_status}

  Try:
    "What can you do?"
    "Run prediction for october 23 2014 with 3 steps"
    "Analyze all channels and rank by MSE"

  Type 'help' for commands, 'exit' to quit
===========================================================
        """)
        print(f"Session log: {self.logger.get_log_path()}\n")

    def show_help(self):
        """Show help text."""
        print("""
NATURAL LANGUAGE QUERIES (when LLM is ON)
==========================================
  "What channel should I use for solar flares?"
  "Show me the corona observations"
  "Run a prediction for the big flare day"
  "How does prediction accuracy change over time?"
  "Compare all channels and tell me which is best"
  "Analyze the magnetic field evolution"

BASIC COMMANDS (always work)
============================
  run inference              Run with default settings
  run inference --dry-run    Preview settings only
  predict 8 hours ahead      Set rollout steps
  analyze                    Analyze latest predictions
  analyze aia171             Analyze specific channel
  rank channels              Rank by prediction error
  show mse trend             Plot error over time
  help                       Show this help
  status                     Show system status
  exit                       Quit

CHANNELS
========
  Flares:    aia94, aia131
  Corona:    aia171, aia193
  Active:    aia211, aia335
  Chromo:    aia304
  UV:        aia1600
  Magnetic:  hmi_m, hmi_bx, hmi_by, hmi_bz
  Velocity:  hmi_v
        """)

    def show_examples(self):
        """Show example queries."""
        print("""
EXAMPLE NATURAL LANGUAGE QUERIES
================================

# Understanding the system
> What channels show solar flares?
> Tell me about the magnetic field observations
> Which channel is best for seeing the corona?

# Running predictions
> Run a prediction for the solar flare on October 23rd
> Predict 10 hours ahead starting at noon
> I want to see what happens during the X1.6 flare

# Analysis
> How accurate are the magnetic field predictions?
> Compare all channels and rank them by error
> Show me how the error grows over time
> Which channel does the model predict best?

# Combined workflows
> Predict the flare and then analyze the hot plasma channel
> Run inference and show me the magnetic field analysis
        """)

    def show_config(self):
        """Display current config."""
        print(f"\nConfiguration file: {CONFIG_PATH}")
        print("-" * 50)

        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                print(f.read())
        else:
            print("Config file not found!")

    def list_channels(self):
        """List available channels."""
        print("""
AVAILABLE CHANNELS (13 total)
=============================

AIA (Atmospheric Imaging Assembly):
  aia94    - 94A   Hot flare plasma (~6.3 MK)
  aia131   - 131A  Flare plasma + cooler
  aia171   - 171A  Quiet corona (~0.6 MK)
  aia193   - 193A  Corona and hot flares
  aia211   - 211A  Active regions (~2 MK)
  aia304   - 304A  Chromosphere (~50,000 K)
  aia335   - 335A  Active regions (~2.5 MK)
  aia1600  - 1600A Upper photosphere

HMI (Helioseismic and Magnetic Imager):
  hmi_m    - Total magnetic field magnitude
  hmi_bx   - Magnetic field X component
  hmi_by   - Magnetic field Y component
  hmi_bz   - Magnetic field Z component
  hmi_v    - Line-of-sight velocity
        """)

    def show_status(self):
        """Show system status."""
        print("\nSYSTEM STATUS")
        print("=" * 40)

        venv_active = sys.prefix != sys.base_prefix
        print(f"Virtual env active: {'Yes' if venv_active else 'No'}")
        print(f"LLM available: {'Yes' if self.llm and self.llm.available else 'No'}")
        print(f"Config exists: {'Yes' if CONFIG_PATH.exists() else 'No'}")

        output_dirs = list(EASY_DIR.glob("outputs_*"))
        print(f"Output directories: {len(output_dirs)}")

        for d in sorted(output_dirs, key=lambda p: p.stat().st_mtime, reverse=True):
            nc = d / "prediction.nc"
            if nc.exists():
                size_mb = nc.stat().st_size / (1024 * 1024)
                print(f"Latest prediction: {nc.name} ({size_mb:.1f} MB)")
                break

        log_files = list(LOG_DIR.glob("session_*.log")) if LOG_DIR.exists() else []
        print(f"Session logs: {len(log_files)}")
        print(f"\nCurrent session: {self.logger.session_id}")

    def handle_unknown(self, cmd: str):
        """Handle unknown commands."""
        print(f"Unknown command: '{cmd}'")
        print("Type 'help' for available commands.")


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Surya Solar Forecasting - Natural Language Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python llm_interface.py                       # Interactive mode selection
  python llm_interface.py --model gpt-oss --reasoning high
  python llm_interface.py --no-llm              # Disable LLM (regex only)
        """
    )

    parser.add_argument("--no-llm", action="store_true", help="Disable LLM (regex-only mode)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name: gpt-oss, gpt-oss-120b, gpt-oss-20b")
    parser.add_argument("--reasoning", type=str, default=None,
                        choices=["low", "medium", "high"],
                        help="Reasoning level (skips interactive menu)")

    args = parser.parse_args()

    if args.no_llm:
        use_llm = False
        model = None
        reasoning = "medium"
    elif args.model:
        model = LocalLLMClient.MODEL_ALIASES.get(args.model, args.model)
        reasoning = args.reasoning or "medium"
        use_llm = True
    else:
        # Interactive selection menu
        result = select_model()
        if result is None:
            use_llm = False
            model = None
            reasoning = "medium"
        else:
            model, reasoning = result
            use_llm = True

    interface = SuryaInterface(use_llm=use_llm, llm_model=model, reasoning=reasoning)
    interface.run()


if __name__ == "__main__":
    main()
