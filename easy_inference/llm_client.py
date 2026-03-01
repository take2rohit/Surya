#!/usr/bin/env python3
"""
LLM client, model selection, session logging, and shared constants.

This module provides:
- Path and channel constants used across the codebase
- ``SessionLogger`` for JSON-lines session logging
- ``LocalLLMClient`` for GPT-OSS inference with Harmony format streaming
- ``select_model()`` interactive model selection menu
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

# ── Path constants ──────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
EASY_DIR = ROOT_DIR / "easy_inference"
CONFIG_PATH = EASY_DIR / "config_easy.yaml"
LOG_DIR = EASY_DIR / "logs"
SCOPE_PATH = EASY_DIR / "SCOPE.md"

# ── Channel constants ───────────────────────────────────────────────────────

CHANNELS = {
    "aia": ["aia94", "aia131", "aia171", "aia193", "aia211", "aia304", "aia335", "aia1600"],
    "hmi": ["hmi_m", "hmi_bx", "hmi_by", "hmi_bz", "hmi_v"],
}
ALL_CHANNELS = CHANNELS["aia"] + CHANNELS["hmi"]

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

# ── Model catalogue ─────────────────────────────────────────────────────────

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


# ── Session logging ─────────────────────────────────────────────────────────

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


# ── LLM client ──────────────────────────────────────────────────────────────

class LocalLLMClient:
    """Client for GPT-OSS on GPU with Harmony format parsing via openai_harmony."""

    DEFAULT_MODEL = "openai/gpt-oss-120b"

    MODEL_ALIASES = {
        "gpt-oss": "openai/gpt-oss-120b",
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "gpt-oss-20b": "openai/gpt-oss-20b",
    }

    REASONING_LEVELS = ("low", "medium", "high")

    # Reserve tokens for generation output
    _GENERATION_RESERVE = 1024

    def __init__(self, model: str | None = None, reasoning: str = "medium"):
        resolved = self.MODEL_ALIASES.get(model, model) if model else None
        self.model = resolved or self.DEFAULT_MODEL
        self.reasoning = reasoning if reasoning in self.REASONING_LEVELS else "medium"
        self.available = False
        self.error_message: str | None = None
        self._tokenizer = None
        self._model = None
        self._harmony_enc = None
        self.context_window: int = 0  # max tokens the model supports
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

            # Read context window from model config
            cfg = self._model.config
            self.context_window = getattr(
                cfg, "max_position_embeddings",
                getattr(cfg, "n_positions", getattr(cfg, "seq_length", 4096)),
            )

            self.available = True
            print(f"OK (context: {self.context_window} tokens)")

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
        """Inject Harmony-aware decoding into a TextIteratorStreamer.

        Why monkey-patching is necessary:
        TextIteratorStreamer decodes tokens with the HF tokenizer, but GPT-OSS
        uses Harmony format where tokens encode structured channels (analysis,
        commentary, final). The standard decoder produces garbage because it
        doesn't understand channel boundaries. By replacing ``streamer.put()``,
        we intercept raw token IDs *before* HF decoding and route them through
        ``openai_harmony.StreamableParser`` instead. Only content from the
        ``analysis`` and ``final`` channels is emitted to the text queue, with
        sentinel strings marking channel transitions so the display layer can
        style them differently (dim thinking vs. bold answer).
        """
        from openai_harmony import Role, StreamableParser

        assert hasattr(streamer, "text_queue"), (
            "Expected TextIteratorStreamer with text_queue attribute"
        )

        _parser = StreamableParser(
            self._harmony_enc, role=Role.ASSISTANT, strict=False
        )
        _seen_analysis = False
        _seen_final = False
        _skip_prompt = True
        _THINK_START = self._THINK_START
        _FINAL_START = self._FINAL_START

        def _put_with_harmony(value):
            nonlocal _seen_analysis, _seen_final, _skip_prompt

            if _skip_prompt:
                _skip_prompt = False
                return
            ids = value.flatten().tolist() if hasattr(value, "flatten") else [value]
            for tid in ids:
                _parser.process(tid)
                delta = _parser.last_content_delta
                ch = _parser.current_channel

                if ch == "analysis" and not _seen_analysis:
                    streamer.text_queue.put(_THINK_START)
                    _seen_analysis = True

                if ch == "final" and not _seen_final:
                    streamer.text_queue.put(_FINAL_START)
                    _seen_final = True

                if delta and ch in ("analysis", "final"):
                    streamer.text_queue.put(delta)

        streamer.put = _put_with_harmony

    def count_tokens(self, text: str) -> int:
        """Count tokens in a string using the loaded tokenizer."""
        if not self._tokenizer:
            return len(text) // 4  # rough fallback
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def _build_messages(
        self,
        prompt: str,
        system: str = "",
        history: list[tuple[str, str]] | None = None,
    ) -> list[dict]:
        """Build HF chat messages list, trimming history to fit context window."""
        sys_content = (
            f"Reasoning: {self.reasoning}\n\n"
            "# Valid channels: analysis, commentary, final. "
            "Channel must be included for every message."
        )
        if system:
            sys_content += f"\n\n{system}"

        # Token budget: context_window - generation reserve
        max_input_tokens = self.context_window - self._GENERATION_RESERVE
        # Count fixed tokens (system + current prompt)
        fixed_tokens = self.count_tokens(sys_content) + self.count_tokens(prompt) + 20  # overhead

        # Trim history from oldest to fit within budget
        trimmed_history: list[tuple[str, str]] = []
        if history:
            history_tokens = 0
            # Walk from newest to oldest, accumulate tokens
            for user_msg, bot_msg in reversed(history):
                pair_tokens = self.count_tokens(user_msg or "") + self.count_tokens(bot_msg or "") + 10
                if fixed_tokens + history_tokens + pair_tokens > max_input_tokens:
                    break
                history_tokens += pair_tokens
                trimmed_history.insert(0, (user_msg, bot_msg))

        messages = [{"role": "system", "content": sys_content}]
        for user_msg, bot_msg in trimmed_history:
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
