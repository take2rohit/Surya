#!/usr/bin/env python3
"""
Surya Easy Inference - Natural Language Interface with LLM Support

A terminal interface for running solar predictions and analysis
using natural language. Supports local Ollama or cloud Groq API.

Usage:
    python llm_interface.py                # Auto-detect (Ollama first, then Groq)
    python llm_interface.py --ollama       # Use local Ollama
    python llm_interface.py --groq         # Use Groq API
    python llm_interface.py --model llama3.2:3b  # Specify model
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
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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


class OllamaClient:
    """Client for local Ollama LLM."""

    DEFAULT_MODEL = "gpt-oss-120b"
    FALLBACK_MODELS = ["llama3.1:8b", "llama3.2:3b", "llama3.2:1b", "mistral:7b", "qwen2.5:7b"]

    def __init__(self, model: str | None = None, host: str = "http://localhost:11434"):
        self.host = host
        self.model = model
        self.available = False
        self._check_connection()

    def _check_connection(self):
        """Check if Ollama is running and find available model."""
        import urllib.request
        import urllib.error

        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                available_models = [m["name"] for m in data.get("models", [])]

                if not available_models:
                    return

                # Use specified model or find first available
                if self.model and self.model in available_models:
                    self.available = True
                elif self.model and any(self.model in m for m in available_models):
                    # Partial match (e.g., "llama3.2" matches "llama3.2:3b")
                    for m in available_models:
                        if self.model in m:
                            self.model = m
                            self.available = True
                            break
                else:
                    # Try default, then fallbacks
                    for candidate in [self.DEFAULT_MODEL] + self.FALLBACK_MODELS:
                        if candidate in available_models:
                            self.model = candidate
                            self.available = True
                            break
                        # Partial match
                        for m in available_models:
                            if candidate.split(":")[0] in m:
                                self.model = m
                                self.available = True
                                break
                        if self.available:
                            break

                    # Use first available if nothing matched
                    if not self.available and available_models:
                        self.model = available_models[0]
                        self.available = True

        except (urllib.error.URLError, TimeoutError, ConnectionRefusedError):
            pass
        except Exception:
            pass

    def generate(self, prompt: str, system: str = "", timeout: int = 90) -> str | None:
        """Generate response from Ollama."""
        if not self.available:
            return None

        import urllib.request
        import urllib.error

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 512,
                }
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode())
                return result.get("response", "").strip()

        except Exception as e:
            print(f"  [Ollama Error: {e}]")
            return None


class GroqClient:
    """Client for Groq LLM inference API."""

    DEFAULT_MODEL = "openai/gpt-oss-120b"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or self.DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.available = bool(self.api_key)
        self._client = None

    def _get_client(self):
        """Lazy load the Groq client."""
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except Exception as e:
                print(f"  [Groq init error: {e}]")
                self.available = False
        return self._client

    def generate(self, prompt: str, system: str = "", timeout: int = 90) -> str | None:
        """Generate response from Groq API."""
        if not self.available:
            return None

        client = self._get_client()
        if not client:
            return None

        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_completion_tokens=512,
                top_p=1,
                stream=False,
            )

            return completion.choices[0].message.content.strip()

        except Exception as e:
            print(f"  [Groq Error: {e}]")
            return None


class LLMHelper:
    """Use LLM for natural language understanding and responses."""

    SYSTEM_PROMPT = """You are a helpful assistant for Surya, a solar forecasting system.

Surya is a 366M parameter foundation model trained on SDO (Solar Dynamics Observatory) data.

## WHAT SURYA CAN DO

### Inference & Prediction
- Run solar predictions for any date from 2010-present
- Multi-step rollout: predict 1-24 hours ahead (each step = 1 hour)
- Outputs predictions for all 13 channels simultaneously
- Auto-downloads required SDO data from AWS S3
- Saves results to prediction.nc (NetCDF format)

### Analysis & Metrics
- MSE (Mean Squared Error): per-step and average
- RMSE, MAE, bias: available in analysis
- Channel ranking: rank channels by prediction accuracy
- MSE trend plots: visualize error over prediction steps
- Comparison plots: Ground Truth vs Prediction vs Difference

### Available Channels (13 total)
AIA EUV (8): aia94 (flares), aia131, aia171 (corona), aia193, aia211, aia304 (chromosphere), aia335, aia1600
HMI Magnetic (5): hmi_m (magnitude), hmi_bx, hmi_by, hmi_bz (vertical), hmi_v (velocity)

### Channel Recommendations
- Solar flares: aia94 or aia131 (hot plasma ~6-10 MK)
- Corona structure: aia171 or aia193
- Magnetic field: hmi_bz (vertical) or hmi_m (total)
- Chromosphere: aia304

## WHAT SURYA CANNOT DO
- Real-time predictions (requires pre-downloaded data)
- Train or fine-tune the model
- Process non-SDO data sources
- Sub-hourly predictions (minimum 1-hour steps)

## METRICS EXPLAINED
- MSE: Mean Squared Error - average of (prediction - ground_truth)^2
- RMSE: Root MSE - square root of MSE, same units as data
- MAE: Mean Absolute Error - average of |prediction - ground_truth|

Keep responses concise (1-3 sentences). When users ask what you can do, explain the capabilities above."""

    INTENT_DETECTION_PROMPT = """You are a command router for Surya, a solar forecasting system.

Classify the user's intent into ONE of these actions:
- INFERENCE: User wants to run the model, get predictions, generate output for a date
- ANALYSIS: User wants to analyze existing predictions, compare channels, see MSE/errors
- QUESTION: User is asking a question about the system, channels, or solar physics

Output ONLY one word: INFERENCE, ANALYSIS, or QUESTION

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

    def __init__(self, llm: GroqClient):
        self.llm = llm

    def ask(self, user_input: str) -> str | None:
        """Get a natural language response from the LLM."""
        if not self.llm.available:
            return None
        return self.llm.generate(user_input, self.SYSTEM_PROMPT)

    def detect_intent(self, user_input: str) -> str:
        """Detect user intent: INFERENCE, ANALYSIS, or QUESTION."""
        if not self.llm.available:
            return "QUESTION"  # Default fallback

        prompt = f"{self.INTENT_DETECTION_PROMPT} \"{user_input}\""
        response = self.llm.generate(prompt, "", timeout=15)

        if response:
            response = response.strip().upper()
            if "INFERENCE" in response:
                return "INFERENCE"
            elif "ANALYSIS" in response:
                return "ANALYSIS"
        return "QUESTION"

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

            response = self.llm.generate(prompt, "", timeout=45)

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
        return self.llm.generate(prompt, "", timeout=30)

    def generate_response(self, date: str, rollout: int) -> str | None:
        """Generate a friendly response explaining what will happen."""
        if not self.llm.available:
            return None

        prompt = self.RESPONSE_PROMPT.format(
            date=date,
            rollout=rollout
        )
        return self.llm.generate(prompt, "", timeout=30)

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

    def __init__(self, use_llm: bool = True, llm_backend: str = "auto", llm_model: str | None = None):
        self.logger = SessionLogger()
        self.date_parser = DateParser()
        self.last_output_dir: str | None = None

        # Initialize LLM
        self.llm: GroqClient | OllamaClient | None = None
        self.llm_helper: LLMHelper | None = None
        self.pending_inference: dict | None = None  # For multi-turn conversations
        self.llm_backend = llm_backend

        if use_llm:
            self._init_llm(llm_backend, llm_model)

    def _init_llm(self, backend: str, model: str | None):
        """Initialize LLM based on backend preference."""
        if backend == "ollama":
            self._init_ollama(model)
        elif backend == "groq":
            self._init_groq(model)
        else:  # auto - try Ollama first, then Groq
            print("Checking for local Ollama...", end=" ", flush=True)
            self.llm = OllamaClient(model=model)
            if self.llm.available:
                self.llm_helper = LLMHelper(self.llm)
                print(f"OK (using {self.llm.model})")
            else:
                print("not found")
                self._init_groq(model)

    def _init_ollama(self, model: str | None):
        """Initialize Ollama client."""
        print("Initializing Ollama...", end=" ", flush=True)
        self.llm = OllamaClient(model=model)
        if self.llm.available:
            self.llm_helper = LLMHelper(self.llm)
            print(f"OK (using {self.llm.model})")
        else:
            print("Not available (is Ollama running?)")
            self.llm = None

    def _init_groq(self, model: str | None):
        """Initialize Groq client."""
        print("Initializing Groq API...", end=" ", flush=True)
        self.llm = GroqClient(model=model)
        if self.llm.available:
            self.llm_helper = LLMHelper(self.llm)
            print(f"OK (using {self.llm.model})")
        else:
            print("Not available (check GROQ_API_KEY)")
            self.llm = None

    def run(self):
        """Main REPL loop."""
        self.show_welcome()

        while True:
            try:
                cmd = input("\n[surya] > ").strip()
                if not cmd:
                    continue
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
            else:  # QUESTION
                response = self.llm_helper.ask(cmd)
                if response:
                    print(f"\n  {response}")
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
        print("\n  Analyzing your request...")
        params = self.llm_helper.extract_params(cmd)

        if not params:
            print("\n  I couldn't understand that request. Could you try rephrasing?")
            print("  Example: 'run prediction for october 23 2014 with 3 rollout steps'")
            return

        # Show what LLM understood
        understood = params.get("understood", "Processing your request")
        print(f"\n  {understood}")

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
            print(f"\n  Using default: {rollout} rollout step")

        # Generate friendly response
        response = self.llm_helper.generate_response(start_dt, rollout)
        if response:
            print(f"\n  {response}")

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
            print(f"\n  {understood}")

            start_dt = params.get("start_datetime")
            rollout = params.get("rollout_steps") or 1

            # Generate response
            response = self.llm_helper.generate_response(start_dt, rollout)
            if response:
                print(f"\n  {response}")

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

        print(f"\n  Parameters:")
        print(f"    Start: {start_dt}")
        print(f"    Rollout steps: {rollout}")
        print(f"    End: (auto-calculated)")

        print("\n>>> Running Surya model...")

        self.logger.log("inference_start", {
            "start_datetime": start_dt,
            "rollout_steps": rollout,
        })

        result = subprocess.run(args, cwd=ROOT_DIR)
        self.logger.log("inference_end", {"returncode": result.returncode})

        if result.returncode == 0:
            print("\n>>> Prediction complete!")
        else:
            print(f"\n>>> Inference failed with code {result.returncode}")

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
            backend_name = "Ollama" if isinstance(self.llm, OllamaClient) else "Groq"
            llm_status = f"LLM: {backend_name} ({self.llm.model})"
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
  python llm_interface.py                    # Auto-detect (Ollama first, then Groq)
  python llm_interface.py --ollama           # Use local Ollama
  python llm_interface.py --groq             # Use Groq API
  python llm_interface.py --ollama --model llama3.2:3b
  python llm_interface.py --no-llm           # Disable LLM (regex only)
        """
    )

    parser.add_argument("--no-llm", action="store_true", help="Disable LLM (regex-only mode)")
    parser.add_argument("--ollama", action="store_true", help="Use local Ollama LLM")
    parser.add_argument("--groq", action="store_true", help="Use Groq API")
    parser.add_argument("--model", type=str, default=None, help="Specify LLM model name")

    args = parser.parse_args()

    # Determine backend
    if args.no_llm:
        use_llm = False
        backend = "none"
    elif args.ollama:
        use_llm = True
        backend = "ollama"
    elif args.groq:
        use_llm = True
        backend = "groq"
    else:
        use_llm = True
        backend = "auto"

    interface = SuryaInterface(use_llm=use_llm, llm_backend=backend, llm_model=args.model)
    interface.run()


if __name__ == "__main__":
    main()
