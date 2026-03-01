#!/usr/bin/env python3
"""
SuryaInterface: REPL loop, command routing, inference/analysis execution.

This module provides the interactive terminal interface that ties together
the LLM client, agent helpers, and the Surya inference/analysis scripts.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from llm_client import (
    ALL_CHANNELS,
    CONFIG_PATH,
    EASY_DIR,
    LOG_DIR,
    LocalLLMClient,
    ROOT_DIR,
    SessionLogger,
)
from llm_agents import (
    DateParser,
    Intent,
    InferenceParams,
    LLMHelper,
    PendingInference,
)


def _find_latest_prediction() -> Path | None:
    """Return the most-recent ``prediction.nc`` under ``EASY_DIR/outputs_*``."""
    output_dirs = sorted(
        EASY_DIR.glob("outputs_*"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for d in output_dirs:
        nc_file = d / "prediction.nc"
        if nc_file.exists():
            return nc_file
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
        self.pending_inference: PendingInference | None = None
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

    # ── Routing ─────────────────────────────────────────────────────────────

    def process(self, cmd: str):
        """Route command to the appropriate handler."""
        cmd_lower = cmd.lower()

        # Quick commands (bypass LLM for efficiency)
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
            if self.complete_pending_inference(cmd):
                return

        # Use LLM to detect intent
        if self.llm_helper:
            intent = self.llm_helper.detect_intent(cmd)

            if intent is Intent.INFERENCE:
                self.run_inference(cmd)
            elif intent is Intent.ANALYSIS:
                self.run_analysis(cmd)
            elif intent in (Intent.CHAT, Intent.QUESTION):
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

    # ── Inference ───────────────────────────────────────────────────────────

    def run_inference(self, cmd: str):
        """Execute inference with LLM extraction, validation, and follow-up questions."""
        dry_run = "--dry-run" in cmd or "dry run" in cmd.lower()
        skip_download = "--skip-download" in cmd or "skip download" in cmd.lower()

        if not self.llm_helper:
            self._run_inference_basic(cmd, dry_run, skip_download)
            return

        params = self.llm_helper.extract_params(cmd)

        if not params:
            print("  Extraction failed, trying regex fallback...")
            self._run_inference_basic(cmd, dry_run, skip_download)
            return

        if not params.is_complete:
            missing = params.missing or ["date", "year", "time"]
            followup = self.llm_helper.generate_followup(
                cmd, params.understood, missing,
            )
            if not followup:
                print(f"  Missing: {', '.join(missing)}. Please provide date, year, and start time (e.g., 'october 23 2014 at 10am').")

            self.pending_inference = PendingInference(
                original_cmd=cmd,
                params=params,
                dry_run=dry_run,
                skip_download=skip_download,
            )
            return

        self.pending_inference = None

        rollout = params.rollout_steps
        if rollout is None:
            rollout = 0
            print(f"  Using default: no rollout (single 1-hour-ahead prediction)")

        self._execute_inference(params.start_datetime, rollout, dry_run, skip_download)

    def complete_pending_inference(self, additional_info: str) -> bool:
        """Complete a pending inference with additional user input."""
        if not self.pending_inference:
            return False

        original = self.pending_inference.original_cmd
        combined = f"{original} {additional_info}"

        print(f"\n  Combining with previous request: '{original}'")

        params = self.llm_helper.extract_params(combined)

        if params and params.start_datetime:
            print(f"\n  > {params.understood or 'Got it!'}")

            rollout = params.rollout_steps or 0

            self._execute_inference(
                params.start_datetime,
                rollout,
                self.pending_inference.dry_run,
                self.pending_inference.skip_download,
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
        rollout = self.date_parser.parse_rollout(cmd) or 0

        if not start_dt:
            print("  No date found. Please use format: 'YYYY-MM-DD' (e.g., 2014-10-23)")
            return

        self._execute_inference(start_dt, rollout, dry_run, skip_download)

    # ── Analysis ────────────────────────────────────────────────────────────

    def run_analysis(
        self,
        cmd: str = "",
        *,
        channel: str = "aia94",
        analyze_all: bool = False,
        rank: bool = False,
        mse_trend: bool = False,
        generate_plots: bool = True,
    ):
        """Run analysis on the latest prediction.

        Can be called with just a natural-language *cmd* string (parsed via
        regex keywords) or with explicit keyword arguments.
        """
        prediction_nc = _find_latest_prediction()
        if not prediction_nc:
            print("\nNo prediction.nc found. Run inference first.")
            return

        # If called with a cmd string, parse keywords out of it
        if cmd:
            cmd_lower = cmd.lower()
            for ch in ALL_CHANNELS:
                if ch in cmd_lower:
                    channel = ch
                    break
            if "all" in cmd_lower:
                analyze_all = True
            if "mse" in cmd_lower or "trend" in cmd_lower:
                mse_trend = True
            if "rank" in cmd_lower:
                rank = True

        print(f"\n>>> Analyzing: {prediction_nc}")
        print(f"    Channel: {'all' if analyze_all else channel}")

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

    # ── Display / informational ─────────────────────────────────────────────

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
        if self.llm and self.llm.available:
            print(f"  Model: {self.llm.model}")
            print(f"  Reasoning: {self._reasoning}")
            print(f"  Context window: {self.llm.context_window:,} tokens")
            hist_tokens = sum(
                self.llm.count_tokens(u or "") + self.llm.count_tokens(b or "")
                for u, b in self.conversation_history
            )
            print(f"  History: {len(self.conversation_history)} turns ({hist_tokens:,} tokens)")
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
