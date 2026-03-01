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

Module layout:
    llm_client.py   - LocalLLMClient, SessionLogger, model selection, constants
    llm_agents.py   - Intent enum, InferenceParams, LLMHelper, DateParser
    surya_runner.py - SuryaInterface (REPL, routing, execution, display)
    llm_interface.py - This file: entry point, re-exports, main()
"""

from __future__ import annotations

# ── Re-exports (preserves ``from llm_interface import ...`` for chat_ui.py) ──

from llm_client import (  # noqa: F401
    ALL_CHANNELS,
    AVAILABLE_MODELS,
    CHANNEL_INFO,
    CHANNELS,
    CONFIG_PATH,
    EASY_DIR,
    LOG_DIR,
    ROOT_DIR,
    SCOPE_PATH,
    LocalLLMClient,
    SessionLogger,
    select_model,
)

from llm_agents import (  # noqa: F401
    PROMPTS_DIR,
    SYSTEM_PROMPT_PATH,
    DateParser,
    InferenceParams,
    Intent,
    LLMHelper,
    PendingInference,
)

from surya_runner import SuryaInterface  # noqa: F401

__all__ = [
    # Constants
    "ROOT_DIR",
    "EASY_DIR",
    "CONFIG_PATH",
    "LOG_DIR",
    "SCOPE_PATH",
    "CHANNELS",
    "ALL_CHANNELS",
    "CHANNEL_INFO",
    "AVAILABLE_MODELS",
    "PROMPTS_DIR",
    "SYSTEM_PROMPT_PATH",
    # Client & logging
    "LocalLLMClient",
    "SessionLogger",
    "select_model",
    # Agents & types
    "Intent",
    "InferenceParams",
    "PendingInference",
    "LLMHelper",
    "DateParser",
    # Interface
    "SuryaInterface",
]


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
