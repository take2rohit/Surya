# Surya Easy Inference — Architecture

## Overview

The Easy Inference subsystem provides a natural-language terminal interface
(and optional Gradio web UI) for running the Surya solar-forecasting model.
Users type queries like *"predict the flare on October 23 2014 with 3 steps"*
and the system detects intent, extracts parameters, asks follow-up questions
when information is incomplete, and dispatches inference or analysis scripts.

A local GPT-OSS model (20B or 120B) handles all NLU via **Harmony format**
streaming. When the LLM is disabled (`--no-llm`), the system falls back to
regex-based keyword matching and ISO-date parsing.

---

## Module Structure

```
easy_inference/
├── llm_interface.py    Entry point, re-exports, main()
├── llm_client.py       LocalLLMClient, SessionLogger, model selection, constants
├── llm_agents.py       Intent enum, dataclasses, LLMHelper, DateParser, prompt loading
├── surya_runner.py     SuryaInterface (REPL, routing, execution, display)
├── chat_ui.py          Gradio web interface (imports from llm_interface.py)
├── run_easy_inference.py   Surya model inference script
├── analyze_predictions.py  Post-inference analysis script
├── config_easy.yaml    Runtime configuration
├── prompts/            Agent prompt templates
│   ├── system_prompt.txt
│   ├── intent_agent.txt
│   ├── params_agent.txt
│   ├── followup_agent.txt
│   └── response_agent.txt
└── logs/               JSON-lines session logs
```

### Dependency Graph (no cycles)

```
llm_client  ←──  llm_agents  ←──  surya_runner  ←──  llm_interface
                                                  ←──  chat_ui
```

- **llm_client.py** has no internal imports (only stdlib + HF/Harmony).
- **llm_agents.py** imports `LocalLLMClient` and constants from `llm_client`.
- **surya_runner.py** imports from both `llm_client` and `llm_agents`.
- **llm_interface.py** re-exports everything from the three modules.
- **chat_ui.py** imports only from `llm_interface` (backward compatible).

---

## Agent Pipeline

Every user command flows through a multi-agent pipeline:

```
User Input
    │
    ▼
┌──────────────┐
│ Intent Agent │  Classifies into: INFERENCE | ANALYSIS | QUESTION | CHAT
└──────┬───────┘
       │
       ├── INFERENCE ──► Params Agent ──► InferenceParams
       │                      │
       │                 ┌────┴────┐
       │                 │complete?│
       │                 └────┬────┘
       │                 yes  │  no
       │                  ▼      ▼
       │              Execute  Follow-up Agent → ask user → loop back
       │
       ├── ANALYSIS ──► regex keyword extraction → run analyze_predictions.py
       │
       └── QUESTION/CHAT ──► Response Agent (streaming) → display
```

### Agent Descriptions

| Agent | Prompt File | Purpose |
|-------|-------------|---------|
| Intent Agent | `intent_agent.txt` | Classify user command into one of 4 intents |
| Params Agent | `params_agent.txt` | Extract structured JSON: `{start_datetime, rollout_steps, missing, understood}` |
| Follow-up Agent | `followup_agent.txt` | Generate a natural-language question for missing parameters |
| Response Agent | `response_agent.txt` / `system_prompt.txt` | General Q&A and friendly confirmations |

---

## Streaming Architecture

### Harmony Format

GPT-OSS uses **Harmony**, a structured token format with named channels:

- **analysis** — internal reasoning (shown as dim "thinking" text)
- **commentary** — intermediate commentary (ignored)
- **final** — the user-facing answer

Standard HF `TextIteratorStreamer` decodes all tokens as flat text, which
produces garbage when channels are interleaved. The solution:

### Monkey-Patching (`_patch_streamer`)

`LocalLLMClient._patch_streamer()` replaces `streamer.put()` with a custom
function that:

1. Intercepts raw token IDs before HF decoding
2. Routes them through `openai_harmony.StreamableParser`
3. Emits only `analysis` and `final` channel content
4. Inserts sentinel strings (`\x00T\x00`, `\x00F\x00`) at channel transitions

The display layer in `generate()` uses these sentinels to switch ANSI colors:
- Thinking: `\033[90m` (dark gray)
- Final answer: configurable per-agent (green for responses, white for params)

### Non-Streaming Path

For silent internal calls (`stream=False`), `_parse_harmony_output()` decodes
the full output tensor via `parse_messages_from_completion_tokens()` and returns
only the `final` channel text.

---

## Context Window Management

### Token Budget

```
context_window (from model config)
  - _GENERATION_RESERVE (1024 tokens)
  = max_input_tokens
    - system prompt tokens
    - current prompt tokens
    - overhead (~20 tokens)
    = available for history
```

### History Trimming Algorithm

In `_build_messages()`:

1. Count fixed tokens (system + current prompt + overhead)
2. Walk conversation history from **newest to oldest**
3. Accumulate token counts per (user, assistant) pair
4. Stop adding when the next pair would exceed the budget
5. Result: most-recent history that fits within the window

History is also bounded to 20 turns maximum in `SuryaInterface.process()`.

---

## Data Flow Diagrams

### Inference Path

```
User: "predict october 23 2014 with 4 steps"
  │
  ├─ Intent Agent → "INFERENCE"
  ├─ Params Agent → InferenceParams(start_datetime="2014-10-23 10:00",
  │                                  rollout_steps=4, missing=[], ...)
  ├─ is_complete → True
  │
  └─ _execute_inference()
       ├─ Build args: [python, run_easy_inference.py, --start-datetime, ...]
       ├─ Print inference plan (timeline table)
       └─ subprocess.run(args, cwd=ROOT_DIR)
```

### Multi-Turn Inference Path

```
User: "run a prediction for the big flare"
  │
  ├─ Intent Agent → "INFERENCE"
  ├─ Params Agent → InferenceParams(start_datetime=None, missing=["date","year"])
  ├─ is_complete → False
  ├─ Follow-up Agent → "Which date? Surya has data from 2010-present."
  └─ Store PendingInference(original_cmd=..., params=...)

User: "october 23 2014"
  │
  ├─ complete_pending_inference()
  │   ├─ Combine: "run a prediction for the big flare october 23 2014"
  │   ├─ Re-extract → InferenceParams(start_datetime="2014-10-23 10:00", ...)
  │   └─ _execute_inference()
```

### Analysis Path

```
User: "compare all channels and rank by mse"
  │
  ├─ Intent Agent → "ANALYSIS"
  ├─ Keyword regex: all=True, rank=True, mse_trend=False
  ├─ _find_latest_prediction() → outputs_*/prediction.nc
  └─ subprocess.run([analyze_predictions.py, --all-channels, --rank-channels])
```

### Chat Path

```
User: "what channel shows solar flares?"
  │
  ├─ Intent Agent → "QUESTION"
  ├─ Response Agent (streaming) → "The AIA 94Å channel..."
  └─ Append to conversation_history
```

---

## Configuration Reference

### `config_easy.yaml`

Runtime configuration for the Surya model (paths, checkpoint, data source, etc.).
Read by `run_easy_inference.py` and `analyze_predictions.py`.

### Constants (`llm_client.py`)

| Constant | Description |
|----------|-------------|
| `ROOT_DIR` | Project root (parent of `easy_inference/`) |
| `EASY_DIR` | `easy_inference/` directory |
| `CONFIG_PATH` | Path to `config_easy.yaml` |
| `LOG_DIR` | Session log directory |
| `SCOPE_PATH` | Path to `SCOPE.md` |
| `CHANNELS` | Dict mapping instrument → channel list |
| `ALL_CHANNELS` | Flat list of all 13 channel names |
| `CHANNEL_INFO` | Channel name → human description |
| `AVAILABLE_MODELS` | List of model configs for selection menu |

### Model Selection

`select_model()` displays an interactive menu of 6 options (2 models × 3
reasoning levels). Returns `(hf_id, reasoning)` or `None` for regex-only mode.

---

## Class / Method Reference

### `llm_client.py`

**`SessionLogger`**
- `log(event, data)` — append JSON-lines entry
- `get_log_path()` — return log file path

**`LocalLLMClient`**
- `__init__(model, reasoning)` — load model + tokenizer + Harmony encoding
- `check_connection()` → `(bool, str)`
- `generate(prompt, system, stream, history, agent, final_color)` → `str | None`
- `chat(message, history, system)` → `Generator[str]` (for Gradio)
- `count_tokens(text)` → `int`
- `display_name` — property, e.g. `"GPT-OSS-120B"`

### `llm_agents.py`

**`Intent`** — Enum: `INFERENCE`, `ANALYSIS`, `QUESTION`, `CHAT`

**`InferenceParams`** — Dataclass
- `start_datetime`, `rollout_steps`, `missing`, `understood`
- `is_complete` — property: True when all required info present
- `from_dict(d)` — classmethod: build from LLM-extracted dict

**`PendingInference`** — Dataclass: `original_cmd`, `params`, `dry_run`, `skip_download`

**`LLMHelper`**
- `detect_intent(user_input)` → `Intent`
- `extract_params(user_input, max_retries)` → `InferenceParams | None`
- `generate_followup(query, understood, missing)` → `str | None`
- `generate_response(date, rollout)` → `str | None`
- `ask(user_input, stream, history)` → `str | None`

**`DateParser`** — Static methods for regex-only fallback
- `parse(text)` → `(start_dt, end_dt)` or `(None, None)`
- `parse_rollout(text)` → `int | None`

### `surya_runner.py`

**`SuryaInterface`**
- `__init__(use_llm, llm_model, reasoning)`
- `run()` — REPL loop
- `process(cmd)` — unified routing (quick commands + LLM intent)
- `run_inference(cmd)` — LLM-based parameter extraction + execution
- `run_analysis(cmd, *, channel, analyze_all, rank, mse_trend, generate_plots)`
- `complete_pending_inference(additional_info)` → `bool`
- `show_welcome()`, `show_help()`, `show_examples()`, `show_config()`
- `show_status()`, `list_channels()`, `handle_unknown(cmd)`

**`_find_latest_prediction()`** → `Path | None` — module-level helper

---

## Multi-Turn Conversation State Machine

```
                    ┌──────────┐
                    │  IDLE    │
                    └────┬─────┘
                         │ user input
                         ▼
                    ┌──────────┐
                    │ PROCESS  │
                    └────┬─────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         INFERENCE   ANALYSIS   CHAT/Q
              │          │          │
              ▼          │          │
        params_ok?       │          │
         yes  no         │          │
          │    │         │          │
          │    ▼         │          │
          │  PENDING     │          │
          │    │         │          │
          │    │ user    │          │
          │    │ reply   │          │
          │    ▼         │          │
          │  combine +   │          │
          │  re-extract  │          │
          │    │         │          │
          ▼    ▼         ▼          ▼
        ┌──────────────────────────────┐
        │          IDLE                │
        └──────────────────────────────┘
```

The `pending_inference` field holds a `PendingInference` dataclass when the
system is waiting for the user to supply missing information. Any subsequent
input first checks this field; if set, `complete_pending_inference()` combines
the original and new input, re-extracts parameters, and either executes or
stays in the pending state.

Saying "cancel", "nevermind", or "reset" clears the pending state.
