# SuryaGPT - Solar Forecasting with Natural Language

A natural language interface for **Surya**, a 366M-parameter foundation model trained on NASA's Solar Dynamics Observatory (SDO) data. Talk to Surya in plain English — ask questions, run predictions, and analyze results.

Powered by **GPT-OSS** (20B / 120B) running locally on GPU with Harmony format streaming.

---

## Quick Start

```bash
source .venv/bin/activate
bash easy_inference/surya.sh
```

Select a model (press Enter for default), then start talking:

```
SuryaGPT > Run prediction for October 23 2014 at 10am with 5 steps
```

---

## Interfaces

| Method | Command | Description |
|--------|---------|-------------|
| **Terminal** | `bash easy_inference/surya.sh` | Interactive CLI with streaming LLM |
| **Web UI** | `bash easy_inference/run_chat_ui.sh` | Browser chat + image gallery at localhost:7860 |
| **Script** | `bash easy_inference/run_easy_inference.sh` | Direct inference (no LLM) |

---

## Agent Architecture

SuryaGPT uses a multi-agent pipeline. Every user query first goes through the **Intent Agent**, which classifies it into one of four intents. Each intent triggers a different workflow through specialized agents.

### Master Flow

```
                          ┌─────────────────────┐
                          │     User Input       │
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │    Intent Agent      │
                          │  (classifies query)  │
                          └──────────┬──────────┘
                                     │
                  ┌──────────────────┼──────────────────┐
                  │                  │                   │
                  ▼                  ▼                   ▼
          ┌───────────┐     ┌──────────────┐    ┌──────────────┐
          │ INFERENCE  │     │   ANALYSIS   │    │ QUESTION /   │
          │            │     │              │    │ CHAT         │
          └─────┬─────┘     └──────┬───────┘    └──────┬───────┘
                │                  │                    │
                ▼                  ▼                    ▼
        Params Agent        run_analysis()       Response Agent
                │                  │                    │
                ▼                  ▼                    ▼
     ┌──── complete? ────┐   analyze_predictions.py   Green answer
     │                   │                             to user
    YES                  NO
     │                   │
     ▼                   ▼
  Surya Model      Follow-up Agent
  (inference)      (asks for missing info)
     │                   │
     ▼                   ▼
  prediction.nc    Pending state
                   (waits for next input)
```

---

### INFERENCE Flow (detailed)

When the Intent Agent classifies a query as `INFERENCE`, the system extracts parameters and validates them before running the model.

```
┌──────────────────────────────────────────────────────────────────────┐
│ INFERENCE FLOW                                                       │
│                                                                      │
│  User: "predict october 23 2014 at 10am with 3 steps"              │
│                                                                      │
│  1. Intent Agent ──────────────────────────────────────► INFERENCE   │
│                                                                      │
│  2. Params Agent ──────────────────────────────────────► JSON        │
│     Extracts: date, year, time, rollout                             │
│     Output: {"start_datetime": "2014-10-23 10:00",                  │
│              "rollout_steps": 3, "missing": []}                     │
│                                                                      │
│  3. Validation ────────────────────────────────────────► Check       │
│     ├─ Has date?  ✓                                                 │
│     ├─ Has year?  ✓                                                 │
│     ├─ Has time?  ✓                                                 │
│     └─ missing == [] ?  ✓                                           │
│                                                                      │
│  4. Show Inference Plan:                                            │
│     Step 1/4  [GT Oct 23 09:00, GT Oct 23 10:00] -> Pred Oct 23 11 │
│     Step 2/4  [GT Oct 23 10:00, Pred Oct 23 11:00] -> Pred Oct 23  │
│     Step 3/4  [Pred Oct 23 11:00, Pred Oct 23 12:00] -> Pred ...   │
│     Step 4/4  [Pred Oct 23 12:00, Pred Oct 23 13:00] -> Pred ...   │
│                                                                      │
│  5. Execute: run_easy_inference.py                                  │
│     └─► prediction.nc                                               │
└──────────────────────────────────────────────────────────────────────┘
```

### INFERENCE Flow — Missing Parameters

When required info is missing, the Follow-up Agent asks and the system waits:

```
┌──────────────────────────────────────────────────────────────────────┐
│ INFERENCE FLOW (incomplete query)                                    │
│                                                                      │
│  User: "run prediction for October 23"                              │
│                                                                      │
│  1. Intent Agent ──────────────────────────────────────► INFERENCE   │
│                                                                      │
│  2. Params Agent ──────────────────────────────────────► JSON        │
│     Output: {"start_datetime": null,                                │
│              "missing": ["year", "time"]}                           │
│                                                                      │
│  3. Validation ────────────────────────────────────────► FAIL        │
│     ├─ Has year?  ✗                                                 │
│     └─ Has time?  ✗                                                 │
│                                                                      │
│  4. Follow-up Agent ───────────────────────────────────► Question    │
│     "I need the year and start time! e.g. '2014 at 10am'"          │
│                                                                      │
│  5. Save pending state (original_cmd + partial params)              │
│                                                                      │
│  ─── Next user input ───                                            │
│                                                                      │
│  User: "2014 at 10am with 3 steps"                                 │
│                                                                      │
│  6. Combine: "run prediction for October 23" + "2014 at 10am ..."  │
│  7. Params Agent re-extracts ──────────────────────────► Complete    │
│  8. Execute inference                                               │
└──────────────────────────────────────────────────────────────────────┘
```

### ANALYSIS Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│ ANALYSIS FLOW                                                        │
│                                                                      │
│  User: "analyze all channels and rank by MSE"                       │
│                                                                      │
│  1. Intent Agent ──────────────────────────────────────► ANALYSIS    │
│                                                                      │
│  2. Keyword parsing (no LLM needed):                                │
│     ├─ "all" in query?  → --all-channels                            │
│     ├─ "rank" in query? → --rank-channels                           │
│     ├─ "mse"/"trend"?   → --show-mse-trend                         │
│     └─ channel name?    → --channel <name>                          │
│                                                                      │
│  3. Find latest prediction.nc in outputs_*/                         │
│                                                                      │
│  4. Execute: analyze_predictions.py                                 │
│     └─► comparison_*.png, mse_trend.png, channel ranking            │
└──────────────────────────────────────────────────────────────────────┘
```

### QUESTION / CHAT Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│ QUESTION / CHAT FLOW                                                 │
│                                                                      │
│  User: "What channel shows solar flares?"                           │
│                                                                      │
│  1. Intent Agent ──────────────────────────────────────► QUESTION    │
│                                                                      │
│  2. Response Agent                                                  │
│     ├─ System prompt: prompts/system_prompt.txt                     │
│     ├─ Conversation history (up to 20 turns)                        │
│     └─ Streams: [thinking] gray... [/thinking]                      │
│                  Green final answer                                  │
│                                                                      │
│  3. Save to conversation history for context                        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Agents Reference

### Intent Agent

| | |
|---|---|
| **Prompt file** | `prompts/intent_agent.txt` |
| **Input** | Raw user query |
| **Output** | One word: `INFERENCE`, `ANALYSIS`, `QUESTION`, or `CHAT` |
| **Color** | Light gray (thinking + output) |
| **Streaming** | Yes |

### Params Agent

| | |
|---|---|
| **Prompt file** | `prompts/params_agent.txt` |
| **Input** | User query (inference requests only) |
| **Output** | JSON: `{start_datetime, rollout_steps, missing, understood}` |
| **Color** | Light gray (thinking + output) |
| **Streaming** | Yes |
| **Retries** | Up to 2 attempts if JSON parse fails |

### Follow-up Agent

| | |
|---|---|
| **Prompt file** | `prompts/followup_agent.txt` |
| **Input** | Original query + what was understood + list of missing fields |
| **Output** | Friendly question asking for missing info |
| **Color** | Light gray (thinking + output) |
| **Streaming** | Yes |
| **Triggers** | Only when Params Agent reports missing required fields |

### Response Agent

| | |
|---|---|
| **Prompt file** | `prompts/system_prompt.txt` (system) + `prompts/response_agent.txt` (pre-inference) |
| **Input** | User question + conversation history (up to 20 turns) |
| **Output** | Natural language answer |
| **Color** | Green final answer, gray thinking |
| **Streaming** | Yes |
| **Used for** | QUESTION and CHAT intents |

---

## Required Parameters for Inference

All three are **mandatory**. If any is missing, the Follow-up Agent will ask.

| Parameter | Required | Examples | What happens if missing |
|-----------|----------|---------|------------------------|
| **Date** | Yes | "october 23", "may 20", "2020-05-20" | Follow-up asks for date |
| **Year** | Yes | "2014", "twenty twenty" | Follow-up asks for year |
| **Start time** | Yes | "at 10am", "14:00", "noon" | Follow-up asks for time |
| **Rollout steps** | No | "5 steps", "10 hours ahead" | Defaults to 0 (single 1-hour prediction) |

---

## Prompt Files

All agent prompts live in `easy_inference/prompts/` and can be edited without touching code:

```
easy_inference/prompts/
├── intent_agent.txt      # Intent classification prompt
├── params_agent.txt      # Date/time/rollout extraction prompt
├── followup_agent.txt    # Missing-info follow-up question prompt
├── response_agent.txt    # Pre-inference response generation prompt
└── system_prompt.txt     # System prompt for Response Agent (chat/questions)
```

---

## Streaming Output Format

All agents stream their output to the terminal with color coding:

```
[Agent Name] [thinking] <gray analysis text> [/thinking]
<final output in agent's color>
```

| Element | Color | ANSI Code |
|---------|-------|-----------|
| Agent tag `[Intent Agent]` | Dark gray | `\033[90m` |
| `[thinking]` content | Dark gray | `\033[90m` |
| Intent/Params/Follow-up final output | Light gray | `\033[37m` |
| Response Agent final output | Green | `\033[32m` |

**Example terminal session:**
```
SuryaGPT > What channel shows solar flares?

[Intent Agent] [thinking] classifying user intent [/thinking]
QUESTION
[Response Agent] [thinking] looking up channel info [/thinking]
For solar flares, aia94 (94Å) is the best channel. It captures hot
flare plasma at ~6.3 million Kelvin...
```

---

## Example Queries

### Inference
```
"Run prediction for october 23 2014 at 10am"
"Predict 10 hours ahead starting at noon on 2020-05-20"
"Forecast the X1.6 flare day at 14:00 with 5 rollout steps"
"Show me what happens on october 12 2016 at 8pm"
"2014-10-23 14:00 with 3 steps, skip download"
```

### Analysis
```
"Analyze the latest predictions"
"Compare all channels and rank by MSE"
"Show me the MSE trend for aia94"
"Which channel does the model predict best?"
"How does prediction accuracy change over time?"
```

### Questions
```
"What channel shows solar flares?"
"Tell me about the AIA 94 angstrom wavelength"
"What's the difference between hmi_bz and hmi_m?"
"How does the model work?"
"What data does Surya use?"
```

---

## Model Selection

On startup, SuryaGPT shows an interactive model menu:

```
  Available Modes:
  ----------------------------------------------------------------------
    #  Model          Reasoning  Description              Cached
  ----------------------------------------------------------------------
    1  GPT-OSS-20B    low        Fast, single GPU                  (default)
    2  GPT-OSS-20B    medium     Balanced, single GPU
    3  GPT-OSS-20B    high       Deep reasoning, single GPU
    4  GPT-OSS-120B   low        Fast, minimal thinking
    5  GPT-OSS-120B   medium     Balanced
    6  GPT-OSS-120B   high       Deep reasoning
  ----------------------------------------------------------------------
    0 / q  = No LLM (regex mode)

  Select mode [1]:
```

| Model | Size | GPUs | Best For |
|-------|------|------|----------|
| GPT-OSS-20B | ~20B params | 1 GPU | Fast responses, interactive use |
| GPT-OSS-120B | ~120B params | Multi-GPU | Complex reasoning, detailed analysis |

| Reasoning | Thinking Depth | Speed |
|-----------|---------------|-------|
| low | Minimal internal analysis | Fastest |
| medium | Balanced thinking | Moderate |
| high | Deep multi-step reasoning | Slowest |

CLI flags to skip the menu:

```bash
python llm_interface.py --model gpt-oss-20b --reasoning low
python llm_interface.py --model gpt-oss-120b --reasoning high
python llm_interface.py --no-llm    # Regex-only, no GPU for LLM
```

---

## Channels

### AIA — Extreme Ultraviolet (8 channels)

| Channel | Wavelength | Temperature | Best For |
|---------|-----------|-------------|----------|
| `aia94` | 94 Å | ~6.3 MK | **Solar flares**, hot plasma |
| `aia131` | 131 Å | ~10 MK + 0.4 MK | Flare plasma + cooler regions |
| `aia171` | 171 Å | ~0.6 MK | Quiet corona, coronal loops |
| `aia193` | 193 Å | ~1.6 MK | Corona + hot flares (general purpose) |
| `aia211` | 211 Å | ~2 MK | Active regions |
| `aia304` | 304 Å | ~50,000 K | Chromosphere, prominences |
| `aia335` | 335 Å | ~2.5 MK | Active regions |
| `aia1600` | 1600 Å | — | Upper photosphere, UV continuum |

### HMI — Magnetic Field (5 channels)

| Channel | Measures |
|---------|----------|
| `hmi_m` | Total magnetic field magnitude |
| `hmi_bx` | Magnetic field X component |
| `hmi_by` | Magnetic field Y component |
| `hmi_bz` | Magnetic field Z (vertical) |
| `hmi_v` | Line-of-sight velocity (Doppler) |

---

## Inference Progress Output

During inference, each step shows human-readable input/output with MSE:

```
[progress] Step 1/4  [GT Oct 23 10:00, GT Oct 23 11:00] -> Pred Oct 23 12:00  MSE=5351.09
[progress] Step 2/4  [GT Oct 23 11:00, Pred Oct 23 12:00] -> Pred Oct 23 13:00  MSE=5374.81
[progress] Step 3/4  [Pred Oct 23 12:00, Pred Oct 23 13:00] -> Pred Oct 23 14:00  MSE=5436.44
[progress] Step 4/4  [Pred Oct 23 13:00, Pred Oct 23 14:00] -> Pred Oct 23 15:00  MSE=5448.03
```

- **GT** = Ground Truth (real observed SDO data used as input)
- **Pred** = Model Prediction (autoregressive — feeds back into next step)
- MSE increases with each step as prediction errors compound

---

## Output Files

```
easy_inference/outputs_24h/
├── prediction.nc           # NetCDF: predictions + ground truth for all 13 channels
├── comparison_aia94.png    # Side-by-side: GT vs Prediction vs Difference
├── comparison_*.png        # Per-channel comparison plots
└── mse_trend.png           # MSE over prediction steps (all channels)
```

## Session Logging

All commands and responses are logged to `easy_inference/logs/session_*.log` (JSON lines):

```json
{"timestamp": "2024-10-23T14:30:00", "event": "command", "input": "predict oct 23 2014 at 10am"}
{"timestamp": "2024-10-23T14:30:05", "event": "inference_start", "start_datetime": "2014-10-23 10:00"}
{"timestamp": "2024-10-23T14:31:12", "event": "inference_end", "returncode": 0}
```

---

## Configuration

Edit `easy_inference/config_easy.yaml`:

```yaml
user:
  start_datetime: "2014-10-23 10:00:00"
  end_datetime: ""          # Leave empty to auto-calculate
  output_dir: easy_inference/outputs_24h
  rollout_steps: 5
```

### Debug Mode

```yaml
advanced:
  debug_mode: true
  debug_log_path: "easy_inference/inference_debug.txt"
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.8.0 | Model inference with `torch.compile` |
| `triton` | 3.4.0 | Inductor backend (must match torch) |
| `kernels` | 0.12+ | MXFP4 quantization kernels for GPT-OSS |
| `transformers` | latest | GPT-OSS model loading (HuggingFace) |
| `openai_harmony` | latest | Harmony format streaming for GPT-OSS |
| `gradio` | latest | Web chat UI |
| `xarray`, `h5netcdf` | latest | NetCDF prediction output |

---

## Files

| File | Purpose |
|------|---------|
| `llm_interface.py` | Terminal interface with multi-agent LLM pipeline |
| `chat_ui.py` | Web chat interface (Gradio) |
| `run_easy_inference.py` | Core Surya inference engine |
| `analyze_predictions.py` | Post-inference analysis and visualization |
| `config_easy.yaml` | User + advanced configuration |
| `prompts/` | All agent prompt files (editable without code changes) |
| `surya.sh` | Launch script (terminal) |
| `run_llm_interface.sh` | Launch script (terminal, activates venv) |
| `run_chat_ui.sh` | Launch script (web UI) |
| `run_easy_inference.sh` | Launch script (direct inference) |
