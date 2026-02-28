# Easy Inference

Use this folder for the simplest Surya flow:
1. choose date window,
2. download only required hourly files,
3. run rollout inference,
4. save one `prediction.nc`.

## Running Surya

| Method | Command | Description |
|--------|---------|-------------|
| **Web UI** | `python chat_ui.py` | Browser-based chat interface at http://localhost:7860 |
| **Terminal** | `python llm_interface.py` | Command-line interface with LLM |
| **Script** | `python run_easy_inference.py` | Direct script execution |

## Quick start

```bash
source .venv/bin/activate
bash easy_inference/run_easy_inference.sh
```

Non-interactive defaults:

```bash
bash easy_inference/run_easy_inference.sh --no-prompt
```

---

## Web Chat UI

A modern web interface with chat, image gallery, and inference controls powered by Groq LLM.

### Quick Start

```bash
source .venv/bin/activate
python chat_ui.py
```

Then open http://localhost:7860 in your browser.

### Features

- **Chat Interface**: Ask questions about solar physics, channels, and predictions
- **Run Inference**: Type "run prediction for 2014-10-23" or use the Inference tab
- **Image Gallery**: View generated prediction images
- **Streaming Responses**: Real-time LLM responses via Groq API

### Commands in Chat

| Command | Example |
|---------|---------|
| Run inference | `run prediction for 2014-10-23 12:00` |
| With rollout | `forecast 2014-10-23 with 4 steps` |
| Show images | `show latest images` |
| Questions | `what channel shows solar flares?` |

### Configuration

Set your Groq API key in `.env`:

```
GROQ_API_KEY=your_api_key_here
```

---

## Terminal Interface (No UI)

A terminal-based interface for running predictions and analysis without a web browser.

### Usage

```bash
# Interactive mode
./easy_inference/surya.sh

# Or directly
source .venv/bin/activate
python easy_inference/llm_interface.py

# Disable LLM (regex-only)
python easy_inference/llm_interface.py --no-llm
```

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Input                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    SuryaInterface                            │
│                                                              │
│   ┌─────────────┐          ┌─────────────────────────────┐  │
│   │  Question?  │───YES───►│  LLMHelper (Ollama API)     │  │
│   │  (ends ?)   │          │  llama3.2:1b / qwen2.5      │  │
│   └──────┬──────┘          └─────────────────────────────┘  │
│          │ NO                                                │
│          ▼                                                   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              DateParser (Regex)                      │   │
│   │  "oct 23 2014 from 1pm to 5pm" → 2014-10-23 13:00   │   │
│   │  "2 steps" → rollout_steps=2                         │   │
│   └──────────────────────────┬──────────────────────────┘   │
│                              │                               │
│          ┌───────────────────┼───────────────────┐          │
│          ▼                   ▼                   ▼          │
│   run_inference()     run_analysis()      show_help()       │
└──────────┬───────────────────┬──────────────────────────────┘
           │                   │
           ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                   Subprocess Calls                           │
│                                                              │
│   python run_easy_inference.py     python analyze_predictions.py
│     --start-datetime ...             --prediction-nc ...    │
│     --end-datetime ...               --all-channels         │
│     --rollout-steps N                --rank-channels        │
└─────────────────────────────────────────────────────────────┘
```

### How It Works

**1. Input Classification**

| Input Type | Detection | Handler |
|------------|-----------|---------|
| Questions | Ends with `?` or starts with `what/how/why` | LLM |
| Inference | Contains `run`, `predict`, `forecast` | Regex |
| Analysis | Contains `analyze`, `mse`, `rank` | Regex |
| Unknown | Fallback | LLM |

**2. LLM Integration**

Uses Ollama HTTP API at `localhost:11434`:

```
POST /api/generate
{
  "model": "llama3.2:1b",
  "prompt": "user question",
  "system": "You are a helpful assistant for Surya...",
  "stream": false
}
```

Model auto-selection order:
1. `llama3.2:1b` (1.3 GB) - recommended
2. `qwen2.5:1.5b` (~1 GB)
3. `qwen2.5:0.5b` (397 MB) - fastest

**3. Date/Time Parsing**

Regex extracts dates, times, and rollout steps:

| Input | Result |
|-------|--------|
| `oct 23 2014` | `2014-10-23` |
| `from 1pm to 5pm` | `13:00` to `17:00` |
| `2 steps` | `rollout_steps=2` |

### Example Session

```
[surya] > what channel shows solar flares?
  [Thinking...]

For solar flares, use aia94 (94Å) which captures hot flare plasma at ~6.3 MK.

[surya] > predict oct 23 2014 from 1pm to 5pm with 2 steps
  Start: 2014-10-23 13:00
  End: 2014-10-23 17:00
  Rollout steps: 2

>>> Executing inference...
>>> Inference completed successfully!

[surya] > analyze all channels and rank by mse

CHANNEL RANKING (by average MSE)
========================================
Rank   Channel      Avg MSE
---------------------------------
1      aia94        0.744463
2      aia335       2.439224
...

[surya] > exit
```

### Output Files

```
outputs_24h/
├── prediction.nc           # Predictions + ground truth
├── comparison_aia94.png    # GT vs Pred vs Diff maps
├── comparison_*.png        # Per-channel comparisons
└── mse_trend.png           # MSE over prediction steps
```

### Session Logging

Commands logged to `logs/session_*.log` (JSON lines):

```json
{"timestamp": "...", "event": "command", "input": "predict oct 23 2014"}
{"timestamp": "...", "event": "llm_response", "input": "...", "response": "..."}
```

---

## Config

Edit `easy_inference/config_easy.yaml`.

- Normal users: edit only the top `user:` section.
- Advanced users: optional changes in `advanced:`.

Default and override behavior:

```bash
# Uses easy_inference/config_easy.yaml by default.
python easy_inference/run_easy_inference.py

# Optional: use a different YAML file.
python easy_inference/run_easy_inference.py --config-path /path/to/custom_easy.yaml
```

### Debug mode

Set in `advanced:`:
- `debug_mode: true`
- optional `debug_log_path: "path/to/inference_debug.txt"` (default is `<user.output_dir>/inference_debug.txt`)

When enabled, the text log contains stage timings and per-step diagnostics with line number + UTC timestamp:
- input file read / transform timing
- GT file read timing
- per-step forward / CPU-copy / inverse-transform / write timing
- per-step memory stats (`CUDA` peak/allocated/reserved when available)

---

## Available Channels

### AIA (8 channels)
| Channel | Wavelength | Best For |
|---------|------------|----------|
| aia94 | 94Å | Solar flares |
| aia131 | 131Å | Flares + cooler |
| aia171 | 171Å | Quiet corona |
| aia193 | 193Å | Corona, flares |
| aia211 | 211Å | Active regions |
| aia304 | 304Å | Chromosphere |
| aia335 | 335Å | Active regions |
| aia1600 | 1600Å | Upper photosphere |

### HMI (5 channels)
| Channel | Description |
|---------|-------------|
| hmi_m | Magnetic field magnitude |
| hmi_bx | Magnetic field X |
| hmi_by | Magnetic field Y |
| hmi_bz | Magnetic field Z (vertical) |
| hmi_v | Line-of-sight velocity |

---

## Metrics Notebook

Use `easy_inference/compare_prediction_groundtruth.ipynb` to compare `prediction.nc` vs GT and compute:
- overall metrics (`MSE`, `RMSE`, `MAE`, `bias`, `max_abs_error`)
- per-channel metrics
- per-step metrics
- visual prediction vs ground-truth plots

---

## Files

| File | Purpose |
|------|---------|
| `chat_ui.py` | Web chat interface (Gradio) |
| `llm_interface.py` | Terminal interface |
| `run_easy_inference.py` | Core inference script |
| `analyze_predictions.py` | Standalone analysis script |
| `config_easy.yaml` | Configuration |
| `.env` | API keys (not committed) |
| `logs/` | Session logs |
| `outputs_*/` | Outputs and plots |
