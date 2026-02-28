#!/usr/bin/env python3
"""
Surya Chat UI - Beautiful Interface with Groq LLM & Image Support

A modern web interface for solar forecasting with:
- Chat interface with streaming responses via Groq
- Inline image display and upload
- Integration with Surya model
- Real-time connection status

Usage:
    source .venv/bin/activate
    python chat_ui.py
"""

import base64
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
import gradio as gr

# Load environment variables from .env file
load_dotenv()
from PIL import Image

# Project paths
ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config_easy.yaml"
LOG_DIR = ROOT_DIR / "logs"

# Channel information
CHANNEL_INFO = {
    "aia94": ("94Å", "Hot flare plasma ~6.3 MK", "#FF6B6B"),
    "aia131": ("131Å", "Flare plasma + cooler", "#FF8E53"),
    "aia171": ("171Å", "Quiet corona ~0.6 MK", "#4ECDC4"),
    "aia193": ("193Å", "Corona and hot flares", "#45B7D1"),
    "aia211": ("211Å", "Active regions ~2 MK", "#96CEB4"),
    "aia304": ("304Å", "Chromosphere ~50,000 K", "#FFEAA7"),
    "aia335": ("335Å", "Active regions ~2.5 MK", "#DDA0DD"),
    "aia1600": ("1600Å", "Upper photosphere", "#C9B1FF"),
    "hmi_m": ("HMI M", "Total magnetic field", "#74B9FF"),
    "hmi_bx": ("HMI Bx", "Magnetic field X", "#A29BFE"),
    "hmi_by": ("HMI By", "Magnetic field Y", "#FD79A8"),
    "hmi_bz": ("HMI Bz", "Magnetic field Z (vertical)", "#FDCB6E"),
    "hmi_v": ("HMI V", "Line-of-sight velocity", "#E17055"),
}


class GroqClient:
    """Client for Groq LLM inference API with streaming."""

    DEFAULT_MODEL = "openai/gpt-oss-120b"

    SYSTEM_PROMPT = """You are Surya Assistant, an AI helper for the Surya solar forecasting system.

Surya is a 366M parameter foundation model trained on SDO (Solar Dynamics Observatory) data.
It predicts solar observations across multiple wavelengths and magnetic field measurements.

## SURYA CAPABILITIES - WHAT IT CAN DO

### Inference & Prediction
- **Run solar predictions**: Generate forecasts for any date from 2010-present
- **Multi-step rollout**: Predict 1-24 hours ahead (each step = 1 hour)
- **13 channel output**: Predicts all AIA and HMI channels simultaneously
- **Auto-download data**: Fetches required SDO data from AWS S3 automatically
- **Save to NetCDF**: Outputs prediction.nc with predictions + ground truth

### Analysis & Metrics
- **MSE (Mean Squared Error)**: Compute per-step and average MSE
- **RMSE, MAE, bias**: Available in analysis notebook
- **Channel ranking**: Rank channels by prediction accuracy
- **MSE trend plots**: Visualize error over prediction steps
- **Comparison plots**: Side-by-side Ground Truth vs Prediction vs Difference

### Visualization
- **Per-channel images**: Generate PNG comparisons for each channel
- **MSE trend charts**: Plot error progression over time
- **Gallery view**: Browse all generated images by channel

### Data Handling
- **Date range selection**: Specify start datetime and rollout steps
- **Automatic end time**: Calculates required data window
- **Missing data handling**: Continues with available data, marks missing as NaN
- **Ground truth comparison**: Loads GT for loss computation when available

## WHAT SURYA CANNOT DO
- Real-time predictions (requires pre-downloaded data)
- Train or fine-tune the model
- Process non-SDO data sources
- Predict beyond available SDO data (2010-present)
- Sub-hourly predictions (minimum 1-hour steps)

## CHAT COMMANDS

| Command | Example | What it does |
|---------|---------|--------------|
| Run prediction | `run prediction for 2014-10-23 12:00` | Runs inference for that date |
| With rollout | `predict 2014-10-23 with 4 steps` | 4-hour forecast (5 predictions) |
| Show images | `show images` | Lists latest prediction images |
| Ask questions | `what is MSE?` | Answers solar/Surya questions |

## AVAILABLE CHANNELS (13 total)

**AIA EUV Imaging (8 channels):**
- aia94: Hot flare plasma ~6.3 MK (best for solar flares)
- aia131: Flare plasma + cooler material
- aia171: Quiet corona ~0.6 MK (coronal loops)
- aia193: Corona and hot flares ~1.2 MK
- aia211: Active regions ~2 MK
- aia304: Chromosphere ~50,000 K
- aia335: Active regions ~2.5 MK
- aia1600: Upper photosphere

**HMI Magnetic Field (5 channels):**
- hmi_m: Total magnetic field magnitude
- hmi_bx: Magnetic field X component
- hmi_by: Magnetic field Y component
- hmi_bz: Magnetic field Z (vertical) - best for sunspot polarity
- hmi_v: Line-of-sight velocity (Dopplergrams)

## CHANNEL RECOMMENDATIONS

| Use Case | Recommended Channels |
|----------|---------------------|
| Solar flares | aia94, aia131 |
| Corona structure | aia171, aia193 |
| Active regions | aia211, aia335 |
| Chromosphere | aia304 |
| Magnetic field | hmi_bz, hmi_m |
| Sunspot analysis | hmi_bz, aia1600 |

## METRICS EXPLAINED
- **MSE**: Mean Squared Error - average of (prediction - ground_truth)^2
- **RMSE**: Root MSE - square root of MSE, same units as data
- **MAE**: Mean Absolute Error - average of |prediction - ground_truth|
- **Bias**: Mean of (prediction - ground_truth) - shows systematic over/under prediction

Be concise and scientifically accurate. When users ask "what can you do" or "what are your capabilities", explain the inference, analysis, and visualization features above."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = self.DEFAULT_MODEL
        self.client = None
        self.available = False
        self.error_message = None

        self._init_client()

    def _init_client(self):
        """Initialize the Groq client."""
        if not self.api_key:
            self.error_message = "No API key provided"
            return

        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            # Test connection
            self.client.models.list()
            self.available = True
            self.error_message = None
        except Exception as e:
            self.error_message = str(e)
            self.available = False

    def check_connection(self) -> tuple[bool, str]:
        """Check if Groq is connected and return status message."""
        if self.available:
            return True, f"Connected to Groq ({self.model})"
        return False, f"Disconnected: {self.error_message or 'Unknown error'}"

    def chat(
        self,
        message: str,
        history: list[tuple[str, str]],
    ) -> Generator[str, None, None]:
        """Stream a chat response from Groq."""
        if not self.available:
            yield f"Groq API not available: {self.error_message}"
            return

        # Build messages from history
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        for user_msg, bot_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})

        messages.append({"role": "user", "content": message})

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                stream=True,
            )

            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    yield full_response

        except Exception as e:
            yield f"Error: {str(e)}"


class SuryaRunner:
    """Handle Surya inference operations."""

    def __init__(self):
        self.last_output_dir: Path | None = None

    def run_inference(
        self,
        start_datetime: str,
        rollout_steps: int = 1,
        dry_run: bool = False,
    ) -> tuple[str, list[str]]:
        """Run Surya inference and return status and generated images."""
        args = [
            sys.executable,
            str(ROOT_DIR / "run_easy_inference.py"),
            "--config-path", str(CONFIG_PATH),
            "--no-prompt",
            "--start-datetime", start_datetime,
            "--rollout-steps", str(rollout_steps),
        ]

        if dry_run:
            args.append("--dry-run")

        try:
            result = subprocess.run(
                args,
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode == 0:
                output_dirs = sorted(
                    ROOT_DIR.glob("outputs_*"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )

                images = []
                if output_dirs:
                    self.last_output_dir = output_dirs[0]
                    for img in sorted(self.last_output_dir.glob("*.png")):
                        images.append(str(img))

                return f"Inference completed!\nOutput: {self.last_output_dir}", images
            else:
                return f"Inference failed:\n{result.stderr[:500]}", []

        except subprocess.TimeoutExpired:
            return "Inference timed out after 10 minutes.", []
        except Exception as e:
            return f"Error: {str(e)}", []

    def get_latest_images(self, channel: str | None = None) -> list[str]:
        """Get images from the latest output directory."""
        output_dirs = sorted(
            ROOT_DIR.glob("outputs_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not output_dirs:
            return []

        latest = output_dirs[0]
        images = []

        for img in sorted(latest.glob("*.png")):
            if channel is None or channel in img.name:
                images.append(str(img))

        return images[:20]


# Initialize clients
groq_client = GroqClient()
surya_runner = SuryaRunner()


def get_status_html() -> str:
    """Generate HTML for connection status."""
    connected, message = groq_client.check_connection()

    if connected:
        status_class = "status-online"
        icon = "🟢"
    else:
        status_class = "status-offline"
        icon = "🔴"

    return f"""
    <div class="status-container {status_class}">
        <span class="status-icon">{icon}</span>
        <span class="status-text">{message}</span>
    </div>
    """


def process_message(
    message: dict,
    history: list,
) -> Generator[list, None, None]:
    """Process a chat message and stream the response."""
    text = message.get("text", "").strip()
    files = message.get("files", [])

    if not text:
        yield history
        return

    # Convert history from messages format to tuples for Groq API
    history_tuples = []
    i = 0
    while i < len(history):
        if history[i].get("role") == "user":
            user_msg = history[i].get("content", "")
            assistant_msg = ""
            if i + 1 < len(history) and history[i + 1].get("role") == "assistant":
                assistant_msg = history[i + 1].get("content", "")
                i += 1
            history_tuples.append((user_msg, assistant_msg))
        i += 1

    # Check for inference commands
    is_inference_command = any(kw in text.lower() for kw in ["run inference", "run prediction", "run forecast", "run surya", "make prediction", "generate forecast"])

    # Check for date-based pattern
    date_match = re.search(
        r'(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?)',
        text,
    )

    if is_inference_command or (date_match and any(kw in text.lower() for kw in ["run", "predict", "inference", "forecast"])):
        # If no date provided, ask for one
        if not date_match:
            yield history + [
                {"role": "user", "content": text},
                {"role": "assistant", "content": "📅 Please provide a date for the prediction.\n\n**Format:** `YYYY-MM-DD` or `YYYY-MM-DD HH:MM`\n\n**Example:** `run prediction for 2014-10-23 12:00`\n\nSDO data is available from 2010 to present."},
            ]
            return

        start_dt = date_match.group(1)
        if " " not in start_dt:
            start_dt += " 10:00"

        rollout_match = re.search(r'(\d+)\s*(?:steps?|hours?|rollout)', text.lower())
        rollout = int(rollout_match.group(1)) if rollout_match else 1

        yield history + [
            {"role": "user", "content": text},
            {"role": "assistant", "content": f"🚀 Running Surya prediction for **{start_dt}** with {rollout} rollout steps..."},
        ]

        status, generated_images = surya_runner.run_inference(start_dt, rollout)

        response = f"✅ {status}"
        if generated_images:
            response += f"\n\n📊 Generated **{len(generated_images)}** images. Check the Gallery tab!"

        yield history + [
            {"role": "user", "content": text},
            {"role": "assistant", "content": response},
        ]
        return

    # Check for show images command
    if any(kw in text.lower() for kw in ["show image", "latest image", "prediction image", "gallery"]):
        images = surya_runner.get_latest_images()
        if images:
            yield history + [
                {"role": "user", "content": text},
                {"role": "assistant", "content": f"📸 Found **{len(images)}** images in the latest output. Check the **Gallery** tab to view them!"},
            ]
        else:
            yield history + [
                {"role": "user", "content": text},
                {"role": "assistant", "content": "No prediction images found. Run an inference first!"},
            ]
        return

    # Regular chat with Groq
    if not groq_client.available:
        yield history + [
            {"role": "user", "content": text},
            {"role": "assistant", "content": f"⚠️ Groq API unavailable: {groq_client.error_message}"},
        ]
        return

    yield history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": ""},
    ]

    for partial_response in groq_client.chat(text, history_tuples):
        yield history + [
            {"role": "user", "content": text},
            {"role": "assistant", "content": partial_response},
        ]


def show_images(channel: str) -> list[str]:
    """Show images from the latest inference."""
    images = surya_runner.get_latest_images(channel if channel != "all" else None)
    return images


def refresh_status() -> str:
    """Refresh and return connection status HTML."""
    return get_status_html()


def run_inference_ui(start_dt: str, steps: int, dry: bool) -> tuple[str, list]:
    """Run inference from UI controls."""
    status, images = surya_runner.run_inference(start_dt, int(steps), dry)
    return status, images


# Custom CSS
CUSTOM_CSS = """
/* Root variables */
:root {
    --primary-color: #F59E0B;
    --primary-dark: #D97706;
    --bg-primary: #0F172A;
    --bg-secondary: #1E293B;
    --bg-tertiary: #334155;
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --border-color: #475569;
    --success-color: #10B981;
    --error-color: #EF4444;
}

/* Dark theme override */
.dark {
    --bg-primary: #0F172A;
    --bg-secondary: #1E293B;
}

/* Main container */
.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
    background: var(--bg-primary) !important;
}

/* Header */
.header-container {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    border: 1px solid var(--border-color);
    text-align: center;
}

.header-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #F59E0B 0%, #FBBF24 50%, #FCD34D 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

.header-subtitle {
    color: var(--text-secondary);
    font-size: 1.1rem;
}

/* Status container */
.status-container {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 9999px;
    font-size: 0.875rem;
    font-weight: 500;
}

.status-online {
    background: rgba(16, 185, 129, 0.15);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.status-offline {
    background: rgba(239, 68, 68, 0.15);
    color: #EF4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.status-icon {
    font-size: 0.75rem;
}

/* Chatbot styling */
.chatbot-container {
    border-radius: 12px !important;
    border: 1px solid var(--border-color) !important;
}

/* Channel badges */
.channel-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 0.5rem;
    padding: 1rem;
}

.channel-badge {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 0.75rem;
    background: var(--bg-tertiary);
    border-radius: 8px;
    border: 1px solid var(--border-color);
    transition: all 0.2s;
}

.channel-badge:hover {
    border-color: var(--primary-color);
    transform: translateY(-2px);
}

.channel-name {
    font-weight: 600;
    font-size: 0.875rem;
    color: var(--text-primary);
}

.channel-desc {
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-align: center;
}

/* Sidebar panel */
.sidebar-panel {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 1.25rem;
    border: 1px solid var(--border-color);
}

.sidebar-title {
    font-weight: 600;
    font-size: 1rem;
    color: var(--text-primary);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Quick action buttons */
.quick-action {
    width: 100%;
    text-align: left !important;
    justify-content: flex-start !important;
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    padding: 0.75rem 1rem !important;
    border-radius: 8px !important;
    margin-bottom: 0.5rem !important;
    transition: all 0.2s !important;
}

.quick-action:hover {
    border-color: var(--primary-color) !important;
    background: var(--bg-secondary) !important;
}

/* Primary button */
.primary-btn {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}

.primary-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);
}

/* Tabs */
.tabs {
    border-radius: 12px !important;
    overflow: hidden;
}

/* Gallery */
.gallery-container {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 1rem;
}

/* Footer */
.footer {
    text-align: center;
    padding: 1.5rem;
    color: var(--text-secondary);
    font-size: 0.875rem;
    border-top: 1px solid var(--border-color);
    margin-top: 2rem;
}

/* Info cards */
.info-card {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid var(--border-color);
}

.info-card h3 {
    color: var(--primary-color);
    margin-bottom: 1rem;
}

.info-card ul {
    color: var(--text-secondary);
    padding-left: 1.25rem;
}

.info-card li {
    margin-bottom: 0.5rem;
}
"""


def create_channel_badges_html() -> str:
    """Create HTML for channel badges."""
    badges = ""
    for ch, (name, desc, color) in CHANNEL_INFO.items():
        badges += f"""
        <div class="channel-badge" style="border-left: 3px solid {color};">
            <span class="channel-name">{name}</span>
            <span class="channel-desc">{desc[:25]}...</span>
        </div>
        """
    return f'<div class="channel-grid">{badges}</div>'


def create_ui() -> gr.Blocks:
    """Create the Gradio UI."""

    with gr.Blocks(title="Surya - Solar Forecasting") as demo:
        # Header
        gr.HTML("""
            <div class="header-container">
                <h1 class="header-title">☀️ Surya</h1>
                <p class="header-subtitle">AI-Powered Solar Forecasting • 366M Parameter Foundation Model</p>
            </div>
        """)

        # Status bar
        with gr.Row():
            status_html = gr.HTML(value=get_status_html())
            refresh_btn = gr.Button("🔄 Refresh Status", size="sm", scale=0)

        refresh_btn.click(fn=refresh_status, outputs=[status_html])

        with gr.Tabs() as tabs:
            # === CHAT TAB ===
            with gr.Tab("💬 Chat", id="chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="",
                            height=500,
                            placeholder="Ask me about solar physics, run predictions, or analyze results...",
                        )

                        chat_input = gr.MultimodalTextbox(
                            placeholder="Type a message... (e.g., 'What channel shows solar flares?' or 'Run prediction for 2014-10-23')",
                            show_label=False,
                            file_types=["image"],
                            file_count="multiple",
                            submit_btn=True,
                        )

                    with gr.Column(scale=1, min_width=280):
                        gr.HTML('<div class="sidebar-panel"><div class="sidebar-title">⚡ Quick Actions</div></div>')

                        examples = [
                            ("🌟", "What channel shows solar flares?"),
                            ("🔬", "Explain the AIA 94Å wavelength"),
                            ("🧲", "How do I analyze magnetic fields?"),
                            ("📊", "Show latest prediction images"),
                        ]

                        for icon, text in examples:
                            btn = gr.Button(f"{icon} {text}", elem_classes=["quick-action"])
                            btn.click(
                                fn=lambda t=text: {"text": t, "files": []},
                                outputs=[chat_input],
                            )

                        gr.HTML('<div class="sidebar-panel" style="margin-top: 1rem;"><div class="sidebar-title">📡 Channels</div></div>')
                        gr.HTML("""
                            <div style="display: flex; flex-wrap: wrap; gap: 0.375rem; padding: 0.5rem;">
                                <span style="background: #FF6B6B22; color: #FF6B6B; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem;">aia94</span>
                                <span style="background: #4ECDC422; color: #4ECDC4; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem;">aia171</span>
                                <span style="background: #45B7D122; color: #45B7D1; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem;">aia193</span>
                                <span style="background: #FFEAA722; color: #FFEAA7; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem;">aia304</span>
                                <span style="background: #FDCB6E22; color: #FDCB6E; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem;">hmi_bz</span>
                            </div>
                        """)

                chat_input.submit(
                    fn=process_message,
                    inputs=[chat_input, chatbot],
                    outputs=[chatbot],
                )

            # === GALLERY TAB ===
            with gr.Tab("🖼️ Gallery", id="gallery"):
                with gr.Row():
                    channel_dropdown = gr.Dropdown(
                        choices=["all"] + list(CHANNEL_INFO.keys()),
                        value="all",
                        label="Filter by Channel",
                        scale=2,
                    )
                    gallery_refresh = gr.Button("🔄 Load Images", variant="primary", scale=1)

                gallery = gr.Gallery(
                    label="Prediction Images",
                    columns=4,
                    height=500,
                    object_fit="contain",
                )

                gallery_refresh.click(
                    fn=show_images,
                    inputs=[channel_dropdown],
                    outputs=[gallery],
                )

                channel_dropdown.change(
                    fn=show_images,
                    inputs=[channel_dropdown],
                    outputs=[gallery],
                )

            # === INFERENCE TAB ===
            with gr.Tab("🚀 Run Inference", id="inference"):
                with gr.Row():
                    with gr.Column():
                        gr.HTML('<div class="info-card"><h3>🔧 Prediction Settings</h3></div>')

                        start_date = gr.Textbox(
                            label="Start Date/Time",
                            placeholder="YYYY-MM-DD HH:MM",
                            value="2014-10-23 10:00",
                            info="Format: 2014-10-23 10:00",
                        )

                        rollout_steps = gr.Slider(
                            minimum=1,
                            maximum=24,
                            value=1,
                            step=1,
                            label="Rollout Steps",
                            info="Number of hours to predict ahead",
                        )

                        dry_run = gr.Checkbox(
                            label="🔍 Dry Run (preview only)",
                            value=False,
                        )

                        run_btn = gr.Button(
                            "🚀 Run Prediction",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column():
                        inference_output = gr.Textbox(
                            label="Output Log",
                            lines=8,
                            interactive=False,
                        )

                        inference_gallery = gr.Gallery(
                            label="Generated Images",
                            columns=3,
                            height=300,
                        )

                run_btn.click(
                    fn=run_inference_ui,
                    inputs=[start_date, rollout_steps, dry_run],
                    outputs=[inference_output, inference_gallery],
                )

            # === ABOUT TAB ===
            with gr.Tab("ℹ️ About", id="about"):
                with gr.Row():
                    with gr.Column():
                        gr.HTML("""
                            <div class="info-card">
                                <h3>🌞 About Surya</h3>
                                <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                                    Surya is a <strong>366M parameter foundation model</strong> for solar forecasting,
                                    trained on data from NASA's Solar Dynamics Observatory (SDO).
                                </p>
                                <ul>
                                    <li>Predicts solar observations across <strong>13 channels</strong></li>
                                    <li>Forecasts magnetic field evolution</li>
                                    <li>Multi-step rollout predictions (up to 24+ hours)</li>
                                    <li>Powered by Groq LLM for natural language interaction</li>
                                </ul>
                            </div>
                        """)

                    with gr.Column():
                        gr.HTML("""
                            <div class="info-card">
                                <h3>📡 Data Sources</h3>
                                <ul>
                                    <li><strong>AIA</strong> - Atmospheric Imaging Assembly: EUV wavelengths (94Å - 1600Å)</li>
                                    <li><strong>HMI</strong> - Helioseismic and Magnetic Imager: Magnetic field components</li>
                                </ul>
                                <h3 style="margin-top: 1.5rem;">🔗 Commands</h3>
                                <ul>
                                    <li><code>run prediction for 2014-10-23</code> - Run inference</li>
                                    <li><code>show images</code> - View latest predictions</li>
                                    <li><code>what channel shows flares?</code> - Ask questions</li>
                                </ul>
                            </div>
                        """)

                gr.HTML(create_channel_badges_html())

        # Footer
        gr.HTML("""
            <div class="footer">
                <p>☀️ <strong>Surya Solar Forecasting</strong> • Powered by Groq LLM & Gradio</p>
                <p style="font-size: 0.75rem; margin-top: 0.5rem;">NASA SDO Data • 366M Parameter Model</p>
            </div>
        """)

    return demo


def main():
    """Launch the UI."""
    print("\n" + "=" * 60)
    print("  ☀️  SURYA CHAT UI")
    print("=" * 60)

    connected, status = groq_client.check_connection()
    if connected:
        print(f"  ✅ Groq API: {status}")
    else:
        print(f"  ❌ Groq API: {status}")

    print("\n  Starting server...")
    print("=" * 60 + "\n")

    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
