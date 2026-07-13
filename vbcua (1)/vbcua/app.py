"""
VBCUA - Voice-Based Concept Understanding Analyser
"""

import os
import io
import re
import json
import base64
import uuid
import datetime

import numpy as np
import requests
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load variables from a .env file (if present) into the environment,
# so GEMINI_API_KEY doesn't have to be set manually every session.
load_dotenv()

from flask import Flask, request, jsonify, send_from_directory, render_template
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Configuration

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"          # multimodal (audio-in, text-out)
GEMINI_EMBED_MODEL = "text-embedding-004"  # embeddings for semantic similarity

GEMINI_GENERATE_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)
GEMINI_EMBED_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_EMBED_MODEL}:embedContent?key={GEMINI_API_KEY}"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

FILLER_WORDS = [
    "um", "uh", "umm", "uhh", "like", "you know", "sort of", "kind of",
    "basically", "actually", "literally", "i mean", "so yeah", "right",
    "okay so", "well", "hmm"
]

# A small built-in reference-concept repository (Supporting Data & AI Services Layer)
REFERENCE_CONCEPTS = {
    "Machine Learning": (
        "Machine Learning is a subfield of artificial intelligence where "
        "systems learn patterns from data instead of being explicitly "
        "programmed. It involves training models on datasets to make "
        "predictions or decisions, using techniques such as supervised, "
        "unsupervised, and reinforcement learning. Core ideas include "
        "features, labels, model training, loss functions, overfitting, "
        "and generalization to unseen data."
    ),
    "Cloud Computing": (
        "Cloud Computing is the on-demand delivery of computing resources "
        "such as servers, storage, databases, and networking over the "
        "internet, typically on a pay-as-you-go basis. It includes service "
        "models like IaaS, PaaS, and SaaS, deployment models such as "
        "public, private, and hybrid cloud, and key benefits like "
        "scalability, elasticity, and reduced infrastructure management."
    ),
    "Blockchain": (
        "Blockchain is a distributed, immutable ledger technology where "
        "records (blocks) are cryptographically linked in a chain and "
        "replicated across a decentralized network of nodes. It relies on "
        "consensus mechanisms to validate transactions without a central "
        "authority, ensuring transparency, security, and tamper resistance."
    ),
    "Computer Networks": (
        "A computer network connects multiple devices so they can "
        "communicate and share resources. Core concepts include the "
        "OSI/TCP-IP layered models, IP addressing, routing and switching, "
        "protocols such as TCP, UDP, HTTP, and DNS, and concerns like "
        "bandwidth, latency, and network security."
    ),
}

app = Flask(__name__)

# Gemini helpers (Supporting Data & AI Services Layer)

def gemini_transcribe_and_summarize(audio_path: str) -> dict:
    """Send audio to Gemini for transcription (Speech-to-Text Module)."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Export it as an environment variable "
            "before starting the server."
        )

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    mime_type = "audio/wav"
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in (".mp3",):
        mime_type = "audio/mp3"
    elif ext in (".m4a",):
        mime_type = "audio/mp4"
    elif ext in (".ogg",):
        mime_type = "audio/ogg"

    prompt = (
        "You are the Speech-to-Text Module of an educational analyser. "
        "Transcribe the spoken audio exactly as spoken (verbatim, including "
        "filler words like um/uh/like). Respond ONLY with strict JSON, no "
        "markdown, no backticks, in this exact schema:\n"
        '{"transcript": "<verbatim transcript>"}'
    )

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": audio_b64}},
            ]
        }],
        "generationConfig": {"temperature": 0.2},
    }

    resp = requests.post(GEMINI_GENERATE_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    text = _strip_json_fences(text)
    parsed = json.loads(text)
    return parsed


def gemini_embed(text: str) -> np.ndarray:
    """Get an embedding vector for a piece of text (Semantic Understanding Module)."""
    payload = {"content": {"parts": [{"text": text}]}}
    resp = requests.post(GEMINI_EMBED_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    values = data["embedding"]["values"]
    return np.array(values, dtype=np.float32)


def gemini_ai_summary(transcript: str, reference: str, semantic_score: float,
                       filler_ratio: float, pause_ratio: float) -> str:
    """Evaluation Scoring / Report Generation Module: qualitative AI feedback."""
    prompt = f"""You are an educational speech evaluator.

Reference concept explanation:
\"\"\"{reference}\"\"\"

Student's spoken explanation (transcribed):
\"\"\"{transcript}\"\"\"

Computed metrics:
- Semantic similarity to reference: {semantic_score:.1f}/100
- Filler word ratio: {filler_ratio:.1f}%
- Pause ratio: {pause_ratio:.1f}%

Write a concise (4-6 sentence) qualitative feedback summary covering:
1) which core ideas were correctly explained,
2) any important points that were missed or incorrect,
3) fluency/delivery observations,
4) one concrete suggestion for improvement.
Respond with plain text only, no markdown headers."""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4},
    }
    resp = requests.post(GEMINI_GENERATE_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()

# Filler Word Analysis Module

def analyze_filler_words(transcript: str) -> dict:
    words = re.findall(r"[a-zA-Z']+", transcript.lower())
    total_words = max(len(words), 1)

    text_lower = " " + transcript.lower() + " "
    counts = {}
    total_fillers = 0
    for fw in FILLER_WORDS:
        pattern = r"\b" + re.escape(fw) + r"\b"
        n = len(re.findall(pattern, text_lower))
        if n > 0:
            counts[fw] = n
            total_fillers += n

    ratio = (total_fillers / total_words) * 100
    return {
        "total_words": total_words,
        "total_fillers": total_fillers,
        "filler_ratio": round(ratio, 2),
        "breakdown": counts,
    }


# Audio Feature Extraction Module (Librosa)

def extract_audio_features(audio_path: str) -> dict:
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    mean_rms = float(np.mean(rms))

    # Pause ratio via silence intervals (non-silent intervals detection)
    intervals = librosa.effects.split(y, top_db=30)
    voiced_samples = sum((end - start) for start, end in intervals)
    total_samples = len(y)
    silent_samples = max(total_samples - voiced_samples, 0)
    pause_ratio = (silent_samples / total_samples) * 100 if total_samples else 0

    return {
        "duration_sec": round(float(duration), 2),
        "sample_rate": int(sr),
        "mean_rms_energy": round(mean_rms, 5),
        "pause_ratio": round(float(pause_ratio), 2),
        "waveform": y,
        "sr": sr,
    }


def generate_waveform_image(y: np.ndarray, sr: int, out_path: str):
    plt.figure(figsize=(8, 2.2))
    times = np.linspace(0, len(y) / sr, num=len(y))
    plt.plot(times, y, linewidth=0.6, color="#5b3fd6")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=150, transparent=True)
    plt.close()


# Evaluation Scoring Module

def compute_comprehension_score(semantic_score: float, filler_ratio: float,
                                 pause_ratio: float) -> dict:
    """
    Weighted comprehension score:
      70% semantic similarity (understanding of concept)
      15% fluency (inverse of filler ratio)
      15% pacing (inverse of excessive pause ratio)
    """
    fluency_score = max(0.0, 100 - filler_ratio * 4)   # penalize fillers
    pacing_score = max(0.0, 100 - max(0, pause_ratio - 20) * 2)  # tolerate up to 20% pause

    final_score = (
        0.70 * semantic_score +
        0.15 * fluency_score +
        0.15 * pacing_score
    )
    final_score = round(min(max(final_score, 0), 100), 2)

    if final_score >= 75:
        classification = "Strong Understanding"
    elif final_score >= 50:
        classification = "Moderate Understanding"
    else:
        classification = "Poor Understanding"

    return {
        "comprehension_score": final_score,
        "fluency_score": round(fluency_score, 2),
        "pacing_score": round(pacing_score, 2),
        "classification": classification,
    }

# Report Generation Module (ReportLab)

def generate_pdf_report(result: dict, waveform_img_path: str) -> str:
    filename = f"VBCUA_Report_{uuid.uuid4().hex[:8]}.pdf"
    filepath = os.path.join(REPORT_DIR, filename)

    doc = SimpleDocTemplate(filepath, pagesize=A4,
                             topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"],
                                  textColor=colors.HexColor("#3f2b96"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                         textColor=colors.HexColor("#5b3fd6"))
    body = styles["BodyText"]

    story = []
    story.append(Paragraph("Voice-Based Concept Understanding Analyser", title_style))
    story.append(Paragraph("Evaluation Report", styles["Heading3"]))
    story.append(Paragraph(datetime.datetime.now().strftime("Generated on %d %b %Y, %H:%M"), body))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("Concept", h2))
    story.append(Paragraph(result["concept"], body))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Transcript", h2))
    story.append(Paragraph(result["transcript"], body))
    story.append(Spacer(1, 0.3 * cm))

    if os.path.exists(waveform_img_path):
        story.append(Paragraph("Waveform Visualization", h2))
        story.append(Image(waveform_img_path, width=16 * cm, height=4 * cm))
        story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Evaluation Metrics", h2))
    table_data = [
        ["Metric", "Value"],
        ["Comprehension Score", f'{result["scoring"]["comprehension_score"]} / 100'],
        ["Classification", result["scoring"]["classification"]],
        ["Semantic Similarity", f'{result["semantic_score"]} / 100'],
        ["Filler Word Ratio", f'{result["filler"]["filler_ratio"]}%'],
        ["Total Fillers Detected", str(result["filler"]["total_fillers"])],
        ["Pause Ratio", f'{result["audio"]["pause_ratio"]}%'],
        ["Mean RMS Energy", str(result["audio"]["mean_rms_energy"])],
        ["Duration", f'{result["audio"]["duration_sec"]} sec'],
    ]
    tbl = Table(table_data, colWidths=[7 * cm, 8 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#5b3fd6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("AI-Generated Summary & Feedback", h2))
    story.append(Paragraph(result["ai_summary"], body))

    doc.build(story)
    return filepath

# Routes

@app.route("/")
def index():
    return render_template("index.html", concepts=list(REFERENCE_CONCEPTS.keys()))


@app.route("/api/concepts")
def api_concepts():
    return jsonify(REFERENCE_CONCEPTS)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file uploaded."}), 400

        audio_file = request.files["audio"]
        concept = request.form.get("concept", "").strip()
        custom_reference = request.form.get("custom_reference", "").strip()

        reference_text = custom_reference or REFERENCE_CONCEPTS.get(concept)
        if not reference_text:
            return jsonify({"error": "No reference concept provided."}), 400
        if not concept:
            concept = "Custom Concept"

        ext = os.path.splitext(audio_file.filename)[1] or ".wav"
        uid = uuid.uuid4().hex[:10]
        raw_path = os.path.join(UPLOAD_DIR, f"{uid}{ext}")
        audio_file.save(raw_path)

        # Ensure it's readable by librosa/soundfile; re-encode to wav if needed
        wav_path = os.path.join(UPLOAD_DIR, f"{uid}.wav")
        try:
            y_tmp, sr_tmp = librosa.load(raw_path, sr=None, mono=True)
            sf.write(wav_path, y_tmp, sr_tmp)
        except Exception:
            wav_path = raw_path  # fall back to original

        # 1. Speech-to-Text (Gemini)
        stt = gemini_transcribe_and_summarize(raw_path)
        transcript = stt.get("transcript", "").strip()
        if not transcript:
            return jsonify({"error": "Transcription failed or empty."}), 500

        # 2. Semantic Understanding (Gemini embeddings + cosine similarity)
        emb_transcript = gemini_embed(transcript)
        emb_reference = gemini_embed(reference_text)
        cos_sim = float(
            np.dot(emb_transcript, emb_reference) /
            (np.linalg.norm(emb_transcript) * np.linalg.norm(emb_reference) + 1e-9)
        )
        semantic_score = round(max(0.0, min(1.0, cos_sim)) * 100, 2)

        # 3. Filler Word Analysis
        filler_result = analyze_filler_words(transcript)

        # 4. Audio Feature Extraction (Librosa)
        audio_feats = extract_audio_features(wav_path)
        waveform_img_path = os.path.join(REPORT_DIR, f"waveform_{uid}.png")
        generate_waveform_image(audio_feats["waveform"], audio_feats["sr"], waveform_img_path)

        # 5. Evaluation Scoring
        scoring = compute_comprehension_score(
            semantic_score, filler_result["filler_ratio"], audio_feats["pause_ratio"]
        )

        # 6. AI Summary (Gemini)
        ai_summary = gemini_ai_summary(
            transcript, reference_text, semantic_score,
            filler_result["filler_ratio"], audio_feats["pause_ratio"]
        )

        result = {
            "concept": concept,
            "reference": reference_text,
            "transcript": transcript,
            "semantic_score": semantic_score,
            "filler": filler_result,
            "audio": {
                "duration_sec": audio_feats["duration_sec"],
                "sample_rate": audio_feats["sample_rate"],
                "mean_rms_energy": audio_feats["mean_rms_energy"],
                "pause_ratio": audio_feats["pause_ratio"],
            },
            "scoring": scoring,
            "ai_summary": ai_summary,
        }

        # 7. Report Generation (PDF)
        pdf_path = generate_pdf_report(result, waveform_img_path)
        result["pdf_filename"] = os.path.basename(pdf_path)
        result["waveform_image"] = os.path.basename(waveform_img_path)

        return jsonify(result)

    except requests.HTTPError as e:
        return jsonify({"error": f"Gemini API error: {e.response.text if e.response else str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/report/<filename>")
def download_report(filename):
    return send_from_directory(REPORT_DIR, filename, as_attachment=True)


@app.route("/api/waveform/<filename>")
def get_waveform_image(filename):
    return send_from_directory(REPORT_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
