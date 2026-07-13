# VBCUA — Voice-Based Concept Understanding Analyser

An AI-powered web app that evaluates how well a person explains a concept out loud.
It transcribes speech, compares it semantically against a reference explanation,
analyses filler words and pacing, scores overall comprehension, and generates a
downloadable PDF report — all powered by the **Google Gemini API**.

## Architecture

```
Presentation Layer (HTML/CSS/JS)
        │
        ▼
Flask API (app.py) — Core Intelligence Layer
   ├─ Speech-to-Text          → Gemini multimodal (audio in / JSON out)
   ├─ Semantic Understanding  → Gemini text-embedding-004 + cosine similarity
   ├─ Filler Word Analysis    → regex-based filler detection
   ├─ Audio Feature Extraction→ Librosa (pause ratio, RMS energy)
   ├─ Evaluation Scoring      → weighted comprehension score + classification
   └─ Report Generation       → ReportLab PDF (waveform + metrics + AI summary)
        │
        ▼
Supporting Data & AI Services
   ├─ Reference concept repository (built-in + custom text)
   ├─ Gemini embeddings
   ├─ Librosa/SoundFile
   └─ ReportLab
```

## 1. Setup

```bash
cd vbcua
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get a Gemini API key

Create a key at https://aistudio.google.com/app/apikey.

Then set it up using **one** of these two methods:

### Method A — `.env` file (recommended, no need to re-set it every session)

1. In the `vbcua` folder, copy `.env.example` to a new file named `.env`.
2. Open `.env` and paste your key:
   ```
   GEMINI_API_KEY=your_actual_key_here
   ```
3. Save the file. `app.py` automatically loads it on startup (via `python-dotenv`) —
   nothing else to do.

### Method B — environment variable (manual, per terminal session)

```bash
export GEMINI_API_KEY="your_api_key_here"     # Windows PowerShell: $env:GEMINI_API_KEY="your_api_key_here"
```

## 3. Run the app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## 4. Using it

1. Pick a reference concept (Machine Learning, Cloud Computing, Blockchain,
   Computer Networks) or choose "Custom concept…" and paste your own reference
   explanation.
2. Upload a `.wav`, `.mp3`, or `.m4a` recording of yourself explaining the topic.
3. Preview the waveform and playback, then click **Evaluate My Explanation**.
4. Review the comprehension score, semantic similarity, filler word ratio,
   pause ratio, RMS energy, transcript, and AI-generated feedback.
5. Download the PDF report for record-keeping or review.

## Notes

- The comprehension score is weighted: 70% semantic similarity, 15% fluency
  (inverse of filler ratio), 15% pacing (inverse of excess pause ratio). You
  can tune these weights in `compute_comprehension_score()` in `app.py`.
- The built-in reference concepts live in `REFERENCE_CONCEPTS` in `app.py` —
  add more entries there (and they'll automatically appear in the dropdown).
- Uploaded audio and generated reports are stored in `uploads/` and `reports/`
  respectively; both are safe to clear periodically.
- Because this app calls the Gemini API, it needs a live internet connection
  and a valid `GEMINI_API_KEY` at runtime.
