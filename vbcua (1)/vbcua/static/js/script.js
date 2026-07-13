const conceptSelect = document.getElementById("conceptSelect");
const customWrap = document.getElementById("customConceptWrap");
const customReference = document.getElementById("customReference");
const audioFileInput = document.getElementById("audioFile");
const playerWrap = document.getElementById("playerWrap");
const audioPlayer = document.getElementById("audioPlayer");
const waveformCanvas = document.getElementById("waveformCanvas");
const evaluateBtn = document.getElementById("evaluateBtn");
const statusMsg = document.getElementById("statusMsg");
const resultsPanel = document.getElementById("resultsPanel");

let selectedFile = null;

conceptSelect.addEventListener("change", () => {
  customWrap.classList.toggle("hidden", conceptSelect.value !== "__custom__");
});

audioFileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  selectedFile = file;

  const url = URL.createObjectURL(file);
  audioPlayer.src = url;
  playerWrap.classList.remove("hidden");
  evaluateBtn.disabled = false;
  statusMsg.textContent = "";
  statusMsg.classList.remove("error");

  drawWaveform(file);
});

async function drawWaveform(file) {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
    const rawData = audioBuffer.getChannelData(0);

    const canvas = waveformCanvas;
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 600;
    const height = canvas.clientHeight || 86;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const samples = 400;
    const blockSize = Math.floor(rawData.length / samples);
    const filtered = [];
    for (let i = 0; i < samples; i++) {
      const start = blockSize * i;
      let sum = 0;
      for (let j = 0; j < blockSize; j++) {
        sum += Math.abs(rawData[start + j] || 0);
      }
      filtered.push(sum / blockSize);
    }
    const max = Math.max(...filtered, 0.0001);

    ctx.fillStyle = "#e8a33d";
    const barWidth = width / samples;
    filtered.forEach((v, i) => {
      const barHeight = Math.max((v / max) * (height * 0.8), 1);
      const x = i * barWidth;
      const y = (height - barHeight) / 2;
      ctx.fillRect(x, y, Math.max(barWidth - 1, 1), barHeight);
    });
  } catch (err) {
    console.warn("Waveform preview unavailable:", err);
  }
}

evaluateBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  let concept = conceptSelect.value;
  let custom = "";
  if (concept === "__custom__") {
    custom = customReference.value.trim();
    if (!custom) {
      showStatus("Please paste a custom reference explanation.", true);
      return;
    }
  }

  const formData = new FormData();
  formData.append("audio", selectedFile);
  formData.append("concept", concept === "__custom__" ? "" : concept);
  formData.append("custom_reference", custom);

  evaluateBtn.disabled = true;
  showStatus("RUNNING TAPE… transcribing, scoring, and printing report.");

  try {
    const resp = await fetch("/api/analyze", { method: "POST", body: formData });
    const data = await resp.json();

    if (!resp.ok) {
      showStatus(data.error || "Something went wrong during analysis.", true);
      evaluateBtn.disabled = false;
      return;
    }

    renderResults(data);
    showStatus("EVALUATION COMPLETE.");
  } catch (err) {
    showStatus("Network error: " + err.message, true);
  } finally {
    evaluateBtn.disabled = false;
  }
});

function showStatus(msg, isError = false) {
  statusMsg.textContent = msg;
  statusMsg.classList.toggle("error", isError);
}

function setFader(fillId, capId, percent) {
  const clamped = Math.max(0, Math.min(100, percent));
  const fill = document.getElementById(fillId);
  const cap = document.getElementById(capId);
  if (fill) fill.style.height = clamped + "%";
  if (cap) cap.style.bottom = clamped + "%";
}

function setNeedle(score) {
  const clamped = Math.max(0, Math.min(100, score));
  const angle = -90 + (clamped / 100) * 180; // -90deg (0) to +90deg (100)
  const needleGroup = document.getElementById("vuNeedleGroup");
  if (needleGroup) needleGroup.style.transform = `rotate(${angle}deg)`;
}

function renderResults(data) {
  resultsPanel.classList.remove("hidden");

  document.getElementById("scoreValue").textContent = data.scoring.comprehension_score;
  document.getElementById("classificationLabel").textContent =
    data.scoring.classification.toUpperCase();
  document.getElementById("conceptLabel").textContent = data.concept;

  setNeedle(data.scoring.comprehension_score);

  document.getElementById("semanticScore").textContent = data.semantic_score + "%";
  document.getElementById("fillerRatio").textContent = data.filler.filler_ratio + "%";
  document.getElementById("pauseRatio").textContent = data.audio.pause_ratio + "%";
  document.getElementById("rmsEnergy").textContent =
    Number(data.audio.mean_rms_energy).toFixed(4);

  setFader("semanticFill", "semanticCap", data.semantic_score);
  setFader("fillerFill", "fillerCap", data.filler.filler_ratio);
  setFader("pauseFill", "pauseCap", data.audio.pause_ratio);

  document.getElementById("transcriptText").textContent = data.transcript;
  document.getElementById("aiSummary").textContent = data.ai_summary;

  const breakdownEl = document.getElementById("fillerBreakdown");
  breakdownEl.innerHTML = "";
  const entries = Object.entries(data.filler.breakdown || {});
  if (entries.length === 0) {
    breakdownEl.innerHTML = '<span class="chip">No filler words detected</span>';
  } else {
    entries.forEach(([word, count]) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = `${word} × ${count}`;
      breakdownEl.appendChild(chip);
    });
  }

  const downloadBtn = document.getElementById("downloadPdfBtn");
  downloadBtn.href = `/api/report/${data.pdf_filename}`;

  resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}