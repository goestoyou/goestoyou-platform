import os, re, json, asyncio, tempfile, subprocess
from pathlib import Path
from typing import Optional, AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GROQ_KEY      = os.environ.get("GROQ_API_KEY", "")
CLAUDE_MODEL  = os.environ.get("CLAUDE_MODEL", "claude-opus-4-6")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Utils ────────────────────────────────────────────────────────────────────

def get_drive_id(url: str) -> Optional[str]:
    m = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url or "")
    return m.group(1) if m else None

def _download_video(file_id: str, dest: Path) -> bool:
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}&confirm=t"
    try:
        gdown.download(url, str(dest), quiet=True)
        return dest.exists() and dest.stat().st_size > 10_000
    except Exception:
        return False

def _extract_audio(video: Path, audio: Path) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-i", str(video), "-ar", "16000", "-ac", "1",
         "-c:a", "pcm_s16le", str(audio), "-y", "-loglevel", "error"],
        capture_output=True
    )
    return r.returncode == 0

def _transcribe(audio: Path) -> str:
    from groq import Groq
    client = Groq(api_key=GROQ_KEY)
    with open(audio, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=("audio.wav", f, "audio/wav"),
            language="it",
            response_format="text",
        )
    return str(result).strip()

def _analyze(transcript: str) -> dict:
    ai = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    resp = ai.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system="Sei un esperto coach di video marketing e contenuti digitali. Analizza trascrizioni e rispondi SOLO in JSON valido, senza testo aggiuntivo.",
        messages=[{"role": "user", "content": f"""Analizza questa trascrizione e rispondi ESCLUSIVAMENTE con questo JSON:
{{
  "score": <intero 1-10>,
  "commento": "<analisi 3-5 frasi: punti di forza, debolezze, chiarezza del messaggio, coinvolgimento>",
  "prompt": "<prompt 4-6 frasi per migliorare il prossimo video sullo stesso argomento>"
}}

Criteri: 9-10=eccellente, 7-8=buono, 5-6=sufficiente, 3-4=scarso, 1-2=molto scarso.

TRASCRIZIONE:
{transcript[:4000]}{"..." if len(transcript) > 4000 else ""}"""}]
    )
    raw = resp.content[0].text.strip()
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        d = json.loads(m.group())
        return {
            "score":    max(1, min(10, int(d.get("score", 5)))),
            "commento": str(d.get("commento", "")).strip(),
            "prompt":   str(d.get("prompt", "")).strip(),
        }
    return {"score": 0, "commento": "Analisi non disponibile.", "prompt": ""}


# ── SSE Stream ───────────────────────────────────────────────────────────────

def evt(step: str, message: str, data: dict = None) -> str:
    p = {"step": step, "message": message}
    if data:
        p["data"] = data
    return f"data: {json.dumps(p, ensure_ascii=False)}\n\n"

async def run_sync(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)

async def analyze_stream(drive_url: str) -> AsyncGenerator[str, None]:
    file_id = get_drive_id(drive_url)
    if not file_id:
        yield evt("error", "URL Google Drive non valido. Deve contenere /file/d/...")
        return

    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "video.mov"
        audio = Path(tmp) / "audio.wav"

        yield evt("download", "Scaricando il video da Google Drive...")
        ok = await run_sync(_download_video, file_id, video)
        if not ok:
            yield evt("error", "Download fallito. Verifica che il link Drive sia pubblico ('Chiunque con il link').")
            return

        size_mb = video.stat().st_size / 1_048_576
        yield evt("audio", f"Estraendo la traccia audio ({size_mb:.1f} MB)...")
        ok = await run_sync(_extract_audio, video, audio)
        if not ok:
            yield evt("error", "Errore estrazione audio. ffmpeg non trovato o formato video non supportato.")
            return

        yield evt("transcribe", "Trascrivendo con Whisper AI...")
        try:
            transcript = await run_sync(_transcribe, audio)
        except Exception as e:
            yield evt("error", f"Errore trascrizione: {e}")
            return

        word_count = len(transcript.split())
        yield evt("analyze", f"Analizzando con Claude AI ({word_count} parole trascritte)...")
        try:
            analysis = await run_sync(_analyze, transcript)
        except Exception as e:
            yield evt("error", f"Errore analisi AI: {e}")
            return

        yield evt("done", "Analisi completata!", {
            "transcript": transcript,
            "word_count":  word_count,
            **analysis,
        })


# ── Routes ───────────────────────────────────────────────────────────────────

class AnalyzeReq(BaseModel):
    drive_url: str

@app.post("/api/analyze")
async def analyze(req: AnalyzeReq):
    return StreamingResponse(
        analyze_stream(req.drive_url),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/api/health")
def health():
    return {
        "ok":        True,
        "anthropic": bool(ANTHROPIC_KEY),
        "groq":      bool(GROQ_KEY),
        "model":     CLAUDE_MODEL,
    }

@app.get("/")
def index():
    return HTMLResponse(HTML)


# ── HTML ─────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GoesToYou.video — Analisi AI</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #09090d;
  --surface: #13131a;
  --surface2: #1c1c26;
  --border: rgba(255,255,255,0.07);
  --primary: #7f5af0;
  --primary-light: #a78bfa;
  --text: #e2e8f0;
  --muted: #64748b;
  --green: #22d3a5;
  --amber: #f59e0b;
  --red: #f87171;
}

body {
  font-family: 'Inter', sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* ── Header ── */
header {
  padding: 1.1rem 2rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 0.75rem;
  position: sticky;
  top: 0;
  background: rgba(9,9,13,0.85);
  backdrop-filter: blur(12px);
  z-index: 100;
}

.logo {
  font-size: 1rem;
  font-weight: 700;
  background: linear-gradient(135deg, #a78bfa, #38bdf8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: -0.02em;
}

.badge {
  font-size: 0.6rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--primary);
  border: 1px solid rgba(127,90,240,0.35);
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  background: rgba(127,90,240,0.08);
}

/* ── Main ── */
main {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 4rem 1.5rem 3rem;
  width: 100%;
  max-width: 860px;
  margin: 0 auto;
}

/* ── Input Phase ── */
#input-section { width: 100%; text-align: center; }

.hero-eyebrow {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--primary-light);
  margin-bottom: 1rem;
}

.hero-title {
  font-size: clamp(2.2rem, 5vw, 3.4rem);
  font-weight: 800;
  line-height: 1.12;
  letter-spacing: -0.03em;
  margin-bottom: 0.85rem;
  background: linear-gradient(160deg, #fff 50%, #a78bfa 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-sub {
  color: var(--muted);
  font-size: 1.05rem;
  line-height: 1.6;
  margin-bottom: 2.75rem;
  max-width: 520px;
  margin-left: auto;
  margin-right: auto;
}

.input-box {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  overflow: hidden;
  display: flex;
  transition: border-color 0.2s, box-shadow 0.2s;
  margin-bottom: 0.85rem;
}

.input-box:focus-within {
  border-color: rgba(127,90,240,0.55);
  box-shadow: 0 0 0 4px rgba(127,90,240,0.1);
}

.input-box input {
  flex: 1;
  background: transparent;
  border: none;
  outline: none;
  padding: 1.1rem 1.4rem;
  font-size: 0.95rem;
  color: var(--text);
  font-family: 'Inter', sans-serif;
}

.input-box input::placeholder { color: var(--muted); }

.btn-analyze {
  background: linear-gradient(135deg, #7f5af0, #6366f1);
  color: white;
  border: none;
  padding: 0 1.8rem;
  font-size: 0.92rem;
  font-weight: 600;
  font-family: 'Inter', sans-serif;
  cursor: pointer;
  white-space: nowrap;
  transition: opacity 0.15s, transform 0.1s;
  letter-spacing: -0.01em;
}

.btn-analyze:hover { opacity: 0.88; }
.btn-analyze:active { transform: scale(0.98); }
.btn-analyze:disabled { opacity: 0.45; cursor: not-allowed; }

.input-hint {
  font-size: 0.78rem;
  color: var(--muted);
  text-align: left;
}

/* ── Progress Phase ── */
#progress-section { width: 100%; }

.progress-header {
  text-align: center;
  margin-bottom: 2rem;
}

.progress-title {
  font-size: 1.3rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin-bottom: 0.4rem;
}

.progress-sub {
  font-size: 0.88rem;
  color: var(--muted);
  min-height: 1.3em;
  transition: opacity 0.3s;
}

.steps-list {
  display: flex;
  flex-direction: column;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  overflow: hidden;
}

.step-row {
  display: flex;
  align-items: center;
  gap: 1.1rem;
  padding: 1.15rem 1.5rem;
  border-bottom: 1px solid var(--border);
  transition: background 0.3s;
}
.step-row:last-child { border-bottom: none; }
.step-row.active { background: rgba(127,90,240,0.06); }
.step-row.done   { background: rgba(34,211,165,0.04); }

.step-dot {
  width: 38px; height: 38px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.05rem;
  flex-shrink: 0;
  border: 1.5px solid var(--border);
  background: var(--surface2);
  transition: all 0.3s;
}
.step-row.active .step-dot {
  border-color: var(--primary);
  background: rgba(127,90,240,0.12);
  animation: pulse-ring 1.8s infinite;
}
.step-row.done .step-dot {
  border-color: var(--green);
  background: rgba(34,211,165,0.1);
}

@keyframes pulse-ring {
  0%, 100% { box-shadow: 0 0 0 0 rgba(127,90,240,0.5); }
  60%       { box-shadow: 0 0 0 7px rgba(127,90,240,0); }
}

.step-info { flex: 1; }
.step-name { font-size: 0.9rem; font-weight: 600; margin-bottom: 0.15rem; }
.step-detail { font-size: 0.76rem; color: var(--muted); }

.step-badge {
  font-size: 0.75rem;
  font-weight: 600;
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
}
.step-row.pending .step-badge { color: var(--muted); }
.step-row.active  .step-badge { color: var(--primary-light); background: rgba(127,90,240,0.12); }
.step-row.done    .step-badge { color: var(--green);          background: rgba(34,211,165,0.1); }

/* ── Results Phase ── */
#results-section { width: 100%; }

.score-card {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 20px;
  padding: 2rem 2.25rem;
  margin-bottom: 1.1rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: relative;
  overflow: hidden;
  background: var(--surface);
}

.score-glow {
  position: absolute;
  inset: 0;
  background: var(--score-color, #7f5af0);
  opacity: 0.07;
  pointer-events: none;
}

.score-left { position: relative; z-index: 1; }

.score-eyebrow {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-bottom: 0.35rem;
}

.score-big {
  font-size: 4.5rem;
  font-weight: 900;
  line-height: 1;
  letter-spacing: -0.04em;
}
.score-denom { font-size: 1.6rem; font-weight: 400; color: var(--muted); }
.score-stars { font-size: 1.15rem; margin-top: 0.4rem; letter-spacing: 0.05em; }

.score-right { position: relative; z-index: 1; text-align: right; }

.score-stat-label { font-size: 0.75rem; color: var(--muted); margin-bottom: 0.25rem; }
.score-stat-value { font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; }

/* Cards */
.two-col {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-bottom: 1rem;
}

@media (max-width: 580px) { .two-col { grid-template-columns: 1fr; } }

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.4rem 1.5rem;
}

.card-eyebrow {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.7rem;
}

.card-body {
  font-size: 0.9rem;
  line-height: 1.75;
  color: #cbd5e1;
}

/* Transcript */
.transcript-wrap {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  overflow: hidden;
  margin-bottom: 1.25rem;
}

.transcript-toggle {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.1rem 1.5rem;
  cursor: pointer;
  background: transparent;
  border: none;
  color: var(--text);
  font-family: 'Inter', sans-serif;
  font-size: 0.88rem;
  font-weight: 600;
  text-align: left;
  transition: background 0.15s;
}

.transcript-toggle:hover { background: var(--surface2); }

.transcript-arrow { transition: transform 0.25s; color: var(--muted); font-size: 0.7rem; }
.transcript-wrap.open .transcript-arrow { transform: rotate(180deg); }

.transcript-body {
  display: none;
  padding: 0 1.5rem 1.4rem;
  font-size: 0.86rem;
  line-height: 1.85;
  color: #94a3b8;
  white-space: pre-wrap;
  border-top: 1px solid var(--border);
}
.transcript-wrap.open .transcript-body { display: block; }

/* Actions */
.actions { display: flex; gap: 0.75rem; flex-wrap: wrap; }

.btn {
  flex: 1;
  min-width: 130px;
  padding: 0.8rem 1.2rem;
  border-radius: 11px;
  font-size: 0.86rem;
  font-weight: 600;
  font-family: 'Inter', sans-serif;
  cursor: pointer;
  text-align: center;
  transition: all 0.15s;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text);
}
.btn:hover { background: var(--surface2); border-color: rgba(255,255,255,0.14); }
.btn.primary {
  background: linear-gradient(135deg, #7f5af0, #6366f1);
  border: none;
  color: white;
}
.btn.primary:hover { opacity: 0.88; }

/* Error */
.error-box {
  background: rgba(248,113,113,0.07);
  border: 1px solid rgba(248,113,113,0.22);
  border-radius: 14px;
  padding: 1.5rem;
  display: flex;
  gap: 1rem;
  margin-bottom: 1.25rem;
}
.error-icon { font-size: 1.3rem; flex-shrink: 0; }
.error-title { font-weight: 700; color: #fca5a5; margin-bottom: 0.3rem; }
.error-msg { font-size: 0.88rem; color: #f87171; line-height: 1.55; }

/* Footer */
footer {
  text-align: center;
  padding: 1.4rem;
  font-size: 0.74rem;
  color: #374151;
  border-top: 1px solid var(--border);
  margin-top: auto;
}

[hidden] { display: none !important; }
</style>
</head>
<body>

<header>
  <span class="logo">GoesToYou.video</span>
  <span class="badge">AI Analysis</span>
</header>

<main>

  <!-- ── Input Phase ── -->
  <section id="input-section">
    <p class="hero-eyebrow">Powered by Whisper + Claude AI</p>
    <h1 class="hero-title">Analizza il tuo video</h1>
    <p class="hero-sub">Incolla il link Google Drive — trascrizione automatica,<br>voto, commento e prompt di miglioramento in pochi minuti.</p>

    <div class="input-box">
      <input type="url" id="drive-url"
             placeholder="https://drive.google.com/file/d/..." />
      <button class="btn-analyze" id="btn-start" onclick="startAnalysis()">Analizza →</button>
    </div>
    <p class="input-hint">💡 Il video deve essere condiviso con "Chiunque con il link può visualizzare"</p>
  </section>

  <!-- ── Progress Phase ── -->
  <section id="progress-section" hidden>
    <div class="progress-header">
      <h2 class="progress-title">Elaborazione in corso…</h2>
      <p class="progress-sub" id="progress-msg">Avvio…</p>
    </div>

    <div class="steps-list">
      <div class="step-row pending" id="sr-download">
        <div class="step-dot">⬇️</div>
        <div class="step-info">
          <div class="step-name">Download video</div>
          <div class="step-detail">Scarica il file da Google Drive</div>
        </div>
        <span class="step-badge">In attesa</span>
      </div>
      <div class="step-row pending" id="sr-audio">
        <div class="step-dot">🎵</div>
        <div class="step-info">
          <div class="step-name">Estrazione audio</div>
          <div class="step-detail">Isola la traccia audio con ffmpeg</div>
        </div>
        <span class="step-badge">In attesa</span>
      </div>
      <div class="step-row pending" id="sr-transcribe">
        <div class="step-dot">🎤</div>
        <div class="step-info">
          <div class="step-name">Trascrizione Whisper</div>
          <div class="step-detail">Converte l'audio in testo</div>
        </div>
        <span class="step-badge">In attesa</span>
      </div>
      <div class="step-row pending" id="sr-analyze">
        <div class="step-dot">🤖</div>
        <div class="step-info">
          <div class="step-name">Analisi Claude AI</div>
          <div class="step-detail">Voto, commento e prompt di miglioramento</div>
        </div>
        <span class="step-badge">In attesa</span>
      </div>
    </div>
  </section>

  <!-- ── Results Phase ── -->
  <section id="results-section" hidden>

    <div class="score-card" id="score-card">
      <div class="score-glow"></div>
      <div class="score-left">
        <div class="score-eyebrow" id="res-eyebrow">Voto complessivo</div>
        <div>
          <span class="score-big" id="res-score">—</span>
          <span class="score-denom">/10</span>
        </div>
        <div class="score-stars" id="res-stars"></div>
      </div>
      <div class="score-right">
        <div class="score-stat-label">Parole trascritte</div>
        <div class="score-stat-value" id="res-words">—</div>
      </div>
    </div>

    <div class="two-col">
      <div class="card">
        <div class="card-eyebrow">💬 Commento</div>
        <div class="card-body" id="res-commento"></div>
      </div>
      <div class="card">
        <div class="card-eyebrow">🚀 Prompt di Miglioramento</div>
        <div class="card-body" id="res-prompt"></div>
      </div>
    </div>

    <div class="transcript-wrap" id="transcript-wrap">
      <button class="transcript-toggle" onclick="toggleTranscript()">
        <span>📝 Trascrizione completa</span>
        <span class="transcript-arrow">▼</span>
      </button>
      <div class="transcript-body" id="res-transcript"></div>
    </div>

    <div class="actions">
      <button class="btn" onclick="copyText()">📋 Copia testo</button>
      <button class="btn" onclick="exportPDF()">📄 Esporta PDF</button>
      <button class="btn primary" onclick="reset()">✨ Nuovo video</button>
    </div>

  </section>

  <!-- ── Error Phase ── -->
  <section id="error-section" hidden>
    <div class="error-box">
      <div class="error-icon">❌</div>
      <div>
        <div class="error-title">Errore nell'analisi</div>
        <div class="error-msg" id="error-msg"></div>
      </div>
    </div>
    <div class="actions">
      <button class="btn primary" onclick="reset()">← Riprova</button>
    </div>
  </section>

</main>

<footer>GoesToYou.video · Whisper AI + Claude · Analisi video automatica</footer>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let resultData = null;

const STEP_IDS = {
  download:  'sr-download',
  audio:     'sr-audio',
  transcribe:'sr-transcribe',
  analyze:   'sr-analyze',
};
const STEP_ORDER = ['download', 'audio', 'transcribe', 'analyze'];

// ── Helpers ────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

function setStep(key, state) {
  const el = $(STEP_IDS[key]);
  if (!el) return;
  el.className = 'step-row ' + state;
  const badge = el.querySelector('.step-badge');
  badge.textContent = state === 'active' ? '⟳ In corso'
                    : state === 'done'   ? '✓ Fatto'
                    :                     'In attesa';
}

function activateStep(key) {
  const idx = STEP_ORDER.indexOf(key);
  STEP_ORDER.forEach((k, i) => {
    setStep(k, i < idx ? 'done' : i === idx ? 'active' : 'pending');
  });
}

function allDone() { STEP_ORDER.forEach(k => setStep(k, 'done')); }

function scoreColor(s) {
  return s >= 8 ? '#22d3a5' : s >= 5 ? '#f59e0b' : '#f87171';
}

function buildStars(s) { return '★'.repeat(s) + '☆'.repeat(10 - s); }

// ── Analysis ───────────────────────────────────────────────────────────────
async function startAnalysis() {
  const url = $('drive-url').value.trim();
  if (!url) { $('drive-url').focus(); return; }

  $('btn-start').disabled = true;
  ['results-section','error-section'].forEach(id => $(id).hidden = true);
  $('input-section').hidden = true;
  $('progress-section').hidden = false;
  STEP_ORDER.forEach(k => setStep(k, 'pending'));
  $('progress-msg').textContent = 'Avvio analisi…';

  try {
    const res = await fetch('/api/analyze', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ drive_url: url }),
    });

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\\n');
      buf = lines.pop();
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try { handleEvent(JSON.parse(line.slice(6))); } catch {}
        }
      }
    }
  } catch (err) {
    showError('Errore di connessione: ' + err.message);
  } finally {
    $('btn-start').disabled = false;
  }
}

function handleEvent({ step, message, data }) {
  $('progress-msg').textContent = message;

  if (step === 'error')  { showError(message); return; }
  if (step === 'done')   { allDone(); resultData = data; showResults(data); return; }
  if (STEP_IDS[step])    { activateStep(step); }
}

// ── Results ────────────────────────────────────────────────────────────────
function showResults({ score, commento, prompt, transcript, word_count }) {
  const color = scoreColor(score);

  // Score card
  $('res-score').textContent = score;
  $('res-stars').textContent = buildStars(score);
  $('res-words').textContent = word_count.toLocaleString('it-IT');
  $('res-eyebrow').style.color = color;
  $('res-score').style.color   = color;
  $('res-stars').style.color   = color;
  $('res-words').style.color   = color;
  const card = $('score-card');
  card.style.borderColor = color + '44';
  card.querySelector('.score-glow').style.background = color;

  // Cards
  $('res-commento').textContent = commento;
  $('res-prompt').textContent   = prompt;
  $('res-transcript').textContent = transcript;

  $('progress-section').hidden = true;
  $('results-section').hidden  = false;
}

function showError(msg) {
  $('error-msg').textContent  = msg;
  $('progress-section').hidden = true;
  $('error-section').hidden    = false;
}

// ── Transcript toggle ──────────────────────────────────────────────────────
function toggleTranscript() {
  $('transcript-wrap').classList.toggle('open');
}

// ── Export ─────────────────────────────────────────────────────────────────
function copyText() {
  if (!resultData) return;
  const { score, commento, prompt, transcript } = resultData;
  const txt = [
    'GoesToYou.video — Analisi AI',
    '='.repeat(40),
    '',
    `VOTO: ${score}/10`,
    '',
    'COMMENTO',
    commento,
    '',
    'PROMPT DI MIGLIORAMENTO',
    prompt,
    '',
    'TRASCRIZIONE',
    transcript,
  ].join('\\n');
  navigator.clipboard.writeText(txt).then(() => {
    const btn = event.target;
    btn.textContent = '✓ Copiato!';
    setTimeout(() => btn.textContent = '📋 Copia testo', 2200);
  });
}

function exportPDF() {
  if (!resultData || !window.jspdf) return;
  const { score, commento, prompt, transcript, word_count } = resultData;
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF();
  const W = doc.internal.pageSize.getWidth();
  let y = 22;

  const write = (text, opts = {}) => {
    const { size = 11, bold = false, color = [30,30,30] } = opts;
    doc.setFontSize(size);
    doc.setFont('helvetica', bold ? 'bold' : 'normal');
    doc.setTextColor(...color);
    const lines = doc.splitTextToSize(String(text), W - 40);
    lines.forEach(l => {
      if (y > 272) { doc.addPage(); y = 22; }
      doc.text(l, 20, y);
      y += size * 0.52 + 1.8;
    });
    y += 3;
  };

  write('GoesToYou.video — Analisi AI', { size: 18, bold: true, color: [127,90,240] });
  write(new Date().toLocaleDateString('it-IT', { dateStyle: 'long' }), { size: 9, color: [120,120,120] });
  y += 4;
  write(`Voto: ${score}/10    (${word_count} parole)`, { size: 16, bold: true });
  y += 3;
  write('COMMENTO', { size: 8, bold: true, color: [100,100,100] });
  write(commento);
  y += 2;
  write('PROMPT DI MIGLIORAMENTO', { size: 8, bold: true, color: [100,100,100] });
  write(prompt);
  y += 2;
  write('TRASCRIZIONE COMPLETA', { size: 8, bold: true, color: [100,100,100] });
  write(transcript, { size: 10, color: [70,70,70] });

  doc.save(`analisi-video-${new Date().toISOString().slice(0,10)}.pdf`);
}

// ── Reset ──────────────────────────────────────────────────────────────────
function reset() {
  resultData = null;
  $('drive-url').value = '';
  $('transcript-wrap').classList.remove('open');
  ['results-section','error-section','progress-section'].forEach(id => $(id).hidden = true);
  $('input-section').hidden = false;
}

// Enter to submit
$('drive-url').addEventListener('keydown', e => { if (e.key === 'Enter') startAnalysis(); });
</script>
</body>
</html>"""
