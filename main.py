import os, re, json, asyncio, tempfile, subprocess, glob, time
from pathlib import Path
from typing import Optional, AsyncGenerator
from urllib.parse import urlparse

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

GROQ_KEY      = os.environ.get("GROQ_API_KEY", "")
APP_ENV       = os.environ.get("APP_ENV", "production").lower()
IS_PROD       = APP_ENV not in {"dev", "development", "local", "test"}

MAX_TEXT_CHARS   = int(os.environ.get("MAX_TEXT_CHARS", "12000"))
MAX_URL_CHARS    = int(os.environ.get("MAX_URL_CHARS", "2048"))
MAX_DRIVE_MB     = int(os.environ.get("MAX_DRIVE_MB", "120"))
RATE_LIMIT_COUNT = int(os.environ.get("RATE_LIMIT_COUNT", "25"))
RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", "3600"))
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()
]

app = FastAPI(
    docs_url=None if IS_PROD else "/docs",
    redoc_url=None if IS_PROD else "/redoc",
    openapi_url=None if IS_PROD else "/openapi.json",
)

if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["POST", "GET"],
        allow_headers=["Content-Type"],
    )

_rate_buckets: dict[str, list[float]] = {}


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'self';",
    )
    if IS_PROD:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _check_rate_limit(key: str) -> None:
    now = time.monotonic()
    cutoff = now - RATE_LIMIT_WINDOW
    hits = [t for t in _rate_buckets.get(key, []) if t > cutoff]
    if len(hits) >= RATE_LIMIT_COUNT:
        raise HTTPException(status_code=429, detail="Troppe richieste. Riprova piu tardi.")
    hits.append(now)
    _rate_buckets[key] = hits


# ── URL Detection ─────────────────────────────────────────────────────────────

def _hostname(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().strip(".")
    except Exception:
        return ""


def _host_matches(host: str, allowed: tuple[str, ...]) -> bool:
    return any(host == d or host.endswith("." + d) for d in allowed)


def detect_url_type(url: str) -> str:
    host = _hostname(url)
    if _host_matches(host, ("drive.google.com",)):                 return "drive"
    if _host_matches(host, ("youtube.com", "youtu.be")):           return "youtube"
    if _host_matches(host, ("tiktok.com",)):                       return "tiktok"
    if _host_matches(host, ("instagram.com",)):                    return "instagram"
    if _host_matches(host, ("facebook.com", "fb.watch")):          return "facebook"
    if _host_matches(host, ("twitter.com", "x.com")):              return "twitter"
    return "unsupported"

def get_drive_id(url: str) -> Optional[str]:
    if detect_url_type(url) != "drive":
        return None
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url or "")
    return m.group(1) if m else None

def get_youtube_id(url: str) -> Optional[str]:
    if detect_url_type(url) != "youtube":
        return None
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})", url or "")
    return m.group(1) if m else None


# ── Subtitle / Caption Fetching (fast — no video download) ────────────────────

def _parse_vtt(vtt: str) -> str:
    """Convert WebVTT text to plain string, deduplicating rolling captions."""
    seen, out = set(), []
    for line in vtt.split("\n"):
        line = line.strip()
        if not line or "-->" in line or line.startswith("WEBVTT") or line.isdigit():
            continue
        clean = re.sub(r"<[^>]+>", "", line).strip()
        if clean and clean not in seen:
            out.append(clean)
            seen.add(clean)
    return " ".join(out)

def _fetch_youtube_transcript(video_id: str) -> Optional[str]:
    """Use youtube-transcript-api for fastest YouTube caption retrieval."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
        for langs in [["it"], ["it-IT"], ["en"], ["en-US"], None]:
            try:
                kwargs = {"languages": langs} if langs else {}
                data = YouTubeTranscriptApi.get_transcript(video_id, **kwargs)
                return " ".join(t["text"] for t in data).strip()
            except (NoTranscriptFound, Exception):
                continue
    except Exception:
        pass
    return None

def _fetch_ytdlp_subs(url: str) -> Optional[str]:
    """Fetch auto-captions via yt-dlp WITHOUT downloading the video."""
    try:
        import yt_dlp
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "sub")
            ydl_opts = {
                "skip_download": True,
                "writeautomaticsub": True,
                "writesubtitles": True,
                "subtitleslangs": ["it", "it-IT", "en", "en-US"],
                "subtitlesformat": "vtt",
                "outtmpl": base,
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            files = glob.glob(base + "*.vtt")
            if files:
                with open(files[0], "r", encoding="utf-8") as f:
                    text = _parse_vtt(f.read())
                if text and len(text) > 50:
                    return text
    except Exception:
        pass
    return None

def _fetch_subtitles(url: str, url_type: str) -> Optional[str]:
    """Main subtitle fetcher: tries fastest method first."""
    if url_type == "youtube":
        vid_id = get_youtube_id(url)
        if vid_id:
            t = _fetch_youtube_transcript(vid_id)
            if t:
                return t
    # Fallback: yt-dlp subtitle extraction (works for TikTok, Instagram, etc.)
    return _fetch_ytdlp_subs(url)


# ── Google Drive fallback: download + Whisper ─────────────────────────────────

def _download_drive(file_id: str, dest: Path) -> bool:
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}&confirm=t"
    try:
        gdown.download(url, str(dest), quiet=True)
        if not dest.exists() or dest.stat().st_size <= 10_000:
            return False
        return dest.stat().st_size <= MAX_DRIVE_MB * 1_048_576
    except Exception:
        return False

def _extract_audio(video: Path, audio: Path) -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg", "-i", str(video), "-ar", "16000", "-ac", "1",
             "-c:a", "pcm_s16le", str(audio), "-y", "-loglevel", "error"],
            capture_output=True,
            timeout=180,
        )
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False

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


# ── AI Analysis ───────────────────────────────────────────────────────────────

def _analyze(transcript: str) -> dict:
    text = re.sub(r"\s+", " ", transcript.strip())
    words = re.findall(r"\b[\wÀ-ÿ']+\b", text.lower())
    word_count = len(words)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    avg_sentence = word_count / max(1, len(sentences))
    unique_ratio = len(set(words)) / max(1, word_count)
    question_count = transcript.count("?")
    hook_terms = {"oggi", "ecco", "perche", "perché", "scopri", "attenzione", "errore", "segreto", "come"}
    action_terms = {"clicca", "commenta", "salva", "condividi", "scrivi", "prova", "guarda", "seguimi", "iscriviti"}
    hook_hits = sum(1 for w in words[:40] if w in hook_terms)
    action_hits = sum(1 for w in words[-80:] if w in action_terms)

    score = 5
    if 70 <= word_count <= 450:
        score += 1
    if 8 <= avg_sentence <= 24:
        score += 1
    if unique_ratio >= 0.45:
        score += 1
    if hook_hits:
        score += 1
    if question_count or action_hits:
        score += 1
    if word_count < 40:
        score -= 2
    if avg_sentence > 32:
        score -= 1
    score = max(1, min(10, score))

    strengths = []
    weaknesses = []
    if hook_hits:
        strengths.append("l'apertura contiene parole che aiutano ad agganciare l'attenzione")
    else:
        weaknesses.append("l'apertura puo essere piu incisiva nei primi secondi")
    if 8 <= avg_sentence <= 24:
        strengths.append("il ritmo delle frasi e abbastanza leggibile")
    else:
        weaknesses.append("le frasi sembrano troppo lunghe o troppo frammentate")
    if action_hits:
        strengths.append("c'e una call to action riconoscibile")
    else:
        weaknesses.append("manca una call to action chiara nel finale")
    if unique_ratio >= 0.45:
        strengths.append("il vocabolario e sufficientemente vario")
    else:
        weaknesses.append("alcuni termini si ripetono e possono rendere il messaggio meno dinamico")

    commento = (
        f"Il contenuto ottiene {score}/10: "
        f"{'; '.join(strengths[:2]) if strengths else 'ha una base comprensibile'}. "
        f"Da migliorare: {'; '.join(weaknesses[:2]) if weaknesses else 'rafforza ancora promessa e chiusura'}. "
        f"Il testo conta circa {word_count} parole con frasi da {avg_sentence:.0f} parole in media, quindi lavora soprattutto su chiarezza, ritmo e invito finale."
    )
    prompt = (
        "Riscrivi questo video con un hook piu forte nei primi 3 secondi, una promessa specifica e un esempio concreto. "
        "Mantieni frasi brevi, elimina ripetizioni e organizza il messaggio in problema, soluzione e beneficio. "
        "Chiudi con una call to action semplice e coerente con l'obiettivo del video. "
        "Tono: diretto, naturale, utile, senza sembrare pubblicita forzata."
    )
    return {"score": score, "commento": commento, "prompt": prompt}


# ── SSE helpers ───────────────────────────────────────────────────────────────

def evt(step: str, message: str, data: dict = None) -> str:
    p = {"step": step, "message": message}
    if data:
        p["data"] = data
    return f"data: {json.dumps(p, ensure_ascii=False)}\n\n"

async def run_sync(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)


# ── Stream Generators ─────────────────────────────────────────────────────────

async def analyze_text_stream(text: str) -> AsyncGenerator[str, None]:
    text = text.strip()
    if len(text) < 20:
        yield evt("error", "Testo troppo corto. Inserisci almeno qualche frase.")
        return
    word_count = len(text.split())
    yield evt("analyze", f"Analizzando lo script ({word_count} parole)...")
    try:
        analysis = await run_sync(_analyze, text)
    except Exception as e:
        yield evt("error", f"Errore analisi AI: {e}")
        return
    yield evt("done", "Analisi completata!", {"transcript": text, "word_count": word_count, **analysis})


async def analyze_url_stream(url: str, url_type: str) -> AsyncGenerator[str, None]:
    if url_type == "unsupported":
        yield evt("error", "Link non supportato. Usa YouTube, TikTok, Instagram, Facebook/X o Google Drive.")
        return

    # ── FAST PATH: fetch subtitles without downloading ────────────────────────
    if url_type != "drive":
        platform_names = {
            "youtube": "YouTube", "tiktok": "TikTok",
            "instagram": "Instagram", "facebook": "Facebook",
            "twitter": "Twitter/X", "social": "piattaforma",
        }
        pname = platform_names.get(url_type, "piattaforma")
        yield evt("fetch", f"Recuperando i sottotitoli da {pname}...")

        transcript = await run_sync(_fetch_subtitles, url, url_type)

        if transcript:
            word_count = len(transcript.split())
            yield evt("analyze", f"Analizzando il testo ({word_count} parole)...")
            try:
                analysis = await run_sync(_analyze, transcript)
            except Exception as e:
                yield evt("error", f"Errore analisi AI: {e}")
                return
            yield evt("done", "Analisi completata!", {
                "transcript": transcript, "word_count": word_count, **analysis
            })
            return
        else:
            yield evt("error",
                f"Nessun sottotitolo disponibile su {pname}. "
                "Copia il testo del video nella tab 📝 Script per analizzarlo.")
            return

    # ── SLOW PATH: Google Drive → download + Whisper ──────────────────────────
    file_id = get_drive_id(url)
    if not file_id:
        yield evt("error", "URL Google Drive non valido. Deve contenere /file/d/...")
        return

    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "video.mp4"
        audio = Path(tmp) / "audio.wav"

        yield evt("download", "Scaricando il video da Google Drive...")
        ok = await run_sync(_download_drive, file_id, video)
        if not ok:
            yield evt("error", "Download fallito. Il link Drive deve essere pubblico.")
            return

        size_mb = video.stat().st_size / 1_048_576
        yield evt("audio", f"Estraendo la traccia audio ({size_mb:.1f} MB)...")
        ok = await run_sync(_extract_audio, video, audio)
        if not ok:
            yield evt("error", "Errore estrazione audio.")
            return

        yield evt("transcribe", "Trascrivendo con Whisper AI...")
        try:
            transcript = await run_sync(_transcribe, audio)
        except Exception as e:
            yield evt("error", f"Errore trascrizione: {e}")
            return

        word_count = len(transcript.split())
        yield evt("analyze", f"Analizzando il testo ({word_count} parole)...")
        try:
            analysis = await run_sync(_analyze, transcript)
        except Exception as e:
            yield evt("error", f"Errore analisi AI: {e}")
            return

        yield evt("done", "Analisi completata!", {
            "transcript": transcript, "word_count": word_count, **analysis
        })


# ── Routes ────────────────────────────────────────────────────────────────────

class AnalyzeReq(BaseModel):
    input_type: str = Field(pattern="^(text|url)$")
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str, info):
        cleaned = value.strip()
        input_type = info.data.get("input_type")
        if input_type == "text":
            if len(cleaned) > MAX_TEXT_CHARS:
                raise ValueError(f"Testo troppo lungo. Limite: {MAX_TEXT_CHARS} caratteri.")
            return cleaned
        if len(cleaned) > MAX_URL_CHARS:
            raise ValueError("URL troppo lungo.")
        parsed = urlparse(cleaned)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("URL non valido.")
        if detect_url_type(cleaned) == "unsupported":
            raise ValueError("Piattaforma non supportata.")
        return cleaned

@app.post("/api/analyze")
async def analyze(req: AnalyzeReq, request: Request):
    _check_rate_limit(_client_ip(request))
    if req.input_type == "text":
        gen = analyze_text_stream(req.content)
    else:
        gen = analyze_url_stream(req.content, detect_url_type(req.content))
    return StreamingResponse(gen, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/api/health")
def health():
    return {"ok": True}

@app.get("/")
def index():
    return HTMLResponse(HTML)


# ── HTML ──────────────────────────────────────────────────────────────────────

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
  --bg:#09090d; --surface:#13131a; --surface2:#1c1c26;
  --border:rgba(255,255,255,0.07); --primary:#7f5af0; --primary-light:#a78bfa;
  --text:#e2e8f0; --muted:#64748b; --green:#22d3a5; --amber:#f59e0b; --red:#f87171;
}
body { font-family:'Inter',sans-serif; background:var(--bg); color:var(--text); min-height:100vh; display:flex; flex-direction:column; }

header { padding:1.1rem 2rem; border-bottom:1px solid var(--border); display:flex; align-items:center; gap:.75rem; position:sticky; top:0; background:rgba(9,9,13,.85); backdrop-filter:blur(12px); z-index:100; }
.logo { font-size:1rem; font-weight:700; background:linear-gradient(135deg,#a78bfa,#38bdf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.badge { font-size:.6rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:var(--primary); border:1px solid rgba(127,90,240,.35); padding:.2rem .55rem; border-radius:999px; background:rgba(127,90,240,.08); }

main { flex:1; display:flex; flex-direction:column; align-items:center; padding:4rem 1.5rem 3rem; width:100%; max-width:860px; margin:0 auto; }

/* Input */
#input-section { width:100%; text-align:center; }
.hero-eyebrow { font-size:.72rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:var(--primary-light); margin-bottom:1rem; }
.hero-title { font-size:clamp(2rem,5vw,3.2rem); font-weight:800; line-height:1.12; letter-spacing:-.03em; margin-bottom:.85rem; background:linear-gradient(160deg,#fff 50%,#a78bfa 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.hero-sub { color:var(--muted); font-size:1rem; line-height:1.65; margin-bottom:2.25rem; max-width:540px; margin-left:auto; margin-right:auto; }

.tab-switcher { display:inline-flex; gap:.35rem; background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:.35rem; margin-bottom:1.5rem; }
.tab-btn { padding:.55rem 1.35rem; border-radius:9px; border:none; background:transparent; color:var(--muted); font-family:'Inter',sans-serif; font-size:.87rem; font-weight:600; cursor:pointer; transition:all .2s; }
.tab-btn.active { background:var(--primary); color:#fff; }
.tab-btn:hover:not(.active) { color:var(--text); background:var(--surface2); }

/* Script panel */
.script-area { width:100%; min-height:160px; background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:1.2rem 1.4rem; color:var(--text); font-family:'Inter',sans-serif; font-size:.92rem; line-height:1.75; resize:vertical; outline:none; transition:border-color .2s,box-shadow .2s; display:block; }
.script-area:focus { border-color:rgba(127,90,240,.55); box-shadow:0 0 0 4px rgba(127,90,240,.1); }
.script-area::placeholder { color:var(--muted); }
.word-count { font-size:.76rem; color:var(--muted); text-align:right; margin-top:.5rem; margin-bottom:.85rem; }
.btn-full { width:100%; background:linear-gradient(135deg,#7f5af0,#6366f1); color:#fff; border:none; padding:.95rem 2rem; font-size:.95rem; font-weight:600; font-family:'Inter',sans-serif; cursor:pointer; border-radius:12px; transition:opacity .15s,transform .1s; }
.btn-full:hover { opacity:.88; } .btn-full:active { transform:scale(.99); } .btn-full:disabled { opacity:.45; cursor:not-allowed; }

/* URL panel */
.platform-grid { display:flex; flex-wrap:wrap; gap:.5rem; justify-content:center; margin-bottom:1.1rem; }
.pill { display:inline-flex; align-items:center; gap:.35rem; padding:.3rem .8rem; border-radius:999px; font-size:.74rem; font-weight:600; border:1px solid var(--border); background:var(--surface2); color:var(--muted); }
.url-detect { display:inline-flex; align-items:center; gap:.45rem; padding:.28rem .8rem; border-radius:999px; font-size:.74rem; font-weight:600; border:1px solid var(--border); background:var(--surface2); color:var(--muted); margin-bottom:.75rem; transition:all .2s; }
.url-detect.youtube   { color:#f87171; border-color:rgba(248,113,113,.3); background:rgba(248,113,113,.07); }
.url-detect.tiktok    { color:#a78bfa; border-color:rgba(167,139,250,.3); background:rgba(167,139,250,.07); }
.url-detect.instagram { color:#f9a8d4; border-color:rgba(249,168,212,.3); background:rgba(249,168,212,.07); }
.url-detect.drive     { color:#4ade80; border-color:rgba(74,222,128,.3);  background:rgba(74,222,128,.07); }
.url-detect.social    { color:#38bdf8; border-color:rgba(56,189,248,.3);  background:rgba(56,189,248,.07); }
.input-box { background:var(--surface); border:1px solid var(--border); border-radius:16px; overflow:hidden; display:flex; transition:border-color .2s,box-shadow .2s; margin-bottom:.75rem; }
.input-box:focus-within { border-color:rgba(127,90,240,.55); box-shadow:0 0 0 4px rgba(127,90,240,.1); }
.input-box input { flex:1; background:transparent; border:none; outline:none; padding:1.1rem 1.4rem; font-size:.92rem; color:var(--text); font-family:'Inter',sans-serif; }
.input-box input::placeholder { color:var(--muted); }
.btn-go { background:linear-gradient(135deg,#7f5af0,#6366f1); color:#fff; border:none; padding:0 1.8rem; font-size:.92rem; font-weight:600; font-family:'Inter',sans-serif; cursor:pointer; white-space:nowrap; transition:opacity .15s; }
.btn-go:hover { opacity:.88; } .btn-go:disabled { opacity:.45; cursor:not-allowed; }
.input-hint { font-size:.78rem; color:var(--muted); text-align:left; }
.drive-note { font-size:.76rem; color:var(--muted); margin-top:.5rem; display:none; }
.drive-note.show { display:block; }

/* Progress */
#progress-section { width:100%; }
.progress-header { text-align:center; margin-bottom:2rem; }
.progress-title { font-size:1.3rem; font-weight:700; letter-spacing:-.02em; margin-bottom:.4rem; }
.progress-sub { font-size:.88rem; color:var(--muted); min-height:1.3em; }
.steps-list { display:flex; flex-direction:column; background:var(--surface); border:1px solid var(--border); border-radius:16px; overflow:hidden; }
.step-row { display:flex; align-items:center; gap:1.1rem; padding:1.15rem 1.5rem; border-bottom:1px solid var(--border); transition:all .3s; }
.step-row:last-child { border-bottom:none; }
.step-row.active  { background:rgba(127,90,240,.06); }
.step-row.done    { background:rgba(34,211,165,.04); }
.step-row.skipped { opacity:.3; }
.step-dot { width:38px; height:38px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:1.05rem; flex-shrink:0; border:1.5px solid var(--border); background:var(--surface2); transition:all .3s; }
.step-row.active .step-dot { border-color:var(--primary); background:rgba(127,90,240,.12); animation:pulse 1.8s infinite; }
.step-row.done   .step-dot { border-color:var(--green); background:rgba(34,211,165,.1); }
@keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(127,90,240,.5)}60%{box-shadow:0 0 0 7px rgba(127,90,240,0)} }
.step-info { flex:1; }
.step-name   { font-size:.9rem; font-weight:600; margin-bottom:.15rem; }
.step-detail { font-size:.76rem; color:var(--muted); }
.step-badge  { font-size:.75rem; font-weight:600; padding:.2rem .6rem; border-radius:999px; }
.step-row.pending .step-badge { color:var(--muted); }
.step-row.active  .step-badge { color:var(--primary-light); background:rgba(127,90,240,.12); }
.step-row.done    .step-badge { color:var(--green); background:rgba(34,211,165,.1); }
.step-row.skipped .step-badge { color:var(--muted); }

/* Results */
#results-section { width:100%; }
.score-card { border:1px solid rgba(255,255,255,.08); border-radius:20px; padding:2rem 2.25rem; margin-bottom:1.1rem; display:flex; align-items:center; justify-content:space-between; position:relative; overflow:hidden; background:var(--surface); }
.score-glow { position:absolute; inset:0; opacity:.07; pointer-events:none; }
.score-left { position:relative; z-index:1; }
.score-eyebrow { font-size:.68rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.35rem; }
.score-big { font-size:4.5rem; font-weight:900; line-height:1; letter-spacing:-.04em; }
.score-denom { font-size:1.6rem; font-weight:400; color:var(--muted); }
.score-stars { font-size:1.15rem; margin-top:.4rem; letter-spacing:.05em; }
.score-right { position:relative; z-index:1; text-align:right; }
.score-stat-label { font-size:.75rem; color:var(--muted); margin-bottom:.25rem; }
.score-stat-value { font-size:2rem; font-weight:800; letter-spacing:-.03em; }
.two-col { display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem; }
@media(max-width:580px){.two-col{grid-template-columns:1fr}}
.card { background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:1.4rem 1.5rem; }
.card-eyebrow { font-size:.68rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); margin-bottom:.7rem; }
.card-body { font-size:.9rem; line-height:1.75; color:#cbd5e1; }
.transcript-wrap { background:var(--surface); border:1px solid var(--border); border-radius:14px; overflow:hidden; margin-bottom:1.25rem; }
.transcript-toggle { width:100%; display:flex; align-items:center; justify-content:space-between; padding:1.1rem 1.5rem; cursor:pointer; background:transparent; border:none; color:var(--text); font-family:'Inter',sans-serif; font-size:.88rem; font-weight:600; text-align:left; transition:background .15s; }
.transcript-toggle:hover { background:var(--surface2); }
.t-arrow { transition:transform .25s; color:var(--muted); font-size:.7rem; }
.transcript-wrap.open .t-arrow { transform:rotate(180deg); }
.transcript-body { display:none; padding:0 1.5rem 1.4rem; font-size:.86rem; line-height:1.85; color:#94a3b8; white-space:pre-wrap; border-top:1px solid var(--border); }
.transcript-wrap.open .transcript-body { display:block; }
.actions { display:flex; gap:.75rem; flex-wrap:wrap; }
.btn { flex:1; min-width:130px; padding:.8rem 1.2rem; border-radius:11px; font-size:.86rem; font-weight:600; font-family:'Inter',sans-serif; cursor:pointer; text-align:center; transition:all .15s; border:1px solid var(--border); background:var(--surface); color:var(--text); }
.btn:hover { background:var(--surface2); } .btn.primary { background:linear-gradient(135deg,#7f5af0,#6366f1); border:none; color:#fff; } .btn.primary:hover { opacity:.88; }
.error-box { background:rgba(248,113,113,.07); border:1px solid rgba(248,113,113,.22); border-radius:14px; padding:1.5rem; display:flex; gap:1rem; margin-bottom:1.25rem; }
.error-icon { font-size:1.3rem; flex-shrink:0; }
.error-title { font-weight:700; color:#fca5a5; margin-bottom:.3rem; }
.error-msg { font-size:.88rem; color:#f87171; line-height:1.55; }
footer { text-align:center; padding:1.4rem; font-size:.74rem; color:#374151; border-top:1px solid var(--border); margin-top:auto; }
[hidden] { display:none !important; }
</style>
</head>
<body>

<header>
  <span class="logo">GoesToYou.video</span>
  <span class="badge">AI Analysis</span>
</header>

<main>

  <!-- INPUT -->
  <section id="input-section">
    <p class="hero-eyebrow">Powered by Whisper + Smart Scoring</p>
    <h1 class="hero-title">Valida la tua idea video</h1>
    <p class="hero-sub">Incolla lo script oppure il link del video — YouTube, TikTok, Instagram e altri.<br>Risultati in pochi secondi.</p>

    <div class="tab-switcher">
      <button class="tab-btn active" id="tab-text" onclick="setTab('text')">📝 Script</button>
      <button class="tab-btn" id="tab-link" onclick="setTab('link')">🔗 Link Video</button>
    </div>

    <!-- Script -->
    <div id="panel-text">
      <textarea class="script-area" id="script-input"
        placeholder="Incolla qui lo script o il testo del tuo video...&#10;&#10;Funziona anche con sottotitoli copiati, trascrizioni, bozze di contenuto."
        oninput="updateWordCount()"></textarea>
      <p class="word-count"><span id="word-count">0</span> parole</p>
      <button class="btn-full" id="btn-text" onclick="startAnalysis()">Analizza →</button>
    </div>

    <!-- Link -->
    <div id="panel-link" hidden>
      <div class="platform-grid">
        <span class="pill">▶️ YouTube</span>
        <span class="pill">🎵 TikTok</span>
        <span class="pill">📸 Instagram</span>
        <span class="pill">📘 Facebook</span>
        <span class="pill">☁️ Google Drive</span>
      </div>
      <div id="url-badge" class="url-detect">🔗 Incolla un link per rilevare la piattaforma</div>
      <div class="input-box">
        <input type="url" id="url-input"
               placeholder="https://youtube.com/...  •  tiktok.com/...  •  instagram.com/..."
               oninput="onUrlChange()" />
        <button class="btn-go" id="btn-link" onclick="startAnalysis()">Analizza →</button>
      </div>
      <p class="input-hint" id="url-hint">💡 Vengono recuperati i sottotitoli senza scaricare il video — velocissimo</p>
      <p class="drive-note" id="drive-note">⚠️ Google Drive richiede trascrizione audio (2-4 min). Per video brevi è OK.</p>
    </div>
  </section>

  <!-- PROGRESS -->
  <section id="progress-section" hidden>
    <div class="progress-header">
      <h2 class="progress-title">Elaborazione in corso…</h2>
      <p class="progress-sub" id="progress-msg">Avvio…</p>
    </div>
    <div class="steps-list">
      <div class="step-row pending" id="sr-fetch">
        <div class="step-dot">⚡</div>
        <div class="step-info">
          <div class="step-name">Recupero sottotitoli</div>
          <div class="step-detail" id="sd-fetch">Prende le caption dalla piattaforma</div>
        </div>
        <span class="step-badge">In attesa</span>
      </div>
      <div class="step-row pending" id="sr-download">
        <div class="step-dot">⬇️</div>
        <div class="step-info">
          <div class="step-name">Download video</div>
          <div class="step-detail">Solo per Google Drive</div>
        </div>
        <span class="step-badge">In attesa</span>
      </div>
      <div class="step-row pending" id="sr-audio">
        <div class="step-dot">🎵</div>
        <div class="step-info">
          <div class="step-name">Estrazione audio</div>
          <div class="step-detail">Isola la traccia con ffmpeg</div>
        </div>
        <span class="step-badge">In attesa</span>
      </div>
      <div class="step-row pending" id="sr-transcribe">
        <div class="step-dot">🎤</div>
        <div class="step-info">
          <div class="step-name">Trascrizione Whisper</div>
          <div class="step-detail">Solo per Google Drive</div>
        </div>
        <span class="step-badge">In attesa</span>
      </div>
      <div class="step-row pending" id="sr-analyze">
        <div class="step-dot">🤖</div>
        <div class="step-info">
          <div class="step-name">Analisi contenuto</div>
          <div class="step-detail">Voto, commento e prompt di miglioramento</div>
        </div>
        <span class="step-badge">In attesa</span>
      </div>
    </div>
  </section>

  <!-- RESULTS -->
  <section id="results-section" hidden>
    <div class="score-card" id="score-card">
      <div class="score-glow" id="score-glow"></div>
      <div class="score-left">
        <div class="score-eyebrow" id="res-eyebrow">Voto complessivo</div>
        <div><span class="score-big" id="res-score">—</span><span class="score-denom">/10</span></div>
        <div class="score-stars" id="res-stars"></div>
      </div>
      <div class="score-right">
        <div class="score-stat-label" id="res-words-label">Parole analizzate</div>
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
        <span id="transcript-label">📝 Testo analizzato</span>
        <span class="t-arrow">▼</span>
      </button>
      <div class="transcript-body" id="res-transcript"></div>
    </div>
    <div class="actions">
      <button class="btn" onclick="copyText()">📋 Copia testo</button>
      <button class="btn" onclick="exportPDF()">📄 Esporta PDF</button>
      <button class="btn primary" onclick="reset()">✨ Nuova analisi</button>
    </div>
  </section>

  <!-- ERROR -->
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

<footer>GoesToYou.video · Sottotitoli istantanei + Smart Scoring · Analisi video rapida</footer>

<script>
let resultData  = null;
let currentMode = 'text';
let currentUrlType = '';

const STEP_IDS = { fetch:'sr-fetch', download:'sr-download', audio:'sr-audio', transcribe:'sr-transcribe', analyze:'sr-analyze' };
const ALL_STEPS = ['fetch','download','audio','transcribe','analyze'];
const $ = id => document.getElementById(id);

function setStep(key, state) {
  const el = $(STEP_IDS[key]); if (!el) return;
  el.className = 'step-row ' + state;
  el.querySelector('.step-badge').textContent =
    state==='active' ? '⟳ In corso' : state==='done' ? '✓ Fatto' : state==='skipped' ? '—' : 'In attesa';
}

function activateStep(key) {
  ALL_STEPS.forEach((k,i) => {
    const idx = ALL_STEPS.indexOf(key);
    if (i < idx) setStep(k, shouldSkip(k) ? 'skipped' : 'done');
    else if (i === idx) setStep(k, 'active');
    else setStep(k, shouldSkip(k) ? 'skipped' : 'pending');
  });
}

function shouldSkip(key) {
  if (currentMode === 'text') return key !== 'analyze';
  if (currentUrlType === 'drive') return key === 'fetch';
  return ['download','audio','transcribe'].includes(key);
}

function prepareSteps() {
  ALL_STEPS.forEach(k => setStep(k, shouldSkip(k) ? 'skipped' : 'pending'));
}

function allDone() { ALL_STEPS.forEach(k => setStep(k, shouldSkip(k) ? 'skipped' : 'done')); }

function scoreColor(s) { return s>=8?'#22d3a5':s>=5?'#f59e0b':'#f87171'; }
function buildStars(s) { return '★'.repeat(s)+'☆'.repeat(10-s); }

// Tabs
function setTab(mode) {
  currentMode = mode;
  $('tab-text').classList.toggle('active', mode==='text');
  $('tab-link').classList.toggle('active', mode==='link');
  $('panel-text').hidden = mode!=='text';
  $('panel-link').hidden = mode!=='link';
}

function updateWordCount() {
  const w = $('script-input').value.trim().split(/\\s+/).filter(Boolean).length;
  $('word-count').textContent = w;
}

// URL detection
const PLATFORMS = {
  youtube:   { label:'YouTube',     icon:'▶️', class:'youtube' },
  tiktok:    { label:'TikTok',      icon:'🎵', class:'tiktok' },
  instagram: { label:'Instagram',   icon:'📸', class:'instagram' },
  facebook:  { label:'Facebook',    icon:'📘', class:'social' },
  twitter:   { label:'Twitter/X',   icon:'🐦', class:'social' },
  drive:     { label:'Google Drive (lento)', icon:'☁️', class:'drive' },
  social:    { label:'Video',       icon:'🔗', class:'social' },
};

function detectType(url) {
  const u = url.toLowerCase();
  if (u.includes('drive.google.com'))            return 'drive';
  if (u.includes('youtube.com')||u.includes('youtu.be')) return 'youtube';
  if (u.includes('tiktok.com'))                  return 'tiktok';
  if (u.includes('instagram.com'))               return 'instagram';
  if (u.includes('facebook.com')||u.includes('fb.watch')) return 'facebook';
  if (u.includes('twitter.com')||u.includes('x.com')) return 'twitter';
  if (url.startsWith('http'))                    return 'social';
  return '';
}

function onUrlChange() {
  const url = $('url-input').value.trim();
  const badge = $('url-badge');
  const hint  = $('url-hint');
  const dnote = $('drive-note');
  const type  = detectType(url);
  currentUrlType = type;

  if (type && PLATFORMS[type]) {
    const p = PLATFORMS[type];
    badge.className = 'url-detect ' + p.class;
    badge.textContent = p.icon + '  ' + p.label + ' rilevato';
    if (type === 'drive') {
      hint.textContent = '💡 Recupero audio e trascrizione Whisper (2-4 min)';
      dnote.classList.add('show');
    } else {
      hint.textContent = '⚡ Sottotitoli istantanei — nessun download necessario';
      dnote.classList.remove('show');
    }
  } else {
    badge.className = 'url-detect';
    badge.textContent = '🔗 Incolla un link per rilevare la piattaforma';
    hint.textContent = '💡 Vengono recuperati i sottotitoli senza scaricare il video — velocissimo';
    dnote.classList.remove('show');
    currentUrlType = '';
  }
}

// Analysis
async function startAnalysis() {
  let inputType, content;
  if (currentMode === 'text') {
    content = $('script-input').value.trim();
    if (!content) { $('script-input').focus(); return; }
    inputType = 'text';
    $('btn-text').disabled = true;
    $('res-words-label').textContent  = 'Parole nello script';
    $('transcript-label').textContent = '📝 Script analizzato';
  } else {
    content = $('url-input').value.trim();
    if (!content) { $('url-input').focus(); return; }
    inputType = 'url';
    currentUrlType = detectType(content);
    $('btn-link').disabled = true;
    $('res-words-label').textContent  = currentUrlType==='drive' ? 'Parole trascritte' : 'Parole nei sottotitoli';
    $('transcript-label').textContent = currentUrlType==='drive' ? '📝 Trascrizione' : '📝 Sottotitoli';
  }

  ['results-section','error-section'].forEach(id=>$(id).hidden=true);
  $('input-section').hidden  = true;
  $('progress-section').hidden = false;
  prepareSteps();
  $('progress-msg').textContent = 'Avvio…';

  try {
    const res = await fetch('/api/analyze', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ input_type:inputType, content }),
    });
    if (!res.ok) {
      let msg = 'Richiesta non valida.';
      try {
        const err = await res.json();
        if (Array.isArray(err.detail)) msg = err.detail.map(e => e.msg).join(' ');
        else if (err.detail) msg = err.detail;
      } catch {}
      showError(msg);
      return;
    }
    if (!res.body) {
      showError('Risposta streaming non disponibile.');
      return;
    }
    const reader=res.body.getReader(), decoder=new TextDecoder();
    let buf='';
    while(true){
      const{done,value}=await reader.read(); if(done)break;
      buf+=decoder.decode(value,{stream:true});
      const lines=buf.split('\\n'); buf=lines.pop();
      for(const line of lines){
        if(line.startsWith('data: ')){try{handleEvent(JSON.parse(line.slice(6)))}catch{}}
      }
    }
  } catch(err){ showError('Errore di connessione: '+err.message); }
  finally { $('btn-text').disabled=false; $('btn-link').disabled=false; }
}

function handleEvent({step,message,data}){
  $('progress-msg').textContent=message;
  if(step==='error'){showError(message);return;}
  if(step==='done'){allDone();resultData=data;showResults(data);return;}
  if(STEP_IDS[step]) activateStep(step);
}

function showResults({score,commento,prompt,transcript,word_count}){
  const color=scoreColor(score);
  $('res-score').textContent=score; $('res-stars').textContent=buildStars(score);
  $('res-words').textContent=word_count.toLocaleString('it-IT');
  ['res-eyebrow','res-score','res-stars','res-words'].forEach(id=>$(id).style.color=color);
  $('score-card').style.borderColor=color+'44';
  $('score-glow').style.background=color;
  $('res-commento').textContent=commento; $('res-prompt').textContent=prompt;
  $('res-transcript').textContent=transcript;
  $('progress-section').hidden=true; $('results-section').hidden=false;
}

function showError(msg){
  $('error-msg').textContent=msg; $('progress-section').hidden=true; $('error-section').hidden=false;
}

function toggleTranscript(){ $('transcript-wrap').classList.toggle('open'); }

function copyText(){
  if(!resultData)return;
  const{score,commento,prompt,transcript}=resultData;
  navigator.clipboard.writeText(
    ['GoesToYou.video — Analisi AI','='.repeat(40),'','VOTO: '+score+'/10','','COMMENTO',commento,'','PROMPT DI MIGLIORAMENTO',prompt,'','TESTO ANALIZZATO',transcript].join('\\n')
  ).then(()=>{const b=event.target;b.textContent='✓ Copiato!';setTimeout(()=>b.textContent='📋 Copia testo',2200);});
}

function exportPDF(){
  if(!resultData||!window.jspdf)return;
  const{score,commento,prompt,transcript,word_count}=resultData;
  const{jsPDF}=window.jspdf,doc=new jsPDF();
  const W=doc.internal.pageSize.getWidth();let y=22;
  const w=(text,o={})=>{const{size=11,bold=false,color=[30,30,30]}=o;doc.setFontSize(size);doc.setFont('helvetica',bold?'bold':'normal');doc.setTextColor(...color);doc.splitTextToSize(String(text),W-40).forEach(l=>{if(y>272){doc.addPage();y=22;}doc.text(l,20,y);y+=size*.52+1.8;});y+=3;};
  w('GoesToYou.video — Analisi AI',{size:18,bold:true,color:[127,90,240]});
  w(new Date().toLocaleDateString('it-IT',{dateStyle:'long'}),{size:9,color:[120,120,120]});
  y+=4;w('Voto: '+score+'/10  ('+word_count+' parole)',{size:16,bold:true});y+=3;
  w('COMMENTO',{size:8,bold:true,color:[100,100,100]});w(commento);y+=2;
  w('PROMPT DI MIGLIORAMENTO',{size:8,bold:true,color:[100,100,100]});w(prompt);y+=2;
  w('TESTO ANALIZZATO',{size:8,bold:true,color:[100,100,100]});w(transcript,{size:10,color:[70,70,70]});
  doc.save('analisi-video-'+new Date().toISOString().slice(0,10)+'.pdf');
}

function reset(){
  resultData=null; currentUrlType='';
  $('script-input').value=''; $('url-input').value='';
  $('word-count').textContent='0';
  $('transcript-wrap').classList.remove('open');
  ['results-section','error-section','progress-section'].forEach(id=>$(id).hidden=true);
  $('input-section').hidden=false; onUrlChange();
}

$('url-input').addEventListener('keydown',e=>{if(e.key==='Enter')startAnalysis();});
$('script-input').addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key==='Enter')startAnalysis();});
</script>
</body>
</html>"""
