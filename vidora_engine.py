#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 VIDORA AI - Vidora Engine (vidora_engine.py)
============================================================================
Runs on YOUR GPU and connects to the Vidora AI Supabase database, polling
the `jobs` queue and processing each job:

  1. Dubbing    : download video -> Whisper transcription -> translate to
                  Bengali -> Edge-TTS voice-over -> FFmpeg final .mp4
  2. Animation  : AnimateDiff / Deforum text-to-video generation (optional,
                  heavy) via diffusers on the T4 GPU
  3. Saves the output MP4 to your Google Drive (or Supabase Storage
     fallback when Drive isn't available, e.g. on a plain GPU machine)
  4. Auto-publishes to your YouTube channel + Facebook Page (if enabled in
     the Customer Dashboard)

Where it runs (the engine you pick on the Vidora dashboard):
  * Google Colab (default)   : T4 GPU  -> heartbeats colab_engines
  * Kaggle notebook          : set VIDORA_NOTEBOOK=kaggle -> kaggle_engines
  * Your OWN NVIDIA GPU      : set VIDORA_ENGINE=nvidia (local RTX PC or a
    cloud NVIDIA VM)         -> heartbeats nvidia_engines

SETUP (in a Colab cell - use the values shown on vidora's /colab page):
    import os
    os.environ['SUPABASE_URL']    = 'https://YOUR-PROJECT-REF.supabase.co'
    os.environ['SUPABASE_ANON_KEY'] = 'YOUR_PUBLIC_ANON_KEY'
    os.environ['SUPABASE_ACCESS_TOKEN']  = 'YOUR_ACCESS_TOKEN'   # dashboard -> Colab guide
    os.environ['SUPABASE_REFRESH_TOKEN'] = 'YOUR_REFRESH_TOKEN'  # engine auto-refreshes
    !wget -q https://raw.githubusercontent.com/<you>/<repo>/main/colab/vidora_engine.py -O vidora_engine.py
    !python vidora_engine.py

SETUP (on your own NVIDIA GPU machine - /colab?via=nvidia):
    pip install -r requirements.txt
    export SUPABASE_URL=... SUPABASE_ANON_KEY=... \
           SUPABASE_ACCESS_TOKEN=... SUPABASE_REFRESH_TOKEN=... \
           VIDORA_ENGINE=nvidia
    python vidora_engine.py

SECURITY: the engine uses ONLY the signed-in user's Supabase access token
(YOUR access token, YOUR RLS scope). It never needs the service-role key and
never holds app secrets. RLS confines every table read/write to your own rows.
============================================================================
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone

import requests

# ---------------------------------------------------------------------------
# 0. CONFIG & BOOTSTRAP (install only what's missing)
# ---------------------------------------------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
# Access token is EITHER the customer's own session token (interactive Colab/
# Kaggle/NVIDIA) OR a short-lived per-job JWT minted by the pairing code /
# the RunPod poller (hosted "Vidora Cloud" worker). Both confine the engine to
# the user's own rows via the same RLS policies - no service key anywhere.
SUPABASE_ACCESS_TOKEN = os.environ.get("SUPABASE_ACCESS_TOKEN") or os.environ.get("VIDORA_JOB_TOKEN", "")
SUPABASE_REFRESH_TOKEN = os.environ.get("SUPABASE_REFRESH_TOKEN", "")
# Hosted mode: 1 = running as the Vidora Cloud worker (Drive tokens come from
# the stored customer tokens via the drive-token-refresh Edge Function).
VIDORA_HOSTED = os.environ.get("VIDORA_HOSTED", "") == "1"
DRIVE_ROOT = os.environ.get("DRIVE_ROOT", "/content/drive/MyDrive/VidoraAI/outputs")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "10"))  # seconds
LOG_THROTTLE = 3.0  # seconds between job-log DB writes
STALE_AFTER = os.environ.get("STALE_AFTER", "00:45:00")  # requeue threshold
MAX_POLL_BACKOFF = int(os.environ.get("MAX_POLL_BACKOFF", "120"))  # seconds

REQUIRED_PIP = [
    "openai-whisper",
    "edge-tts",
    "ffmpeg-python",
    "google-api-python-client",
    "google-auth-oauthlib",
    "facebook-sdk",
    "deep-translator",
    "requests",
]
# yt-dlp needed for YouTube URL extraction (installed separately - fast)
OPTIONAL_PIP = ["yt-dlp"]

if not SUPABASE_URL or not SUPABASE_ANON_KEY or not SUPABASE_ACCESS_TOKEN:
    raise SystemExit(
        "Set SUPABASE_URL, SUPABASE_ANON_KEY and SUPABASE_ACCESS_TOKEN first. "
        "Get a fresh pairing code from the Vidora AI dashboard -> 'Connect your machine'."
    )

REST = f"{SUPABASE_URL}/rest/v1"

# in-memory token store + thread-safe rotation (refresh token optional)
_tokens = {"access": SUPABASE_ACCESS_TOKEN, "refresh": SUPABASE_REFRESH_TOKEN}
_token_lock = threading.Lock()


def _current_token() -> str:
    with _token_lock:
        return _tokens["access"]


def _auth_headers(extra: dict | None = None) -> dict:
    h = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {_current_token()}",
        "Content-Type": "application/json",
    }
    if extra:
        h.update(extra)
    return h


def _refresh_access_token() -> bool:
    """Exchange a refresh token for a fresh access token (no service role)."""
    with _token_lock:
        refresh = _tokens["refresh"]
    if not refresh:
        return False
    try:
        r = requests.post(
            f"{SUPABASE_URL}/auth/v1/token?grant_type=refresh_token",
            json={"refresh_token": refresh},
            headers=_auth_headers({"Content-Type": "application/json"}),
            timeout=30,
        )
        data = r.json()
        new_access = data.get("access_token")
        if not new_access:
            return False
        with _token_lock:
            _tokens["access"] = new_access
            if data.get("refresh_token"):
                _tokens["refresh"] = data["refresh_token"]
        return True
    except Exception:
        return False


def shell(cmd: str, timeout: int = 1200) -> str:
    """Run a shell command and return (stdout+stderr)."""
    proc = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{proc.stderr[-2000:]}")
    return proc.stdout + proc.stderr


def install_packages() -> None:
    print("[setup] installing/verifying dependencies…", flush=True)
    for pkg in REQUIRED_PIP + OPTIONAL_PIP:
        shell(f"pip install -q -U {pkg}")
    # ffmpeg binary
    if shutil.which("ffmpeg") is None:
        shell("apt-get -qq update && apt-get -qq install -y ffmpeg >/dev/null 2>&1")
    print("[setup] dependencies ready", flush=True)


# ---------------------------------------------------------------------------
# 1. SUPABASE ACCESS (the user's OWN access token -> RLS confines us)
# ---------------------------------------------------------------------------
def _req(method: str, url_path: str, **kwargs):
    """HTTP helper with automatic token refresh on 401 + retry with backoff."""
    timeout = kwargs.pop("timeout", 30)
    last_exc = None
    for attempt in range(4):
        try:
            headers = dict(kwargs.pop("headers", {}))
            r = requests.request(
                method, f"{SUPABASE_URL}/{url_path}",
                headers=_auth_headers(headers), timeout=timeout,
                **kwargs,
            )
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 401 and attempt == 0 and _refresh_access_token():
            continue
        if 500 <= r.status_code < 600 and attempt < 3:
            time.sleep(2 ** attempt)
            continue
        return r
    raise RuntimeError(
        f"{method} {url_path} failed after retries: "
        + (str(last_exc) if last_exc else "server error")
    )


def supabase_get(table: str, query: dict | None = None, select: str | None = None):
    params = {}
    if select:
        params["select"] = select
    if query:
        params.update(query)
    return _req("GET", f"rest/v1/{table}?{'&'.join(f'{k}={v}' for k, v in params.items())}").json()


def supabase_patch(table: str, row_id: str, payload: dict, extra_cond: dict | None = None):
    cond = [f"id=eq.{row_id}"]
    if extra_cond:
        cond.extend(f"{k}=eq.{v}" for k, v in extra_cond.items())
    _req("PATCH", f"rest/v1/{table}?{'&'.join(cond)}", json=payload)


def supabase_upsert(table: str, payload: dict, on_conflict: str):
    _req(
        "POST", f"rest/v1/{table}?on_conflict={on_conflict}",
        json=payload,
        headers={"Prefer": "resolution=merge-duplicates"},
    )


def supabase_rpc(name: str, body: dict | None = None):
    return _req("POST", f"rest/v1/rpc/{name}", json=body or {}).json()


def fetch_user_id() -> str:
    """Resolve the token's user. Always the engine's own user (RLS-scoped)."""
    r = _req("GET", "auth/v1/user")
    if r.status_code != 200:
        raise SystemExit(f"Invalid access token (HTTP {r.status_code}). "
                         "Get a fresh one from the dashboard -> Colab guide.")
    return r.json()["id"]


# ---------------------------------------------------------------------------
# 2. HEARTBEAT - tells the dashboard the engine is online
# ---------------------------------------------------------------------------
def heartbeat_loop(user_id: str, stop_event: threading.Event):
    # Which heartbeat table we touch depends on where the engine runs:
    #   VIDORA_ENGINE=nvidia  -> nvidia_engines (your own NVIDIA GPU machine)
    #   VIDORA_NOTEBOOK=kaggle-> kaggle_engines (Kaggle notebook; legacy flag)
    #   anything else         -> colab_engines (Google Colab)
    engine = os.environ.get("VIDORA_ENGINE") or os.environ.get("VIDORA_NOTEBOOK") or "colab"
    table = {
        "colab": "colab_engines",
        "kaggle": "kaggle_engines",
        "nvidia": "nvidia_engines",
    }.get(engine, "colab_engines")
    # Immediate heartbeat so the dashboard shows "Connected!" instantly.
    try:
        supabase_upsert(
            table,
            {"user_id": user_id, "last_seen": now_iso()},
            "user_id",
        )
    except Exception:
        pass
    while not stop_event.is_set():
        try:
            supabase_upsert(
                table,
                {"user_id": user_id, "last_seen": now_iso()},
                "user_id",
            )
        except Exception:
            pass  # non-critical
        stop_event.wait(30)  # sleep 30s but wake instantly on shutdown


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# 3. QUEUE POLLING
# ---------------------------------------------------------------------------
def claim_job(job_id: str, user_id: str) -> bool:
    """ATOMIC claim: PATCH only succeeds while the row is still 'queued'
    and owned by us, and returns the row only in that case."""
    r = _req(
        "PATCH",
        f"rest/v1/jobs?id=eq.{job_id}&user_id=eq.{user_id}&status=eq.queued",
        json={"status": "processing", "started_at": now_iso()},
        headers={"Prefer": "return=representation"},
    )
    return len(r.json()) == 1


def next_job(user_id: str) -> dict | None:
    """Claim the oldest PAID queued job for this user (conditional update).
    paid=true is the only gate: license holders are pre-paid at submission,
    per-video jobs are paid via bKash before an engine may pick them up.
    NEVER claim unpaid jobs - that would be free work."""
    jobs = supabase_get(
        "jobs",
        query={
            "status": "eq.queued",
            "paid": "eq.true",
            "user_id": f"eq.{user_id}",
            "order": "created_at.asc",
            "limit": "1",
        },
        select="*",
    )
    for job in jobs:
        if claim_job(job["id"], user_id):
            return job
    return None


def has_license(user_id: str) -> bool:
    """True when the user holds an active (non-expired) license.

    NOTE: not used as a processing gate anymore. Jobs are processed only when
    paid=true (license holders are pre-paid at submission; per-video jobs are
    paid via bKash). Kept for diagnostics/tools."""
    try:
        rows = supabase_get(
            "licenses",
            query={
                "user_id": f"eq.{user_id}",
                "is_active": "eq.true",
                "order": "expires_at.desc",
                "limit": "1",
            },
            select="expires_at",
        )
    except Exception:
        return True
    if not rows:
        return False
    exp = rows[0].get("expires_at")
    if not exp:
        return True
    try:
        return datetime.fromisoformat(exp.replace("Z", "+00:00")) > datetime.now(timezone.utc)
    except Exception:
        return True


def requeue_stale_jobs() -> int:
    """Requeue any job stuck in 'processing' without a fresh heartbeat.
    Runs as security-definer RPC - only touches abandoned jobs."""
    try:
        n = supabase_rpc("requeue_stale_jobs", {"p_stale_after": STALE_AFTER})
        if n:
            print(f"[engine] requeued {n} stale job(s)", flush=True)
        return n or 0
    except Exception as e:
        print(f"[warn] requeue_stale_jobs failed: {e}", flush=True)
        return 0


# ---------------------------------------------------------------------------
# 4. JOB LOGGING (progress + log lines, throttled writes)
# ---------------------------------------------------------------------------
class JobLogger:
    def __init__(self, job_id: str, user_id: str):
        self.job_id = job_id
        self.user_id = user_id
        self.buffer: list[str] = []
        self.progress = 0
        self._lock = threading.Lock()
        self._last_write = 0.0

    def write(self, line: str):
        with self._lock:
            stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
            self.buffer.append(f"[{stamp}] {line}")
            print(line, flush=True)
            if time.time() - self._last_write > LOG_THROTTLE:
                self.flush_locked()

    def set_progress(self, pct: int):
        with self._lock:
            self.progress = max(0, min(100, pct))
            if time.time() - self._last_write > LOG_THROTTLE:
                self.flush_locked()

    def flush(self):
        with self._lock:
            self.flush_locked()

    def flush_locked(self):
        self._last_write = time.time()
        try:
            supabase_patch(
                "jobs", self.job_id,
                {"progress": self.progress, "log": "\n".join(self.buffer[-400:])},
            )
        except Exception as e:
            print(f"[warn] log write failed: {e}", flush=True)

    def finish(self, completed: bool, extra: dict | None = None):
        self.flush()
        payload = {
            "status": "completed" if completed else "failed",
            "progress": 100 if completed else self.progress,
            "completed_at": now_iso() if completed else None,
            "error": None if completed else "\n".join(self.buffer[-3:]),
        }
        if extra:
            payload.update(extra)
        try:
            supabase_patch("jobs", self.job_id, payload)
        except Exception as e:
            print(f"[warn] finish failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# 5. DUBBING PIPELINE
# ---------------------------------------------------------------------------
def download_video(url: str, workdir: str) -> str:
    """Download a video from YouTube (yt-dlp) or direct URL (requests)."""
    out = os.path.join(workdir, "source")
    os.makedirs(out, exist_ok=True)
    if re.search(r"(youtube\.com|youtu\.be)", url):
        shell(
            f"yt-dlp -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best' "
            f"-o '{out}/src.%(ext)s' --merge-output-format mp4 '{url}'"
        )
        path = os.path.join(out, "src.mp4")
    else:
        # direct media URL
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        path = os.path.join(out, "src.mp4")
        with open(path, "wb") as fh:
            shutil.copyfileobj(r.raw, fh)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError("Could not download the source video.")
    return path


def transcribe(video_path: str, logger: JobLogger) -> list[dict]:
    """Whisper transcription with per-segment timestamps."""
    import whisper

    logger.write("loading Whisper model (base)…")
    model = whisper.load_model("base")  # 'small'/'medium' for higher accuracy
    logger.set_progress(25)
    result = model.transcribe(video_path, language=None)  # auto-detect
    logger.write(f"transcription done - {len(result['segments'])} segments")
    return [
        {
            "start": s["start"],
            "end": s["end"],
            "text": s["text"].strip(),
        }
        for s in result["segments"]
        if s["text"].strip()
    ]


def translate_segments(segments: list[dict], source_lang: str, target_lang: str, logger: JobLogger) -> list[dict]:
    """Translate each segment to the target language (free GoogleTranslator)."""
    if source_lang and source_lang != "auto":
        # skip translation if source already equals target
        pass
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source="auto", target=target_lang)
    out = []
    for i, seg in enumerate(segments):
        try:
            seg["text"] = translator.translate(seg["text"])
        except Exception as e:
            logger.write(f"[warn] segment {i} translation failed: {e}")
        out.append(seg)
        if i % 10 == 0:
            logger.set_progress(30 + int(20 * (i / max(len(segments), 1))))
    logger.write(f"translated {len(out)} segments -> {target_lang}")
    return out


def synthesize_voiceover(segments: list[dict], voice: str, workdir: str, logger: JobLogger) -> str:
    """Edge-TTS per segment -> concatenated target-language audio track.

    Multi-character support: each segment may carry its own `voice` (assigned
    by assign_character_voices) so different characters sound distinct."""
    import asyncio
    import edge_tts

    async def gen_all():
        tasks = []
        for i, seg in enumerate(segments):
            out = os.path.join(workdir, "tts", f"{i:04d}.mp3")
            tasks.append((seg, out))
        os.makedirs(os.path.join(workdir, "tts"), exist_ok=True)

        warned_voices: set[str] = set()
        for idx, (seg, out) in enumerate(tasks):
            # speed up slightly to fit the original timing
            rate = "+5%"
            seg_voice = seg.get("voice") or voice
            # Robust fallback chain: assigned voice -> base voice -> English
            for attempt_voice in (seg_voice, voice, "en-US-JennyNeural"):
                try:
                    communicate = edge_tts.Communicate(seg["text"], attempt_voice, rate=rate)
                    await communicate.save(out)
                    break
                except Exception as e:
                    if attempt_voice not in warned_voices:
                        warned_voices.add(attempt_voice)
                        logger.write(f"[warn] voice {attempt_voice} unavailable ({e}); trying fallback")
            if idx % 10 == 0:
                logger.set_progress(55 + int(25 * (idx / max(len(tasks), 1))))

    asyncio.run(gen_all())
    logger.write("voice-over synthesized with Edge-TTS")
    return os.path.join(workdir, "tts")


# Distinct neural-voice pools per language prefix (female/male mixed). Used to
# give every character their own voice when num_speakers > 1.
VOICE_POOLS: dict[str, list[str]] = {
    "bn": ["bn-BD-NusratNeural", "bn-BD-PradeepNeural", "bn-IN-SwaraNeural", "bn-IN-RanbirNeural"],
    "en": ["en-US-JennyNeural", "en-US-GuyNeural", "en-US-AriaNeural", "en-US-ChristopherNeural", "en-GB-SoniaNeural", "en-GB-RyanNeural"],
    "hi": ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"],
    "ar": ["ar-SA-ZariyahNeural", "ar-SA-HamedNeural", "ar-EG-SalmaNeural", "ar-EG-ShakirNeural"],
    "es": ["es-ES-ElviraNeural", "es-ES-AlvaroNeural", "es-MX-DaliaNeural", "es-MX-JorgeNeural"],
    "fr": ["fr-FR-DeniseNeural", "fr-FR-HenriNeural", "fr-CA-SylvieNeural", "fr-CA-AntoineNeural"],
    "de": ["de-DE-KatjaNeural", "de-DE-ConradNeural", "de-DE-AmalaNeural"],
    "pt": ["pt-BR-FranciscaNeural", "pt-BR-AntonioNeural", "pt-PT-FernandaNeural"],
    "ru": ["ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"],
    "ja": ["ja-JP-NanamiNeural", "ja-JP-KeitaNeural"],
    "ko": ["ko-KR-SunHiNeural", "ko-KR-InJoonNeural"],
    "zh": ["zh-CN-XiaoxiaoNeural", "zh-CN-YunjianNeural", "zh-CN-XiaoyiNeural", "zh-CN-YunxiNeural"],
    "id": ["id-ID-GadisNeural", "id-ID-ArdiNeural"],
    "tr": ["tr-TR-EmelNeural", "tr-TR-AhmetNeural"],
    "vi": ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"],
    "ur": ["ur-PK-UzmaNeural", "ur-PK-AsadNeural"],
    "fa": ["fa-IR-DilaraNeural", "fa-IR-FaridNeural"],
}


def assign_character_voices(
    segments: list[dict], base_voice: str, num_speakers: int, logger: JobLogger
) -> None:
    """Give each segment one of `num_speakers` distinct voices.

    Heuristic diarization (no heavy pyannote dependency): a silence gap longer
    than ~1.2s is treated as a likely speaker change and we cycle through the
    requested number of characters. The customer's chosen voice stays
    character #1 so the result always matches what they picked."""
    if num_speakers <= 1 or not segments:
        for seg in segments:
            seg["voice"] = base_voice
        return

    prefix = (base_voice.split("-")[0] or "en").lower()
    pool = VOICE_POOLS.get(prefix)
    if not pool:
        # Unknown language: alternate between the chosen voice and a neutral
        # counterpart so characters still sound different.
        pool = [base_voice, "en-US-GuyNeural" if "Female" not in base_voice else "en-US-JennyNeural"]
    # Put the customer's chosen voice first so character #1 matches the UI.
    ordered = [base_voice] + [v for v in pool if v != base_voice]

    speakers: list[str] = []
    while len(speakers) < min(num_speakers, max(len(ordered), num_speakers)):
        nxt = ordered[len(speakers) % len(ordered)]
        if nxt in speakers:
            # Pool exhausted; synthesize numbered variants of the base locale.
            locale = "-".join(base_voice.split("-")[:2]) or "en-US"
            nxt = f"{locale}-Char{len(speakers) + 1}Neural"
        speakers.append(nxt)

    current = 0
    last_end = segments[0].get("start", 0)
    for seg in segments:
        gap = float(seg.get("start", 0)) - last_end
        if gap > 1.2 and len(speakers) > 0:
            current = (current + 1) % len(speakers)
        seg["voice"] = speakers[current]
        last_end = float(seg.get("end", last_end))

    logger.write(f"character voices assigned: {len(speakers)} distinct speakers")


def merge_video(video_path: str, tts_dir: str, segments: list[dict], out_path: str, logger: JobLogger) -> str:
    """Position each TTS clip at its segment start time and mux onto the video."""
    import ffmpeg

    # build filter chain: each clip delayed by (start*1000) ms, all mixed
    inputs = []
    filters = []
    for i, seg in enumerate(segments):
        clip = os.path.join(tts_dir, f"{i:04d}.mp3")
        if not os.path.exists(clip):
            continue
        inputs.append(ffmpeg.input(clip))
        # 'adelay' delays in ms; clips are finite so 'amix' ends at the last one
        filters.append(
            ffmpeg.filter(inputs[-1], "adelay", delays=f"{int(seg['start']*1000)}|{int(seg['start']*1000)}")
            .filter("volume", 1.0)
        )
    if not filters:
        raise RuntimeError("No TTS audio produced - check translation step.")

    mixed = filters[0]
    for f in filters[1:]:
        # Each chain step mixes exactly two streams (amix input count = 2).
        mixed = ffmpeg.filter([mixed, f], "amix", inputs=2, duration="longest")

    video = ffmpeg.input(video_path)
    final = ffmpeg.output(
        video.video, mixed,
        out_path,
        acodec="aac", strict="experimental",
        **{"c:v": "copy"},  # keep original video stream untouched (fast)
    )
    ffmpeg.run(final, overwrite_output=True, quiet=False)
    logger.write(f"final video rendered: {out_path}")
    return out_path


def probe_duration_seconds(path: str) -> int:
    """Exact media duration via ffprobe (the same binary ffmpeg uses)."""
    out = shell(f'ffprobe -v error -show_entries format=duration -of csv=p=0 "{path}"')
    try:
        return int(math.ceil(float(out.strip())))
    except Exception:
        return 0


def process_dubbing(job: dict, workdir: str, logger: JobLogger) -> dict:
    logger.write(f"dubbing job: {job.get('video_url')}")
    logger.write(f"voice: {job.get('voice')} target: {job.get('target_language')}")

    src = download_video(job["video_url"], workdir)
    logger.write(f"downloaded source ({os.path.getsize(src) // 1024} KB)")

    # Billing verification: the customer declares video_seconds at creation and
    # it is locked once paid. If the ACTUAL video is meaningfully longer, the
    # job was underpaid - refuse to process (free work). Allow a small
    # tolerance so rounding/streaming variance doesn't nuke honest jobs.
    claimed = job.get("video_seconds") or 0
    if claimed > 0:
        actual = probe_duration_seconds(src)
        if actual <= 0:
            raise RuntimeError(
                "Could not verify the video duration (ffprobe failed) - refusing to process. "
                "Re-run the engine and try again; contact support if this persists."
            )
        allowance = max(5, int(0.15 * claimed))
        if actual > claimed + allowance:
            raise RuntimeError(
                f"Billing check failed: you paid for {claimed}s but the video is {actual}s. "
                f"Re-create the job with the correct length (over {claimed + allowance}s longer "
                f"is refused). Contact support if you already paid and need a refund."
            )
        logger.write(f"duration verified: {actual}s (declared {claimed}s, tolerance {allowance}s)")

    segments = transcribe(src, logger)
    if not segments:
        raise RuntimeError("Whisper produced no segments - is the video audio-only?")
    logger.write(f"found {len(segments)} speech segments")

    segments = translate_segments(segments, job.get("source_language"), job.get("target_language"), logger)

    voice = job.get("voice") or "bn-BD-NusratNeural"
    # Multi-character dubbing: jobs.num_speakers (or options.num_speakers)
    # gives every detected character a distinct neural voice.
    opts = job.get("options") if isinstance(job.get("options"), dict) else {}
    num_speakers = int(job.get("num_speakers") or opts.get("num_speakers") or 1)
    assign_character_voices(segments, voice, max(1, min(50, num_speakers)), logger)
    tts_dir = synthesize_voiceover(segments, voice, workdir, logger)

    final = os.path.join(workdir, "output.mp4")
    merge_video(src, tts_dir, segments, final, logger)
    logger.set_progress(85)
    return {"final_path": final}


# ---------------------------------------------------------------------------
# 6. ANIMATION PIPELINE (AnimateDiff via diffusers - optional heavy install)
# ---------------------------------------------------------------------------
def process_animation(job: dict, workdir: str, logger: JobLogger) -> dict:
    logger.write("animation job: loading AnimateDiff pipeline (first run installs ~4GB)…")
    logger.write("run: pip install -q diffusers accelerate transformers xformers opencv-python")
    shell("pip install -q diffusers accelerate transformers xformers opencv-python")
    import torch
    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler

    prompt = job.get("text_prompt") or "cinematic 3D animation, slow zoom"

    logger.write("loading motion adapter…")
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
    pipe = AnimateDiffPipeline.from_pretrained(
        "guoyww/animatediff-txt2img-v1-5",
        motion_adapter=adapter,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DDIMScheduler.from_pretrained(
        "guoyww/animatediff-txt2img-v1-5",
        subfolder="scheduler",
        beta_schedule="linear",
        clip_sample=False,
        timestep_spacing="linspace",
        steps_offset=1,
    )
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()  # works well on the T4

    logger.set_progress(40)
    # Advanced options (jobs.options): steps, guidance and an optional seed so
    # customers can tune quality / reproducibility from the Studio.
    opts = job.get("options") if isinstance(job.get("options"), dict) else {}
    try:
        steps = max(10, min(50, int(opts.get("steps") or 20)))
    except (TypeError, ValueError):
        steps = 20
    try:
        guidance = max(1.0, min(21.0, float(opts.get("guidance") or 7.5)))
    except (TypeError, ValueError):
        guidance = 7.5
    seed = opts.get("seed")
    generator = None
    if seed not in (None, ""):
        try:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        except (TypeError, ValueError):
            generator = None
    logger.write(f"animation settings: steps={steps} guidance={guidance} seed={seed or 'random'}")

    # Duration-driven: the customer paid per minute of OUTPUT (video_seconds).
    # AnimateDiff generates 16-frame (2s @ 8fps) chunks; chain them to reach
    # the requested length. Capped at 5 minutes to keep runtimes sane.
    target_seconds = min(max(int(job.get("video_seconds") or 60), 1), 300)
    chunk_count = max(1, int((target_seconds + 1) / 2))
    logger.write(f"generating ~{target_seconds}s animation ({chunk_count} x 2s chunks)…")
    all_frames: list = []
    for chunk in range(chunk_count):
        frames = pipe(
            prompt,
            num_frames=16,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator,
        ).frames[0]
        all_frames.extend(frames)
        logger.write(f"chunk {chunk + 1}/{chunk_count} done ({len(all_frames) * 8}s so far)")
    all_frames = all_frames[: max(1, target_seconds * 8)]

    logger.write("rendering frames to video (FFmpeg)…")
    import cv2  # noqa: F401  (imported by opencv-python)

    frames_dir = os.path.join(workdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i, f in enumerate(all_frames):
        f.save(os.path.join(frames_dir, f"{i:04d}.png"))
    out = os.path.join(workdir, "output.mp4")
    shell(
        f"ffmpeg -y -framerate 8 -i {frames_dir}/%04d.png -c:v libx264 "
        f"-pix_fmt yuv420p {out}"
    )
    logger.set_progress(85)
    return {"final_path": out}


# ---------------------------------------------------------------------------
# 7. OUTPUT DELIVERY (owner's Cloudflare R2 -> public download link)
# ---------------------------------------------------------------------------
def save_to_r2(local_path: str, job: dict, logger: JobLogger) -> tuple[str, str, int]:
    """Upload the finished video STRAIGHT to the owner's Cloudflare R2 bucket
    via a short-lived presigned URL minted by the r2-upload-url Edge Function,
    and return (public_url, expires_at, size_bytes). The file never touches
    Supabase storage, the customer's Drive, or the engine's own disks for
    delivery. Anyone with the public link can download until auto-delete."""
    r = _req(
        "POST",
        "functions/v1/r2-upload-url",
        json={"job_id": job["id"]},
        headers={"Authorization": f"Bearer {_current_token()}"},
        timeout=90,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"R2 upload URL failed (HTTP {r.status_code}): "
            f"{(r.json() or {}).get('error', 'unknown error')}"
        )
    data = r.json() or {}
    upload_url = data.get("upload_url")
    public_url = data.get("public_url")
    expires_at = data.get("expires_at")
    if not upload_url or not public_url or not expires_at:
        raise RuntimeError("R2 upload URL returned incomplete data")

    size = os.path.getsize(local_path)
    logger.write("uploading to cloud storage…")
    with open(local_path, "rb") as fh:
        blob = fh.read()
    put = requests.put(
        upload_url,
        data=blob,
        headers={"Content-Type": "video/mp4"},
        timeout=1800,
    )
    if put.status_code not in (200, 201):
        raise RuntimeError(
            f"cloud upload failed (HTTP {put.status_code}): {put.text[:300]}"
        )
    logger.write(f"uploaded (expires {expires_at}): {public_url}")
    return public_url, expires_at, size


# ---------------------------------------------------------------------------
# 8. AUTO-PUBLISHING (YouTube Data API v3 + Facebook Graph API)
# ---------------------------------------------------------------------------
def publish_to_youtube(local_path: str, creds: dict, job: dict, logger: JobLogger) -> bool:
    """Upload the video to the user's YouTube channel using OAuth refresh token."""
    if not creds.get("youtube_refresh_token"):
        logger.write("youtube: no refresh token stored - skipping publish")
        return False

    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    creds_obj = Credentials(
        token=None,
        refresh_token=creds["youtube_refresh_token"],
        client_id=creds.get("youtube_client_id"),
        client_secret=creds.get("youtube_client_secret"),
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    import google.auth.transport.requests as greq

    creds_obj.refresh(greq.Request())

    yt = build("youtube", "v3", credentials=creds_obj)
    tags = [t.strip() for t in (creds.get("default_tags") or "").split(",") if t.strip()]

    body = {
        "snippet": {
            "title": creds.get("default_title") or f"AI {job['type']} video {job['id'][:6]}",
            "description": creds.get("default_description") or "Created with Vidora AI",
            "tags": tags,
            "categoryId": "22",  # People & Blogs
            "defaultLanguage": "en",
            "defaultAudioLanguage": "bn",
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": bool(creds.get("made_for_kids")),
            "publishAt": None,
            "madeForKids": False,
        },
    }

    media = MediaFileUpload(local_path, mimetype="video/mp4", resumable=True)
    request = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            logger.set_progress(92 + int(8 * status.progress() / 100))
    logger.write(f"[PUBLISH] youtube ok #https://youtu.be/{response['id']}")
    return True


def publish_to_facebook(local_path: str, creds: dict, logger: JobLogger) -> bool:
    """Post the video to the user's Facebook Page (Reels-ready) via Graph API."""
    page_id = creds.get("fb_page_id")
    token = creds.get("fb_page_access_token")
    if not page_id or not token:
        logger.write("facebook: no page token stored - skipping publish")
        return False

    desc = (creds.get("default_description") or "").strip()[:2000]
    url = f"https://graph.facebook.com/v21.0/{page_id}/videos"
    with open(local_path, "rb") as fh:
        r = requests.post(
            url,
            data={
                "access_token": token,
                "description": desc,
                "title": (creds.get("default_title") or "AI video by Vidora")[:255],
            },
            files={"source": fh},
            timeout=600,
        )
    body = r.json()
    if "id" not in body:
        logger.write(f"[PUBLISH] facebook FAILED: {body}")
        return False
    logger.write(f"[PUBLISH] facebook ok #post {body['id']}")
    return True


def auto_publish(local_path: str, job: dict, logger: JobLogger) -> None:
    """Read the user's publish settings and push to YouTube/Facebook."""
    user_id = job["user_id"]
    rows = supabase_get("user_social_credentials", {"user_id": f"eq.{user_id}"})
    if not rows:
        logger.write("publishing: no credentials saved - skipping")
        return
    creds = rows[0]
    if creds.get("auto_publish_youtube"):
        try:
            publish_to_youtube(local_path, creds, job, logger)
        except Exception as e:
            logger.write(f"[PUBLISH] youtube error: {e}")
    if creds.get("auto_publish_facebook"):
        try:
            publish_to_facebook(local_path, creds, logger)
        except Exception as e:
            logger.write(f"[PUBLISH] facebook error: {e}")


# ---------------------------------------------------------------------------
# 9. MAIN LOOP
# ---------------------------------------------------------------------------
def _job_watchdog(job_id: str, user_id: str, stop_event: threading.Event):
    """Touch started_at every 30s so a crashed/frozen engine is detected and
    the job is requeued by requeue_stale_jobs instead of being stuck."""
    while not stop_event.is_set():
        time.sleep(30)
        try:
            supabase_patch(
                "jobs", job_id,
                {"started_at": now_iso()},
                extra_cond={"user_id": user_id},
            )
        except Exception:
            pass


def process_one(user_id: str, job: dict) -> None:
    logger = JobLogger(job["id"], job["user_id"])
    workdir = os.path.join(
        os.environ.get("VIDORA_WORKDIR") or tempfile.gettempdir(), f"vidora_work/{job['id']}"
    )
    os.makedirs(workdir, exist_ok=True)
    watchdog_stop = threading.Event()
    watchdog = threading.Thread(
        target=_job_watchdog, args=(job["id"], user_id, watchdog_stop), daemon=True
    )
    watchdog.start()
    try:
        logger.write(f"job claimed: {job['type']} ({job['id']})")

        result = (
            process_dubbing(job, workdir, logger)
            if job["type"] == "dubbing"
            else process_animation(job, workdir, logger)
        )
        final_path = result["final_path"]

        # Deliver the finished video straight to the owner's Cloudflare R2
        # bucket as a public download link with an auto-expiry (retention days
        # from admin). No Drive, no customer consent, no Supabase storage.
        output_url, output_expires_at, output_size = save_to_r2(final_path, job, logger)

        # auto-publish is best-effort: never let it fail an otherwise-completed
        # job (a transient credential/network error must not lose the output).
        try:
            auto_publish(final_path, job, logger)
        except Exception as e:
            logger.write(f"[warn] auto-publish failed ({e}) - continuing")

        logger.set_progress(100)
        logger.finish(
            completed=True,
            extra={
                "output_file_url": output_url,
                "output_drive_link": None,
                "output_expires_at": output_expires_at,
                "output_size_bytes": output_size,
            },
        )
        logger.write(f"job {job['id']} DONE")
    except Exception as e:
        logger.write(f"FAILED: {e}")
        logger.write(traceback.format_exc()[-1500:])
        logger.finish(completed=False)
    finally:
        watchdog_stop.set()
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    install_packages()
    print(f"[engine] connected to {SUPABASE_URL}", flush=True)

    # The engine only ever serves the signed-in user's own jobs.
    user_id = fetch_user_id()
    print(f"[engine] serving YOUR jobs (user {user_id[:8]}…)", flush=True)
    print("[engine] using your access token - RLS keeps every read/write scoped to you", flush=True)

    stop_heartbeat = threading.Event()
    threading.Thread(target=heartbeat_loop, args=(user_id, stop_heartbeat), daemon=True).start()
    print(f"[engine] heartbeat started", flush=True)

    print(f"[engine] polling every {POLL_INTERVAL}s… (Ctrl+C to stop)", flush=True)
    poll_backoff = 1
    while True:
        try:
            requeue_stale_jobs()

            # Processing gate is the job's OWN paid flag (see next_job). No
            # global license check: per-video customers pay per job via bKash.
            job = next_job(user_id)
            if job:
                process_one(user_id, job)
            poll_backoff = 1  # healthy poll - reset
        except Exception as e:
            print(f"[engine] poll error: {e}", flush=True)
            # Exponential backoff so a persistent outage (bad token, Supabase
            # down) doesn't hammer the API every few seconds.
            time.sleep(min(poll_backoff, MAX_POLL_BACKOFF))
            poll_backoff = min(poll_backoff * 2, MAX_POLL_BACKOFF)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()