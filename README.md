# YouTube Pipeline

A small MongoDB-backed pipeline for discovering YouTube videos, downloading new ones,
tracking processing stages, and viewing stage state in a browser.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

Set the channel URL and MongoDB details:

```bash
export YT_CHANNEL_URL="https://www.youtube.com/@channel/videos"
export MONGO_URI="mongodb://localhost:27017"
export MONGO_DB="yt_pipeline"
```

Transcripts are written to `transcripts/<videoId>.json` and
`transcripts/<videoId>.srt`.

Downloads are split by purpose: highest-quality merged video files go to
`downloads/video/`, and transcription audio goes to `downloads/audio/`.

Vertical reel candidates are written to `reels/<videoId>/`. By default the
pipeline creates up to 20 centered 1080x1920 clips at 30 seconds each:

```bash
export REEL_CLIP_SECONDS=30
export REEL_MAX_CLIPS=20
```

## Commands

Run one discovery/download pass:

```bash
yt-pipeline run-once
```

Run the scheduler:

```bash
yt-pipeline scheduler
```

Run the UI:

```bash
yt-pipeline ui
```

Then open `http://127.0.0.1:8000`.

## Local AI Review

AI review runs locally through Ollama. The model output is advisory only:
manual approval remains mandatory and no clip is automatically uploaded.

Install and start Ollama:

```bash
brew install ollama
ollama serve
```

Download and verify the default local model:

```bash
ollama pull qwen3-vl:8b
ollama run qwen3-vl:8b
```

Configure AI review:

```bash
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="qwen3-vl:8b"
export OLLAMA_TIMEOUT_SECONDS=180
export OLLAMA_MAX_RETRIES=2
export AI_REVIEW_ENABLED=true
```

Review and rank generated reels:

```bash
yt-pipeline review-video <videoId>
yt-pipeline review-video <videoId> --force --limit 5
yt-pipeline review-reel <reelId> --force
yt-pipeline rank-video <videoId>
yt-pipeline retry-ai-review <reelId>
yt-pipeline build-review-assets <reelId> --force
yt-pipeline ai-health
```

Review proxies and frames are stored under `review-assets/`. Original
1080x1920 reel files remain unchanged and are still the upload candidates.
The review model receives representative frames, transcript windows, and
activity metrics through the local Ollama instance.
