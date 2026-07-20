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
