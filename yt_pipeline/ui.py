"""FastAPI UI for viewing video stages and ranked reel reviews."""

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from yt_pipeline.config import Settings, get_settings
from yt_pipeline.database import VideoRepository
from yt_pipeline.models import HumanReviewUpdateDTO, VideoSummaryDTO
from yt_pipeline.stages.ai_review import AIReviewService, RankingService


def create_app(repo: VideoRepository, settings: Settings | None = None) -> FastAPI:
    """Create a FastAPI app bound to a video repository."""

    settings = settings or get_settings()
    app = FastAPI(title="YouTube Pipeline")

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        """Render the stage dashboard shell."""

        return _html()

    @app.get("/api/videos", response_model=list[VideoSummaryDTO])
    def videos() -> list[VideoSummaryDTO]:
        """Return recent videos and their current stage state."""

        return [
            VideoSummaryDTO(
                video_id=video.video_id,
                title=video.title,
                status=video.status,
                updated_at=video.updated_at,
                stages=video.stages,
            )
            for video in repo.list_recent()
        ]

    @app.get("/videos/{video_id}/reels", response_class=HTMLResponse)
    def video_reels_page(video_id: str) -> str:
        """Render the ranked reels page for one source video."""

        return _video_reels_html(video_id)

    @app.get("/reels/{reel_id}", response_class=HTMLResponse)
    def reel_page(reel_id: str) -> str:
        """Render a detailed review page for one reel."""

        return _reel_html(reel_id)

    @app.get("/api/videos/{video_id}/reels")
    def video_reels(
        video_id: str,
        recommendation: str | None = None,
        humanStatus: str | None = None,
        aiStatus: str | None = None,
        minimumScore: float | None = None,
        sort: str = "rank",
        order: str = "asc",
    ) -> list[dict[str, Any]]:
        """Return filtered and sorted reel metadata for one video."""

        try:
            reels = repo.list_reels(video_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return sort_reels(filter_reels(reels, recommendation, humanStatus, aiStatus, minimumScore), sort, order)

    @app.get("/api/reels/{reel_id}")
    def reel_detail(reel_id: str) -> dict[str, Any]:
        """Return one reel with its source video id."""

        try:
            video, reel = repo.find_reel(reel_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"videoId": video.video_id, **reel}

    @app.patch("/api/reels/{reel_id}/human-review")
    def human_review(reel_id: str, update: HumanReviewUpdateDTO) -> dict[str, Any]:
        """Persist a human review decision."""

        try:
            return repo.update_human_review(reel_id, update)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/reels/{reel_id}/ai-review")
    async def review_reel(reel_id: str, force: bool = Query(default=False)) -> dict[str, Any]:
        """Retry or run AI review for a single reel."""

        try:
            video, _reel = repo.find_reel(reel_id)
            result = await AIReviewService(repo, settings).review_reel(video.video_id, reel_id, force=force)
            return result.model_dump(by_alias=True, mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/reels/{reel_id}/review-assets")
    def build_review_assets(reel_id: str, force: bool = Query(default=False)) -> dict[str, Any]:
        """Regenerate review proxy and representative frames for one reel."""

        try:
            video, reel = repo.find_reel(reel_id)
            assets = AIReviewService(repo, settings).asset_builder.build(video.video_id, reel, force=force)
            return repo.update_reel(video.video_id, reel_id, {"reviewAssets": assets})
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/videos/{video_id}/ai-review")
    async def review_video(video_id: str, force: bool = Query(default=False), limit: int | None = None) -> dict[str, Any]:
        """Run AI review for pending reels on one source video."""

        return await AIReviewService(repo, settings).review_video(video_id, force=force, limit=limit)

    @app.post("/api/videos/{video_id}/rank")
    def rank_video(video_id: str) -> dict[str, Any]:
        """Rank reviewed reels for one video."""

        return RankingService(repo, settings).rank_video(video_id)

    @app.get("/api/ai/health")
    async def ai_health() -> dict[str, Any]:
        """Return local Ollama health information."""

        try:
            return await AIReviewService(repo, settings).ollama.health()
        except Exception as exc:
            return {"ok": False, "error": str(exc), "model": settings.ollama_model}

    @app.get("/media/reels/{reel_id}")
    def reel_media(reel_id: str) -> FileResponse:
        """Serve an original generated reel for local dashboard preview."""

        try:
            _video, reel = repo.find_reel(reel_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        path = Path(str(reel.get("path", "")))
        if not path.exists():
            raise HTTPException(status_code=404, detail="Reel file not found.")
        return FileResponse(path)

    return app


def filter_reels(
    reels: list[dict[str, Any]],
    recommendation: str | None,
    human_status: str | None,
    ai_status: str | None,
    minimum_score: float | None,
) -> list[dict[str, Any]]:
    """Apply API filters to reel metadata."""

    filtered = []
    for reel in reels:
        ai_review = reel.get("aiReview") or {}
        human_review = reel.get("humanReview") or {}
        ranking = reel.get("ranking") or {}
        if recommendation and ai_review.get("recommendation") != recommendation:
            continue
        if human_status and human_review.get("status") != human_status:
            continue
        if ai_status and ai_review.get("status") != ai_status:
            continue
        if minimum_score is not None and float(ranking.get("adjustedScore", 0.0) or 0.0) < minimum_score:
            continue
        filtered.append(reel)
    return filtered


def sort_reels(reels: list[dict[str, Any]], sort: str, order: str) -> list[dict[str, Any]]:
    """Sort reel metadata for API and dashboard responses."""

    reverse = order.lower() == "desc"
    key_map = {
        "rank": lambda reel: (reel.get("ranking") or {}).get("rank") or 9999,
        "adjustedScore": lambda reel: (reel.get("ranking") or {}).get("adjustedScore") or 0,
        "uploadPotential": lambda reel: ((reel.get("aiReview") or {}).get("scores") or {}).get("uploadPotential") or 0,
        "confidence": lambda reel: (reel.get("aiReview") or {}).get("confidence") or 0,
        "start": lambda reel: reel.get("start") or 0,
        "duration": lambda reel: reel.get("duration") or 0,
    }
    return sorted(reels, key=key_map.get(sort, key_map["rank"]), reverse=reverse)


def _html() -> str:
    """Return the dashboard HTML with small embedded styles and JavaScript."""

    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YouTube Pipeline</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f6f7f9; color: #17202a; }
    header { border-bottom: 1px solid #d9dde3; background: #ffffff; padding: 18px 24px; }
    h1 { font-size: 22px; margin: 0; font-weight: 700; letter-spacing: 0; }
    main { max-width: 1180px; margin: 0 auto; padding: 24px; }
    table { width: 100%; border-collapse: collapse; background: #ffffff; border: 1px solid #d9dde3; }
    th, td { padding: 12px 14px; border-bottom: 1px solid #e8ebef; text-align: left; vertical-align: top; }
    th { font-size: 12px; text-transform: uppercase; color: #5b6573; background: #fbfcfd; }
    .title { font-weight: 650; max-width: 320px; }
    .status { display: inline-block; min-width: 124px; padding: 4px 8px; border-radius: 6px; background: #eef2ff; color: #263b80; font-size: 12px; font-weight: 700; text-align: center; }
    .stages { display: flex; flex-wrap: wrap; gap: 6px; }
    .stage { border: 1px solid #ccd3dc; border-radius: 6px; padding: 4px 8px; font-size: 12px; color: #516071; background: #ffffff; }
    .done { border-color: #93c5a4; background: #edf8f0; color: #166534; }
    .failed { border-color: #f0aaa4; background: #fff0ef; color: #a22b24; }
    .empty { padding: 40px; text-align: center; color: #657184; background: #ffffff; border: 1px solid #d9dde3; }
    @media (max-width: 760px) {
      main { padding: 16px; }
      table, thead, tbody, tr, th, td { display: block; }
      thead { display: none; }
      tr { border-bottom: 1px solid #d9dde3; }
      td { border-bottom: 0; }
    }
  </style>
</head>
<body>
  <header><h1>YouTube Pipeline</h1></header>
  <main id="app"><div class="empty">Loading stages...</div></main>
  <script>
    const labels = { download: "Download", transcription: "Transcript", reels: "Reels", aiReview: "AI Review", instagram: "Upload" };
    function escapeHtml(value) {
      return String(value || "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      })[char]);
    }
    function stageClass(stage) {
      if (stage.error) return "stage failed";
      return stage.completed ? "stage done" : "stage";
    }
    function render(videos) {
      if (!videos.length) {
        document.getElementById("app").innerHTML = '<div class="empty">No videos have been discovered yet.</div>';
        return;
      }
      const rows = videos.map(video => `
        <tr>
          <td class="title">${escapeHtml(video.title)}</td>
          <td><span class="status">${escapeHtml(video.status)}</span></td>
          <td><div class="stages">${Object.entries(labels).map(([key, label]) => {
            const stage = video.stages[key] || {};
            return `<span class="${stageClass(stage)}" title="${escapeHtml(stage.error)}">${label}</span>`;
          }).join("")}</div></td>
          <td>${new Date(video.updated_at).toLocaleString()}</td>
        </tr>`).join("");
      document.getElementById("app").innerHTML = `<table>
        <thead><tr><th>Video</th><th>Status</th><th>Stages</th><th>Updated</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
    }
    async function load() {
      const response = await fetch("/api/videos");
      render(await response.json());
    }
    load();
    setInterval(load, 15000);
  </script>
</body>
</html>
"""


def _video_reels_html(video_id: str) -> str:
    """Return a ranked reels dashboard page."""

    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ranked Reels</title>
  <style>
    body {{ margin: 0; background: #f6f7f9; color: #17202a; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ border-bottom: 1px solid #d9dde3; background: #fff; padding: 16px 24px; display: flex; justify-content: space-between; gap: 12px; align-items: center; }}
    h1 {{ font-size: 20px; margin: 0; letter-spacing: 0; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 20px; }}
    .toolbar {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }}
    select, button {{ border: 1px solid #cdd4df; background: #fff; border-radius: 6px; padding: 8px 10px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d9dde3; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #e8ebef; text-align: left; vertical-align: top; font-size: 14px; }}
    th {{ color: #5b6573; font-size: 12px; text-transform: uppercase; background: #fbfcfd; }}
    .label {{ display: inline-block; min-width: 64px; padding: 3px 6px; border-radius: 6px; border: 1px solid #cdd4df; font-size: 12px; text-align: center; font-weight: 700; }}
    .UPLOAD {{ background: #eaf7ee; color: #166534; border-color: #9bd0aa; }}
    .REVIEW {{ background: #fff7e6; color: #8a5200; border-color: #efc56c; }}
    .SKIP {{ background: #f2f3f5; color: #4b5563; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
  </style>
</head>
<body>
  <header><h1>Ranked Reels</h1><a href="/">Videos</a></header>
  <main>
    <div class="toolbar">
      <select id="filter">
        <option value="">All</option><option value="UPLOAD">Upload</option><option value="REVIEW">Review</option><option value="SKIP">Skip</option>
      </select>
      <select id="sort">
        <option value="rank">Rank</option><option value="adjustedScore">Adjusted score</option><option value="uploadPotential">Upload potential</option><option value="confidence">Confidence</option><option value="start">Start time</option><option value="duration">Duration</option>
      </select>
      <button onclick="reviewVideo()">Run AI review</button>
      <button onclick="rankVideo()">Rank</button>
    </div>
    <div id="app">Loading...</div>
  </main>
  <script>
    const videoId = {json.dumps(video_id)};
    function esc(value) {{ return String(value ?? "").replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}})[c]); }}
    async function load() {{
      const recommendation = document.getElementById("filter").value;
      const sort = document.getElementById("sort").value;
      const url = `/api/videos/${{encodeURIComponent(videoId)}}/reels?sort=${{sort}}${{recommendation ? `&recommendation=${{recommendation}}` : ""}}`;
      const reels = await (await fetch(url)).json();
      document.getElementById("app").innerHTML = `<table><thead><tr><th>Rank</th><th>Reel</th><th>AI</th><th>Score</th><th>Reason</th><th>Human</th></tr></thead><tbody>${{reels.map(row).join("")}}</tbody></table>`;
    }}
    function row(reel) {{
      const ai = reel.aiReview || {{}};
      const ranking = reel.ranking || {{}};
      const human = reel.humanReview || {{}};
      return `<tr><td>${{esc(ranking.rank || "")}}</td><td><a href="/reels/${{encodeURIComponent(reel.id)}}">${{esc(reel.id)}}</a><br>${{esc(reel.start)}}s-${{esc(reel.end)}}s</td><td><span class="label ${{esc(ai.recommendation)}}">${{esc(ai.recommendation || ai.status || "PENDING")}}</span><br>${{esc(ai.detectedMoment || "")}}</td><td>${{esc(ranking.adjustedScore || "")}}<br>conf ${{esc(ai.confidence || "")}}</td><td>${{esc(ai.reason || "")}}</td><td>${{esc(human.status || "PENDING")}}</td></tr>`;
    }}
    async function reviewVideo() {{ await fetch(`/api/videos/${{encodeURIComponent(videoId)}}/ai-review`, {{method: "POST"}}); await load(); }}
    async function rankVideo() {{ await fetch(`/api/videos/${{encodeURIComponent(videoId)}}/rank`, {{method: "POST"}}); await load(); }}
    document.getElementById("filter").addEventListener("change", load);
    document.getElementById("sort").addEventListener("change", load);
    load();
  </script>
</body>
</html>
"""


def _reel_html(reel_id: str) -> str:
    """Return a detailed reel review page."""

    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reel Review</title>
  <style>
    body {{ margin: 0; background: #f6f7f9; color: #17202a; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ border-bottom: 1px solid #d9dde3; background: #fff; padding: 16px 24px; }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 20px; display: grid; grid-template-columns: minmax(260px, 360px) 1fr; gap: 20px; }}
    video {{ width: 100%; background: #000; max-height: 80vh; }}
    section {{ background: #fff; border: 1px solid #d9dde3; padding: 14px; margin-bottom: 12px; }}
    button, input, textarea {{ border: 1px solid #cdd4df; border-radius: 6px; padding: 8px; width: 100%; box-sizing: border-box; margin-top: 6px; }}
    button {{ background: #fff; cursor: pointer; }}
    .actions {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <header><h1>Reel Review</h1></header>
  <main>
    <div><video id="player" controls></video></div>
    <div id="app">Loading...</div>
  </main>
  <script>
    const reelId = {json.dumps(reel_id)};
    function esc(value) {{ return String(value ?? "").replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}})[c]); }}
    async function load() {{
      const reel = await (await fetch(`/api/reels/${{encodeURIComponent(reelId)}}`)).json();
      document.getElementById("player").src = `/media/reels/${{encodeURIComponent(reelId)}}`;
      const ai = reel.aiReview || {{}};
      const transcript = reel.transcriptReview || {{}};
      document.getElementById("app").innerHTML = `
        <section><h2>${{esc(reel.id)}}</h2><p>${{esc(reel.start)}}s-${{esc(reel.end)}}s · ${{esc(reel.duration)}}s</p></section>
        <section><h3>AI Review</h3><pre>${{esc(JSON.stringify(ai, null, 2))}}</pre></section>
        <section><h3>Transcript</h3><p>${{esc(transcript.contextBefore)}}</p><strong>${{esc(transcript.clipText || "(no speech)")}}</strong><p>${{esc(transcript.contextAfter)}}</p></section>
        <section><h3>Human Review</h3><div class="actions"><button onclick="human('APPROVED')">Approve</button><button onclick="human('REJECTED')">Reject</button><button onclick="human('NEEDS_EDIT')">Needs Edit</button></div><textarea id="notes" placeholder="Notes"></textarea></section>
        <section><button onclick="retry()">Retry AI Review</button><button onclick="assets()">Regenerate Review Assets</button></section>`;
    }}
    async function human(status) {{ await fetch(`/api/reels/${{encodeURIComponent(reelId)}}/human-review`, {{method:"PATCH", headers:{{"content-type":"application/json"}}, body:JSON.stringify({{status, notes:document.getElementById("notes").value}})}}); await load(); }}
    async function retry() {{ await fetch(`/api/reels/${{encodeURIComponent(reelId)}}/ai-review?force=true`, {{method:"POST"}}); await load(); }}
    async function assets() {{ await fetch(`/api/reels/${{encodeURIComponent(reelId)}}/review-assets?force=true`, {{method:"POST"}}); await load(); }}
    load();
  </script>
</body>
</html>
"""
