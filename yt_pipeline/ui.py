"""FastAPI UI for viewing video stages."""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from yt_pipeline.database import VideoRepository
from yt_pipeline.models import VideoSummaryDTO


def create_app(repo: VideoRepository) -> FastAPI:
    """Create a FastAPI app bound to a video repository."""

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

    return app


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
