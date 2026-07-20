"""Unit tests for local AI review and ranking helpers."""

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from yt_pipeline.config import Settings
from yt_pipeline.models import AIReviewResult
from yt_pipeline.stages.ai_review import (
    RankingService,
    TranscriptWindowExtractor,
    duplicate_analysis,
    overlap_ratio,
    parse_ollama_review_response,
    representative_timestamps,
    weighted_score,
)
from yt_pipeline.ui import filter_reels, sort_reels


def valid_ai_payload(**overrides: object) -> dict:
    """Return a valid AI review payload with optional overrides."""

    payload = {
        "scores": {
            "hook": 8,
            "gameplay": 9,
            "excitement": 8,
            "visualClarity": 7,
            "contextIndependence": 8,
            "payoff": 9,
            "pacing": 8,
            "audioClarity": 7,
            "technicalQuality": 8,
            "uploadPotential": 9,
        },
        "recommendation": "UPLOAD",
        "confidence": 0.9,
        "reason": "Clear fight with payoff.",
        "detectedMoment": "Squad wipe",
        "suggestedTitle": "Wrong push",
        "suggestedCaption": "One push, one squad wipe.",
        "hashtags": ["BGMI", "#GamingReels"],
        "issues": [],
    }
    payload.update(overrides)
    return payload


class AIReviewModelTests(unittest.TestCase):
    """Tests for strict model validation and Ollama response parsing."""

    def test_valid_response_normalizes_hashtags(self) -> None:
        """AI response validation keeps recommendations strict and normalizes tags."""

        result = AIReviewResult.model_validate(valid_ai_payload())

        self.assertEqual(result.recommendation, "UPLOAD")
        self.assertEqual(result.hashtags, ["#BGMI", "#GamingReels"])

    def test_invalid_recommendation_is_rejected(self) -> None:
        """Unknown recommendation strings are rejected."""

        with self.assertRaises(ValidationError):
            AIReviewResult.model_validate(valid_ai_payload(recommendation="MAYBE"))

    def test_score_range_is_enforced(self) -> None:
        """Individual scores must stay in the zero-to-ten range."""

        payload = valid_ai_payload()
        payload["scores"]["hook"] = 12

        with self.assertRaises(ValidationError):
            AIReviewResult.model_validate(payload)

    def test_parse_ollama_structured_response(self) -> None:
        """Ollama chat content is parsed as JSON and validated."""

        payload = {"message": {"content": json.dumps(valid_ai_payload())}}

        self.assertEqual(parse_ollama_review_response(payload).recommendation, "UPLOAD")


class ReviewHelperTests(unittest.TestCase):
    """Tests for transcript, frame, and ranking helper functions."""

    def test_representative_timestamps_cover_clip(self) -> None:
        """Eight default timestamps cover the clip from first to near-final frame."""

        self.assertEqual(representative_timestamps(30, 8), [0.0, 3.0, 7.5, 12.0, 16.5, 21.0, 25.5, 29.4])

    def test_transcript_window_extracts_overlap_and_context(self) -> None:
        """Transcript extractor separates clip text from neighboring context."""

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "abc.json"
            path.write_text(
                json.dumps(
                    {
                        "language": "en",
                        "duration": 20,
                        "segments": [
                            {"start": 6, "end": 7, "text": "before"},
                            {"start": 10, "end": 12, "text": "inside"},
                            {"start": 15, "end": 16, "text": "after"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            window = TranscriptWindowExtractor(Path(temp_dir), context_seconds=5).extract("abc", 10, 14)

            self.assertEqual(window["clipText"], "inside")
            self.assertEqual(window["contextBefore"], "before")
            self.assertEqual(window["contextAfter"], "after")

    def test_empty_transcript_is_allowed(self) -> None:
        """Missing transcript files produce empty no-speech review input."""

        with tempfile.TemporaryDirectory() as temp_dir:
            window = TranscriptWindowExtractor(Path(temp_dir), context_seconds=5).extract("abc", 0, 10)

            self.assertEqual(window["clipText"], "")
            self.assertEqual(window["segments"], [])

    def test_weighted_score_uses_application_weights(self) -> None:
        """Weighted score is deterministic and owned by the application."""

        self.assertAlmostEqual(weighted_score(valid_ai_payload()["scores"]), 8.55)

    def test_overlap_ratio_uses_shorter_clip(self) -> None:
        """Overlap ratio is normalized by the shorter candidate."""

        self.assertAlmostEqual(overlap_ratio(0, 10, 5, 15), 0.5)

    def test_duplicate_analysis_flags_related_reels(self) -> None:
        """Duplicate analysis keeps related reel ids instead of deleting clips."""

        reels = [{"id": "a", "start": 0, "end": 10}, {"id": "b", "start": 2, "end": 11}]

        result = duplicate_analysis(reels[0], reels, threshold=0.6)

        self.assertTrue(result["isPossibleDuplicate"])
        self.assertEqual(result["relatedReelIds"], ["b"])


class FakeRankingRepository:
    """Repository fake used by ranking service tests."""

    def __init__(self, video: object) -> None:
        """Create a fake repository with one video."""

        self.video = video
        self.metadata_patch: dict | None = None
        self.summary: dict | None = None

    def get(self, video_id: str) -> object:
        """Return the fake video."""

        return self.video

    def update_reels_metadata(self, video_id: str, metadata_patch: dict) -> None:
        """Capture updated reel metadata."""

        self.metadata_patch = metadata_patch

    def update_ai_review_stage(self, video_id: str, summary: dict) -> None:
        """Capture video-level AI review summary."""

        self.summary = summary


class RankingTests(unittest.TestCase):
    """Tests for deterministic ranking behavior."""

    def test_ranker_assigns_rank_and_duplicate_penalty(self) -> None:
        """Ranker stores ranks and duplicate warnings deterministically."""

        class Stage:
            """Simple stage object with metadata."""

            metadata = {
                "clips": [
                    reviewed_reel("a", start=0, end=10, upload_potential=9, confidence=0.8),
                    reviewed_reel("b", start=2, end=11, upload_potential=8, confidence=0.9),
                ]
            }

        class Video:
            """Simple video object with reel stage."""

            stages = {"reels": Stage()}

        repo = FakeRankingRepository(Video())

        result = RankingService(repo, Settings()).rank_video("abc")

        self.assertEqual(result["ranked"], 2)
        self.assertEqual(repo.metadata_patch["clips"][0]["ranking"]["rank"], 1)
        self.assertTrue(repo.metadata_patch["clips"][0]["duplicateAnalysis"]["isPossibleDuplicate"])


def reviewed_reel(reel_id: str, *, start: float, end: float, upload_potential: float, confidence: float) -> dict:
    """Return a reviewed reel dictionary for ranking tests."""

    payload = valid_ai_payload(confidence=confidence)
    payload["scores"]["uploadPotential"] = upload_potential
    return {
        "id": reel_id,
        "start": start,
        "end": end,
        "duration": end - start,
        "aiReview": {"status": "COMPLETED", **payload},
        "activityMetrics": {"audio": {"speechCoverage": 0.6}, "visual": {"motionScore": 0.8, "blackFrameRatio": 0, "frozenFrameRatio": 0}},
    }


class ReelApiHelperTests(unittest.TestCase):
    """Tests for API filter and sort helpers."""

    def test_filter_and_sort_reels(self) -> None:
        """API helper functions filter by AI recommendation and sort by score."""

        reels = [
            {"id": "a", "aiReview": {"recommendation": "SKIP"}, "ranking": {"adjustedScore": 2}},
            {"id": "b", "aiReview": {"recommendation": "UPLOAD"}, "ranking": {"adjustedScore": 9}},
        ]

        filtered = filter_reels(reels, "UPLOAD", None, None, None)
        sorted_reels = sort_reels(filtered, "adjustedScore", "desc")

        self.assertEqual([reel["id"] for reel in sorted_reels], ["b"])


if __name__ == "__main__":
    unittest.main()
