"""Unit tests for MongoDB repository update shapes."""

import unittest

from yt_pipeline.database import VideoRepository


class FakeCollection:
    """Tiny collection fake that captures update_one calls."""

    def __init__(self) -> None:
        """Create an empty fake collection."""

        self.update: dict | None = None

    def update_one(self, filter_: dict, update: dict, **kwargs: object) -> None:
        """Capture the update document sent by the repository."""

        self.update = update


class VideoRepositoryTests(unittest.TestCase):
    """Tests for write documents emitted by the video repository."""

    def test_complete_transcription_replaces_stage_without_subpath_unset(self) -> None:
        """Completed transcription update avoids Mongo path conflicts."""

        collection = FakeCollection()
        repo = VideoRepository(collection)

        repo.complete_transcription(
            "abc",
            transcript_path="transcripts/abc.json",
            subtitle_path="transcripts/abc.srt",
            language="en",
            total_segments=2,
            model="large-v3",
        )

        self.assertNotIn("$unset", collection.update)
        self.assertEqual(collection.update["$set"]["stages.transcription"]["completed"], True)
        self.assertNotIn("error", collection.update["$set"]["stages.transcription"])


if __name__ == "__main__":
    unittest.main()

