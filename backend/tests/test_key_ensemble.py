"""Tests for key detection ensemble."""

import numpy as np
import pytest


class TestKeyEnsemble:
    """Test key detection ensemble."""

    def test_c_major_scale_detection(self, sample_audio):
        """A C major scale signal should be detected as C major."""
        from app.engines.key.profiles import chroma_profile_match

        # Create a synthetic C major chroma vector
        # C major: C, D, E, F, G, A, B have energy
        chroma = np.zeros((12, 100), dtype=np.float32)
        c_major_notes = [0, 2, 4, 5, 7, 9, 11]  # C D E F G A B
        for note in c_major_notes:
            chroma[note, :] = 1.0
        chroma[0, :] = 2.0  # Emphasize C (tonic)

        result = chroma_profile_match(chroma)
        assert result["key"] == "C"
        assert result["scale"] == "major"
        assert result["confidence"] > 0.5

    def test_empty_chroma_returns_fallback(self):
        """Empty chroma should return C major with zero confidence."""
        from app.engines.key.profiles import chroma_profile_match

        result = chroma_profile_match(None)
        assert result["confidence"] == 0.0


class TestCamelotWheel:
    """Test Camelot wheel notation."""

    def test_key_to_camelot(self):
        from app.utils.music_theory import key_to_camelot

        assert key_to_camelot("A", "minor") == "8A"
        assert key_to_camelot("C", "major") == "8B"
        assert key_to_camelot("F#", "minor") == "11A"

    def test_compatible_keys(self):
        from app.utils.music_theory import compatible_keys

        compat = compatible_keys("8A")
        assert "8A" in compat  # Same key
        assert "8B" in compat  # Parallel major
