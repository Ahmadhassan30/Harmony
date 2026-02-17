"""Tests for BPM detection ensemble."""

import numpy as np
import pytest


class TestBPMEnsemble:
    """Test BPM ensemble detection."""

    def test_click_track_120bpm(self, sample_click_track):
        """A perfect 120 BPM click track should be detected accurately."""
        from app.engines.bpm.ensemble import BPMEnsemble

        audio, sr = sample_click_track
        features = self._extract_features(audio, sr)
        ensemble = BPMEnsemble()
        result = ensemble.detect(audio, sr, features)

        # Should be within ±2 BPM of 120
        assert abs(result.bpm - 120.0) <= 2.0, f"Expected ~120 BPM, got {result.bpm}"
        assert result.confidence > 0.5

    def test_empty_audio_returns_fallback(self):
        """Empty audio should return a fallback BPM with zero confidence."""
        from app.engines.bpm.ensemble import BPMEnsemble

        audio = np.zeros(22050, dtype=np.float32)
        sr = 22050
        features = self._extract_features(audio, sr)
        ensemble = BPMEnsemble()
        result = ensemble.detect(audio, sr, features)

        assert result.bpm > 0
        assert result.confidence < 0.5

    @staticmethod
    def _extract_features(audio, sr):
        """Helper to extract features for testing."""
        import librosa

        harmonic, percussive = librosa.effects.hpss(audio)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr)
        onset = librosa.onset.onset_strength(y=percussive, sr=sr)

        return {
            "mel_spectrogram": mel_db,
            "chroma_cqt": chroma,
            "onset_envelope": onset,
            "harmonic": harmonic,
            "percussive": percussive,
        }
