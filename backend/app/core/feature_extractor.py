"""Shared feature extraction with caching support."""

from __future__ import annotations

import hashlib
from pathlib import Path

import librosa
import numpy as np
import structlog

from app.config import settings

logger = structlog.get_logger()


class FeatureExtractor:
    """Extract and cache common audio features used across engines."""

    def __init__(self) -> None:
        self._cache_dir = settings.cache_dir / "features"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def extract_all(self, audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
        """Extract all shared features in a single pass.

        Returns a dict with:
        - mel_spectrogram: (n_mels, T) mel-scaled spectrogram
        - chroma_cqt: (12, T) constant-Q chromagram
        - onset_envelope: (T,) onset strength envelope
        - harmonic: (N,) harmonic component
        - percussive: (N,) percussive component
        """
        logger.debug("features.extracting", sr=sr, samples=len(audio))

        # HPSS for harmonic/percussive separation
        harmonic, percussive = librosa.effects.hpss(audio)

        # Mel spectrogram (for CNN/EfficientNet models)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=128, fmax=8000, hop_length=512
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # CQT Chromagram (for key detection)
        chroma = librosa.feature.chroma_cqt(
            y=harmonic, sr=sr, hop_length=512, n_chroma=12
        )

        # Onset envelope (for BPM detection)
        onset_env = librosa.onset.onset_strength(y=percussive, sr=sr, hop_length=512)

        features = {
            "mel_spectrogram": mel_db,
            "chroma_cqt": chroma,
            "onset_envelope": onset_env,
            "harmonic": harmonic,
            "percussive": percussive,
        }

        logger.debug(
            "features.extracted",
            mel_shape=mel_db.shape,
            chroma_shape=chroma.shape,
            onset_len=len(onset_env),
        )

        return features

    @staticmethod
    def file_hash(file_path: str | Path) -> str:
        """Compute MD5 hash of a file for cache keying."""
        h = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
