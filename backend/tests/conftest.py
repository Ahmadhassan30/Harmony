"""Shared pytest fixtures for Harmony backend tests."""

import numpy as np
import pytest


@pytest.fixture
def sample_audio() -> tuple[np.ndarray, int]:
    """Generate a synthetic test audio signal (440Hz sine wave, 5 seconds)."""
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 440 Hz sine wave with some harmonics
    audio = (
        0.5 * np.sin(2 * np.pi * 440 * t)
        + 0.3 * np.sin(2 * np.pi * 880 * t)
        + 0.1 * np.sin(2 * np.pi * 1320 * t)
    ).astype(np.float32)
    return audio, sr


@pytest.fixture
def sample_click_track() -> tuple[np.ndarray, int]:
    """Generate a click track at 120 BPM for testing."""
    sr = 22050
    duration = 10.0
    bpm = 120.0
    beat_interval = 60.0 / bpm

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.zeros_like(t, dtype=np.float32)

    # Add clicks at beat positions
    for beat_time in np.arange(0, duration, beat_interval):
        idx = int(beat_time * sr)
        click_len = min(int(0.01 * sr), len(audio) - idx)  # 10ms click
        if click_len > 0:
            audio[idx : idx + click_len] = 0.8

    return audio, sr
