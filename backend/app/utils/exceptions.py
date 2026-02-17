"""Typed exception hierarchy for Harmony."""


class HarmonyError(Exception):
    """Base exception for all Harmony errors."""


class AudioLoadError(HarmonyError):
    """Failed to load or decode an audio file."""


class UnsupportedFormatError(HarmonyError):
    """Audio format is not supported."""


class AnalysisError(HarmonyError):
    """An error occurred during audio analysis."""


class ModelLoadError(HarmonyError):
    """Failed to load an ML model."""


class ModelInferenceError(HarmonyError):
    """An error occurred during model inference."""
