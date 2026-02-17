"""ONNX model registry — loads, caches, and serves models."""

from __future__ import annotations

from pathlib import Path

import structlog

from app.config import settings

logger = structlog.get_logger()


class ModelRegistry:
    """Manages ONNX model loading and lifecycle."""

    def __init__(self) -> None:
        self._sessions: dict[str, object] = {}
        self.is_loaded = False

    async def preload(self) -> None:
        """Pre-load all ONNX models into memory."""
        models_dir = settings.models_dir

        if not models_dir.exists():
            logger.warning("models.dir_missing", path=str(models_dir))
            self.is_loaded = True  # No models to load is OK for initial setup
            return

        for onnx_path in models_dir.glob("*.onnx"):
            try:
                self._load_model(onnx_path)
                logger.info("models.loaded", model=onnx_path.stem)
            except Exception as e:
                logger.error("models.load_failed", model=onnx_path.stem, error=str(e))

        self.is_loaded = True

    def _load_model(self, path: Path) -> None:
        """Load a single ONNX model with optimal execution provider."""
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning("onnxruntime not installed, skipping model loading")
            return

        providers = self._get_providers()
        session = ort.InferenceSession(str(path), providers=providers)
        self._sessions[path.stem] = session

    def _get_providers(self) -> list[str]:
        """Select best available ONNX execution providers."""
        device = settings.onnx_device

        if device == "auto":
            providers = []
            try:
                import onnxruntime as ort
                available = ort.get_available_providers()
                # Priority order
                for p in [
                    "CUDAExecutionProvider",
                    "CoreMLExecutionProvider",
                    "DmlExecutionProvider",
                    "CPUExecutionProvider",
                ]:
                    if p in available:
                        providers.append(p)
            except ImportError:
                providers = ["CPUExecutionProvider"]
            return providers or ["CPUExecutionProvider"]

        provider_map = {
            "cpu": ["CPUExecutionProvider"],
            "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
            "directml": ["DmlExecutionProvider", "CPUExecutionProvider"],
        }
        return provider_map.get(device, ["CPUExecutionProvider"])

    def get_session(self, model_name: str) -> object | None:
        """Get a loaded ONNX inference session by model name."""
        return self._sessions.get(model_name)


# Singleton
registry = ModelRegistry()
