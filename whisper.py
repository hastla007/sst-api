"""Compatibility shim for the Whisper dependency.

This module prefers the real ``openai-whisper`` package when it is installed
and available on the Python path. If the package cannot be imported (for
example, in constrained test environments), it falls back to a lightweight
mock implementation so the application can still start.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import os
import sys
from typing import Any, Optional


logger = logging.getLogger(__name__)


def _load_real_whisper() -> Optional[object]:
    """Attempt to import the installed ``whisper`` package, bypassing this file.

    Because this shim shares the same module name as the installed package, the
    standard import machinery would resolve back to this file. To avoid the
    loop, we ask the :class:`PathFinder` to search *other* entries in
    ``sys.path`` (excluding the current working directory and the empty string
    that also points to it). If a different module is found, we load and return
    it; otherwise ``None`` is returned and the caller can decide how to proceed.
    """

    search_path = [p for p in sys.path if p not in {"", os.getcwd()}]
    spec = importlib.machinery.PathFinder.find_spec("whisper", search_path)
    if spec and spec.loader and spec.origin != __file__:
        module = importlib.util.module_from_spec(spec)
        # Ensure the module is registered before execution so relative imports
        # inside the real package (e.g., ``from .audio import ...``) resolve
        # correctly. Without this, the loader may treat the module as a plain
        # file rather than a package, leading to ``'whisper' is not a package``
        # errors during Docker builds.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        return module
    return None


_real_whisper = _load_real_whisper()

if _real_whisper:
    logger.info("Using installed openai-whisper implementation")
    # Expose the real module's public attributes so consumers behave normally.
    for attr in dir(_real_whisper):
        if attr.startswith("__"):
            continue
        globals()[attr] = getattr(_real_whisper, attr)
else:
    logger.warning("Falling back to mock Whisper implementation; audio will not be transcribed")

    class MockWhisperModel:
        def __init__(self, name: str, device: str = "cpu"):
            self.name = name
            self.device = device

        def transcribe(self, audio_path: str, **kwargs: Any) -> dict[str, Any]:
            """Mock transcribe method"""
            return {
                "text": "This is a mock transcription",
                "language": kwargs.get("language", "en"),
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "This is a mock transcription",
                    }
                ],
            }

    def load_model(name: str, device: str = "cpu") -> MockWhisperModel:
        """Mock load_model function"""

        return MockWhisperModel(name, device)
