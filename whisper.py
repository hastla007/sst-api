"""
Mock whisper module for testing purposes
This allows tests to run without installing the full openai-whisper package
"""


class MockWhisperModel:
    def __init__(self, name, device="cpu"):
        self.name = name
        self.device = device

    def transcribe(self, audio_path, **kwargs):
        """Mock transcribe method"""
        return {
            "text": "This is a mock transcription",
            "language": kwargs.get("language", "en"),
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "This is a mock transcription"
                }
            ]
        }


def load_model(name, device="cpu"):
    """Mock load_model function"""
    return MockWhisperModel(name, device)
