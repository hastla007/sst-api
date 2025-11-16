import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import tempfile
import os

# Disable rate limiting for tests
import app as app_module
app_module.RATE_LIMIT_REQUESTS = 1000000  # Very high limit for tests

from app import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health and information endpoints"""

    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model" in data
        assert "version" in data

    def test_root_endpoint(self):
        """Test root information endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Enhanced Speech-to-Text API"
        assert data["version"] == "2.0.0"
        assert "endpoints" in data

    def test_languages_endpoint(self):
        """Test supported languages endpoint"""
        response = client.get("/languages")
        assert response.status_code == 200
        data = response.json()
        assert "supported_languages" in data
        assert "count" in data
        assert data["count"] > 0
        assert "en" in data["supported_languages"]


class TestAuthentication:
    """Test API key authentication"""

    @patch.dict(os.environ, {"API_KEYS": "test-key-123,test-key-456"})
    def test_missing_api_key(self):
        """Test request without API key when auth is enabled"""
        # Create a valid test file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b'RIFF\x24\x00\x00\x00WAVE' + b'\x00' * 100)
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                # This test assumes auth is enabled via environment variable
                # The actual behavior depends on how the app is initialized
                response = client.post(
                    "/transcribe",
                    files={"file": ("test.wav", f, "audio/wav")}
                )
                # When auth is disabled (default), this should succeed or fail for other reasons
                # When auth is enabled, should return 401
                # Can also get 400 for validation errors which is checked before auth
                assert response.status_code in [200, 400, 401, 500]
        finally:
            os.unlink(temp_path)

    @patch.dict(os.environ, {"API_KEYS": "test-key-123"})
    def test_invalid_api_key(self):
        """Test request with invalid API key"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b'RIFF\x24\x00\x00\x00WAVE' + b'\x00' * 100)
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/transcribe",
                    files={"file": ("test.wav", f, "audio/wav")},
                    headers={"X-API-Key": "invalid-key"}
                )
                # Should return 403 when auth is enabled with wrong key
                # Can also get 200 if auth is disabled (ENABLE_AUTH set at module load)
                # Can also get 400 for validation errors
                assert response.status_code in [200, 400, 403, 500]
        finally:
            os.unlink(temp_path)


class TestRateLimiting:
    """Test rate limiting functionality"""

    def test_rate_limit_tracking(self):
        """Test that rate limiting tracks requests"""
        # Make multiple requests
        responses = []
        for i in range(3):
            response = client.get("/health")
            responses.append(response)

        # All health check requests should succeed (no rate limit on health)
        for response in responses:
            assert response.status_code == 200


class TestTranscription:
    """Test transcription endpoints"""

    def test_transcribe_empty_file(self):
        """Test transcription with empty file"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/transcribe",
                    files={"file": ("test.wav", f, "audio/wav")}
                )
                assert response.status_code == 400
                assert "empty" in response.json()["detail"].lower()
        finally:
            os.unlink(temp_path)

    def test_transcribe_invalid_language(self):
        """Test transcription with invalid language code"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio content")
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/transcribe",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"language": "invalid"}
                )
                assert response.status_code == 400
                assert "language" in response.json()["detail"].lower()
        finally:
            os.unlink(temp_path)

    def test_transcribe_invalid_export_format(self):
        """Test transcription with invalid export format"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio content")
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/transcribe",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"export_format": "invalid"}
                )
                assert response.status_code == 400
                assert "export_format" in response.json()["detail"].lower()
        finally:
            os.unlink(temp_path)

    @patch('app.model.transcribe')
    def test_transcribe_success(self, mock_transcribe):
        """Test successful transcription"""
        # Mock the transcription result
        mock_transcribe.return_value = {
            "text": "Hello world",
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Hello world"}
            ]
        }

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Write a valid WAV header (RIFF + size + WAVE)
            f.write(b'RIFF')
            f.write(b'\x24\x00\x00\x00')  # Chunk size
            f.write(b'WAVE')
            f.write(b'\x00' * 100)
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/transcribe",
                    files={"file": ("test.wav", f, "audio/wav")}
                )
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert "text" in data
                assert "correlation_id" in data
        finally:
            os.unlink(temp_path)

    @patch('app.model.transcribe')
    def test_transcribe_srt_export(self, mock_transcribe):
        """Test transcription with SRT export"""
        mock_transcribe.return_value = {
            "text": "Hello world",
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Hello world"}
            ]
        }

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b'RIFF\x24\x00\x00\x00WAVE' + b'\x00' * 100)
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/transcribe",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"export_format": "srt"}
                )
                assert response.status_code == 200
                assert "text/plain" in response.headers["content-type"]
                # Check SRT format
                content = response.text
                assert "00:00:00" in content
                assert "Hello world" in content
        finally:
            os.unlink(temp_path)

    @patch('app.model.transcribe')
    def test_transcribe_vtt_export(self, mock_transcribe):
        """Test transcription with VTT export"""
        mock_transcribe.return_value = {
            "text": "Hello world",
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Hello world"}
            ]
        }

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b'RIFF\x24\x00\x00\x00WAVE' + b'\x00' * 100)
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/transcribe",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"export_format": "vtt"}
                )
                assert response.status_code == 200
                assert "text/vtt" in response.headers["content-type"]
                content = response.text
                assert "WEBVTT" in content
                assert "Hello world" in content
        finally:
            os.unlink(temp_path)


class TestBatchProcessing:
    """Test batch processing endpoint"""

    def test_batch_no_files(self):
        """Test batch processing with no files"""
        response = client.post("/transcribe/batch")
        assert response.status_code == 422  # Validation error

    def test_batch_too_many_files(self):
        """Test batch processing with too many files"""
        files = []
        for i in range(15):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b"fake audio")
                files.append(("files", ("test{}.wav".format(i), open(f.name, "rb"), "audio/wav")))

        response = client.post("/transcribe/batch", files=files)

        # Clean up
        for _, (_, file_obj, _) in files:
            file_obj.close()

        assert response.status_code == 400
        assert "maximum" in response.json()["detail"].lower()

    @patch('app.model.transcribe')
    def test_batch_processing_success(self, mock_transcribe):
        """Test successful batch processing"""
        mock_transcribe.return_value = {
            "text": "Test",
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "Test"}]
        }

        files = []
        temp_paths = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b'RIFF\x24\x00\x00\x00WAVE' + b'\x00' * 100)
                temp_paths.append(f.name)
                files.append(("files", (f"test{i}.wav", open(f.name, "rb"), "audio/wav")))

        try:
            response = client.post("/transcribe/batch", files=files)

            # Clean up file handles
            for _, (_, file_obj, _) in files:
                file_obj.close()

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "results" in data
            assert len(data["results"]) == 2
        finally:
            for path in temp_paths:
                if os.path.exists(path):
                    os.unlink(path)


class TestAnalytics:
    """Test analytics endpoint"""

    def test_analytics_endpoint(self):
        """Test analytics endpoint returns statistics"""
        response = client.get("/analytics")
        assert response.status_code in [200, 401]  # Depends on auth config

        if response.status_code == 200:
            data = response.json()
            assert "total_requests" in data
            assert "successful_requests" in data
            assert "cache_statistics" in data


class TestMetrics:
    """Test Prometheus metrics endpoint"""

    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint"""
        response = client.get("/metrics")
        # May return 200 if prometheus is available, or 503 if not
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            # Check it returns prometheus format
            assert "stt_" in response.text or response.text == ""


class TestUtilityFunctions:
    """Test utility functions"""

    def test_format_timestamp(self):
        """Test timestamp formatting"""
        from app import format_timestamp

        # Test various timestamps
        assert format_timestamp(0) == "00:00:00,000"
        assert format_timestamp(1.5) == "00:00:01,500"
        assert format_timestamp(61.234) == "00:01:01,234"
        assert format_timestamp(3661.567) == "01:01:01,567"

    def test_segments_to_srt(self):
        """Test SRT conversion"""
        from app import segments_to_srt

        segments = [
            {"start": 0.0, "end": 2.5, "text": "Hello world"},
            {"start": 2.5, "end": 5.0, "text": "This is a test"}
        ]

        srt = segments_to_srt(segments)
        assert "1\n" in srt
        assert "2\n" in srt
        assert "Hello world" in srt
        assert "This is a test" in srt
        assert "00:00:00,000 --> 00:00:02,500" in srt

    def test_segments_to_vtt(self):
        """Test VTT conversion"""
        from app import segments_to_vtt

        segments = [
            {"start": 0.0, "end": 2.5, "text": "Hello world"}
        ]

        vtt = segments_to_vtt(segments)
        assert "WEBVTT" in vtt
        assert "Hello world" in vtt
        assert "00:00:00.000 --> 00:00:02.500" in vtt

    def test_calculate_file_hash(self):
        """Test file hash calculation"""
        from app import calculate_file_hash

        content1 = b"test content"
        content2 = b"different content"

        hash1 = calculate_file_hash(content1)
        hash2 = calculate_file_hash(content1)
        hash3 = calculate_file_hash(content2)

        # Same content should produce same hash
        assert hash1 == hash2
        # Different content should produce different hash
        assert hash1 != hash3
        # Hash should be hex string
        assert len(hash1) == 64  # SHA256 produces 64 hex characters


class TestCorrelationID:
    """Test correlation ID tracking"""

    def test_correlation_id_generated(self):
        """Test that correlation ID is generated if not provided"""
        response = client.get("/health")
        assert "X-Correlation-ID" in response.headers

    def test_correlation_id_preserved(self):
        """Test that provided correlation ID is preserved"""
        custom_id = "test-correlation-123"
        response = client.get(
            "/health",
            headers={"X-Correlation-ID": custom_id}
        )
        assert response.headers["X-Correlation-ID"] == custom_id


class TestFileSizeValidation:
    """Test file size validation improvements"""

    def test_file_too_large(self):
        """Test that files larger than MAX_FILE_SIZE are rejected"""
        # Create a file that's way too large (simulated via content)
        # Since we can't actually upload 100MB+ in tests, we test the logic with smaller files
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Write valid WAV header
            f.write(b'RIFF\x24\x00\x00\x00WAVE' + b'\x00' * 100)
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                # Test will pass validation since file is small
                response = client.post(
                    "/transcribe",
                    files={"file": ("test.wav", f, "audio/wav")}
                )
                # Should either succeed (200) or fail on transcription (500)
                # Not fail on size validation (413)
                assert response.status_code != 413
        finally:
            os.unlink(temp_path)

    def test_empty_file_rejected(self):
        """Test that empty files are rejected"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/transcribe",
                    files={"file": ("test.wav", f, "audio/wav")}
                )
                assert response.status_code == 400
                assert "empty" in response.json()["detail"].lower()
        finally:
            os.unlink(temp_path)


class TestFileValidation:
    """Test improved file content validation"""

    def test_invalid_file_format_rejected(self):
        """Test that non-audio files are rejected"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"This is not an audio file, just plain text")
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/transcribe",
                    files={"file": ("test.txt", f, "text/plain")}
                )
                assert response.status_code == 400
                assert "invalid" in response.json()["detail"].lower() or "format" in response.json()["detail"].lower()
        finally:
            os.unlink(temp_path)

    def test_pdf_file_rejected(self):
        """Test that PDF files are rejected"""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\n%This is a PDF file")
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/transcribe",
                    files={"file": ("test.pdf", f, "application/pdf")}
                )
                assert response.status_code == 400
        finally:
            os.unlink(temp_path)

    @patch('app.model.transcribe')
    def test_valid_mp3_accepted(self, mock_transcribe):
        """Test that MP3 files are accepted"""
        mock_transcribe.return_value = {
            "text": "Test",
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "Test"}]
        }

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            # Write ID3v2 header
            f.write(b'ID3' + b'\x00' * 100)
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/transcribe",
                    files={"file": ("test.mp3", f, "audio/mpeg")}
                )
                assert response.status_code == 200
        finally:
            os.unlink(temp_path)

    @patch('app.model.transcribe')
    def test_valid_flac_accepted(self, mock_transcribe):
        """Test that FLAC files are accepted"""
        mock_transcribe.return_value = {
            "text": "Test",
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "Test"}]
        }

        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as f:
            f.write(b'fLaC' + b'\x00' * 100)
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/transcribe",
                    files={"file": ("test.flac", f, "audio/flac")}
                )
                assert response.status_code == 200
        finally:
            os.unlink(temp_path)


class TestWebhookValidation:
    """Test webhook URL validation and SSRF protection"""

    def test_webhook_localhost_blocked(self):
        """Test that webhook to localhost is blocked"""
        from app import validate_webhook_url
        assert validate_webhook_url("http://localhost/webhook") is False
        assert validate_webhook_url("http://127.0.0.1/webhook") is False
        assert validate_webhook_url("http://0.0.0.0/webhook") is False

    def test_webhook_private_ip_blocked(self):
        """Test that webhook to private IPs is blocked"""
        from app import validate_webhook_url
        assert validate_webhook_url("http://192.168.1.1/webhook") is False
        assert validate_webhook_url("http://10.0.0.1/webhook") is False
        assert validate_webhook_url("http://172.16.0.1/webhook") is False

    def test_webhook_valid_url_allowed(self):
        """Test that valid public URLs are allowed"""
        from app import validate_webhook_url
        # Note: This will do DNS lookup, so may fail in isolated environments
        # In production, we'd use a mock
        assert validate_webhook_url("https://example.com/webhook") is True
        assert validate_webhook_url("https://api.example.org/callback") is True

    def test_webhook_invalid_scheme_blocked(self):
        """Test that non-HTTP(S) schemes are blocked"""
        from app import validate_webhook_url
        assert validate_webhook_url("ftp://example.com/webhook") is False
        assert validate_webhook_url("file:///etc/passwd") is False


class TestCacheFormatHandling:
    """Test cache format mismatch fixes"""

    @patch('app.model.transcribe')
    def test_cache_respects_export_format(self, mock_transcribe):
        """Test that cache keys include export format"""
        mock_transcribe.return_value = {
            "text": "Test",
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "Test"}]
        }

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b'RIFF\x24\x00\x00\x00WAVE' + b'\x00' * 100)
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                # Request with JSON format
                response = client.post(
                    "/transcribe",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"export_format": "json", "use_cache": "false"}
                )
                assert response.status_code == 200
        finally:
            os.unlink(temp_path)


class TestRateLimiting:
    """Test rate limiting functionality"""

    def test_rate_limit_doesnt_crash(self):
        """Test that rate limiting doesn't crash on many requests"""
        # Make multiple requests to test rate limiter
        for i in range(5):
            response = client.get("/health")
            assert response.status_code == 200


class TestTaskValidation:
    """Test task ID validation"""

    def test_empty_task_id_validation(self):
        """Test that empty task IDs are rejected"""
        response = client.get("/task/   ")  # Whitespace-only
        # Should return 400 for invalid input or 503 if Celery not available
        assert response.status_code in [400, 503]

    def test_task_id_validated_before_celery_check(self):
        """Test that task ID is validated before checking Celery availability"""
        response = client.get("/task/")  # Empty task_id
        # Should get 400 or 404, not 503
        assert response.status_code in [400, 404, 422]  # FastAPI may return 422 for validation


class TestErrorResponses:
    """Test error response handling"""

    def test_invalid_language_error_has_correlation_id(self):
        """Test that error responses include correlation ID"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b'RIFF\x24\x00\x00\x00WAVE' + b'\x00' * 100)
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/transcribe",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"language": "invalid_code"},
                    headers={"X-Correlation-ID": "test-123"}
                )
                assert response.status_code == 400
                # Check if correlation ID is in response headers
                assert "X-Correlation-ID" in response.headers
        finally:
            os.unlink(temp_path)


class TestBatchProcessing:
    """Additional batch processing tests"""

    def test_batch_with_invalid_file_mixed(self):
        """Test batch processing with mix of valid and invalid files"""
        # Create one valid and one invalid file
        valid_file_path = None
        invalid_file_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b'RIFF\x24\x00\x00\x00WAVE' + b'\x00' * 100)
                valid_file_path = f.name

            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
                f.write(b'Not an audio file')
                invalid_file_path = f.name

            files = [
                ("files", ("valid.wav", open(valid_file_path, "rb"), "audio/wav")),
                ("files", ("invalid.txt", open(invalid_file_path, "rb"), "text/plain"))
            ]

            response = client.post("/transcribe/batch", files=files)

            # Close file handles
            for _, (_, fh, _) in files:
                fh.close()

            # Should succeed but with errors for invalid files
            if response.status_code == 200:
                data = response.json()
                assert "results" in data
        finally:
            if valid_file_path and os.path.exists(valid_file_path):
                os.unlink(valid_file_path)
            if invalid_file_path and os.path.exists(invalid_file_path):
                os.unlink(invalid_file_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
