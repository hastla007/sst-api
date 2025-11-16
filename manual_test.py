#!/usr/bin/env python3
"""
Manual API Testing Script
Tests all endpoints of the STT API to verify functionality
"""

import requests
import json
import io
import wave
import struct
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
TEST_API_KEY = None  # Set to your API key if authentication is enabled

def create_test_wav_file() -> bytes:
    """Create a minimal valid WAV file for testing"""
    # Create a 1-second silent WAV file
    sample_rate = 16000
    duration = 1
    num_samples = sample_rate * duration

    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        # Write silence
        for _ in range(num_samples):
            wav_file.writeframes(struct.pack('<h', 0))

    return wav_buffer.getvalue()

def print_test_result(test_name: str, success: bool, response: Any = None, error: str = None):
    """Print formatted test result"""
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"\n{status} - {test_name}")
    if response:
        print(f"  Status Code: {response.status_code if hasattr(response, 'status_code') else 'N/A'}")
        if hasattr(response, 'json'):
            try:
                print(f"  Response: {json.dumps(response.json(), indent=2)[:200]}...")
            except:
                print(f"  Response: {str(response.text)[:200]}...")
    if error:
        print(f"  Error: {error}")

def get_headers() -> Dict[str, str]:
    """Get request headers with API key if configured"""
    headers = {}
    if TEST_API_KEY:
        headers["X-API-Key"] = TEST_API_KEY
    return headers

def test_health_endpoint():
    """Test /health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        success = response.status_code == 200
        data = response.json()
        success = success and data.get("status") == "healthy"
        print_test_result("Health Check", success, response)
        return success
    except Exception as e:
        print_test_result("Health Check", False, error=str(e))
        return False

def test_root_endpoint():
    """Test / endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        success = response.status_code == 200
        data = response.json()
        success = success and "service" in data
        print_test_result("Root Endpoint", success, response)
        return success
    except Exception as e:
        print_test_result("Root Endpoint", False, error=str(e))
        return False

def test_languages_endpoint():
    """Test /languages endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/languages", timeout=5)
        success = response.status_code == 200
        data = response.json()
        success = success and "supported_languages" in data and len(data["supported_languages"]) > 0
        print_test_result("Languages Endpoint", success, response)
        return success
    except Exception as e:
        print_test_result("Languages Endpoint", False, error=str(e))
        return False

def test_analytics_endpoint():
    """Test /analytics endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/analytics", headers=get_headers(), timeout=5)
        success = response.status_code in [200, 401]  # 401 if auth is enabled
        print_test_result("Analytics Endpoint", success, response)
        return success
    except Exception as e:
        print_test_result("Analytics Endpoint", False, error=str(e))
        return False

def test_metrics_endpoint():
    """Test /metrics endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/metrics", timeout=5)
        success = response.status_code in [200, 503]  # 503 if prometheus not available
        print_test_result("Metrics Endpoint", success, response)
        return success
    except Exception as e:
        print_test_result("Metrics Endpoint", False, error=str(e))
        return False

def test_transcribe_empty_file():
    """Test /transcribe with empty file (should fail)"""
    try:
        files = {"file": ("test.wav", io.BytesIO(b""), "audio/wav")}
        response = requests.post(
            f"{BASE_URL}/transcribe",
            files=files,
            headers=get_headers(),
            timeout=30
        )
        success = response.status_code == 400  # Should reject empty file
        print_test_result("Transcribe Empty File (Validation)", success, response)
        return success
    except Exception as e:
        print_test_result("Transcribe Empty File (Validation)", False, error=str(e))
        return False

def test_transcribe_invalid_language():
    """Test /transcribe with invalid language code (should fail)"""
    try:
        wav_data = create_test_wav_file()
        files = {"file": ("test.wav", io.BytesIO(wav_data), "audio/wav")}
        data = {"language": "invalid_lang_code"}
        response = requests.post(
            f"{BASE_URL}/transcribe",
            files=files,
            data=data,
            headers=get_headers(),
            timeout=30
        )
        success = response.status_code == 400  # Should reject invalid language
        print_test_result("Transcribe Invalid Language (Validation)", success, response)
        return success
    except Exception as e:
        print_test_result("Transcribe Invalid Language (Validation)", False, error=str(e))
        return False

def test_transcribe_invalid_format():
    """Test /transcribe with invalid export format (should fail)"""
    try:
        wav_data = create_test_wav_file()
        files = {"file": ("test.wav", io.BytesIO(wav_data), "audio/wav")}
        data = {"export_format": "invalid_format"}
        response = requests.post(
            f"{BASE_URL}/transcribe",
            files=files,
            data=data,
            headers=get_headers(),
            timeout=30
        )
        success = response.status_code == 400  # Should reject invalid format
        print_test_result("Transcribe Invalid Export Format (Validation)", success, response)
        return success
    except Exception as e:
        print_test_result("Transcribe Invalid Export Format (Validation)", False, error=str(e))
        return False

def test_transcribe_invalid_webhook():
    """Test /transcribe with invalid webhook URL (should fail)"""
    try:
        wav_data = create_test_wav_file()
        files = {"file": ("test.wav", io.BytesIO(wav_data), "audio/wav")}
        data = {"webhook_url": "not-a-valid-url"}
        response = requests.post(
            f"{BASE_URL}/transcribe",
            files=files,
            data=data,
            headers=get_headers(),
            timeout=30
        )
        success = response.status_code == 400  # Should reject invalid webhook URL
        print_test_result("Transcribe Invalid Webhook URL (Validation)", success, response)
        return success
    except Exception as e:
        print_test_result("Transcribe Invalid Webhook URL (Validation)", False, error=str(e))
        return False

def test_transcribe_valid_json():
    """Test /transcribe with valid file (JSON export)"""
    try:
        wav_data = create_test_wav_file()
        files = {"file": ("test.wav", io.BytesIO(wav_data), "audio/wav")}
        data = {"export_format": "json", "language": "en"}
        response = requests.post(
            f"{BASE_URL}/transcribe",
            files=files,
            data=data,
            headers=get_headers(),
            timeout=60
        )
        success = response.status_code in [200, 401, 429]  # 401 if auth required, 429 if rate limited
        if response.status_code == 200:
            resp_data = response.json()
            success = success and "text" in resp_data and "correlation_id" in resp_data
        print_test_result("Transcribe Valid File (JSON)", success, response)
        return success
    except Exception as e:
        print_test_result("Transcribe Valid File (JSON)", False, error=str(e))
        return False

def test_transcribe_valid_srt():
    """Test /transcribe with SRT export"""
    try:
        wav_data = create_test_wav_file()
        files = {"file": ("test.wav", io.BytesIO(wav_data), "audio/wav")}
        data = {"export_format": "srt", "language": "en"}
        response = requests.post(
            f"{BASE_URL}/transcribe",
            files=files,
            data=data,
            headers=get_headers(),
            timeout=60
        )
        success = response.status_code in [200, 401, 429]
        if response.status_code == 200:
            # SRT should contain timestamp format
            success = success and ("-->" in response.text or response.text == "")
        print_test_result("Transcribe Valid File (SRT)", success, response)
        return success
    except Exception as e:
        print_test_result("Transcribe Valid File (SRT)", False, error=str(e))
        return False

def test_transcribe_valid_vtt():
    """Test /transcribe with VTT export"""
    try:
        wav_data = create_test_wav_file()
        files = {"file": ("test.wav", io.BytesIO(wav_data), "audio/wav")}
        data = {"export_format": "vtt", "language": "en"}
        response = requests.post(
            f"{BASE_URL}/transcribe",
            files=files,
            data=data,
            headers=get_headers(),
            timeout=60
        )
        success = response.status_code in [200, 401, 429]
        if response.status_code == 200:
            # VTT should start with WEBVTT
            success = success and ("WEBVTT" in response.text or response.text == "WEBVTT\n\n")
        print_test_result("Transcribe Valid File (VTT)", success, response)
        return success
    except Exception as e:
        print_test_result("Transcribe Valid File (VTT)", False, error=str(e))
        return False

def test_batch_no_files():
    """Test /transcribe/batch with no files (should fail)"""
    try:
        response = requests.post(
            f"{BASE_URL}/transcribe/batch",
            headers=get_headers(),
            timeout=30
        )
        success = response.status_code in [400, 422]  # Should reject
        print_test_result("Batch Transcribe No Files (Validation)", success, response)
        return success
    except Exception as e:
        print_test_result("Batch Transcribe No Files (Validation)", False, error=str(e))
        return False

def test_batch_invalid_language():
    """Test /transcribe/batch with invalid language (should fail)"""
    try:
        wav_data = create_test_wav_file()
        files = [
            ("files", ("test1.wav", io.BytesIO(wav_data), "audio/wav")),
        ]
        data = {"language": "invalid_code"}
        response = requests.post(
            f"{BASE_URL}/transcribe/batch",
            files=files,
            data=data,
            headers=get_headers(),
            timeout=60
        )
        success = response.status_code == 400  # Should reject invalid language
        print_test_result("Batch Transcribe Invalid Language (Validation)", success, response)
        return success
    except Exception as e:
        print_test_result("Batch Transcribe Invalid Language (Validation)", False, error=str(e))
        return False

def test_batch_valid():
    """Test /transcribe/batch with valid files"""
    try:
        wav_data = create_test_wav_file()
        files = [
            ("files", ("test1.wav", io.BytesIO(wav_data), "audio/wav")),
            ("files", ("test2.wav", io.BytesIO(wav_data), "audio/wav")),
        ]
        response = requests.post(
            f"{BASE_URL}/transcribe/batch",
            files=files,
            headers=get_headers(),
            timeout=120
        )
        success = response.status_code in [200, 401, 429]
        if response.status_code == 200:
            resp_data = response.json()
            success = success and "results" in resp_data
        print_test_result("Batch Transcribe Valid Files", success, response)
        return success
    except Exception as e:
        print_test_result("Batch Transcribe Valid Files", False, error=str(e))
        return False

def test_task_status_invalid():
    """Test /task/{task_id} with invalid task ID"""
    try:
        response = requests.get(
            f"{BASE_URL}/task/invalid-task-id",
            headers=get_headers(),
            timeout=10
        )
        # Could be 400, 500, or 503 depending on whether Celery is available
        success = response.status_code in [400, 500, 503]
        print_test_result("Task Status Invalid ID", success, response)
        return success
    except Exception as e:
        print_test_result("Task Status Invalid ID", False, error=str(e))
        return False

def test_correlation_id():
    """Test that correlation ID is handled properly"""
    try:
        custom_id = "test-correlation-12345"
        response = requests.get(
            f"{BASE_URL}/health",
            headers={"X-Correlation-ID": custom_id},
            timeout=5
        )
        success = response.status_code == 200
        # Check if correlation ID is in response headers
        if success:
            success = response.headers.get("X-Correlation-ID") == custom_id
        print_test_result("Correlation ID Preservation", success, response)
        return success
    except Exception as e:
        print_test_result("Correlation ID Preservation", False, error=str(e))
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("STT API Manual Test Suite")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print(f"API Key: {'Configured' if TEST_API_KEY else 'Not configured'}")
    print("=" * 60)

    # Wait for server to be ready
    print("\nWaiting for server to be ready...")
    for i in range(10):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("Server is ready!\n")
                break
        except:
            pass
        time.sleep(1)
    else:
        print("WARNING: Server may not be ready. Proceeding with tests anyway.\n")

    # Run tests
    tests = [
        # Information endpoints
        test_health_endpoint,
        test_root_endpoint,
        test_languages_endpoint,
        test_analytics_endpoint,
        test_metrics_endpoint,
        test_correlation_id,

        # Validation tests
        test_transcribe_empty_file,
        test_transcribe_invalid_language,
        test_transcribe_invalid_format,
        test_transcribe_invalid_webhook,

        # Functional tests
        test_transcribe_valid_json,
        test_transcribe_valid_srt,
        test_transcribe_valid_vtt,

        # Batch tests
        test_batch_no_files,
        test_batch_invalid_language,
        test_batch_valid,

        # Task status
        test_task_status_invalid,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)
        time.sleep(0.5)  # Small delay between tests

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print("=" * 60)

    return passed == total

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
