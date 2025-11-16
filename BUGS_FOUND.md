# Bugs and Issues Found in STT API

## Critical Bugs

### 1. File Size Validation Timing (app.py:519-527)
**Severity:** Critical - Can cause memory exhaustion
**Issue:** File content is read entirely into memory BEFORE checking file size
```python
content = await file.read()  # Line 519 - reads full file
file_size = len(content)     # Line 520
if file_size > MAX_FILE_SIZE:  # Line 523 - checks AFTER reading
```
**Impact:** Attacker can send 10GB file, crash server with OOM
**Fix:** Check Content-Length header before reading, or use streaming

### 2. Webhook URL SSRF Vulnerability (app.py:281-287)
**Severity:** High - Security vulnerability
**Issue:** Only validates URL format, doesn't prevent internal network access
```python
def validate_webhook_url(url: str) -> bool:
    result = urlparse(url)
    return all([result.scheme in ['http', 'https'], result.netloc])
```
**Impact:** Attacker can make server send requests to localhost, internal services, cloud metadata endpoints
**Fix:** Block private IP ranges, localhost, link-local addresses

### 3. Cache Format Mismatch (app.py:543-563)
**Severity:** Medium - Returns wrong data format
**Issue:** Cached results don't include export_format in cache key
**Impact:** Request with export_format=json caches result, then request with srt gets JSON
**Fix:** Include export_format in cache key

## High Priority Bugs

### 4. Rate Limiter Memory Leak (app.py:258-264)
**Severity:** High - Memory leak
**Issue:** In-memory rate_limit_storage defaultdict never removes old keys
```python
rate_limit_storage = defaultdict(lambda: deque())  # Line 137
# Keys accumulate forever, only timestamps are removed
```
**Impact:** Memory grows unbounded with unique identifiers
**Fix:** Periodically clean up keys with empty deques or old data

### 5. Task ID Validation Order (app.py:804-808)
**Severity:** Medium - Poor error messages
**Issue:** Checks celery_app before validating task_id input
```python
if not celery_app:  # Line 798
    raise HTTPException(...)
if not task_id or not task_id.strip():  # Line 804
```
**Impact:** User gets "Celery not configured" instead of "Invalid task_id"
**Fix:** Validate inputs first, then check system state

### 6. Batch Processing Resource Leak (app.py:756-770)
**Severity:** Medium - Resource leak
**Issue:** Temporary files in batch processing not cleaned up on errors
```python
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
    temp_file.write(content)
    temp_file_path = temp_file.name
result = transcribe_file(temp_file_path, language)  # Can fail
os.unlink(temp_file_path)  # Never reached if error
```
**Impact:** Disk space leak on errors
**Fix:** Use try/finally for cleanup

## Medium Priority Issues

### 7. File Content Validation (app.py:533-534)
**Severity:** Medium - Weak validation
**Issue:** Only logs warning for invalid files, doesn't reject them
**Impact:** Wastes resources processing non-audio files
**Fix:** Validate content-type header and magic numbers, reject invalid files

### 8. Missing Content-Type Enforcement (app.py:461-517)
**Severity:** Medium - Validation gap
**Issue:** SUPPORTED_AUDIO_TYPES defined but never enforced
**Impact:** Server accepts any file type, even PDFs or executables
**Fix:** Validate file.content_type against SUPPORTED_AUDIO_TYPES

### 9. Empty File Validation Placement (app.py:529-530)
**Severity:** Low - Inefficient validation order
**Issue:** Empty file check happens after file size check
**Fix:** Check for empty file first (cheaper validation)

### 10. Correlation ID Not in All Errors
**Severity:** Low - Debugging difficulty
**Issue:** HTTPExceptions don't include correlation_id in detail
**Impact:** Harder to correlate error logs with requests
**Fix:** Include correlation_id in error responses

## Code Quality Issues

### 11. Duplicate Webhook Validation
**Location:** app.py:512-516 and 719-724
**Issue:** Same validation code duplicated
**Fix:** Already using validate_webhook_url function, no issue actually

### 12. Incomplete Task ID Validation (app.py:804-808)
**Issue:** Only checks if task_id exists, not if format is valid
**Fix:** Validate UUID format

## Test Coverage Gaps

1. No test for max file size enforcement
2. No test for file content-type validation
3. No test for webhook SSRF protection
4. No test for cache format mismatch
5. No test for rate limiter memory cleanup
6. No test for batch processing error cleanup
7. No test for correlation ID in errors
8. No test for concurrent requests
9. No test for very large files (streaming)
10. No integration test with real audio files

## Summary
- **Critical Bugs:** 3
- **High Priority:** 3
- **Medium Priority:** 4
- **Low Priority:** 2
- **Total:** 12 bugs identified
