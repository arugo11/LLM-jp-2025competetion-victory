"""
Simple Load Balancer for tool_server instances.

Features:
- Round-robin with retry on failure
- Backend health tracking (temporary exclusion on consecutive failures)
- Reused AsyncClient for connection pooling
- Proper error handling and response formatting
- Observability with logging (backend, latency, failure reason)

Usage:
    TOOL_SERVER_BASE_URLS="http://localhost:1214,http://localhost:1215" \
    uvicorn simple_loadbalancer:app --host 0.0.0.0 --port 1213
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Set

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

# Response schema (consistent for success and failure)
class CodeResponse(BaseModel):
    output: str
    error: Optional[str] = None
    success: bool


class BackendStatus:
    """Track status and failure count for each backend."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.execute_url = f"{self.base_url}/execute"
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.is_healthy = True
        # Temporary exclusion after MAX_CONSECUTIVE_FAILURES
        self.MAX_CONSECUTIVE_FAILURES = 3
        self.EXCLUSION_DURATION_SECONDS = 30.0  # Exclude for 30 seconds

    def record_success(self):
        """Reset failure count on success."""
        self.failure_count = 0
        self.is_healthy = True
        self.last_failure_time = None

    def record_failure(self):
        """Increment failure count and check if should be excluded."""
        self.failure_count += 1
        self.last_failure_time = time.monotonic()  # Use monotonic time
        if self.failure_count >= self.MAX_CONSECUTIVE_FAILURES:
            self.is_healthy = False
            logger.warning(
                f"Backend {self.base_url} marked as unhealthy after {self.failure_count} consecutive failures"
            )

    def is_exclusion_expired(self) -> bool:
        """Check if exclusion period has expired (pure function, no side effects)."""
        if self.is_healthy:
            return True
        if self.last_failure_time is None:
            return True
        return time.monotonic() - self.last_failure_time > self.EXCLUSION_DURATION_SECONDS  # Use monotonic time

    def should_retry(self) -> bool:
        """Check if backend should be retried (no side effects)."""
        if self.is_healthy:
            return True
        # Check if exclusion period has expired
        if self.is_exclusion_expired():
            return True
        return False

    def reset_if_expired(self):
        """Reset exclusion if expired (separate method for side effects)."""
        if not self.is_healthy and self.is_exclusion_expired():
            logger.info(f"Backend {self.base_url} exclusion expired, resetting")
            self.is_healthy = True
            self.failure_count = 0


class LoadBalancer:
    """Load balancer with round-robin, retry, and health tracking."""

    def __init__(self, base_urls: List[str]):
        self.backends = [BackendStatus(url) for url in base_urls]
        self.round_robin_counter = 0
        # Use asyncio.Lock for async context
        self.counter_lock = asyncio.Lock()
        # Lock for AsyncClient creation to prevent race conditions
        self.client_lock = asyncio.Lock()
        # Reused AsyncClient for connection pooling
        self.client: Optional[httpx.AsyncClient] = None
        logger.info(f"LoadBalancer initialized with {len(self.backends)} backends")

    @asynccontextmanager
    async def get_client(self):
        """Get or create AsyncClient (reused across requests) with lock to prevent race conditions."""
        # Double-check pattern with lock to prevent multiple clients from being created
        if self.client is None:
            async with self.client_lock:
                # Check again after acquiring lock
                if self.client is None:
                    self.client = httpx.AsyncClient(
                        timeout=httpx.Timeout(30.0, connect=5.0),
                        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                    )
                    logger.info("AsyncClient created for load balancer")
        yield self.client

    async def close(self):
        """Close AsyncClient on shutdown."""
        if self.client:
            async with self.client_lock:
                if self.client:
                    await self.client.aclose()
                    self.client = None
                    logger.info("AsyncClient closed")

    async def get_next_backend(self) -> Optional[BackendStatus]:
        """Get next backend using round-robin, skipping unhealthy ones."""
        # Reset expired exclusions before selecting
        for backend in self.backends:
            backend.reset_if_expired()

        healthy_backends = [b for b in self.backends if b.should_retry()]
        if not healthy_backends:
            # If all are unhealthy, try all backends anyway
            healthy_backends = self.backends
            logger.warning("All backends marked unhealthy, trying anyway")

        if not healthy_backends:
            return None

        async with self.counter_lock:
            backend = healthy_backends[self.round_robin_counter % len(healthy_backends)]
            self.round_robin_counter += 1
            return backend

    async def forward_request(self, request_data: dict) -> CodeResponse:
        """Forward request to backends with retry logic."""
        tried_backends: Set[str] = set()  # Use set instead of list
        last_error: Optional[str] = None

        # Try all backends in round-robin order until one succeeds
        for attempt in range(len(self.backends)):
            backend = await self.get_next_backend()
            if backend is None:
                break

            if backend.base_url in tried_backends:
                # Already tried this backend, skip
                continue

            tried_backends.add(backend.base_url)  # Use set.add()
            start_time = time.perf_counter()

            try:
                async with self.get_client() as client:
                    response = await client.post(
                        backend.execute_url,
                        json=request_data,
                        # Content-Type header removed - httpx sets it automatically
                    )
                    latency_ms = (time.perf_counter() - start_time) * 1000

                    # Check HTTP status first
                    if response.status_code != 200:
                        error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                        backend.record_failure()
                        logger.warning(
                            f"Request failed: backend={backend.base_url}, "
                            f"status={response.status_code}, latency={latency_ms:.1f}ms, "
                            f"error={error_msg[:100]}"
                        )
                        last_error = error_msg
                        continue

                    # Parse JSON response
                    try:
                        result_dict = response.json()
                    except Exception as e:
                        # JSON parse failure is treated as failure
                        error_msg = f"JSON parse error: {str(e)}"
                        backend.record_failure()
                        logger.warning(
                            f"Request failed: backend={backend.base_url}, "
                            f"JSON parse error, latency={latency_ms:.1f}ms, "
                            f"error={error_msg[:100]}"
                        )
                        last_error = error_msg
                        continue

                    # Ensure consistent schema and check success flag
                    try:
                        result = CodeResponse(
                            output=result_dict.get("output", ""),
                            error=result_dict.get("error"),
                            success=result_dict.get("success", True),
                        )
                    except Exception as e:
                        # Schema validation failure is treated as failure
                        error_msg = f"Schema validation error: {str(e)}"
                        backend.record_failure()
                        logger.warning(
                            f"Request failed: backend={backend.base_url}, "
                            f"Schema validation error, latency={latency_ms:.1f}ms, "
                            f"error={error_msg[:100]}"
                        )
                        last_error = error_msg
                        continue

                    # Check success flag - HTTP 200 but success=False is still a failure
                    if not result.success:
                        error_msg = result.error or "success=False in response"
                        backend.record_failure()
                        logger.warning(
                            f"Request failed: backend={backend.base_url}, "
                            f"success=False, latency={latency_ms:.1f}ms, "
                            f"error={error_msg[:100]}"
                        )
                        last_error = error_msg
                        continue

                    # Success: HTTP 200 AND success=True
                    backend.record_success()
                    logger.info(
                        f"Request succeeded: backend={backend.base_url}, "
                        f"latency={latency_ms:.1f}ms, attempt={attempt + 1}"
                    )
                    return result

            except httpx.HTTPError as e:
                # HTTPError covers both network errors (RequestError) and HTTP status errors
                # (HTTPStatusError). This is future-safe if we ever use response.raise_for_status().
                latency_ms = (time.perf_counter() - start_time) * 1000

                status_code_part = ""
                if isinstance(e, httpx.HTTPStatusError):
                    status_code_part = f" status={e.response.status_code}"

                error_msg = f"HTTPError: {type(e).__name__}:{status_code_part} {str(e)}"
                backend.record_failure()
                logger.warning(
                    f"Request failed: backend={backend.base_url}, "
                    f"HTTPError={type(e).__name__}{status_code_part}, latency={latency_ms:.1f}ms, "
                    f"error={str(e)[:100]}"
                )
                last_error = error_msg
                continue

            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                error_msg = f"Unexpected error: {type(e).__name__}: {str(e)}"
                backend.record_failure()
                logger.error(
                    f"Request failed: backend={backend.base_url}, "
                    f"Exception={type(e).__name__}, latency={latency_ms:.1f}ms, "
                    f"error={str(e)[:100]}",
                    exc_info=True,
                )
                last_error = error_msg
                continue

        # All backends failed - return consistent error response
        error_response = CodeResponse(
            output="",
            error=f"All backends failed. Last error: {last_error}. Tried: {', '.join(sorted(tried_backends))}",
            success=False,
        )
        logger.error(
            f"All backends failed after {len(tried_backends)} attempts. "
            f"Tried: {sorted(tried_backends)}, Last error: {last_error}"
        )
        return error_response


# Initialize load balancer
load_balancer: Optional[LoadBalancer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage load balancer lifecycle."""
    global load_balancer

    # Startup: Initialize load balancer
    base_urls_str = os.getenv("TOOL_SERVER_BASE_URLS", "")
    if not base_urls_str:
        raise ValueError("TOOL_SERVER_BASE_URLS environment variable must be set")

    base_urls = [url.strip() for url in base_urls_str.split(",") if url.strip()]
    if not base_urls:
        raise ValueError("TOOL_SERVER_BASE_URLS must contain at least one base URL")

    load_balancer = LoadBalancer(base_urls)
    logger.info(f"Load balancer started with backends: {base_urls}")

    yield

    # Shutdown: Close connections
    if load_balancer:
        await load_balancer.close()
        logger.info("Load balancer shut down")


app = FastAPI(title="Tool Server Load Balancer", lifespan=lifespan)


@app.post("/execute", response_model=CodeResponse)
async def execute_code(request: Request) -> CodeResponse:
    """Forward execute request to backends with load balancing."""
    if load_balancer is None:
        raise HTTPException(status_code=503, detail="Load balancer not initialized")

    try:
        request_data = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse request JSON: {e}")
        return CodeResponse(
            output="",
            error=f"Invalid JSON in request: {str(e)}",
            success=False,
        )

    result = await load_balancer.forward_request(request_data)
    return result


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if load_balancer is None:
        return JSONResponse(status_code=503, content={"status": "not_ready"})

    # Reset expired exclusions before health check
    for backend in load_balancer.backends:
        backend.reset_if_expired()

    healthy_count = sum(1 for b in load_balancer.backends if b.is_healthy)
    return {
        "status": "healthy",
        "total_backends": len(load_balancer.backends),
        "healthy_backends": healthy_count,
        "backends": [
            {
                "url": b.base_url,
                "healthy": b.is_healthy,
                "failure_count": b.failure_count,
            }
            for b in load_balancer.backends
        ],
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Tool Server Load Balancer",
        "endpoints": {
            "POST /execute": "Forward code execution request to backends",
            "GET /health": "Check load balancer and backend health status",
        },
    }

