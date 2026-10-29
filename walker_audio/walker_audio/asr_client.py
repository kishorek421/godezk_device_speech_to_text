"""Robust Remote ASR Client with Circuit Breaker, Connect/Read Timeouts, and Diagnostic Logging."""

from __future__ import annotations

import enum
import logging
import os
import time
from typing import Any, Optional
import requests

logger = logging.getLogger(__name__)


class CircuitState(str, enum.Enum):
    CLOSED = "CLOSED"          # Normal operation, requests allowed
    OPEN = "OPEN"              # Server failing, fast-fail without network call
    HALF_OPEN = "HALF_OPEN"    # Probing server with a single canary request


class AsrCircuitBreaker:
    """In-memory circuit breaker protecting the pipeline from network hangs and request storms."""

    def __init__(self, failure_threshold: int = 2, cooldown_seconds: float = 15.0):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.last_state_change = time.time()
        self.last_error_reason = ""

    def allow_request(self) -> bool:
        now = time.time()
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if now - self.last_state_change >= self.cooldown_seconds:
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = now
                logger.info("[CIRCUIT-BREAKER] Cooldown expired; entering HALF_OPEN to probe ASR server.")
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            # Allow only one probe request in half-open state
            return True
        return False

    def record_success(self) -> None:
        self.failure_count = 0
        self.last_error_reason = ""
        if self.state != CircuitState.CLOSED:
            logger.info("[CIRCUIT-BREAKER] Probe succeeded. Circuit is now CLOSED (healthy).")
            self.state = CircuitState.CLOSED
            self.last_state_change = time.time()

    def record_failure(self, reason: str) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.last_error_reason = reason

        if self.state == CircuitState.HALF_OPEN:
            logger.warning("[CIRCUIT-BREAKER] Probe failed (%s). Re-opening circuit for %ss.", reason, self.cooldown_seconds)
            self.state = CircuitState.OPEN
            self.last_state_change = time.time()
        elif self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
            logger.warning("[CIRCUIT-BREAKER] %d consecutive failures (%s). Opening circuit for %ss.", self.failure_count, reason, self.cooldown_seconds)
            self.state = CircuitState.OPEN
            self.last_state_change = time.time()

    def get_status(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_error_reason": self.last_error_reason,
            "cooldown_remaining_sec": max(0.0, round(self.cooldown_seconds - (time.time() - self.last_state_change), 1)) if self.state == CircuitState.OPEN else 0.0,
        }


class RemoteAsrClient:
    """Client for remote Canary-Qwen GPU transcription with strict timeouts and circuit breaker."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        connect_timeout: float = 3.0,
        interim_read_timeout: float = 5.0,
        final_read_timeout: float = 15.0,
        failure_threshold: int = 3,
        cooldown_seconds: float = 15.0,
    ):
        self.base_url = (base_url or os.getenv("REMOTE_GPU_API", "http://124.123.89.162:8001")).rstrip("/")
        self.connect_timeout = connect_timeout
        self.interim_read_timeout = interim_read_timeout
        self.final_read_timeout = final_read_timeout
        self.circuit_breaker = AsrCircuitBreaker(
            failure_threshold=failure_threshold,
            cooldown_seconds=cooldown_seconds,
        )

    def diagnose_connection(self) -> dict[str, Any]:
        """Diagnoses connectivity distinguishing TCP handshake vs HTTP application responsiveness."""
        import urllib.parse
        parsed = urllib.parse.urlparse(self.base_url)
        host = parsed.hostname or "124.123.89.162"
        port = parsed.port or 8001

        diag = {
            "target": self.base_url,
            "host": host,
            "port": port,
            "tcp_connect": False,
            "tcp_latency_ms": None,
            "http_responsive": False,
            "http_status": None,
            "http_latency_ms": None,
            "error": None,
        }

        # 1. Probe TCP socket
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.connect_timeout)
        t0 = time.perf_counter()
        try:
            err = s.connect_ex((host, port))
            tcp_lat = (time.perf_counter() - t0) * 1000
            diag["tcp_connect"] = (err == 0)
            diag["tcp_latency_ms"] = round(tcp_lat, 1)
        except Exception as exc:
            diag["error"] = f"TCP socket error: {exc}"
        finally:
            s.close()

        # 2. Probe HTTP GET /health
        if diag["tcp_connect"]:
            t1 = time.perf_counter()
            try:
                resp = requests.get(
                    f"{self.base_url}/health",
                    timeout=(self.connect_timeout, 2.5),
                )
                http_lat = (time.perf_counter() - t1) * 1000
                diag["http_responsive"] = resp.ok
                diag["http_status"] = resp.status_code
                diag["http_latency_ms"] = round(http_lat, 1)
            except requests.exceptions.ConnectTimeout:
                diag["error"] = "HTTP ConnectTimeout"
            except requests.exceptions.ReadTimeout:
                diag["error"] = "HTTP ReadTimeout (server accepted TCP but failed to respond with HTTP bytes)"
            except requests.exceptions.ConnectionError as exc:
                diag["error"] = f"HTTP ConnectionError: {exc}"
            except Exception as exc:
                diag["error"] = f"HTTP request exception: {exc}"

        return diag

    def transcribe(
        self,
        wav_bytes: bytes,
        mode: str = "salm",
        domain: str = "general",
        dict_enabled: bool = False,
        is_interim: bool = False,
    ) -> tuple[str, float, str]:
        """Transcribes audio WAV bytes via Canary-Qwen API.
        
        Returns:
            (transcript, latency_sec, status_str)
            status_str in: 'ok', 'circuit_open', 'connect_timeout', 'read_timeout', 'connection_error', 'http_error'
        """
        if not self.circuit_breaker.allow_request():
            logger.debug("[ASR-CLIENT] Fast-failing request: circuit breaker is OPEN.")
            return "", 0.0, "circuit_open"

        read_timeout = self.interim_read_timeout if is_interim else self.final_read_timeout
        t0 = time.perf_counter()

        try:
            resp = requests.post(
                f"{self.base_url}/v1/transcribe",
                files={"audio": ("live_chunk.wav", wav_bytes, "audio/wav")},
                data={"mode": mode, "domain": domain, "dictionary_enabled": str(dict_enabled)},
                timeout=(self.connect_timeout, read_timeout),
            )
            lat = time.perf_counter() - t0

            if resp.ok:
                data = resp.json()
                raw_text = data.get("original_transcript") or data.get("transcript") or ""
                self.circuit_breaker.record_success()
                return raw_text, lat, "ok"
            else:
                reason = f"http_{resp.status_code}"
                self.circuit_breaker.record_failure(reason)
                return "", lat, reason

        except requests.exceptions.ConnectTimeout:
            lat = time.perf_counter() - t0
            self.circuit_breaker.record_failure("connect_timeout")
            logger.warning("[ASR-CLIENT] Connect timeout after %.2fs to %s", lat, self.base_url)
            return "", lat, "connect_timeout"

        except requests.exceptions.ReadTimeout:
            lat = time.perf_counter() - t0
            self.circuit_breaker.record_failure("read_timeout")
            logger.warning("[ASR-CLIENT] Read timeout after %.2fs on %s request", lat, "interim" if is_interim else "final")
            return "", lat, "read_timeout"

        except requests.exceptions.ConnectionError as exc:
            lat = time.perf_counter() - t0
            self.circuit_breaker.record_failure("connection_error")
            logger.warning("[ASR-CLIENT] Connection error after %.2fs: %s", lat, exc)
            return "", lat, "connection_error"

        except Exception as exc:
            lat = time.perf_counter() - t0
            self.circuit_breaker.record_failure("unexpected_error")
            logger.error("[ASR-CLIENT] Unexpected error after %.2fs: %s", lat, exc)
            return "", lat, "unexpected_error"
