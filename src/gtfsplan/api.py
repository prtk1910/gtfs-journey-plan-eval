"""OpenCode Zen client with content-addressed caching and bounded retries."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests


API_URL = os.environ.get(
    "OX_API_URL",
    "https://opencode.ai/zen/v1/chat/completions",
)

DEFAULT_MODEL = "x-preview-f-free"

DEFAULT_MAX_RETRIES = 4
DEFAULT_TIMEOUT_S = 300
INITIAL_RETRY_DELAY_S = 3.0
MAX_RETRY_DELAY_S = 20.0


@dataclass
class CallResult:
    key: str
    model: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    cached: bool
    finish_reason: str = ""
    reasoning: str = ""


class Ledger:
    def __init__(self, path: Path):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.con = sqlite3.connect(
            path,
            check_same_thread=False,
        )

        self.lock = threading.Lock()

        with self.lock:
            self.con.execute(
                """
                CREATE TABLE IF NOT EXISTS calls (
                    key TEXT PRIMARY KEY,
                    model TEXT,
                    request_json TEXT,
                    response_json TEXT,
                    content TEXT,
                    prompt_tokens INT,
                    completion_tokens INT,
                    latency_ms INT,
                    ts TEXT
                )
                """
            )
            self.con.commit()

    def get(
        self,
        key: str,
    ) -> tuple[str, str] | None:
        with self.lock:
            row = self.con.execute(
                """
                SELECT
                    response_json,
                    content
                FROM calls
                WHERE key = ?
                """,
                (key,),
            ).fetchone()

        if row is None:
            return None

        return (
            row[0],
            row[1],
        )

    def put(
        self,
        key: str,
        model: str,
        request_json: str,
        response_json: str,
        content: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
    ) -> None:
        with self.lock:
            self.con.execute(
                """
                INSERT OR REPLACE INTO calls
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?,
                    datetime('now')
                )
                """,
                (
                    key,
                    model,
                    request_json,
                    response_json,
                    content,
                    prompt_tokens,
                    completion_tokens,
                    latency_ms,
                ),
            )

            self.con.commit()

    def stats(self) -> list[tuple]:
        with self.lock:
            return self.con.execute(
                """
                SELECT
                    model,
                    COUNT(*),
                    SUM(prompt_tokens),
                    SUM(completion_tokens)
                FROM calls
                GROUP BY model
                """
            ).fetchall()


class OpenCodeZenClient:
    def __init__(
        self,
        ledger_path: Path,
        model: str | None = None,
    ):
        self.key = os.environ.get(
            "OPENCODE_ZEN_API_KEY",
            "",
        )

        if not self.key:
            raise RuntimeError(
                "OPENCODE_ZEN_API_KEY missing; "
                "set it in the root .env"
            )

        self.model = (
            model
            or os.environ.get(
                "OX_MODEL",
                DEFAULT_MODEL,
            )
        )

        self.ledger = Ledger(
            ledger_path
        )

        # Each evaluation worker gets its own HTTP session and
        # response metadata. This avoids cross-thread state races.
        self._local = threading.local()

        self._headers = {
            "Authorization": (
                f"Bearer {self.key}"
            ),
            "Content-Type": "application/json",
        }

    def _session(
        self,
    ) -> requests.Session:
        session = getattr(
            self._local,
            "session",
            None,
        )

        if session is None:
            session = requests.Session()

            session.headers.update(
                self._headers
            )

            self._local.session = session

        return session

    @property
    def last_finish_reason(
        self,
    ) -> str:
        return getattr(
            self._local,
            "finish_reason",
            "",
        )

    @last_finish_reason.setter
    def last_finish_reason(
        self,
        value: str,
    ) -> None:
        self._local.finish_reason = (
            value or ""
        )

    @property
    def last_reasoning_len(
        self,
    ) -> int:
        return getattr(
            self._local,
            "reasoning_len",
            0,
        )

    @last_reasoning_len.setter
    def last_reasoning_len(
        self,
        value: int,
    ) -> None:
        self._local.reasoning_len = int(
            value or 0
        )

    def _key(
        self,
        payload: dict,
    ) -> str:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
        ).encode(
            "utf-8"
        )

        return hashlib.sha256(
            encoded
        ).hexdigest()

    def _response_metadata(
        self,
        body: dict,
    ) -> tuple[str, str, str]:
        choices = (
            body.get("choices")
            or []
        )

        if not choices:
            return (
                "",
                "",
                "",
            )

        choice = (
            choices[0]
            or {}
        )

        message = (
            choice.get("message")
            or {}
        )

        content = (
            message.get("content")
            or ""
        )

        # Different OpenAI-compatible gateways have used both names.
        reasoning = (
            message.get("reasoning")
            or message.get("reasoning_content")
            or ""
        )

        finish_reason = (
            choice.get("finish_reason")
            or ""
        )

        return (
            content,
            reasoning,
            finish_reason,
        )

    def _set_last_metadata(
        self,
        reasoning: str,
        finish_reason: str,
    ) -> None:
        self.last_finish_reason = (
            finish_reason
        )

        self.last_reasoning_len = len(
            reasoning or ""
        )

    def _retry_delay(
        self,
        delay: float,
        response: requests.Response | None = None,
    ) -> float:
        if response is not None:
            retry_after = (
                response.headers.get(
                    "Retry-After"
                )
            )

            if retry_after:
                try:
                    seconds = float(
                        retry_after
                    )

                    return min(
                        max(
                            seconds,
                            0.0,
                        ),
                        MAX_RETRY_DELAY_S,
                    )
                except ValueError:
                    pass

        return min(
            delay,
            MAX_RETRY_DELAY_S,
        )

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> CallResult:
        # Do not allow a failed call to inherit metadata from the
        # preceding call on the same worker.
        self._set_last_metadata(
            "",
            "",
        )

        payload: dict = {
            "model": self.model,
            "messages": messages,
        }

        if temperature is not None:
            payload[
                "temperature"
            ] = temperature

        if max_tokens is not None:
            payload[
                "max_tokens"
            ] = max_tokens

        if response_format is not None:
            payload[
                "response_format"
            ] = response_format

        key = self._key(
            payload
        )

        cached = self.ledger.get(
            key
        )

        if cached is not None:
            response_json, content = (
                cached
            )

            body = json.loads(
                response_json
            )

            (
                cached_content,
                reasoning,
                finish_reason,
            ) = self._response_metadata(
                body
            )

            if not content:
                content = cached_content

            self._set_last_metadata(
                reasoning,
                finish_reason,
            )

            usage = (
                body.get("usage")
                or {}
            )

            return CallResult(
                key=key,
                model=body.get(
                    "model",
                    self.model,
                ),
                content=content,
                prompt_tokens=usage.get(
                    "prompt_tokens",
                    0,
                ),
                completion_tokens=usage.get(
                    "completion_tokens",
                    0,
                ),
                latency_ms=0,
                cached=True,
                finish_reason=finish_reason,
                reasoning=reasoning,
            )

        if max_retries < 1:
            raise ValueError(
                "max_retries must be >= 1"
            )

        delay = (
            INITIAL_RETRY_DELAY_S
        )

        last_error: Exception | None = (
            None
        )

        session = self._session()

        for attempt in range(
            max_retries
        ):
            started = time.monotonic()

            try:
                response = session.post(
                    API_URL,
                    json=payload,
                    timeout=DEFAULT_TIMEOUT_S,
                )

            except requests.RequestException as exc:
                last_error = exc

                if (
                    attempt + 1
                    >= max_retries
                ):
                    break

                time.sleep(
                    self._retry_delay(
                        delay
                    )
                )

                delay = min(
                    delay * 1.7,
                    MAX_RETRY_DELAY_S,
                )

                continue

            if response.status_code == 200:
                try:
                    body = response.json()

                except ValueError as exc:
                    last_error = RuntimeError(
                        "OpenCode Zen returned "
                        "HTTP 200 with invalid JSON: "
                        f"{exc}"
                    )

                    if (
                        attempt + 1
                        >= max_retries
                    ):
                        break

                    time.sleep(
                        self._retry_delay(
                            delay
                        )
                    )

                    delay = min(
                        delay * 1.7,
                        MAX_RETRY_DELAY_S,
                    )

                    continue

                (
                    content,
                    reasoning,
                    finish_reason,
                ) = self._response_metadata(
                    body
                )

                usage = (
                    body.get("usage")
                    or {}
                )

                latency_ms = int(
                    (
                        time.monotonic()
                        - started
                    )
                    * 1000
                )

                returned_model = body.get(
                    "model",
                    self.model,
                )

                self.ledger.put(
                    key=key,
                    model=returned_model,
                    request_json=json.dumps(
                        payload,
                        sort_keys=True,
                        ensure_ascii=False,
                    ),
                    response_json=json.dumps(
                        body,
                        ensure_ascii=False,
                    ),
                    content=content,
                    prompt_tokens=usage.get(
                        "prompt_tokens",
                        0,
                    ),
                    completion_tokens=usage.get(
                        "completion_tokens",
                        0,
                    ),
                    latency_ms=latency_ms,
                )

                self._set_last_metadata(
                    reasoning,
                    finish_reason,
                )

                return CallResult(
                    key=key,
                    model=returned_model,
                    content=content,
                    prompt_tokens=usage.get(
                        "prompt_tokens",
                        0,
                    ),
                    completion_tokens=usage.get(
                        "completion_tokens",
                        0,
                    ),
                    latency_ms=latency_ms,
                    cached=False,
                    finish_reason=finish_reason,
                    reasoning=reasoning,
                )

            retryable = (
                response.status_code
                in (
                    408,
                    409,
                    429,
                )
                or response.status_code
                >= 500
            )

            if retryable:
                last_error = RuntimeError(
                    "HTTP "
                    f"{response.status_code}: "
                    f"{response.text[:500]}"
                )

                if (
                    attempt + 1
                    >= max_retries
                ):
                    break

                time.sleep(
                    self._retry_delay(
                        delay,
                        response,
                    )
                )

                delay = min(
                    delay * 1.7,
                    MAX_RETRY_DELAY_S,
                )

                continue

            raise RuntimeError(
                "OpenCode Zen HTTP "
                f"{response.status_code}: "
                f"{response.text[:1000]}"
            )

        raise RuntimeError(
            "OpenCode Zen request failed "
            f"after {max_retries} attempts; "
            f"last error: {last_error}"
        )

    def stats(
        self,
    ) -> list[tuple]:
        return self.ledger.stats()


# Backwards-compatible alias in case anything outside runner.py imports
# the old client class directly.
OpenRouterClient = OpenCodeZenClient


def default_ledger_path(
    root: Path,
) -> Path:
    return (
        root
        / "artifacts"
        / "ledger.sqlite"
    )


def load_client(
    root: Path,
) -> OpenCodeZenClient:
    return OpenCodeZenClient(
        default_ledger_path(
            root
        )
    )
