"""OpenRouter chat client with content-addressed ledger, retries, usage logging."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"


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
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.con.execute(
            """CREATE TABLE IF NOT EXISTS calls (
            key TEXT PRIMARY KEY, model TEXT, request_json TEXT,
            response_json TEXT, content TEXT,
            prompt_tokens INT, completion_tokens INT, latency_ms INT, ts TEXT)"""
        )
        self.con.commit()

    def get(self, key: str) -> tuple[str, str] | None:
        with self.lock:
            row = self.con.execute(
                "SELECT response_json, content FROM calls WHERE key = ?", (key,)
            ).fetchone()
        return (row[0], row[1]) if row else None

    def put(
        self,
        key: str,
        model: str,
        request_json: str,
        response_json: str,
        content: str,
        ptok: int,
        ctok: int,
        lat: int,
    ) -> None:
        with self.lock:
            self.con.execute(
            "INSERT OR REPLACE INTO calls VALUES (?,?,?,?,?,?,?,?,datetime('now'))",
            (key, model, request_json, response_json, content, ptok, ctok, lat),
        )
        self.con.commit()


class OpenRouterClient:
    def __init__(self, ledger_path: Path, model: str | None = None):
        self.key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.key or "replace-with" in self.key:
            raise RuntimeError("OPENROUTER_API_KEY missing; set it in the root .env")
        self.model = model or os.environ.get("OX_MODEL", "stealth/ox-alpha")
        self.ledger = Ledger(ledger_path)
        self.last_finish_reason = ""
        self.last_reasoning_len = 0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.key}",
                "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", ""),
                "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "research-evals"),
                "Content-Type": "application/json",
            }
        )

    def _key(self, payload: dict) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        max_retries: int = 12,
    ) -> CallResult:
        payload: dict = {"model": self.model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        key = self._key(payload)
        hit = self.ledger.get(key)
        if hit is not None:
            resp_json, content = hit
            body = json.loads(resp_json)
            usage = body.get("usage", {})
            fr = (body.get("choices") or [{}])[0].get("finish_reason", "") if body.get("choices") else ""
            return CallResult(key, self.model, content,
                              usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                              0, cached=True, finish_reason=fr)
        delay = 3.0
        last_err = None
        for attempt in range(max_retries):
            t0 = time.monotonic()
            try:
                r = self.session.post(API_URL, data=json.dumps(payload), timeout=300)
            except requests.RequestException as e:
                last_err = e
                time.sleep(delay)
                delay *= 1.7
                continue
            if r.status_code == 200:
                body = r.json()
                content = ""
                reasoning = ""
                finish_reason = ""
                choices = body.get("choices") or []
                if choices:
                    msg = choices[0].get("message", {})
                    content = msg.get("content") or ""
                    reasoning = msg.get("reasoning") or ""
                    finish_reason = choices[0].get("finish_reason") or ""
                usage = body.get("usage", {})
                lat = int((time.monotonic() - t0) * 1000)
                self.ledger.put(key, body.get("model", self.model),
                                json.dumps(payload, sort_keys=True), json.dumps(body),
                                content, usage.get("prompt_tokens", 0),
                                usage.get("completion_tokens", 0), lat)
                self.last_finish_reason = finish_reason
                self.last_reasoning_len = len(reasoning or "")
                return CallResult(key, body.get("model", self.model), content,
                                  usage.get("prompt_tokens", 0),
                                  usage.get("completion_tokens", 0), lat, cached=False,
                                  finish_reason=finish_reason, reasoning=reasoning)
            if r.status_code in (429,) or r.status_code >= 500:
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(delay)
                delay *= 1.7
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        raise RuntimeError(f"max retries exceeded; last error: {last_err}")

    def stats(self) -> list[tuple]:
        return self.ledger.con.execute(
            "SELECT model, COUNT(*), SUM(prompt_tokens), SUM(completion_tokens) FROM calls GROUP BY model"
        ).fetchall()


def default_ledger_path(root: Path) -> Path:
    return root / "artifacts" / "ledger.sqlite"


def load_client(root: Path) -> OpenRouterClient:
    return OpenRouterClient(default_ledger_path(root))
