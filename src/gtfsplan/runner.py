"""Concurrent model evaluation of gold journey queries across context arms."""

from __future__ import annotations

import datetime as dt
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .api import load_client
from .context import FeedContext, build_excerpt
from .goldgen import next_weekday
from .network import load_network
from .querygen import build_messages
from .scoring import ScheduleAuditor, chat_json, score_itinerary

ARMS = ("closed_book", "schedule_excerpt")


def load_gold(root: Path, slug: str) -> list[dict]:
    files = sorted((root / "artifacts" / "gold").glob(f"{slug}-*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no gold file for {slug}; run make gold FEED={slug}")
    items = []
    with open(files[-1]) as fh:
        for line in fh:
            it = json.loads(line)
            if it["pareto"]:
                items.append(it)
    return items


def run_eval(
    root: Path,
    feeds: list[str],
    arms: list[str],
    n_per_feed: int | None = None,
    workers: int = 8,
    seed: int = 11,
) -> Path:
    root = root.resolve()
    client = load_client(root)
    out_path = root / "artifacts" / "results" / "eval.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    if out_path.exists():
        with open(out_path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    if not r.get("error"):
                        done_keys.add((r["feed"], r["origin"], r["destination"],
                                       r["dep_time"], r["arm"]))
                except Exception:
                    pass

    tasks = []
    for slug in feeds:
        items = load_gold(root, slug)
        if n_per_feed:
            items = items[:n_per_feed]
        day = dt.date.fromisoformat(items[0]["day"])
        index_db = root / "artifacts" / "index" / f"{slug}.sqlite"
        net = load_network(index_db, day, max_walk_m=300.0,
                           time_window=(4 * 3600, 23 * 3600))
        auditor = ScheduleAuditor(index_db, net)
        fctx = FeedContext(index_db, day)
        for it in items:
            for arm in arms:
                key = (slug, it["origin"], it["destination"], it["dep_time"], arm)
                if key in done_keys:
                    continue
                excerpt = None
                if arm != "closed_book":
                    excerpt = build_excerpt(index_db, it, arm, ctx=fctx)
                tasks.append({
                    "feed": slug, "item": it, "arm": arm,
                    "messages": build_messages(it, arm, excerpt),
                    "auditor": auditor, "index_db": index_db,
                })
        print(f"[{slug}] queued {sum(1 for t in tasks if t['feed']==slug)} tasks")

    lock = threading.Lock()
    stats = {"done": 0, "parse_fail": 0}

    def work(t):
        it = t["item"]
        finish = getattr(client, "last_finish_reason", "")
        reasoning_len = getattr(client, "last_reasoning_len", 0)
        try:
            obj = chat_json(client, t["messages"], max_tokens=16000)
            content = json.dumps(obj)
        except Exception as e:
            with lock:
                stats["parse_fail"] += 1
            rec = {
                "feed": t["feed"], "arm": t["arm"],
                "origin": it["origin"], "destination": it["destination"],
                "dep_time": it["dep_time"], "dist_band": it["dist_band"],
                "day": it["day"],
                "error": str(e)[:200], "score": {"parse_ok": False},
                "raw": "", "finish_reason": finish,
                "reasoning_len": reasoning_len,
            }
        else:
            sc = score_itinerary(t["auditor"], it, content).to_dict()
            rec = {
                "feed": t["feed"], "arm": t["arm"],
                "origin": it["origin"], "destination": it["destination"],
                "dep_time": it["dep_time"], "dist_band": it["dist_band"],
                "day": it["day"],
                "error": "", "score": sc, "raw": content[:4000],
                "finish_reason": getattr(client, "last_finish_reason", ""),
                "reasoning_len": getattr(client, "last_reasoning_len", 0),
            }
        with lock:
            with open(out_path, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["done"] += 1
            if stats["done"] % 10 == 0:
                print(f"progress: {stats['done']} scored ({stats['parse_fail']} failures)",
                      flush=True)
        return rec

    t0 = time.time()
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, t) for t in tasks]
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc is not None:
                failures += 1
                print(f"TASK FAILURE: {exc}", flush=True)
    print(f"eval complete: {stats['done']} records, {failures} task failures "
          f"in {time.time()-t0:.0f}s -> {out_path}")
    return out_path
