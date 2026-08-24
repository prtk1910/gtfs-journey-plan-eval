"""Parse model itinerary JSON and audit each leg against the GTFS schedule."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import unicodedata
from dataclasses import dataclass, field, asdict

ITIN_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "itinerary",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "legs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "route": {"type": "string"},
                            "from": {"type": "string"},
                            "to": {"type": "string"},
                            "depart": {"type": "string"},
                            "arrive": {"type": "string"},
                            "verified": {"type": "boolean"},
                        },
                        "required": ["type", "route", "from", "to", "depart", "arrive"],
                    },
                }
            },
            "required": ["legs"],
        },
    },
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return s.strip()


def _parse_hhmm(s: str) -> int | None:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::\d{2})?", (s or "").strip())
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60


class ScheduleAuditor:
    def __init__(self, index_db, net):
        self.con = sqlite3.connect(index_db, check_same_thread=False)
        self.lock = threading.Lock()
        self.net = net
        self._name_to_id: dict[str, list[str]] = {}
        for sid, name in self.con.execute("SELECT stop_id, stop_name FROM stops"):
            self._name_to_id.setdefault(_norm(name), []).append(sid)
        self._route_by_norm: dict[str, list[str]] = {}
        for rid, short, long in self.con.execute(
            "SELECT route_id, route_short_name, route_long_name FROM routes"
        ):
            for label in (short, long):
                n = _norm(label or "")
                if n:
                    self._route_by_norm.setdefault(n, []).append(rid)

    def close(self):
        self.con.close()

    # ---- stop resolution -------------------------------------------------
    def resolve_stop(self, name: str) -> tuple[str | None, bool]:
        """Return (stop_id, fuzzy?). None if unresolvable."""
        n = _norm(name)
        if not n:
            return None, False
        exact = self._name_to_id.get(n)
        if exact:
            return sorted(exact)[0], False
        import difflib

        cands = difflib.get_close_matches(n, self._name_to_id.keys(), n=1, cutoff=0.85)
        if cands:
            return sorted(self._name_to_id[cands[0]])[0], True
        return None, False

    def resolve_route(self, label: str) -> str | None:
        n = _norm(label)
        ids = self._route_by_norm.get(n)
        if ids:
            return sorted(ids)[0]
        for key, ids in self._route_by_norm.items():
            if n and (n in key or key in n):
                return sorted(ids)[0]
        return None

    # ---- leg audit -------------------------------------------------------
    def _ride_exists_exact(self, route_id: str, o: str, d: str, dep: int, arr: int) -> bool:
        row = self.con.execute(
            """SELECT 1
               FROM stop_times a
               JOIN stop_times b ON a.trip_id = b.trip_id
               JOIN trips t ON t.trip_id = a.trip_id
               WHERE a.stop_id=? AND b.stop_id=?
                 AND a.stop_sequence < b.stop_sequence
                 AND a.departure_s=? AND b.arrival_s=?
                 AND t.route_id=?
               LIMIT 1""",
            (o, d, dep, arr, route_id),
        ).fetchone()
        return row is not None

    def _ride_exists_order(self, route_id: str, o: str, d: str, t0: int, t1: int) -> int | None:
        """Any trip of route visiting o then d with departure near window; returns arrival."""
        row = self.con.execute(
            """SELECT b.arrival_s
               FROM stop_times a
               JOIN stop_times b ON a.trip_id = b.trip_id
               JOIN trips t ON t.trip_id = a.trip_id
               WHERE a.stop_id=? AND b.stop_id=?
                 AND a.stop_sequence < b.stop_sequence
                 AND a.departure_s BETWEEN ? AND ?
                 AND t.route_id=?
               ORDER BY b.arrival_s LIMIT 1""",
            (o, d, t0, t1, route_id),
        ).fetchone()
        return row[0] if row else None


@dataclass
class ScoreResult:
    parse_ok: bool = False
    empty_itinerary: bool = False
    reaches_dest: bool = False
    feasible_strict: bool = False
    feasible_lenient: bool = False
    optimality_gap_min: float | None = None
    stated_arrival: int | None = None
    n_legs: int = 0
    n_rides: int = 0
    unresolved_stops: int = 0
    hallucinated_routes: int = 0
    time_mismatches: int = 0
    bad_transitions: int = 0
    walks_too_long: int = 0
    notes: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def score_itinerary(
    auditor: ScheduleAuditor,
    item: dict,
    content: str,
    max_walk_m: float = 800.0,
) -> ScoreResult:
    res = ScoreResult()
    try:
        obj = _strip_fences_and_parse(content)
    except Exception:
        return res
    res.parse_ok = True
    legs = obj.get("legs") or []
    res.n_legs = len(legs)
    if not legs:
        res.empty_itinerary = True
        return res

    resolved_ids: list[str | None] = []
    cur_time: int | None = item["dep_time"]
    prev_stop: str | None = item["origin"]
    chain_ok = True

    for i, leg in enumerate(legs):
        ltype = (leg.get("type") or "").lower()
        o_id, _of = auditor.resolve_stop(leg.get("from", ""))
        d_id, _df = auditor.resolve_stop(leg.get("to", ""))
        if o_id is None or d_id is None:
            res.unresolved_stops += 1
            chain_ok = False
            resolved_ids.append(None)
            continue
        resolved_ids.append(d_id)
        if prev_stop is not None and o_id != prev_stop:
            res.bad_transitions += 1
            chain_ok = False
        dep_t = _parse_hhmm(leg.get("depart"))
        arr_t = _parse_hhmm(leg.get("arrive"))

        if ltype == "walk":
            dist = auditor.net.haversine_m(o_id, d_id) or 10_000.0
            if dist > max_walk_m:
                res.walks_too_long += 1
                chain_ok = False
            if dep_t is not None:
                cur_time = dep_t
            if arr_t is not None:
                cur_time = arr_t
            prev_stop = d_id
            continue

        res.n_rides += 1
        route_label = leg.get("route") or ""
        rid = auditor.resolve_route(route_label)
        if rid is None:
            res.hallucinated_routes += 1
            chain_ok = False
            prev_stop = d_id
            if arr_t is not None:
                cur_time = arr_t
            continue

        arr_eff: int | None = None
        exact = dep_t is not None and arr_t is not None and auditor._ride_exists_exact(
            rid, o_id, d_id, dep_t, arr_t
        )
        if not exact:
            res.time_mismatches += 1
            near = auditor._ride_exists_order(
                rid, o_id, d_id,
                (dep_t - 1800) if dep_t is not None else ((cur_time or item["dep_time"]) - 1800),
                (dep_t + 5400) if dep_t is not None else ((cur_time or item["dep_time"]) + 5400),
            )
            if near is None:
                chain_ok = False
            else:
                arr_eff = near
        else:
            arr_eff = arr_t
        if dep_t is not None and cur_time is not None and dep_t < cur_time - 60:
            res.bad_transitions += 1
            chain_ok = False
        if dep_t is not None:
            cur_time = dep_t
        elif cur_time is None:
            pass
        if arr_eff is not None:
            cur_time = max(cur_time or 0, arr_eff) if cur_time is not None else arr_eff
        prev_stop = d_id

        if i == len(legs) - 1:
            if d_id == item["destination"]:
                res.reaches_dest = True
                res.stated_arrival = arr_t

    last = legs[-1]
    d_id_last, _ = auditor.resolve_stop(last.get("to", ""))
    if d_id_last == item["destination"]:
        res.reaches_dest = True
        if res.stated_arrival is None:
            res.stated_arrival = _parse_hhmm(last.get("arrive"))

    pareto_min = min(int(v) for v in item["pareto"].values())
    res.feasible_lenient = chain_ok and res.reaches_dest and res.unresolved_stops == 0 \
        and res.hallucinated_routes == 0 and res.walks_too_long == 0
    res.feasible_strict = res.feasible_lenient and res.time_mismatches == 0 \
        and res.bad_transitions == 0
    if res.reaches_dest and res.stated_arrival is not None:
        res.optimality_gap_min = round((res.stated_arrival - pareto_min) / 60.0, 2)
    res.notes["pareto_min"] = pareto_min
    return res


def _strip_fences_and_parse(content: str) -> dict:
    t = (content or "").strip()
    if t.startswith("```"):
        lines = [ln for ln in t.splitlines() if not ln.strip().startswith("```")]
        t = "\n".join(lines).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        t = m.group(0)
    return json.loads(t)


def chat_json(client, messages, schema=ITIN_SCHEMA, max_tokens=4000,
              validation_retries=3) -> dict:
    """Strict-JSON elicitation with schema-in-prompt fallback (ox-alpha ignores
    response_format). Mirrors popstats.api.chat_json."""
    import jsonschema

    inner = schema.get("json_schema", {}).get("schema", schema)

    def validate(text):
        obj = _strip_fences_and_parse(text)
        jsonschema.validate(obj, inner)
        return obj

    r = client.chat(messages, max_tokens=max_tokens, response_format=schema)
    try:
        return validate(r.content)
    except Exception:
        pass
    augmented = list(messages) + [
        {"role": "assistant", "content": r.content},
        {"role": "user", "content": (
            "Reply again with ONLY the raw JSON object matching this schema; "
            "no prose, no markdown:\n" + json.dumps(inner, indent=2))},
    ]
    err = None
    for _ in range(validation_retries):
        r = client.chat(augmented, max_tokens=max_tokens)
        try:
            return validate(r.content)
        except Exception as e:
            err = e
            augmented.append({"role": "assistant", "content": r.content})
            augmented.append({"role": "user", "content":
                              "Still invalid. Output ONLY the raw JSON object."})
    raise RuntimeError(f"chat_json failed after retries: {err}")
