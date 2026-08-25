"""Parse model itinerary JSON and audit each leg against the GTFS schedule.

The evaluator operates at the same abstraction level exposed to the model:

- model times are HH:MM, so strict schedule matching is minute-exact rather
  than second-exact;
- rider-visible stop names may correspond to multiple GTFS stop/platform IDs,
  so all compatible IDs are considered;
- route and stop compatibility are evaluated jointly against actual GTFS trips;
- journey continuity is checked without arbitrarily choosing one platform ID.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
import threading
import unicodedata
from dataclasses import dataclass, field, asdict

from .network import resolve_active_services


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
                            "type": {
                                "type": "string",
                            },
                            "route": {
                                "type": "string",
                            },
                            "from": {
                                "type": "string",
                            },
                            "to": {
                                "type": "string",
                            },
                            "depart": {
                                "type": "string",
                            },
                            "arrive": {
                                "type": "string",
                            },
                            "verified": {
                                "type": "boolean",
                            },
                        },
                        "required": [
                            "type",
                            "route",
                            "from",
                            "to",
                            "depart",
                            "arrive",
                        ],
                    },
                }
            },
            "required": [
                "legs",
            ],
        },
    },
}


def _norm(s: str) -> str:
    s = (
        unicodedata.normalize(
            "NFKD",
            s or "",
        )
        .encode(
            "ascii",
            "ignore",
        )
        .decode()
    )

    s = re.sub(
        r"[^a-z0-9]+",
        " ",
        s.lower(),
    )

    return s.strip()


def _parse_hhmm(
    s: str,
) -> int | None:
    """
    Parse model-visible HH:MM into the first second of that minute.

    GTFS may contain seconds, while the model output schema intentionally
    exposes only minute precision.
    """
    m = re.fullmatch(
        r"(\d{1,2}):(\d{2})(?::\d{2})?",
        (s or "").strip(),
    )

    if not m:
        return None

    hours = int(m.group(1))
    minutes = int(m.group(2))

    if minutes < 0 or minutes > 59:
        return None

    return (
        hours * 3600
        + minutes * 60
    )


def _same_display_minute(
    actual_seconds: int,
    claimed_minute_seconds: int,
) -> bool:
    return (
        int(actual_seconds) // 60
        == int(claimed_minute_seconds) // 60
    )


class ScheduleAuditor:
    def __init__(
        self,
        index_db,
        net,
    ):
        self.con = sqlite3.connect(
            index_db,
            check_same_thread=False,
        )

        self.lock = threading.RLock()
        self.net = net

        self._name_to_id: dict[
            str,
            list[str],
        ] = {}

        with self.lock:
            for sid, name in self.con.execute(
                """
                SELECT
                    stop_id,
                    stop_name
                FROM stops
                """
            ):
                self._name_to_id.setdefault(
                    _norm(name),
                    [],
                ).append(sid)

        for ids in self._name_to_id.values():
            ids.sort()

        self._route_by_norm: dict[
            str,
            list[str],
        ] = {}

        with self.lock:
            route_rows = self.con.execute(
                """
                SELECT
                    route_id,
                    route_short_name,
                    route_long_name
                FROM routes
                """
            ).fetchall()

        for rid, short, long_name in route_rows:
            for label in (
                short,
                long_name,
            ):
                n = _norm(label or "")

                if n:
                    self._route_by_norm.setdefault(
                        n,
                        [],
                    ).append(rid)

        for ids in self._route_by_norm.values():
            ids.sort()

        self._active_cache: dict[
            str,
            tuple[str, ...],
        ] = {}

    def close(self):
        with self.lock:
            self.con.close()

    # ------------------------------------------------------------------
    # Active services
    # ------------------------------------------------------------------

    def active_services(
        self,
        day: str,
    ) -> tuple[str, ...]:
        cached = self._active_cache.get(
            day
        )

        if cached is not None:
            return cached

        date = dt.date.fromisoformat(
            day
        )

        with self.lock:
            active = tuple(
                sorted(
                    resolve_active_services(
                        self.con,
                        date,
                    )
                )
            )

        self._active_cache[day] = active
        return active

    # ------------------------------------------------------------------
    # Stop resolution
    # ------------------------------------------------------------------

    def resolve_stops(
        self,
        name: str,
    ) -> tuple[list[str], bool]:
        """
        Resolve a rider-visible stop name to every compatible GTFS stop ID.

        Returns:
            (candidate_ids, used_fuzzy_match)
        """
        n = _norm(name)

        if not n:
            return [], False

        exact = self._name_to_id.get(
            n
        )

        if exact:
            return list(exact), False

        import difflib

        matches = difflib.get_close_matches(
            n,
            self._name_to_id.keys(),
            n=1,
            cutoff=0.85,
        )

        if not matches:
            return [], False

        return (
            list(
                self._name_to_id[
                    matches[0]
                ]
            ),
            True,
        )

    def resolve_stop(
        self,
        name: str,
    ) -> tuple[str | None, bool]:
        """
        Backward-compatible single-ID resolver.

        New scoring code should prefer resolve_stops().
        """
        ids, fuzzy = self.resolve_stops(
            name
        )

        if not ids:
            return None, fuzzy

        return ids[0], fuzzy

    # ------------------------------------------------------------------
    # Route resolution
    # ------------------------------------------------------------------

    def resolve_routes(
        self,
        label: str,
    ) -> list[str]:
        n = _norm(label)

        if not n:
            return []

        ids = self._route_by_norm.get(
            n
        )

        if ids:
            return sorted(
                set(ids)
            )

        matches: set[str] = set()

        for key, route_ids in (
            self._route_by_norm.items()
        ):
            if (
                n in key
                or key in n
            ):
                matches.update(
                    route_ids
                )

        return sorted(matches)

    def resolve_route(
        self,
        label: str,
    ) -> str | None:
        ids = self.resolve_routes(
            label
        )

        return (
            ids[0]
            if ids
            else None
        )

    # ------------------------------------------------------------------
    # Ride matching
    # ------------------------------------------------------------------

    def _strict_ride_match(
        self,
        route_ids: list[str],
        from_ids: list[str],
        to_ids: list[str],
        dep_minute: int,
        arr_minute: int,
        active_services: tuple[str, ...],
    ) -> tuple | None:
        """
        Find an actual scheduled ride matching the model's displayed minute.

        The model can only emit HH:MM, so a GTFS departure at 11:21:37
        legitimately matches a model claim of 11:21.
        """
        if (
            not route_ids
            or not from_ids
            or not to_ids
            or not active_services
        ):
            return None

        from_ph = ",".join(
            "?" * len(from_ids)
        )

        to_ph = ",".join(
            "?" * len(to_ids)
        )

        route_ph = ",".join(
            "?" * len(route_ids)
        )

        service_ph = ",".join(
            "?" * len(active_services)
        )

        dep_lo = dep_minute
        dep_hi = dep_minute + 59

        arr_lo = arr_minute
        arr_hi = arr_minute + 59

        sql = f"""
            SELECT
                a.trip_id,
                t.route_id,
                a.stop_id,
                b.stop_id,
                a.departure_s,
                b.arrival_s
            FROM stop_times a
            JOIN stop_times b
              ON a.trip_id = b.trip_id
            JOIN trips t
              ON t.trip_id = a.trip_id
            WHERE a.stop_id IN ({from_ph})
              AND b.stop_id IN ({to_ph})
              AND a.stop_sequence < b.stop_sequence
              AND t.route_id IN ({route_ph})
              AND t.service_id IN ({service_ph})
              AND a.departure_s BETWEEN ? AND ?
              AND b.arrival_s BETWEEN ? AND ?
            ORDER BY
                a.departure_s,
                b.arrival_s
            LIMIT 1
        """

        params = (
            *from_ids,
            *to_ids,
            *route_ids,
            *active_services,
            dep_lo,
            dep_hi,
            arr_lo,
            arr_hi,
        )

        with self.lock:
            return self.con.execute(
                sql,
                params,
            ).fetchone()

    def _lenient_ride_match(
        self,
        route_ids: list[str],
        from_ids: list[str],
        to_ids: list[str],
        earliest: int,
        latest: int,
        active_services: tuple[str, ...],
    ) -> tuple | None:
        """
        Find any compatible scheduled ride for the named route and stops.

        This retains the original evaluator's lenient concept: the route chain
        must exist even if the model's stated minute is wrong.
        """
        if (
            not route_ids
            or not from_ids
            or not to_ids
            or not active_services
        ):
            return None

        from_ph = ",".join(
            "?" * len(from_ids)
        )

        to_ph = ",".join(
            "?" * len(to_ids)
        )

        route_ph = ",".join(
            "?" * len(route_ids)
        )

        service_ph = ",".join(
            "?" * len(active_services)
        )

        sql = f"""
            SELECT
                a.trip_id,
                t.route_id,
                a.stop_id,
                b.stop_id,
                a.departure_s,
                b.arrival_s
            FROM stop_times a
            JOIN stop_times b
              ON a.trip_id = b.trip_id
            JOIN trips t
              ON t.trip_id = a.trip_id
            WHERE a.stop_id IN ({from_ph})
              AND b.stop_id IN ({to_ph})
              AND a.stop_sequence < b.stop_sequence
              AND t.route_id IN ({route_ph})
              AND t.service_id IN ({service_ph})
              AND a.departure_s BETWEEN ? AND ?
            ORDER BY
                a.departure_s,
                b.arrival_s
            LIMIT 1
        """

        params = (
            *from_ids,
            *to_ids,
            *route_ids,
            *active_services,
            earliest,
            latest,
        )

        with self.lock:
            return self.con.execute(
                sql,
                params,
            ).fetchone()

    # ------------------------------------------------------------------
    # Walking
    # ------------------------------------------------------------------

    def walk_pair_exists(
        self,
        from_ids: list[str],
        to_ids: list[str],
        max_walk_m: float,
    ) -> bool:
        """
        Accept a walk when at least one compatible platform pair is within
        the allowed walking distance.

        Importantly, zero-distance/same-platform walks are valid rather than
        being converted into a large fallback distance.
        """
        if (
            not from_ids
            or not to_ids
        ):
            return False

        for from_id in from_ids:
            for to_id in to_ids:
                if from_id == to_id:
                    return True

                dist = self.net.haversine_m(
                    from_id,
                    to_id,
                )

                if (
                    dist is not None
                    and dist <= max_walk_m
                ):
                    return True

        return False


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
    notes: dict = field(
        default_factory=dict
    )

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
        obj = _strip_fences_and_parse(
            content
        )
    except Exception:
        return res

    res.parse_ok = True

    legs = (
        obj.get("legs")
        or []
    )

    res.n_legs = len(legs)

    if not legs:
        res.empty_itinerary = True
        return res

    active_services = auditor.active_services(
        item["day"]
    )

    strict_ok = True
    lenient_ok = True

    current_time: int | None = (
        item["dep_time"]
    )

    previous_to_ids: set[str] | None = {
        item["origin"]
    }

    previous_to_name = _norm(
        item.get(
            "origin_name",
            "",
        )
    )

    ambiguous_stop_mentions = 0
    fuzzy_stop_mentions = 0

    final_to_ids: list[str] = []
    final_to_name = ""

    for i, leg in enumerate(legs):
        leg_type = (
            leg.get("type")
            or ""
        ).lower()

        from_text = (
            leg.get("from")
            or ""
        )

        to_text = (
            leg.get("to")
            or ""
        )

        from_ids, from_fuzzy = (
            auditor.resolve_stops(
                from_text
            )
        )

        to_ids, to_fuzzy = (
            auditor.resolve_stops(
                to_text
            )
        )

        if from_fuzzy:
            fuzzy_stop_mentions += 1

        if to_fuzzy:
            fuzzy_stop_mentions += 1

        if len(from_ids) > 1:
            ambiguous_stop_mentions += 1

        if len(to_ids) > 1:
            ambiguous_stop_mentions += 1

        if (
            not from_ids
            or not to_ids
        ):
            res.unresolved_stops += 1
            strict_ok = False
            lenient_ok = False

            previous_to_ids = (
                set(to_ids)
                if to_ids
                else None
            )

            previous_to_name = _norm(
                to_text
            )

            continue

        # --------------------------------------------------------------
        # Journey-chain continuity
        #
        # Prefer actual candidate-ID overlap. If two rider-visible names
        # normalize identically, also treat them as the same public stop
        # abstraction even when GTFS assigns distinct platform IDs.
        # --------------------------------------------------------------

        current_from_name = _norm(
            from_text
        )

        transition_ok = False

        if previous_to_ids is None:
            transition_ok = True

        elif (
            previous_to_ids
            & set(from_ids)
        ):
            transition_ok = True

        elif (
            previous_to_name
            and current_from_name
            and previous_to_name
            == current_from_name
        ):
            transition_ok = True

        if not transition_ok:
            res.bad_transitions += 1
            strict_ok = False
            lenient_ok = False

        dep_t = _parse_hhmm(
            leg.get("depart")
        )

        arr_t = _parse_hhmm(
            leg.get("arrive")
        )

        # Model times are minute-level. Reject obvious backwards time.
        if (
            dep_t is not None
            and current_time is not None
        ):
            # Compare displayed minutes rather than hidden GTFS seconds.
            if (
                dep_t // 60
                < current_time // 60
            ):
                res.bad_transitions += 1
                strict_ok = False
                lenient_ok = False

        # --------------------------------------------------------------
        # Walking leg
        # --------------------------------------------------------------

        if leg_type == "walk":
            walk_ok = (
                auditor.walk_pair_exists(
                    from_ids,
                    to_ids,
                    max_walk_m,
                )
            )

            if not walk_ok:
                res.walks_too_long += 1
                strict_ok = False
                lenient_ok = False

            if arr_t is not None:
                current_time = arr_t
            elif dep_t is not None:
                current_time = dep_t

            previous_to_ids = set(
                to_ids
            )

            previous_to_name = _norm(
                to_text
            )

            final_to_ids = to_ids
            final_to_name = _norm(
                to_text
            )

            continue

        # --------------------------------------------------------------
        # Ride leg
        # --------------------------------------------------------------

        res.n_rides += 1

        route_label = (
            leg.get("route")
            or ""
        )

        route_ids = (
            auditor.resolve_routes(
                route_label
            )
        )

        if not route_ids:
            res.hallucinated_routes += 1
            strict_ok = False
            lenient_ok = False

            if arr_t is not None:
                current_time = arr_t

            previous_to_ids = set(
                to_ids
            )

            previous_to_name = _norm(
                to_text
            )

            final_to_ids = to_ids
            final_to_name = _norm(
                to_text
            )

            continue

        strict_match = None

        if (
            dep_t is not None
            and arr_t is not None
        ):
            strict_match = (
                auditor._strict_ride_match(
                    route_ids,
                    from_ids,
                    to_ids,
                    dep_t,
                    arr_t,
                    active_services,
                )
            )

        if strict_match is None:
            res.time_mismatches += 1
            strict_ok = False

            anchor = (
                dep_t
                if dep_t is not None
                else (
                    current_time
                    if current_time is not None
                    else item["dep_time"]
                )
            )

            lenient_match = (
                auditor._lenient_ride_match(
                    route_ids,
                    from_ids,
                    to_ids,
                    anchor - 1800,
                    anchor + 5400,
                    active_services,
                )
            )

            if lenient_match is None:
                lenient_ok = False

                if arr_t is not None:
                    current_time = arr_t

            else:
                actual_arrival = int(
                    lenient_match[5]
                )

                current_time = (
                    actual_arrival
                )

        else:
            actual_arrival = int(
                strict_match[5]
            )

            # For onward time consistency, retain minute precision because
            # that is all the model was allowed to express.
            current_time = (
                actual_arrival // 60
            ) * 60

        previous_to_ids = set(
            to_ids
        )

        previous_to_name = _norm(
            to_text
        )

        final_to_ids = to_ids
        final_to_name = _norm(
            to_text
        )

        if (
            i
            == len(legs) - 1
            and arr_t is not None
        ):
            res.stated_arrival = arr_t

    # ------------------------------------------------------------------
    # Destination check
    # ------------------------------------------------------------------

    target_id = item[
        "destination"
    ]

    target_name = _norm(
        item.get(
            "destination_name",
            "",
        )
    )

    if (
        target_id in final_to_ids
        or (
            target_name
            and final_to_name
            == target_name
        )
    ):
        res.reaches_dest = True

    if (
        res.reaches_dest
        and res.stated_arrival is None
    ):
        last = legs[-1]

        res.stated_arrival = (
            _parse_hhmm(
                last.get("arrive")
            )
        )

    # ------------------------------------------------------------------
    # Feasibility
    # ------------------------------------------------------------------

    res.feasible_lenient = (
        lenient_ok
        and res.reaches_dest
        and res.unresolved_stops == 0
        and res.hallucinated_routes == 0
        and res.walks_too_long == 0
        and res.bad_transitions == 0
    )

    res.feasible_strict = (
        strict_ok
        and res.feasible_lenient
        and res.time_mismatches == 0
    )

    # ------------------------------------------------------------------
    # Optimality
    #
    # The public output schema is minute-resolution, so compare against the
    # gold frontier at minute resolution too. A model saying 12:08 should
    # not be credited or penalized because the hidden GTFS value is 12:08:08.
    # ------------------------------------------------------------------

    pareto_min = min(
        int(v)
        for v in item[
            "pareto"
        ].values()
    )

    if (
        res.reaches_dest
        and res.stated_arrival is not None
    ):
        stated_minute = (
            res.stated_arrival
            // 60
        )

        gold_minute = (
            pareto_min
            // 60
        )

        res.optimality_gap_min = float(
            stated_minute
            - gold_minute
        )

    res.notes[
        "pareto_min"
    ] = pareto_min

    res.notes[
        "pareto_minute"
    ] = pareto_min // 60

    res.notes[
        "ambiguous_stop_mentions"
    ] = ambiguous_stop_mentions

    res.notes[
        "fuzzy_stop_mentions"
    ] = fuzzy_stop_mentions

    return res


def _strip_fences_and_parse(
    content: str,
) -> dict:
    t = (
        content
        or ""
    ).strip()

    if t.startswith("```"):
        lines = [
            ln
            for ln in t.splitlines()
            if not ln.strip().startswith(
                "```"
            )
        ]

        t = "\n".join(
            lines
        ).strip()

    m = re.search(
        r"\{.*\}",
        t,
        re.DOTALL,
    )

    if m:
        t = m.group(0)

    return json.loads(t)


def chat_json(
    client,
    messages,
    schema=ITIN_SCHEMA,
    max_tokens=4000,
    validation_retries=3,
) -> dict:
    """
    Strict-JSON elicitation with schema-in-prompt fallback.

    ox-alpha may ignore response_format, so malformed output is followed by
    explicit JSON-only repair attempts.
    """
    import jsonschema

    inner = (
        schema.get(
            "json_schema",
            {},
        ).get(
            "schema",
            schema,
        )
    )

    def validate(text):
        obj = (
            _strip_fences_and_parse(
                text
            )
        )

        jsonschema.validate(
            obj,
            inner,
        )

        return obj

    r = client.chat(
        messages,
        max_tokens=max_tokens,
        response_format=schema,
    )

    try:
        return validate(
            r.content
        )
    except Exception:
        pass

    augmented = list(
        messages
    ) + [
        {
            "role": "assistant",
            "content": r.content,
        },
        {
            "role": "user",
            "content": (
                "Reply again with ONLY the raw JSON object "
                "matching this schema; no prose, no markdown:\n"
                + json.dumps(
                    inner,
                    indent=2,
                )
            ),
        },
    ]

    err = None

    for _ in range(
        validation_retries
    ):
        r = client.chat(
            augmented,
            max_tokens=max_tokens,
        )

        try:
            return validate(
                r.content
            )

        except Exception as e:
            err = e

            augmented.append(
                {
                    "role": "assistant",
                    "content": r.content,
                }
            )

            augmented.append(
                {
                    "role": "user",
                    "content": (
                        "Still invalid. "
                        "Output ONLY the raw JSON object."
                    ),
                }
            )

    raise RuntimeError(
        "chat_json failed after retries: "
        f"{err}"
    )
