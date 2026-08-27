"""Schedule-aware itinerary scoring and strict JSON elicitation."""

from __future__ import annotations

import difflib
import json
import re
import sqlite3
import threading
import unicodedata
from dataclasses import asdict, dataclass, field

from .network import resolve_active_services


ITIN_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "transit_itinerary",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "legs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": [
                                    "ride",
                                    "walk",
                                ],
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


def _norm(
    value: str,
) -> str:
    value = unicodedata.normalize(
        "NFKD",
        value or "",
    ).encode(
        "ascii",
        "ignore",
    ).decode()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value.lower(),
    )

    return value.strip()


def _parse_hhmm(
    value: str,
) -> int | None:
    """
    Convert a model HH:MM timestamp to the beginning of that minute.

    Because the model only emits minute-resolution timestamps, a claimed
    time HH:MM corresponds to the GTFS interval:

        [HH:MM:00, HH:MM:59]
    """
    match = re.fullmatch(
        r"(\d{1,2}):(\d{2})(?::\d{2})?",
        (value or "").strip(),
    )

    if not match:
        return None

    hour = int(
        match.group(1)
    )

    minute = int(
        match.group(2)
    )

    if (
        hour < 0
        or minute < 0
        or minute > 59
    ):
        return None

    return (
        hour * 3600
        + minute * 60
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

        self.lock = threading.Lock()

        self.net = net

        # A public stop name may correspond to several GTFS platform IDs.
        # Keep all of them rather than selecting one arbitrary platform.
        self._name_to_ids: dict[
            str,
            list[str],
        ] = {}

        self._stop_name_by_id: dict[
            str,
            str,
        ] = {}

        for (
            stop_id,
            stop_name,
        ) in self.con.execute(
            """
            SELECT stop_id, stop_name
            FROM stops
            """
        ):
            normalized = _norm(
                stop_name or ""
            )

            if normalized:
                self._name_to_ids.setdefault(
                    normalized,
                    [],
                ).append(
                    stop_id
                )

            self._stop_name_by_id[
                stop_id
            ] = (
                stop_name
                or ""
            )

        for ids in self._name_to_ids.values():
            ids.sort()

        # A rider-facing route label may map to more than one internal
        # route_id. Keep all compatible IDs.
        self._route_by_norm: dict[
            str,
            list[str],
        ] = {}

        for (
            route_id,
            short_name,
            long_name,
        ) in self.con.execute(
            """
            SELECT
                route_id,
                route_short_name,
                route_long_name
            FROM routes
            """
        ):
            labels = {
                _norm(
                    short_name
                    or ""
                ),
                _norm(
                    long_name
                    or ""
                ),
            }

            for label in labels:
                if not label:
                    continue

                self._route_by_norm.setdefault(
                    label,
                    [],
                ).append(
                    route_id
                )

        for ids in self._route_by_norm.values():
            ids.sort()

        self.active_services = tuple(
            sorted(
                resolve_active_services(
                    self.con,
                    self.net.day,
                )
            )
        )

    def close(
        self,
    ):
        self.con.close()

    # ------------------------------------------------------------------
    # Stop resolution
    # ------------------------------------------------------------------

    def resolve_stops(
        self,
        name: str,
    ) -> tuple[
        list[str],
        bool,
    ]:
        """
        Resolve a public stop name to all matching GTFS stop/platform IDs.

        Returns:
            (candidate_stop_ids, used_fuzzy_match)
        """
        normalized = _norm(
            name
        )

        if not normalized:
            return [], False

        exact = self._name_to_ids.get(
            normalized
        )

        if exact:
            return (
                list(
                    exact
                ),
                False,
            )

        candidates = (
            difflib.get_close_matches(
                normalized,
                self._name_to_ids.keys(),
                n=1,
                cutoff=0.85,
            )
        )

        if candidates:
            return (
                list(
                    self._name_to_ids[
                        candidates[0]
                    ]
                ),
                True,
            )

        return [], False

    def resolve_stop(
        self,
        name: str,
    ) -> tuple[
        str | None,
        bool,
    ]:
        """
        Compatibility helper for older callers.

        The main evaluator uses resolve_stops() because choosing one
        arbitrary platform would create false negatives.
        """
        ids, fuzzy = (
            self.resolve_stops(
                name
            )
        )

        return (
            ids[0]
            if ids
            else None,
            fuzzy,
        )

    # ------------------------------------------------------------------
    # Route resolution
    # ------------------------------------------------------------------

    def resolve_routes(
        self,
        label: str,
    ) -> list[str]:
        normalized = _norm(
            label
        )

        if not normalized:
            return []

        exact = (
            self._route_by_norm.get(
                normalized
            )
        )

        if exact:
            return list(
                exact
            )

        matches: set[
            str
        ] = set()

        for (
            route_label,
            route_ids,
        ) in self._route_by_norm.items():
            if (
                normalized in route_label
                or route_label in normalized
            ):
                matches.update(
                    route_ids
                )

        return sorted(
            matches
        )

    def resolve_route(
        self,
        label: str,
    ) -> str | None:
        """
        Compatibility helper for older callers.
        """
        ids = (
            self.resolve_routes(
                label
            )
        )

        return (
            ids[0]
            if ids
            else None
        )

    def public_name(
        self,
        stop_id: str,
    ) -> str:
        return (
            self._stop_name_by_id.get(
                stop_id,
                "",
            )
        )

    # ------------------------------------------------------------------
    # SQL helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _placeholders(
        values,
    ) -> str:
        return ",".join(
            "?"
            for _ in values
        )

    # ------------------------------------------------------------------
    # Strict ride audit
    # ------------------------------------------------------------------

    def _strict_ride_match(
        self,
        route_ids: list[str],
        from_ids: list[str],
        to_ids: list[str],
        depart_minute: int | None,
        arrive_minute: int | None,
    ):
        """
        Match the claimed leg against one real active-service trip.

        Model output only has HH:MM precision, so a displayed minute is
        accepted when the GTFS timestamp lies anywhere in that minute.
        """
        if (
            not route_ids
            or not from_ids
            or not to_ids
            or depart_minute is None
            or arrive_minute is None
            or not self.active_services
        ):
            return None

        from_ph = (
            self._placeholders(
                from_ids
            )
        )

        to_ph = (
            self._placeholders(
                to_ids
            )
        )

        route_ph = (
            self._placeholders(
                route_ids
            )
        )

        service_ph = (
            self._placeholders(
                self.active_services
            )
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
              AND t.route_id IN ({route_ph})
              AND t.service_id IN ({service_ph})
              AND a.stop_sequence < b.stop_sequence
              AND a.departure_s BETWEEN ? AND ?
              AND b.arrival_s BETWEEN ? AND ?
            ORDER BY
                a.departure_s,
                b.arrival_s,
                a.trip_id
            LIMIT 1
        """

        params = (
            *from_ids,
            *to_ids,
            *route_ids,
            *self.active_services,
            depart_minute,
            depart_minute + 59,
            arrive_minute,
            arrive_minute + 59,
        )

        with self.lock:
            return self.con.execute(
                sql,
                params,
            ).fetchone()

    # ------------------------------------------------------------------
    # Lenient ride audit
    # ------------------------------------------------------------------

    def _lenient_ride_match(
        self,
        route_ids: list[str],
        from_ids: list[str],
        to_ids: list[str],
        depart_minute: int | None,
        arrive_minute: int | None,
        window_s: int = 1800,
    ):
        """
        Test whether the claimed route connects the claimed stops on an
        active scheduled trip within a broader temporal window.

        This separates structural route-chain correctness from exact
        minute-level timetable fidelity.
        """
        if (
            not route_ids
            or not from_ids
            or not to_ids
            or not self.active_services
        ):
            return None

        from_ph = (
            self._placeholders(
                from_ids
            )
        )

        to_ph = (
            self._placeholders(
                to_ids
            )
        )

        route_ph = (
            self._placeholders(
                route_ids
            )
        )

        service_ph = (
            self._placeholders(
                self.active_services
            )
        )

        conditions = [
            f"a.stop_id IN ({from_ph})",
            f"b.stop_id IN ({to_ph})",
            f"t.route_id IN ({route_ph})",
            f"t.service_id IN ({service_ph})",
            "a.stop_sequence < b.stop_sequence",
        ]

        params: list = [
            *from_ids,
            *to_ids,
            *route_ids,
            *self.active_services,
        ]

        if depart_minute is not None:
            conditions.append(
                """
                a.departure_s
                BETWEEN ? AND ?
                """
            )

            params.extend(
                [
                    depart_minute
                    - window_s,
                    depart_minute
                    + 59
                    + window_s,
                ]
            )

        if arrive_minute is not None:
            conditions.append(
                """
                b.arrival_s
                BETWEEN ? AND ?
                """
            )

            params.extend(
                [
                    arrive_minute
                    - window_s,
                    arrive_minute
                    + 59
                    + window_s,
                ]
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
            WHERE {" AND ".join(conditions)}
            ORDER BY
                a.departure_s,
                b.arrival_s,
                a.trip_id
            LIMIT 1
        """

        with self.lock:
            return self.con.execute(
                sql,
                tuple(
                    params
                ),
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
        Accept one direct walking/transfer edge.

        The routing network contains both:

        - explicit GTFS transfer edges
        - generated proximity footpaths

        The geographic fallback handles equivalent public-stop/platform
        representations while respecting the same configured walk limit.

        Importantly, this does not recursively chain walking edges.
        """
        if (
            not from_ids
            or not to_ids
        ):
            return False

        for from_id in from_ids:
            direct_targets = {
                target
                for (
                    target,
                    _seconds,
                )
                in self.net.footpaths.get(
                    from_id,
                    (),
                )
            }

            for to_id in to_ids:
                if (
                    from_id
                    == to_id
                ):
                    return True

                if (
                    to_id
                    in direct_targets
                ):
                    return True

                dist = (
                    self.net.haversine_m(
                        from_id,
                        to_id,
                    )
                )

                if (
                    dist is not None
                    and dist
                    <= max_walk_m
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

    optimality_gap_min: (
        float | None
    ) = None

    stated_arrival: (
        int | None
    ) = None

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

    def to_dict(
        self,
    ):
        return asdict(
            self
        )


def _places_connect(
    previous_ids: list[str],
    previous_name: str,
    current_ids: list[str],
    current_name: str,
) -> bool:
    """
    Test continuity without collapsing duplicate platform IDs.
    """
    if (
        previous_ids
        and current_ids
        and set(
            previous_ids
        ).intersection(
            current_ids
        )
    ):
        return True

    previous_norm = _norm(
        previous_name
    )

    current_norm = _norm(
        current_name
    )

    return (
        bool(
            previous_norm
        )
        and previous_norm
        == current_norm
    )


def score_itinerary(
    auditor: ScheduleAuditor,
    item: dict,
    content: str,
    max_walk_m: float = 300.0,
) -> ScoreResult:
    """
    Score one model-generated itinerary.

    Strict feasibility requires:
    - public-stop continuity
    - valid direct walking transfers
    - real route/stop pairs
    - real active trips
    - scheduled departure and arrival timestamps within the model's
      displayed HH:MM minute

    Lenient feasibility preserves the same structural constraints while
    allowing a broader schedule-time window for ride legs.
    """
    res = ScoreResult()

    try:
        obj = (
            _strip_fences_and_parse(
                content
            )
        )

    except Exception:
        return res

    res.parse_ok = True

    legs = (
        obj.get(
            "legs"
        )
        or []
    )

    res.n_legs = len(
        legs
    )

    if not legs:
        res.empty_itinerary = True
        return res

    origin_name = (
        item.get(
            "origin_name"
        )
        or auditor.public_name(
            item[
                "origin"
            ]
        )
    )

    destination_name = (
        item.get(
            "destination_name"
        )
        or auditor.public_name(
            item[
                "destination"
            ]
        )
    )

    previous_ids = [
        item[
            "origin"
        ]
    ]

    previous_name = (
        origin_name
    )

    # At model-visible HH:MM precision, treat the requested departure
    # as its containing minute.
    previous_arrive_minute = (
        item[
            "dep_time"
        ]
        // 60
        * 60
    )

    chain_ok = True

    strict_ok = True

    lenient_ok = True

    final_to_ids: list[
        str
    ] = []

    final_to_name = ""

    for leg in legs:
        leg_type = (
            leg.get(
                "type"
            )
            or ""
        ).strip().lower()

        from_name = (
            leg.get(
                "from"
            )
            or ""
        )

        to_name = (
            leg.get(
                "to"
            )
            or ""
        )

        (
            from_ids,
            _from_fuzzy,
        ) = auditor.resolve_stops(
            from_name
        )

        (
            to_ids,
            _to_fuzzy,
        ) = auditor.resolve_stops(
            to_name
        )

        if not from_ids:
            res.unresolved_stops += 1

        if not to_ids:
            res.unresolved_stops += 1

        if (
            not from_ids
            or not to_ids
        ):
            chain_ok = False
            strict_ok = False
            lenient_ok = False

        if not _places_connect(
            previous_ids,
            previous_name,
            from_ids,
            from_name,
        ):
            res.bad_transitions += 1

            chain_ok = False

        depart_minute = (
            _parse_hhmm(
                leg.get(
                    "depart",
                    "",
                )
            )
        )

        arrive_minute = (
            _parse_hhmm(
                leg.get(
                    "arrive",
                    "",
                )
            )
        )

        if (
            depart_minute
            is None
            or arrive_minute
            is None
        ):
            res.time_mismatches += 1

            strict_ok = False

        else:
            if (
                depart_minute
                < previous_arrive_minute
            ):
                res.bad_transitions += 1

                chain_ok = False

            if (
                arrive_minute
                < depart_minute
            ):
                res.bad_transitions += 1

                chain_ok = False

        if (
            leg_type
            == "walk"
        ):
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

        elif (
            leg_type
            == "ride"
        ):
            res.n_rides += 1

            route_ids = (
                auditor.resolve_routes(
                    leg.get(
                        "route",
                        "",
                    )
                )
            )

            if not route_ids:
                res.hallucinated_routes += 1

                strict_ok = False
                lenient_ok = False

            else:
                strict_match = (
                    auditor._strict_ride_match(
                        route_ids,
                        from_ids,
                        to_ids,
                        depart_minute,
                        arrive_minute,
                    )
                )

                if (
                    strict_match
                    is None
                ):
                    res.time_mismatches += 1

                    strict_ok = False

                lenient_match = (
                    auditor._lenient_ride_match(
                        route_ids,
                        from_ids,
                        to_ids,
                        depart_minute,
                        arrive_minute,
                    )
                )

                if (
                    lenient_match
                    is None
                ):
                    lenient_ok = False

        else:
            chain_ok = False
            strict_ok = False
            lenient_ok = False

        previous_ids = (
            to_ids
        )

        previous_name = (
            to_name
        )

        if (
            arrive_minute
            is not None
        ):
            previous_arrive_minute = (
                arrive_minute
            )

        final_to_ids = (
            to_ids
        )

        final_to_name = (
            to_name
        )

    res.reaches_dest = (
        item[
            "destination"
        ]
        in final_to_ids
        or (
            bool(
                _norm(
                    final_to_name
                )
            )
            and _norm(
                final_to_name
            )
            == _norm(
                destination_name
            )
        )
    )

    final_arrival = (
        _parse_hhmm(
            legs[
                -1
            ].get(
                "arrive",
                "",
            )
        )
    )

    if (
        final_arrival
        is not None
    ):
        res.stated_arrival = (
            final_arrival
        )

    res.feasible_lenient = (
        chain_ok
        and lenient_ok
        and res.reaches_dest
        and res.unresolved_stops
        == 0
        and res.hallucinated_routes
        == 0
        and res.walks_too_long
        == 0
        and res.bad_transitions
        == 0
    )

    res.feasible_strict = (
        res.feasible_lenient
        and strict_ok
        and res.time_mismatches
        == 0
    )

    pareto_values = [
        int(
            value
        )
        for value
        in item.get(
            "pareto",
            {},
        ).values()
    ]

    if pareto_values:
        pareto_min = min(
            pareto_values
        )

        res.notes[
            "pareto_min"
        ] = pareto_min

        if (
            res.reaches_dest
            and res.stated_arrival
            is not None
        ):
            # Compare at the same minute resolution available to the model.
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

    return res


def _strip_fences_and_parse(
    content: str,
) -> dict:
    text = (
        content
        or ""
    ).strip()

    if text.startswith(
        "```"
    ):
        lines = [
            line
            for line
            in text.splitlines()
            if not line.strip().startswith(
                "```"
            )
        ]

        text = "\n".join(
            lines
        ).strip()

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL,
    )

    if match:
        text = (
            match.group(
                0
            )
        )

    parsed = json.loads(
        text
    )

    if not isinstance(
        parsed,
        dict,
    ):
        raise ValueError(
            "itinerary response must be a JSON object"
        )

    return parsed


def chat_json(
    client,
    messages,
    schema=ITIN_SCHEMA,
    max_tokens=4000,
    validation_retries=3,
) -> dict:
    """
    Strict-JSON elicitation with schema-in-prompt fallback.

    Some models/providers may ignore response_format, so invalid responses
    are retried with the schema explicitly included in the conversation.
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

    def validate(
        text,
    ):
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

    response = client.chat(
        messages,
        max_tokens=max_tokens,
        response_format=schema,
    )

    try:
        return validate(
            response.content
        )

    except Exception:
        pass

    augmented = (
        list(
            messages
        )
        + [
            {
                "role": "assistant",
                "content": (
                    response.content
                ),
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
    )

    error = None

    for _ in range(
        validation_retries
    ):
        response = (
            client.chat(
                augmented,
                max_tokens=max_tokens,
            )
        )

        try:
            return validate(
                response.content
            )

        except Exception as exc:
            error = exc

            augmented.append(
                {
                    "role": "assistant",
                    "content": (
                        response.content
                    ),
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
        f"{error}"
    )
