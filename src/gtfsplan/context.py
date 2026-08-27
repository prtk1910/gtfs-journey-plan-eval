"""Context-arm builders: schedule excerpts pulled from the routing index.

Performance-critical: a stop->routes map is precomputed once per feed and reused
for every item; proximity transfer search is vectorized with numpy.
"""

from __future__ import annotations

import datetime as dt
import math
import sqlite3

import numpy as np

from .network import resolve_active_services


def _fmt_min(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    return f"{h:02d}:{rem // 60:02d}"


def _route_label(
    con: sqlite3.Connection,
    route_id: str,
    cache: dict,
) -> str:
    if route_id not in cache:
        row = con.execute(
            "SELECT COALESCE(NULLIF(route_short_name,''), route_long_name) "
            "FROM routes WHERE route_id=?",
            (route_id,),
        ).fetchone()

        cache[route_id] = (
            row[0]
            if row
            else route_id
        )

    return cache[route_id]


def _stop_name(
    con: sqlite3.Connection,
    stop_id: str,
) -> str:
    row = con.execute(
        "SELECT stop_name FROM stops WHERE stop_id=?",
        (stop_id,),
    ).fetchone()

    return (
        row[0]
        if row
        else stop_id
    )


class FeedContext:
    """Precomputed per-feed structures shared across all items."""

    def __init__(
        self,
        index_db,
        day: dt.date,
    ):
        self.con = sqlite3.connect(
            index_db
        )

        self.con.execute(
            "PRAGMA journal_mode=WAL"
        )

        self.day = day

        self.frag, self.args = (
            self._active_sql(
                day
            )
        )

        self.route_cache: dict[
            str,
            str,
        ] = {}

        ph = self.frag
        args = self.args

        self.stop_routes: dict[
            str,
            list[str],
        ] = {}

        for sid, rid in self.con.execute(
            f"""
            SELECT DISTINCT
                st.stop_id,
                t.route_id
            FROM stop_times st
            JOIN trips t
              ON t.trip_id = st.trip_id
            WHERE t.service_id IN ({ph})
            """,
            args,
        ):
            self.stop_routes.setdefault(
                sid,
                [],
            ).append(
                rid
            )

        for key in self.stop_routes:
            self.stop_routes[
                key
            ].sort()

    def _active_sql(
        self,
        day: dt.date,
    ):
        active = sorted(
            resolve_active_services(
                self.con,
                day,
            )
        )

        return (
            ",".join(
                "?" * len(active)
            ),
            tuple(
                active
            ),
        )

    def close(
        self,
    ):
        self.con.close()


def build_excerpt(
    index_db,
    item: dict,
    arm: str,
    ctx: "FeedContext | None" = None,
    window_before_s: int = 1800,
    window_after_s: int = 7200,
    max_transfer_stops: int = 5,
    max_lines: int = 120,
) -> str:
    con = (
        ctx.con
        if ctx is not None
        else sqlite3.connect(
            index_db
        )
    )

    owns = (
        ctx is None
    )

    try:
        if owns:
            con.execute(
                "PRAGMA journal_mode=WAL"
            )

        day = dt.date.fromisoformat(
            item["day"]
        )

        dep = item[
            "dep_time"
        ]

        t0 = (
            dep
            - window_before_s
        )

        t1 = (
            dep
            + window_after_s
        )

        if ctx is not None:
            frag = ctx.frag
            args = ctx.args
            sr = ctx.stop_routes

        else:
            active = sorted(
                resolve_active_services(
                    con,
                    day,
                )
            )

            frag = ",".join(
                "?" * len(active)
            )

            args = tuple(
                active
            )

            sr = {}

        rcache = (
            ctx.route_cache
            if ctx
            else {}
        )

        o = item[
            "origin"
        ]

        d = item[
            "destination"
        ]

        def fmt_route(
            rid: str,
        ) -> str:
            return (
                f"route "
                f"{_route_label(con, rid, rcache)}"
            )

        lines: list[str] = []

        lines.append(
            f"Services departing ORIGIN "
            f"[{_stop_name(con, o)}] "
            f"{_fmt_min(t0)}-"
            f"{_fmt_min(t1)}:"
        )

        for (
            rid,
            dsec,
            head,
        ) in _departures(
            con,
            frag,
            args,
            o,
            t0,
            t1,
            8,
        ):
            lines.append(
                f"  {_fmt_min(dsec)} "
                f"{fmt_route(rid)}"
                + (
                    f" towards {head}"
                    if head
                    else ""
                )
            )

        lines.append(
            f"Services arriving DESTINATION "
            f"[{_stop_name(con, d)}] "
            f"{_fmt_min(t0)}-"
            f"{_fmt_min(t1)}:"
        )

        for (
            rid,
            asec,
        ) in _arrivals(
            con,
            frag,
            args,
            d,
            t0,
            t1,
            8,
        ):
            lines.append(
                f"  {_fmt_min(asec)} "
                f"{fmt_route(rid)}"
            )

        def routes_at(
            stop_id: str,
        ) -> set[str]:
            if ctx is not None:
                return set(
                    sr.get(
                        stop_id,
                        (),
                    )
                )

            rows = con.execute(
                f"""
                SELECT DISTINCT
                    t.route_id
                FROM stop_times st
                JOIN trips t
                  ON t.trip_id = st.trip_id
                WHERE st.stop_id = ?
                  AND t.service_id
                      IN ({frag})
                """,
                (
                    stop_id,
                    *args,
                ),
            ).fetchall()

            return {
                row[0]
                for row in rows
            }

        routes_o = routes_at(
            o
        )

        routes_d = routes_at(
            d
        )

        def stops_of_routes(
            route_ids: set[str],
        ) -> dict[
            str,
            tuple[float, float],
        ]:
            ph = (
                ",".join(
                    "?" * len(
                        route_ids
                    )
                )
                if route_ids
                else "NULL"
            )

            rows = con.execute(
                f"""
                SELECT DISTINCT
                    st.stop_id,
                    s.stop_lat,
                    s.stop_lon
                FROM stop_times st
                JOIN trips t
                  ON t.trip_id = st.trip_id
                JOIN stops s
                  ON s.stop_id = st.stop_id
                WHERE t.route_id IN ({ph})
                  AND t.service_id IN ({frag})
                  AND s.stop_lat IS NOT NULL
                """,
                (
                    *route_ids,
                    *args,
                ),
            ).fetchall()

            return {
                row[0]: (
                    row[1],
                    row[2],
                )
                for row in rows
            }

        so = stops_of_routes(
            routes_o
        )

        sd = stops_of_routes(
            routes_d
        )

        shared_o: dict[
            str,
            str,
        ] = {}

        for sid in sorted(
            (
                so.keys()
                & sd.keys()
            )
            - {
                o,
                d,
            }
        )[
            :max_transfer_stops
        ]:
            shared_o[
                sid
            ] = sid

        if (
            len(shared_o)
            < max_transfer_stops
        ):
            A = [
                (
                    stop,
                    coords,
                )
                for (
                    stop,
                    coords,
                )
                in so.items()
                if (
                    stop not in shared_o
                    and stop
                    not in (
                        o,
                        d,
                    )
                )
            ]

            B = [
                (
                    stop,
                    coords,
                )
                for (
                    stop,
                    coords,
                )
                in sd.items()
                if stop
                not in (
                    o,
                    d,
                )
            ]

            if A and B:
                la = np.array(
                    [
                        coords[0]
                        for _stop, coords
                        in A
                    ]
                )

                loa = np.array(
                    [
                        coords[1]
                        for _stop, coords
                        in A
                    ]
                )

                lb = np.array(
                    [
                        coords[0]
                        for _stop, coords
                        in B
                    ]
                )

                lob = np.array(
                    [
                        coords[1]
                        for _stop, coords
                        in B
                    ]
                )

                lat1 = np.radians(
                    la
                )[:, None]

                lon1 = np.radians(
                    loa
                )[:, None]

                lat2 = np.radians(
                    lb
                )[None, :]

                lon2 = np.radians(
                    lob
                )[None, :]

                dlat = (
                    lat2
                    - lat1
                )

                dlon = (
                    lon2
                    - lon1
                )

                h = (
                    np.sin(
                        dlat / 2
                    )
                    ** 2
                    + np.cos(
                        lat1
                    )
                    * np.cos(
                        lat2
                    )
                    * np.sin(
                        dlon / 2
                    )
                    ** 2
                )

                dists = (
                    6_371_000.0
                    * 2
                    * np.arcsin(
                        np.sqrt(
                            h
                        )
                    )
                )

                # Keep schedule-evidence transfer generation aligned
                # with the 300 m walking policy used by gold/scoring.
                close = np.argwhere(
                    dists <= 300.0
                )

                order = sorted(
                    close.tolist(),
                    key=lambda pair: (
                        dists[
                            pair[0],
                            pair[1],
                        ]
                    ),
                )

                for i, j in order:
                    s1 = A[
                        i
                    ][0]

                    s2 = B[
                        j
                    ][0]

                    if (
                        s1 == o
                        or s2 == d
                        or s1 == d
                        or s2 == o
                    ):
                        continue

                    shared_o.setdefault(
                        s1,
                        s2,
                    )

                    if (
                        len(
                            shared_o
                        )
                        >= max_transfer_stops
                    ):
                        break

        if not shared_o:
            lines.append(
                "No shared or walk-connected stop "
                "between origin-serving and "
                "destination-serving routes was "
                "found in the index."
            )

        for sid_o in list(
            shared_o
        ):
            sid_d = (
                shared_o[
                    sid_o
                ]
            )

            label = _stop_name(
                con,
                sid_o,
            )

            if (
                sid_d
                != sid_o
            ):
                label += (
                    f" (then walk to "
                    f"[{_stop_name(con, sid_d)}])"
                )

            lines.append(
                f"Transfer point [{label}]:"
            )

            seen_any = False

            for (
                rid,
                asec,
            ) in _arrivals(
                con,
                frag,
                args,
                sid_o,
                t0,
                dep + 10800,
                4,
            ):
                if (
                    rid
                    in routes_o
                ):
                    lines.append(
                        f"  arrive "
                        f"{_fmt_min(asec)} "
                        f"{fmt_route(rid)}"
                    )

                    seen_any = True

            for (
                rid,
                dsec,
                head,
            ) in _departures(
                con,
                frag,
                args,
                sid_d,
                t0,
                dep + 12600,
                4,
            ):
                if (
                    rid
                    in routes_d
                ):
                    lines.append(
                        f"  depart "
                        f"{_fmt_min(dsec)} "
                        f"{fmt_route(rid)}"
                        + (
                            f" towards {head}"
                            if head
                            else ""
                        )
                    )

                    seen_any = True

            if not seen_any:
                lines.append(
                    "  (no timed connections in window)"
                )

        def trip_sequence_block(
            stop_id: str,
            direction: str,
            cap: int = 2,
            max_stops: int = 45,
        ) -> None:
            trips = con.execute(
                f"""
                SELECT
                    st.trip_id,
                    st.departure_s
                FROM stop_times st
                JOIN trips t
                  ON t.trip_id = st.trip_id
                WHERE st.stop_id = ?
                  AND st.departure_s
                      BETWEEN ? AND ?
                  AND t.service_id IN ({frag})
                  AND st.departure_s
                      IS NOT NULL
                ORDER BY
                    st.departure_s
                LIMIT {cap}
                """,
                (
                    stop_id,
                    max(
                        t0,
                        dep - 900,
                    ),
                    t1,
                    *args,
                ),
            ).fetchall()

            for (
                tid,
                dsec,
            ) in trips:
                rid_row = con.execute(
                    """
                    SELECT route_id
                    FROM trips
                    WHERE trip_id = ?
                    """,
                    (
                        tid,
                    ),
                ).fetchone()

                if not rid_row:
                    continue

                rid = (
                    rid_row[
                        0
                    ]
                )

                rows = con.execute(
                    """
                    SELECT
                        st.stop_sequence,
                        st.stop_id,
                        st.departure_s,
                        s.stop_name
                    FROM stop_times st
                    JOIN stops s
                      ON s.stop_id = st.stop_id
                    WHERE st.trip_id = ?
                    ORDER BY
                        st.stop_sequence
                    """,
                    (
                        tid,
                    ),
                ).fetchall()

                idx = next(
                    (
                        i
                        for i, row
                        in enumerate(
                            rows
                        )
                        if row[1]
                        == stop_id
                    ),
                    None,
                )

                if (
                    idx is None
                ):
                    continue

                if (
                    direction
                    == "arr"
                ):
                    shown = (
                        rows[
                            :idx
                        ][
                            ::-1
                        ][
                            :max_stops
                        ]
                    )

                    head_line = (
                        f"{fmt_route(rid)} "
                        f"trip arriving DESTINATION at "
                        f"{_fmt_min(rows[idx][2])} "
                        f"(earlier stops, "
                        f"most recent first):"
                    )

                else:
                    shown = (
                        rows[
                            idx + 1:
                        ][
                            :max_stops
                        ]
                    )

                    head_line = (
                        f"{fmt_route(rid)} "
                        f"trip departing ORIGIN at "
                        f"{_fmt_min(dsec)} "
                        f"(subsequent stops):"
                    )

                lines.append(
                    head_line
                )

                for (
                    _seq,
                    sid2,
                    tsec,
                    name,
                ) in shown:
                    others = (
                        len(
                            set(
                                sr.get(
                                    sid2,
                                    [],
                                )
                            )
                            - {
                                rid
                            }
                        )
                        if ctx is not None
                        else ""
                    )

                    tag = (
                        f" [+{others} other routes]"
                        if others
                        else ""
                    )

                    lines.append(
                        f"    "
                        f"{_fmt_min(tsec)} "
                        f"[{name}]"
                        f"{tag}"
                    )

        trip_sequence_block(
            o,
            "dep",
        )

        trip_sequence_block(
            d,
            "arr",
        )

        return "\n".join(
            lines[
                :max_lines
            ]
        )

    finally:
        if owns:
            con.close()


def _departures(
    con,
    sql_frag,
    args,
    stop_id: str,
    t0: int,
    t1: int,
    limit: int,
):
    return con.execute(
        f"""
        SELECT
            r.route_id,
            st.departure_s,
            t.trip_headsign
        FROM stop_times st
        JOIN trips t
          ON t.trip_id = st.trip_id
        JOIN routes r
          ON r.route_id = t.route_id
        WHERE st.stop_id = ?
          AND st.departure_s
              BETWEEN ? AND ?
          AND t.service_id
              IN ({sql_frag})
          AND st.departure_s
              IS NOT NULL
        ORDER BY
            st.departure_s
        LIMIT {limit}
        """,
        (
            stop_id,
            t0,
            t1,
            *args,
        ),
    ).fetchall()


def _arrivals(
    con,
    sql_frag,
    args,
    stop_id: str,
    t0: int,
    t1: int,
    limit: int,
):
    return con.execute(
        f"""
        SELECT
            r.route_id,
            st.arrival_s
        FROM stop_times st
        JOIN trips t
          ON t.trip_id = st.trip_id
        JOIN routes r
          ON r.route_id = t.route_id
        WHERE st.stop_id = ?
          AND st.arrival_s
              BETWEEN ? AND ?
          AND t.service_id
              IN ({sql_frag})
          AND st.arrival_s
              IS NOT NULL
        ORDER BY
            st.arrival_s
        LIMIT {limit}
        """,
        (
            stop_id,
            t0,
            t1,
            *args,
        ),
    ).fetchall()


def build_oracle_complete_schedule(
    index_db,
    item: dict,
    ctx: "FeedContext | None" = None,
    distractors_per_leg: int = 4,
) -> str:
    """
    Build schedule evidence guaranteed internally to contain a feasible
    earliest-arriving journey.

    The RAPTOR fastest journey is used only behind the scenes to identify
    timetable facts that must be present.

    The model is not told which facts belong to the hidden journey.

    For each ride leg:
    - the exact hidden-gold scheduled ride is guaranteed to be present;
    - nearby real trips on the same route and stop pair are included;
    - if the initial local window contains fewer than three distinct
      visible options, the search window is widened adaptively.

    No synthetic timetable facts are generated.

    Ride and walk blocks are sorted independently rather than emitted
    in hidden-gold leg order.
    """
    journey = item.get(
        "fastest_journey"
    )

    if (
        not journey
        or not journey.get(
            "legs"
        )
    ):
        raise ValueError(
            "oracle_complete_schedule "
            "requires fastest_journey"
        )

    con = (
        ctx.con
        if ctx is not None
        else sqlite3.connect(
            index_db
        )
    )

    owns = (
        ctx is None
    )

    try:
        if owns:
            con.execute(
                "PRAGMA journal_mode=WAL"
            )

        day = dt.date.fromisoformat(
            item[
                "day"
            ]
        )

        if (
            ctx is not None
        ):
            frag = (
                ctx.frag
            )

            args = (
                ctx.args
            )

            rcache = (
                ctx.route_cache
            )

        else:
            active = sorted(
                resolve_active_services(
                    con,
                    day,
                )
            )

            frag = ",".join(
                "?"
                * len(
                    active
                )
            )

            args = tuple(
                active
            )

            rcache = {}

        evidence: list[
            str
        ] = [
            (
                "Candidate schedule facts for the requested journey. "
                "Select and combine the appropriate facts; "
                "not every listed option must be used."
            )
        ]

        ride_blocks: list[
            str
        ] = []

        walk_blocks: list[
            str
        ] = []

        # Start locally around the hidden ride. Only widen the
        # timetable search for sparse route/stop pairs.
        search_half_windows = (
            30 * 60,
            60 * 60,
            2 * 60 * 60,
            4 * 60 * 60,
            8 * 60 * 60,
            12 * 60 * 60,
        )

        max_visible_options = max(
            1,
            distractors_per_leg
            + 1,
        )

        preferred_min_options = min(
            3,
            max_visible_options,
        )

        for leg in journey[
            "legs"
        ]:
            kind = leg.get(
                "kind"
            )

            # ------------------------------------------------------
            # Walking facts
            # ------------------------------------------------------

            if (
                kind
                == "walk"
            ):
                from_stop = leg[
                    "from_stop"
                ]

                to_stop = leg[
                    "to_stop"
                ]

                from_name = _stop_name(
                    con,
                    from_stop,
                )

                to_name = _stop_name(
                    con,
                    to_stop,
                )

                walk_seconds = max(
                    0,
                    int(
                        leg[
                            "arrive"
                        ]
                    )
                    - int(
                        leg[
                            "depart"
                        ]
                    ),
                )

                walk_minutes = max(
                    1,
                    int(
                        round(
                            walk_seconds
                            / 60
                        )
                    ),
                )

                walk_blocks.append(
                    (
                        "Walking connection available "
                        f"between [{from_name}] "
                        f"and [{to_name}] "
                        f"(approximately "
                        f"{walk_minutes} "
                        f"{'minute' if walk_minutes == 1 else 'minutes'})."
                    )
                )

                continue

            if (
                kind
                != "ride"
            ):
                continue

            # ------------------------------------------------------
            # Hidden ride information
            # ------------------------------------------------------

            route_id = leg[
                "route_id"
            ]

            from_stop = leg[
                "from_stop"
            ]

            to_stop = leg[
                "to_stop"
            ]

            gold_dep = int(
                leg[
                    "depart"
                ]
            )

            gold_arr = int(
                leg[
                    "arrive"
                ]
            )

            gold_trip_id = (
                leg.get(
                    "trip_id"
                )
            )

            route_name = _route_label(
                con,
                route_id,
                rcache,
            )

            from_name = _stop_name(
                con,
                from_stop,
            )

            to_name = _stop_name(
                con,
                to_stop,
            )

            # ------------------------------------------------------
            # Retrieve real timetable distractors
            # ------------------------------------------------------

            rows_by_trip: dict[
                str,
                tuple,
            ] = {}

            # We retrieve more database rows than will be shown because
            # duplicate visible times/headsigns may collapse.
            query_limit = max(
                20,
                max_visible_options
                * 8,
            )

            for half_window in (
                search_half_windows
            ):
                candidate_rows = (
                    con.execute(
                        f"""
                        SELECT
                            a.trip_id,
                            a.departure_s,
                            b.arrival_s,
                            t.trip_headsign
                        FROM stop_times a
                        JOIN stop_times b
                          ON a.trip_id = b.trip_id
                        JOIN trips t
                          ON t.trip_id = a.trip_id
                        WHERE a.stop_id = ?
                          AND b.stop_id = ?
                          AND a.stop_sequence
                              < b.stop_sequence
                          AND t.route_id = ?
                          AND t.service_id
                              IN ({frag})
                          AND a.departure_s
                              BETWEEN ? AND ?
                          AND a.departure_s
                              IS NOT NULL
                          AND b.arrival_s
                              IS NOT NULL
                        ORDER BY
                            ABS(
                                a.departure_s - ?
                            ),
                            a.departure_s,
                            a.trip_id
                        LIMIT ?
                        """,
                        (
                            from_stop,
                            to_stop,
                            route_id,
                            *args,
                            gold_dep
                            - half_window,
                            gold_dep
                            + half_window,
                            gold_dep,
                            query_limit,
                        ),
                    ).fetchall()
                )

                for row in candidate_rows:
                    rows_by_trip[
                        row[
                            0
                        ]
                    ] = row

                visible_options = {
                    (
                        int(
                            dep_s
                        ),
                        int(
                            arr_s
                        ),
                        head
                        or "",
                    )
                    for (
                        _trip_id,
                        dep_s,
                        arr_s,
                        head,
                    )
                    in rows_by_trip.values()
                }

                if (
                    len(
                        visible_options
                    )
                    >= preferred_min_options
                ):
                    break

            # ------------------------------------------------------
            # Guarantee the exact hidden gold trip is available
            # ------------------------------------------------------

            if (
                gold_trip_id
                and gold_trip_id
                not in rows_by_trip
            ):
                exact = con.execute(
                    f"""
                    SELECT
                        a.trip_id,
                        a.departure_s,
                        b.arrival_s,
                        t.trip_headsign
                    FROM stop_times a
                    JOIN stop_times b
                      ON a.trip_id = b.trip_id
                    JOIN trips t
                      ON t.trip_id = a.trip_id
                    WHERE a.trip_id = ?
                      AND a.stop_id = ?
                      AND b.stop_id = ?
                      AND a.stop_sequence
                          < b.stop_sequence
                      AND t.service_id
                          IN ({frag})
                      AND a.departure_s
                          IS NOT NULL
                      AND b.arrival_s
                          IS NOT NULL
                    LIMIT 1
                    """,
                    (
                        gold_trip_id,
                        from_stop,
                        to_stop,
                        *args,
                    ),
                ).fetchone()

                if exact:
                    rows_by_trip[
                        exact[
                            0
                        ]
                    ] = exact

            all_options = {
                (
                    int(
                        dep_s
                    ),
                    int(
                        arr_s
                    ),
                    head
                    or "",
                )
                for (
                    _trip_id,
                    dep_s,
                    arr_s,
                    head,
                )
                in rows_by_trip.values()
            }

            if not all_options:
                raise RuntimeError(
                    "No timetable rows found "
                    "for oracle ride leg "
                    f"route={route_id} "
                    f"from={from_stop} "
                    f"to={to_stop}"
                )

            gold_options = [
                option
                for option
                in all_options
                if (
                    option[
                        0
                    ]
                    == gold_dep
                    and option[
                        1
                    ]
                    == gold_arr
                )
            ]

            if not gold_options:
                raise RuntimeError(
                    "Oracle evidence failed "
                    "to contain the gold "
                    "scheduled ride: "
                    f"route={route_id}, "
                    f"from={from_stop}, "
                    f"to={to_stop}, "
                    f"depart="
                    f"{_fmt_min(gold_dep)}, "
                    f"arrive="
                    f"{_fmt_min(gold_arr)}"
                )

            gold_option = sorted(
                gold_options
            )[0]

            # ------------------------------------------------------
            # Select real options nearest the hidden ride
            # ------------------------------------------------------

            nearest_options = sorted(
                all_options,
                key=lambda option: (
                    abs(
                        option[
                            0
                        ]
                        - gold_dep
                    ),
                    option[
                        0
                    ],
                    option[
                        1
                    ],
                    option[
                        2
                    ],
                ),
            )

            selected = list(
                nearest_options[
                    :max_visible_options
                ]
            )

            # Defensive guarantee. Never allow later selection logic to
            # accidentally remove the exact hidden timetable fact.
            if (
                gold_option
                not in selected
            ):
                if (
                    len(
                        selected
                    )
                    >= max_visible_options
                ):
                    selected[
                        -1
                    ] = gold_option

                else:
                    selected.append(
                        gold_option
                    )

            # Display chronologically. Do not expose which option was
            # nearest to or selected by the hidden gold computation.
            options = sorted(
                set(
                    selected
                ),
                key=lambda option: (
                    option[
                        0
                    ],
                    option[
                        1
                    ],
                    option[
                        2
                    ],
                ),
            )

            block = [
                (
                    f"Route {route_name}: "
                    f"[{from_name}] -> "
                    f"[{to_name}]"
                )
            ]

            for (
                dep_s,
                arr_s,
                head,
            ) in options:
                suffix = (
                    f" towards {head}"
                    if head
                    else ""
                )

                block.append(
                    (
                        f"  depart "
                        f"{_fmt_min(dep_s)}, "
                        f"arrive "
                        f"{_fmt_min(arr_s)}"
                        f"{suffix}"
                    )
                )

            ride_blocks.append(
                "\n".join(
                    block
                )
            )

        # Deliberately avoid preserving hidden journey leg order.
        ride_blocks.sort()
        walk_blocks.sort()

        if ride_blocks:
            evidence.append(
                "\nCandidate scheduled rides:"
            )

            evidence.extend(
                ride_blocks
            )

        if walk_blocks:
            evidence.append(
                "\nCandidate walking connections:"
            )

            evidence.extend(
                walk_blocks
            )

        return "\n\n".join(
            evidence
        )

    finally:
        if owns:
            con.close()
