"""Round-based multi-criteria RAPTOR (arrival time x transfers) with journey reconstruction."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .network import Network


@dataclass(frozen=True)
class Leg:
    kind: str
    trip_id: str | None
    route_id: str | None
    route_short: str | None
    from_stop: str
    to_stop: str
    depart: int
    arrive: int


@dataclass(frozen=True)
class Journey:
    legs: tuple[Leg, ...]

    @property
    def n_rides(self) -> int:
        return sum(
            1
            for leg in self.legs
            if leg.kind == "ride"
        )

    @property
    def transfers(self) -> int:
        return max(
            0,
            self.n_rides - 1,
        )

    @property
    def arrival(self) -> int:
        return self.legs[-1].arrive

    @property
    def departure(self) -> int:
        return self.legs[0].depart


@dataclass(frozen=True)
class PatternGroup:
    route_id: str
    stop_ids: tuple[str, ...]
    trips: tuple


def pattern_groups(
    net: Network,
) -> dict[
    tuple[str, tuple[str, ...]],
    PatternGroup,
]:
    """
    Group trips by route and exact stop sequence.

    The grouping is cached on the Network because it is reused for every
    RAPTOR query on the same feed.
    """
    cached = getattr(
        net,
        "_pattern_cache",
        None,
    )

    if cached is not None:
        return cached

    grouped: dict[
        tuple[str, tuple[str, ...]],
        list,
    ] = {}

    for route_id, trip_list in net.trips_by_route.items():
        for trip in trip_list:
            stop_ids = tuple(
                stop_time.stop_id
                for stop_time in trip.stop_times
            )

            key = (
                route_id,
                stop_ids,
            )

            grouped.setdefault(
                key,
                [],
            ).append(
                trip
            )

    result: dict[
        tuple[str, tuple[str, ...]],
        PatternGroup,
    ] = {}

    for (
        route_id,
        stop_ids,
    ), trip_list in grouped.items():

        sorted_trips = sorted(
            trip_list,
            key=lambda trip: (
                trip.stop_times[0].departure,
                trip.trip_id,
            ),
        )

        result[
            (
                route_id,
                stop_ids,
            )
        ] = PatternGroup(
            route_id=route_id,
            stop_ids=stop_ids,
            trips=tuple(
                sorted_trips
            ),
        )

    net._pattern_cache = result  # type: ignore[attr-defined]

    return result


def stop_to_patterns(
    groups: dict[
        tuple[str, tuple[str, ...]],
        PatternGroup,
    ],
) -> dict[
    str,
    list[tuple[str, tuple[str, ...]]],
]:
    result: dict[
        str,
        list[tuple[str, tuple[str, ...]]],
    ] = {}

    for key, group in groups.items():
        for stop_id in group.stop_ids:
            result.setdefault(
                stop_id,
                [],
            ).append(
                key
            )

    return result


@dataclass
class RaparResult:
    pareto: dict[int, int]
    journeys: dict[int, Journey]
    explored_rounds: int


def run_raptor(
    net: Network,
    origin: str,
    destination: str,
    dep_time: int,
    max_rounds: int = 4,
) -> RaparResult:
    """
    Compute Pareto-optimal journeys by number of rides and arrival time.

    Walking policy
    --------------
    Exactly one direct walking/transfer edge is allowed:

    * before the first transit ride, and
    * after each transit ride.

    Newly reached walking stops are NOT recursively expanded through
    additional walking edges during the same transfer phase.

    Path snapshots are stored together with each arrival label. This is
    important because a mutable predecessor table can accidentally replace
    a ride predecessor with a walking predecessor and reconstruct multiple
    consecutive walks even when relaxation itself was non-recursive.
    """
    if origin == destination:
        return RaparResult(
            pareto={},
            journeys={},
            explored_rounds=0,
        )

    if (
        origin not in net.stops
        or destination not in net.stops
    ):
        raise KeyError(
            f"unknown stop(s): {origin} / {destination}"
        )

    groups = pattern_groups(
        net
    )

    stop_patterns = stop_to_patterns(
        groups
    )

    INF = float(
        "inf"
    )

    # labels[k][stop] = earliest arrival using exactly k rides.
    labels: dict[
        int,
        dict[str, float],
    ] = {
        0: {
            origin: float(
                dep_time
            ),
        }
    }

    # paths[k][stop] = immutable path corresponding to labels[k][stop].
    paths: dict[
        int,
        dict[str, tuple[Leg, ...]],
    ] = {
        0: {
            origin: (),
        }
    }

    def relax_footpaths(
        round_no: int,
        seeds: dict[str, float],
    ) -> set[str]:
        """
        Relax exactly one walking edge from every seed.

        Crucially, both the seed times and seed paths are snapshotted before
        any output label/path is updated. Therefore, if walking from A reaches
        B and B is itself another seed, updating B cannot cause B's new walked
        path to be used for another walk during this same relaxation.
        """
        seed_times = dict(
            seeds
        )

        seed_paths = {
            stop_id: paths[
                round_no
            ][stop_id]
            for stop_id in seeds
            if stop_id
            in paths[
                round_no
            ]
        }

        newly: dict[
            str,
            float,
        ] = {}

        new_paths: dict[
            str,
            tuple[Leg, ...],
        ] = {}

        for source, source_time in seed_times.items():
            source_path = (
                seed_paths.get(
                    source
                )
            )

            if source_path is None:
                continue

            for (
                target,
                walk_seconds,
            ) in net.footpaths.get(
                source,
                (),
            ):
                candidate = (
                    source_time
                    + walk_seconds
                )

                current = labels[
                    round_no
                ].get(
                    target,
                    INF,
                )

                candidate_current = (
                    newly.get(
                        target,
                        INF,
                    )
                )

                if (
                    candidate >= current
                    or candidate >= candidate_current
                ):
                    continue

                leg = Leg(
                    kind="walk",
                    trip_id=None,
                    route_id=None,
                    route_short=None,
                    from_stop=source,
                    to_stop=target,
                    depart=int(
                        source_time
                    ),
                    arrive=int(
                        candidate
                    ),
                )

                newly[
                    target
                ] = candidate

                new_paths[
                    target
                ] = (
                    source_path
                    + (
                        leg,
                    )
                )

        for target, arrival in newly.items():
            labels[
                round_no
            ][target] = arrival

            paths[
                round_no
            ][target] = new_paths[
                target
            ]

        return set(
            newly
        )

    # ------------------------------------------------------------------
    # Round 0: origin plus one optional access walk
    # ------------------------------------------------------------------

    initial_seeds = {
        origin: float(
            dep_time
        )
    }

    initial_walk_reached = (
        relax_footpaths(
            0,
            initial_seeds,
        )
    )

    marked: dict[
        str,
        float,
    ] = {
        origin: labels[
            0
        ][origin]
    }

    for stop_id in initial_walk_reached:
        marked[
            stop_id
        ] = labels[
            0
        ][stop_id]

    final_round = 0

    # ------------------------------------------------------------------
    # Transit rounds
    # ------------------------------------------------------------------

    for k in range(
        1,
        max_rounds + 1,
    ):
        labels[
            k
        ] = {}

        paths[
            k
        ] = {}

        if not marked:
            break

        previous_labels = (
            labels[
                k - 1
            ]
        )

        previous_paths = (
            paths[
                k - 1
            ]
        )

        relevant_patterns: set[
            tuple[str, tuple[str, ...]]
        ] = set()

        for stop_id in marked:
            relevant_patterns.update(
                stop_patterns.get(
                    stop_id,
                    (),
                )
            )

        # These are ride arrivals before applying the optional one-edge
        # walking transfer.
        improved: dict[
            str,
            float,
        ] = {}

        for key in relevant_patterns:
            group = (
                groups[
                    key
                ]
            )

            stop_ids = (
                group.stop_ids
            )

            number_of_stops = len(
                stop_ids
            )

            for trip in group.trips:
                board_position = None

                # Find the first stop on this trip that can be boarded from
                # a label produced in the previous ride round.
                for position in range(
                    number_of_stops - 1
                ):
                    source = (
                        stop_ids[
                            position
                        ]
                    )

                    available = (
                        previous_labels.get(
                            source,
                            INF,
                        )
                    )

                    departure = (
                        trip.stop_times[
                            position
                        ].departure
                    )

                    if (
                        available
                        <= departure
                    ):
                        board_position = (
                            position
                        )
                        break

                if (
                    board_position
                    is None
                ):
                    continue

                board_stop = (
                    stop_ids[
                        board_position
                    ]
                )

                board_path = (
                    previous_paths.get(
                        board_stop
                    )
                )

                if (
                    board_path
                    is None
                ):
                    continue

                board_departure = (
                    trip.stop_times[
                        board_position
                    ].departure
                )

                for position in range(
                    board_position + 1,
                    number_of_stops,
                ):
                    target = (
                        stop_ids[
                            position
                        ]
                    )

                    if (
                        target
                        == board_stop
                    ):
                        continue

                    arrival = (
                        trip.stop_times[
                            position
                        ].arrival
                    )

                    current_this_round = (
                        labels[
                            k
                        ].get(
                            target,
                            INF,
                        )
                    )

                    best_previous_round = min(
                        labels[
                            previous_round
                        ].get(
                            target,
                            INF,
                        )
                        for previous_round
                        in range(
                            0,
                            k,
                        )
                    )

                    # If an equal-or-better arrival already exists using
                    # fewer rides, this candidate cannot be Pareto useful.
                    if (
                        arrival
                        >= best_previous_round
                    ):
                        continue

                    # Within the same ride count, keep only the earliest
                    # arrival.
                    if (
                        arrival
                        >= current_this_round
                    ):
                        continue

                    route_short = None

                    if (
                        group.route_id
                        in net.routes
                    ):
                        route = (
                            net.routes[
                                group.route_id
                            ]
                        )

                        route_short = (
                            route.short_name
                            or route.long_name
                        )

                    ride_leg = Leg(
                        kind="ride",
                        trip_id=trip.trip_id,
                        route_id=group.route_id,
                        route_short=route_short,
                        from_stop=board_stop,
                        to_stop=target,
                        depart=int(
                            board_departure
                        ),
                        arrive=int(
                            arrival
                        ),
                    )

                    labels[
                        k
                    ][target] = (
                        arrival
                    )

                    paths[
                        k
                    ][target] = (
                        board_path
                        + (
                            ride_leg,
                        )
                    )

                    improved[
                        target
                    ] = (
                        arrival
                    )

        if not improved:
            final_round = (
                k - 1
            )
            break

        # --------------------------------------------------------------
        # Exactly one optional walking transfer after this ride round.
        # --------------------------------------------------------------

        walk_reached = (
            relax_footpaths(
                k,
                improved,
            )
        )

        # The next transit round may board from either:
        #   * a stop reached directly by this ride, or
        #   * a stop reached by one walking transfer after this ride.
        #
        # Use the final label after walk relaxation because a direct walk
        # from another ride arrival may improve one of the ride-arrival
        # stops itself.
        next_marked_stops = (
            set(
                improved
            )
            | walk_reached
        )

        marked = {
            stop_id: labels[
                k
            ][stop_id]
            for stop_id
            in next_marked_stops
            if stop_id
            in labels[
                k
            ]
        }

        final_round = k

    # ------------------------------------------------------------------
    # Pareto frontier
    # ------------------------------------------------------------------

    pareto: dict[
        int,
        int,
    ] = {}

    for k in range(
        final_round + 1
    ):
        arrival = (
            labels.get(
                k,
                {},
            ).get(
                destination,
                INF,
            )
        )

        if (
            arrival
            == INF
        ):
            continue

        dominated = False

        for (
            existing_rides,
            existing_arrival,
        ) in pareto.items():
            if (
                existing_rides
                <= k
                and existing_arrival
                <= arrival
            ):
                dominated = True
                break

        if dominated:
            continue

        pareto = {
            existing_rides: existing_arrival
            for (
                existing_rides,
                existing_arrival,
            )
            in pareto.items()
            if not (
                k
                <= existing_rides
                and arrival
                <= existing_arrival
            )
        }

        pareto[
            k
        ] = int(
            arrival
        )

    # ------------------------------------------------------------------
    # Journey materialization
    # ------------------------------------------------------------------

    journeys: dict[
        int,
        Journey,
    ] = {}

    for rides in pareto:
        path = (
            paths.get(
                rides,
                {},
            ).get(
                destination
            )
        )

        if not path:
            continue

        # Defensive validation. A path produced by this implementation
        # should never contain consecutive walks.
        previous_kind = None
        valid = True

        for leg in path:
            if (
                previous_kind
                == "walk"
                and leg.kind
                == "walk"
            ):
                valid = False
                break

            previous_kind = (
                leg.kind
            )

        if not valid:
            continue

        if (
            path[
                0
            ].from_stop
            != origin
        ):
            continue

        if (
            path[
                -1
            ].to_stop
            != destination
        ):
            continue

        journeys[
            rides
        ] = Journey(
            tuple(
                path
            )
        )

    # Keep the Pareto map and reconstructed-journey map consistent.
    pareto = {
        rides: arrival
        for (
            rides,
            arrival,
        )
        in pareto.items()
        if rides
        in journeys
    }

    return RaparResult(
        pareto=pareto,
        journeys=journeys,
        explored_rounds=final_round,
    )


def fmt_hhmm(
    seconds_after_midnight: int,
) -> str:
    base = (
        dt.datetime(
            2000,
            1,
            1,
        )
        + dt.timedelta(
            seconds=seconds_after_midnight
        )
    )

    return base.strftime(
        "%H:%M"
    )
