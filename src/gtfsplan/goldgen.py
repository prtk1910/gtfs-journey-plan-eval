"""Stratified origin-destination query generation and gold journey computation."""

from __future__ import annotations

import datetime as dt
import json
import random
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from .network import Network, load_network
from .raptor import Journey, run_raptor


DEFAULT_DAY_OFFSET_DAYS = 0


def next_weekday(
    start: dt.date,
    weekday: int = 2,
) -> dt.date:
    d = start

    while d.weekday() != weekday:
        d += dt.timedelta(days=1)

    return d


def haversine_m(
    net: Network,
    a: str,
    b: str,
) -> float | None:
    return net.haversine_m(
        a,
        b,
    )


def _norm_stop_name(
    name: str,
) -> str:
    """
    Normalize public stop names exactly like the itinerary scorer.

    The model sees rider-facing stop names rather than GTFS stop IDs, so
    gold endpoint semantics must use the same public-name equivalence.
    """
    text = (
        unicodedata.normalize(
            "NFKD",
            name or "",
        )
        .encode(
            "ascii",
            "ignore",
        )
        .decode()
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text.lower(),
    )

    return text.strip()


def _public_name_to_ids(
    net: Network,
) -> dict[
    str,
    tuple[str, ...],
]:
    grouped: dict[
        str,
        list[str],
    ] = {}

    for stop_id, stop in net.stops.items():
        key = _norm_stop_name(
            stop.name
        )

        if key:
            grouped.setdefault(
                key,
                [],
            ).append(
                stop_id
            )

    return {
        key: tuple(
            sorted(
                ids
            )
        )
        for key, ids
        in grouped.items()
    }


def sample_od_pairs(
    net: Network,
    n_per_stratum: int,
    seed: int,
    min_km: float = 0.8,
    max_km: float = 18.0,
) -> list[
    tuple[str, str, int]
]:
    """
    Sample representative GTFS OD pairs in four distance bands.

    Gold routing later expands each sampled endpoint to every GTFS stop ID
    with the same normalized rider-facing stop name.
    """
    rng = random.Random(
        seed
    )

    candidates = [
        stop_id
        for stop_id, routes
        in net.stop_routes.items()
        if len(routes) >= 1
    ]

    if len(candidates) < 2:
        raise RuntimeError(
            "feed too small for OD sampling"
        )

    bands = [
        (
            min_km,
            max_km / 4,
        ),
        (
            max_km / 4,
            max_km / 2,
        ),
        (
            max_km / 2,
            3 * max_km / 4,
        ),
        (
            3 * max_km / 4,
            max_km,
        ),
    ]

    out: list[
        tuple[str, str, int]
    ] = []

    seen_ids: set[
        tuple[str, str]
    ] = set()

    seen_public: set[
        tuple[str, str]
    ] = set()

    counts = [
        0,
        0,
        0,
        0,
    ]

    tries = 0

    while (
        sum(counts)
        < 4 * n_per_stratum
        and tries < 200_000
    ):
        tries += 1

        o = rng.choice(
            candidates
        )

        d = rng.choice(
            candidates
        )

        if (
            o == d
            or (
                o,
                d,
            )
            in seen_ids
        ):
            continue

        o_name = _norm_stop_name(
            net.stops[
                o
            ].name
        )

        d_name = _norm_stop_name(
            net.stops[
                d
            ].name
        )

        public_key = (
            o_name,
            d_name,
        )

        # Avoid asking for a journey from a public stop name to itself
        # just because GTFS represents the place with several IDs.
        if (
            not o_name
            or not d_name
            or o_name
            == d_name
            or public_key
            in seen_public
        ):
            continue

        dist_m = haversine_m(
            net,
            o,
            d,
        )

        if dist_m is None:
            continue

        km = (
            dist_m
            / 1000.0
        )

        if not (
            min_km
            <= km
            <= max_km
        ):
            continue

        for band, (
            lo,
            hi,
        ) in enumerate(
            bands
        ):
            if (
                lo <= km < hi
                and counts[
                    band
                ]
                < n_per_stratum
            ):
                seen_ids.add(
                    (
                        o,
                        d,
                    )
                )

                seen_public.add(
                    public_key
                )

                out.append(
                    (
                        o,
                        d,
                        band,
                    )
                )

                counts[
                    band
                ] += 1

                break

    return out


def departure_times(
    n: int,
    seed: int,
    start_s: int = 7 * 3600,
    end_s: int = 19 * 3600,
) -> list[int]:
    rng = random.Random(
        seed + 1
    )

    return [
        rng.randint(
            start_s,
            end_s,
        )
        for _ in range(
            n
        )
    ]


@dataclass
class GoldItem:
    feed: str
    day: str

    # Representative sampled GTFS IDs, retained for deterministic identity
    # and distance-stratum bookkeeping.
    origin: str
    destination: str

    # Rider-facing endpoint names shown to the model.
    origin_name: str
    destination_name: str

    dist_band: int
    dep_time: int

    # All GTFS IDs equivalent to the public endpoint names.
    origin_ids: list[str]
    destination_ids: list[str]

    # Public-name Pareto frontier:
    # number of rides -> earliest arrival.
    pareto: dict[
        str,
        int,
    ]

    # Fewest-rides Pareto journey, retained for backward compatibility.
    best_journey: dict | None

    # Earliest-arriving Pareto journey.
    # Equal-arrival ties prefer fewer rides.
    fastest_journey: dict | None


def journey_to_dict(
    journey: Journey | None,
) -> dict | None:
    if journey is None:
        return None

    return {
        "n_rides": journey.n_rides,
        "transfers": journey.transfers,
        "departure": journey.departure,
        "arrival": journey.arrival,
        "legs": [
            asdict(
                leg
            )
            for leg
            in journey.legs
        ],
    }


def _journey_tie_key(
    journey: Journey,
) -> tuple:
    """
    Stable tie break when equivalent endpoint IDs produce equal labels.
    """
    return tuple(
        (
            leg.kind,
            leg.from_stop,
            leg.to_stop,
            leg.depart,
            leg.arrive,
            leg.route_id or "",
            leg.trip_id or "",
        )
        for leg
        in journey.legs
    )


def _public_endpoint_gold(
    net: Network,
    origin_ids: tuple[str, ...],
    destination_ids: tuple[str, ...],
    dep_time: int,
    max_rounds: int = 4,
) -> tuple[
    dict[int, int],
    dict[int, Journey],
]:
    """
    Compute exact gold under rider-facing endpoint semantics.

    The traveler may start at any GTFS stop ID represented by the displayed
    origin name and may finish at any GTFS stop ID represented by the
    displayed destination name.

    For each ride count, retain the earliest arrival across all endpoint-ID
    combinations, then Pareto-prune points dominated by fewer rides.
    """
    best_arrival_by_rides: dict[
        int,
        int,
    ] = {}

    best_journey_by_rides: dict[
        int,
        Journey,
    ] = {}

    for origin_id in origin_ids:
        for destination_id in destination_ids:
            if (
                origin_id
                == destination_id
            ):
                continue

            result = run_raptor(
                net,
                origin_id,
                destination_id,
                dep_time,
                max_rounds=max_rounds,
            )

            for (
                rides,
                arrival,
            ) in result.pareto.items():
                journey = (
                    result.journeys.get(
                        rides
                    )
                )

                if journey is None:
                    continue

                old_arrival = (
                    best_arrival_by_rides.get(
                        rides
                    )
                )

                if (
                    old_arrival is None
                    or arrival
                    < old_arrival
                ):
                    best_arrival_by_rides[
                        rides
                    ] = int(
                        arrival
                    )

                    best_journey_by_rides[
                        rides
                    ] = journey

                elif (
                    arrival
                    == old_arrival
                ):
                    old_journey = (
                        best_journey_by_rides[
                            rides
                        ]
                    )

                    if (
                        _journey_tie_key(
                            journey
                        )
                        < _journey_tie_key(
                            old_journey
                        )
                    ):
                        best_journey_by_rides[
                            rides
                        ] = journey

    pareto: dict[
        int,
        int,
    ] = {}

    journeys: dict[
        int,
        Journey,
    ] = {}

    best_arrival_with_fewer_rides: (
        int
        | None
    ) = None

    for rides in sorted(
        best_arrival_by_rides
    ):
        arrival = (
            best_arrival_by_rides[
                rides
            ]
        )

        if (
            best_arrival_with_fewer_rides
            is not None
            and best_arrival_with_fewer_rides
            <= arrival
        ):
            # Dominated by a journey with fewer rides and an
            # equal-or-earlier arrival.
            continue

        pareto[
            rides
        ] = arrival

        journeys[
            rides
        ] = (
            best_journey_by_rides[
                rides
            ]
        )

        best_arrival_with_fewer_rides = (
            arrival
        )

    return (
        pareto,
        journeys,
    )


def generate_gold(
    root: Path,
    slug: str,
    day: dt.date,
    n_per_stratum: int,
    seed: int,
    index_db: Path | None = None,
    max_walk_m: float = 0.0,
    time_window: tuple[int, int] = (
        5 * 3600,
        22 * 3600,
    ),
    out_path: Path | None = None,
) -> Path:
    index_db = (
        Path(
            index_db
        )
        if index_db
        else (
            root
            / "artifacts"
            / "index"
            / f"{slug}.sqlite"
        )
    )

    net = load_network(
        index_db,
        day,
        max_walk_m=max_walk_m,
        time_window=time_window,
    )

    if (
        net.n_trips_active
        == 0
    ):
        raise RuntimeError(
            f"{slug}: no active trips on {day}"
        )

    pairs = sample_od_pairs(
        net,
        n_per_stratum,
        seed,
    )

    deps = departure_times(
        len(
            pairs
        ),
        seed,
    )

    name_to_ids = (
        _public_name_to_ids(
            net
        )
    )

    out_path = (
        out_path
        or (
            root
            / "artifacts"
            / "gold"
            / f"{slug}-{day.isoformat()}.jsonl"
        )
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    solved = 0

    with open(
        out_path,
        "w",
    ) as fh:
        for (
            (
                representative_origin,
                representative_destination,
                band,
            ),
            dep_time,
        ) in zip(
            pairs,
            deps,
        ):
            origin_name = (
                net.stops[
                    representative_origin
                ].name
            )

            destination_name = (
                net.stops[
                    representative_destination
                ].name
            )

            origin_ids = (
                name_to_ids.get(
                    _norm_stop_name(
                        origin_name
                    ),
                    (
                        representative_origin,
                    ),
                )
            )

            destination_ids = (
                name_to_ids.get(
                    _norm_stop_name(
                        destination_name
                    ),
                    (
                        representative_destination,
                    ),
                )
            )

            (
                public_pareto,
                public_journeys,
            ) = _public_endpoint_gold(
                net,
                origin_ids,
                destination_ids,
                dep_time,
                max_rounds=4,
            )

            if public_pareto:
                fewest_rides_k = min(
                    public_pareto
                )

                fastest_k = min(
                    public_pareto,
                    key=lambda k: (
                        public_pareto[
                            k
                        ],
                        k,
                    ),
                )

                best_journey = (
                    journey_to_dict(
                        public_journeys.get(
                            fewest_rides_k
                        )
                    )
                )

                fastest_journey = (
                    journey_to_dict(
                        public_journeys.get(
                            fastest_k
                        )
                    )
                )

            else:
                best_journey = None
                fastest_journey = None

            item = GoldItem(
                feed=slug,
                day=day.isoformat(),
                origin=representative_origin,
                destination=representative_destination,
                origin_name=origin_name,
                destination_name=destination_name,
                dist_band=band,
                dep_time=dep_time,
                origin_ids=list(
                    origin_ids
                ),
                destination_ids=list(
                    destination_ids
                ),
                pareto={
                    str(
                        rides
                    ): arrival
                    for (
                        rides,
                        arrival,
                    )
                    in sorted(
                        public_pareto.items()
                    )
                },
                best_journey=best_journey,
                fastest_journey=fastest_journey,
            )

            fh.write(
                json.dumps(
                    asdict(
                        item
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )

            if public_pareto:
                solved += 1

    print(
        f"[{slug}] gold: "
        f"{solved}/{len(pairs)} "
        "reachable public-name pairs "
        f"-> {out_path.name}"
    )

    return out_path
