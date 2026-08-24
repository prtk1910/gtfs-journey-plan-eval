"""Stratified origin-destination query generation and gold journey computation."""

from __future__ import annotations

import datetime as dt
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

from .network import Network, load_network
from .raptor import Journey, run_raptor

DEFAULT_DAY_OFFSET_DAYS = 0


def next_weekday(start: dt.date, weekday: int = 2) -> dt.date:
    d = start
    while d.weekday() != weekday:
        d += dt.timedelta(days=1)
    return d


def haversine_m(net: Network, a: str, b: str) -> float | None:
    return net.haversine_m(a, b)


def sample_od_pairs(
    net: Network,
    n_per_stratum: int,
    seed: int,
    min_km: float = 0.8,
    max_km: float = 18.0,
) -> list[tuple[str, str, int]]:
    """Sample OD pairs stratified into 4 distance bands; returns (origin, dest, band)."""
    rng = random.Random(seed)
    candidates = [s for s, routes in net.stop_routes.items() if len(routes) >= 1]
    if len(candidates) < 2:
        raise RuntimeError("feed too small for OD sampling")
    bands = [(min_km, max_km / 4), (max_km / 4, max_km / 2),
             (max_km / 2, 3 * max_km / 4), (3 * max_km / 4, max_km)]
    out: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str]] = set()
    per_band_target = n_per_stratum
    tries = 0
    counts = [0, 0, 0, 0]
    while sum(counts) < 4 * per_band_target and tries < 200_000:
        tries += 1
        o = rng.choice(candidates)
        d = rng.choice(candidates)
        if o == d or (o, d) in seen:
            continue
        dist_m = haversine_m(net, o, d)
        if dist_m is None:
            continue
        km = dist_m / 1000.0
        if not (min_km <= km <= max_km):
            continue
        for bi, (lo, hi) in enumerate(bands):
            if lo <= km < hi and counts[bi] < per_band_target:
                seen.add((o, d))
                out.append((o, d, bi))
                counts[bi] += 1
                break
    return out


def departure_times(n: int, seed: int, start_s: int = 7 * 3600, end_s: int = 19 * 3600):
    rng = random.Random(seed + 1)
    return [rng.randint(start_s, end_s) for _ in range(n)]


@dataclass
class GoldItem:
    feed: str
    day: str
    origin: str
    destination: str
    origin_name: str
    destination_name: str
    dist_band: int
    dep_time: int
    pareto: dict[str, int]
    best_journey: dict | None


def journey_to_dict(j: Journey | None) -> dict | None:
    if j is None:
        return None
    return {
        "n_rides": j.n_rides,
        "transfers": j.transfers,
        "departure": j.departure,
        "arrival": j.arrival,
        "legs": [asdict(lg) for lg in j.legs],
    }


def generate_gold(
    root: Path,
    slug: str,
    day: dt.date,
    n_per_stratum: int,
    seed: int,
    index_db: Path | None = None,
    max_walk_m: float = 0.0,
    time_window: tuple[int, int] = (5 * 3600, 22 * 3600),
    out_path: Path | None = None,
) -> Path:
    index_db = Path(index_db) if index_db else root / "artifacts" / "index" / f"{slug}.sqlite"
    net = load_network(index_db, day, max_walk_m=max_walk_m, time_window=time_window)
    if net.n_trips_active == 0:
        raise RuntimeError(f"{slug}: no active trips on {day}")
    pairs = sample_od_pairs(net, n_per_stratum, seed)
    deps = departure_times(len(pairs), seed)

    out_path = out_path or root / "artifacts" / "gold" / f"{slug}-{day.isoformat()}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    solved = 0
    with open(out_path, "w") as fh:
        for (o, d, band), t in zip(pairs, deps):
            res = run_raptor(net, o, d, t, max_rounds=4)
            item = GoldItem(
                feed=slug,
                day=day.isoformat(),
                origin=o,
                destination=d,
                origin_name=net.stops[o].name,
                destination_name=net.stops[d].name,
                dist_band=band,
                dep_time=t,
                pareto={str(k): v for k, v in sorted(res.pareto.items())},
                best_journey=journey_to_dict(res.journeys.get(min(res.pareto)) if res.pareto else None),
            )
            if res.pareto:
                solved += 1
            fh.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    print(f"[{slug}] gold: {solved}/{len(pairs)} reachable pairs -> {out_path.name}")
    return out_path
