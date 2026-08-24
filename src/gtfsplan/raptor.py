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
        return sum(1 for lg in self.legs if lg.kind == "ride")

    @property
    def transfers(self) -> int:
        return max(0, self.n_rides - 1)

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
    trips: tuple  # tuple[Trip]


def pattern_groups(net: Network) -> dict[tuple[str, tuple[str, ...]], PatternGroup]:
    """Group trips by (route_id, exact stop sequence); cached on the network."""
    cached = getattr(net, "_pattern_cache", None)
    if cached is not None:
        return cached
    grouped: dict[tuple[str, tuple[str, ...]], list] = {}
    for route_id, tl in net.trips_by_route.items():
        for tr in tl:
            key = (route_id, tuple(st.stop_id for st in tr.stop_times))
            grouped.setdefault(key, []).append(tr)
    out = {}
    for (rid, stops_seq), tl in grouped.items():
        tl_sorted = sorted(tl, key=lambda t: t.stop_times[0].departure)
        out[(rid, stops_seq)] = PatternGroup(rid, stops_seq, tuple(tl_sorted))
    net._pattern_cache = out  # type: ignore[attr-defined]
    return out


def stop_to_patterns(
    groups: dict[tuple[str, tuple[str, ...]], PatternGroup],
) -> dict[str, list[tuple[str, tuple[str, ...]]]]:
    idx: dict[str, list] = {}
    for key, g in groups.items():
        for s in g.stop_ids:
            idx.setdefault(s, []).append(key)
    return idx


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
    """Compute Pareto-optimal (rides, arrival) labels from origin to destination."""
    if origin == destination:
        return RaparResult({}, {}, 0)
    if origin not in net.stops or destination not in net.stops:
        raise KeyError(f"unknown stop(s): {origin} / {destination}")

    groups = pattern_groups(net)
    stop_pat = stop_to_patterns(groups)

    INF = float("inf")
    labels: dict[int, dict[str, float]] = {0: {origin: float(dep_time)}}
    parents: dict[tuple[int, str], tuple] = {}

    def relax_footpaths(round_no: int, seeds: dict[str, float]) -> set[str]:
        newly = {}
        frontier = dict(seeds)
        seen = set(seeds)
        while frontier:
            nxt = {}
            for s, t in frontier.items():
                for tgt, secs in net.footpaths.get(s, ()):
                    cand = t + secs
                    cur = labels[round_no].get(tgt, INF)
                    if cand < cur and cand < newly.get(tgt, INF):
                        newly[tgt] = cand
                        parents[(round_no, tgt)] = ("walk", (round_no, s), s, tgt, int(t), int(cand))
                        if tgt not in seen:
                            nxt[tgt] = cand
                            seen.add(tgt)
            frontier = nxt
        labels[round_no].update(newly)
        return set(newly)

    marked: dict[str, float] = {origin: float(dep_time)}
    relax_footpaths(0, marked)

    reached_rounds: dict[int, int] = {}
    final_round = 0
    for k in range(1, max_rounds + 1):
        labels[k] = {}
        prev_labels = labels[k - 1]
        if not marked:
            break
        relevant: set = set()
        for s in marked:
            relevant.update(stop_pat.get(s, ()))

        improved: dict[str, float] = {}
        for key in relevant:
            g = groups[key]
            sts = g.stop_ids
            n = len(sts)
            for tr in g.trips:
                board_p = None
                for p in range(n - 1):
                    dep = tr.stop_times[p].departure
                    src = sts[p]
                    avail = prev_labels.get(src, marked.get(src, INF))
                    if avail <= dep:
                        board_p = p
                        break
                if board_p is None:
                    continue
                board_dep = tr.stop_times[board_p].departure
                board_stop = sts[board_p]
                for q in range(board_p + 1, n):
                    arr = tr.stop_times[q].arrival
                    tgt = sts[q]
                    if tgt == board_stop:
                        continue
                    cur_all = labels[k].get(tgt, INF)
                    best_anywhere = min(
                        labels[r].get(tgt, INF) for r in range(0, k + 1)
                    )
                    if arr >= best_anywhere and arr >= cur_all:
                        continue
                    if arr < cur_all:
                        labels[k][tgt] = arr
                        parents[(k, tgt)] = (
                            "ride",
                            (k - 1, board_stop),
                            tr.trip_id,
                            g.route_id,
                            board_stop,
                            tgt,
                            int(board_dep),
                            int(arr),
                        )
                        if arr < improved.get(tgt, INF):
                            improved[tgt] = arr
                    elif tgt == destination and tgt not in reached_rounds:
                        parents.setdefault(
                            (k, tgt),
                            (
                                "ride",
                                (k - 1, board_stop),
                                tr.trip_id,
                                g.route_id,
                                board_stop,
                                tgt,
                                int(board_dep),
                                int(arr),
                            ),
                        )
                        labels[k].setdefault(tgt, arr)

        if not improved:
            final_round = k - 1
            break
        walk_reached = relax_footpaths(k, improved)
        marked = dict(improved)
        for s in walk_reached:
            marked.setdefault(s, labels[k][s])
        final_round = k
        dest_arr = labels[k].get(destination, INF)
        if dest_arr < INF:
            reached_rounds[k] = int(dest_arr)

    pareto: dict[int, int] = {}
    for k in range(final_round + 1):
        arr = labels.get(k, {}).get(destination, INF)
        if arr == INF:
            continue
        dominated = False
        for kk, aa in pareto.items():
            if kk <= k and aa <= arr:
                dominated = True
                break
        if not dominated:
            pareto = {
                kk: aa for kk, aa in pareto.items() if not (k <= kk and arr <= aa)
            }
            pareto[k] = int(arr)

    journeys: dict[int, Journey] = {}
    for k in pareto:
        legs: list[Leg] = []
        cur = (k, destination)
        while cur != (0, origin):
            rec = parents.get(cur)
            if rec is None:
                break
            if rec[0] == "walk":
                _, pk, s, t, dep, arr = rec
                legs.append(
                    Leg("walk", None, None, None, s, t, dep, arr)
                )
                cur = pk
            else:
                _, pk, tid, rid, bs, ts, dep, arr = rec
                legs.append(
                    Leg(
                        "ride", tid, rid,
                        net.routes.get(rid).short_name or net.routes.get(rid).long_name
                        if rid in net.routes
                        else None,
                        bs, ts, dep, arr,
                    )
                )
                cur = pk
        legs.reverse()
        if legs and legs[0].from_stop == origin and legs[-1].to_stop == destination:
            journeys[k] = Journey(tuple(legs))

    return RaparResult(pareto=pareto, journeys=journeys, explored_rounds=final_round)


def fmt_hhmm(seconds_after_midnight: int) -> str:
    base = dt.datetime(2000, 1, 1) + dt.timedelta(seconds=seconds_after_midnight)
    return base.strftime("%H:%M")
