"""RAPTOR correctness tests against exhaustive brute-force enumeration."""

from __future__ import annotations

import unittest

from gtfsplan.network import Network, RouteInfo, Stop, StopTime, Trip
from gtfsplan.raptor import run_raptor


def st(seq, stop_id, arr, dep):
    return StopTime(seq, stop_id, arr, dep)


def build_fixture():
    stops = {
        s: Stop(s, f"Stop {s}", 47.0 + i * 0.001, 8.0 + i * 0.001)
        for i, s in enumerate(["A", "B", "C", "D", "E", "F"])
    }
    routes = {
        "RED": RouteInfo("RED", "R", "Red Line", 0),
        "BLUE": RouteInfo("BLUE", "B", "Blue Line", 0),
        "GREEN": RouteInfo("GREEN", "G", "Green Shuttle", 3),
        "YELLOW": RouteInfo("YELLOW", "Y", "Yellow Express", 0),
    }

    def mk(tid, rid, rows):
        return Trip(tid, rid, tuple(st(i + 1, sid, a, d) for i, (sid, a, d) in enumerate(rows)))

    trips_by_route = {
        "RED": [
            mk("red1", "RED", [("A", 28800, 28800), ("B", 28860, 28860), ("C", 28920, 28920), ("D", 28980, 28980)]),
            mk("red2", "RED", [("A", 29400, 29400), ("B", 29460, 29460), ("C", 29520, 29520), ("D", 29580, 29580)]),
        ],
        "BLUE": [
            mk("blue1", "BLUE", [("C", 29100, 29100), ("E", 29160, 29160), ("F", 29220, 29220)]),
            mk("blue2", "BLUE", [("C", 29400, 29400), ("E", 29460, 29460), ("F", 29520, 29520)]),
        ],
        "GREEN": [
            mk("green1", "GREEN", [("B", 28920, 28920), ("E", 29040, 29040)]),
            mk("green2", "GREEN", [("B", 29520, 29520), ("E", 29640, 29640)]),
        ],
        "YELLOW": [
            mk("yellow1", "YELLOW", [("A", 30600, 30600), ("F", 31200, 31200)]),
        ],
    }
    stop_routes: dict[str, set[str]] = {}
    for tl in trips_by_route.values():
        for tr in tl:
            for x in tr.stop_times:
                stop_routes.setdefault(x.stop_id, set()).add(tr.route_id)
    footpaths = {"E": [("F", 300)], "F": [("E", 300)]}
    return Network(
        day=None,
        stops=stops,
        routes=routes,
        trips_by_route=trips_by_route,
        stop_routes=stop_routes,
        footpaths=footpaths,
        n_trips_total=7,
        n_trips_active=7,
    )


def connections(net: Network):
    conns = []
    for tl in net.trips_by_route.values():
        for tr in tl:
            xs = tr.stop_times
            for i in range(len(xs) - 1):
                conns.append(
                    (
                        xs[i].stop_id,
                        xs[i + 1].stop_id,
                        xs[i].departure,
                        xs[i + 1].arrival,
                        tr.trip_id,
                    )
                )
    return conns


def brute_force(net: Network, origin: str, dest: str, dep_time: int, max_rides: int = 4):
    """Earliest arrival at dest using <= k vehicle legs, for each k."""
    conns = connections(net)
    best: dict[int, float] = {}
    best_at: dict[tuple[str, int], float] = {}

    def dfs(stop: str, time: float, rides: int, cur_trip: str | None):
        if rides > max_rides:
            return
        key = (stop, rides)
        if time >= best_at.get(key, float("inf")):
            return
        best_at[key] = time
        if stop == dest:
            cur = best.get(rides, float("inf"))
            if time < cur:
                best[rides] = time
        for fs, ts, dep, arr, tid in conns:
            if fs == stop and dep >= time:
                nrides = rides if tid == cur_trip else rides + 1
                dfs(ts, arr, nrides, tid)
        for tgt, secs in net.footpaths.get(stop, ()):
            dfs(tgt, time + secs, rides, None)

    dfs(origin, dep_time, 0, None)
    out: dict[int, float] = {}
    for k in range(max_rides + 1):
        vals = [v for kk, v in best.items() if kk <= k]
        if vals:
            out[k] = min(vals)
    return out


class TestRaptorAgainstBruteForce(unittest.TestCase):
    def setUp(self):
        self.net = build_fixture()

    def check(self, origin, dest, dep_time):
        res = run_raptor(self.net, origin, dest, dep_time, max_rounds=4)
        bf = brute_force(self.net, origin, dest, dep_time)
        rap_cum: dict[int, int] = {}
        running = float("inf")
        for k in sorted(set(res.pareto) | set(range(5))):
            if k in res.pareto:
                running = min(running, res.pareto[k])
            if running != float("inf"):
                rap_cum[k] = int(running)
        self.assertEqual(
            {k: int(v) for k, v in bf.items()},
            rap_cum,
            msg=f"Mismatch OD=({origin},{dest}) dep={dep_time}",
        )

    def test_all_pairs_multiple_departures(self):
        stops = ["A", "B", "C", "D", "E", "F"]
        deps = [28700, 28800, 28900, 29000, 29100, 29300, 29500, 30500]
        for o in stops:
            for d in stops:
                if o == d:
                    continue
                for t in deps:
                    self.check(o, d, t)

    def test_journey_reconstruction_is_consistent(self):
        res = run_raptor(self.net, "A", "F", 28800, max_rounds=4)
        self.assertTrue(res.journeys)
        for k, j in res.journeys.items():
            self.assertEqual(j.legs[0].from_stop, "A")
            self.assertEqual(j.legs[-1].to_stop, "F")
            self.assertLessEqual(j.n_rides, k)
            self.assertEqual(j.arrival, res.pareto[k])
            for a, b in zip(j.legs, j.legs[1:]):
                self.assertEqual(a.to_stop, b.from_stop)
                self.assertLessEqual(a.arrive, b.depart)

    def test_unknown_stop_raises(self):
        with self.assertRaises(KeyError):
            run_raptor(self.net, "A", "ZZZ", 28800)


if __name__ == "__main__":
    unittest.main()
