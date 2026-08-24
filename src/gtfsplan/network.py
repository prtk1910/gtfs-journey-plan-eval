"""Build an in-memory transit network from Parquet extracts for routing."""

from __future__ import annotations

import datetime as dt
import sqlite3
from dataclasses import dataclass
from pathlib import Path


def parse_gtfs_time(value: str | None) -> int | None:
    """Parse HH:MM:SS (hours may exceed 23) into seconds after midnight."""
    if value is None or value == "":
        return None
    parts = value.strip().split(":")
    if len(parts) != 3:
        return None
    h, m, s = (int(p) for p in parts)
    return h * 3600 + m * 60 + s


@dataclass(frozen=True)
class Stop:
    stop_id: str
    name: str
    lat: float | None
    lon: float | None


@dataclass(frozen=True)
class StopTime:
    stop_sequence: int
    stop_id: str
    arrival: int
    departure: int


@dataclass(frozen=True)
class Trip:
    trip_id: str
    route_id: str
    stop_times: tuple[StopTime, ...]


@dataclass(frozen=True)
class RouteInfo:
    route_id: str
    short_name: str
    long_name: str
    route_type: int


@dataclass
class Network:
    day: dt.date
    stops: dict[str, Stop]
    routes: dict[str, RouteInfo]
    trips_by_route: dict[str, list[Trip]]
    stop_routes: dict[str, set[str]]
    footpaths: dict[str, list[tuple[str, int]]]
    n_trips_total: int
    n_trips_active: int

    def haversine_m(self, a: str, b: str) -> float | None:
        s1 = self.stops.get(a)
        s2 = self.stops.get(b)
        if s1 is None or s2 is None or s1.lat is None or s2.lat is None:
            return None
        import math

        lon1, lat1, lon2, lat2 = map(math.radians, [s1.lon, s1.lat, s2.lon, s2.lat])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 6371000.0 * 2 * math.asin(math.sqrt(h))


def resolve_active_services(
    con: sqlite3.Connection, day: dt.date
) -> set[str]:
    """Return service_ids active on `day` given calendar + calendar_dates tables.

    Dates are stored as integers in YYYYMMDD form for fast comparison.
    """
    active: set[str] = set()
    weekday_col = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][
        day.weekday()
    ]
    ymd = int(day.strftime("%Y%m%d"))
    for row in con.execute(
        f"""SELECT service_id FROM calendar
            WHERE start_date <= ? AND end_date >= ?
              AND {weekday_col} = 1""",
        (ymd, ymd),
    ):
        active.add(row[0])
    for service_id, etype in con.execute(
        "SELECT service_id, exception_type FROM calendar_dates WHERE date = ?",
        (ymd,),
    ):
        if etype == 1:
            active.add(service_id)
        elif etype == 2:
            active.discard(service_id)
    return active


def _grid_cells(lat: float, lon: float, cell_deg: float) -> tuple[int, int]:
    return (int(lat / cell_deg), int(lon / cell_deg))


def load_network(
    index_db: Path,
    day: dt.date,
    walk_speed_mps: float = 1.0,
    max_walk_m: float = 0.0,
    time_window: tuple[int, int] | None = None,
) -> Network:
    """Load the SQLite index built by pipeline.build into a routable Network.

    Footpaths come from transfers.txt rows with min_transfer_time; optionally add
    distance-based footpaths between stops within max_walk_m using a spatial grid.
    If time_window=(t0,t1) is given, only trips whose service interval intersects
    it are kept.
    """
    import math

    con = sqlite3.connect(index_db)
    con.execute("PRAGMA journal_mode=WAL")
    try:
        active = resolve_active_services(con, day)

        stops = {
            sid: Stop(sid, name, lat, lon)
            for sid, name, lat, lon in con.execute(
                "SELECT stop_id, COALESCE(stop_name,''), stop_lat, stop_lon FROM stops"
            )
        }
        routes = {
            rid: RouteInfo(rid, short or "", long or "", rtype)
            for rid, short, long, rtype in con.execute(
                "SELECT route_id, route_short_name, route_long_name, route_type FROM routes"
            )
        }

        placeholders = ",".join("?" * len(active))
        trip_rows = con.execute(
            f"""SELECT t.trip_id, t.route_id, st.stop_sequence, st.stop_id,
                       st.arrival_s, st.departure_s
                FROM trips t JOIN stop_times st ON st.trip_id = t.trip_id
                WHERE t.service_id IN ({placeholders})
                ORDER BY t.trip_id, st.stop_sequence""",
            tuple(active),
        ).fetchall()

        trips_by_route: dict[str, dict[str, list[StopTime]]] = {}
        trip_route: dict[str, str] = {}
        seen_incomplete: set[str] = set()
        for trip_id, route_id, seq, stop_id, arr, dep in trip_rows:
            if arr is None or dep is None:
                seen_incomplete.add(trip_id)
                continue
            trips_by_route.setdefault(route_id, {}).setdefault(trip_id, []).append(
                StopTime(int(seq), stop_id, int(arr), int(dep))
            )
            trip_route[trip_id] = route_id

        trips: dict[str, list[Trip]] = {}
        total = 0
        for route_id, tm in trips_by_route.items():
            tl = []
            for tid, sts in tm.items():
                if tid in seen_incomplete or not sts:
                    continue
                if time_window is not None:
                    first_dep = min(s.departure for s in sts)
                    last_arr = max(s.arrival for s in sts)
                    if last_arr < time_window[0] or first_dep > time_window[1]:
                        continue
                tl.append(Trip(tid, route_id, tuple(sts)))
            tl.sort(key=lambda t: (t.stop_times[0].departure, t.trip_id))
            trips[route_id] = tl
            total += len(tl)

        stop_routes: dict[str, set[str]] = {}
        for route_id, tl in trips.items():
            for tr in tl:
                for st in tr.stop_times:
                    stop_routes.setdefault(st.stop_id, set()).add(route_id)

        footpaths: dict[str, list[tuple[str, int]]] = {}
        for src, dst, secs in con.execute(
            """SELECT from_stop_id, to_stop_id, MIN(min_transfer_time)
               FROM transfers WHERE min_transfer_time IS NOT NULL
               GROUP BY from_stop_id, to_stop_id"""
        ):
            if secs is not None and int(secs) >= 0:
                footpaths.setdefault(src, []).append((dst, int(secs)))
        if max_walk_m > 0:
            cell_deg = max_walk_m / 111_320.0
            grid: dict[tuple[int, int], list[str]] = {}
            for sid, st in stops.items():
                if st.lat is None or st.lon is None:
                    continue
                grid.setdefault(_grid_cells(st.lat, st.lon, cell_deg), []).append(sid)

            def haversine(a: Stop, b: Stop) -> float:
                lon1, lat1, lon2, lat2 = map(
                    math.radians, [a.lon, a.lat, b.lon, b.lat]
                )
                dlat, dlon = lat2 - lat1, lon2 - lon1
                h = (
                    math.sin(dlat / 2) ** 2
                    + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
                )
                return 6_371_000.0 * 2 * math.asin(math.sqrt(h))

            for src in stops:
                s = stops[src]
                if s.lat is None or s.lon is None:
                    continue
                cx, cy = _grid_cells(s.lat, s.lon, cell_deg)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dst in grid.get((cx + dx, cy + dy), ()):
                            if dst == src:
                                continue
                            d = haversine(s, stops[dst])
                            if d <= max_walk_m:
                                secs = max(30, int(round(d / walk_speed_mps)))
                                existing = dict(footpaths.get(src, []))
                                cur = existing.get(dst)
                                if cur is None or secs < cur:
                                    footpaths.setdefault(src, []).append((dst, secs))

        return Network(
            day=day,
            stops=stops,
            routes=routes,
            trips_by_route=trips,
            stop_routes=stop_routes,
            footpaths=footpaths,
            n_trips_total=len({tid for tl in trips.values() for tid in (t.trip_id for t in tl)})
            + len(seen_incomplete),
            n_trips_active=total,
        )
    finally:
        con.close()
