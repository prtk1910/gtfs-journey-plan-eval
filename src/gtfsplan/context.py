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


def _route_label(con: sqlite3.Connection, route_id: str, cache: dict) -> str:
    if route_id not in cache:
        row = con.execute(
            "SELECT COALESCE(NULLIF(route_short_name,''), route_long_name) "
            "FROM routes WHERE route_id=?",
            (route_id,),
        ).fetchone()
        cache[route_id] = row[0] if row else route_id
    return cache[route_id]


def _stop_name(con: sqlite3.Connection, stop_id: str) -> str:
    row = con.execute("SELECT stop_name FROM stops WHERE stop_id=?", (stop_id,)).fetchone()
    return row[0] if row else stop_id


class FeedContext:
    """Precomputed per-feed structures shared across all items."""

    def __init__(self, index_db, day: dt.date):
        self.con = sqlite3.connect(index_db)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.day = day
        self.frag, self.args = self._active_sql(day)
        self.route_cache: dict[str, str] = {}
        ph, args = self.frag, self.args
        self.stop_routes: dict[str, list[str]] = {}
        for sid, rid in self.con.execute(
            f"""SELECT DISTINCT st.stop_id, t.route_id FROM stop_times st
                JOIN trips t ON t.trip_id=st.trip_id
                WHERE t.service_id IN ({ph})""",
            args,
        ):
            self.stop_routes.setdefault(sid, []).append(rid)
        for k in self.stop_routes:
            self.stop_routes[k].sort()

    def _active_sql(self, day: dt.date):
        active = sorted(resolve_active_services(self.con, day))
        return ",".join("?" * len(active)), tuple(active)

    def close(self):
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
    con = ctx.con if ctx is not None else sqlite3.connect(index_db)
    owns = ctx is None
    try:
        if owns:
            con.execute("PRAGMA journal_mode=WAL")
        day = dt.date.fromisoformat(item["day"])
        dep = item["dep_time"]
        t0 = dep - window_before_s
        t1 = dep + window_after_s
        if ctx is not None:
            frag, args = ctx.frag, ctx.args
            sr = ctx.stop_routes
        else:
            active = sorted(resolve_active_services(con, day))
            frag = ",".join("?" * len(active))
            args = tuple(active)
            sr = {}
        rcache = ctx.route_cache if ctx else {}
        o, d = item["origin"], item["destination"]

        def fmt_route(rid: str) -> str:
            return f"route {_route_label(con, rid, rcache)}"

        lines: list[str] = []
        lines.append(f"Services departing ORIGIN [{_stop_name(con, o)}] "
                     f"{_fmt_min(t0)}-{_fmt_min(t1)}:")
        for rid, dsec, head in _departures(con, frag, args, o, t0, t1, 8):
            lines.append(f"  {_fmt_min(dsec)} {fmt_route(rid)}"
                         + (f" towards {head}" if head else ""))
        lines.append(f"Services arriving DESTINATION [{_stop_name(con, d)}] "
                     f"{_fmt_min(t0)}-{_fmt_min(t1)}:")
        for rid, asec in _arrivals(con, frag, args, d, t0, t1, 8):
            lines.append(f"  {_fmt_min(asec)} {fmt_route(rid)}")

        def routes_at(stop_id: str) -> set[str]:
            if ctx is not None:
                return set(sr.get(stop_id, ()))
            rows = con.execute(
                f"""SELECT DISTINCT t.route_id FROM stop_times st
                    JOIN trips t ON t.trip_id=st.trip_id
                    WHERE st.stop_id=? AND t.service_id IN ({frag})""",
                (stop_id, *args),
            ).fetchall()
            return {r[0] for r in rows}

        routes_o = routes_at(o)
        routes_d = routes_at(d)

        def stops_of_routes(route_ids: set[str]) -> dict[str, tuple[float, float]]:
            ph = ",".join("?" * len(route_ids)) if route_ids else "NULL"
            rows = con.execute(
                f"""SELECT DISTINCT st.stop_id, s.stop_lat, s.stop_lon
                    FROM stop_times st
                    JOIN trips t ON t.trip_id = st.trip_id
                    JOIN stops s ON s.stop_id = st.stop_id
                    WHERE t.route_id IN ({ph}) AND t.service_id IN ({frag})
                      AND s.stop_lat IS NOT NULL""",
                (*route_ids, *args),
            ).fetchall()
            return {r[0]: (r[1], r[2]) for r in rows}

        so = stops_of_routes(routes_o)
        sd = stops_of_routes(routes_d)
        shared_o: dict[str, str] = {}
        for sid in sorted((so.keys() & sd.keys()) - {o, d})[:max_transfer_stops]:
            shared_o[sid] = sid
        if len(shared_o) < max_transfer_stops:
            A = [(s, c) for s, c in so.items() if s not in shared_o and s not in (o, d)]
            B = [(s, c) for s, c in sd.items() if s not in (o, d)]
            if A and B:
                la = np.array([c[0] for _, c in A]); loa = np.array([c[1] for _, c in A])
                lb = np.array([c[0] for _, c in B]); lob = np.array([c[1] for _, c in B])
                lat1 = np.radians(la)[:, None]; lon1 = np.radians(loa)[:, None]
                lat2 = np.radians(lb)[None, :]; lon2 = np.radians(lob)[None, :]
                dlat = lat2 - lat1; dlon = lon2 - lon1
                h = (np.sin(dlat / 2) ** 2
                     + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
                dists = 6_371_000.0 * 2 * np.arcsin(np.sqrt(h))
                close = np.argwhere(dists <= 400.0)
                order = sorted(close.tolist(), key=lambda p: dists[p[0], p[1]])
                for i, j in order:
                    s1, s2 = A[i][0], B[j][0]
                    if s1 == o or s2 == d or s1 == d or s2 == o:
                        continue
                    shared_o.setdefault(s1, s2)
                    if len(shared_o) >= max_transfer_stops:
                        break

        if not shared_o:
            lines.append(
                "No shared or walk-connected stop between origin-serving and "
                "destination-serving routes was found in the index.")
        for sid_o in list(shared_o):
            sid_d = shared_o[sid_o]
            label = _stop_name(con, sid_o)
            if sid_d != sid_o:
                label += f" (then walk to [{_stop_name(con, sid_d)}])"
            lines.append(f"Transfer point [{label}]:")
            seen_any = False
            for rid, asec in _arrivals(con, frag, args, sid_o, t0, dep + 10800, 4):
                if rid in routes_o:
                    lines.append(f"  arrive {_fmt_min(asec)} {fmt_route(rid)}")
                    seen_any = True
            for rid, dsec, head in _departures(con, frag, args, sid_d, t0, dep + 12600, 4):
                if rid in routes_d:
                    lines.append(f"  depart {_fmt_min(dsec)} {fmt_route(rid)}"
                                 + (f" towards {head}" if head else ""))
                    seen_any = True
            if not seen_any:
                lines.append("  (no timed connections in window)")

        def trip_sequence_block(stop_id: str, direction: str, cap: int = 2,
                                max_stops: int = 45) -> None:
            trips = con.execute(
                f"""SELECT st.trip_id, st.departure_s
                    FROM stop_times st
                    JOIN trips t ON t.trip_id = st.trip_id
                    WHERE st.stop_id=? AND st.departure_s BETWEEN ? AND ?
                      AND t.service_id IN ({frag})
                      AND st.departure_s IS NOT NULL
                    ORDER BY st.departure_s LIMIT {cap}""",
                (stop_id, max(t0, dep - 900), t1, *args),
            ).fetchall()
            for tid, dsec in trips:
                rid = con.execute(
                    "SELECT route_id FROM trips WHERE trip_id=?", (tid,)
                ).fetchone()[0]
                rows = con.execute(
                    """SELECT st.stop_sequence, st.stop_id, st.departure_s, s.stop_name
                       FROM stop_times st JOIN stops s ON s.stop_id = st.stop_id
                       WHERE st.trip_id=? ORDER BY st.stop_sequence""",
                    (tid,),
                ).fetchall()
                idx = next((i for i, r in enumerate(rows) if r[1] == stop_id), None)
                if idx is None:
                    continue
                if direction == "arr":
                    shown = [r for r in rows[:idx]][::-1][:max_stops]
                    head_line = (f"{fmt_route(rid)} trip arriving DESTINATION at "
                                 f"{_fmt_min(rows[idx][2])} (earlier stops, "
                                 "most recent first):")
                else:
                    shown = rows[idx + 1:][:max_stops]
                    head_line = (f"{fmt_route(rid)} trip departing ORIGIN at "
                                 f"{_fmt_min(dsec)} (subsequent stops):")
                lines.append(head_line)
                for _seq, sid2, tsec, name in shown:
                    others = len(set(sr.get(sid2, [])) - {rid}) if ctx is not None else ""
                    tag = f" [+{others} other routes]" if others else ""
                    lines.append(f"    {_fmt_min(tsec)} [{name}]{tag}")

        trip_sequence_block(o, "dep")
        trip_sequence_block(d, "arr")
        return "\n".join(lines[:max_lines])
    finally:
        if owns:
            con.close()


def _departures(con, sql_frag, args, stop_id: str, t0: int, t1: int, limit: int):
    return con.execute(
        f"""SELECT r.route_id, st.departure_s, t.trip_headsign
            FROM stop_times st
            JOIN trips t ON t.trip_id = st.trip_id
            JOIN routes r ON r.route_id = t.route_id
            WHERE st.stop_id=? AND st.departure_s BETWEEN ? AND ?
              AND t.service_id IN ({sql_frag})
              AND st.departure_s IS NOT NULL
            ORDER BY st.departure_s LIMIT {limit}""",
        (stop_id, t0, t1, *args),
    ).fetchall()


def _arrivals(con, sql_frag, args, stop_id: str, t0: int, t1: int, limit: int):
    return con.execute(
        f"""SELECT r.route_id, st.arrival_s
            FROM stop_times st
            JOIN trips t ON t.trip_id = st.trip_id
            JOIN routes r ON r.route_id = t.route_id
            WHERE st.stop_id=? AND st.arrival_s BETWEEN ? AND ?
              AND t.service_id IN ({sql_frag})
              AND st.arrival_s IS NOT NULL
            ORDER BY st.arrival_s LIMIT {limit}""",
        (stop_id, t0, t1, *args),
    ).fetchall()
