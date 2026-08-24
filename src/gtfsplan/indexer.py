"""Build a compact SQLite routing index from Parquet extracts."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import duckdb

SCHEMA = """
CREATE TABLE IF NOT EXISTS stops (
    stop_id TEXT PRIMARY KEY, stop_name TEXT, stop_lat DOUBLE, stop_lon DOUBLE);
CREATE TABLE IF NOT EXISTS routes (
    route_id TEXT PRIMARY KEY, route_short_name TEXT, route_long_name TEXT, route_type INT);
CREATE TABLE IF NOT EXISTS trips (
    trip_id TEXT, route_id TEXT, service_id TEXT, trip_headsign TEXT);
CREATE INDEX IF NOT EXISTS ix_trips_service ON trips(service_id);
CREATE TABLE IF NOT EXISTS stop_times (
    trip_id TEXT, stop_sequence INT, stop_id TEXT, arrival_s INT, departure_s INT);
CREATE INDEX IF NOT EXISTS ix_stop_times_trip ON stop_times(trip_id);
CREATE INDEX IF NOT EXISTS ix_stop_times_stop ON stop_times(stop_id);
CREATE TABLE IF NOT EXISTS calendar (
    service_id TEXT, monday INT, tuesday INT, wednesday INT, thursday INT,
    friday INT, saturday INT, sunday INT, start_date INT, end_date INT);
CREATE TABLE IF NOT EXISTS calendar_dates (
    service_id TEXT, date INT, exception_type INT);
CREATE TABLE IF NOT EXISTS transfers (
    from_stop_id TEXT, to_stop_id TEXT, min_transfer_time INT);
"""

TIME_TO_S = (
    "(try_cast(split_part(COALESCE({col},''), ':', 1) AS INT) * 3600 + "
    "try_cast(split_part({col}, ':', 2) AS INT) * 60 + "
    "try_cast(split_part({col}, ':', 3) AS INT))"
)


def _rows(con_duck, sql):
    cur = con_duck.execute(sql)
    while True:
        batch = cur.fetchmany(50_000)
        if not batch:
            return
        yield from batch


def build_index(paths) -> Path:
    pq = paths.parquet_dir
    out = paths.index_path
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    con_sql = sqlite3.connect(out)
    con_sql.executescript(SCHEMA)

    duck = duckdb.connect()
    try:
        con_sql.executemany(
            "INSERT OR REPLACE INTO stops VALUES (?,?,?,?)",
            _rows(
                duck,
                f"""SELECT stop_id, CAST(stop_name AS VARCHAR),
                           try_cast(stop_lat AS DOUBLE), try_cast(stop_lon AS DOUBLE)
                    FROM read_parquet('{(pq / "stops").as_posix()}/data.parquet')""",
            ),
        )
        con_sql.executemany(
            "INSERT OR REPLACE INTO routes VALUES (?,?,?,?)",
            _rows(
                duck,
                f"""SELECT route_id, CAST(route_short_name AS VARCHAR),
                           CAST(route_long_name AS VARCHAR), try_cast(route_type AS INT)
                    FROM read_parquet('{(pq / "routes").as_posix()}/data.parquet')""",
            ),
        )
        trips_cols = {r[0] for r in duck.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{(pq / 'trips').as_posix()}/data.parquet')"
        ).fetchall()}
        head = ("CAST(trip_headsign AS VARCHAR)" if "trip_headsign" in trips_cols else "NULL")
        con_sql.executemany(
            "INSERT INTO trips VALUES (?,?,?,?)",
            _rows(
                duck,
                f"""SELECT trip_id, route_id, service_id, {head}
                    FROM read_parquet('{(pq / 'trips').as_posix()}/data.parquet')""",
            ),
        )

        def st_rows():
            files = f"{(pq / 'stop_times').as_posix()}/data.parquet"
            for row in _rows(
                duck,
                f"""SELECT trip_id, try_cast(stop_sequence AS INT), stop_id,
                           {TIME_TO_S.format(col='arrival_time')},
                           {TIME_TO_S.format(col='departure_time')}
                    FROM read_parquet('{files}')""",
            ):
                yield row

        con_sql.executemany("INSERT INTO stop_times VALUES (?,?,?,?,?)", st_rows())

        cal_path = pq / "calendar" / "data.parquet"
        if cal_path.exists():
            cols = ", ".join(f"try_cast({d} AS INT)" for d in
                             ["monday", "tuesday", "wednesday", "thursday",
                              "friday", "saturday", "sunday"])
            con_sql.executemany(
                "INSERT OR REPLACE INTO calendar VALUES (?,?,?,?,?,?,?,?,?,?)",
                _rows(
                    duck,
                    f"""SELECT service_id, {cols},
                               try_cast(start_date AS BIGINT), try_cast(end_date AS BIGINT)
                        FROM read_parquet('{cal_path.as_posix()}')""",
                ),
            )
        cd_path = pq / "calendar_dates" / "data.parquet"
        if cd_path.exists():
            con_sql.executemany(
                "INSERT INTO calendar_dates VALUES (?,?,?)",
                _rows(
                    duck,
                    f"""SELECT service_id, try_cast(date AS BIGINT),
                               try_cast(exception_type AS INT)
                        FROM read_parquet('{cd_path.as_posix()}')""",
                ),
            )
        tr_path = pq / "transfers" / "data.parquet"
        if tr_path.exists():
            tr_cols = {r[0] for r in duck.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{tr_path.as_posix()}')"
            ).fetchall()}
            mtt = ("try_cast(min_transfer_time AS INT)"
                   if "min_transfer_time" in tr_cols else "NULL")
            con_sql.executemany(
                "INSERT OR REPLACE INTO transfers VALUES (?,?,?)",
                _rows(
                    duck,
                    f"""SELECT from_stop_id, to_stop_id, {mtt}
                        FROM read_parquet('{tr_path.as_posix()}')""",
                ),
            )
        con_sql.commit()
        n = con_sql.execute("SELECT COUNT(*) FROM stop_times").fetchone()[0]
        print(f"[{paths.slug}] index built: {n:,} stop_times -> {out.name}")
    finally:
        duck.close()
        con_sql.close()
    return out
