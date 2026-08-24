"""Feed discovery, streamed download, selective extraction, Parquet conversion.

Disk discipline: stream the zip, extract only whitelisted GTFS txt files, convert each
to Parquet via DuckDB, delete extracted CSVs; raw zip deleted after successful build.
"""

from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

# Hand-maintained direct URLs (stable agency endpoints). Update here if an agency
# changes its URL; keep each feed under ~200MB zipped.
FEED_SOURCES: dict[str, str] = {
    "trimet": "https://developer.trimet.org/schedule/gtfs.zip",
    "cta": "https://www.transitchicago.com/downloads/sch_data/google_transit.zip",
    "hsl": "https://dev.hsl.fi/gtfs/hsl.zip",
    "mta": "http://web.mta.info/developers/data/nyct/subway/google_transit.zip",
}

# GTFS files required for journey planning.
REQUIRED_FILES = [
    "agency.txt",
    "stops.txt",
    "routes.txt",
    "trips.txt",
    "stop_times.txt",
    "calendar.txt",
]
OPTIONAL_FILES = ["calendar_dates.txt", "transfers.txt", "feed_info.txt"]

MAX_ZIP_BYTES = 400 * 1024 * 1024  # abort streams beyond 400MB


@dataclass(frozen=True)
class FeedPaths:
    slug: str
    root: Path

    @property
    def zip_path(self) -> Path:
        return self.root / "raw" / f"{self.slug}.zip"

    @property
    def work_dir(self) -> Path:
        return self.root / "work" / self.slug

    @property
    def parquet_dir(self) -> Path:
        return self.root / "parquet" / self.slug

    @property
    def index_path(self) -> Path:
        return self.root / "index" / f"{self.slug}.sqlite"


def feed_paths(root: Path, slug: str) -> FeedPaths:
    if slug not in FEED_SOURCES:
        raise KeyError(f"unknown feed {slug!r}; known: {sorted(FEED_SOURCES)}")
    return FeedPaths(slug=slug, root=root)


def download_feed(paths: FeedPaths, chunk_size: int = 1 << 20) -> Path:
    """Stream-download the feed zip with a size guard. Resumable via .part rename."""
    url = FEED_SOURCES[paths.slug]
    paths.zip_path.parent.mkdir(parents=True, exist_ok=True)
    part = paths.zip_path.with_suffix(".zip.part")
    if paths.zip_path.exists():
        return paths.zip_path
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        if total and total > MAX_ZIP_BYTES:
            raise RuntimeError(f"{paths.slug}: content-length {total} exceeds guard")
        written = 0
        with open(part, "wb") as fh:
            for chunk in r.iter_content(chunk_size):
                written += len(chunk)
                if written > MAX_ZIP_BYTES:
                    raise RuntimeError(f"{paths.slug}: stream exceeded size guard")
                fh.write(chunk)
    part.replace(paths.zip_path)
    return paths.zip_path


def _csv_stem(name: str) -> str:
    return name[:-4] if name.endswith(".txt") else Path(name).stem


def extract_and_convert(paths: FeedPaths) -> dict[str, Path]:
    """Selectively unzip required+optional files, convert each to Parquet, clean up.

    Returns mapping of GTFS file stem -> parquet path.
    """
    import duckdb

    wanted = {f for f in REQUIRED_FILES + OPTIONAL_FILES}
    work = paths.work_dir
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    with zipfile.ZipFile(paths.zip_path) as zf:
        names = zf.namelist()
        present = {}
        for n in names:
            base = n.split("/")[-1].lower()
            if base in wanted and base not in present:
                present[base] = n
        missing = [f for f in REQUIRED_FILES if f not in present]
        if missing:
            raise RuntimeError(f"{paths.slug}: feed missing required files: {missing}")
        for base, arcname in sorted(present.items()):
            zf.extract(arcname, work)

    con = duckdb.connect()
    try:
        for base in sorted(present):
            stem = _csv_stem(base)
            csv_path = work / base
            pq_dir = paths.parquet_dir / stem
            pq_dir.mkdir(parents=True, exist_ok=True)
            pq_path = pq_dir / "data.parquet"
            con.execute(
                f"COPY (SELECT * FROM read_csv_auto('{csv_path.as_posix()}', "
                f"header=true, sample_size=-1)) TO '{pq_path.as_posix()}' "
                f"(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            out[stem] = pq_path
            csv_path.unlink(missing_ok=True)
    finally:
        con.close()

    # Work dir should now be empty; remove it. Keep the raw zip by default so a
    # rebuild never re-downloads; `clean` removes it explicitly.
    shutil.rmtree(work, ignore_errors=True)
    return out
