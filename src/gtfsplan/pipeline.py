"""Command-line entry points for the gtfs-journey-plan-eval pipeline."""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

from .feeds import FEED_SOURCES, download_feed, extract_and_convert, feed_paths
from .indexer import build_index
from .goldgen import generate_gold, next_weekday

ROOT = Path(__file__).resolve().parents[2]


def cmd_fetch(args) -> None:
    paths = feed_paths(ROOT / "artifacts", args.feed)
    zp = download_feed(paths)
    print(f"[{args.feed}] downloaded -> {zp} ({zp.stat().st_size/1e6:.1f} MB)")


def cmd_build(args) -> None:
    paths = feed_paths(ROOT / "artifacts", args.feed)
    download_feed(paths)
    extract_and_convert(paths)
    build_index(paths)


def cmd_gold(args) -> None:
    day = dt.date.fromisoformat(args.day) if args.day else next_weekday(dt.date.today())
    paths = feed_paths(ROOT / "artifacts", args.feed)
    generate_gold(
        ROOT, args.feed, day,
        n_per_stratum=args.n_per_stratum, seed=args.seed,
        max_walk_m=args.max_walk_m,
    )


def cmd_smoke(_args) -> None:
    from gtfsplan.raptor import run_raptor, fmt_hhmm
    from tests.test_raptor import build_fixture

    net = build_fixture()
    res = run_raptor(net, "A", "F", 28800)
    assert res.pareto, "smoke raptor found nothing"
    k = min(res.pareto)
    print(f"smoke OK: A->F best arrival {fmt_hhmm(res.pareto[k])} with {k} rides; "
          f"{len(res.journeys)} pareto journeys reconstructed")


def cmd_run(args) -> None:
    from .runner import ARMS, run_eval

    feeds = args.feeds.split(",") if args.feeds else sorted(FEED_SOURCES)
    arms = [a for a in args.arms.split(",") if a] or list(ARMS)
    run_eval(ROOT, feeds, arms, n_per_feed=args.n_per_feed, workers=args.workers)


def cmd_paper(_args) -> None:
    from .analysis import write_paper
    from .paperio import md_to_docx

    md = write_paper(ROOT, config_note="stealth/ox-alpha via OpenRouter")
    docx = md_to_docx(md, ROOT / "PAPER.docx")
    print(f"paper written: {md} and {docx}")


def cmd_clean(_args) -> None:
    art = ROOT / "artifacts"
    if art.exists():
        for sub in ["raw", "work"]:
            p = art / sub
            if p.exists():
                shutil.rmtree(p)
                print(f"removed {p}")
        zips = list((art / "raw").glob("*.zip")) if (art / "raw").exists() else []
        for z in zips:
            z.unlink()
            print(f"removed {z}")
    else:
        print("nothing to clean")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gtfsplan")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def feed_arg(p):
        p.add_argument("--feed", default="trimet", choices=sorted(FEED_SOURCES))

    p = sub.add_parser("fetch"); feed_arg(p); p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("build"); feed_arg(p); p.set_defaults(fn=cmd_build)
    p = sub.add_parser("gold"); feed_arg(p)
    p.add_argument("--day", default=None)
    p.add_argument("--n-per-stratum", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-walk-m", type=float, default=0.0)
    p.set_defaults(fn=cmd_gold)
    sub.add_parser("smoke").set_defaults(fn=cmd_smoke)
    p = sub.add_parser("run")
    p.add_argument("--feeds", default=None)
    p.add_argument("--arms", default="closed_book,schedule_excerpt")
    p.add_argument("--n-per-feed", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=cmd_run)
    sub.add_parser("paper").set_defaults(fn=cmd_paper)
    sub.add_parser("clean").set_defaults(fn=cmd_clean)

    args = ap.parse_args(argv)
    try:
        args.fn(args)
    except NotImplementedError as e:
        print(f"not implemented yet: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
