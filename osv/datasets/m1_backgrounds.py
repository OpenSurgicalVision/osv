"""M1 backgrounds: review sheets of tool-free CholecT50 frames, and the reviewed list.

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

    python -m osv.datasets.m1_backgrounds sheets --census ~/osv-runs/m1_census.json \\
        --out ~/osv-runs/m1_sheets
    # a person looks at every sheet and writes rejects.txt (format below)
    python -m osv.datasets.m1_backgrounds collect --sheets ~/osv-runs/m1_sheets \\
        --rejects ~/osv-runs/m1_sheets/rejects.txt --reviewer "Name" \\
        --out ~/osv-runs/m1_backgrounds.json

"Tool-free" in the census means only that the CholecT50 triplet labels name none of
its six instruments. A look at 48 such frames (RFC 0001, 2026-10-23) found about 40 %
clean tissue, 20 % with an instrument or object the labels do not name (trocars,
suction tips, drains, a looped rod, possibly grasper and hook shafts), 15 % black and
15 % unreadable (inside a trocar, blown out, out of focus). A visible instrument left
in a "background" would teach M1 that real instruments are background, which is the
one error that decides H1 against the synthetic arm - so every background is looked
at by a person before it is used.

`sheets` keeps every `--stride`-th frame of each run of consecutive tool-free frames
(1 fps: neighbours are nearly the same scene), drops frames the M5 haze score already
calls black or blown out (`cholect50.haze_score` == 1.0), and lays the rest out in
numbered sheets of 6 x 6 tiles. Each tile is labelled ``row.column`` in its corner.
Frames come from the CholecT50 train and val videos only; test stays untouched.

`rejects.txt` has one line per sheet - every sheet, also those with nothing to
reject, so an unreviewed sheet can never pass as clean::

    # reject a tile showing any instrument, trocar, drain, bag, tube or other object
    # (clips and suture threads are tissue), and any frame where tissue is not clearly
    # visible
    001: 1.2 3.3 6.6
    002:
    003: 2.1

`collect` checks that every sheet is listed exactly once and every tile exists, and
writes the accepted frames per split with the reviewer and the census hash. That file
is the reviewed input of the M1 compositing stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROWS = COLS = 6
TILE = (320, 180)
SPLITS = ("train", "val")
LINE = re.compile(r"^\s*(\d+)\s*:\s*(.*?)\s*$")
TILE_ID = re.compile(r"^(\d+)\.(\d+)$")


def candidates(videos: Mapping[str, Mapping[str, Any]], stride: int) -> list[tuple[str, int]]:
    """(video, frame index) of every stride-th frame of each tool-free run, train and val."""
    picked = []
    for video in sorted(videos):
        record = videos[video]
        if record["split"] not in SPLITS:
            continue
        for first, last in record["runs"]:
            picked += [(video, index) for index in range(first, last + 1, stride)]
    return picked


def tile_name(position: int) -> str:
    return f"{position // COLS + 1}.{position % COLS + 1}"


def _label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    font = ImageFont.load_default(size=26)
    box = draw.textbbox(xy, text, font=font)
    draw.rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), fill=(0, 0, 0))
    draw.text(xy, text, fill=(255, 255, 0), font=font)


def render_sheet(images: list[np.ndarray]) -> Image.Image:
    sheet = Image.new("RGB", (COLS * TILE[0], ROWS * TILE[1]), (40, 40, 40))
    draw = ImageDraw.Draw(sheet)
    for position, rgb in enumerate(images):
        x, y = (position % COLS) * TILE[0], (position // COLS) * TILE[1]
        sheet.paste(Image.fromarray(rgb).resize(TILE), (x, y))
        draw.rectangle((x, y, x + TILE[0] - 1, y + TILE[1] - 1), outline=(40, 40, 40), width=2)
        _label(draw, (x + 10, y + 8), tile_name(position))
    return sheet


def make_sheets(census_path: Path, out: Path, *, source, stride: int) -> dict[str, Any]:
    from osv.datasets import cholect50 as C

    census_bytes = census_path.read_bytes()
    videos = json.loads(census_bytes)["cholect50"]["videos"]
    frames = {(f.video, f.index): f for f in source.frames()}
    picked = candidates(videos, stride)
    missing = [f"{v}_{i:06d}" for v, i in picked if (v, i) not in frames]
    if missing:
        raise ValueError(f"{len(missing)} census frames have no png, e.g. {missing[:3]}")
    out.mkdir(parents=True, exist_ok=True)
    kept: list[tuple[str, str, np.ndarray]] = []
    dark: list[str] = []
    sheets: dict[str, dict[str, str]] = {}

    def flush() -> None:
        number = f"{len(sheets) + 1:03d}"
        render_sheet([rgb for _, _, rgb in kept]).save(out / f"sheet_{number}.png")
        sheets[number] = {tile_name(p): frame_id for p, (frame_id, _, _) in enumerate(kept)}
        kept.clear()

    for video, index in picked:
        frame = frames[(video, index)]
        rgb = source.read(frame)
        if C.haze_score(rgb) >= 1.0:
            dark.append(frame.frame_id)
            continue
        kept.append((frame.frame_id, videos[video]["split"], rgb))
        if len(kept) == ROWS * COLS:
            flush()
    if kept:
        flush()
    index_record = {
        "census_sha256": hashlib.sha256(census_bytes).hexdigest(),
        "stride": stride,
        "candidates": len(picked),
        "auto_rejected_dark_or_blown": dark,
        "split_of": {
            f"{v}_{i:06d}": videos[v]["split"] for v, i in picked if f"{v}_{i:06d}" not in dark
        },
        "sheets": sheets,
    }
    (out / "sheets.json").write_text(json.dumps(index_record, indent=2) + "\n")
    template = ["# one line per sheet: '<sheet>: <tiles to reject>' (see the module docstring)"]
    template += [f"{number}:" for number in sheets]
    rejects = out / "rejects.txt"
    if not rejects.exists():
        rejects.write_text("\n".join(template) + "\n")
    return index_record


def parse_rejects(lines: Iterable[str], sheets: Mapping[str, Mapping[str, str]]) -> set[str]:
    """frame ids rejected by the reviewer; every sheet must be listed exactly once."""
    seen: set[str] = set()
    rejected: set[str] = set()
    for number, raw in enumerate(lines, start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = LINE.match(line)
        if not match:
            raise ValueError(f"rejects line {number}: expected '<sheet>: <tiles>', got {raw!r}")
        sheet = f"{int(match[1]):03d}"
        if sheet not in sheets:
            raise ValueError(f"rejects line {number}: there is no sheet {sheet}")
        if sheet in seen:
            raise ValueError(f"rejects line {number}: sheet {sheet} is listed twice")
        seen.add(sheet)
        for token in match[2].replace(",", " ").split():
            if not TILE_ID.match(token) or token not in sheets[sheet]:
                raise ValueError(f"rejects line {number}: sheet {sheet} has no tile {token!r}")
            rejected.add(sheets[sheet][token])
    unreviewed = sorted(set(sheets) - seen)
    if unreviewed:
        raise ValueError(f"sheets not in rejects.txt (unreviewed): {unreviewed}")
    return rejected


def collect(sheets_dir: Path, rejects: Path, *, reviewer: str) -> dict[str, Any]:
    index = json.loads((sheets_dir / "sheets.json").read_text())
    rejected = parse_rejects(rejects.read_text().splitlines(), index["sheets"])
    reviewed = [frame for tiles in index["sheets"].values() for frame in tiles.values()]
    accepted: dict[str, list[str]] = {split: [] for split in SPLITS}
    for frame in sorted(reviewed):
        if frame not in rejected:
            accepted[index["split_of"][frame]].append(frame)
    return {
        "disclaimer": "RESEARCH USE ONLY - NOT FOR CLINICAL USE",
        "census_sha256": index["census_sha256"],
        "rejects_sha256": hashlib.sha256(rejects.read_bytes()).hexdigest(),
        "stride": index["stride"],
        "reviewer": reviewer,
        "reviewed_on": date.today().isoformat(),
        "counts": {
            "candidates": index["candidates"],
            "auto_rejected_dark_or_blown": len(index["auto_rejected_dark_or_blown"]),
            "reviewed": len(reviewed),
            "rejected_by_reviewer": len(rejected),
            **{f"accepted_{split}": len(frames) for split, frames in accepted.items()},
        },
        "accepted": accepted,
        "rejected_by_reviewer": sorted(rejected),
    }


def main(argv: list[str] | None = None) -> int:
    from osv.datasets import cholect50 as C

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    sheet = commands.add_parser("sheets", help="write review sheets")
    sheet.add_argument("--census", type=Path, required=True)
    sheet.add_argument("--out", type=Path, required=True, help="a folder outside the repo")
    sheet.add_argument("--stride", type=int, default=3)
    sheet.add_argument("--cholect50", help="default: CHOLECT50_RAW_ZIP or CHOLECT50_DIR")
    done = commands.add_parser("collect", help="turn rejects.txt into the reviewed list")
    done.add_argument("--sheets", type=Path, required=True)
    done.add_argument("--rejects", type=Path, required=True)
    done.add_argument("--reviewer", required=True)
    done.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "sheets":
        source = C.FrameSource.from_env(args.cholect50)
        try:
            index = make_sheets(args.census, args.out, source=source, stride=args.stride)
        finally:
            source.close()
        print(
            f"[m1-backgrounds] {index['candidates']} candidates (stride {index['stride']}), "
            f"{len(index['auto_rejected_dark_or_blown'])} black or blown out, "
            f"{sum(len(s) for s in index['sheets'].values())} to review on "
            f"{len(index['sheets'])} sheets in {args.out}"
        )
        print(f"[m1-backgrounds] fill in {args.out / 'rejects.txt'}, then run collect")
        return 0

    result = collect(args.sheets, args.rejects, reviewer=args.reviewer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[m1-backgrounds] {result['counts']} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["candidates", "collect", "make_sheets", "parse_rejects", "render_sheet", "tile_name"]
