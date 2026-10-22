"""M1 census: tool-free CholecT50 frames, and the CholecSeg8k videos left for a clean test.

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

    python -m osv.datasets.m1_census --out /outside/the/repo/m1_census.json

Inputs come from outside the repository (R-01/R-02): CHOLECT50_RAW_ZIP or CHOLECT50_DIR
for CholecT50, CHOLECSEG8K_RAW_ZIP (or --cholecseg8k) for CholecSeg8k. Nothing is
written but the report.

RFC 0001 trains a two-class instrument segmenter (grasper, L-hook) only on Blender
instruments composited over CholecT50 frames in which no instrument is visible, and
tests it on real CholecSeg8k frames. Two numbers decide whether that is feasible and
fair, and this census reports both:

* **Tool-free frames.** CholecT50 labels every frame (1 fps) with triplet vectors
  ``[triplet, instrument, ..., phase]`` (15 values; instrument ids 0-5, -1 for none).
  A frame is tool-free when no vector names an instrument. They are rare in a
  cholecystectomy and cluster in a few phases, so the report counts them per video,
  per official split, per phase, and as runs of consecutive frames (one long run is
  one scene, not many backgrounds).
* **The clean test videos.** CholecSeg8k and CholecT50 both come from Cholec80, and
  several videos are in both. A test video whose frames also served as synthetic
  backgrounds would leak; the report lists the CholecSeg8k videos absent from
  CholecT50 (the only admissible test videos) and the overlapping ones (real-label
  training only).

An unknown instrument id or a vector of the wrong shape is an error, not a frame to skip:
the label format is then not the one this census was written for.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

INSTRUMENTS = ("grasper", "bipolar", "hook", "scissors", "clipper", "irrigator")
NO_INSTRUMENT = -1
VECTOR_LENGTH = 15
PHASE_INDEX = 14
LABEL_FILE = re.compile(r"(?:^|/)labels/(VID\d+)\.json$")
SEG_VIDEO = re.compile(r"^video(\d+)$")


def read_labels(path: str | Path) -> dict[str, dict[str, Any]]:
    """video -> parsed labels/VIDnn.json, from the CholecT50 archive or its extracted folder."""
    path = Path(path)
    found: dict[str, dict[str, Any]] = {}
    if path.is_file():
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                match = LABEL_FILE.search(name)
                if match:
                    found[match[1]] = json.loads(archive.read(name))
    else:
        for file in sorted(path.rglob("*.json")):
            match = LABEL_FILE.search(file.relative_to(path).as_posix())
            if match:
                found[match[1]] = json.loads(file.read_text())
    if not found:
        raise FileNotFoundError(f"no labels/VIDnn.json under {path}")
    return found


def frame_instruments(vectors: Sequence[Sequence[float]], *, where: str) -> set[int]:
    """Instrument ids visible in one frame; empty for a tool-free frame."""
    present = set()
    for vector in vectors:
        if len(vector) != VECTOR_LENGTH:
            raise ValueError(f"{where}: vector of length {len(vector)}, expected {VECTOR_LENGTH}")
        instrument = int(vector[1])
        if instrument == NO_INSTRUMENT:
            continue
        if not 0 <= instrument < len(INSTRUMENTS):
            raise ValueError(f"{where}: unknown instrument id {instrument}")
        present.add(instrument)
    return present


def frame_phase(vectors: Sequence[Sequence[float]]) -> int | None:
    phases = {int(vector[PHASE_INDEX]) for vector in vectors if len(vector) > PHASE_INDEX}
    phases.discard(-1)
    return phases.pop() if len(phases) == 1 else None


def _phase_names(labels: Mapping[str, Any]) -> dict[int, str]:
    names = labels.get("categories", {}).get("phase", {})
    return {int(key): str(value) for key, value in names.items()} if isinstance(names, dict) else {}


def runs(indices: Iterable[int]) -> list[list[int]]:
    """[first, last] of each run of consecutive frame indices."""
    result: list[list[int]] = []
    for index in sorted(indices):
        if result and index == result[-1][1] + 1:
            result[-1][1] = index
        else:
            result.append([index, index])
    return result


def census(labels: Mapping[str, Mapping[str, Any]], split_of: Mapping[str, str]) -> dict:
    unknown = sorted(set(labels) - set(split_of))
    if unknown:
        raise ValueError(f"videos missing from the official split: {unknown}")
    videos: dict[str, dict[str, Any]] = {}
    by_split: dict[str, Counter] = {}
    by_phase: Counter = Counter()
    phase_names: dict[int, str] = {}
    for video in sorted(labels):
        annotations = labels[video]["annotations"]
        phase_names |= _phase_names(labels[video])
        free = []
        for key, vectors in annotations.items():
            where = f"{video} frame {key}"
            if not frame_instruments(vectors, where=where):
                free.append(int(key))
                by_phase[frame_phase(vectors)] += 1
        split = split_of[video]
        totals = by_split.setdefault(split, Counter())
        totals["frames"] += len(annotations)
        totals["tool_free"] += len(free)
        totals["videos"] += 1
        totals["videos_with_tool_free"] += bool(free)
        videos[video] = {
            "split": split,
            "frames": len(annotations),
            "tool_free": len(free),
            "runs": runs(free),
        }
    return {
        "by_split": {split: dict(counts) for split, counts in sorted(by_split.items())},
        "by_phase": {
            phase_names.get(phase, str(phase)) if phase is not None else "unknown": count
            for phase, count in sorted(by_phase.items(), key=lambda kv: (kv[0] is None, kv[0]))
        },
        "videos": videos,
    }


def cholec80_number(video: str) -> int:
    """'VID12' or 'video12' -> 12: both datasets keep Cholec80's video numbers."""
    match = re.fullmatch(r"(?:VID|video)(\d+)", video)
    if not match:
        raise ValueError(f"not a CholecT50 or CholecSeg8k video id: {video!r}")
    return int(match[1])


def admissible_test_videos(seg_frames: Mapping[str, int], cholect50_videos: Iterable[str]) -> dict:
    """CholecSeg8k videos split into the admissible test set and the overlap with CholecT50."""
    in_t50 = {cholec80_number(video) for video in cholect50_videos}
    test = sorted(v for v in seg_frames if cholec80_number(v) not in in_t50)
    overlap = sorted(v for v in seg_frames if cholec80_number(v) in in_t50)
    return {
        "test": {"videos": test, "frames": sum(seg_frames[v] for v in test)},
        "overlap_with_cholect50": {
            "videos": overlap,
            "frames": sum(seg_frames[v] for v in overlap),
        },
        "frames_per_video": dict(sorted(seg_frames.items())),
    }


def _seg_frames(source: str | Path) -> dict[str, int]:
    from osv.datasets import cholecseg8k as S

    path = Path(source)
    records = S.iter_frames_from_zip(path) if path.is_file() else S.iter_frames_from_dir(path)
    return dict(Counter(record.video_id for record in records))


def main(argv: list[str] | None = None) -> int:
    from osv.datasets import cholect50 as C

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="JSON report, outside the repo")
    parser.add_argument(
        "--cholect50",
        default=os.environ.get("CHOLECT50_RAW_ZIP") or os.environ.get("CHOLECT50_DIR"),
    )
    parser.add_argument("--cholecseg8k", default=os.environ.get("CHOLECSEG8K_RAW_ZIP"))
    args = parser.parse_args(argv)
    if not args.cholect50:
        parser.error("set CHOLECT50_RAW_ZIP / CHOLECT50_DIR or pass --cholect50")
    split_of = C.official_split(C.load_manifest())
    report: dict[str, Any] = {"cholect50": census(read_labels(args.cholect50), split_of)}
    if args.cholecseg8k:
        report["cholecseg8k"] = admissible_test_videos(_seg_frames(args.cholecseg8k), split_of)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    for split, counts in report["cholect50"]["by_split"].items():
        share = counts["tool_free"] / max(counts["frames"], 1)
        print(
            f"[m1-census] {split}: {counts['tool_free']} tool-free of {counts['frames']} frames "
            f"({share:.1%}), in {counts['videos_with_tool_free']}/{counts['videos']} videos"
        )
    print(f"[m1-census] tool-free by phase: {report['cholect50']['by_phase']}")
    if "cholecseg8k" in report:
        seg = report["cholecseg8k"]
        print(
            f"[m1-census] CholecSeg8k test videos (absent from CholecT50): "
            f"{seg['test']['videos']} = {seg['test']['frames']} frames"
        )
        print(
            f"[m1-census] overlapping videos (real-label training only): "
            f"{seg['overlap_with_cholect50']['videos']} = "
            f"{seg['overlap_with_cholect50']['frames']} frames"
        )
    print(f"[m1-census] report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "INSTRUMENTS",
    "admissible_test_videos",
    "census",
    "cholec80_number",
    "frame_instruments",
    "frame_phase",
    "read_labels",
    "runs",
]
