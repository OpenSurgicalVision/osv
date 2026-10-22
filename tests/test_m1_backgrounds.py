"""M1 background review: sheets of tool-free frames and the reviewer's rejects."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from osv.datasets import cholect50 as C
from osv.datasets import m1_backgrounds as B


def tissue(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.array([150, 50, 40], np.float32) + rng.normal(0, 12, (36, 64, 3))
    return np.clip(base, 0, 255).astype(np.uint8)


@pytest.fixture
def data(tmp_path):
    """40 tool-free train frames (one black), 2 val, 3 in test that must never appear."""
    root = tmp_path / "CholecT50"
    plan = {"VID01": ("train", 40), "VID12": ("val", 2), "VID06": ("test", 3)}
    videos = {}
    for video, (split, count) in plan.items():
        folder = root / "videos" / video
        folder.mkdir(parents=True)
        for index in range(count):
            rgb = (
                np.zeros((36, 64, 3), np.uint8) if (video, index) == ("VID01", 5) else tissue(index)
            )
            Image.fromarray(rgb).save(folder / f"{index:06d}.png")
        videos[video] = {
            "split": split,
            "frames": count,
            "tool_free": count,
            "runs": [[0, count - 1]],
        }
    census = tmp_path / "census.json"
    census.write_text(json.dumps({"cholect50": {"videos": videos}}))
    return root, census


def test_candidates_take_every_stride_th_frame_of_each_run_in_train_and_val():
    videos = {
        "VID01": {"split": "train", "runs": [[0, 6], [10, 10]]},
        "VID06": {"split": "test", "runs": [[0, 9]]},
    }
    assert B.candidates(videos, 3) == [("VID01", 0), ("VID01", 3), ("VID01", 6), ("VID01", 10)]


def test_sheets_drop_black_frames_and_number_every_tile(data, tmp_path):
    root, census = data
    out = tmp_path / "sheets"
    index = B.make_sheets(census, out, source=C.FrameSource(root), stride=1)
    assert index["auto_rejected_dark_or_blown"] == ["VID01_000005"]
    assert list(index["sheets"]) == ["001", "002"]
    assert len(index["sheets"]["001"]) == 36 and len(index["sheets"]["002"]) == 5
    assert index["sheets"]["001"]["1.1"] == "VID01_000000"
    assert not any("VID06" in f for tiles in index["sheets"].values() for f in tiles.values())
    with Image.open(out / "sheet_001.png") as sheet:
        assert sheet.size == (6 * 320, 6 * 180)
    assert (out / "rejects.txt").read_text().splitlines()[1:] == ["001:", "002:"]


def test_collect_needs_every_sheet_and_real_tiles(data, tmp_path):
    root, census = data
    out = tmp_path / "sheets"
    B.make_sheets(census, out, source=C.FrameSource(root), stride=1)
    rejects = out / "rejects.txt"

    rejects.write_text("001: 1.2, 6.6\n")
    with pytest.raises(ValueError, match="unreviewed"):
        B.collect(out, rejects, reviewer="R")
    rejects.write_text("001: 1.2\n002: 6.6\n")
    with pytest.raises(ValueError, match="no tile '6.6'"):
        B.collect(out, rejects, reviewer="R")
    rejects.write_text("001: 1.2\n001:\n002:\n")
    with pytest.raises(ValueError, match="twice"):
        B.collect(out, rejects, reviewer="R")

    rejects.write_text("# reviewed\n001: 1.2 6.6  # trocar, shaft\n002: 1.1\n")
    result = B.collect(out, rejects, reviewer="R")
    assert result["counts"]["reviewed"] == 41
    assert result["counts"]["rejected_by_reviewer"] == 3
    assert result["counts"]["accepted_train"] == 36 and result["counts"]["accepted_val"] == 2
    assert "VID01_000001" in result["rejected_by_reviewer"]
    assert "VID01_000001" not in result["accepted"]["train"]
