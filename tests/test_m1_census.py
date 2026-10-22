"""M1 census: tool-free CholecT50 frames and admissible CholecSeg8k test videos."""

from __future__ import annotations

import json
import zipfile

import pytest

from osv.datasets import m1_census as M

NONE = [-1] * 14


def vector(instrument: int, phase: int) -> list[int]:
    return [0, instrument, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, phase]


def video_labels(frames: dict[int, list[list[int]]]) -> dict:
    return {
        "categories": {"phase": {"0": "preparation", "1": "calot-triangle-dissection"}},
        "annotations": {str(index): vectors for index, vectors in frames.items()},
    }


LABELS = {
    "VID01": video_labels(
        {
            0: [NONE + [0]],
            1: [NONE + [0]],
            2: [vector(0, 0)],  # grasper
            3: [NONE + [1]],
            4: [vector(2, 1), vector(0, 1)],  # hook + grasper
        }
    ),
    "VID12": video_labels({0: [vector(2, 1)], 1: [vector(5, 1)]}),
}
SPLIT = {"VID01": "train", "VID12": "val"}


def test_a_frame_is_tool_free_only_when_no_vector_names_an_instrument():
    assert M.frame_instruments([NONE + [0]], where="f") == set()
    assert M.frame_instruments([vector(2, 1), vector(0, 1)], where="f") == {0, 2}
    with pytest.raises(ValueError, match="unknown instrument id 9"):
        M.frame_instruments([vector(9, 1)], where="f")
    with pytest.raises(ValueError, match="length 3"):
        M.frame_instruments([[0, 1, 1]], where="f")


def test_census_counts_per_video_split_phase_and_runs():
    report = M.census(LABELS, SPLIT)
    assert report["videos"]["VID01"] == {
        "split": "train",
        "frames": 5,
        "tool_free": 3,
        "runs": [[0, 1], [3, 3]],
    }
    assert report["by_split"]["train"] == {
        "frames": 5,
        "tool_free": 3,
        "videos": 1,
        "videos_with_tool_free": 1,
    }
    assert report["by_split"]["val"]["tool_free"] == 0
    assert report["by_phase"] == {"preparation": 2, "calot-triangle-dissection": 1}
    with pytest.raises(ValueError, match="missing from the official split"):
        M.census(LABELS, {"VID01": "train"})


def test_labels_are_read_from_the_archive_and_from_a_folder(tmp_path):
    archive = tmp_path / "CholecT50.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for video, labels in LABELS.items():
            handle.writestr(f"CholecT50/labels/{video}.json", json.dumps(labels))
        handle.writestr("CholecT50/videos/VID01/000000.png", b"")
    folder = tmp_path / "extracted" / "labels"
    folder.mkdir(parents=True)
    for video, labels in LABELS.items():
        (folder / f"{video}.json").write_text(json.dumps(labels))
    assert set(M.read_labels(archive)) == set(M.read_labels(tmp_path / "extracted")) == set(SPLIT)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        M.read_labels(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no labels"):
        M.read_labels(tmp_path / "empty")


def test_only_cholecseg8k_videos_absent_from_cholect50_are_admissible_for_test():
    seg = {"video01": 100, "video09": 50, "video12": 30, "video55": 20}
    result = M.admissible_test_videos(seg, ["VID01", "VID12", "VID92"])
    assert result["test"] == {"videos": ["video09", "video55"], "frames": 70}
    assert result["overlap_with_cholect50"] == {"videos": ["video01", "video12"], "frames": 130}
    assert M.cholec80_number("VID07") == M.cholec80_number("video07") == 7
    with pytest.raises(ValueError):
        M.cholec80_number("frame07")
