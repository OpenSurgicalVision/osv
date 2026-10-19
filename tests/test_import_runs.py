"""Importing a GPU pod's local MLflow record into the tracking server keeps what R-24 reads."""

from __future__ import annotations

import pytest

pytest.importorskip("mlflow")

from mlflow.tracking import MlflowClient  # noqa: E402

from osv.tracking import import_runs as I  # noqa: E402

HOUR = 3_600_000


def pod_store(tmp_path):
    uri = f"sqlite:///{tmp_path / 'pod.db'}"
    client = MlflowClient(tracking_uri=uri)
    experiment = client.create_experiment(
        "osv-m5-desmoke-gan", artifact_location=str(tmp_path / "pod-artifacts")
    )
    runs = []
    for number, start in enumerate((1_000 * HOUR, 1_002 * HOUR)):
        run = client.create_run(experiment, start_time=start, run_name=f"pix2pix-{number}")
        client.log_param(run.info.run_id, "config_hash", "abc123")
        client.log_param(run.info.run_id, "gpu_count", "1")
        for step in range(3):
            client.log_metric(run.info.run_id, "psnr", 20.0 + step, step=step, timestamp=start)
        client.set_terminated(run.info.run_id, status="FINISHED", end_time=start + HOUR)
        runs.append(run.info.run_id)
    crashed = client.create_run(experiment, start_time=1_004 * HOUR, run_name="crashed")
    return uri, runs, crashed.info.run_id


def test_runs_are_copied_once_with_history_times_and_one_checkpoint(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # the target's default artifact root is ./mlruns
    source, (first, second), crashed = pod_store(tmp_path)
    outputs = tmp_path / "runs" / "pix2pix"
    outputs.mkdir(parents=True)
    (outputs / "report_abc123.json").write_text("{}")
    (outputs / "ckpt_abc123.pt").write_bytes(b"weights")
    target = f"sqlite:///{tmp_path / 'server.db'}"
    argv = ["--source", source, "--target", target, "--outputs", str(tmp_path / "runs")]

    assert I.main(argv + ["--dry-run"]) == 1  # the crashed run is reported, not copied
    assert MlflowClient(tracking_uri=target).search_experiments()[0].name == "Default"
    assert I.main(argv) == 1
    assert "not copied, RUNNING" in capsys.readouterr().err

    client = MlflowClient(tracking_uri=target)
    experiment = client.get_experiment_by_name("osv-m5-desmoke-gan")
    copied = {r.data.tags[I.SOURCE_TAG]: r for r in client.search_runs([experiment.experiment_id])}
    assert set(copied) == {first, second}
    for source_id, start in ((first, 1_000 * HOUR), (second, 1_002 * HOUR)):
        run = copied[source_id]
        assert (run.info.start_time, run.info.end_time) == (start, start + HOUR)
        assert run.info.status == "FINISHED"
        assert run.data.params == {"config_hash": "abc123", "gpu_count": "1"}
        history = client.get_metric_history(run.info.run_id, "psnr")
        assert [(m.step, m.value) for m in history] == [(0, 20.0), (1, 21.0), (2, 22.0)]
    # The resumed experiment shares one checkpoint: it goes to the latest run only.
    assert {f.path for f in client.list_artifacts(copied[second].info.run_id, "outputs")} == {
        "outputs/report_abc123.json",
        "outputs/ckpt_abc123.pt",
    }
    assert client.list_artifacts(copied[first].info.run_id) == []

    assert I.main(argv) == 1  # idempotent
    assert "copied 0, already imported 2, unfinished 1" in capsys.readouterr().out
    assert len(client.search_runs([experiment.experiment_id])) == 2


def test_ambiguous_outputs_and_same_store_are_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source, _, _ = pod_store(tmp_path)
    for session in ("a", "b"):
        (tmp_path / session).mkdir()
        (tmp_path / session / "ckpt_abc123.pt").write_bytes(b"w")
    with pytest.raises(SystemExit, match="one session"):
        I.find_outputs(tmp_path, "abc123")
    with pytest.raises(SystemExit, match="same store"):
        I.main(["--source", source, "--target", source])
