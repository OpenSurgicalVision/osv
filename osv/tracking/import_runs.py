"""Copy MLflow runs from a GPU pod's local store into the project tracking server.

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

    kubectl -n gpu port-forward svc/mlflow 5000:5000      # osv-infra ADR 0009
    python -m osv.tracking.import_runs \\
        --source sqlite:///$HOME/osv-runs/2026-10-20/mlflow.db \\
        --target http://localhost:5000 \\
        --outputs $HOME/osv-runs/2026-10-20/runs

A manually rented pod never reaches the tracking server (osv-infra ADR 0009): it records
into a local sqlite store, and the owner imports that record afterwards. The copy keeps
what R-24 and R-15 read: params, every metric step, tags, status, and the original start
and end times, which the quota gate turns into GPU-hours.

* Idempotent: each imported run carries `osv.import.source_run_id`; a run already on the
  target is skipped, so re-running after a partial import is safe.
* Unfinished runs (status RUNNING or SCHEDULED) are not copied: half a run would be
  counted as a finished one.
* `--outputs` attaches `report_<config_hash>.json` and `ckpt_<config_hash>.pt` found
  under that folder to the latest run with that config_hash (a resumed experiment is
  several runs sharing one checkpoint), so the checkpoint is uploaded once.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

SOURCE_TAG = "osv.import.source_run_id"
UNFINISHED = {"RUNNING", "SCHEDULED"}
BATCH = 900  # MLflow accepts at most 1000 entities per log_batch
OUTPUT_PATTERNS = ("report_{}.json", "ckpt_{}.pt")


def _client(uri: str):
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=uri)


def _pages(search, **kwargs) -> Iterator[Any]:
    token = None
    while True:
        page = search(page_token=token, **kwargs)
        yield from page
        token = page.token
        if not token:
            return


def _chunks(items: list, size: int = BATCH) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def source_runs(client) -> list[tuple[Any, Any]]:
    """(experiment, run) for every run of every experiment, oldest first."""
    from mlflow.entities import ViewType

    found = []
    for experiment in _pages(client.search_experiments, view_type=ViewType.ALL):
        for run in _pages(
            client.search_runs,
            experiment_ids=[experiment.experiment_id],
            run_view_type=ViewType.ALL,
            max_results=1000,
        ):
            found.append((experiment, run))
    return sorted(found, key=lambda pair: pair[1].info.start_time or 0)


def imported_ids(client) -> set[str]:
    from mlflow.entities import ViewType

    ids = set()
    for experiment in _pages(client.search_experiments, view_type=ViewType.ALL):
        for run in _pages(
            client.search_runs,
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.`{SOURCE_TAG}` != ''",
            run_view_type=ViewType.ALL,
            max_results=1000,
        ):
            ids.add(run.data.tags[SOURCE_TAG])
    return ids


def latest_by_config(runs: list[tuple[Any, Any]]) -> dict[str, str]:
    """config_hash -> source run_id of the latest finished run with it."""
    latest: dict[str, Any] = {}
    for _experiment, run in runs:
        config_hash = run.data.params.get("config_hash")
        if not config_hash or run.info.status in UNFINISHED:
            continue
        if config_hash not in latest or run.info.start_time > latest[config_hash].info.start_time:
            latest[config_hash] = run
    return {config_hash: run.info.run_id for config_hash, run in latest.items()}


def find_outputs(folder: Path | None, config_hash: str) -> list[Path]:
    if folder is None:
        return []
    found = []
    for pattern in OUTPUT_PATTERNS:
        matches = sorted(folder.rglob(pattern.format(config_hash)))
        if len(matches) > 1:
            raise SystemExit(
                f"[osv.import] {len(matches)} files named {pattern.format(config_hash)} under "
                f"{folder}: pass the folder of one session"
            )
        found += matches
    return found


def copy_run(source, target, experiment, run, *, outputs: list[Path]) -> str:
    from mlflow.entities import Param

    existing = target.get_experiment_by_name(experiment.name)
    experiment_id = (
        existing.experiment_id if existing else target.create_experiment(experiment.name)
    )
    tags = {
        key: value for key, value in run.data.tags.items() if not key.startswith("mlflow.log-model")
    }
    tags[SOURCE_TAG] = run.info.run_id
    created = target.create_run(
        experiment_id,
        start_time=run.info.start_time,
        tags=tags,
        run_name=run.info.run_name,
    )
    run_id = created.info.run_id
    params = [Param(key, value) for key, value in sorted(run.data.params.items())]
    metrics = [
        metric
        for key in sorted(run.data.metrics)
        for metric in source.get_metric_history(run.info.run_id, key)
    ]
    for chunk in _chunks(params, 100):
        target.log_batch(run_id, params=chunk)
    for chunk in _chunks(metrics):
        target.log_batch(run_id, metrics=chunk)
    for path in outputs:
        target.log_artifact(run_id, str(path), artifact_path="outputs")
    target.set_terminated(run_id, status=run.info.status, end_time=run.info.end_time)
    return run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--source", required=True, help="the pod's store, e.g. sqlite:///…/mlflow.db"
    )
    parser.add_argument(
        "--target", required=True, help="the tracking server, e.g. http://localhost:5000"
    )
    parser.add_argument("--outputs", type=Path, help="folder with report_*.json / ckpt_*.pt")
    parser.add_argument("--dry-run", action="store_true", help="list what would be copied")
    args = parser.parse_args(argv)
    if args.source == args.target:
        raise SystemExit("[osv.import] source and target are the same store")
    if args.outputs is not None and not args.outputs.is_dir():
        raise SystemExit(f"[osv.import] --outputs {args.outputs} is not a folder")
    source, target = _client(args.source), _client(args.target)
    runs = source_runs(source)
    done = imported_ids(target)
    owners = latest_by_config(runs)
    copied = skipped = unfinished = 0
    for experiment, run in runs:
        name = f"{experiment.name}/{run.info.run_name or run.info.run_id}"
        if run.info.status in UNFINISHED:
            unfinished += 1
            print(f"[osv.import] not copied, {run.info.status}: {name}", file=sys.stderr)
            continue
        if run.info.run_id in done:
            skipped += 1
            continue
        config_hash = run.data.params.get("config_hash", "")
        outputs = (
            find_outputs(args.outputs, config_hash)
            if config_hash and owners.get(config_hash) == run.info.run_id
            else []
        )
        attached = f" + {', '.join(p.name for p in outputs)}" if outputs else ""
        if args.dry_run:
            print(f"[osv.import] would copy {name} ({run.info.status}){attached}")
        else:
            new_id = copy_run(source, target, experiment, run, outputs=outputs)
            print(f"[osv.import] copied {name} -> {new_id}{attached}")
        copied += 1
    verb = "would copy" if args.dry_run else "copied"
    print(f"[osv.import] {verb} {copied}, already imported {skipped}, unfinished {unfinished}")
    return 1 if unfinished else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SOURCE_TAG", "copy_run", "find_outputs", "latest_by_config", "main", "source_runs"]
