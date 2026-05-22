from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BATCH_RESULTS_DIR = ROOT_DIR / "batch_results"
DEFAULT_OUTPUT_DIR = DEFAULT_BATCH_RESULTS_DIR / "_aggregate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate all task/backend batch_results into summary CSV files."
    )
    parser.add_argument(
        "--batch_results_dir",
        default=str(DEFAULT_BATCH_RESULTS_DIR),
        help="Directory containing per-task batch result folders.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where aggregate CSV/JSON/Markdown files will be written.",
    )
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def load_trial_records(batch_results_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for batch_dir in sorted(path for path in batch_results_dir.iterdir() if path.is_dir()):
        results_path = batch_dir / "results.json"
        if not results_path.exists():
            continue
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            record = dict(item)
            record["batch_dir_name"] = batch_dir.name
            records.append(record)
    return records


def build_summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        task = str(record.get("task") or "")
        backend = str(record.get("backend") or "")
        grouped[(task, backend)].append(record)

    rows: list[dict[str, Any]] = []
    for (task, backend), group in sorted(grouped.items()):
        success_count = sum(1 for item in group if item.get("final_task_success") is True)
        completed_count = sum(1 for item in group if item.get("status") == "completed")
        non_null_status_count = sum(1 for item in group if item.get("status") is not None)
        planner_rounds = [_safe_float(item.get("planner_round_count")) for item in group]
        planner_rounds = [value for value in planner_rounds if value is not None]
        actor_steps = [_safe_float(item.get("total_actor_steps")) for item in group]
        actor_steps = [value for value in actor_steps if value is not None]
        planner_tokens = [_safe_float(item.get("planner_tokens_total")) for item in group]
        planner_tokens = [value for value in planner_tokens if value is not None]
        actor_tokens = [_safe_float(item.get("actor_tokens_total")) for item in group]
        actor_tokens = [value for value in actor_tokens if value is not None]
        verifier_tokens = [_safe_float(item.get("verifier_tokens_total")) for item in group]
        verifier_tokens = [value for value in verifier_tokens if value is not None]
        total_tokens = [_safe_float(item.get("tokens_total")) for item in group]
        total_tokens = [value for value in total_tokens if value is not None]

        failure_status_counts: dict[str, int] = defaultdict(int)
        for item in group:
            status = item.get("status")
            if status is None:
                failure_status_counts["missing"] += 1
            elif status != "completed":
                failure_status_counts[str(status)] += 1

        rows.append(
            {
                "task": task,
                "backend": backend,
                "trial_count": len(group),
                "observed_status_count": non_null_status_count,
                "success_count": success_count,
                "success_rate": (success_count / len(group)) if group else 0.0,
                "completed_count": completed_count,
                "avg_planner_round_count": _mean(planner_rounds),
                "avg_total_actor_steps": _mean(actor_steps),
                "avg_planner_tokens_total": _mean(planner_tokens),
                "avg_actor_tokens_total": _mean(actor_tokens),
                "avg_verifier_tokens_total": _mean(verifier_tokens),
                "avg_tokens_total": _mean(total_tokens),
                "failure_status_breakdown": json.dumps(dict(sorted(failure_status_counts.items())), ensure_ascii=False),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_outputs(output_dir: Path, trial_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_fieldnames = [
        "task",
        "backend",
        "trial_count",
        "observed_status_count",
        "success_count",
        "success_rate",
        "completed_count",
        "avg_planner_round_count",
        "avg_total_actor_steps",
        "avg_planner_tokens_total",
        "avg_actor_tokens_total",
        "avg_verifier_tokens_total",
        "avg_tokens_total",
        "failure_status_breakdown",
    ]
    trial_fieldnames = [
        "task",
        "backend",
        "trial_index",
        "status",
        "final_task_success",
        "planner_round_count",
        "total_actor_steps",
        "planner_tokens_total",
        "actor_tokens_total",
        "verifier_tokens_total",
        "tokens_total",
        "completion_message",
        "run_dir",
        "batch_dir_name",
    ]

    write_csv(output_dir / "batch_results_summary.csv", summary_rows, summary_fieldnames)
    write_csv(output_dir / "batch_results_trials.csv", trial_rows, trial_fieldnames)
    (output_dir / "batch_results_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "batch_results_trials.json").write_text(
        json.dumps(trial_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    backends = sorted({str(row.get("backend") or "") for row in summary_rows})
    tasks = sorted({str(row.get("task") or "") for row in summary_rows})
    summary_lines = [
        "# Aggregated Batch Results",
        "",
        f"- Tasks: {len(tasks)}",
        f"- Backends: {', '.join(backends) if backends else 'None'}",
        f"- Task-backend groups: {len(summary_rows)}",
        f"- Trial rows: {len(trial_rows)}",
        "",
        "## Files",
        "",
        "- `batch_results_summary.csv`: one row per task-backend aggregate",
        "- `batch_results_trials.csv`: one row per trial",
        "- `batch_results_summary.json`: JSON version of the aggregate summary",
        "- `batch_results_trials.json`: JSON version of the trial details",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def _sorted_backends(rows: list[dict[str, Any]]) -> list[str]:
    preferred = ["none", "static", "dms"]
    seen = {str(row.get("backend") or "") for row in rows}
    ordered = [backend for backend in preferred if backend in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def _safe_label(task: str) -> str:
    return task.replace("/", "_").replace("\\", "_").replace(" ", "_")


def generate_plots(output_dir: Path, trial_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> list[str]:
    plt = _try_import_matplotlib()
    if plt is None:
        return []

    created: list[str] = []
    plots_dir = output_dir / "plots"
    per_task_dir = plots_dir / "per_task"
    plots_dir.mkdir(parents=True, exist_ok=True)
    per_task_dir.mkdir(parents=True, exist_ok=True)

    backends = _sorted_backends(summary_rows)

    def save_current(filename: str) -> None:
        path = plots_dir / filename
        plt.tight_layout()
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        created.append(str(path))

    if summary_rows:
        overall_by_backend: dict[str, dict[str, float]] = defaultdict(lambda: {"success_sum": 0.0, "count": 0.0, "tokens_sum": 0.0, "steps_sum": 0.0})
        for row in summary_rows:
            backend = str(row.get("backend") or "")
            overall_by_backend[backend]["success_sum"] += float(row.get("success_rate") or 0.0)
            overall_by_backend[backend]["tokens_sum"] += float(row.get("avg_tokens_total") or 0.0)
            overall_by_backend[backend]["steps_sum"] += float(row.get("avg_total_actor_steps") or 0.0)
            overall_by_backend[backend]["count"] += 1.0

        labels = backends
        success_values = [
            (overall_by_backend[backend]["success_sum"] / overall_by_backend[backend]["count"])
            if overall_by_backend[backend]["count"]
            else 0.0
            for backend in labels
        ]
        token_values = [
            (overall_by_backend[backend]["tokens_sum"] / overall_by_backend[backend]["count"])
            if overall_by_backend[backend]["count"]
            else 0.0
            for backend in labels
        ]
        step_values = [
            (overall_by_backend[backend]["steps_sum"] / overall_by_backend[backend]["count"])
            if overall_by_backend[backend]["count"]
            else 0.0
            for backend in labels
        ]

        plt.figure(figsize=(7, 4))
        plt.bar(labels, success_values, color=["#7f8c8d", "#3498db", "#2ecc71"][: len(labels)])
        plt.ylim(0, 1.0)
        plt.ylabel("Average Success Rate")
        plt.title("Average Success Rate by Backend")
        save_current("overall_success_rate_by_backend.png")

        plt.figure(figsize=(7, 4))
        plt.bar(labels, token_values, color=["#7f8c8d", "#3498db", "#2ecc71"][: len(labels)])
        plt.ylabel("Average Total Tokens")
        plt.title("Average Total Tokens by Backend")
        save_current("overall_avg_tokens_by_backend.png")

        plt.figure(figsize=(7, 4))
        plt.bar(labels, step_values, color=["#7f8c8d", "#3498db", "#2ecc71"][: len(labels)])
        plt.ylabel("Average Actor Steps")
        plt.title("Average Actor Steps by Backend")
        save_current("overall_avg_steps_by_backend.png")

        tasks = sorted({str(row.get("task") or "") for row in summary_rows})
        x_positions = list(range(len(tasks)))
        width = 0.25 if len(backends) >= 3 else 0.35
        backend_colors = {"none": "#7f8c8d", "static": "#3498db", "dms": "#2ecc71"}

        def plot_grouped(metric_key: str, ylabel: str, title: str, filename: str, ylim: tuple[float, float] | None = None) -> None:
            plt.figure(figsize=(max(10, len(tasks) * 0.8), 5))
            for backend_index, backend in enumerate(backends):
                series = []
                for task in tasks:
                    row = next((item for item in summary_rows if item.get("task") == task and item.get("backend") == backend), None)
                    value = float((row or {}).get(metric_key) or 0.0)
                    series.append(value)
                offset = (backend_index - ((len(backends) - 1) / 2.0)) * width
                plt.bar(
                    [position + offset for position in x_positions],
                    series,
                    width=width,
                    label=backend,
                    color=backend_colors.get(backend),
                )
            plt.xticks(x_positions, tasks, rotation=45, ha="right")
            plt.ylabel(ylabel)
            plt.title(title)
            if ylim is not None:
                plt.ylim(*ylim)
            plt.legend()
            save_current(filename)

        plot_grouped(
            "success_rate",
            "Success Rate",
            "Success Rate by Task and Backend",
            "task_backend_success_rate.png",
            ylim=(0.0, 1.0),
        )
        plot_grouped(
            "avg_tokens_total",
            "Average Total Tokens",
            "Average Total Tokens by Task and Backend",
            "task_backend_avg_tokens.png",
        )
        plot_grouped(
            "avg_total_actor_steps",
            "Average Actor Steps",
            "Average Actor Steps by Task and Backend",
            "task_backend_avg_steps.png",
        )

    grouped_trials: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trial_rows:
        grouped_trials[str(row.get("task") or "")].append(row)

    backend_colors = {"none": "#7f8c8d", "static": "#3498db", "dms": "#2ecc71"}

    for task, rows in sorted(grouped_trials.items()):
        rows_by_backend: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            rows_by_backend[str(row.get("backend") or "")].append(row)

        # cumulative success curve
        plt.figure(figsize=(7, 4))
        plotted = False
        for backend in _sorted_backends(rows):
            series = sorted(rows_by_backend.get(backend, []), key=lambda item: _safe_int(item.get("trial_index")) or 0)
            if not series:
                continue
            successes = 0
            x_vals: list[int] = []
            y_vals: list[float] = []
            for index, item in enumerate(series, start=1):
                if item.get("final_task_success") is True:
                    successes += 1
                x_vals.append(index)
                y_vals.append(successes / index)
            plt.plot(x_vals, y_vals, marker="o", label=backend, color=backend_colors.get(backend))
            plotted = True
        if plotted:
            plt.ylim(0.0, 1.0)
            plt.xlabel("Trial Index")
            plt.ylabel("Cumulative Success Rate")
            plt.title(f"{task}: Cumulative Success Rate")
            plt.legend()
            path = per_task_dir / f"{_safe_label(task)}_cumulative_success_rate.png"
            plt.tight_layout()
            plt.savefig(path, dpi=160, bbox_inches="tight")
            plt.close()
            created.append(str(path))
        else:
            plt.close()

        # token curve
        plt.figure(figsize=(7, 4))
        plotted = False
        for backend in _sorted_backends(rows):
            series = sorted(rows_by_backend.get(backend, []), key=lambda item: _safe_int(item.get("trial_index")) or 0)
            x_vals: list[int] = []
            y_vals: list[float] = []
            for item in series:
                trial_index = _safe_int(item.get("trial_index"))
                tokens_total = _safe_float(item.get("tokens_total"))
                if trial_index is None or tokens_total is None:
                    continue
                x_vals.append(trial_index)
                y_vals.append(tokens_total)
            if not x_vals:
                continue
            plt.plot(x_vals, y_vals, marker="o", label=backend, color=backend_colors.get(backend))
            plotted = True
        if plotted:
            plt.xlabel("Trial Index")
            plt.ylabel("Total Tokens")
            plt.title(f"{task}: Total Tokens by Trial")
            plt.legend()
            path = per_task_dir / f"{_safe_label(task)}_tokens_by_trial.png"
            plt.tight_layout()
            plt.savefig(path, dpi=160, bbox_inches="tight")
            plt.close()
            created.append(str(path))
        else:
            plt.close()

        # steps curve
        plt.figure(figsize=(7, 4))
        plotted = False
        for backend in _sorted_backends(rows):
            series = sorted(rows_by_backend.get(backend, []), key=lambda item: _safe_int(item.get("trial_index")) or 0)
            x_vals: list[int] = []
            y_vals: list[float] = []
            for item in series:
                trial_index = _safe_int(item.get("trial_index"))
                steps = _safe_float(item.get("total_actor_steps"))
                if trial_index is None or steps is None:
                    continue
                x_vals.append(trial_index)
                y_vals.append(steps)
            if not x_vals:
                continue
            plt.plot(x_vals, y_vals, marker="o", label=backend, color=backend_colors.get(backend))
            plotted = True
        if plotted:
            plt.xlabel("Trial Index")
            plt.ylabel("Actor Steps")
            plt.title(f"{task}: Actor Steps by Trial")
            plt.legend()
            path = per_task_dir / f"{_safe_label(task)}_steps_by_trial.png"
            plt.tight_layout()
            plt.savefig(path, dpi=160, bbox_inches="tight")
            plt.close()
            created.append(str(path))
        else:
            plt.close()

    return created


def main() -> int:
    args = parse_args()
    batch_results_dir = Path(args.batch_results_dir)
    output_dir = Path(args.output_dir)

    if not batch_results_dir.exists():
        raise SystemExit(f"batch_results_dir does not exist: {batch_results_dir}")

    trial_rows = load_trial_records(batch_results_dir)
    summary_rows = build_summary_rows(trial_rows)
    write_outputs(output_dir, trial_rows, summary_rows)
    plot_paths = generate_plots(output_dir, trial_rows, summary_rows)

    print(f"Loaded {len(trial_rows)} trial rows from: {batch_results_dir}")
    print(f"Wrote aggregate outputs to: {output_dir}")
    print(f"Summary CSV: {output_dir / 'batch_results_summary.csv'}")
    print(f"Trial CSV: {output_dir / 'batch_results_trials.csv'}")
    print(f"Generated plots: {len(plot_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
