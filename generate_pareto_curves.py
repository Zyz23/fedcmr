#!/usr/bin/env python3
"""Generate reproducible cost--accuracy Pareto curves from the supplied logs.

The logs are Tiny-ImageNet experiments (90,000 training samples, 20 clients,
4 participants per round, 10 local epochs).  Costs count Conv/Linear MACs,
use 1 MAC = 2 FLOPs, and estimate a training step as 3x forward FLOPs.

FedCMR is the only method for which the frozen round-initial reference forward
is intentionally excluded, following the requested reporting convention.
Communication counts float32 model parameters in both directions only.
"""

from __future__ import annotations

import csv
import json
import re
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path("/workspace/scratch/78168ddb2e52")
LOG_DIR = ROOT / "upload"
OUT_DIR = ROOT / "pareto_curves_tiny"

N_ROUNDS = 200
SELECT_FRACTIONS = (0.15, 0.40, 0.65, 0.82, 0.94)

# Costs obtained analytically from HeteroCNNtiny / MetaCNN3_tiny.  The
# communication number is download + upload for all four selected clients.
METHODS = OrderedDict(
    [
        (
            "HeteroFL",
            {
                "file": "hetero(1).txt",
                "round_tflops": 63.28042488,
                "round_mb": 243.719952,
            },
        ),
        (
            "FedRolex",
            {
                "file": "fedrolex(1).txt",
                "round_tflops": 63.28042488,
                "round_mb": 243.719952,
            },
        ),
        (
            "FedDropout",
            {
                "file": "feddroupt(1).txt",
                "round_tflops": 63.28042488,
                "round_mb": 243.719952,
            },
        ),
        (
            "FedDF",
            {
                "file": "feddf(1).txt",
                "round_tflops": 72.08704712,
                "round_mb": 243.719952,
            },
        ),
        (
            "FedMD",
            {
                "file": "fedmd(1).txt",
                "round_tflops": 72.08704712,
                "round_mb": 243.719952,
            },
        ),
        (
            "FjORD",
            {
                "file": "fjord(1).txt",
                "round_tflops": 91.00875132,
                "round_mb": 243.719952,
            },
        ),
        (
            "FIARSE",
            {
                "file": "fiarse(1).txt",
                "round_tflops": 74.14530048,
                "round_mb": 284.148992,
            },
        ),
        (
            "FedDSE",
            {
                "file": "feddse(1).txt",
                "round_tflops": 63.63192852672,
                "round_mb": 263.934472,
            },
        ),
        (
            "FedGML",
            {
                "file": "fedgml(1).txt",
                "round_tflops": 74.14530048,
                "round_mb": 284.148992,
            },
        ),
        (
            "FedCMR (Ours)",
            {
                "file": "fedcmr(1).txt",
                "dynamic": True,
            },
        ),
    ]
)

# FedCMR's 20 fixed client ranks, reconstructed from the model-creation list
# and checked against every [MASTER SEND] rank/numel record in the log.
FEDCMR_CLIENT_RANK = (
    0.75,
    1.00,
    1.00,
    0.25,
    0.75,
    1.00,
    0.25,
    0.50,
    0.75,
    0.75,
    0.75,
    0.25,
    0.25,
    1.00,
    1.00,
    0.25,
    0.75,
    0.75,
    0.50,
    1.00,
)

# Per selected FedCMR client for 4,500 local samples x 10 epochs.  The
# calculation includes only the active local model's training graph.
FEDCMR_CLIENT_TFLOPS = {
    0.25: 16.92610560,
    0.50: 17.51003136,
    0.75: 18.09395712,
    1.00: 18.60710400,
}

# Bidirectional float32 model size for one selected client.  These are unique
# trainable parameter counts, not duplicated state_dict aliases.
FEDCMR_CLIENT_MB = {
    0.25: 22.514184,
    0.50: 39.815688,
    0.75: 57.117192,
    1.00: 72.321544,
}

TOP1_RE = re.compile(
    r"The validation performance = .*?'top1':\s*([0-9]+(?:\.[0-9]+)?)"
)
SELECTED_RE = re.compile(
    r"Master selected 4 from 20 clients:\s*\[([^\]]+)\]"
)

STYLES = {
    "HeteroFL": ("#7f7f7f", "s", "-"),
    "FedRolex": ("#bcbd22", "o", "-"),
    "FedDropout": ("#1f77b4", "^", "-"),
    "FedDF": ("#9467bd", "v", "-"),
    "FedMD": ("#17becf", "D", "-"),
    "FjORD": ("#8c564b", ">", "-"),
    "FIARSE": ("#7a8b00", "h", "-"),
    "FedDSE": ("#2ca02c", "*", "-"),
    "FedGML": ("#e377c2", "x", "-"),
    "FedCMR (Ours)": ("#e41a1c", "p", "-"),
}


def parse_accuracies(path: Path) -> list[float]:
    values = [float(value) for value in TOP1_RE.findall(path.read_text(errors="ignore"))]
    if len(values) != N_ROUNDS:
        raise RuntimeError(f"Expected {N_ROUNDS} validation records in {path}, got {len(values)}")
    return values


def running_best(values: list[float]) -> list[float]:
    result = []
    best = float("-inf")
    for value in values:
        best = max(best, value)
        result.append(best)
    return result


def parse_fedcmr_selected_clients(path: Path) -> list[list[int]]:
    rounds = []
    for payload in SELECTED_RE.findall(path.read_text(errors="ignore")):
        clients = [int(token.strip()) for token in payload.split(",")]
        if len(clients) != 4:
            raise RuntimeError(f"Malformed FedCMR client selection: {payload}")
        rounds.append(clients)
    if len(rounds) != N_ROUNDS:
        raise RuntimeError(f"Expected {N_ROUNDS} FedCMR selections, got {len(rounds)}")
    return rounds


def cumulative_costs(method: str, config: dict, log_path: Path) -> tuple[list[float], list[float]]:
    if not config.get("dynamic", False):
        compute = [config["round_tflops"] * round_id for round_id in range(1, N_ROUNDS + 1)]
        communication = [config["round_mb"] * round_id for round_id in range(1, N_ROUNDS + 1)]
        return compute, communication

    selections = parse_fedcmr_selected_clients(log_path)
    compute = []
    communication = []
    cumulative_compute = 0.0
    cumulative_communication = 0.0

    for clients in selections:
        ranks = [FEDCMR_CLIENT_RANK[client_id] for client_id in clients]
        cumulative_compute += sum(FEDCMR_CLIENT_TFLOPS[rank] for rank in ranks)
        cumulative_communication += sum(FEDCMR_CLIENT_MB[rank] for rank in ranks)
        compute.append(cumulative_compute)
        communication.append(cumulative_communication)

    return compute, communication


def select_representative_points(best: list[float]) -> list[dict]:
    maximum = max(best)
    selected = []
    used_rounds = set()

    for fraction in SELECT_FRACTIONS:
        target = maximum * fraction
        round_id = next(index for index, value in enumerate(best, start=1) if value >= target)
        if round_id in used_rounds:
            continue
        used_rounds.add(round_id)
        selected.append(
            {
                "target_fraction": fraction,
                "round": round_id,
                "accuracy": best[round_id - 1],
            }
        )

    return selected


def build_records() -> tuple[dict, list[dict], list[dict]]:
    method_data = OrderedDict()
    all_rounds = []
    selected_rows = []

    for method, config in METHODS.items():
        log_path = LOG_DIR / config["file"]
        raw_accuracy = parse_accuracies(log_path)
        best_accuracy = running_best(raw_accuracy)
        compute, communication = cumulative_costs(method, config, log_path)
        selected = select_representative_points(best_accuracy)

        for row in selected:
            index = row["round"] - 1
            row["compute_tflops"] = compute[index]
            row["communication_mb"] = communication[index]
            selected_rows.append({"method": method, **row})

        for round_id, (raw, best, comp, comm) in enumerate(
            zip(raw_accuracy, best_accuracy, compute, communication), start=1
        ):
            all_rounds.append(
                {
                    "method": method,
                    "round": round_id,
                    "raw_accuracy": raw,
                    "running_best_accuracy": best,
                    "cumulative_compute_tflops": comp,
                    "cumulative_communication_mb": comm,
                }
            )

        method_data[method] = {
            "raw_accuracy": raw_accuracy,
            "best_accuracy": best_accuracy,
            "compute": compute,
            "communication": communication,
            "selected": selected,
            "observed_max": max(raw_accuracy),
            "best_round": raw_accuracy.index(max(raw_accuracy)) + 1,
        }

    return method_data, all_rounds, selected_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(method_data: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": "Nimbus Roman",
            "font.size": 11,
            "axes.labelsize": 15,
            "axes.labelweight": "bold",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.15), sharey=True)
    metric_specs = (
        ("compute_tflops", "Computational Cost (TFLOPs)"),
        ("communication_mb", "Communication Cost (MB)"),
    )

    handles = []
    labels = []

    for axis, (metric, xlabel) in zip(axes, metric_specs):
        for method, data in method_data.items():
            color, marker, linestyle = STYLES[method]
            selected = data["selected"]
            x = [0.0] + [row[metric] for row in selected]
            y = [0.0] + [row["accuracy"] for row in selected]
            ours = method == "FedCMR (Ours)"

            line, = axis.plot(
                x,
                y,
                color=color,
                marker=marker,
                markevery=range(1, len(x)),
                linestyle=linestyle,
                linewidth=2.7 if ours else 1.45,
                markersize=8.5 if ours else 5.7,
                markeredgewidth=1.2 if ours else 0.8,
                markerfacecolor=color if marker not in {"x", "+"} else "none",
                label=method,
                zorder=10 if ours else 3,
            )

            if axis is axes[0]:
                handles.append(line)
                labels.append(method)

        axis.set_xlabel(xlabel)
        axis.grid(True, linestyle="--", linewidth=0.55, alpha=0.25)
        axis.set_axisbelow(True)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        axis.set_xlim(left=0)
        axis.tick_params(direction="out", length=4, width=1)

    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 33)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=True,
        fancybox=False,
        edgecolor="#777777",
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=1.35,
        handlelength=2.1,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.98, bottom=0.235, wspace=0.18)

    for suffix in ("png", "pdf", "svg", "eps"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(OUT_DIR / f"pareto_curves.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_summary(method_data: dict, selected_rows: list[dict]) -> None:
    lines = [
        "# Pareto curve node audit",
        "",
        "Dataset: Tiny-ImageNet; 20 clients; participation ratio 0.2; 10 local epochs; 200 rounds.",
        "",
        "Five nodes per method are selected reproducibly at the first round reaching "
        "15%, 40%, 65%, 82%, and 94% of that method's observed maximum running-best accuracy.",
        "",
        "FedCMR computation excludes the frozen round-initial model forward. Other methods "
        "retain their method-specific computation. Communication is bidirectional float32 "
        "model size only.",
        "",
        "| Method | Observed max (%) | Best round | Selected rounds |",
        "|---|---:|---:|---|",
    ]

    for method, data in method_data.items():
        rounds = ", ".join(str(row["round"]) for row in data["selected"])
        lines.append(
            f"| {method} | {data['observed_max']:.2f} | {data['best_round']} | {rounds} |"
        )

    lines.extend(
        [
            "",
            "## Selected nodes",
            "",
            "| Method | Round | Accuracy (%) | Compute (TFLOPs) | Communication (MB) |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for row in selected_rows:
        lines.append(
            f"| {row['method']} | {row['round']} | {row['accuracy']:.2f} | "
            f"{row['compute_tflops']:.2f} | {row['communication_mb']:.2f} |"
        )

    (OUT_DIR / "pareto_curve_node_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    method_data, all_rounds, selected_rows = build_records()

    write_csv(
        OUT_DIR / "pareto_all_rounds.csv",
        all_rounds,
        [
            "method",
            "round",
            "raw_accuracy",
            "running_best_accuracy",
            "cumulative_compute_tflops",
            "cumulative_communication_mb",
        ],
    )
    write_csv(
        OUT_DIR / "pareto_selected_points.csv",
        selected_rows,
        [
            "method",
            "target_fraction",
            "round",
            "accuracy",
            "compute_tflops",
            "communication_mb",
        ],
    )

    json_payload = {
        method: {
            "observed_max": data["observed_max"],
            "best_round": data["best_round"],
            "selected": data["selected"],
        }
        for method, data in method_data.items()
    }
    (OUT_DIR / "pareto_selected_points.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    plot_curves(method_data)
    write_summary(method_data, selected_rows)


if __name__ == "__main__":
    main()
