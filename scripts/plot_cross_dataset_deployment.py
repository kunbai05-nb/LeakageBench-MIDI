#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


def value(row: dict, name: str) -> float | None:
    raw = row.get(name, "")
    return None if raw in ("", "None") else float(raw)


def percent(number: float | None) -> str:
    return "--" if number is None else f"{100 * number:.1f}"


def latex(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: float(row["detected_random_test_file_exposure_mean"]))

    font_file = os.environ.get("LEAKAGEBENCH_FIGURE_FONT")
    if font_file:
        mpl.font_manager.fontManager.addfont(font_file)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 3.35), constrained_layout=True)
    y = list(range(len(rows)))
    detected = [100 * float(row["detected_random_test_file_exposure_mean"]) for row in rows]
    low = [100 * float(row["detected_random_test_file_exposure_q025"]) for row in rows]
    high = [100 * float(row["detected_random_test_file_exposure_q975"]) for row in rows]
    axes[0].errorbar(
        detected,
        y,
        xerr=[[point - bound for point, bound in zip(detected, low)],
              [bound - point for point, bound in zip(detected, high)]],
        fmt="o",
        color="#0072B2",
        capsize=2,
        markersize=4.5,
        label="Detector-inferred random split",
    )
    reference_label = True
    official_label = True
    valid_label = True
    for position, row in enumerate(rows):
        reference = value(row, "reference_random_test_file_exposure")
        if reference is not None:
            axes[0].scatter(
                100 * reference,
                position,
                marker="s",
                s=22,
                facecolors="white",
                edgecolors="#666666",
                label="Metadata/reference relation" if reference_label else None,
                zorder=3,
            )
            reference_label = False
        official = value(row, "official_test_file_exposure")
        if official is not None:
            axes[0].scatter(
                100 * official,
                position,
                marker="^",
                s=26,
                color="#D55E00",
                label="Official split" if official_label else None,
                zorder=3,
            )
            official_label = False
        valid = value(row, "valid_random_test_file_exposure_mean")
        if valid is not None and not math.isclose(valid * 100, detected[position], abs_tol=0.05):
            axes[0].scatter(
                100 * valid,
                position,
                marker="o",
                s=28,
                facecolors="white",
                edgecolors="#0072B2",
                label="Valid MIDI subset" if valid_label else None,
                zorder=3,
            )
            valid_label = False
    axes[0].set_yticks(y, [row["dataset"] for row in rows])
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Test-file exposure (%)")
    axes[0].grid(axis="x", color="#dddddd", linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    axes[0].text(-0.2, 1.04, "a", transform=axes[0].transAxes, fontweight="bold")

    sizes = [float(row["files"]) for row in rows]
    runtimes = [float(row["runtime_seconds"]) / 60 for row in rows]
    scatter = axes[1].scatter(
        sizes,
        runtimes,
        c=detected,
        cmap="viridis",
        vmin=0,
        vmax=100,
        s=34,
        edgecolor="black",
        linewidth=0.5,
    )
    offsets = {
        "ASAP": (5, 5),
        "MAESTRO": (5, -1),
        "POP909": (5, 6),
        "LMD": (-30, -10),
        "PDMX": (6, 8),
        "GigaMIDI": (6, 4),
        "Aria-MIDI Full": (6, 5),
    }
    for row, x, runtime in zip(rows, sizes, runtimes):
        axes[1].annotate(
            row["dataset"],
            (x, runtime),
            xytext=offsets[row["dataset"]],
            textcoords="offset points",
            fontsize=7,
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Dataset size (MIDI files)")
    axes[1].set_ylabel("CPU runtime (min)")
    axes[1].grid(color="#dddddd", linewidth=0.6, which="both")
    colorbar = figure.colorbar(scatter, ax=axes[1], pad=0.02)
    colorbar.set_label("Random test-file exposure (%)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    axes[1].text(-0.18, 1.04, "b", transform=axes[1].transAxes, fontweight="bold")

    figure_dir = args.output_dir / "figures"
    table_dir = args.output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(
            figure_dir / f"fig_cross_dataset_deployment.{suffix}",
            dpi=600,
            bbox_inches="tight",
        )
    plt.close(figure)

    table_data = []
    table_rows = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cross-dataset deployment of the frozen structural detector. Random test exposure is averaged over 200 fixed 80/10/10 file splits; brackets give empirical 95\% intervals.}",
        r"\label{tab:cross-dataset-detector}",
        r"\footnotesize",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrcrr}",
        r"\toprule",
        r"Dataset & MIDI files & Valid MIDI & Multi-version files & Random test exposure & Official exposure & Wall time \\",
        r" & & (\%) & (\%) & \% [95\% interval] & (\%) & (min) \\",
        r"\midrule",
    ]
    for row in reversed(rows):
        mean = 100 * float(row["detected_random_test_file_exposure_mean"])
        low = 100 * float(row["detected_random_test_file_exposure_q025"])
        high = 100 * float(row["detected_random_test_file_exposure_q975"])
        table_data.append(
            {
                "Dataset": row["dataset"],
                "MIDI files": f"{int(row['files']):,}",
                "Valid MIDI (%)": f"{100 * float(row['valid_file_rate']):.1f}",
                "Multi-version files (%)": f"{100 * float(row['inferred_multi_member_file_rate']):.1f}",
                "Random test exposure, % [95% EI]": f"{mean:.1f} [{low:.1f}, {high:.1f}]",
                "Official exposure (%)": percent(value(row, "official_test_file_exposure")),
                "Wall time (min)": f"{float(row['runtime_seconds']) / 60:.1f}",
            }
        )
        table_rows.append(
            f"{latex(row['dataset'])} & {int(row['files']):,} & "
            f"{100 * float(row['valid_file_rate']):.1f} & "
            f"{100 * float(row['inferred_multi_member_file_rate']):.1f} & "
            f"{mean:.1f} [{low:.1f}, {high:.1f}] & "
            f"{percent(value(row, 'official_test_file_exposure'))} & "
            f"{float(row['runtime_seconds']) / 60:.1f} \\\\"
        )
    table_rows.extend(
        [
            r"\bottomrule",
            r"\end{tabular*}",
            r"\vspace{2pt}",
            r"\parbox{\textwidth}{\footnotesize Multi-version files belong to inferred components containing at least two files. GigaMIDI's random exposure is 54.5\% when restricted to files with extractable non-drum notes. Wall-clock times were measured on the same 128-vCPU Linux server. Feature extraction used 8 workers for POP909, 4 for ASAP and MAESTRO, 16 for PDMX and LMD, and 24 for GigaMIDI and Aria-MIDI Full; FAISS used one thread.}",
            r"\end{table*}",
        ]
    )
    (table_dir / "table_cross_dataset_detector.tex").write_text(
        "\n".join(table_rows) + "\n"
    )
    with (table_dir / "table_cross_dataset_detector.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_data[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table_data)


if __name__ == "__main__":
    main()
