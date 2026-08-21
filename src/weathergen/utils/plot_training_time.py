"""
plot_training_time.py

Plots training loss with wall-clock time on the x-axis.

Instead of approximating time from throughput, we directly linearly map
each stage's num_samples range to its exact known wall-clock duration:

    t_i = t_start + (s_i - s_min) / (s_max - s_min) * stage_duration_hours

Known durations:
    Curriculum: L3=30min, L4=30min, L5=90min, L6=210min  (total: 6h)
    Baseline:   6h total
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Add WeatherGenerator to path
wg_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../WeatherGenerator/src"))
sys.path.append(wg_src)

try:
    from weathergen.utils.train_logger import TrainLogger
except ImportError as e:
    print(f"Could not import WeatherGenerator modules: {e}")
    sys.exit(1)

# ── Styling ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
})

# ── Stage definitions ──────────────────────────────────────────────────────────
# Each entry: run_id -> label, color, duration in hours, linestyle
CURRICULUM_STAGES = [
    ("ylmg7r8y", "Stage 1 (Level 3)", 30 / 60),
    ("w7s3sxbj", "Stage 2 (Level 4)", 30 / 60),
    ("yxvv3tvj", "Stage 3 (Level 5)", 90 / 60),
    ("uak7v5j0", "Stage 4 (Level 6)", 210 / 60),
]

BASELINE_STAGE = ("wmn145xq", "Baseline (Level 6)", 6.0)

COLORS = sns.color_palette("husl", len(CURRICULUM_STAGES))


def load_run(run_id: str, model_base_dir: Path):
    """Load training loss data for a run. Returns (samples_array, loss_array) or (None, None)."""
    run_data = TrainLogger.read(run_id, model_path=model_base_dir, cols_patterns=["loss_avg_mean"])
    if run_data.train.is_empty():
        print(f"  Warning: no training data found for run '{run_id}'")
        return None, None

    raw_samples = np.array(run_data.train["num_samples"])
    y_vals = np.array(run_data.train["loss_avg_mean"])
    mask = ~np.isnan(raw_samples) & ~np.isnan(y_vals)
    return raw_samples[mask], y_vals[mask]


def samples_to_time(samples, duration_hours, cumulative_start_hours=0.0):
    """
    Linearly map a samples array to wall-clock time (hours).

    Maps [samples[0], samples[-1]]  ->  [cumulative_start, cumulative_start + duration_hours].
    """
    s_min, s_max = samples[0], samples[-1]
    if s_max == s_min:
        # degenerate case: all points at same sample count
        return np.full_like(samples, cumulative_start_hours, dtype=float)
    fraction = (samples - s_min) / (s_max - s_min)
    return cumulative_start_hours + fraction * duration_hours


def plot_training_loss_time(model_base_dir: Path, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    # ── Curriculum (cumulative) ────────────────────────────────────────────────
    cumulative_hours = 0.0
    for i, (run_id, label, duration_hours) in enumerate(CURRICULUM_STAGES):
        print(f"Loading curriculum run '{run_id}' ({label})...")
        samples, losses = load_run(run_id, model_base_dir)
        if samples is None:
            cumulative_hours += duration_hours
            continue

        time_hours = samples_to_time(samples, duration_hours, cumulative_hours)
        ax.plot(time_hours, losses, label=label, color=COLORS[i], linewidth=2)
        cumulative_hours += duration_hours

    # ── Baseline ───────────────────────────────────────────────────────────────
    bl_run_id, bl_label, bl_duration = BASELINE_STAGE
    print(f"Loading baseline run '{bl_run_id}' ({bl_label})...")
    samples, losses = load_run(bl_run_id, model_base_dir)
    if samples is not None:
        time_hours = samples_to_time(samples, bl_duration, 0.0)
        ax.plot(time_hours, losses, label=bl_label, color="black", linewidth=2, linestyle="--")

    # ── Formatting ─────────────────────────────────────────────────────────────
    ax.set_yscale("log")
    ax.set_title("Training Loss over Wall-Clock Time")
    ax.set_ylabel("Average Loss (MSE)")
    ax.set_xlabel("Wall-Clock Time (Hours)")
    ax.set_xlim(left=0)
    ax.legend(loc="upper right")
    ax.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "training_loss_time.png"
    pdf_path = out_dir / "training_loss_time.pdf"
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="./official_comparison_plot",
        type=Path,
        help="Directory where plots are saved (default: ./official_comparison_plot)",
    )
    parser.add_argument(
        "-m", "--model_base_dir",
        default="../../../WeatherGenerator/model",
        type=Path,
        help="Base-directory where WeatherGenerator models are saved",
    )
    args = parser.parse_args()
    plot_training_loss_time(args.model_base_dir, args.output_dir)


if __name__ == "__main__":
    main()
