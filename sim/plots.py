"""Generate the 6 required visualizations and save them as PNG files."""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10})


def savefig(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)


def plot_machine_utilization_over_time(hourly_df):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hourly_df["time"], hourly_df["machine_utilization"], lw=0.8, color="#1f6f8b")
    ax.set_xlabel("Simulation time (hours)")
    ax.set_ylabel("Machine utilization")
    ax.set_title("Machine Utilization Over Time — Representative Base Run")
    ax.set_ylim(0, 1.05)
    savefig(fig, "01_machine_utilization_over_time.png")


def plot_queue_length_over_time(hourly_df):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hourly_df["time"], hourly_df["queue_length"], lw=0.8, color="#c1440e")
    ax.set_xlabel("Simulation time (hours)")
    ax.set_ylabel("Buffer (queue) length")
    ax.set_title("Buffer Queue Length Over Time — Representative Base Run")
    savefig(fig, "02_queue_length_over_time.png")


def plot_job_outcomes(job_df):
    counts = job_df["status"].value_counts().reindex(["completed", "rejected", "censored"]).fillna(0)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    colors = ["#2e8b57", "#c1440e", "#7f7f7f"]
    ax.bar(counts.index, counts.values, color=colors)
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, int(v), ha="center")
    ax.set_ylabel("Number of jobs")
    ax.set_title("Job Outcomes — Representative Base Run")
    savefig(fig, "03_job_outcomes.png")


def plot_rejection_reasons(job_df):
    rejected = job_df[job_df.status == "rejected"]
    counts = rejected["rejection_reason"].value_counts()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar(counts.index, counts.values, color="#8a3324")
    ax.set_ylabel("Number of jobs")
    ax.set_title("Rejection Reasons — Representative Base Run")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    savefig(fig, "04_rejection_reasons.png")


def plot_processing_time_by_job_type(job_df):
    completed = job_df[job_df.status == "completed"].copy()
    completed["processing_time_est"] = completed["completion_time"] - completed["start_time"]
    order = ["Bulk", "Rush", "Standard", "Precision", "Custom"]
    data = [completed.loc[completed.job_type == jt, "processing_time_est"].dropna().values for jt in order]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.boxplot(data, tick_labels=order, showmeans=True)
    ax.set_ylabel("Processing time (hours)")
    ax.set_title("Processing Duration by Job Type — Representative Base Run")
    savefig(fig, "05_processing_time_by_job_type.png")


def plot_scenario_comparison(summary_df, metric="avg_wait_hours", label="Average waiting time (hours)"):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    names = summary_df["scenario"]
    means = summary_df[f"{metric}_mean"]
    lo = summary_df[f"{metric}_mean"] - summary_df[f"{metric}_ci_low"]
    hi = summary_df[f"{metric}_ci_high"] - summary_df[f"{metric}_mean"]
    ax.bar(names, means, yerr=[lo, hi], capsize=6, color="#1f6f8b")
    ax.set_ylabel(label)
    ax.set_title(f"Scenario Comparison with 95% CI — {label}")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    savefig(fig, "06_scenario_comparison_avg_wait.png")


def main():
    hourly_df = pd.read_csv(os.path.join(DATA_DIR, "hourly_monitoring_base_run.csv"))
    job_df = pd.read_csv(os.path.join(DATA_DIR, "job_level_base_run.csv"))
    summary_df = pd.read_csv(os.path.join(DATA_DIR, "scenario_summary.csv"))

    plot_machine_utilization_over_time(hourly_df)
    plot_queue_length_over_time(hourly_df)
    plot_job_outcomes(job_df)
    plot_rejection_reasons(job_df)
    plot_processing_time_by_job_type(job_df)
    plot_scenario_comparison(summary_df)

# ----------------------------------------------------------------------
# 7. Gantt chart – machine timelines
# ----------------------------------------------------------------------
def plot_gantt_chart(job_df):
    """
    Horizontal bar chart per machine showing processing intervals.
    Each bar is coloured by job type.
    """
    # Use only completed jobs (they have start & completion)
    completed = job_df[job_df.status == "completed"].copy()
    if completed.empty:
        print("No completed jobs to plot Gantt chart.")
        return

    # Assign a colour per job type
    job_types = completed["job_type"].unique()
    cmap = plt.cm.tab10
    colors = {jt: cmap(i % 10) for i, jt in enumerate(job_types)}

    fig, ax = plt.subplots(figsize=(12, 6))
    machines = sorted(completed["machine_number"].unique())
    y_pos = {m: i for i, m in enumerate(machines)}

    for _, row in completed.iterrows():
        y = y_pos[row["machine_number"]]
        start = row["start_time"]
        end = row["completion_time"]
        duration = end - start
        ax.barh(y, duration, left=start,
                color=colors.get(row["job_type"], "gray"),
                edgecolor="black", linewidth=0.5)

    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels([f"Machine {m}" for m in machines])
    ax.set_xlabel("Simulation time (hours)")
    ax.set_title("Machine Gantt Chart – Representative Base Run")

    # Legend
    handles = [Rectangle((0,0),1,1, color=colors[t]) for t in job_types]
    ax.legend(handles, job_types, loc="upper right")

    savefig(fig, "07_gantt_chart.png")


# ----------------------------------------------------------------------
# 8. Animated dashboard – machine status + queue length
# ----------------------------------------------------------------------
def plot_animated_dashboard(job_df, hourly_df):
    """
    Animated GIF showing machine statuses and queue length.
    Uses completion_time, rejection_time, or processing_time to determine end.
    """
    # Build busy intervals per machine from all jobs that started
    intervals = {}
    for _, row in job_df.iterrows():
        if pd.isna(row["start_time"]):
            continue
        # Determine end time
        end = None
        if not pd.isna(row["completion_time"]):
            end = row["completion_time"]
        elif not pd.isna(row.get("rejection_time", np.nan)):
            end = row["rejection_time"]
        elif not pd.isna(row.get("processing_time", np.nan)):
            # fallback: approximate end as start + processing
            end = row["start_time"] + row["processing_time"]
        if end is None:
            continue
        machine = int(row["machine_number"])
        intervals.setdefault(machine, []).append((row["start_time"], end))

    N_MACHINES = 10  # base scenario has 10 machines
    times = hourly_df["time"].values
    queue_lengths = hourly_df["queue_length"].values

    # For each frame, determine which machines are busy
    busy_statuses = []
    for t in times:
        busy = [False] * N_MACHINES
        for m, ints in intervals.items():
            if m >= N_MACHINES:
                continue
            for start, end in ints:
                if start <= t < end:
                    busy[m] = True
                    break
        busy_statuses.append(busy)


    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: 2 rows × 5 columns of machine rectangles
    rects = []
    for i in range(2):
        for j in range(5):
            rect = Rectangle((j * 1.2, i * 1.2), 1, 1,
                             fc="lightgray", ec="black")
            ax1.add_patch(rect)
            rects.append(rect)
    ax1.set_xlim(-0.5, 7)
    ax1.set_ylim(-0.5, 3)
    ax1.set_aspect("equal")
    ax1.set_title("Machine Status (Green=Idle, Red=Busy)")
    ax1.axis("off")

    # Right: bar for queue length
    bars = ax2.bar(["Queue Length"], [0], color="orange")
    ax2.set_ylim(0, max(queue_lengths) + 2)
    ax2.set_title("Buffer Queue Length")

    # Animation update function
    def update(frame):
        busy = busy_statuses[frame]
        for rect, is_busy in zip(rects, busy):
            rect.set_facecolor("red" if is_busy else "lightgreen")
        q = queue_lengths[frame]
        bars[0].set_height(q)
        ax2.set_title(f"Buffer Queue Length (Hour {int(times[frame])})")
        return rects + [bars[0]]

    anim = animation.FuncAnimation(
        fig, update, frames=len(times),
        interval=200, repeat=True
    )

    gif_path = os.path.join(PLOT_DIR, "08_shop_floor_animation.gif")
    anim.save(gif_path, writer="pillow", fps=5)
    plt.close(fig)
    print("saved", gif_path)


# ----------------------------------------------------------------------
# Update main() to call the new plots
# ----------------------------------------------------------------------
def main():
    hourly_df = pd.read_csv(os.path.join(DATA_DIR, "hourly_monitoring_base_run.csv"))
    job_df = pd.read_csv(os.path.join(DATA_DIR, "job_level_base_run.csv"))
    summary_df = pd.read_csv(os.path.join(DATA_DIR, "scenario_summary.csv"))

    # Existing plots
    plot_machine_utilization_over_time(hourly_df)
    plot_queue_length_over_time(hourly_df)
    plot_job_outcomes(job_df)
    plot_rejection_reasons(job_df)
    plot_processing_time_by_job_type(job_df)
    plot_scenario_comparison(summary_df)

    # NEW plots
    plot_gantt_chart(job_df)
    plot_animated_dashboard(job_df, hourly_df)
    
    
if __name__ == "__main__":
    main()
