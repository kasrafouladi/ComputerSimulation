"""Generate the 6 required visualizations and save them as PNG files."""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    main()
