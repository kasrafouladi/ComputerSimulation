"""Minimum verification checks (analogous to Appendix B of the assignment),
run against the saved base-run job-level / hourly tables and the
replication-level metrics table.
"""
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def main():
    job_df = pd.read_csv(os.path.join(DATA_DIR, "job_level_base_run.csv"))
    hourly_df = pd.read_csv(os.path.join(DATA_DIR, "hourly_monitoring_base_run.csv"))
    rep_df = pd.read_csv(os.path.join(DATA_DIR, "replication_metrics.csv"))

    results = []
    n_machines_base = 10
    buffer_capacity_base = 8

    results.append(check(
        "Number of busy machines never exceeds machine capacity",
        (hourly_df["machines_busy"] <= n_machines_base).all()
    ))
    results.append(check(
        "Active queue length never exceeds buffer capacity",
        (hourly_df["queue_length"] <= buffer_capacity_base).all()
    ))
    results.append(check(
        "Every completed job has machine_number, start_time, completion_time",
        job_df[job_df.status == "completed"][["machine_number", "start_time", "completion_time"]].notna().all().all()
    ))
    results.append(check(
        "Every rejected job has rejection_reason",
        job_df[job_df.status == "rejected"]["rejection_reason"].notna().all()
    ))
    results.append(check(
        "Every censored job (still in system at horizon end) is explicitly flagged",
        (job_df.status.isin(["completed", "rejected", "censored"])).all()
    ))
    results.append(check(
        "Replications use different random seeds",
        rep_df.groupby("scenario")["seed"].nunique().eq(30).all()
    ))
    results.append(check(
        "Replication count per scenario is at least 30",
        rep_df.groupby("scenario").size().ge(30).all()
    ))
    results.append(check(
        "No machine assigned to two jobs at once "
        "(no duplicate active machine_number at the same start_time)",
        not job_df[job_df.status == "completed"].duplicated(subset=["machine_number", "start_time"]).any()
    ))

    n_pass = sum(results)
    print(f"\n{n_pass}/{len(results)} verification checks passed.")


if __name__ == "__main__":
    main()
