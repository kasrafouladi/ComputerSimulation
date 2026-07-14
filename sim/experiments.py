"""Run baseline + 3 alternative scenarios + combined scenario,
each with 30 replications, using Poisson arrivals and realistic distributions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__) + "/..")

import numpy as np
import pandas as pd
from scipy import stats
from sim.model import ScenarioParams, run_single_replication

N_REPS = 30
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

SCENARIOS = {
    "base": ScenarioParams(name="base", arrival_mean=0.45),
    "capacity": ScenarioParams(name="capacity", n_machines=13, buffer_capacity=10, arrival_mean=0.45),
    "policy_priority": ScenarioParams(name="policy_priority", queue_discipline="PRIORITY", arrival_mean=0.45),
    "demand_surge": ScenarioParams(name="demand_surge", arrival_mean=0.35),
    "priority_demand_surge": ScenarioParams(
        name="priority_demand_surge",
        queue_discipline="PRIORITY",
        arrival_mean=0.35
    ),
}

METRIC_KEYS = [
    "admission_rate", "rejection_rate", "avg_wait_hours", "avg_machine_utilization",
    "peak_queue_length", "pct_time_queue_full", "pct_time_under_pressure",
    "avg_cycle_time_hours", "rejection_rate_low_priority", "rejection_rate_medium_priority",
    "rejection_rate_high_priority", "scrap_rate", "avg_machine_downtime_fraction",
]


def ci95(values):
    values = np.array([v for v in values if not np.isnan(v)])
    n = len(values)
    if n < 2:
        return values.mean() if n else np.nan, np.nan, np.nan
    mean = values.mean()
    sem = stats.sem(values)
    h = sem * stats.t.ppf(0.975, n - 1)
    return mean, mean - h, mean + h


def main():
    all_replication_rows = []
    representative_base = None

    for scen_name, params in SCENARIOS.items():
        for rep in range(N_REPS):
            seed = hash((scen_name, rep)) % (2**31)
            result = run_single_replication(params, seed=seed)
            row = {"scenario": scen_name, "replication": rep, "seed": seed}
            row.update(result["metrics"])
            all_replication_rows.append(row)

            if scen_name == "base" and rep == 0:
                representative_base = result

        print(f"done: {scen_name}")

    rep_df = pd.DataFrame(all_replication_rows)
    rep_df.to_csv(os.path.join(DATA_DIR, "replication_metrics.csv"), index=False)

    summary_rows = []
    for scen_name in SCENARIOS:
        sub = rep_df[rep_df.scenario == scen_name]
        row = {"scenario": scen_name}
        for m in METRIC_KEYS:
            mean, lo, hi = ci95(sub[m].values)
            row[f"{m}_mean"] = mean
            row[f"{m}_ci_low"] = lo
            row[f"{m}_ci_high"] = hi
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(DATA_DIR, "scenario_summary.csv"), index=False)

    representative_base["job_df"].to_csv(os.path.join(DATA_DIR, "job_level_base_run.csv"), index=False)
    representative_base["hourly_df"].to_csv(os.path.join(DATA_DIR, "hourly_monitoring_base_run.csv"), index=False)

    print("\nScenario summary (mean [95% CI]):")
    for _, r in summary_df.iterrows():
        print(f"\n== {r['scenario']} ==")
        for m in ["admission_rate", "rejection_rate", "avg_wait_hours", "avg_machine_utilization", "avg_cycle_time_hours"]:
            print(f"  {m}: {r[m+'_mean']:.4f}  [{r[m+'_ci_low']:.4f}, {r[m+'_ci_high']:.4f}]")

    return rep_df, summary_df, representative_base


if __name__ == "__main__":
    main()