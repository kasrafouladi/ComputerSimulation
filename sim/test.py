from sim.model import ScenarioParams, run_single_replication
import os

params = ScenarioParams(name="base", arrival_mean=0.45)
result = run_single_replication(params, seed=0)  # use seed 0 for reproducibility

# Overwrite the existing base files
result["job_df"].to_csv("data/job_level_base_run.csv", index=False)
result["hourly_df"].to_csv("data/hourly_monitoring_base_run.csv", index=False)
print("Base run CSV files updated.")