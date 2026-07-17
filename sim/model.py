"""
Discrete-event simulation of a job-shop machining work center.
(Version with realistic probability distributions and Poisson arrivals)
"""

import simpy
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict

# --------------------------------------------------------------------------
# Baseline reference data
# --------------------------------------------------------------------------

PRIORITY_CLASSES = {
    "Low": (0.30, 1),
    "Standard": (0.40, 2),
    "High": (0.20, 3),
    "Critical": (0.10, 5),
}

JOB_TYPES = {
    # name: (probability, base_complexity, proc_mean, proc_cv, subtypes)
    "Standard":  (0.25, 2, 3.0, 0.35, ["SingleOp", "TwoOp", "ThreeOp"]),
    "Precision": (0.20, 4, 5.0, 0.35, ["TightTolerance", "UltraPrecision", "MultiAxis"]),
    "Bulk":      (0.20, 1, 1.5, 0.35, ["SmallBatch", "MediumBatch", "LargeBatch"]),
    "Rush":      (0.20, 3, 2.25, 0.35, ["SingleItem", "SmallLot"]),
    "Custom":    (0.15, 5, 6.5, 0.35, ["Prototype", "OneOff", "EngineeringChange"]),
}

@dataclass
class ScenarioParams:
    name: str = "base"
    horizon_hours: float = 720.0
    n_machines: int = 10
    buffer_capacity: int = 8
    arrival_mean: float = 0.45               # mean interarrival time (exponential)
    queue_discipline: str = "FIFO"           # "FIFO" or "PRIORITY"
    mtbf_mean: float = 120.0
    mttr_low: float = 1.0
    mttr_high: float = 4.0
    p_scrap_on_breakdown: float = 0.25
    seed: int = 0


@dataclass
class Job:
    job_id: int
    arrival_time: float
    job_type: str
    subtype: str
    priority_class: str
    priority_weight: int
    complexity_score: float
    priority_level: float
    processing_time: float
    safe_wait_mean: float          # mean of exponential wait tolerance
    status: str = "in_system"
    machine_number: Optional[int] = None
    queue_entry_time: Optional[float] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    rejection_time: Optional[float] = None
    rejection_reason: Optional[str] = None
    remaining_processing: Optional[float] = None


def lognormal_params(mean, cv):
    """Return (mu, sigma) for log-normal distribution given mean and CV."""
    sigma = np.sqrt(np.log(1 + cv**2))
    mu = np.log(mean) - 0.5 * sigma**2
    return mu, sigma


def sample_job_attributes(rng: np.random.Generator, job_id: int, arrival_time: float) -> Job:
    # Priority class
    p_names = list(PRIORITY_CLASSES.keys())
    p_probs = [PRIORITY_CLASSES[n][0] for n in p_names]
    priority_class = rng.choice(p_names, p=p_probs)
    priority_weight = PRIORITY_CLASSES[priority_class][1]

    # Job type
    j_names = list(JOB_TYPES.keys())
    j_probs = [JOB_TYPES[n][0] for n in j_names]
    job_type = rng.choice(j_names, p=j_probs)
    base_complexity, proc_mean, proc_cv, subtypes = (
        JOB_TYPES[job_type][1],
        JOB_TYPES[job_type][2],
        JOB_TYPES[job_type][3],
        JOB_TYPES[job_type][4],
    )
    subtype = rng.choice(subtypes)

    # Processing time (log-normal)
    mu, sigma = lognormal_params(proc_mean, proc_cv)
    processing_time = rng.lognormal(mu, sigma)

    # Quality risk factor (Beta(2,2) scaled to [0,2])
    quality_risk = 2.0 * rng.beta(2, 2)

    complexity_score = base_complexity + quality_risk

    # Priority level: combination of priority weight and complexity
    norm_complexity = complexity_score / 7.0
    priority_level = 0.6 * priority_weight + 0.4 * (norm_complexity * 5)

    # Safe wait mean: higher priority -> shorter tolerance
    safe_wait_mean = max(0.5, 10.0 / (1.0 + priority_level))

    return Job(
        job_id=job_id,
        arrival_time=arrival_time,
        job_type=job_type,
        subtype=subtype,
        priority_class=priority_class,
        priority_weight=priority_weight,
        complexity_score=round(complexity_score, 3),
        priority_level=round(priority_level, 3),
        processing_time=round(processing_time, 3),
        safe_wait_mean=round(safe_wait_mean, 3),
        remaining_processing=round(processing_time, 3),
    )


class WorkCenter:
    def __init__(self, env: simpy.Environment, params: ScenarioParams, rng: np.random.Generator):
        self.env = env
        self.p = params
        self.rng = rng

        self.machine_free = [True] * params.n_machines
        self.machine_down = [False] * params.n_machines
        self.machine_current_job: List[Optional[Job]] = [None] * params.n_machines
        self.machine_busy_hours = [0.0] * params.n_machines
        self.machine_process_ref: List[Optional[simpy.Process]] = [None] * params.n_machines

        self.buffer: List[Job] = []
        self.all_jobs: Dict[int, Job] = {}
        self.hourly_records = []

        self.job_counter = 0

        for m in range(params.n_machines):
            env.process(self.machine_breakdown_process(m))

        env.process(self.monitor_process())

    # ---------- Arrivals (Poisson process) ----------
    def job_generator(self):
        while self.env.now < self.p.horizon_hours:
            interarrival = self.rng.exponential(self.p.arrival_mean)
            yield self.env.timeout(interarrival)
            if self.env.now >= self.p.horizon_hours:
                break
            self.job_counter += 1
            job = sample_job_attributes(self.rng, self.job_counter, self.env.now)
            self.all_jobs[job.job_id] = job
            self.env.process(self.handle_new_job(job))

    def handle_new_job(self, job: Job):
        # Inspection time (Gamma distribution)
        # Gamma parameters for mean=0.25, sd=0.08
        mean_insp = 0.25
        sd_insp = 0.08
        shape = mean_insp**2 / sd_insp**2
        scale = sd_insp**2 / mean_insp
        inspection_time = max(0.05, self.rng.gamma(shape, scale))
        yield self.env.timeout(inspection_time)

        self.try_assign_or_queue(job)

    def try_assign_or_queue(self, job: Job):
        free_machine = self.first_free_machine()
        if free_machine is not None:
            self.assign_machine(job, free_machine)
        elif len(self.buffer) < self.p.buffer_capacity:
            job.queue_entry_time = self.env.now
            self.buffer.append(job)
            self.env.process(self.timeout_watcher(job))
        else:
            self.reject(job, "buffer_full")

    def first_free_machine(self) -> Optional[int]:
        for m in range(self.p.n_machines):
            if self.machine_free[m] and not self.machine_down[m]:
                return m
        return None

    def assign_machine(self, job: Job, machine_id: int):
        self.machine_free[machine_id] = False
        self.machine_current_job[machine_id] = job
        job.machine_number = machine_id
        job.start_time = self.env.now
        if job in self.buffer:
            self.buffer.remove(job)
        self.machine_process_ref[machine_id] = self.env.process(self.run_job(job, machine_id))

    def run_job(self, job: Job, machine_id: int):
        remaining = job.remaining_processing
        while remaining > 0:
            start = self.env.now
            try:
                yield self.env.timeout(remaining)
                remaining = 0
            except simpy.Interrupt:
                elapsed = self.env.now - start
                remaining -= elapsed
                remaining = max(remaining, 0)
                job.remaining_processing = remaining
                if self.rng.random() < self.p.p_scrap_on_breakdown:
                    self.reject(job, "machine_failure")
                    self.machine_current_job[machine_id] = None
                    return
                while self.machine_down[machine_id]:
                    try:
                        yield self.env.timeout(0.1)
                    except simpy.Interrupt:
                        continue
                continue

        self.machine_busy_hours[machine_id] += job.processing_time
        scrap_prob = min(0.20, 0.02 * job.complexity_score)
        if self.rng.random() < scrap_prob:
            job.status = "rejected"
            job.rejection_time = self.env.now
            job.rejection_reason = "quality_reject"
        else:
            job.status = "completed"
            job.completion_time = self.env.now

        self.machine_free[machine_id] = True
        self.machine_current_job[machine_id] = None
        self.pull_next_job(machine_id)

    def pull_next_job(self, machine_id: int):
        if not self.buffer:
            return
        if self.p.queue_discipline == "PRIORITY":
            self.buffer.sort(key=lambda j: (-j.priority_level, j.queue_entry_time))
        next_job = self.buffer[0]
        self.assign_machine(next_job, machine_id)

    def timeout_watcher(self, job: Job):
        # Exponential wait tolerance with mean = safe_wait_mean
        wait_limit = self.rng.exponential(job.safe_wait_mean)
        yield self.env.timeout(wait_limit)
        if job in self.buffer and job.status == "in_system":
            self.buffer.remove(job)
            self.reject(job, "timeout")

    def reject(self, job: Job, reason: str):
        if job.status != "in_system":
            return
        job.status = "rejected"
        job.rejection_time = self.env.now
        job.rejection_reason = reason

    # ---------- Machine breakdown process ----------
    def machine_breakdown_process(self, machine_id: int):
        while True:
            time_to_failure = self.rng.exponential(self.p.mtbf_mean)
            yield self.env.timeout(time_to_failure)
            if self.env.now >= self.p.horizon_hours:
                break
            self.machine_down[machine_id] = True
            was_free = self.machine_free[machine_id]
            self.machine_free[machine_id] = False
            current_job = self.machine_current_job[machine_id]
            proc_ref = self.machine_process_ref[machine_id]
            if current_job is not None and proc_ref is not None and proc_ref.is_alive:
                try:
                    proc_ref.interrupt("breakdown")
                except RuntimeError:
                    pass

            repair_time = self.rng.uniform(self.p.mttr_low, self.p.mttr_high)
            yield self.env.timeout(repair_time)
            self.machine_down[machine_id] = False
            if self.machine_current_job[machine_id] is None:
                self.machine_free[machine_id] = True
                self.pull_next_job(machine_id)

    # ---------- Hourly monitoring ----------
    def monitor_process(self):
        while True:
            machines_busy = sum(1 for f in self.machine_free if not f)
            available = self.p.n_machines - machines_busy
            queue_length = len(self.buffer)
            queue_full = queue_length >= self.p.buffer_capacity
            utilization = machines_busy / self.p.n_machines
            pressure = machines_busy + queue_length
            over_capacity_pressure = pressure > self.p.n_machines
            machines_down_count = sum(self.machine_down)

            self.hourly_records.append({
                "time": self.env.now,
                "machines_busy": machines_busy,
                "available_machines": available,
                "queue_length": queue_length,
                "queue_full": queue_full,
                "machine_utilization": utilization,
                "system_pressure": pressure,
                "over_capacity_pressure": over_capacity_pressure,
                "machines_down": machines_down_count,
            })
            yield self.env.timeout(1.0)


def run_single_replication(params: ScenarioParams, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    wc = WorkCenter(env, params, rng)
    env.process(wc.job_generator())
    env.run(until=params.horizon_hours + 200)

    for job in wc.all_jobs.values():
        if job.status == "in_system":
            job.status = "censored"

    job_rows = []
    for job in wc.all_jobs.values():
        wait_to_machine = None
        if job.start_time is not None and job.queue_entry_time is not None:
            wait_to_machine = job.start_time - job.queue_entry_time
        elif job.start_time is not None:
            wait_to_machine = 0.0
        total_time = None
        if job.completion_time is not None:
            total_time = job.completion_time - job.arrival_time
        elif job.rejection_time is not None:
            total_time = job.rejection_time - job.arrival_time

        job_rows.append({
            "job_id": job.job_id, "arrival_time": round(job.arrival_time, 3),
            "job_type": job.job_type, "subtype": job.subtype,
            "priority_class": job.priority_class, "complexity_score": job.complexity_score,
            "priority_level": job.priority_level, "status": job.status,
            "machine_number": job.machine_number,
            "queue_entry_time": job.queue_entry_time,
            "start_time": job.start_time, "completion_time": job.completion_time,
            "wait_to_machine": wait_to_machine,
            "total_time_in_system": total_time,
            "rejection_reason": job.rejection_reason,
        })

    job_df = pd.DataFrame(job_rows)
    hourly_df = pd.DataFrame(wc.hourly_records)

    metrics = compute_metrics(job_df, hourly_df, params)
    return {"job_df": job_df, "hourly_df": hourly_df, "metrics": metrics}


def compute_metrics(job_df: pd.DataFrame, hourly_df: pd.DataFrame, params: ScenarioParams) -> Dict:
    total = len(job_df)
    completed = job_df[job_df.status == "completed"]
    rejected = job_df[job_df.status == "rejected"]
    admitted = job_df[job_df.status.isin(["completed", "censored"]) | job_df.start_time.notna()]

    admission_rate = job_df.start_time.notna().sum() / total if total else np.nan
    rejection_rate = len(rejected) / total if total else np.nan

    wait_admitted = job_df[job_df.wait_to_machine.notna()].wait_to_machine
    avg_wait = wait_admitted.mean() if len(wait_admitted) else np.nan

    avg_utilization = hourly_df.machine_utilization.mean() if len(hourly_df) else np.nan
    peak_queue = hourly_df.queue_length.max() if len(hourly_df) else np.nan
    pct_queue_full = hourly_df.queue_full.mean() if len(hourly_df) else np.nan
    pct_pressure = hourly_df.over_capacity_pressure.mean() if len(hourly_df) else np.nan

    cycle_times = completed.total_time_in_system.dropna()
    avg_cycle_time = cycle_times.mean() if len(cycle_times) else np.nan

    def rej_rate_for(mask):
        sub = job_df[mask]
        if len(sub) == 0:
            return np.nan
        return (sub.status == "rejected").mean()

    rej_low = rej_rate_for(job_df.priority_class == "Low")
    rej_med = rej_rate_for(job_df.priority_class == "Standard")
    rej_high = rej_rate_for(job_df.priority_class.isin(["High", "Critical"]))

    scrap_rate = (job_df.rejection_reason == "quality_reject").sum() / total if total else np.nan
    machine_downtime_frac = hourly_df.machines_down.mean() / params.n_machines if len(hourly_df) else np.nan

    return {
        "admission_rate": admission_rate,
        "rejection_rate": rejection_rate,
        "avg_wait_hours": avg_wait,
        "avg_machine_utilization": avg_utilization,
        "peak_queue_length": peak_queue,
        "pct_time_queue_full": pct_queue_full,
        "pct_time_under_pressure": pct_pressure,
        "avg_cycle_time_hours": avg_cycle_time,
        "rejection_rate_low_priority": rej_low,
        "rejection_rate_medium_priority": rej_med,
        "rejection_rate_high_priority": rej_high,
        "scrap_rate": scrap_rate,
        "avg_machine_downtime_fraction": machine_downtime_frac,
        "n_jobs": total,
    }