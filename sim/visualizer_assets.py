import sys
import os
import tkinter as tk
from tkinter import ttk
import simpy
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# ----------------------------------------------------------------------
# Data structures and distributions
# ----------------------------------------------------------------------

PRIORITY_CLASSES = {
    "Low": (0.30, 1),
    "Standard": (0.40, 2),
    "High": (0.20, 3),
    "Critical": (0.10, 5),
}

JOB_TYPES = {
    "Standard":  (0.25, 2, 3.0, 0.35, ["SingleOp", "TwoOp", "ThreeOp"]),
    "Precision": (0.20, 4, 5.0, 0.35, ["TightTolerance", "UltraPrecision", "MultiAxis"]),
    "Bulk":      (0.20, 1, 1.5, 0.35, ["SmallBatch", "MediumBatch", "LargeBatch"]),
    "Rush":      (0.20, 3, 2.25, 0.35, ["SingleItem", "SmallLot"]),
    "Custom":    (0.15, 5, 6.5, 0.35, ["Prototype", "OneOff", "EngineeringChange"]),
}

# Machine types
MACHINE_TYPES = ['Milling', 'Turning', 'Drilling']
TYPE_COLORS = {'Milling': 'dodgerblue', 'Turning': 'orange', 'Drilling': 'forestgreen'}

# Job routes: (weight, [(type, mean_processing_time, cv), ...])
ROUTES = [
    (0.25, [('Milling', 2.0, 0.35), ('Turning', 1.5, 0.30)]),
    (0.20, [('Turning', 1.8, 0.30), ('Drilling', 1.2, 0.35)]),
    (0.15, [('Milling', 2.5, 0.35), ('Drilling', 1.5, 0.35), ('Turning', 1.0, 0.30)]),
    (0.20, [('Turning', 2.2, 0.30), ('Milling', 1.8, 0.35)]),
    (0.10, [('Drilling', 1.0, 0.35), ('Milling', 3.0, 0.35), ('Turning', 2.0, 0.30)]),
    (0.10, [('Milling', 1.5, 0.35), ('Turning', 1.5, 0.30), ('Drilling', 1.5, 0.35)]),
]

def lognormal_params(mean, cv):
    sigma = np.sqrt(np.log(1 + cv**2))
    mu = np.log(mean) - 0.5 * sigma**2
    return mu, sigma

@dataclass
class ScenarioParams:
    name: str = "base"
    horizon_hours: float = 720.0
    machine_counts: Dict[str, int] = field(default_factory=lambda: {'Milling': 4, 'Turning': 3, 'Drilling': 3})
    buffer_capacity: int = 8
    arrival_mean: float = 0.45
    queue_discipline: str = "FIFO"          # "FIFO" or "PRIORITY"
    mtbf_mean: float = 120.0
    mttr_low: float = 1.0
    mttr_high: float = 4.0
    p_scrap_on_breakdown: float = 0.25
    seed: int = 0
    qc_mean: float = 0.2                    # mean QC time (exponential)
    qc_stations: int = 2                    # number of shared QC stations

@dataclass
class Job:
    # Fields without defaults (must come first)
    job_id: int
    arrival_time: float
    job_type: str
    subtype: str
    priority_class: str
    priority_weight: int
    complexity_score: float
    priority_level: float
    operations: List[Dict]          # each dict: {'type': str, 'processing_time': float}
    safe_wait_mean: float

    # Fields with defaults
    current_op_idx: int = 0
    status: str = "in_system"
    machine_number: Optional[int] = None
    queue_entry_time: Optional[float] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    rejection_time: Optional[float] = None
    rejection_reason: Optional[str] = None
    remaining_processing: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.remaining_processing:
            self.remaining_processing = [op['processing_time'] for op in self.operations]

    def current_op_type(self):
        return self.operations[self.current_op_idx]['type']

    def is_finished(self):
        return self.current_op_idx >= len(self.operations)

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

    quality_risk = 2.0 * rng.beta(2, 2)
    complexity_score = base_complexity + quality_risk

    norm_complexity = complexity_score / 7.0
    priority_level = 0.6 * priority_weight + 0.4 * (norm_complexity * 5)

    safe_wait_mean = max(0.5, 10.0 / (1.0 + priority_level))

    # Choose a route
    route_weights = [w for w, _ in ROUTES]
    chosen_route = rng.choice([r for _, r in ROUTES], p=np.array(route_weights)/sum(route_weights))

    operations = []
    for op_type, op_mean, op_cv in chosen_route:
        adj_mean = op_mean * (0.7 + 0.3 * (complexity_score / 5.0))
        mu, sigma = lognormal_params(adj_mean, op_cv)
        proc_time = max(0.05, rng.lognormal(mu, sigma))
        operations.append({'type': op_type, 'processing_time': round(proc_time, 3)})

    return Job(
        job_id=job_id,
        arrival_time=arrival_time,
        job_type=job_type,
        subtype=subtype,
        priority_class=priority_class,
        priority_weight=priority_weight,
        complexity_score=round(complexity_score, 3),
        priority_level=round(priority_level, 3),
        operations=operations,
        safe_wait_mean=round(safe_wait_mean, 3),
    )

# ----------------------------------------------------------------------
# WorkCenter with shared QC resource
# ----------------------------------------------------------------------
class WorkCenterWithQC:
    def __init__(self, env: simpy.Environment, params: ScenarioParams, rng: np.random.Generator):
        self.env = env
        self.p = params
        self.rng = rng

        # Build machines
        self.machine_types = []
        self.machine_free = []
        self.machine_down = []
        self.machine_current_job = []
        self.machine_busy_hours = []
        self.machine_process_ref = []

        for mtype, count in params.machine_counts.items():
            for i in range(count):
                idx = len(self.machine_types)
                self.machine_types.append(mtype)
                self.machine_free.append(True)
                self.machine_down.append(False)
                self.machine_current_job.append(None)
                self.machine_busy_hours.append(0.0)
                self.machine_process_ref.append(None)

        self.n_machines = len(self.machine_types)

        # Buffer (waiting for CNC machines)
        self.buffer: List[Job] = []
        self.all_jobs: Dict[int, Job] = {}
        self.hourly_records = []
        self.job_counter = 0

        # ---------- Shared QC ----------
        self.qc_resource = simpy.Resource(env, capacity=params.qc_stations)
        self.qc_waiting: List[Job] = []      # jobs that have finished machining but are waiting for QC
        self.qc_active: List[Job] = []       # jobs currently being inspected
        self.good_count = 0
        self.scrap_count = 0

        # Event log for GUI
        self.event_log: List[Dict[str, Any]] = []
        self.job_attrs: Dict[int, Dict[str, Any]] = {}

        # Start machine breakdown processes
        for m in range(self.n_machines):
            env.process(self.machine_breakdown_process(m))

        env.process(self.monitor_process())
        self.record_snapshot()

    def record_snapshot(self):
        """Capture current state for GUI."""
        snapshot = {
            'time': self.env.now,
            'buffer': [job.job_id for job in self.buffer],
            'machines': [
                {
                    'state': 'broken' if self.machine_down[i] else
                              ('working' if self.machine_current_job[i] is not None else 'idle'),
                    'job_id': self.machine_current_job[i].job_id if self.machine_current_job[i] else None,
                    'type': self.machine_types[i],
                }
                for i in range(self.n_machines)
            ],
            'qc_waiting': [job.job_id for job in self.qc_waiting],
            'qc_active': [job.job_id for job in self.qc_active],
            'qc_free': self.qc_resource.capacity - len(self.qc_active),
            'good_count': self.good_count,
            'scrap_count': self.scrap_count,
        }
        self.event_log.append(snapshot)

    # ---------- Job arrival ----------
    def job_generator(self):
        while self.env.now < self.p.horizon_hours:
            interarrival = self.rng.exponential(self.p.arrival_mean)
            yield self.env.timeout(interarrival)
            if self.env.now >= self.p.horizon_hours:
                break
            self.job_counter += 1
            job = sample_job_attributes(self.rng, self.job_counter, self.env.now)
            self.all_jobs[job.job_id] = job
            op_info = f"{job.current_op_idx+1}/{len(job.operations)}"
            self.job_attrs[job.job_id] = {
                'color': self._priority_color(job.priority_class),
                'label': f'J{job.job_id}',
                'priority': job.priority_class,
                'op_info': op_info,
            }
            self.env.process(self.handle_new_job(job))

    def _priority_color(self, pclass):
        colors = {'Low': 'blue', 'Standard': 'green', 'High': 'orange', 'Critical': 'red'}
        return colors.get(pclass, 'gray')

    def handle_new_job(self, job: Job):
        # Initial inspection (Gamma)
        mean_insp = 0.25
        sd_insp = 0.08
        shape = mean_insp**2 / sd_insp**2
        scale = sd_insp**2 / mean_insp
        inspection_time = max(0.05, self.rng.gamma(shape, scale))
        yield self.env.timeout(inspection_time)
        self.record_snapshot()
        self.try_assign_or_queue(job)

    def try_assign_or_queue(self, job: Job):
        if job.is_finished():
            # All operations done: go to QC queue
            self.qc_waiting.append(job)
            self.record_snapshot()
            self.env.process(self.quality_check(job))
            return

        needed_type = job.current_op_type()
        free_machine = self.find_free_machine(needed_type)

        if free_machine is not None:
            self.assign_machine(job, free_machine)
        elif len(self.buffer) < self.p.buffer_capacity:
            job.queue_entry_time = self.env.now
            self.buffer.append(job)
            self.record_snapshot()
            self.env.process(self.timeout_watcher(job))
        else:
            self.reject(job, "buffer_full")
            self.record_snapshot()

    def find_free_machine(self, mtype: str) -> Optional[int]:
        for i in range(self.n_machines):
            if (self.machine_types[i] == mtype and
                self.machine_free[i] and not self.machine_down[i]):
                return i
        return None

    def assign_machine(self, job: Job, machine_id: int):
        self.machine_free[machine_id] = False
        self.machine_current_job[machine_id] = job
        job.machine_number = machine_id
        job.start_time = self.env.now
        if job in self.buffer:
            self.buffer.remove(job)
        self.machine_process_ref[machine_id] = self.env.process(self.run_operation(job, machine_id))
        self.record_snapshot()

    def run_operation(self, job: Job, machine_id: int):
        op_idx = job.current_op_idx
        remaining = job.remaining_processing[op_idx]

        while remaining > 0:
            start = self.env.now
            try:
                yield self.env.timeout(remaining)
                remaining = 0
            except simpy.Interrupt:
                elapsed = self.env.now - start
                remaining -= elapsed
                remaining = max(remaining, 0)
                job.remaining_processing[op_idx] = remaining
                if self.rng.random() < self.p.p_scrap_on_breakdown:
                    self.reject(job, "machine_failure")
                    self.machine_current_job[machine_id] = None
                    self.record_snapshot()
                    return
                while self.machine_down[machine_id]:
                    try:
                        yield self.env.timeout(0.1)
                    except simpy.Interrupt:
                        continue

        # Operation finished
        self.machine_busy_hours[machine_id] += (job.operations[op_idx]['processing_time'] -
                                                job.remaining_processing[op_idx])
        job.current_op_idx += 1
        self.machine_free[machine_id] = True
        self.machine_current_job[machine_id] = None
        self.record_snapshot()

        # Pull next job for this machine
        self.pull_next_job(machine_id)

        # Re-enter the routing logic
        self.try_assign_or_queue(job)

    def pull_next_job(self, machine_id: int):
        needed_type = self.machine_types[machine_id]
        candidates = [j for j in self.buffer if not j.is_finished() and j.current_op_type() == needed_type]
        if not candidates:
            return
        if self.p.queue_discipline == "PRIORITY":
            candidates.sort(key=lambda j: (-j.priority_level, j.queue_entry_time))
        next_job = candidates[0]
        self.assign_machine(next_job, machine_id)
        self.record_snapshot()

    def timeout_watcher(self, job: Job):
        wait_limit = self.rng.exponential(job.safe_wait_mean)
        yield self.env.timeout(wait_limit)
        if job in self.buffer and job.status == "in_system":
            self.buffer.remove(job)
            self.reject(job, "timeout")
            self.record_snapshot()

    def reject(self, job: Job, reason: str):
        if job.status != "in_system":
            return
        job.status = "rejected"
        job.rejection_time = self.env.now
        job.rejection_reason = reason
        # Remove from any queues
        if job in self.buffer:
            self.buffer.remove(job)
        if job in self.qc_waiting:
            self.qc_waiting.remove(job)
        self.record_snapshot()

    # ---------- Shared Quality Check ----------
    def quality_check(self, job: Job):
        # Remove from waiting queue (if still there)
        if job in self.qc_waiting:
            self.qc_waiting.remove(job)
        self.record_snapshot()

        # Request a QC station
        with self.qc_resource.request() as req:
            yield req
            # Now we have a QC station
            self.qc_active.append(job)
            self.record_snapshot()

            qc_time = self.rng.exponential(self.p.qc_mean)
            yield self.env.timeout(qc_time)

            # Determine pass/fail
            scrap_prob = min(0.20, 0.02 * job.complexity_score)
            if self.rng.random() < scrap_prob:
                self.scrap_count += 1
                job.status = "rejected"
                job.rejection_time = self.env.now
                job.rejection_reason = "quality_reject"
            else:
                self.good_count += 1
                job.status = "completed"
                job.completion_time = self.env.now

            self.qc_active.remove(job)
            self.record_snapshot()

    # ---------- Machine breakdown ----------
    def machine_breakdown_process(self, machine_id: int):
        while True:
            time_to_failure = self.rng.exponential(self.p.mtbf_mean)
            yield self.env.timeout(time_to_failure)
            if self.env.now >= self.p.horizon_hours:
                break
            self.machine_down[machine_id] = True
            self.machine_free[machine_id] = False
            current_job = self.machine_current_job[machine_id]
            proc_ref = self.machine_process_ref[machine_id]
            if current_job is not None and proc_ref is not None and proc_ref.is_alive:
                try:
                    proc_ref.interrupt("breakdown")
                except RuntimeError:
                    pass
            self.record_snapshot()

            repair_time = self.rng.uniform(self.p.mttr_low, self.p.mttr_high)
            yield self.env.timeout(repair_time)
            self.machine_down[machine_id] = False
            if self.machine_current_job[machine_id] is None:
                self.machine_free[machine_id] = True
                self.pull_next_job(machine_id)
            self.record_snapshot()

    # ---------- Hourly monitoring ----------
    def monitor_process(self):
        while True:
            busy = sum(1 for i in range(self.n_machines) if not self.machine_free[i] and not self.machine_down[i])
            machines_down = sum(self.machine_down)
            available = self.n_machines - busy - machines_down
            queue_length = len(self.buffer)
            queue_full = queue_length >= self.p.buffer_capacity
            utilization = busy / self.n_machines
            pressure = busy + queue_length
            over_capacity_pressure = pressure > self.n_machines

            self.hourly_records.append({
                "time": self.env.now,
                "machines_busy": busy,
                "available_machines": available,
                "queue_length": queue_length,
                "queue_full": queue_full,
                "machine_utilization": utilization,
                "system_pressure": pressure,
                "over_capacity_pressure": over_capacity_pressure,
                "machines_down": machines_down,
            })
            yield self.env.timeout(1.0)

# ----------------------------------------------------------------------
# Run simulation
# ----------------------------------------------------------------------
def run_single_replication_with_events(params: ScenarioParams, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    wc = WorkCenterWithQC(env, params, rng)
    env.process(wc.job_generator())
    env.run(until=params.horizon_hours + 200)

    # Build job dataframe
    job_rows = []
    for job in wc.all_jobs.values():
        if job.status == "in_system":
            job.status = "censored"
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
            "route": [op['type'] for op in job.operations],
        })
    job_df = pd.DataFrame(job_rows)
    hourly_df = pd.DataFrame(wc.hourly_records)

    # Metrics
    total = len(job_df)
    completed = job_df[job_df.status == "completed"]
    rejected = job_df[job_df.status == "rejected"]

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

    metrics = {
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

    return {
        "job_df": job_df,
        "hourly_df": hourly_df,
        "metrics": metrics,
        "event_log": wc.event_log,
        "job_attrs": wc.job_attrs,
        "params": params,
    }

# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------
SCENARIOS = {
    "base": ScenarioParams(
        name="base",
        machine_counts={'Milling': 4, 'Turning': 3, 'Drilling': 3},
        qc_stations=2,
        arrival_mean=0.45
    ),
    "capacity": ScenarioParams(
        name="capacity",
        machine_counts={'Milling': 5, 'Turning': 4, 'Drilling': 4},
        buffer_capacity=10,
        qc_stations=3,
        arrival_mean=0.45
    ),
    "policy_priority": ScenarioParams(
        name="policy_priority",
        machine_counts={'Milling': 4, 'Turning': 3, 'Drilling': 3},
        qc_stations=2,
        queue_discipline="PRIORITY",
        arrival_mean=0.45
    ),
    "demand_surge": ScenarioParams(
        name="demand_surge",
        machine_counts={'Milling': 4, 'Turning': 3, 'Drilling': 3},
        qc_stations=2,
        arrival_mean=0.35
    ),
    "priority_demand_surge": ScenarioParams(
        name="priority_demand_surge",
        machine_counts={'Milling': 4, 'Turning': 3, 'Drilling': 3},
        qc_stations=2,
        queue_discipline="PRIORITY",
        arrival_mean=0.35
    ),
}

class CNC_GUI:
    def __init__(self, master):
        self.master = master
        master.title("CNC Manufacturing Line (Multi‑Op + Shared QC)")

        # Control frame
        ctrl_frame = ttk.Frame(master)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(ctrl_frame, text="Scenario:").pack(side=tk.LEFT, padx=2)
        self.scenario_var = tk.StringVar(value="base")
        scenario_menu = ttk.Combobox(ctrl_frame, textvariable=self.scenario_var,
                                     values=list(SCENARIOS.keys()), state="readonly", width=15)
        scenario_menu.pack(side=tk.LEFT, padx=5)

        self.run_btn = ttk.Button(ctrl_frame, text="Run Simulation", command=self.run_simulation)
        self.run_btn.pack(side=tk.LEFT, padx=5)

        self.play_btn = ttk.Button(ctrl_frame, text="Play", command=self.play, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.step_btn = ttk.Button(ctrl_frame, text="Step", command=self.step, state=tk.DISABLED)
        self.step_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = ttk.Button(ctrl_frame, text="Reset", command=self.reset, state=tk.DISABLED)
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl_frame, text="Speed:").pack(side=tk.LEFT, padx=(20,2))
        self.speed_var = tk.IntVar(value=500)
        speed_scale = ttk.Scale(ctrl_frame, from_=100, to=2000, variable=self.speed_var,
                                orient=tk.HORIZONTAL, length=100)
        speed_scale.pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(ctrl_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Canvas
        self.canvas = tk.Canvas(master, bg="white", width=1100, height=600)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # State
        self.event_log = []
        self.current_index = -1
        self.job_canvas_ovals = {}   # job_id -> canvas oval id
        self.playing = False
        self.after_id = None

        # Positions (computed dynamically)
        self.machine_positions = []
        self.qc_station_positions = []
        self.qc_waiting_positions = []
        self.buffer_positions = []
        self.good_box_pos = None
        self.scrap_box_pos = None
        self.animating = False

        self.draw_empty()

    def draw_empty(self):
        self.canvas.delete("all")

    def run_simulation(self):
        self.reset()
        scenario_name = self.scenario_var.get()
        params = SCENARIOS[scenario_name]
        seed = 12345
        self.status_label.config(text="Running simulation...")
        self.master.update()

        result = run_single_replication_with_events(params, seed)
        self.event_log = result["event_log"]
        self.job_attrs = result["job_attrs"]
        self.params = params

        self.status_label.config(text=f"Simulation done. {len(self.event_log)} events recorded.")
        self.current_index = 0
        self.play_btn.config(state=tk.NORMAL)
        self.step_btn.config(state=tk.NORMAL)
        self.reset_btn.config(state=tk.NORMAL)
        self.run_btn.config(state=tk.DISABLED)

        self.setup_positions()
        self.draw_snapshot(self.current_index)

    def setup_positions(self):
        n = self.params.n_machines
        canvas_width = int(self.canvas.cget("width"))
        canvas_height = int(self.canvas.cget("height"))

        # Buffer (left)
        buffer_x = 60
        buffer_y_start = 150
        buffer_spacing = 28
        self.buffer_positions = [(buffer_x, buffer_y_start + i*buffer_spacing) for i in range(self.params.buffer_capacity)]

        # Machines (middle)
        machine_y = 200
        x_start = 180
        x_spacing = (canvas_width - x_start - 300) // (n + 1)   # leave room for QC on right
        self.machine_positions = []
        for i in range(n):
            x = x_start + (i+1)*x_spacing
            self.machine_positions.append((x, machine_y))

        # QC Waiting area (between machines and QC stations)
        qc_wait_x = x_start + (n+1)*x_spacing + 20
        qc_wait_y_start = 150
        qc_wait_spacing = 28
        self.qc_waiting_positions = [(qc_wait_x, qc_wait_y_start + i*qc_wait_spacing) for i in range(10)]  # max 10 waiting

        # QC Stations (far right)
        qc_station_x = qc_wait_x + 70
        qc_station_y_start = 180
        qc_station_spacing = 60
        self.qc_station_positions = []
        for i in range(self.params.qc_stations):
            y = qc_station_y_start + i*qc_station_spacing
            self.qc_station_positions.append((qc_station_x, y))

        # Good / Scrap boxes (far right, below QC)
        self.good_box_pos = (canvas_width - 130, 150)
        self.scrap_box_pos = (canvas_width - 130, 400)

    def draw_snapshot(self, idx):
        self.canvas.delete("all")
        snapshot = self.event_log[idx]
        time = snapshot['time']
        buffer_ids = snapshot['buffer']
        machines = snapshot['machines']
        qc_waiting_ids = snapshot['qc_waiting']
        qc_active_ids = snapshot['qc_active']
        qc_free = snapshot['qc_free']
        good_count = snapshot['good_count']
        scrap_count = snapshot['scrap_count']

        # ---- Buffer ----
        for pos_idx, job_id in enumerate(buffer_ids):
            if pos_idx < len(self.buffer_positions):
                x, y = self.buffer_positions[pos_idx]
                color = self.job_attrs.get(job_id, {}).get('color', 'gray')
                op_info = self.job_attrs.get(job_id, {}).get('op_info', '')
                oval = self.canvas.create_oval(x-10, y-10, x+10, y+10, fill=color, outline='black')
                self.canvas.create_text(x, y, text=op_info, font=('Arial', 7))

        # ---- Machines ----
        for i, (x, y) in enumerate(self.machine_positions):
            state = machines[i]['state']
            mtype = machines[i]['type']
            color = {'idle':'lightgray', 'working':'yellow', 'broken':'red'}.get(state, 'gray')
            self.canvas.create_rectangle(x-25, y-25, x+25, y+25, fill=color,
                                         outline=TYPE_COLORS.get(mtype, 'black'), width=3)
            self.canvas.create_text(x, y-40, text=f"M{i+1}\n{mtype}", font=('Arial', 8))
            if state == 'working':
                job_id = machines[i]['job_id']
                if job_id is not None:
                    c = self.job_attrs.get(job_id, {}).get('color', 'gray')
                    op_info = self.job_attrs.get(job_id, {}).get('op_info', '')
                    oval = self.canvas.create_oval(x-12, y-12, x+12, y+12, fill=c, outline='black')
                    self.canvas.create_text(x, y, text=op_info, font=('Arial', 7))

        # ---- QC Waiting Queue ----
        self.canvas.create_text(self.qc_waiting_positions[0][0]-20, 130,
                                text="QC Queue", font=('Arial', 10, 'bold'))
        for pos_idx, job_id in enumerate(qc_waiting_ids):
            if pos_idx < len(self.qc_waiting_positions):
                x, y = self.qc_waiting_positions[pos_idx]
                color = self.job_attrs.get(job_id, {}).get('color', 'gray')
                oval = self.canvas.create_oval(x-10, y-10, x+10, y+10, fill=color, outline='black')
                self.canvas.create_text(x, y, text="QC", font=('Arial', 7))

        # ---- QC Stations ----
        self.canvas.create_text(self.qc_station_positions[0][0], 130,
                                text=f"QC Stations ({self.params.qc_stations})", font=('Arial', 10, 'bold'))
        for i, (x, y) in enumerate(self.qc_station_positions):
            # Check if this station is busy (i < len(qc_active_ids))
            is_busy = i < len(qc_active_ids)
            fill_color = 'darkblue' if is_busy else 'lightblue'
            self.canvas.create_rectangle(x-20, y-15, x+20, y+15, fill=fill_color, outline='black')
            self.canvas.create_text(x, y-30, text=f"QC{i+1}")
            if is_busy:
                job_id = qc_active_ids[i]
                c = self.job_attrs.get(job_id, {}).get('color', 'gray')
                self.canvas.create_oval(x-10, y-10, x+10, y+10, fill=c, outline='black')
                self.canvas.create_text(x, y, text="QC", font=('Arial', 7))
        # Show free stations count
        self.canvas.create_text(self.qc_station_positions[0][0],
                                self.qc_station_positions[-1][1] + 25,
                                text=f"Free: {qc_free}", font=('Arial', 9))

        # ---- Good / Scrap Boxes ----
        gx, gy = self.good_box_pos
        self.canvas.create_rectangle(gx-40, gy-30, gx+40, gy+30, fill='lightgreen', outline='black')
        self.canvas.create_text(gx, gy-40, text="Good Parts")
        self.canvas.create_text(gx, gy, text=f"{good_count}")

        sx, sy = self.scrap_box_pos
        self.canvas.create_rectangle(sx-40, sy-30, sx+40, sy+30, fill='lightcoral', outline='black')
        self.canvas.create_text(sx, sy-40, text="Scrap Parts")
        self.canvas.create_text(sx, sy, text=f"{scrap_count}")

        # Time label
        self.canvas.create_text(50, 50, text=f"Time: {time:.1f} h", anchor='w')

        # Legend
        leg_x = 20
        leg_y = 550
        self.canvas.create_text(leg_x, leg_y-10, text="Machine Types:", anchor='w')
        for i, (mtype, color) in enumerate(TYPE_COLORS.items()):
            self.canvas.create_rectangle(leg_x + i*80, leg_y, leg_x + i*80+15, leg_y+15, fill=color, outline='black')
            self.canvas.create_text(leg_x + i*80+20, leg_y+7, text=mtype, anchor='w')
        self.canvas.create_text(leg_x + 250, leg_y, text="Priority: Low=Blue, Std=Green, High=Orange, Crit=Red", anchor='w')

    def step(self):
        if self.animating:
            return
        if self.current_index < len(self.event_log) - 1:
            self.current_index += 1
            self.draw_snapshot(self.current_index)
            self.update_buttons()
        else:
            self.status_label.config(text="End of simulation.")
            self.play_btn.config(state=tk.DISABLED)
            self.step_btn.config(state=tk.DISABLED)

    def play(self):
        if self.playing:
            self.playing = False
            self.play_btn.config(text="Play")
            if self.after_id:
                self.master.after_cancel(self.after_id)
                self.after_id = None
            return
        if self.current_index >= len(self.event_log) - 1:
            self.reset()
            return
        self.playing = True
        self.play_btn.config(text="Pause")
        self.step_btn.config(state=tk.DISABLED)
        self.auto_step()

    def auto_step(self):
        if not self.playing:
            return
        if self.current_index < len(self.event_log) - 1:
            self.step()
            speed = self.speed_var.get()
            self.after_id = self.master.after(speed, self.auto_step)
        else:
            self.playing = False
            self.play_btn.config(text="Play")
            self.step_btn.config(state=tk.NORMAL)
            self.status_label.config(text="End of simulation.")

    def reset(self):
        self.playing = False
        if self.after_id:
            self.master.after_cancel(self.after_id)
            self.after_id = None
        self.current_index = 0
        self.play_btn.config(text="Play", state=tk.DISABLED)
        self.step_btn.config(state=tk.DISABLED)
        self.reset_btn.config(state=tk.DISABLED)
        self.run_btn.config(state=tk.NORMAL)
        self.canvas.delete("all")
        self.job_canvas_ovals.clear()
        self.event_log = []
        self.status_label.config(text="Ready")
        self.draw_empty()

    def update_buttons(self):
        if self.current_index < len(self.event_log) - 1:
            self.play_btn.config(state=tk.NORMAL)
            self.step_btn.config(state=tk.NORMAL)
        else:
            self.play_btn.config(state=tk.DISABLED)
            self.step_btn.config(state=tk.DISABLED)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    root = tk.Tk()
    app = CNC_GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()