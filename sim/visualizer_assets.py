import sys
import os
import tkinter as tk
from tkinter import ttk
import simpy
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# ----------------------------------------------------------------------
# Copy of the original model (with modifications for quality check + event logging)
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

@dataclass
class ScenarioParams:
    name: str = "base"
    horizon_hours: float = 720.0
    n_machines: int = 10
    buffer_capacity: int = 8
    arrival_mean: float = 0.45
    queue_discipline: str = "FIFO"
    mtbf_mean: float = 120.0
    mttr_low: float = 1.0
    mttr_high: float = 4.0
    p_scrap_on_breakdown: float = 0.25
    seed: int = 0
    qc_mean: float = 0.2  # mean quality check time (exponential)

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
    safe_wait_mean: float
    status: str = "in_system"
    machine_number: Optional[int] = None
    queue_entry_time: Optional[float] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    rejection_time: Optional[float] = None
    rejection_reason: Optional[str] = None
    remaining_processing: Optional[float] = None

def lognormal_params(mean, cv):
    sigma = np.sqrt(np.log(1 + cv**2))
    mu = np.log(mean) - 0.5 * sigma**2
    return mu, sigma

def sample_job_attributes(rng: np.random.Generator, job_id: int, arrival_time: float) -> Job:
    p_names = list(PRIORITY_CLASSES.keys())
    p_probs = [PRIORITY_CLASSES[n][0] for n in p_names]
    priority_class = rng.choice(p_names, p=p_probs)
    priority_weight = PRIORITY_CLASSES[priority_class][1]

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

    mu, sigma = lognormal_params(proc_mean, proc_cv)
    processing_time = rng.lognormal(mu, sigma)

    quality_risk = 2.0 * rng.beta(2, 2)
    complexity_score = base_complexity + quality_risk

    norm_complexity = complexity_score / 7.0
    priority_level = 0.6 * priority_weight + 0.4 * (norm_complexity * 5)

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

# ----------------------------------------------------------------------
# Extended WorkCenter with quality check and event logging
# ----------------------------------------------------------------------
class WorkCenterWithQC:
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

        # Quality check
        self.qc_busy = [False] * params.n_machines
        self.qc_job = [None] * params.n_machines
        self.good_count = 0
        self.scrap_count = 0

        # Event log for GUI
        self.event_log: List[Dict[str, Any]] = []
        self.job_attrs: Dict[int, Dict[str, Any]] = {}  # job_id -> {color, label, ...}

        # Start processes
        for m in range(params.n_machines):
            env.process(self.machine_breakdown_process(m))
        env.process(self.monitor_process())
        self.record_snapshot()

    def record_snapshot(self):
        """Capture current state and append to event_log."""
        snapshot = {
            'time': self.env.now,
            'buffer': [job.job_id for job in self.buffer],
            'machines': [
                {
                    'state': 'broken' if self.machine_down[i] else
                              ('working' if self.machine_current_job[i] is not None else 'idle'),
                    'job_id': self.machine_current_job[i].job_id if self.machine_current_job[i] else None,
                }
                for i in range(self.p.n_machines)
            ],
            'qc': [
                {
                    'state': 'inspecting' if self.qc_job[i] is not None else 'idle',
                    'job_id': self.qc_job[i].job_id if self.qc_job[i] else None,
                }
                for i in range(self.p.n_machines)
            ],
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
            # Store attributes for GUI
            self.job_attrs[job.job_id] = {
                'color': self._priority_color(job.priority_class),
                'label': f'J{job.job_id}',
                'priority': job.priority_class,
            }
            self.env.process(self.handle_new_job(job))

    def _priority_color(self, pclass):
        colors = {'Low': 'blue', 'Standard': 'green', 'High': 'orange', 'Critical': 'red'}
        return colors.get(pclass, 'gray')

    def handle_new_job(self, job: Job):
        # Inspection time (Gamma)
        mean_insp = 0.25
        sd_insp = 0.08
        shape = mean_insp**2 / sd_insp**2
        scale = sd_insp**2 / mean_insp
        inspection_time = max(0.05, self.rng.gamma(shape, scale))
        yield self.env.timeout(inspection_time)
        self.record_snapshot()
        self.try_assign_or_queue(job)

    def try_assign_or_queue(self, job: Job):
        free_machine = self.first_free_machine()
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
        self.record_snapshot()

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
                    self.record_snapshot()
                    return
                while self.machine_down[machine_id]:
                    try:
                        yield self.env.timeout(0.1)
                    except simpy.Interrupt:
                        continue
                # after repair, continue processing

        # Processing finished
        self.machine_busy_hours[machine_id] += job.processing_time
        # Machine becomes free, pull next job
        self.machine_free[machine_id] = True
        self.machine_current_job[machine_id] = None
        self.record_snapshot()
        self.pull_next_job(machine_id)

        # Now start quality check
        self.env.process(self.quality_check(job, machine_id))

    def quality_check(self, job: Job, machine_id: int):
        self.qc_busy[machine_id] = True
        self.qc_job[machine_id] = job
        self.record_snapshot()
        qc_time = self.rng.exponential(self.p.qc_mean)
        yield self.env.timeout(qc_time)
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
        self.qc_busy[machine_id] = False
        self.qc_job[machine_id] = None
        self.record_snapshot()

    def pull_next_job(self, machine_id: int):
        if not self.buffer:
            return
        if self.p.queue_discipline == "PRIORITY":
            self.buffer.sort(key=lambda j: (-j.priority_level, j.queue_entry_time))
        next_job = self.buffer[0]
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
        self.record_snapshot()

    # ---------- Machine breakdown ----------
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
            self.record_snapshot()

            repair_time = self.rng.uniform(self.p.mttr_low, self.p.mttr_high)
            yield self.env.timeout(repair_time)
            self.machine_down[machine_id] = False
            if self.machine_current_job[machine_id] is None:
                self.machine_free[machine_id] = True
                self.pull_next_job(machine_id)
            self.record_snapshot()

    # ---------- Hourly monitoring (for metrics) ----------
    def monitor_process(self):
        while True:
            busy = sum(1 for i in range(self.p.n_machines) if not self.machine_free[i] and not self.machine_down[i])
            machines_down = sum(self.machine_down)
            available = self.p.n_machines - busy - machines_down
            queue_length = len(self.buffer)
            queue_full = queue_length >= self.p.buffer_capacity
            utilization = busy / self.p.n_machines
            pressure = busy + queue_length
            over_capacity_pressure = pressure > self.p.n_machines

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
# Run single replication with event logging
# ----------------------------------------------------------------------
def run_single_replication_with_events(params: ScenarioParams, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    wc = WorkCenterWithQC(env, params, rng)
    env.process(wc.job_generator())
    env.run(until=params.horizon_hours + 200)

    # Build job_df (similar to original)
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
        })
    job_df = pd.DataFrame(job_rows)
    hourly_df = pd.DataFrame(wc.hourly_records)

    # Metrics (simplified)
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

class CNC_GUI:
    def __init__(self, master):
        self.master = master
        master.title("CNC Manufacturing Line Simulation")

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
        self.job_canvas_ids = {}  # job_id -> canvas oval id
        self.playing = False
        self.after_id = None

        # Positions calculated dynamically
        self.machine_positions = []  # (x, y) for each CNC
        self.qc_positions = []       # (x, y) for each QC
        self.buffer_positions = []   # list of (x, y) for buffer slots
        self.good_box_pos = None
        self.scrap_box_pos = None

        # For animation
        self.animating = False

        # Draw initial empty layout
        self.draw_empty()

    def draw_empty(self):
        self.canvas.delete("all")
        # We'll redraw when simulation is run.

    def run_simulation(self):
        self.reset()  # clear previous
        scenario_name = self.scenario_var.get()
        params = SCENARIOS[scenario_name]
        # Use a fixed seed for reproducibility
        seed = 12345
        self.status_label.config(text="Running simulation...")
        self.master.update()

        result = run_single_replication_with_events(params, seed)
        self.event_log = result["event_log"]
        self.job_attrs = result["job_attrs"]
        self.params = params
        self.metrics = result["metrics"]

        self.status_label.config(text=f"Simulation done. {len(self.event_log)} events recorded.")
        self.current_index = 0
        self.play_btn.config(state=tk.NORMAL)
        self.step_btn.config(state=tk.NORMAL)
        self.reset_btn.config(state=tk.NORMAL)
        self.run_btn.config(state=tk.DISABLED)

        # Setup canvas positions based on number of machines
        self.setup_positions()
        # Draw initial state
        self.draw_snapshot(self.current_index)

    def setup_positions(self):
        n = self.params.n_machines
        canvas_width = int(self.canvas.cget("width"))
        canvas_height = int(self.canvas.cget("height"))

        # Buffer on left
        buffer_x = 80
        buffer_y_start = 150
        buffer_spacing = 30
        self.buffer_positions = [(buffer_x, buffer_y_start + i*buffer_spacing) for i in range(self.params.buffer_capacity)]

        # Machines row
        machine_y = 200
        x_start = 200
        x_spacing = (canvas_width - x_start - 200) // (n + 1)
        self.machine_positions = []
        for i in range(n):
            x = x_start + (i+1)*x_spacing
            self.machine_positions.append((x, machine_y))

        # QC row below
        qc_y = machine_y + 100
        self.qc_positions = [(x, qc_y) for x, y in self.machine_positions]

        # Good and scrap boxes on right
        self.good_box_pos = (canvas_width - 150, 150)
        self.scrap_box_pos = (canvas_width - 150, 400)

    def draw_snapshot(self, idx):
        self.canvas.delete("all")
        snapshot = self.event_log[idx]
        time = snapshot['time']
        buffer_ids = snapshot['buffer']
        machines = snapshot['machines']
        qc = snapshot['qc']
        good_count = snapshot['good_count']
        scrap_count = snapshot['scrap_count']

        # Draw static elements: machine labels, boxes, etc.
        self.draw_static_elements()

        # Draw buffer parts
        for pos_idx, job_id in enumerate(buffer_ids):
            if pos_idx < len(self.buffer_positions):
                x, y = self.buffer_positions[pos_idx]
                color = self.job_attrs.get(job_id, {}).get('color', 'gray')
                self.canvas.create_oval(x-10, y-10, x+10, y+10, fill=color, outline='black')
                # store id for animation
                self.job_canvas_ids[job_id] = (x, y, 'buffer', pos_idx)

        # Draw machines
        for i, (x, y) in enumerate(self.machine_positions):
            state = machines[i]['state']
            color = {'idle':'green', 'working':'yellow', 'broken':'red'}.get(state, 'gray')
            # machine rectangle
            self.canvas.create_rectangle(x-25, y-25, x+25, y+25, fill=color, outline='black')
            self.canvas.create_text(x, y-40, text=f"M{i+1}")
            if state == 'working':
                job_id = machines[i]['job_id']
                if job_id is not None:
                    c = self.job_attrs.get(job_id, {}).get('color', 'gray')
                    self.canvas.create_oval(x-12, y-12, x+12, y+12, fill=c, outline='black')
                    self.canvas.create_text(x, y, text=str(job_id), font=('Arial', 8))
                    self.job_canvas_ids[job_id] = (x, y, 'machine', i)

        # Draw QC
        for i, (x, y) in enumerate(self.qc_positions):
            state = qc[i]['state']
            color = 'lightblue' if state == 'idle' else 'darkblue'
            self.canvas.create_rectangle(x-15, y-15, x+15, y+15, fill=color, outline='black')
            self.canvas.create_text(x, y-25, text=f"QC{i+1}")
            if state == 'inspecting':
                job_id = qc[i]['job_id']
                if job_id is not None:
                    c = self.job_attrs.get(job_id, {}).get('color', 'gray')
                    self.canvas.create_oval(x-10, y-10, x+10, y+10, fill=c, outline='black')
                    self.canvas.create_text(x, y, text=str(job_id), font=('Arial', 8))
                    self.job_canvas_ids[job_id] = (x, y, 'qc', i)

        # Draw good/scrap boxes
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

        # Store current snapshot for animation
        self.current_snapshot = snapshot

    def draw_static_elements(self):
        # Draw grid lines, labels, etc.
        n = self.params.n_machines
        # We can add machine numbers, QC numbers already in draw_snapshot

    def step(self):
        if self.animating:
            return
        if self.current_index < len(self.event_log) - 1:
            self.current_index += 1
            self.animate_transition()
        else:
            self.status_label.config(text="End of simulation.")
            self.play_btn.config(state=tk.DISABLED)
            self.step_btn.config(state=tk.DISABLED)

    def animate_transition(self):
        # Animate changes from previous snapshot to current
        if self.current_index == 0:
            self.draw_snapshot(self.current_index)
            return
        prev = self.event_log[self.current_index - 1]
        curr = self.event_log[self.current_index]
        # Determine which jobs moved
        # We'll move jobs from their prev location to curr location
        # Gather all jobs that are in either snapshot
        all_jobs = set()
        for snap in (prev, curr):
            all_jobs.update(snap['buffer'])
            for m in snap['machines']:
                if m['job_id'] is not None:
                    all_jobs.add(m['job_id'])
            for q in snap['qc']:
                if q['job_id'] is not None:
                    all_jobs.add(q['job_id'])

        # We'll animate each job's position change
        # For each job, find its position in prev and curr
        # If different, animate
        # Also handle new jobs appearing in curr
        # We'll create a list of movements: (job_id, start_x, start_y, end_x, end_y)
        movements = []
        for job_id in all_jobs:
            start_pos = self.get_job_position(job_id, prev)
            end_pos = self.get_job_position(job_id, curr)
            if start_pos != end_pos:
                # If job not in prev (new arrival), start at buffer position? Actually we can start at off-screen or buffer
                if start_pos is None:
                    # new job, start from buffer entry point
                    start_pos = (self.buffer_positions[0][0] - 40, self.buffer_positions[0][1])
                # If job not in curr, it might have been removed (scrap/good) - we can move to box
                if end_pos is None:
                    # If job is in good/scrap? We can determine from status? But we can just leave it at its last position.
                    # Actually, when job goes to good/scrap, it disappears from states, so we should move it to box.
                    # Determine if job is in good or scrap by checking curr good/scrap counts? Not easy.
                    # We'll rely on status from job_attrs? Not stored.
                    # Instead, we can check if job's status is completed/rejected in the result.
                    # We'll just move it to the box if it's not in buffer/machine/qc.
                    # We'll infer from curr: if job not in buffer, not on machine, not on qc, then it must have gone to box.
                    # But we don't know which box. We'll use the job status from all_jobs? We don't have it here.
                    # For simplicity, we'll just not animate disappearance; we'll keep it at its last position.
                    # We'll handle new arrivals and moves.
                    continue
                if start_pos != end_pos:
                    movements.append((job_id, start_pos[0], start_pos[1], end_pos[0], end_pos[1]))

        if not movements:
            # No movement, just draw current
            self.draw_snapshot(self.current_index)
            return

        # Animate all movements simultaneously
        self.animating = True
        # For each movement, we need to create or update canvas oval
        # We'll store the canvas ids for jobs
        # If a job doesn't have a canvas id, create one
        for job_id, sx, sy, ex, ey in movements:
            if job_id not in self.job_canvas_ids:
                # create oval at start position
                color = self.job_attrs.get(job_id, {}).get('color', 'gray')
                oval = self.canvas.create_oval(sx-10, sy-10, sx+10, sy+10, fill=color, outline='black')
                self.job_canvas_ids[job_id] = oval
            else:
                oval = self.job_canvas_ids[job_id]
                # move to start
                self.canvas.coords(oval, sx-10, sy-10, sx+10, sy+10)

        # Now animate over 20 steps
        steps = 20
        delay = 20  # ms
        for step in range(1, steps+1):
            frac = step / steps
            for job_id, sx, sy, ex, ey in movements:
                oval = self.job_canvas_ids[job_id]
                x = sx + (ex - sx) * frac
                y = sy + (ey - sy) * frac
                self.canvas.coords(oval, x-10, y-10, x+10, y+10)
            self.master.update()
            if step < steps:
                self.master.after(delay, lambda: None)  # simple wait
            else:
                # final draw
                self.draw_snapshot(self.current_index)
                self.animating = False
                self.update_buttons()

        # For simplicity, we'll just do a quick animation by scheduling updates
        # But we can use after to schedule steps
        # We'll implement a recursive animation function
        # Let's restructure: use after to animate

    def get_job_position(self, job_id, snapshot):
        # Return (x,y) or None if job not present
        # Check buffer
        if job_id in snapshot['buffer']:
            idx = snapshot['buffer'].index(job_id)
            if idx < len(self.buffer_positions):
                return self.buffer_positions[idx]
        # Check machines
        for i, m in enumerate(snapshot['machines']):
            if m['job_id'] == job_id:
                return self.machine_positions[i]
        # Check QC
        for i, q in enumerate(snapshot['qc']):
            if q['job_id'] == job_id:
                return self.qc_positions[i]
        # Check if in good or scrap? We can't know from snapshot.
        return None

    def play(self):
        if self.playing:
            # Pause
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
            # Wait for animation to finish? We'll check animating flag.
            if self.animating:
                self.master.after(100, self.auto_step)  # wait a bit
            else:
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
        self.job_canvas_ids.clear()
        self.event_log = []
        self.status_label.config(text="Ready")
        self.draw_empty()

    def update_buttons(self):
        # Enable/disable based on state
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