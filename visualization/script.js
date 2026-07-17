const loadStatus = document.getElementById('loadStatus');
const loadingEl = document.getElementById('loading');

let pyodide = null;
let simulationFrames = [];
let currentIdx = 0;
let playing = false;
let intervalId = null;

const el = {
    clock: document.getElementById('simClock'),
    machinesGrid: document.getElementById('machinesGrid'),
    bufferArea: document.getElementById('bufferArea'),
    bufferCount: document.getElementById('bufferCount'),
    rejectionBins: document.getElementById('rejectionBins'),
    utilVal: document.getElementById('utilVal'),
    utilBar: document.getElementById('utilBar'),
    queueVal: document.getElementById('queueVal'),
    queueBar: document.getElementById('queueBar'),
    completedVal: document.getElementById('completedVal'),
    sparkline: document.getElementById('sparkline'),
    runBtn: document.getElementById('runBtn'),
    playBtn: document.getElementById('playBtn'),
    stepBackBtn: document.getElementById('stepBackBtn'),
    stepBtn: document.getElementById('stepBtn'),
    scenarioSelect: document.getElementById('scenarioSelect'),
    speedSlider: document.getElementById('speedSlider'),
    guideToggle: document.getElementById('guideToggle'),
    guidePanel: document.getElementById('guidePanel'),
};

async function loadPyodideAndPackages() {
    loadStatus.textContent = 'Loading Pyodide …';
    pyodide = await loadPyodide({
        indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.27.2/full/',
    });
    loadStatus.textContent = 'Installing simpy, numpy, pandas …';
    await pyodide.loadPackage(['micropip']);
    const micropip = pyodide.pyimport('micropip');
    await micropip.install(['simpy', 'numpy', 'pandas']);
    loadStatus.textContent = 'Running Python simulation code …';
    await pyodide.runPythonAsync(PYTHON_CODE);
    loadStatus.textContent = 'Ready!';
    setTimeout(() => loadingEl.classList.add('hidden'), 400);
    el.runBtn.disabled = false;
}

const PYTHON_CODE = `
import simpy
import numpy as np
import json
import math
from collections import defaultdict

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

def lognormal_params(mean, cv):
    sigma = np.sqrt(np.log(1 + cv**2))
    mu = np.log(mean) - 0.5 * sigma**2
    return mu, sigma

def sample_job_attributes(rng, job_id, arrival_time):
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

    return {
        "job_id": job_id,
        "arrival_time": arrival_time,
        "job_type": job_type,
        "subtype": subtype,
        "priority_class": priority_class,
        "priority_weight": priority_weight,
        "complexity_score": round(complexity_score, 3),
        "priority_level": round(priority_level, 3),
        "processing_time": round(processing_time, 3),
        "safe_wait_mean": round(safe_wait_mean, 3),
        "remaining_processing": round(processing_time, 3),
        "status": "waiting",
        "machine_number": None,
        "queue_entry_time": None,
        "start_time": None,
        "completion_time": None,
        "rejection_time": None,
        "rejection_reason": None,
        "timeout_deadline": None,
    }

class WorkCenter:
    def __init__(self, env, params, rng):
        self.env = env
        self.p = params
        self.rng = rng

        self.n_machines = params["n_machines"]
        self.machine_free = [True] * self.n_machines
        self.machine_down = [False] * self.n_machines
        self.machine_current_job = [None] * self.n_machines
        self.machine_busy_hours = [0.0] * self.n_machines
        self.machine_process_ref = [None] * self.n_machines
        self.machine_repair_end = [0.0] * self.n_machines

        self.buffer = []
        self.all_jobs = {}
        self.job_counter = 0
        self.rejection_counts = {
            "buffer_full": 0,
            "timeout": 0,
            "machine_failure": 0,
            "quality_reject": 0
        }
        self.completed_count = 0

        self.frames = []
        self._last_frame_time = -1.0
        self.frame_interval = 0.5

        for m in range(self.n_machines):
            env.process(self.machine_breakdown_process(m))
        env.process(self.monitor_process())

    def _record_visual_frame(self):
        if self.frames and abs(self.frames[-1]["time"] - self.env.now) < 1e-9:
            return
        frame = {
            "time": self.env.now,
            "machines": [],
            "buffer": [],
            "rejections": dict(self.rejection_counts),
            "completed_count": self.completed_count,
            "utilization": sum(1 for f in self.machine_free if not f) / self.n_machines,
            "queue_length": len(self.buffer),
        }
        for idx in range(self.n_machines):
            job = self.machine_current_job[idx]
            status = "idle"
            repair_left = 0.0
            if self.machine_down[idx]:
                if not self.machine_free[idx]:
                    status = "repair"
                    repair_left = max(0, self.machine_repair_end[idx] - self.env.now)
                else:
                    status = "down"
            elif not self.machine_free[idx]:
                status = "busy"
            progress = 0.0
            if job and status == "busy" and job.get("start_time"):
                elapsed = self.env.now - job["start_time"]
                total = job["processing_time"]
                progress = min(1.0, elapsed / total) if total > 0 else 0
            frame["machines"].append({
                "id": idx,
                "status": status,
                "job": {
                    "id": job["job_id"],
                    "priorityClass": job["priority_class"],
                    "jobType": job["job_type"],
                } if job else None,
                "progress": progress,
                "repairTimeLeft": repair_left,
            })
        for job in self.buffer:
            timeout_remaining = max(0, job["timeout_deadline"] - self.env.now) if job.get("timeout_deadline") else job["safe_wait_mean"]
            if job.get("timeout_deadline") and job.get("queue_entry_time"):
                timeout_total = job["timeout_deadline"] - job["queue_entry_time"]
            else:
                timeout_total = job["safe_wait_mean"]
            frame["buffer"].append({
                "id": job["job_id"],
                "priorityClass": job["priority_class"],
                "priorityLevel": job["priority_level"],
                "jobType": job["job_type"],
                "timeoutRemaining": timeout_remaining,
                "timeoutTotal": timeout_total,
                "safeWaitMean": job["safe_wait_mean"],
            })
        self.frames.append(frame)
        self._last_frame_time = self.env.now

    def job_generator(self):
        while self.env.now < self.p["horizon"]:
            interarrival = self.rng.exponential(self.p["arrival_mean"])
            yield self.env.timeout(interarrival)
            if self.env.now >= self.p["horizon"]:
                break
            self.job_counter += 1
            job = sample_job_attributes(self.rng, self.job_counter, self.env.now)
            self.all_jobs[job["job_id"]] = job
            self.env.process(self.handle_new_job(job))

    def handle_new_job(self, job):
        mean_insp = 0.25
        sd_insp = 0.08
        shape = mean_insp**2 / sd_insp**2
        scale = sd_insp**2 / mean_insp
        inspection_time = max(0.05, self.rng.gamma(shape, scale))
        yield self.env.timeout(inspection_time)
        self.try_assign_or_queue(job)

    def try_assign_or_queue(self, job):
        free_machine = self.first_free_machine()
        if free_machine is not None:
            self.assign_machine(job, free_machine)
        elif len(self.buffer) < self.p["buffer_cap"]:
            job["queue_entry_time"] = self.env.now
            job["timeout_deadline"] = self.env.now + self.rng.exponential(job["safe_wait_mean"])
            self.buffer.append(job)
            self.env.process(self.timeout_watcher(job))
            self._record_visual_frame()
        else:
            self.reject(job, "buffer_full")
            self._record_visual_frame()

    def first_free_machine(self):
        for m in range(self.n_machines):
            if self.machine_free[m] and not self.machine_down[m]:
                return m
        return None

    def assign_machine(self, job, machine_id):
        self.machine_free[machine_id] = False
        self.machine_current_job[machine_id] = job
        job["machine_number"] = machine_id
        job["start_time"] = self.env.now
        if job in self.buffer:
            self.buffer.remove(job)
        self.machine_process_ref[machine_id] = self.env.process(
            self.run_job(job, machine_id)
        )
        self._record_visual_frame()

    def run_job(self, job, machine_id):
        remaining = job["remaining_processing"]
        while remaining > 0:
            start = self.env.now
            try:
                yield self.env.timeout(remaining)
                remaining = 0
            except simpy.Interrupt:
                elapsed = self.env.now - start
                remaining -= elapsed
                remaining = max(remaining, 0)
                job["remaining_processing"] = remaining
                if self.rng.random() < self.p["scrap_on_breakdown"]:
                    self.reject(job, "machine_failure")
                    self.machine_current_job[machine_id] = None
                    self._record_visual_frame()
                    return
                while self.machine_down[machine_id]:
                    try:
                        yield self.env.timeout(0.1)
                    except simpy.Interrupt:
                        continue
                continue

        self.machine_busy_hours[machine_id] += job["processing_time"]
        scrap_prob = min(0.20, 0.02 * job["complexity_score"])
        if self.rng.random() < scrap_prob:
            self.reject(job, "quality_reject")
        else:
            job["status"] = "completed"
            job["completion_time"] = self.env.now
            self.completed_count += 1

        self.machine_free[machine_id] = True
        self.machine_current_job[machine_id] = None
        self._record_visual_frame()
        self.pull_next_job(machine_id)

    def pull_next_job(self, machine_id):
        if not self.buffer:
            return
        if self.p["queue_discipline"] == "PRIORITY":
            self.buffer.sort(key=lambda j: (-j["priority_level"], j["queue_entry_time"]))
        next_job = self.buffer[0]
        self.assign_machine(next_job, machine_id)

    def timeout_watcher(self, job):
        wait_limit = self.rng.exponential(job["safe_wait_mean"])
        yield self.env.timeout(wait_limit)
        if job in self.buffer and job["status"] == "waiting":
            self.buffer.remove(job)
            self.reject(job, "timeout")
            self._record_visual_frame()

    def reject(self, job, reason):
        if job["status"] != "waiting" and job["status"] != "processing":
            return
        job["status"] = "rejected"
        job["rejection_time"] = self.env.now
        job["rejection_reason"] = reason
        self.rejection_counts[reason] += 1

    def machine_breakdown_process(self, machine_id):
        while True:
            time_to_failure = self.rng.exponential(self.p["mtbf_mean"])
            yield self.env.timeout(time_to_failure)
            if self.env.now >= self.p["horizon"]:
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

            repair_time = self.rng.uniform(self.p["mttr_low"], self.p["mttr_high"])
            self.machine_repair_end[machine_id] = self.env.now + repair_time
            yield self.env.timeout(repair_time)
            self.machine_down[machine_id] = False
            self.machine_repair_end[machine_id] = 0.0
            self._record_visual_frame()
            if self.machine_current_job[machine_id] is None:
                self.machine_free[machine_id] = True
                self.pull_next_job(machine_id)

    def monitor_process(self):
        while True:
            if self.env.now - self._last_frame_time >= self.frame_interval:
                self._record_visual_frame()
            yield self.env.timeout(0.1)


def run_visual_simulation(scenario_name):
    params = {
        "name": scenario_name,
        "horizon": 720.0,
        "n_machines": 10,
        "buffer_cap": 8,
        "arrival_mean": 0.45,
        "queue_discipline": "FIFO",
        "mtbf_mean": 120.0,
        "mttr_low": 1.0,
        "mttr_high": 4.0,
        "scrap_on_breakdown": 0.25,
        "seed": 42,
    }
    if scenario_name == "capacity":
        params["n_machines"] = 13
        params["buffer_cap"] = 10
    elif scenario_name == "priority":
        params["queue_discipline"] = "PRIORITY"
    elif scenario_name == "demand":
        params["arrival_mean"] = 0.35
    elif scenario_name == "priority_demand":
        params["arrival_mean"] = 0.35
        params["queue_discipline"] = "PRIORITY"

    rng = np.random.default_rng(params["seed"])
    env = simpy.Environment()
    wc = WorkCenter(env, params, rng)
    env.process(wc.job_generator())
    env.run(until=params["horizon"] + 200)

    for job in wc.all_jobs.values():
        if job["status"] in ("waiting", "processing"):
            job["status"] = "censored"

    wc._record_visual_frame()
    return json.dumps(wc.frames)
`;

const app = {
    frames: [],
    currentIdx: 0,
    playing: false,
    intervalId: null,
    scenario: 'base',
    guideOpen: false,

    init() {
        el.runBtn.onclick = () => this.runSimulation();
        el.playBtn.onclick = () => this.togglePlay();
        el.stepBackBtn.onclick = () => this.stepBack();
        el.stepBtn.onclick = () => this.stepForward();
        el.guideToggle.onclick = () => this.toggleGuide();
        el.scenarioSelect.onchange = (e) => {
            this.scenario = e.target.value;
            this.stopPlay();
            this.frames = [];
            this.currentIdx = 0;
            this.renderEmpty();
        };
        this.renderEmpty();
        loadPyodideAndPackages();
    },

    toggleGuide() {
        this.guideOpen = !this.guideOpen;
        el.guidePanel.classList.toggle('open', this.guideOpen);
        el.guideToggle.classList.toggle('active', this.guideOpen);
        el.guideToggle.textContent = this.guideOpen ? '✕ Guide' : 'Guide';
    },

    async runSimulation() {
        if (!pyodide) return;
        this.stopPlay();
        el.runBtn.disabled = true;
        el.runBtn.textContent = 'Running...';
        el.playBtn.disabled = true;
        el.stepBtn.disabled = true;
        el.stepBackBtn.disabled = true;

        try {
            const scenario = this.scenario;
            const result = await pyodide.runPythonAsync(
                `run_visual_simulation('${scenario}')`
            );
            const frames = JSON.parse(result);
            this.frames = frames;
            this.currentIdx = 0;
            el.runBtn.textContent = 'Done';
            el.playBtn.disabled = false;
            el.stepBtn.disabled = false;
            el.stepBackBtn.disabled = true;
            this.renderFrame(0);
        } catch (err) {
            alert('Simulation error: ' + err);
            console.error(err);
        } finally {
            el.runBtn.disabled = false;
            el.runBtn.textContent = 'Run Simulation';
        }
    },

    togglePlay() {
        if (this.playing) this.stopPlay();
        else this.startPlay();
    },

    startPlay() {
        if (!this.frames.length) return;
        this.playing = true;
        el.playBtn.textContent = 'Pause';
        const delay = Math.max(50, 500 / parseInt(el.speedSlider.value));
        this.intervalId = setInterval(() => {
            if (this.currentIdx < this.frames.length - 1) {
                this.currentIdx++;
                this.renderFrame(this.currentIdx);
            } else {
                this.stopPlay();
            }
        }, delay);
    },

    stopPlay() {
        this.playing = false;
        if (this.intervalId) { clearInterval(this.intervalId);
            this.intervalId = null; }
        el.playBtn.textContent = 'Play';
    },

    stepForward() {
        if (!this.frames.length || this.playing) return;
        if (this.currentIdx < this.frames.length - 1) {
            this.currentIdx++;
            this.renderFrame(this.currentIdx);
        }
    },

    stepBack() {
        if (!this.frames.length || this.playing) return;
        if (this.currentIdx > 0) {
            this.currentIdx--;
            this.renderFrame(this.currentIdx);
        }
    },

    renderEmpty() {
        el.clock.textContent = 'Time 0.0 h';
        el.machinesGrid.innerHTML = '';
        el.bufferArea.innerHTML = '';
        el.bufferCount.textContent = '';
        el.rejectionBins.innerHTML = `
                    <div class="bin buffer-full"><div class="count">0</div><div class="label">Buffer Full</div></div>
                    <div class="bin timeout"><div class="count">0</div><div class="label">Timeout</div></div>
                    <div class="bin machine-fail"><div class="count">0</div><div class="label">Machine Fail</div></div>
                    <div class="bin quality-fail"><div class="count">0</div><div class="label">Quality Reject</div></div>
                `;
        el.utilVal.textContent = '0%';
        el.utilBar.style.width = '0%';
        el.queueVal.textContent = '0';
        el.queueBar.style.width = '0%';
        el.completedVal.textContent = '0';
        this.drawSparkline([]);
        el.stepBackBtn.disabled = true;
        el.stepBtn.disabled = true;
        el.playBtn.disabled = true;
    },

    renderFrame(idx) {
        const f = this.frames[idx];
        if (!f) return;
        el.clock.textContent = `Time ${f.time.toFixed(1)} h`;

        let html = '';
        for (const m of f.machines) {
            const cls = m.status;
            const label = m.status.toUpperCase();
            const prog = m.progress || 0;
            let jobTag = '';
            let repairText = '';
            let progressHtml = '';

            if (cls === 'repair') {
                const maxRepair = 4.0;
                const pct = Math.min(100, (m.repairTimeLeft / maxRepair) * 100);
                progressHtml = `<div class="progress-track"><div class="progress-fill repair-fill" style="width:${pct}%"></div></div>`;
                repairText = `<div class="repair-timer">${m.repairTimeLeft.toFixed(1)}h</div>`;
            } else {
                progressHtml = `<div class="progress-track"><div class="progress-fill" style="width:${prog*100}%"></div></div>`;
            }

            if (m.status === 'busy' && m.job) {
                const pc = m.job.priorityClass;
                const jt = m.job.jobType;
                jobTag =
                    `<span class="job-tag priority-${pc}"><span class="type-badge job-type-${jt}">${jt}</span></span>`;
            }

            html += `
                        <div class="machine ${cls}">
                            <div class="mid">M${m.id+1}</div>
                            <div class="status-label">${label}</div>
                            ${progressHtml}
                            ${jobTag}
                            ${repairText}
                        </div>
                    `;
        }
        el.machinesGrid.innerHTML = html;

        const cap = 8;
        let bufHtml = '';
        for (let i = 0; i < Math.max(cap, f.buffer.length); i++) {
            const job = f.buffer[i] || null;
            if (job) {
                const pct = Math.min(100, (job.timeoutRemaining / job.timeoutTotal) * 100);
                const pc = job.priorityClass;
                const jt = job.jobType;
                bufHtml += `
                            <div class="buffer-slot">
                                <div class="slot-content">
                                    <div class="pclass">${pc}</div>
                                    <span class="type-badge job-type-${jt}">${jt}</span>
                                    <div class="timeout-bar"><div class="timeout-fill" style="width:${Math.max(0,pct)}%"></div></div>
                                </div>
                            </div>
                        `;
            } else {
                bufHtml += `<div class="buffer-slot empty"></div>`;
            }
        }
        el.bufferArea.innerHTML = bufHtml;
        el.bufferCount.textContent = `(${f.buffer.length}/${cap})`;

        const r = f.rejections || {};
        const bufferFull = r.buffer_full ?? 0;
        const timeout = r.timeout ?? 0;
        const machineFail = r.machine_failure ?? 0;
        const qualityReject = r.quality_reject ?? 0;

        el.rejectionBins.innerHTML = `
                    <div class="bin buffer-full"><div class="count">${bufferFull}</div><div class="label">Buffer Full</div></div>
                    <div class="bin timeout"><div class="count">${timeout}</div><div class="label">Timeout</div></div>
                    <div class="bin machine-fail"><div class="count">${machineFail}</div><div class="label">Machine Fail</div></div>
                    <div class="bin quality-fail"><div class="count">${qualityReject}</div><div class="label">Quality Reject</div></div>
                `;

        const util = f.utilization || 0;
        const qLen = f.queue_length || 0;
        const completed = f.completed_count || 0;

        el.utilVal.textContent = `${Math.round(util*100)}%`;
        el.utilBar.style.width = `${util*100}%`;
        el.queueVal.textContent = qLen;
        el.queueBar.style.width = `${Math.min(100, (qLen/cap)*100)}%`;
        el.completedVal.textContent = completed;

        el.stepBackBtn.disabled = (this.currentIdx === 0);
        el.stepBtn.disabled = (this.currentIdx >= this.frames.length - 1);

        const hist = this.frames.slice(0, idx + 1).map(fr => fr.utilization || 0);
        this.drawSparkline(hist);
    },

    drawSparkline(data) {
        const canvas = el.sparkline;
        const ctx = canvas.getContext('2d');
        const w = canvas.parentElement.clientWidth || 200;
        const h = canvas.height || 44;
        canvas.width = w;
        canvas.height = h;
        ctx.clearRect(0, 0, w, h);
        if (data.length < 2) {
            ctx.fillStyle = '#30363d';
            ctx.font = '11px sans-serif';
            ctx.fillText('waiting...', 10, 26);
            return;
        }
        const maxVal = Math.max(1, ...data);
        const step = w / (data.length - 1);
        ctx.beginPath();
        ctx.strokeStyle = '#58a6ff';
        ctx.lineWidth = 1.5;
        for (let i = 0; i < data.length; i++) {
            const x = i * step;
            const y = h - (data[i] / maxVal) * (h - 6) - 3;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.lineTo(w, h);
        ctx.lineTo(0, h);
        ctx.closePath();
        ctx.fillStyle = '#58a6ff15';
        ctx.fill();
    }
};

document.addEventListener('DOMContentLoaded', () => app.init());