"""
Visual interface for the CNC simulation.
Allows scenario selection, runs a single replication, and animates the hourly
state of the work center.

Requirements:
    - Python 3.7+
    - tkinter (usually bundled)
    - matplotlib, numpy, pandas, simpy
"""

import tkinter as tk
from tkinter import ttk
import threading
import time

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd

# Import your simulation modules
from model import run_single_replication, ScenarioParams
from experiments import SCENARIOS, N_REPS, METRIC_KEYS


class SimulationVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("CNC Simulation Visualizer")
        self.root.geometry("1200x800")

        # State
        self.hourly_df = None
        self.metrics = None
        self.anim = None
        self.running = False

        # Build GUI
        self._create_widgets()
        self._setup_plot()

    def _create_widgets(self):
        """Top frame with controls."""
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Scenario dropdown
        ttk.Label(control_frame, text="Scenario:").pack(side=tk.LEFT, padx=5)
        self.scenario_var = tk.StringVar(value="base")
        scenario_menu = ttk.Combobox(
            control_frame,
            textvariable=self.scenario_var,
            values=list(SCENARIOS.keys()),
            state="readonly",
            width=20
        )
        scenario_menu.pack(side=tk.LEFT, padx=5)

        # Replication dropdown
        ttk.Label(control_frame, text="Replication:").pack(side=tk.LEFT, padx=5)
        self.rep_var = tk.StringVar(value="0")
        rep_menu = ttk.Combobox(
            control_frame,
            textvariable=self.rep_var,
            values=[str(i) for i in range(N_REPS)],
            state="readonly",
            width=5
        )
        rep_menu.pack(side=tk.LEFT, padx=5)

        # Run button
        self.run_btn = ttk.Button(
            control_frame,
            text="Run Simulation",
            command=self._on_run
        )
        self.run_btn.pack(side=tk.LEFT, padx=10)

        # Progress bar (indeterminate while running)
        self.progress = ttk.Progressbar(
            control_frame,
            mode="indeterminate",
            length=200
        )
        self.progress.pack(side=tk.LEFT, padx=10)
        self.progress.pack_forget()   # hidden initially

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(control_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, padx=10)

        # Stats display (right side)
        self.stats_text = tk.Text(
            control_frame,
            height=6,
            width=50,
            font=("Courier", 9),
            wrap=tk.NONE
        )
        self.stats_text.pack(side=tk.RIGHT, padx=10, fill=tk.Y)

    def _setup_plot(self):
        """Create matplotlib figure and canvas."""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(11, 6))
        self.fig.subplots_adjust(hspace=0.3, wspace=0.3)

        # Axis labels
        self.axes[0, 0].set_title("Machines")
        self.axes[0, 0].set_ylabel("count")
        self.axes[0, 0].set_xlabel("Time (h)")
        self.axes[0, 1].set_title("Queue Length")
        self.axes[0, 1].set_ylabel("jobs")
        self.axes[0, 1].set_xlabel("Time (h)")
        self.axes[1, 0].set_title("Machine Utilization")
        self.axes[1, 0].set_ylabel("fraction")
        self.axes[1, 0].set_xlabel("Time (h)")
        self.axes[1, 1].set_title("System Pressure")
        self.axes[1, 1].set_ylabel("(busy + queue)")
        self.axes[1, 1].set_xlabel("Time (h)")

        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _on_run(self):
        """Start simulation in a background thread."""
        if self.running:
            return
        self.running = True
        self.run_btn.config(state=tk.DISABLED)
        self.progress.pack(side=tk.LEFT, padx=10)
        self.progress.start()
        self.status_var.set("Running simulation...")
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, "Running...\n")

        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()

        # Launch thread
        thread = threading.Thread(target=self._run_simulation, daemon=True)
        thread.start()

    def _run_simulation(self):
        """Execute one replication and then schedule the animation."""
        scen_name = self.scenario_var.get()
        rep = int(self.rep_var.get())
        params = SCENARIOS[scen_name]

        seed = hash((scen_name, rep)) % (2**31)
        result = run_single_replication(params, seed=seed)

        self.hourly_df = result["hourly_df"]
        self.metrics = result["metrics"]

        # Update GUI in main thread
        self.root.after(0, self._simulation_done)

    def _simulation_done(self):
        """Called after simulation finishes. Start animation and show stats."""
        self.progress.stop()
        self.progress.pack_forget()
        self.run_btn.config(state=tk.NORMAL)
        self.running = False
        self.status_var.set("Simulation complete. Animating...")

        # Display key metrics
        self._update_stats()

        # Start animation
        if self.anim is not None:
            self.anim.event_source.stop()
        self.anim = FuncAnimation(
            self.fig,
            self._animate,
            frames=len(self.hourly_df),
            interval=50,          # ms per frame
            repeat=False,
            cache_frame_data=False
        )
        self.canvas.draw()
        self.status_var.set("Animation running")

    def _update_stats(self):
        """Show summary metrics in the text box."""
        if self.metrics is None:
            return
        stats = [
            f"Admission rate   : {self.metrics['admission_rate']:.3f}",
            f"Rejection rate   : {self.metrics['rejection_rate']:.3f}",
            f"Avg wait (h)     : {self.metrics['avg_wait_hours']:.3f}",
            f"Avg utilization  : {self.metrics['avg_machine_utilization']:.3f}",
            f"Peak queue       : {self.metrics['peak_queue_length']:.0f}",
            f"Scrap rate       : {self.metrics['scrap_rate']:.3f}",
        ]
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, "\n".join(stats))

    def _animate(self, i):
        """Update plots for frame i."""
        df = self.hourly_df
        if df is None or len(df) == 0:
            return

        # Get data up to current time
        current_time = df.iloc[i]["time"]
        data = df.iloc[:i+1]

        # Clear and replot each subplot
        ax1 = self.axes[0, 0]
        ax1.clear()
        ax1.plot(data["time"], data["machines_busy"], "b-", label="Busy")
        ax1.plot(data["time"], data["available_machines"], "r--", label="Available")
        ax1.axvline(current_time, color="k", linestyle=":", alpha=0.5)
        ax1.set_title("Machines")
        ax1.set_ylabel("count")
        ax1.set_xlabel("Time (h)")
        ax1.legend(loc="upper right")
        ax1.set_xlim(0, df["time"].max())
        ax1.set_ylim(0, max(df["machines_busy"].max(), df["available_machines"].max()) + 1)

        ax2 = self.axes[0, 1]
        ax2.clear()
        ax2.plot(data["time"], data["queue_length"], "g-")
        ax2.axvline(current_time, color="k", linestyle=":", alpha=0.5)
        ax2.set_title("Queue Length")
        ax2.set_ylabel("jobs")
        ax2.set_xlabel("Time (h)")
        ax2.set_xlim(0, df["time"].max())
        ax2.set_ylim(0, df["queue_length"].max() + 1)

        ax3 = self.axes[1, 0]
        ax3.clear()
        ax3.plot(data["time"], data["machine_utilization"], "m-")
        ax3.axvline(current_time, color="k", linestyle=":", alpha=0.5)
        ax3.set_title("Machine Utilization")
        ax3.set_ylabel("fraction")
        ax3.set_xlabel("Time (h)")
        ax3.set_xlim(0, df["time"].max())
        ax3.set_ylim(0, 1.05)

        ax4 = self.axes[1, 1]
        ax4.clear()
        ax4.plot(data["time"], data["system_pressure"], "c-")
        # Overlay the over‑capacity indicator
        over = data[data["over_capacity_pressure"]]
        if len(over):
            ax4.scatter(over["time"], over["system_pressure"],
                        color="red", s=20, alpha=0.6, label="Over capacity")
        ax4.axvline(current_time, color="k", linestyle=":", alpha=0.5)
        ax4.set_title("System Pressure")
        ax4.set_ylabel("(busy + queue)")
        ax4.set_xlabel("Time (h)")
        ax4.legend(loc="upper right")
        ax4.set_xlim(0, df["time"].max())
        ymax = df["system_pressure"].max() + 2
        ax4.set_ylim(0, ymax)

        self.fig.suptitle(f"Time: {current_time:.1f} h", fontsize=12)
        self.canvas.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationVisualizer(root)
    root.mainloop()