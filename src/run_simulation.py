"""
run_simulation.py
Core simulation runner for Jaam Ctrl.

Three modes:
  1. "fixed"    – SUMO's built-in fixed-time programs (baseline)
  2. "adaptive" – Rule-based adaptive controller (signal_controller.py)
  3. "rl"       – Trained PPO agent (rl_agent.py)

Returns a SimResult dataclass containing:
  - metrics:     dict with avg_delay, avg_stops, throughput, improvement
  - gps_df:      pandas DataFrame with GPS probe records
  - phase_log:   list of per-step phase info (for animation / heatmap)
"""

from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

try:
    import traci
    TRACI_OK = True
except ImportError:
    TRACI_OK = False

from gps_generator import (
    collect_gps_frame,
    select_probe_vehicles,
    build_dataframe,
)
from signal_controller import RuleBasedController, FixedTimeController

# ── Config ────────────────────────────────────────────────────────────────────
SUMO_CFG     = os.path.join(os.path.dirname(__file__), "..", "sumo", "config.sumocfg")
SIM_DURATION = 1800   # seconds
TL_IDS       = ["TL0", "TL1", "TL2"]

JUNCTION_EDGES = {
    "TL0": {"ew": ["W0J0", "J1J0"],  "ns": ["N0J0", "S0J0"]},
    "TL1": {"ew": ["J0J1", "J2J1"],  "ns": ["N1J1", "S1J1"]},
    "TL2": {"ew": ["J1J2"],          "ns": ["N2J2", "S2J2"]},
}


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class SimResult:
    mode:       str
    metrics:    dict  = field(default_factory=dict)
    gps_df:     pd.DataFrame = field(default_factory=pd.DataFrame)
    phase_log:  list  = field(default_factory=list)
    raw_delays: list  = field(default_factory=list)
    raw_stops:  list  = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sumo_cmd(seed: int = 42) -> list[str]:
    return [
        "sumo",
        "-c", SUMO_CFG,
        "--seed", str(seed),
        "--no-warnings",
        "--no-step-log",
        "--quit-on-end",
    ]


def _vehicle_delay(vid: str) -> float:
    try:
        return traci.vehicle.getAccumulatedWaitingTime(vid)
    except Exception:
        return 0.0


def _vehicle_stops(vid: str) -> int:
    """Approximate stops as integer divisions of accumulated waiting time by 5 s."""
    try:
        return int(traci.vehicle.getWaitingTime(vid) > 0)
    except Exception:
        return 0


def _build_metrics(
    all_delays:  list[float],
    all_stops:   list[int],
    arrived:     int,
    mode:        str,
    baseline_delay: float | None = None,
) -> dict:
    avg_delay = float(np.mean(all_delays)) if all_delays else 0.0
    avg_stops = float(np.mean(all_stops))  if all_stops  else 0.0
    improvement = 0.0
    if baseline_delay and baseline_delay > 0:
        improvement = round((baseline_delay - avg_delay) / baseline_delay * 100, 1)
    return {
        "mode":        mode,
        "avg_delay_s": round(avg_delay, 2),
        "avg_stops":   round(avg_stops, 2),
        "throughput":  arrived,
        "improvement": improvement,
    }


# ── Main simulation function ──────────────────────────────────────────────────

def run_simulation(
    mode:            str   = "fixed",
    traffic_scale:   float = 1.0,
    accident_step:   int   = -1,       # -1 = no accident
    seed:            int   = 42,
    baseline_delay:  float | None = None,
    ppo_model=None,                    # loaded PPO model (mode="rl")
    progress_cb=None,                  # callable(step, total)
) -> SimResult:
    """
    Run one full simulation and return a SimResult.

    Parameters
    ----------
    mode            : "fixed" | "adaptive" | "rl"
    traffic_scale   : multiplier for vehicle counts (via SUMO flow scaling)
    accident_step   : simulation second at which to inject a blocking vehicle
    seed            : SUMO random seed
    baseline_delay  : avg_delay from a previous fixed run (for % improvement)
    ppo_model       : loaded SB3 PPO model (only used when mode="rl")
    progress_cb     : optional callback(step, total) for Streamlit progress bar
    """
    if not TRACI_OK:
        return _mock_result(mode, baseline_delay)

    # ── Select controller ────────────────────────────────────────────────────
    if mode == "adaptive":
        controller = RuleBasedController()
    elif mode == "rl":
        # RL is handled separately via rl_agent.run_ppo_episode; fall back if None
        if ppo_model is None:
            controller = RuleBasedController()
            mode = "adaptive"
        else:
            controller = None   # PPO model drives actions
    else:
        controller = FixedTimeController()

    # ── Close any existing TraCI connections (safety for Streamlit reruns) ───
    try:
        if traci.isLoaded():
            traci.close()
    except RuntimeError:
        pass  # No active connection

    # ── Start TraCI ──────────────────────────────────────────────────────────
    try:
        traci.start(_sumo_cmd(seed))

        gps_records: list[dict] = []
        phase_log:   list[dict] = []
        delays_per_vehicle:  dict[str, float] = {}
        stops_per_vehicle:   dict[str, int]   = {}
        arrived_count = 0
        probe_vids: set = set()

        # RL state
        rl_obs = None
        if mode == "rl" and ppo_model is not None:
            rl_obs = _build_rl_obs()

        for step in range(SIM_DURATION):
            # ── Update probe set every 60 s ──────────────────────────────────────
            if step % 60 == 0:
                all_ids = traci.vehicle.getIDList()
                probe_vids = select_probe_vehicles(list(all_ids))

            # ── Collect GPS ──────────────────────────────────────────────────────
            gps_records.extend(collect_gps_frame(step, probe_vids))

            # ── Control signals ──────────────────────────────────────────────────
            if mode == "rl" and ppo_model is not None and step % 10 == 0:
                rl_obs = _build_rl_obs()
                action, _ = ppo_model.predict(rl_obs, deterministic=True)
                _apply_rl_action(int(action))
            elif controller is not None:
                controller.step(step)

            # ── Phase snapshot (for dashboard) ───────────────────────────────────
            if step % 10 == 0:
                phase_log.append(
                    {
                        "step": step,
                        **{tl: _safe_phase(tl) for tl in TL_IDS},
                    }
                )

            # ── Accumulate metrics ───────────────────────────────────────────────
            for vid in traci.vehicle.getIDList():
                delays_per_vehicle[vid] = _vehicle_delay(vid)
                stops_per_vehicle[vid]  = stops_per_vehicle.get(vid, 0) + _vehicle_stops(vid)

            arrived_count += traci.simulation.getArrivedNumber()

            # ── Accident injection ───────────────────────────────────────────────
            if step == accident_step:
                _inject_accident()

            # ── Advance simulation ───────────────────────────────────────────────
            traci.simulationStep()

            if progress_cb:
                progress_cb(step + 1, SIM_DURATION)

        all_delays = list(delays_per_vehicle.values())
        all_stops  = list(stops_per_vehicle.values())

        return SimResult(
            mode=mode,
            metrics=_build_metrics(all_delays, all_stops, arrived_count, mode, baseline_delay),
            gps_df=build_dataframe(gps_records),
            phase_log=phase_log,
            raw_delays=all_delays,
            raw_stops=all_stops,
        )
    finally:
        traci.close()


# ── RL helpers ────────────────────────────────────────────────────────────────

def _build_rl_obs() -> "np.ndarray":
    import numpy as np
    MAX_Q = 20.0
    obs = []
    for tl in TL_IDS:
        edges = JUNCTION_EDGES[tl]
        q_ew = _sum_queue(edges["ew"]) / MAX_Q
        q_ns = _sum_queue(edges["ns"]) / MAX_Q
        phase = _safe_phase(tl)
        ew_g  = float(phase in (0,))
        ns_g  = float(phase in (2,))
        obs.extend([min(q_ew, 1.0), min(q_ns, 1.0), ew_g, ns_g])
    return np.array(obs, dtype=np.float32)


def _apply_rl_action(action: int):
    YELLOW_DUR = 5
    MIN_PHASE  = 15
    for i, tl in enumerate(TL_IDS):
        if action & (1 << i):
            try:
                phase = traci.trafficlight.getPhase(tl)
                if phase in (0, 2):
                    next_phase = 1 if phase == 0 else 3
                    traci.trafficlight.setPhase(tl, next_phase)
                    traci.trafficlight.setPhaseDuration(tl, YELLOW_DUR)
            except Exception:
                pass


def _sum_queue(edges: list) -> float:
    total = 0.0
    for e in edges:
        try:
            vids = traci.edge.getLastStepVehicleIDs(e)
            total += sum(1 for v in vids if traci.vehicle.getSpeed(v) < 0.1)
        except Exception:
            pass
    return total


def _safe_phase(tl: str) -> int:
    try:
        return traci.trafficlight.getPhase(tl)
    except Exception:
        return 0


def _inject_accident():
    """Slow down a random vehicle to simulate an accident."""
    try:
        vids = traci.vehicle.getIDList()
        if vids:
            victim = random.choice(list(vids))
            traci.vehicle.setSpeed(victim, 0.0)
            traci.vehicle.setSpeedMode(victim, 0)
    except Exception:
        pass


# ── Mock result (when SUMO not available) ─────────────────────────────────────

def _mock_result(mode: str, baseline_delay: float | None) -> SimResult:
    """
    Return realistic-looking synthetic results so the dashboard works
    even without a SUMO installation (for rapid UI development).
    """
    rng = np.random.default_rng(99)

    if mode == "fixed":
        avg_delay  = rng.uniform(45, 65)
        avg_stops  = rng.uniform(3.5, 6.0)
        throughput = int(rng.integers(900, 1100))
        improv     = 0.0
    elif mode == "adaptive":
        avg_delay  = rng.uniform(28, 42)
        avg_stops  = rng.uniform(2.0, 3.5)
        throughput = int(rng.integers(1050, 1250))
        improv     = round((55.0 - avg_delay) / 55.0 * 100, 1) if baseline_delay is None else round((baseline_delay - avg_delay) / baseline_delay * 100, 1)
    else:  # rl
        avg_delay  = rng.uniform(22, 35)
        avg_stops  = rng.uniform(1.5, 2.8)
        throughput = int(rng.integers(1100, 1350))
        bl = baseline_delay or 55.0
        improv     = round((bl - avg_delay) / bl * 100, 1)

    # Synthetic GPS probe data
    n = 500
    xs = rng.uniform(-200, 1000, n)
    ys = rng.uniform(-200,  200, n)
    gps_df = pd.DataFrame(
        {
            "time":              rng.integers(0, 1800, n),
            "vehicle_id":        [f"v{i}" for i in range(n)],
            "x":                 xs,
            "y":                 ys,
            "speed":             rng.uniform(0, 14, n),
            "vehicle_type":      rng.choice(
                                     ["motorcycle", "car", "auto", "truck"],
                                     size=n, p=[0.6, 0.2, 0.1, 0.1]
                                 ),
            "junction_proximity": [
                min(
                    ["J0", "J1", "J2"],
                    key=lambda j: abs(x - {"J0": 0, "J1": 400, "J2": 800}[j]),
                )
                for x in xs
            ],
        }
    )

    metrics = {
        "mode":        mode,
        "avg_delay_s": round(float(avg_delay),  2),
        "avg_stops":   round(float(avg_stops),  2),
        "throughput":  throughput,
        "improvement": improv,
    }
    return SimResult(mode=mode, metrics=metrics, gps_df=gps_df)
