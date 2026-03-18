"""
rl_agent.py  -  Jaam Ctrl (जाम Ctrl)
======================================
PPO-based reinforcement learning agent for coordinated 3-intersection
traffic signal control using stable-baselines3 + SUMO/TraCI.

Architecture
------------
Observation space  (18-dim, continuous [0, 1]):
  Per junction × 3:
    [0] queue_ew        – normalised queue on E-W approaches
    [1] queue_ns        – normalised queue on N-S approaches
    [2] phase_ew        – 1.0 if currently EW-green, else 0.0
    [3] phase_ns        – 1.0 if currently NS-green, else 0.0
    [4] time_in_phase   – normalised seconds spent in current phase (0-1)
    [5] throughput_norm – vehicles that passed junction in last 10 s (norm)

Action space  Discrete(8):
  3-bit binary integer. Bit i = 1 requests a phase switch at junction i.
    bit 0 → J0   bit 1 → J1   bit 2 → J2

Reward (composite, per control step):
  R = w1 * delay_reduction
    + w2 * throughput_gain
    - w3 * flow_imbalance_penalty
    - w4 * unnecessary_switch_penalty
    - w5 * long_queue_penalty

  Weights promote even flow across all 3 junctions while
  maximising total throughput and minimising total waiting time.

Training
--------
  Call train_ppo(total_timesteps=3000) – takes ~2 min on CPU.
  Model saved to  models/ppo_jaam_ctrl.zip
  Training curve  models/training_log.json

Usage from Streamlit (app.py)
------------------------------
  from src.rl_agent import train_ppo, load_ppo_model, MODEL_PATH, SB3_AVAILABLE
  model = load_ppo_model()          # load saved model
  obs   = env.reset()
  action, _ = model.predict(obs, deterministic=True)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Callable, Optional

import numpy as np

# ── Gymnasium ──────────────────────────────────────────────────────────────────
import gymnasium as gym
from gymnasium import spaces

# ── stable-baselines3 (optional – only for training) ──────────────────────────
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

# ── TraCI ──────────────────────────────────────────────────────────────────────
try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_HERE)
SUMO_CFG    = os.path.join(_ROOT, "sumo", "config.sumocfg")
MODEL_PATH  = os.path.join(_ROOT, "models", "ppo_jaam_ctrl")
LOG_PATH    = os.path.join(_ROOT, "models", "training_log.json")
MONITOR_DIR = os.path.join(_ROOT, "models", "monitor")

# ── Simulation constants ───────────────────────────────────────────────────────
SIM_DURATION    = 1800    # seconds per episode (30 min)
CONTROL_STEP    = 10      # agent acts every N sim seconds
MIN_PHASE_DUR   = 15      # minimum green seconds before switch allowed
MAX_PHASE_DUR   = 60      # maximum green before forced switch
YELLOW_DUR      = 5       # yellow phase duration (fixed)

# ── Junction / edge mapping ────────────────────────────────────────────────────
# TL IDs match junction IDs in network.net.xml (J0, J1, J2)
TL_IDS = ["J0", "J1", "J2"]

JUNCTION_EDGES: dict[str, dict[str, list[str]]] = {
    "J0": {
        "ew": ["W0J0", "J1J0"],
        "ns": ["N0J0", "S0J0"],
    },
    "J1": {
        "ew": ["J0J1", "J2J1"],
        "ns": ["N1J1", "S1J1"],
    },
    "J2": {
        "ew": ["J1J2"],
        "ns": ["N2J2", "S2J2"],
    },
}

# Phase indices in tlLogic (must match tllogic.tll.xml / build_net.py)
PHASE_EW_GREEN  = 0
PHASE_EW_YELLOW = 1
PHASE_NS_GREEN  = 2
PHASE_NS_YELLOW = 3

# Observation normalisation
MAX_QUEUE       = 25.0    # vehicles
MAX_THROUGHPUT  = 10.0    # vehicles per control step

# Reward weights
W_DELAY_REDUCTION      = 1.0
W_THROUGHPUT_GAIN      = 0.9
W_FLOW_IMBALANCE       = 0.3
W_UNNECESSARY_SWITCH   = 0.1
W_LONG_QUEUE           = 0.4


# ══════════════════════════════════════════════════════════════════════════════
# Gymnasium Environment
# ══════════════════════════════════════════════════════════════════════════════

class CorridorEnv(gym.Env):
    """
    SUMO 3-intersection corridor environment for PPO training.

    One episode = SIM_DURATION simulation seconds.
    The agent is called every CONTROL_STEP seconds.
    """

    metadata = {"render_modes": []}

    def __init__(self, sumo_binary: str = "sumo", seed: int = 42,
                 verbose: bool = False):
        super().__init__()

        self.sumo_binary = sumo_binary
        self._seed       = seed
        self._verbose    = verbose

        # ── Spaces ────────────────────────────────────────────────────────────
        # 18-dim observation: 6 features × 3 junctions
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(18,), dtype=np.float32
        )
        # 8 discrete actions (3-bit binary)
        self.action_space = spaces.Discrete(8)

        # ── Internal state ────────────────────────────────────────────────────
        self._traci_open    = False
        self._phase_timer:  dict[str, int]   = {}  # seconds in current phase
        self._cur_phase:    dict[str, int]   = {}  # current phase index
        self._prev_arrived: int              = 0
        self._prev_delay:   float            = 0.0
        self._switch_count: int              = 0
        self._episode_reward: float          = 0.0
        self._step_count:   int              = 0

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)

        self._close_traci()

        sumo_cmd = [
            self.sumo_binary,
            "-c", SUMO_CFG,
            "--seed", str(self._seed),
            "--no-warnings",
            "--no-step-log",
            "--quit-on-end",
        ]

        try:
            traci.start(sumo_cmd)
        except Exception as e:
            raise RuntimeError(f"Failed to start SUMO: {e}")

        self._traci_open    = True
        self._phase_timer   = {tl: 0   for tl in TL_IDS}
        self._cur_phase     = {tl: PHASE_EW_GREEN for tl in TL_IDS}
        self._prev_arrived  = 0
        self._prev_delay    = 0.0
        self._switch_count  = 0
        self._episode_reward = 0.0
        self._step_count    = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        if not self._traci_open:
            raise RuntimeError("Call reset() before step()")

        # ── Decode 3-bit action ────────────────────────────────────────────────
        requests: dict[str, bool] = {
            "J0": bool(action & 1),
            "J1": bool(action & 2),
            "J2": bool(action & 4),
        }

        # Snapshot before stepping
        delay_before   = self._total_delay()
        arrived_before = traci.simulation.getArrivedNumber()

        # ── Run CONTROL_STEP simulation seconds ────────────────────────────────
        for _ in range(CONTROL_STEP):
            t = traci.simulation.getTime()
            if t >= SIM_DURATION:
                break

            for tl in TL_IDS:
                self._phase_timer[tl] += 1
                self._maybe_switch(tl, requests[tl])

            traci.simulationStep()
            self._step_count += 1

        # ── Snapshot after stepping ────────────────────────────────────────────
        delay_after    = self._total_delay()
        arrived_after  = traci.simulation.getArrivedNumber()
        newly_arrived  = arrived_after  # SUMO resets each step; accumulate
        switches_made  = sum(
            1 for tl in TL_IDS
            if requests[tl] and self._phase_timer[tl] >= MIN_PHASE_DUR
        )

        # ── Reward ────────────────────────────────────────────────────────────
        reward = self._compute_reward(
            delay_before, delay_after,
            newly_arrived, switches_made
        )
        self._episode_reward += reward
        self._switch_count   += switches_made

        # ── Termination ───────────────────────────────────────────────────────
        terminated = traci.simulation.getTime() >= SIM_DURATION
        if terminated:
            if self._verbose:
                print(
                    f"  Episode done | reward={self._episode_reward:.2f} "
                    f"| switches={self._switch_count}"
                )
            self._close_traci()

        obs = self._get_obs() if not terminated else np.zeros(18, dtype=np.float32)
        info: dict = {
            "delay_after":    delay_after,
            "newly_arrived":  newly_arrived,
            "episode_reward": self._episode_reward,
        }
        return obs, reward, terminated, False, info

    def close(self):
        self._close_traci()

    # ── Observation ────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        18-dim observation vector:
          [q_ew, q_ns, phase_ew, phase_ns, time_in_phase, throughput] × 3
        """
        obs: list[float] = []
        for tl in TL_IDS:
            edges   = JUNCTION_EDGES[tl]
            q_ew    = min(self._sum_queue(edges["ew"]) / MAX_QUEUE,    1.0)
            q_ns    = min(self._sum_queue(edges["ns"]) / MAX_QUEUE,    1.0)
            phase   = self._cur_phase[tl]
            ph_ew   = 1.0 if phase == PHASE_EW_GREEN else 0.0
            ph_ns   = 1.0 if phase == PHASE_NS_GREEN else 0.0
            t_norm  = min(self._phase_timer[tl] / MAX_PHASE_DUR, 1.0)
            thru    = min(self._edge_throughput(edges["ew"] + edges["ns"])
                         / MAX_THROUGHPUT, 1.0)
            obs.extend([q_ew, q_ns, ph_ew, ph_ns, t_norm, thru])

        return np.array(obs, dtype=np.float32)

    # ── Reward ─────────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        delay_before:  float,
        delay_after:   float,
        newly_arrived: int,
        switches_made: int,
    ) -> float:
        """
        Composite reward:
          + delay reduction        (main signal: did we cut waiting time?)
          + throughput gain        (did more vehicles complete trips?)
          - flow imbalance         (is one junction way worse than others?)
          - unnecessary switches   (penalise flipping phases too fast)
          - long queue penalty     (any junction with a very long queue)
        """
        # 1. Delay reduction (normalised)
        delay_delta = delay_before - delay_after
        r_delay = np.tanh(delay_delta / max(1.0, delay_before))

        # 2. Throughput gain
        r_throughput = min(newly_arrived / MAX_THROUGHPUT, 1.0)

        # 3. Flow imbalance across junctions
        queues = np.array([
            self._sum_queue(JUNCTION_EDGES[tl]["ew"])
            + self._sum_queue(JUNCTION_EDGES[tl]["ns"])
            for tl in TL_IDS
        ], dtype=np.float32)
        mean_q = np.mean(queues)
        imbalance = float(np.std(queues) / max(1.0, mean_q))

        # 4. Unnecessary switch penalty (switched but phase was too young)
        r_switch_penalty = float(switches_made) * 0.1

        # 5. Long queue penalty (any queue > 70% of MAX_QUEUE)
        long_queue = float(np.sum(queues > 0.7 * MAX_QUEUE))

        reward = (
              W_DELAY_REDUCTION    * r_delay
            + W_THROUGHPUT_GAIN    * r_throughput
            - W_FLOW_IMBALANCE     * imbalance
            - W_UNNECESSARY_SWITCH * r_switch_penalty
            - W_LONG_QUEUE         * long_queue / len(TL_IDS)
        )
        return float(reward)

    # ── Phase control ──────────────────────────────────────────────────────────

    def _maybe_switch(self, tl: str, requested: bool):
        """
        Apply phase switch logic:
          - Force switch if phase exceeded MAX_PHASE_DUR
          - Allow switch if requested and MIN_PHASE_DUR elapsed
          - Never switch during yellow phases
        """
        phase = self._cur_phase[tl]

        # Don't touch yellow phases
        if phase in (PHASE_EW_YELLOW, PHASE_NS_YELLOW):
            return

        timer   = self._phase_timer[tl]
        force   = timer >= MAX_PHASE_DUR
        allowed = requested and timer >= MIN_PHASE_DUR

        if force or allowed:
            self._switch_phase(tl)

    def _switch_phase(self, tl: str):
        """Toggle EW-green ↔ NS-green through yellow."""
        try:
            cur = self._cur_phase[tl]
            if cur == PHASE_EW_GREEN:
                next_phase = PHASE_EW_YELLOW
                after_yellow = PHASE_NS_GREEN
            else:
                next_phase = PHASE_NS_YELLOW
                after_yellow = PHASE_EW_GREEN

            traci.trafficlight.setPhase(tl, next_phase)
            traci.trafficlight.setPhaseDuration(tl, YELLOW_DUR)
            # Schedule the green phase after yellow
            # (SUMO will auto-advance; we track state ourselves)
            self._cur_phase[tl]   = after_yellow
            self._phase_timer[tl] = 0

        except Exception:
            pass

    # ── TraCI helpers ──────────────────────────────────────────────────────────

    def _sum_queue(self, edges: list[str]) -> float:
        """Count stopped vehicles (speed < 0.5 m/s) on given edges."""
        total = 0.0
        for e in edges:
            try:
                vids  = traci.edge.getLastStepVehicleIDs(e)
                total += sum(1 for v in vids if traci.vehicle.getSpeed(v) < 0.5)
            except Exception:
                pass
        return total

    def _edge_throughput(self, edges: list[str]) -> float:
        """Vehicles that left these edges in the last step."""
        total = 0.0
        for e in edges:
            try:
                total += traci.edge.getLastStepVehicleNumber(e)
            except Exception:
                pass
        return total

    def _total_delay(self) -> float:
        """Sum of accumulated waiting time for all active vehicles."""
        try:
            return float(sum(
                traci.vehicle.getAccumulatedWaitingTime(v)
                for v in traci.vehicle.getIDList()
            ))
        except Exception:
            return 0.0

    def _close_traci(self):
        if self._traci_open:
            try:
                traci.close()
            except Exception:
                pass
            self._traci_open = False


# ══════════════════════════════════════════════════════════════════════════════
# SB3 Callback – logs reward/delay per episode, feeds Streamlit progress bar
# ══════════════════════════════════════════════════════════════════════════════

class JaamCallback(BaseCallback):
    """
    Logs per-episode metrics and calls an optional Streamlit progress callback.

    progress_fn(step: int, total: int) -> None
    """

    def __init__(
        self,
        total_timesteps:  int,
        progress_fn:      Optional[Callable[[int, int], None]] = None,
        verbose:          int = 0,
    ):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress_fn     = progress_fn
        self.episode_rewards: list[float] = []
        self.episode_delays:  list[float] = []
        self._ep_reward      = 0.0
        self._n_calls_local  = 0

    def _on_step(self) -> bool:
        self._n_calls_local += 1

        # Collect reward
        rewards = self.locals.get("rewards", [0.0])
        self._ep_reward += float(np.mean(rewards))

        # Check for episode end
        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])
        for done, info in zip(dones, infos):
            if done:
                self.episode_rewards.append(self._ep_reward)
                delay = info.get("delay_after", 0.0)
                self.episode_delays.append(delay)
                if self.verbose >= 1:
                    ep = len(self.episode_rewards)
                    print(
                        f"  Episode {ep:3d} | "
                        f"reward={self._ep_reward:+.3f} | "
                        f"delay={delay:.1f}s"
                    )
                self._ep_reward = 0.0

        # Streamlit progress bar
        if self.progress_fn is not None:
            self.progress_fn(self.num_timesteps, self.total_timesteps)

        return True  # continue training

    def _on_training_end(self):
        """Save training curve to JSON for dashboard display."""
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        log = {
            "episode_rewards": self.episode_rewards,
            "episode_delays":  self.episode_delays,
            "total_episodes":  len(self.episode_rewards),
            "mean_reward":     float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            "best_reward":     float(np.max(self.episode_rewards))  if self.episode_rewards else 0.0,
        }
        with open(LOG_PATH, "w") as f:
            json.dump(log, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def train_ppo(
    total_timesteps: int = 3000,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    learning_rate: float = 3e-4,
    n_steps: int = 256,
    batch_size: int = 64,
    n_epochs: int = 5,
    gamma: float = 0.95,
    ent_coef: float = 0.01,
    verbose_env: bool = False,
) -> str:
    """
    Train a PPO agent on the 3-intersection corridor.

    Parameters
    ----------
    total_timesteps   : Total env steps to train for (2000-5000 recommended).
    progress_callback : Optional fn(step, total) for Streamlit progress bar.
    learning_rate     : PPO learning rate.
    n_steps           : Steps per rollout buffer.
    batch_size        : Mini-batch size for PPO updates.
    n_epochs          : PPO update epochs per rollout.
    gamma             : Discount factor.
    ent_coef          : Entropy coefficient (exploration).
    verbose_env       : Print per-episode stats to stdout.

    Returns
    -------
    str : Path where model was saved (without .zip extension).
    """
    if not SB3_AVAILABLE:
        raise ImportError(
            "stable-baselines3 not installed.\n"
            "Run: pip install stable-baselines3"
        )
    if not TRACI_AVAILABLE:
        raise ImportError(
            "traci not found.\n"
            "Install SUMO and add $SUMO_HOME/tools to PYTHONPATH."
        )

    # Ensure output dirs exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(MONITOR_DIR, exist_ok=True)

    print(f"[JaamCtrl] Training PPO | timesteps={total_timesteps}")
    print(f"[JaamCtrl] SUMO config : {SUMO_CFG}")
    print(f"[JaamCtrl] Model path  : {MODEL_PATH}.zip")

    # Build environment
    env = CorridorEnv(sumo_binary="sumo", verbose=verbose_env)
    env = Monitor(env, filename=os.path.join(MONITOR_DIR, "train"))

    # Build PPO model
    model = PPO(
        policy         = "MlpPolicy",
        env            = env,
        learning_rate  = learning_rate,
        n_steps        = n_steps,
        batch_size     = batch_size,
        n_epochs       = n_epochs,
        gamma          = gamma,
        gae_lambda     = 0.95,
        clip_range     = 0.2,
        ent_coef       = ent_coef,
        vf_coef        = 0.5,
        max_grad_norm  = 0.5,
        verbose        = 0,
        policy_kwargs  = dict(
            net_arch = [dict(pi=[128, 128], vf=[128, 128])]
        ),
    )

    # Build callback
    cb = JaamCallback(
        total_timesteps = total_timesteps,
        progress_fn     = progress_callback,
        verbose         = 1,
    )

    # Train
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=cb)
    elapsed = time.time() - t0

    # Save
    model.save(MODEL_PATH)
    print(f"[JaamCtrl] Saved model to {MODEL_PATH}.zip  ({elapsed:.1f}s)")

    # Final stats
    if cb.episode_rewards:
        print(
            f"[JaamCtrl] Episodes={len(cb.episode_rewards)} | "
            f"mean_reward={np.mean(cb.episode_rewards):.3f} | "
            f"best_reward={np.max(cb.episode_rewards):.3f}"
        )

    env.close()
    return MODEL_PATH


def load_ppo_model() -> Optional["PPO"]:
    """
    Load the most recently saved PPO model.
    Returns None if no model exists or SB3 is not installed.
    """
    if not SB3_AVAILABLE:
        return None
    path = MODEL_PATH + ".zip"
    if not os.path.exists(path):
        return None
    try:
        return PPO.load(MODEL_PATH)
    except Exception as e:
        print(f"[JaamCtrl] Failed to load model: {e}")
        return None


def load_training_log() -> dict:
    """
    Load the training metrics JSON written after training.
    Returns empty dict if not found.
    """
    if not os.path.exists(LOG_PATH):
        return {}
    try:
        with open(LOG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def run_ppo_episode(model: "PPO") -> dict:
    """
    Run one deterministic inference episode with a trained model.

    Returns
    -------
    dict with keys:
      total_reward, steps, final_delay, avg_delay_per_step
    """
    if not TRACI_AVAILABLE:
        raise ImportError("traci not available for RL inference.")

    env      = CorridorEnv(sumo_binary="sumo")
    obs, _   = env.reset()
    total_r  = 0.0
    steps    = 0
    delays   = []
    done     = False

    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(int(action))
        total_r += reward
        steps   += 1
        if "delay_after" in info:
            delays.append(info["delay_after"])

    env.close()

    return {
        "total_reward":        round(total_r, 3),
        "steps":               steps,
        "final_delay":         round(delays[-1], 2) if delays else 0.0,
        "avg_delay_per_step":  round(float(np.mean(delays)), 2) if delays else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI  (python src/rl_agent.py --train --timesteps 3000)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Jaam Ctrl PPO Agent")
    parser.add_argument("--train",      action="store_true", help="Run training")
    parser.add_argument("--infer",      action="store_true", help="Run inference episode")
    parser.add_argument("--check-env",  action="store_true", help="Run SB3 env checker")
    parser.add_argument("--timesteps",  type=int, default=3000)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    if args.check_env:
        if not SB3_AVAILABLE:
            print("stable-baselines3 not installed.")
            sys.exit(1)
        print("Checking environment...")
        env = CorridorEnv(verbose=True)
        check_env(env, warn=True)
        print("Environment check passed.")

    elif args.train:
        saved = train_ppo(
            total_timesteps = args.timesteps,
            learning_rate   = args.lr,
            verbose_env     = args.verbose,
        )
        print(f"Training complete. Model: {saved}.zip")
        log = load_training_log()
        if log:
            print(f"Episodes: {log['total_episodes']}")
            print(f"Mean reward: {log['mean_reward']:.3f}")
            print(f"Best reward: {log['best_reward']:.3f}")

    elif args.infer:
        model = load_ppo_model()
        if model is None:
            print("No trained model found. Run --train first.")
            sys.exit(1)
        print("Running inference episode...")
        result = run_ppo_episode(model)
        print(f"Result: {result}")

    else:
        parser.print_help()
