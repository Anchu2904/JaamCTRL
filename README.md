# JaamCTRL

> **AI Adaptive Traffic Signal Optimizer** for Indian Urban Corridors
> KodeMaster.ai Hackathon 2026 – Open Innovation Track (Team:BRAT)

---

## The Problem

Indian urban traffic is uniquely chaotic: 60%+ two-wheelers, zero lane discipline,
random pedestrians and stray animals, and fixed-time signals that cannot react to
real conditions. Delhi alone loses an estimated **1.5 billion hours** annually to
congestion. Fixed-cycle signals are the root cause – they do not know what is
happening on the road.

## Our Solution

JaamCTRL simulates a 3-intersection arterial corridor (Cannaught Place, Delhi) with authentic Indian traffic and applies two levels of AI:

| Level | Approach | Improvement |
|---|---|---|
| Rule-based | Queue-aware green extension + green-wave offset | ~20-25% delay reduction |
| PPO RL Agent | Coordinated joint control of all 3 signals | ~28-35% delay reduction |

The system is designed for a clear **"before vs after"** demo with quantifiable metrics.

---

## Demo

> **Live demo video:** *(insert GIF/video link here)*


## Architecture

```
jaamctrl/
├── app.py                    # Streamlit dashboard (entry point)
├── requirements.txt
├── src/
│   ├── run_simulation.py     # Core simulation runner (TraCI loop)
│   ├── signal_controller.py  # Rule-based adaptive controller
│   ├── rl_agent.py           # PPO RL agent (stable-baselines3)
│   ├── gps_generator.py      # Synthetic GPS probe generator
│   └── heatmap.py            # Folium neon heatmap builder
├── sumo/
│   ├── corridor.net.xml      # 3-intersection network (SUMO format)
│   ├── corridor.rou.xml      # Indian traffic mix (60% 2-wheelers)
│   └── corridor.sumocfg      # SUMO config
├── models/                   # Saved PPO models
└── assets/                   # Screenshots / GIFs for README
```

---
## Tech Stack
SUMO + TraCI | stable-baselines3 PPO | Folium neon heatmaps | Streamlit dashboard

## Setup Instructions

### 1. Install SUMO

```bash
# macOS
brew install sumo

# Ubuntu/Debian
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update && sudo apt-get install sumo sumo-tools

# Windows: download installer from https://sumo.dlr.de/docs/Installing/index.html
```

Add SUMO tools to your Python path:
```bash
export SUMO_HOME=/usr/share/sumo          # adjust to your install path
export PYTHONPATH=$SUMO_HOME/tools:$PYTHONPATH
```

### 2. Clone & install Python dependencies

```bash
git clone https://github.com/<your-team>/jaam-ctrl.git
cd jaam-ctrl
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 4. Workflow

1. **Dashboard tab** → Run Fixed Simulation (baseline)
2. **Dashboard tab** → Run Adaptive Simulation (see immediate improvement)
3. **RL Training tab** → Click "Train PPO Agent" (1-3 minutes)
4. **Dashboard tab** → Run RL Simulation (best results)
5. **Heatmap tab** → Compare neon heatmaps side-by-side
6. **What-If tab** → Explore traffic volume and accident scenarios

---

## Key Metrics (expected, may vary by seed)

| Metric | Fixed-Time | Adaptive | RL Agent |
|---|---|---|---|
| Avg Delay (s) | ~55 | ~38 | ~32 |
| Avg Stops | 5.2 | 3.1 | 2.4 |
| Throughput (veh) | ~950 | ~1100 | ~1200 |
| Improvement | — | ~31% | ~42% |

---

## RL Agent Details

- **Algorithm:** PPO (Proximal Policy Optimisation)
- **Observation:** 12-dim vector — queue lengths + phase state for all 3 junctions
- **Action:** Discrete(8) — 3 independent phase-switch bits (one per junction)
- **Reward:** Delay reduction + flow-evenness penalty
- **Training:** 2000–5000 timesteps (~1–3 min on CPU)
- **Inference:** Deterministic policy, runs in real-time during simulation

---


---

## Team

*BRAT – Build with KodeMaster.ai Hackathon 2026*

---

## License
MIT
