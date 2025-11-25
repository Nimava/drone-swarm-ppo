PPO-Based Optimization of Drone Swarm Dynamics

Authors: Tzyy-Leng Horng, Nima Vaziri
Repository: https://github.com/Nimava/drone-swarm-ppo

🌟 Overview

This repository contains the implementation and simulation code for the research work:

“PPO-Based Optimization of Swarm Dynamics for Leader–Follower Drone Navigation and Collision Avoidance.”

The project integrates classical swarm dynamics (cohesion–alignment–separation) with Proximal Policy Optimization (PPO) to automatically tune swarm behavior in:

nozzle-shaped tunnels

cluttered jungle environments

moving-obstacle fields

The goal is to improve:

collision avoidance

formation stability

leader–follower performance

without manually tuning swarm parameters.

📁 Repository Structure (drone-swarm-ppo)
.
├── src/                          # All source code (simulation + PPO training)
│   ├── drone_swarm_env.py        # Gym-like environment for PPO
│   ├── train_ppo.py              # PPO training script
│   ├── run_trained_policy.py     # Run the trained PPO policy
│   ├── simulate_jungle_case.py   # Simulation script for the "jungle" scenario
│   ├── case01_nozzle_simulation.py
│   ├── case02_straight_path.py
│   ├── case03_dynamic_obstacles.py
│   ├── ... (other case files)
│
├── docs/                         # Figures, GIFs, documentation images
│   ├── nozzle_2d.png
│   ├── nozzle_3d.png
│   ├── trajectories.gif
│   ├── ...
│
├── config.json                   # Configuration file for simulations (user-editable)
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md                     # Project documentation

🔧 Installation

Install dependencies:

pip install -r requirements.txt

🚀 Running Simulations
1. Simulations
python src/2d_nozzle.py
python src/3d_nozzle.py
python src/jungle.py
python src/moving_obstacle.py

2. Train PPO
python src/train_ppo.py

3. Evaluate trained PPO policy
python src/run_trained_policy.py

4. Run the optimized jungle-case
python src/simulate_jungle_case.py

🧠 Summary of PPO Approach

The PPO agent learns optimal swarm coefficients:

kc – cohesion

ka – alignment

ks – separation / obstacle avoidance

Actions:

[kc, ka, ks]


State includes:

average inter-drone distance

velocity alignment score

leader–drone lag

Reward penalizes collisions and instability, and rewards cohesion, alignment, and leader-following.

📊 Results

Example outputs (in docs/):

2D + 3D nozzle geometry

PPO vs baseline trajectories

GIF animations of swarm movement

These demonstrate smoother trajectories and fewer collisions with PPO.

📜 License

This repository is released under the MIT License.

📄 CITATION

Because the paper is not yet published, do not include a CITATION.cff.
We will add it after acceptance.
