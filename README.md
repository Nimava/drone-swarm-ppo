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

📁 Repository Structure
drone-swarm-ppo/
│
├── src/
│   ├── 2d_nozzle.py                   # 2D nozzle simulation
│   ├── 3d_nozzle.py                   # 3D nozzle simulation
│   ├── jungle.py                      # Jungle simulation
│   ├── moving_obstacle.py             # Moving obstacles simulation
│   ├── drone_swarm_env.py             # RL environment (state, action, rewards)
│   ├── train_ppo.py                   # PPO training script
│   ├── run_trained_policy.py          # Evaluate trained PPO policy
│   ├── simulate_jungle_case.py        # PPO-based jungle-case simulation
│
├── docs/
│   ├── 2d_nozzle_path.png                  
│   ├── 3d_nozzle_path.png                  
│   ├── jungle_path.png
│   ├── moving_obstacles_path.png
│   ├── 2d_nozzle_animation.gif
│   ├── 3d_nozzle_animation.gif
│   ├── jungle_animation.gif
│   └── README_figures.md
│
├── config.json                        # Global configuration (parameters)
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore
└── README.md

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