# Authors: Tzyy-Leng Horng, Nima Vaziri 
import numpy as np
from stable_baselines3 import PPO
from drone_swarm_env import DroneSwarmEnv
from simulate_jungle_case import simulate_jungle_case  # <- use the jungle version

# Load the trained PPO model
model = PPO.load("ppo_drone_swarm")

# Create the environment to get observation space
env = DroneSwarmEnv()
obs = env.reset()[0]  # Gymnasium format: returns (obs, info)

# Get the optimized parameters
action, _ = model.predict(obs, deterministic=True)
action = np.array(action).flatten()
k_c, k_a, k_s = action[:3]

print(f"Optimized Parameters:\nCohesion = {k_c:.3f}, Alignment = {k_a:.3f}, Separation = {k_s:.3f}")

# Run simulation with those parameters
Amin, Bmin, Cmax, *_ = simulate_jungle_case(
    k_c=k_c,
    k_s=k_s,
    k_a=k_a,
    k_l=0.5  # fixed
)

print("\n=== Evaluation Metrics ===")
print(f"Amin (drone-drone):   {min(Amin):.2f} m")
print(f"Bmin (drone-obstacle): {min(Bmin):.2f} m")
print(f"Cmax (lag from leader): {max(Cmax):.2f} m")
