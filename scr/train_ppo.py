
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from drone_swarm_env import DroneSwarmEnv

# Instantiate and validate the environment
env = DroneSwarmEnv()
check_env(env, warn=True)

# Create the PPO agent
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_drone_tensorboard/")

# Train the agent
model.learn(total_timesteps=10000)

# Save the trained model
model.save("ppo_drone_swarm")
