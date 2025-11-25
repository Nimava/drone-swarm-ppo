import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Parameters
num_drones = 5
dt = 0.1
k_c = 0.1  # Cohesion coefficient
k_s = 0.1  # Separation and obstacle avoidance coefficient
k_a = 0.1  # Alignment coefficient
k_l = 0.5  # Leader-following coefficient
leader_k_s = 0.1  # Reduce leader's obstacle avoidance strength
    
#obstacle_radius = 0.5  # Smaller obstacle size
interaction_range = 5.0
max_speed = 5.0
steps = 1000

# Initialize positions and velocities
leader_pos = np.array([0.0, 5.0])  # Start from the left side of the figure (middle height)
positions = leader_pos + np.random.rand(num_drones, 2) * 2 - 1  # Initialize drones close to the leader
velocities = (np.random.rand(num_drones, 2) - 0.5) * 0.5  # Small initial random velocities
leader_velocity_function = lambda t: np.array([0.1, 0.0]) 

# Fixed circular obstacles 
circle_obstacles = [
    {"center": np.array([1, 1]), "radius": 0.4},
    {"center": np.array([3, 7]), "radius": 0.5},
    {"center": np.array([5, 5]), "radius": 0.5},
    {"center": np.array([7, 7]), "radius": 0.3},
    {"center": np.array([9, 9]), "radius": 0.4},
    {"center": np.array([4, 3]), "radius": 0.3},
    {"center": np.array([8, 5]), "radius": 0.2},
    {"center": np.array([2, 6]), "radius": 0.2},
    {"center": np.array([1, 8]), "radius": 0.4},
    {"center": np.array([3, 9]), "radius": 0.3},
    {"center": np.array([5, 1]), "radius": 0.4},
    {"center": np.array([9, 4]), "radius": 0.4},
    {"center": np.array([2, 2]), "radius": 0.4},
    {"center": np.array([3, 4]), "radius": 0.2},
    {"center": np.array([6, 6]), "radius": 0.4},
    {"center": np.array([8, 8]), "radius": 0.4},
    {"center": np.array([5, 4]), "radius": 0.2},
    {"center": np.array([9, 9]), "radius": 0.4},
    {"center": np.array([6, 4]), "radius": 0.3},
    {"center": np.array([2, 4.75]), "radius": 0.3},
    {"center": np.array([9, 3]), "radius": 0.4},
    {"center": np.array([5, 7]), "radius": 0.2},
    {"center": np.array([8, 2]), "radius": 0.3},
]
# To store positions for path visualization
positions_history = [positions.copy()]
leader_history = [leader_pos.copy()]

# Define force functions
def cohesion_force(i, positions):
    neighbors = [j for j in range(num_drones) if j != i and np.linalg.norm(positions[j] - positions[i]) < interaction_range]
    if not neighbors:
        return np.zeros(2)
    avg_pos = np.mean(positions[neighbors], axis=0)
    return k_c * (avg_pos - positions[i])

def separation_force(i, positions):
    force = np.zeros(2)
    for j in range(num_drones):
        if j != i:
            diff = positions[i] - positions[j]
            distance = np.linalg.norm(diff)
            if 0 < distance < 1.0:
                force += k_s * diff / (distance**3 + 1e-6)
    return force

def alignment_force(i, velocities):
    neighbors = [j for j in range(num_drones) if j != i and np.linalg.norm(positions[j] - positions[i]) < interaction_range]
    if not neighbors:
        return np.zeros(2)
    avg_vel = np.mean(velocities[neighbors], axis=0)
    return k_a * (avg_vel - velocities[i])

def obstacle_avoidance_force(i, positions):
    force = np.zeros(2)
    for circle in circle_obstacles:
        diff = positions[i] - circle["center"]
        distance = np.linalg.norm(diff)
        if 0 < distance < circle["radius"]+ 0.15:
            force += k_s * diff / (distance**3 + 1e-6)
    return force

def leader_following_force(i, positions, velocities, leader_pos, leader_vel):
    pos_force = k_l * (leader_pos - positions[i])
    vel_force = k_l * (leader_vel - velocities[i])
    return pos_force + vel_force

def leader_obstacle_avoidance_force(leader_pos):
    force = np.zeros(2)
    for circle in circle_obstacles:
        diff = leader_pos - circle["center"]
        distance = np.linalg.norm(diff)
        if distance < circle["radius"] + 0.15:  # Consider an extra buffer zone
            force += leader_k_s * diff / (distance**3 + 1e-6)  # Reduce power from `distance^3` to `distance^2`
    return force * 1.0    # Further dampen the effect

# Simulation
for step in range(steps):
    new_positions = np.zeros_like(positions)
    new_velocities = np.zeros_like(velocities)
    
    # Compute leader velocity from its velocity function
    leader_vel = leader_velocity_function(step * dt)
    f_leader_obstacle = leader_obstacle_avoidance_force(leader_pos)
    leader_pos += (leader_vel + f_leader_obstacle) * dt

    for i in range(num_drones):
        f_cohesion = cohesion_force(i, positions)
        f_separation = separation_force(i, positions)
        f_alignment = alignment_force(i, velocities)
        f_obstacle = obstacle_avoidance_force(i, positions)
        f_leader = leader_following_force(i, positions, velocities, leader_pos, leader_vel)
        
        total_force = f_cohesion + f_separation + f_alignment + f_obstacle + f_leader
        new_velocities[i] = velocities[i] + total_force * dt
        speed = np.linalg.norm(new_velocities[i])
        if speed > max_speed:
            new_velocities[i] = new_velocities[i] / speed * max_speed
        new_positions[i] = positions[i] + new_velocities[i] * dt
    
    positions = new_positions
    velocities = new_velocities
    
    positions_history.append(positions.copy())
    leader_history.append(leader_pos.copy())
    if step % 200 == 0:  # Save every 200 steps
        plt.figure(figsize=(8, 8))
        plt.xlim(0, 11)  # Adjust based on your setup
        plt.ylim(0, 11)
    
        # Plot drones
        plt.scatter(positions[:, 0], positions[:, 1], c='blue', label='Drones')
        plt.scatter(leader_pos[0], leader_pos[1], c='red', label='Leader')
    
        # Plot circular obstacles
        for circle in circle_obstacles:
            circle_plot = plt.Circle(circle["center"], circle["radius"], color='green', alpha=0.5)
            plt.gca().add_artist(circle_plot)
    
        plt.legend()
        plt.title(f'Drone Swarm at Step {step}')
    
        # Save the figure
        plt.savefig(f"drone_swarm_step_{step}.jpg", dpi=300)
        plt.close()  # Close the figure to avoid excessive memory usage


# Convert positions history to arrays for plotting
positions_history = np.array(positions_history)
leader_history = np.array(leader_history)

# Path plot
plt.figure(figsize=(10, 8))
for i in range(num_drones):
    plt.plot(positions_history[:, i, 0], positions_history[:, i, 1], label=f'Drone {i+1}')
plt.plot(leader_history[:, 0], leader_history[:, 1], 'r-', linewidth=2, label='Leader')

# Plot circular obstacles explicitly
for circle in circle_obstacles:
    circle_plot = plt.Circle(circle["center"], circle["radius"], color='green', alpha=0.5)
    plt.gca().add_artist(circle_plot)

plt.xlim(0, 11)  
plt.ylim(0, 11)  
plt.title('Drone and Leader Path in a Dense Jungle')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()

# Animation
fig, ax = plt.subplots(figsize=(8, 8))
scat_drones = ax.scatter([], [], c='blue', label='Drones')
scat_leader = ax.scatter([], [], c='red', label='Leader')

# Plot circular obstacles
for circle in circle_obstacles:
    circle_plot = plt.Circle(circle["center"], circle["radius"], color='green', alpha=0.5)
    ax.add_artist(circle_plot)

x_min, x_max = 0, 11
y_min, y_max = 0, 11
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
plt.legend()
plt.title('Drone Swarm Animation in a Dense Jungle')

def update(frame):
    scat_drones.set_offsets(positions_history[frame])
    scat_leader.set_offsets(leader_history[frame])
    
    # Add step text annotation
    ax.set_title(f'Drone Swarm Animation in a Dense Jungle\nStep: {frame}')
    
    return scat_drones, scat_leader


ani = FuncAnimation(fig, update, frames=steps, interval=50, blit=True)

# Save the animation as a GIF
output_path = 'drone_swarm_jungle.gif'
ani.save(output_path, writer=PillowWriter(fps=20))
plt.show()

print(f"Animation saved as {output_path}")
