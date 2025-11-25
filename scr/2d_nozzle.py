import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Parameters
num_drones = 10
dt = 0.1
k_c = 0.1  # Cohesion coefficient
k_s = 0.1  # Separation and obstacle avoidance coefficients
k_a = 0.1  # Alignment coefficient
k_l = 0.4  # Leader-following coefficient
interaction_range = 5.0
obstacle_radius = 1.0
max_speed = 2.0
steps = 500

# Leader initial position and velocity
leader_pos = np.array([5.0, 5.0])
leader_velocity_function = lambda t: np.array([1.0, 0.0])  # Leader moves horizontally
positions = leader_pos + np.random.rand(num_drones, 2) * 2 - 1  # Initialize drones close to the leader
velocities = (np.random.rand(num_drones, 2) - 0.5) * 0.5  # Small initial random velocities


# Symmetric channel walls
def channel_walls(x):
    """Define the channel."""
    if x < 10:  # Straight section
        upper_wall = 8
        lower_wall = 2
    elif 10 <= x <= 20:  # Converging section
        upper_wall = 8 - (x - 10) * 0.25
        lower_wall = 2 + (x - 10) * 0.25
    elif 20 < x <= 30: # Throut section
        upper_wall = 5.5
        lower_wall = 4.5
    elif 30 < x <= 40:  # Diverging section
        upper_wall = 5.5 + (x - 30) * 0.25
        lower_wall = 4.5 - (x - 30) * 0.25 
    else:  # Straight section
        upper_wall = 8
        lower_wall = 2    
    return lower_wall, upper_wall


# To store positions for path visualization
positions_history = [positions.copy()]
leader_history = [leader_pos.copy()]

# Define force functions
def obstacle_avoidance_force(i, positions):
    """Avoid the walls of the channel."""
    x, y = positions[i]
    lower_wall, upper_wall = channel_walls(x)
    force = np.zeros(2)
    if y < lower_wall + obstacle_radius:
        force[1] += k_s / ((y - lower_wall)**2 + 1e-6)
    if y > upper_wall - obstacle_radius:
        force[1] -= k_s / ((upper_wall - y)**2 + 1e-6)
    return force

def cohesion_force(i, positions):
    neighbors = [j for j in range(num_drones) if j != i and np.linalg.norm(positions[j] - positions[i]) < interaction_range]
    if not neighbors:
        return np.zeros(2)
    avg_pos = np.mean(positions[neighbors], axis=0)
    return k_c * (avg_pos - positions[i])

def separation_force(i, positions, leader_pos):
    force = np.zeros(2)
    for j in range(num_drones):
        if j != i:
            diff = positions[i] - positions[j]
            distance = np.linalg.norm(diff)
            if 0 < distance < 1.0:
                force += k_s * diff / (distance**3 + 1e-6)

    # Add avoidance force from the leader
    diff_leader = positions[i] - leader_pos
    distance_leader = np.linalg.norm(diff_leader)
    if 0 < distance_leader < 1.0:
        force += k_s * diff_leader / (distance_leader**3 + 1e-6)
    
    return force
    
def alignment_force(i, velocities):
    neighbors = [j for j in range(num_drones) if j != i and np.linalg.norm(positions[j] - positions[i]) < interaction_range]
    if not neighbors:
        return np.zeros(2)
    avg_vel = np.mean(velocities[neighbors], axis=0)
    return k_a * (avg_vel - velocities[i])

def leader_following_force(i, positions, velocities, leader_pos, leader_vel):
    pos_force = k_l * (leader_pos - positions[i])
    vel_force = k_l * (leader_vel - velocities[i])
    return pos_force + vel_force

# Simulation
for step in range(steps):
    new_positions = np.zeros_like(positions)
    new_velocities = np.zeros_like(velocities)
    
    # Compute leader velocity from its velocity function
    leader_vel = leader_velocity_function(step * dt)
    leader_pos += leader_vel * dt

    for i in range(num_drones):
        f_cohesion = cohesion_force(i, positions)
        f_separation = separation_force(i, positions, leader_pos)
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

# Convert positions history to arrays for plotting
positions_history = np.array(positions_history)
leader_history = np.array(leader_history)

# Path plot
plt.figure(figsize=(10, 8))
for i in range(num_drones):
    plt.plot(positions_history[:, i, 0], positions_history[:, i, 1], label=f'Drone {i+1}')
plt.plot(leader_history[:, 0], leader_history[:, 1], 'r-', linewidth=2, label='Leader')

# Plot channel walls
x_vals = np.linspace(0, 50, 500)
lower_walls, upper_walls = zip(*[channel_walls(x) for x in x_vals])
plt.plot(x_vals, lower_walls, 'k--', label='Channel Lower Wall')
plt.plot(x_vals, upper_walls, 'k--', label='Channel Upper Wall')


plt.title('Drone and Leader Path in Converge-Diverge Channel')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()

# Animation
fig, ax = plt.subplots(figsize=(8, 8))
scat_drones = ax.scatter([], [], c='blue', label='Drones')
scat_leader = ax.scatter([], [], c='red', label='Leader')
x_min, x_max = 0, 50
y_min, y_max = 0, 10
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
plt.legend()
plt.title('Drone Swarm Animation in Converge-Diverge Channel')

# Initialize wall lines
line_lower, = ax.plot([], [], 'k--', label='Channel Lower Wall')
line_upper, = ax.plot([], [], 'k--', label='Channel Upper Wall')

# Update function for animation
def update(frame):
    # Update drones and leader positions
    scat_drones.set_offsets(positions_history[frame])
    scat_leader.set_offsets(leader_history[frame])

    # Update walls
    x_vals = np.linspace(0, 50, 500)  # Adjust range for diverging section
    lower_walls, upper_walls = zip(*[channel_walls(x) for x in x_vals])
    line_lower.set_data(x_vals, lower_walls)
    line_upper.set_data(x_vals, upper_walls)

    return scat_drones, scat_leader, line_lower, line_upper


ani = FuncAnimation(fig, update, frames=steps, interval=50, blit=True)

# Save the animation as a GIF
output_path = 'drone_channel_animation.gif'
ani.save(output_path, writer=PillowWriter(fps=20))
plt.show()

print(f"Animation saved as {output_path}")
