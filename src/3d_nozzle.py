import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ===========================
# Parameters
# ===========================
num_drones = 10
dt = 0.1
k_c = 0.1     # Cohesion coefficient
k_s = 0.1     # Separation and obstacle avoidance coefficient
k_a = 0.1     # Alignment coefficient
k_l = 0.5     # Leader-following coefficient
interaction_range = 5.0
obstacle_radius = 1.0
max_speed = 2.0
steps = 500

# ===========================
# Initialize positions and velocities (3D)
# ===========================
leader_pos = np.array([5.0, 5.0, 5.0])  # Leader initial position (center of tunnel)
leader_velocity_function = lambda t: np.array([1.0, 0.0, 0.0])  # Leader moves along +X direction
positions = leader_pos + np.random.rand(num_drones, 3) * 2 - 1  # Drones near leader
velocities = (np.random.rand(num_drones, 3) - 0.5) * 0.5        # Small initial velocities

# ===========================
# Define 3D nozzle (channel) walls
# ===========================
def channel_walls(x):
    if x < 10:
        lower_y, upper_y = 2, 8
        lower_z, upper_z = 2, 8
    elif 10 <= x <= 20:
        lower_y = 2 + (x - 10) * 0.25
        upper_y = 8 - (x - 10) * 0.25
        lower_z = 2 + (x - 10) * 0.25
        upper_z = 8 - (x - 10) * 0.25
#        upper_z = 8
#        lower_z = 2
    elif 20 < x <= 30:
        lower_y, upper_y = 4.5, 5.5   
        lower_z, upper_z = 4.5, 5.5  
#        lower_z, upper_z = 2, 8    
    elif 30 < x <= 40:
        lower_y = 4.5 - (x - 30) * 0.25
        upper_y = 5.5 + (x - 30) * 0.25
        lower_z = 4.5 - (x - 30) * 0.25
        upper_z = 5.5 + (x - 30) * 0.25
#        upper_z = 8
#        lower_z = 2
    else:
        lower_y, upper_y = 2, 8
        lower_z, upper_z = 2, 8
    return lower_y, upper_y, lower_z, upper_z

# ===========================
# Simulation: using your original force functions extended to 3D
# ===========================
# To store positions for path visualization
positions_history = [positions.copy()]
leader_history = [leader_pos.copy()]

def obstacle_avoidance_force(i, positions):
    """3D obstacle (nozzle wall) avoidance based on channel boundaries."""
    x, y, z = positions[i]
    lower_y, upper_y, lower_z, upper_z = channel_walls(x)
    force = np.zeros(3)
    if y < lower_y + obstacle_radius:
        force[1] += k_s / ((y - lower_y)**2 + 1e-6)
    if y > upper_y - obstacle_radius:
        force[1] -= k_s / ((upper_y - y)**2 + 1e-6)
    if z < lower_z + obstacle_radius:
        force[2] += k_s / ((z - lower_z)**2 + 1e-6)
    if z > upper_z - obstacle_radius:
        force[2] -= k_s / ((upper_z - z)**2 + 1e-6)
    return force

def cohesion_force(i, positions):
    neighbors = [j for j in range(num_drones) if j != i and np.linalg.norm(positions[j]-positions[i]) < interaction_range]
    if not neighbors:
        return np.zeros(3)
    avg_pos = np.mean(positions[neighbors], axis=0)
    return k_c * (avg_pos - positions[i])

def separation_force(i, positions, leader_pos):
    force = np.zeros(3)
    for j in range(num_drones):
        if j != i:
            diff = positions[i] - positions[j]
            distance = np.linalg.norm(diff)
            if 0 < distance < 1.0:
                force += k_s * diff / (distance**3 + 1e-6)
    # Also avoid the leader
    diff_leader = positions[i] - leader_pos
    distance_leader = np.linalg.norm(diff_leader)
    if 0 < distance_leader < 1.0:
        force += k_s * diff_leader / (distance_leader**3 + 1e-6)
    return force

def alignment_force(i, velocities):
    neighbors = [j for j in range(num_drones) if j != i and np.linalg.norm(positions[j]-positions[i]) < interaction_range]
    if not neighbors:
        return np.zeros(3)
    avg_vel = np.mean(velocities[neighbors], axis=0)
    return k_a * (avg_vel - velocities[i])

def leader_following_force(i, positions, velocities, leader_pos, leader_vel):
    pos_force = k_l * (leader_pos - positions[i])
    vel_force = k_l * (leader_vel - velocities[i])
    return pos_force + vel_force

# Simulation loop
for step in range(steps):
    new_positions = np.zeros_like(positions)
    new_velocities = np.zeros_like(velocities)
    
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
# Save snapshot every 100 steps
    if step % 100 == 0:
        fig_snapshot = plt.figure(figsize=(8, 8))
        ax_snapshot = fig_snapshot.add_subplot(111, projection='3d')

        ax_snapshot.set_xlim(0, 50)
        ax_snapshot.set_ylim(0, 10)
        ax_snapshot.set_zlim(0, 10)
        ax_snapshot.set_xlabel("X")
        ax_snapshot.set_ylabel("Y")
        ax_snapshot.set_zlabel("Z")
        ax_snapshot.set_title(f'Drone Swarm at Step {step}')

        # Plot drones and leader
        ax_snapshot.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='blue', label='Drones')
        ax_snapshot.scatter(leader_pos[0], leader_pos[1], leader_pos[2], c='red', label='Leader')

        # Plot nozzle walls as surfaces
        x_vals = np.linspace(0, 50, 50)
        X_surf, Z_surf = np.meshgrid(x_vals, np.linspace(0, 10, 10))

        Y_left_surface = np.tile(np.array([channel_walls(x)[0] for x in x_vals]), (10, 1))
        Y_right_surface = np.tile(np.array([channel_walls(x)[1] for x in x_vals]), (10, 1))

        Z_floor_surface = np.tile(np.array([channel_walls(x)[2] for x in x_vals]), (10, 1))
        Z_ceiling_surface = np.tile(np.array([channel_walls(x)[3] for x in x_vals]), (10, 1))

        # Meshgrid creation for the floor and ceiling
        X_floor, Y_floor_surf = np.meshgrid(x_vals, np.linspace(np.min(Y_left_surface), np.max(Y_right_surface), 10))
        X_ceiling, Y_ceiling_surf = np.meshgrid(x_vals, np.linspace(np.min(Y_left_surface), np.max(Y_right_surface), 10))

        # Add nozzle walls
        ax_snapshot.plot_surface(X_surf, Y_left_surface, Z_surf, color="k", alpha=0.2)
        ax_snapshot.plot_surface(X_surf, Y_right_surface, Z_surf, color="k", alpha=0.2)
        ax_snapshot.plot_surface(X_floor, Y_floor_surf, Z_floor_surface, color="k", alpha=0.2)
        ax_snapshot.plot_surface(X_ceiling, Y_ceiling_surf, Z_ceiling_surface, color="k", alpha=0.2)

        # Save the figure
        plt.savefig(f"drone_nozzle_step_{step}.jpg", dpi=300)
        plt.close()  # Close figure to save memory

positions_history = np.array(positions_history)
leader_history = np.array(leader_history)



# ===========================
# 3D Path Plot (static)
# ===========================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for i in range(num_drones):
    ax.plot(positions_history[:, i, 0], positions_history[:, i, 1], positions_history[:, i, 2], label=f'Drone {i+1}')
ax.plot(leader_history[:, 0], leader_history[:, 1], leader_history[:, 2], 'r-', linewidth=2, label='Leader')

# Plot nozzle walls as surfaces
x_vals = np.linspace(0, 50, 50)
# For left wall surface:
X, Z = np.meshgrid(x_vals, np.linspace(0, 10, 10))
Y_left = np.array([channel_walls(x)[0] for x in x_vals])
Y_left_surface = np.tile(Y_left, (10,1))
ax.plot_surface(X, Y_left_surface, Z, color="k", alpha=0.2)

# For right wall surface:
Y_right = np.array([channel_walls(x)[1] for x in x_vals])
Y_right_surface = np.tile(Y_right, (10,1))
ax.plot_surface(X, Y_right_surface, Z, color="k", alpha=0.2)

# For floor surface:
X_floor, Y_floor = np.meshgrid(x_vals, np.linspace(np.min(Y_left), np.max(Y_right), 10))
Z_floor = np.array([channel_walls(x)[2] for x in x_vals])
Z_floor_surface = np.tile(Z_floor, (10,1))
ax.plot_surface(X_floor, Y_floor, Z_floor_surface, color="k", alpha=0.2)

# For ceiling surface:
Z_ceiling = np.array([channel_walls(x)[3] for x in x_vals])
Z_ceiling_surface = np.tile(Z_ceiling, (10,1))
ax.plot_surface(X_floor, Y_floor, Z_ceiling_surface, color="k", alpha=0.2)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Drone and Leader Path in Nozzle")
ax.legend()
plt.show()

# Plot 2D projections
def save_2d_projection(view, filename, xlabel, ylabel, x_idx, y_idx):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot drone paths
    for i in range(num_drones):
        ax.plot(positions_history[:, i, x_idx], positions_history[:, i, y_idx], 
                label=f'Drone {i+1}', linewidth=1.5, alpha=0.7)
    
    # Plot leader path
    ax.plot(leader_history[:, x_idx], leader_history[:, y_idx], 
            color="red", linewidth=2.5, alpha=1, label='Leader')

    # Add nozzle walls
    x_vals = np.linspace(0, 50, 100)
    if view == "XY":
        lower_walls, upper_walls, _, _ = zip(*[channel_walls(x) for x in x_vals])
        ax.plot(x_vals, lower_walls, 'k--', linewidth=1.5, alpha=0.7, label='Lower Nozzle Wall')
        ax.plot(x_vals, upper_walls, 'k--', linewidth=1.5, alpha=0.7, label='Upper Nozzle Wall')

    elif view == "XZ":
        _, _, lower_walls, upper_walls = zip(*[channel_walls(x) for x in x_vals])
        ax.plot(x_vals, lower_walls, 'k--', linewidth=1.5, alpha=0.7, label='Lower Nozzle Wall')
        ax.plot(x_vals, upper_walls, 'k--', linewidth=1.5, alpha=0.7, label='Upper Nozzle Wall')

    elif view == "YZ":
        # Nozzle walls don't depend on X in YZ view, so we assume max/min limits
        ax.axhline(y=2, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label='Lower Nozzle Wall')
        ax.axhline(y=8, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label='Upper Nozzle Wall')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Drone and Leader Path ({view} Plane)")
    ax.legend()
    ax.grid(True)

    # Save the figure
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    plt.show()

# Save 3D path projections
save_2d_projection("XY", "drone_path_xy.png", "X", "Y", 0, 1)  # XY Plane
save_2d_projection("XZ", "drone_path_xz.png", "X", "Z", 0, 2)  # XZ Plane
save_2d_projection("YZ", "drone_path_yz.png", "Y", "Z", 1, 2)  # YZ Plane

# ===========================
# 3D Animation 
# ===========================
fig_anim = plt.figure(figsize=(10, 8))
ax_anim = fig_anim.add_subplot(111, projection='3d')
ax_anim.set_xlim(0, 50)
ax_anim.set_ylim(0, 10)
ax_anim.set_zlim(0, 10)
ax_anim.set_xlabel("X")
ax_anim.set_ylabel("Y")
ax_anim.set_zlabel("Z")
ax_anim.set_title("Drone Swarm Animation in Nozzle")

# Define the nozzle surfaces 
X_surf, Z_surf = np.meshgrid(x_vals, np.linspace(0, 10, 10))

Y_left_surface = np.tile(np.array([channel_walls(x)[0] for x in x_vals]), (10, 1))
Y_right_surface = np.tile(np.array([channel_walls(x)[1] for x in x_vals]), (10, 1))

Z_floor_surface = np.tile(np.array([channel_walls(x)[2] for x in x_vals]), (10, 1))
Z_ceiling_surface = np.tile(np.array([channel_walls(x)[3] for x in x_vals]), (10, 1))

# Meshgrid creation for the floor and ceiling
X_floor, Y_floor_surf = np.meshgrid(x_vals, np.linspace(np.min(Y_left_surface), np.max(Y_right_surface), 10))
X_ceiling, Y_ceiling_surf = np.meshgrid(x_vals, np.linspace(np.min(Y_left_surface), np.max(Y_right_surface), 10))

# Plot nozzle walls in animation as surfaces
ax_anim.plot_surface(X_surf, Y_left_surface, Z_surf, color="k", alpha=0.2)
ax_anim.plot_surface(X_surf, Y_right_surface, Z_surf, color="k", alpha=0.2)

ax_anim.plot_surface(X_floor, Y_floor_surf, Z_floor_surface, color="k", alpha=0.2)
ax_anim.plot_surface(X_ceiling, Y_ceiling_surf, Z_ceiling_surface, color="k", alpha=0.2)

scat_drones = ax_anim.scatter([], [], [], c='blue', label='Drones')
scat_leader = ax_anim.scatter([], [], [], c='red', label='Leader')
ax_anim.legend()

def update_anim(frame):
    ax_anim.clear()  # Clear previous frame

    # Reset labels and limits after clearing
    ax_anim.set_xlim(0, 50)
    ax_anim.set_ylim(0, 10)
    ax_anim.set_zlim(0, 10)
    ax_anim.set_xlabel("X")
    ax_anim.set_ylabel("Y")
    ax_anim.set_zlabel("Z")
    ax_anim.set_title("Drone Swarm Animation in Nozzle")

    # Re-plot nozzle surfaces (this keeps them visible)
    ax_anim.plot_surface(X_surf, Y_left_surface, Z_surf, color="k", alpha=0.2)
    ax_anim.plot_surface(X_surf, Y_right_surface, Z_surf, color="k", alpha=0.2)
    ax_anim.plot_surface(X_floor, Y_floor_surf, Z_floor_surface, color="k", alpha=0.2)
    ax_anim.plot_surface(X_ceiling, Y_ceiling_surf, Z_ceiling_surface, color="k", alpha=0.2)

    # Update drone and leader positions
    ax_anim.scatter(positions_history[frame][:,0], positions_history[frame][:,1], positions_history[frame][:,2], c='blue', label='Drones')
    ax_anim.scatter(leader_history[frame,0], leader_history[frame,1], leader_history[frame,2], c='red', label='Leader')

    # **Display Time/Step in Animation**
    ax_anim.text2D(0.05, 0.95, f"Time Step: {frame}", transform=ax_anim.transAxes, fontsize=12, color='black')

    return ax_anim


ani = FuncAnimation(fig_anim, update_anim, frames=steps, interval=50, blit=False)
output_path = 'drone_nozzle_3D.gif'
ani.save(output_path, writer=PillowWriter(fps=20))
plt.show()

print(f"Animation saved as {output_path}")
