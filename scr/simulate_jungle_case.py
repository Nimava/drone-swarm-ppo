
import numpy as np
import matplotlib.pyplot as plt

def simulate_jungle_case(k_c=1.0, k_s=7.0, k_a=1.0, k_l=2.5, max_speed=5.0, save_outputs=False):
    num_drones = 5
    dt = 0.1
    steps = 1000
    leader_k_s = 2.0
    interaction_range = 5.0

    # Initial positions and velocities
    leader_pos = np.array([0.0, 5.0])
    positions = leader_pos + np.random.rand(num_drones, 2) * 2 - 1
    velocities = (np.random.rand(num_drones, 2) - 0.5) * 0.5
    leader_velocity_function = lambda t: np.array([0.1, 0.0])

    # Circular obstacles
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

    Amin_t, Bmin_t, Cmax_t = [], [], []
    positions_history = [positions.copy()]
    leader_history = [leader_pos.copy()]

    def cohesion_force(i):
        neighbors = [j for j in range(num_drones) if j != i and np.linalg.norm(positions[j] - positions[i]) < interaction_range]
        if not neighbors: return np.zeros(2)
        avg_pos = np.mean(positions[neighbors], axis=0)
        return k_c * (avg_pos - positions[i])

    def separation_force(i):
        force = np.zeros(2)
        for j in range(num_drones):
            if j != i:
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                if 0 < dist < 1.1: #was 1.0
                    force += k_s * diff / (dist**3 + 1e-6)
        return force

    def alignment_force(i):
        neighbors = [j for j in range(num_drones) if j != i and np.linalg.norm(positions[j] - positions[i]) < interaction_range]
        if not neighbors: return np.zeros(2)
        avg_vel = np.mean(velocities[neighbors], axis=0)
        return k_a * (avg_vel - velocities[i])

    def obstacle_avoidance_force(i):
        force = np.zeros(2)
        for circle in circle_obstacles:
            diff = positions[i] - circle["center"]
            dist = np.linalg.norm(diff)
            if 0 < dist < circle["radius"] + 0.15: #was 0.1
                force += k_s * diff / (dist**3 + 1e-6)
        return force

    def leader_following_force(i):
        pos_force = k_l * (leader_pos - positions[i])
        vel_force = k_l * (leader_vel - velocities[i])
        return pos_force + vel_force

    def leader_obstacle_avoidance_force():
        force = np.zeros(2)
        for circle in circle_obstacles:
            diff = leader_pos - circle["center"]
            dist = np.linalg.norm(diff)
            if dist < circle["radius"] + 0.15: #was 0.1
                force += leader_k_s * diff / (dist**3 + 1e-6) #was 2
        return force

    for step in range(steps):
        new_positions = np.zeros_like(positions)
        new_velocities = np.zeros_like(velocities)
        leader_vel = leader_velocity_function(step * dt)
        leader_pos += (leader_vel + leader_obstacle_avoidance_force()) * dt

        for i in range(num_drones):
            f = (cohesion_force(i) + separation_force(i) + alignment_force(i)
                 + obstacle_avoidance_force(i) + leader_following_force(i))
            new_velocities[i] = velocities[i] + f * dt
            speed = np.linalg.norm(new_velocities[i])
            if speed > max_speed:
                new_velocities[i] = new_velocities[i] / speed * max_speed
            new_positions[i] = positions[i] + new_velocities[i] * dt

        positions = new_positions
        velocities = new_velocities
        positions_history.append(positions.copy())
        leader_history.append(leader_pos.copy())

        # Evaluation metrics
        A = [np.linalg.norm(positions[i] - positions[j]) for i in range(num_drones) for j in range(i+1, num_drones)]
        B = [abs(np.linalg.norm(positions[i] - c["center"]) - c["radius"]) for i in range(num_drones) for c in circle_obstacles]
        C = [np.linalg.norm(positions[i] - leader_pos) for i in range(num_drones)]
        Amin_t.append(np.min(A))
        Bmin_t.append(np.min(B))
        Cmax_t.append(np.max(C))

    if save_outputs:
        np.savetxt("Amin_t.txt", Amin_t)
        np.savetxt("Bmin_t.txt", Bmin_t)
        np.savetxt("Cmax_t.txt", Cmax_t)
        np.save("positions_history.npy", np.array(positions_history))
        np.save("leader_history.npy", np.array(leader_history))

    return Amin_t, Bmin_t, Cmax_t, positions_history, leader_history
