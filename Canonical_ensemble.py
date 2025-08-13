import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from scipy.fft import fft, ifft


N = 200
sigma = 1.0
epsilon = 1
rc = 2.5 * sigma
phi = 2
T = 2
T_set = 2
mass = 1.0
z_list = []

dt = 0.01 * sigma * np.sqrt(mass / epsilon)

steps = 200


def apply_heyes_thermostat(velocities, T_set, mass, delta=0.05):
    N = len(velocities)
    dof = 2 * N - 2
    kinetic_energy = 0.5 * mass * np.sum(velocities**2)

    u = np.random.uniform(-delta, delta)
    z = np.exp(-u)
    z2 = z**2

    exponent = -kinetic_energy * (z2 - 1) / (1 * T_set)
    acceptance_ratio = z**dof * np.exp(exponent)
    p_acc = min(1.0, acceptance_ratio)

    if np.random.rand() < p_acc:
        velocities *= z
        z_list.append(z)


def show_box(L, sigma, particles):
    fig, ax = plt.subplots()
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal')
    ax.set_title('Particles\' positions')

    for x, y in particles:
        circle = Circle((x, y), radius=sigma * 0.5, edgecolor='black', facecolor='skyblue', alpha=0.6)
        ax.add_patch(circle)
        ax.plot(x, y, 'ko')

    plt.grid(True)
    plt.show()


def animate_particles(L, sigma, n_particles, positions_over_time, gif_filename='animation1.gif', interval=100):
    if not positions_over_time:
        raise ValueError("positions_over_time must not be empty.")

    n_frames = len(positions_over_time)

    fig, ax = plt.subplots()
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal')
    ax.set_title("Particle Animation")

    circles = [Circle((0, 0), sigma * 0.5, edgecolor='black', facecolor='skyblue', alpha=0.6) for _ in
               range(n_particles)]
    for circle in circles:
        ax.add_patch(circle)

    for circle, (x, y) in zip(circles, positions_over_time[0]):
        circle.center = (x, y)

    def update(frame_index):
        positions = positions_over_time[frame_index]
        for circle, (x, y) in zip(circles, positions):
            circle.center = (x, y)
        return circles

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
    anim.save(gif_filename, writer='imagemagick', fps=1000 / interval)

    print(f"Animation saved as {gif_filename}")


def initialize_system(N, sigma, phi, T, mass=1.0):
    area = N * np.pi * sigma ** 2 / phi
    L = np.sqrt(area)
    box = np.array([L, L])

    positions = []
    too_close_counter = 0
    min_dist = sigma

    while len(positions) < N:
        candidate = np.random.rand(2) * box
        too_close = False
        for pos in positions:
            dist = np.linalg.norm(candidate - pos - box * np.round((candidate - pos) / box))
            if dist < min_dist:
                too_close = True
                too_close_counter += 1
                break
        if not too_close:
            positions.append(candidate)

    if len(positions) < N:
        raise RuntimeError("Could not place all particles without overlap. Try lowering the packing fraction.")

    positions = np.array(positions)

    kB = 1.0
    stddev = np.sqrt(kB * T / mass)
    velocities = np.random.normal(0, stddev, size=(N, 2))

    v_cm = velocities.mean(axis=0)
    velocities -= v_cm

    print(too_close_counter)

    return positions, velocities, box


def lj_potential(r_x, r_y, epsilon=1.0, sigma=1.0, rc=2.5):
    r = np.linalg.norm(np.array([r_x, r_y]))
    inv_r = 1.0 / r
    x = sigma * inv_r
    y = rc * inv_r

    # Potential
    term1 = (x ** 2 - 1)
    term2 = (y ** 2 - 1)
    U = epsilon * term1 * term2 ** 2
    return U


def lj_force_numeric(r, h=1e-5):
    fx = (lj_potential(r[0] + h, r[1]) - lj_potential(r[0] - h, r[1])) / (2 * h)
    fy = (lj_potential(r[0], r[1] + h) - lj_potential(r[0], r[1] - h)) / (2 * h)
    return np.array([fx, fy])


def compute_forces(positions, box, epsilon, sigma, rc):
    N = len(positions)
    forces = np.zeros_like(positions)
    potential_energy = 0.0
    particles_distances = []

    for i in range(N):
        for j in range(i + 1, N):

            rij = positions[j] - positions[i]
            rij -= box * np.round(rij / box)
            r = np.linalg.norm(rij)
            particles_distances.append(r)

            if r < rc:
                U = lj_potential(rij[0], rij[1], epsilon=1.0, sigma=1.0)
                potential_energy += U


                fij = lj_force_numeric(rij, h=1e-5)
                forces[i] += fij
                forces[j] -= fij

    r_max = box[0]
    N_bins = 10000
    epsilon_gr = r_max / 2 / N_bins
    counts, edges = np.histogram(particles_distances, range=(0, r_max), bins=N_bins)
    r = 0.5 * (edges[:-1] + edges[1:])
    shell_volumes = 2 * np.pi * r * epsilon_gr
    g_r = counts*2*box[0]**2 / N**2 / shell_volumes

    return forces, potential_energy, g_r


def velocity_verlet(positions, velocities, box, dt, epsilon, sigma, rc, steps, T_set, apply_every_n=10):
    N = len(positions)
    m = mass

    energies = []
    temperatures = []
    positions_over_time = []
    velocities_over_time = []
    g_r_over_time = []

    forces, potential_energy, g_r = compute_forces(positions, box, epsilon, sigma, rc)

    for step in tqdm(range(steps)):
        velocities_half_step = velocities + forces / 2 / m * dt

        if step % apply_every_n == 0:
            apply_heyes_thermostat(velocities_half_step, T_set, m)

        positions = positions + velocities_half_step * dt
        positions %= box

        forces, potential_energy, g_r = compute_forces(positions, box, epsilon, sigma, rc)
        velocities = velocities_half_step + forces / 2 / m * dt


        if step >= steps - 500:
        #if True:
            positions_over_time.append(positions)
            velocities_over_time.append(velocities)

            kinetic_energy = 0.5 * m * np.sum(velocities ** 2)
            total_energy = kinetic_energy + potential_energy
            temperature = 2*kinetic_energy / (2*N-2-1)

            energies.append(total_energy)
            temperatures.append(temperature)
            g_r_over_time.append(g_r)

        """show_box(box[0], sigma, positions)"""

    return positions_over_time, velocities_over_time, energies, temperatures, g_r_over_time


def plot_energy_and_temperature(energies, temperatures, dt):
    time = np.arange(len(energies)) * dt

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(time, energies)
    plt.xlabel('Time')
    plt.ylabel('Total Energy')
    plt.title('Total Energy vs Time')

    plt.subplot(1, 2, 2)
    plt.plot(time, temperatures)
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title('Temperature vs Time')

    plt.tight_layout()
    plt.show()


def plot_g_r(data):
    g_r = np.mean(data, axis=0)
    _, edges = np.histogram([0, 0], range=(0, box[0]), bins=10000)
    r = 0.5 * (edges[:-1] + edges[1:])

    plt.figure(figsize=(8, 5))
    plt.plot(r, g_r, label="g(r)", drawstyle="steps-mid")
    plt.xlabel("Distance r")
    plt.ylabel("g(r)")
    plt.title("Radial Distribution Function")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def calculate_dif_const(positions_over_time, dt, box):
    positions_over_time = np.array(positions_over_time)
    print(f'positions_over_time shape is {positions_over_time.shape}')
    dimensions = positions_over_time.shape[-1]
    print(f'dimensions are {dimensions}')
    time = np.arange(len(positions)) * dt
    delta_r = positions_over_time - positions_over_time[0]
    print(f'delta_r shape is {delta_r.shape}')
    delta_r2 = np.sum(delta_r ** 2, axis=-1)
    print(f'delta_r2 shape is {delta_r2.shape}')
    mean_delta_r2 = np.mean(delta_r2, axis=1)
    print(f'mean_delta_r2 shape is {mean_delta_r2.shape}')

    time = np.arange(len(mean_delta_r2)) * dt
    cutoff_index = np.argmax(mean_delta_r2 > (box[0]*0.45)**2)
    if not cutoff_index:
        cutoff_index = -1
    slope, _ = np.polyfit(time[:cutoff_index], mean_delta_r2[:cutoff_index], 1)
    D = slope / (2 * dimensions)

    plt.figure(figsize=(8, 5))
    plt.plot(time[:cutoff_index], mean_delta_r2[:cutoff_index], label="Δr^2(t)", drawstyle="steps-mid")
    plt.xlabel("t")
    plt.ylabel("Δr^2(t)")
    plt.title("Δr^2 over time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return D


def velocity_autocorrelation(velocities_over_time, dt):

    N, d = velocities_over_time.shape[1], velocities_over_time.shape[2]  # (timesteps, N, d)

    # Subtract CM velocity at each time step to avoid drift
    velocities_over_time -= np.mean(velocities_over_time, axis=1, keepdims=True)

    # Flatten all particles into a single long velocity vector per timestep
    flattened_velocities = velocities_over_time.reshape(velocities_over_time.shape[0], -1)

    # Zero pad to next power of 2 for FFT performance
    n = 2 ** int(np.ceil(np.log2(2 * len(flattened_velocities) - 1)))

    # Apply FFT
    f_vel = fft(flattened_velocities, n=n, axis=0)
    power_spectrum = f_vel * np.conj(f_vel) / 2 / np.pi / len(velocities_over_time)/dt
    acf = ifft(power_spectrum, axis=0).real
    acf = acf[:len(flattened_velocities)]

    # Normalize
    #acf_raw = acf.copy()
    #acf /= np.arange(len(flattened_velocities), 0, -1)[:, None]
    #acf /= acf[0]  # Normalize to 1

    # Average over all components
    vacf = np.mean(acf, axis=1)
    #cutoff_index = np.argmax(vacf < 0)
    #vacf = vacf[:cutoff_index]

    # Integrate to get diffusion constant using the Green-Kubo relation
    D = np.trapezoid(vacf, dx=dt) / d
    time = np.arange(len(vacf)) * dt

    plt.figure(figsize=(8, 5))
    plt.plot(time, vacf, label="VACF", drawstyle="steps-mid")
    plt.xlabel("Time")
    plt.ylabel("<v(0) · v(t)>")
    plt.title("Velocity Autocorrelation Function")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return D


def velocity_autocorrelation2(velocities_over_time, dt):
    velocities = np.array(velocities_over_time)
    n_steps, n_particles, d = velocities.shape
    vacf = np.zeros(n_steps)

    for t in range(n_steps):
        prod = np.sum(
            velocities[0] * velocities[t], axis=1
        )
        vacf[t] = np.mean(prod)

    #cutoff_index = np.argmax(vacf > 0)
    #vacf = vacf[:cutoff_index]

    positive = vacf > 0
    transitions = []
    for i in range(len(positive)-1):
        if int(positive[i]) - int(positive[i + 1]) == 1:
            transitions.append(i)
    #print(transitions)
    cutoff_index = transitions[3] + 1
    #vacf = vacf[:cutoff_index]

    #vacf /= vacf[0]
    time = np.arange(len(vacf)) * dt

    D = np.sum(vacf)*dt/d

    plt.figure(figsize=(8, 5))
    plt.plot(time, vacf, label="VACF", drawstyle="steps-mid")
    plt.xlabel("Time")
    plt.ylabel("<v(0) · v(t)>")
    plt.title("Velocity Autocorrelation Function")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return D


positions, velocities, box = initialize_system(N, sigma, phi, T, mass)
positions_over_time, velocities_over_time, energies, temperatures, g_r_over_time = velocity_verlet(positions, velocities, box, dt, epsilon, sigma, rc,
                                                                 steps, T_set)
"""_, _, energies_set2, temperatures_set2, g_r_over_time_set2 = velocity_verlet(positions, velocities, box, dt, epsilon, sigma, rc,
                                                                 steps, 15)"""
velocities_over_time = np.array(velocities_over_time)
plot_g_r(g_r_over_time)
#plot_g_r(g_r_over_time_set2)

plot_energy_and_temperature(energies, temperatures, dt)
#plot_energy_and_temperature(energies, temperatures_set2, dt)

print(box[0])
print(np.array(velocities_over_time).shape)
print(calculate_dif_const(positions_over_time, dt, box))
print(velocity_autocorrelation(velocities_over_time, dt))
#print(velocity_autocorrelation2(velocities_over_time, dt))
#print(velocity_autocorrelation3(velocities_over_time, dt))
#print(compute_velocity_autocorrelation(velocities_over_time, dt, False))

#animate_particles(box[0], sigma, N, positions_over_time, gif_filename='100particles.gif')
