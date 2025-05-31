import numpy as np
import pandas as pd

def generate_mass_spring_damper_data(dt, t_total, input_force_steps, noise_std=0.2):
    """
    Simulates a mass-spring-damper system with noisy outputs.

    Args:
        dt (float): Sampling time (seconds).
        t_total (float): Total simulation time (seconds).
        input_force_steps (int): Duration (in seconds) for which the force remains constant.
        noise_std (float): Standard deviation of the Gaussian noise for the outputs.

    Returns:
        pd.DataFrame: A DataFrame containing time, input force, position, and velocity.
    """
    # System parameters
    mass = 1.0      # Mass (kg)
    damping = 0.5   # Damping coefficient
    stiffness = 1.0 # Spring constant (N/m)

    # Time vector
    time = np.arange(0, t_total, dt)

    # Force signal: Changes randomly every 'input_force_steps'
    num_steps = int(t_total / input_force_steps)
    forces = np.random.uniform(-5, 5, num_steps)
    force_signal = np.repeat(forces, int(input_force_steps / dt))

    # Initialize state variables
    position = 0.0
    velocity = 0.0
    positions = []
    velocities = []

    # Simulate the system using Euler's method
    for f in force_signal:
        # Second-order differential equation: m*x'' + c*x' + k*x = F
        acceleration = (f - damping * velocity - stiffness * position) / mass

        # Update states
        velocity += acceleration * dt
        position += velocity * dt

        # Save states (add noise)
        positions.append( position + np.random.normal(0, noise_std))
        velocities.append(velocity + np.random.normal(0, noise_std))

    # Combine data into a DataFrame
    data = pd.DataFrame({
        'time': time[:len(force_signal)],
        'force': force_signal[:len(time)],
        'position': positions,
        'velocity': velocities,
    })

    return data

# Generate Training Data
dt = 0.01  # Sampling time
t_total = 50  # Total simulation time (seconds) for training
training_data = generate_mass_spring_damper_data(dt, t_total, input_force_steps=10, noise_std=0.2)
training_data.to_csv("training_file.csv", index=False)

# Generate Testing Data
t_test = 25  # Total simulation time (seconds) for testing
testing_data = generate_mass_spring_damper_data(dt, t_total, input_force_steps=10, noise_std=0.2)
testing_data.to_csv("testing_file.csv", index=False)
