import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def generate_mass_spring_damper_data(total_time=50, dt=0.01, noise_std=0.2):
    time = np.arange(0, total_time, dt)
    steps = len(time)
    force = np.zeros(steps)
    position = np.zeros(steps)
    velocity = np.zeros(steps)
    random_force_values = np.random.uniform(-5, 5, int(total_time / 10))

    for i in range(int(total_time / 10)):
        force[i * int(10 / dt): (i + 1) * int(10 / dt)] = random_force_values[i]

    for k in range(1, steps):
        acceleration = (-2 * position[k-1] - 1 * velocity[k-1] + force[k]) * dt
        velocity[k] = velocity[k-1] + acceleration
        position[k] = position[k-1] + velocity[k] * dt

    position += np.random.normal(0, noise_std, steps)
    velocity += np.random.normal(0, noise_std, steps)

    return pd.DataFrame({"time": time, "force": force, "position": position, "velocity": velocity})

def add_delayed_terms(data, n_delayed_inputs, n_delayed_outputs):
    n_samples = len(data)
    augmented_data = []
    for t in range(max(n_delayed_inputs, n_delayed_outputs), n_samples):
        row = []
        # Add current inputs
        row.extend(data.iloc[t, 1:2])  # Current force (input)
        # Add delayed inputs
        for d in range(1, n_delayed_inputs + 1):
            row.extend(data.iloc[t-d, 1:2])
        # Add delayed outputs
        for d in range(1, n_delayed_outputs + 1):
            row.extend(data.iloc[t-d, 2:])  # Delayed position and velocity
        # Add current outputs (targets)
        row.extend(data.iloc[t, 2:])
        augmented_data.append(row)

    columns = ["input_current"]
    for d in range(1, n_delayed_inputs + 1):
        columns.append(f"input_t-{d}")
    for d in range(1, n_delayed_outputs + 1):
        columns.extend([f"position_t-{d}", f"velocity_t-{d}"])
    columns.extend(["position_target", "velocity_target"])

    return pd.DataFrame(augmented_data, columns=columns)

# Generate training and testing files
training_data = generate_mass_spring_damper_data()
training_data.to_csv("training_file.csv", index=False)

augmented_training_data = add_delayed_terms(training_data, n_delayed_inputs=3, n_delayed_outputs=3)
augmented_training_data.to_csv("augmented_training_file.csv", index=False)


testing_data  = generate_mass_spring_damper_data()
testing_data.to_csv("testing_file.csv", index=False)

augmented_testing_data  = add_delayed_terms(testing_data, n_delayed_inputs=3, n_delayed_outputs=3)
augmented_testing_data.to_csv("augmented_testing_file.csv", index=False)
