import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
def generate_steps(time, dt, input_stdev, duration=[2, 5], values=[-5, 5]):
    """Generate a noisy steps input sequences."""
    np.random.seed(42)  # For reproducibility
    total_time = time[-1]
    step_duration = np.random.uniform(duration[0], duration[1])  # Random duration between 2-5s
    n_steps = int(total_time / step_duration) + 1
    u_base  = np.random.uniform(values[0], values[1], n_steps)  # Base step values
    t_steps = np.arange(0, total_time + step_duration, step_duration)
    u_continuous = np.zeros(int(total_time / dt) + 1)

    # Interpolate steps
    for i in range(len(t_steps) - 1):
        start_idx = int(t_steps[i] / dt)
        end_idx = int(t_steps[i + 1] / dt)
        u_continuous[start_idx:end_idx] = u_base[i]
    u_continuous[end_idx:] = u_base[-1]

    # Add noise
    u_noisy = u_continuous + np.random.normal(0, input_stdev, len(u_continuous))
    return u_noisy

#-------------------------------------------------------------------------------
def generate_sines(time, dt, input_stdev, duration=[2, 5], values=[-5, 5]):
    """Generate a noisy steps input sequences."""
    np.random.seed(57)  # For reproducibility
    total_time = time[-1]
    step_duration = np.random.uniform(duration[0], duration[1])  # Random duration between 2-5s
    n_steps = int(total_time / step_duration) + 1
    u_base  = np.random.uniform(values[0], values[1], n_steps)  # Base step values
    t_steps = np.arange(0, total_time + step_duration, step_duration)
    u_continuous = np.zeros(int(total_time / dt) + 1)

    time_steps = int(total_time/ dt) + 1
    #time       = np.linspace(start_t, end_t, self.time_steps)

    # Interpolate sines
    for i in range(len(t_steps) - 1):
        start_idx = int(t_steps[i] / dt)
        end_idx = int(t_steps[i + 1] / dt)
        u_continuous[start_idx:end_idx] = u_base[i] * np.sin(step_duration * time[start_idx:end_idx])
    u_continuous[end_idx:] = u_base[-1]

##    sinusoidal_force = u_base[i] * np.sin(step_duration * time)  # 5*sin(2.5*t)
##    noise = np.random.normal(0, input_stdev, time_steps)  # Random noise (N)
##    F_testing  = sinusoidal_force  + noise # Combined force


    # Add noise
    u_noisy = u_continuous + np.random.normal(0, input_stdev, len(u_continuous))
    return u_noisy

#-------------------------------------------------------------------------------
# Model: mass-spring-dashpot system
class MSD:
    def __init__(self, mass=1,
                       damping=0.25,  alpha=0.1,
                       stiffness=0.2, beta=0.05,
                       x0=0, v0=0,
                       start_t=0, end_t=10, dt=0.01):
        # Parameters for mass-spring-dashpot system
        self.m = mass
        self.k = stiffness
        self.c = damping
        self.a = alpha
        self.b = beta

        # Simulation settings
        self.dt      = dt  # time step (seconds)
        self.t_start = start_t
        self.t_end   = end_t  # simulation time (seconds)
        self.time_steps = int((end_t - start_t)/ dt) + 1
        self.time       = np.linspace(start_t, end_t, self.time_steps)

        # Initial conditions
        self.x_0 = x0  # initial position (m)
        self.v_0 = v0  # initial velocity (m/s)

        # Arrays to store results
        self.x = x0*np.ones(self.time_steps)  # position (m)
        self.v = v0*np.ones(self.time_steps)  # velocity (m/s)

    def damper(p, c=0.25, alpha=0.1):
        return (c  +  alpha * log(cosh(p)) * tanh(p))* p

    def spring(x, k=0.2, beta=0.05):
        return x * (k + beta * abs(x))

    def msd(t, F, x, v, m, k, c, alpha, beta):
        v_dot = (F - damper(v, c, alpha) -  spring(x, k, beta)) / m
        x_dot = v
        return v_dot, x_dot # = msd(t, F, x, v, m, k, c, alpha, beta)

    def simulate(self, time_steps=None, force_function=generate_steps, input_stdev = 0.2):
        # Numerical integration (Euler's method) for second-order ODE
        self.force = force_function(self.time, self.dt, input_stdev, duration, values)
        for t in range(1, self.time_steps):
            # Acceleration: m * a = F(t) - c * v - k * x
            a = (self.force[t] - self.c * self.v[t-1] - self.k * self.x[t-1]) / self.m
            self.v[t] = self.v[t-1] + a * self.dt  # Update velocity
            self.x[t] = self.x[t-1] + self.v[t-1] * self.dt  # Update position
        return self.force, self.x, self.v

    def plot_results(self, title='Mass-Spring-Dashpot System'):
        # Optionally plot the training data to visualize
        plt.figure(figsize=(10, 6))
        plt.plot(self.time, self.x, label='Position [m]')
        plt.plot(self.time, self.v, label='Velocity [m/s]')
        plt.plot(self.time, self.force, label='Input Force [N]', alpha=0.5)
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Magnitude')
        plt.title(title)
        plt.show()

    def store_results(self, file_name='msd_data.csv'):
        # Create DataFrame for training data (time, force, position, velocity)
        training_data = pd.DataFrame({
            'time':        self.time,
            'input_force': self.force,
            'position':    self.x,
            'velocity':    self.v
        })

        # Save to CSV for training data
        training_data.to_csv(file_name, index=False)
        self.data = training_data

# Random steps input force and Gaussian noise
input_stdev = 0.2
duration = [ 1, 2]   # step duration & sinewave amplitude
values   = [-5, 5]   # step value    & sinewave frequency
# Maybe add to class simulate

msd_train = MSD(mass=1,
                damping=0.25,  alpha=0.1,
                stiffness=0.2, beta=0.05,
                x0=0, v0=0,
                start_t=0, end_t=10, dt=0.01)
training_force, position, velocity \
= msd_train.simulate(force_function=generate_steps, input_stdev = 0.2)
msd_train.plot_results(title='MSD-training w/ steps')
msd_train.store_results('training_data.csv')

msd_test  = MSD(mass=1,
                damping=0.10,  alpha=0.05,
                stiffness=0.35, beta=0.15,
                x0=0, v0=0,
                start_t=0, end_t=10, dt=0.01)
testing_force, position, velocity \
= msd_test.simulate(force_function=generate_sines, input_stdev = 0.1)
msd_test.plot_results(title='MSD-testing w/ sines')
msd_test.store_results('test_data.csv')
