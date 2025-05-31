import numpy as np
import matplotlib.pyplot as plt
from distributed_steps import *

def initialize_distributed_steps(tspan=10, t_sigma=1, d_sigma=2):
    """
    Generate random steps over time for tspan

    Parameters:
    tspan   -- defines time span (default: 10)
    t_sigma -- defines largest step duration (default: 1)
    d_sigma -- defines largest step size (default: 2)

    Returns:
    cuando  -- vector of step durations (time points)
    que     -- vector of step sizes

    Originally from The University of Texas at Austin
    Mechanical Engineering Department
    (c) 2003-2013 Benito R. Fernandez
    """

    # Handle tspan as either a single value or an array
    if isinstance(tspan, (list, np.ndarray)):
        end_time = tspan[-1]
    else:
        end_time = tspan

    # Initialize arrays
    cuando = [0]
    que = [0]

    # Calculate number of steps
    nsteps = int(end_time / t_sigma)

    # Generate random steps
    for i in range(1, nsteps):
        cuando.append(cuando[i-1] + t_sigma * np.random.rand())
        que.append(d_sigma * (np.random.randn() - 0.5))

    # Scale cuando to match the full time span
    if cuando[-1] > 0:  # Avoid division by zero
        cuando = np.array(cuando) / cuando[-1] * end_time

    return cuando, que

if __name__ == '__main__':
    times = np.linspace(9,10,100)

    cuando, que = initialize_distributed_steps(tspan=10, t_sigma=1, d_sigma=2)
    for i in range(cuando.size):
        print(f"cuando[{i}] = {cuando[i]}, que[{i}] = {que[i]}")

    forces=[]
    for t in times:
        force = distributed_steps(t, cuando, que)
        print(f"F({t}) = {force}")
        forces.append(force)

    plt.figure(figsize=(12, 6))
    plt.plot(times, forces,'.')
    plt.xlabel('Time')
    plt.ylabel('F(t)')
    plt.grid(True)
    plt.title("Test of initialize_distributed_steps")
    plt.show()


