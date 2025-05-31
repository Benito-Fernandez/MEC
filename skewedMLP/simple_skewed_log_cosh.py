import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# === Parameters ===
np.random.seed(int(time.time()) % (2**32 - 1))
n_samples = 5000
mean   = 1.5
sigma  = 2.5
gain   = 2
slope  = 0.1
target = mean + 2 * sigma  # 2 * 2.5 = 5.0
learning_rate = 0.05
momentum      = 0.01
n_epochs   = 5001
n_steps    = 10
xmin, xmax = mean-10, mean+10
x_vals     = np.linspace(xmin, xmax, 1000)

# === Functions ===
def Cost(error, slope=0.1, gain=2):
    clipped_error = np.clip(error, -100, 100)
    log_cosh = np.log(np.cosh(clipped_error))
    return slope**np.tanh(gain * error) * log_cosh

def CostDerivative(error, slope=0.1, gain=2):
    u = np.tanh(gain * error)
    term1 = slope**u * np.tanh(error)
    term2 = np.log(np.cosh(np.clip(error, xmin, xmax))) * slope**u * np.log(slope) * gain * (1 - u**2)
    return term1 + term2

def pdf(x, mean=0, sigma=1):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

def f(r, x, slope=0.1, gain=2, mean=0, sigma=1):
    error = r - x
    return Cost(error, slope, gain) * pdf(x, mean, sigma)

def f_derivative(r, x, slope=0.1, gain=2, mean=0, sigma=1):
    error = r - x
    cost_grad = CostDerivative(error, slope, gain)
    return cost_grad * pdf(x, mean, sigma)

# Termination values
r_accuracy = 0.000005  # 0.5% relative change
gradient_threshold = 0.00001
n_hold = 5  # Number of epochs for no significant change

def should_terminate(epoch, r, r_old, grad, grad_old, r_count, grad_count, n_epochs, n_hold, r_accuracy):
    if epoch >= n_epochs:
        print(f"Terminated at epoch {epoch}: reached max epochs")
        return True, r_count, grad_count
    if r_old is None or grad_old is None:
        return False, r_count, grad_count

    # Check r change
    if abs(r_old) > 1e-10:
        r_change = abs(r - r_old) / abs(r_old)
        if r_change <= r_accuracy:
            r_count += 1
        else:
            r_count = 0
    # Check grad change
    if abs(grad_old) > 1e-10:
        grad_change = abs(grad - grad_old) / abs(grad_old)
        if grad_change <= gradient_threshold:
            grad_count += 1
        else:
            grad_count = 0

    if r_count >= n_hold:
        print(f"Terminated at epoch {epoch}: r stable for {n_hold} epochs")
        return True, r_count, grad_count
    if grad_count >= n_hold:
        print(f"Terminated at epoch {epoch}: grad stable for {n_hold} epochs")
        return True, r_count, grad_count

    return False, r_count, grad_count

# === Generate Gaussian Data ===
data = np.random.normal(mean, sigma, n_samples)

# === Setup Optimization Data ===
r_values = []
cost_values = []
epochs = []
r = 0
r_old = None
grad_old = None
r_count = 0
grad_count = 0
epoch = 0

while True:
    # Compute gradient and cost
    grad = np.mean(f_derivative(r, data, slope, gain, mean, sigma))
    cost = np.mean(f(r, data, slope, gain, mean, sigma))

    # Store values
    cost_values.append(cost)
    epochs.append(epoch)
    if epoch % n_steps == 0 or epoch == n_epochs - 1:
        r_values.append((epoch, r))

    # Check termination
    terminate, r_count, grad_count = should_terminate(
        epoch, r, r_old, grad, grad_old, r_count, grad_count, n_epochs, n_hold, r_accuracy
    )
    if terminate:
        break

    # Update r
    r_new = r - learning_rate * grad

    # Store old values
    r_old = r
    grad_old = grad
    r = r_new
    epoch += 1

print(f"          Final r value: {r}, Cost = {cost}.")

# === Animation Setup ===
fig, ax = plt.subplots(figsize=(10, 6))

def update(frame):
    ax.clear()
    epoch, r = r_values[frame]
    cost = np.mean(f(r, data, slope, gain, mean, sigma))

    # Histogram and PDF
    ax.hist(data, bins=50, density=True, alpha=0.4, color='skyblue', label='Sample Histogram')
    ax.plot(x_vals, pdf(x_vals, mean, sigma), 'r-', lw=2, label='PDF')

    # Lines for r and target
    ax.axvline(r,      color='green', linestyle='-',  lw=2, label=f'r = {r:.2f}')
    ax.axvline(target, color='gray',  linestyle='--', lw=2, label=f'Target (2σ) = {target:.2f}')

    # Compute f(r, x)
    f_vals = f(r, x_vals, slope, gain, mean, sigma)
    # Scale to match PDF magnitude
    scale_factor = np.max(pdf(x_vals, mean, sigma)) / np.max(f_vals) * 0.5
    ax.plot(x_vals, f_vals * scale_factor, 'b--', lw=2, label='f(r, x) (scaled)')

    ax.set_title(f"Epoch {epoch} | cost = {cost:.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)

ani = FuncAnimation(fig, update, frames=len(r_values), interval=250, repeat=False)
plt.tight_layout()
plt.show()

# === Plot Cost vs. Epochs (Semilog) ===
fig_cost, ax_cost = plt.subplots(figsize=(10, 4))
ax_cost.loglog(epochs, cost_values, 'b-', lw=2, label='Cost (mean f(r, x))')
ax_cost.set_title("Cost vs. Epochs (Semilog)")
ax_cost.set_xlabel("Epoch")
ax_cost.set_ylabel("Cost (log scale)")
ax_cost.legend()
ax_cost.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()