import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# === Parameters ===
np.random.seed(int(time.time()) % (2**32 - 1))
n_samples = 2500
mean = 1.25
sigma = 2.5
n_bins = 50
gain = 2
# Slope to capture confidence level o 95%
slope_lower = 125
slope_upper = 1/slope_lower
'''
Sigma (±σ)    Confidence Level
    ±1σ         68.27%
    ±2σ         95.45%
    ±3σ         99.73%
    ±4σ         99.9937%
    ±5σ         99.99994%
'''
target_upper = mean + 2 * sigma  #  6.25 (for mean = 1.25, sigma = 2.5)
target_lower = mean - 2 * sigma  # -3.75 (for mean = 1.25, sigma = 2.5)
learning_rate = 0.005
n_epochs = 10001
n_steps = 100
n_hold = 5
# Termination values
r_accuracy         = 0.000005   # 0.5% relative change
gradient_threshold = 0.00001    # gradient threshold
n_hold = 5  # Number of epochs for no significant change

xmin = mean - 10  # -10
xmax = mean + 10  #  10
x_vals = np.linspace(xmin, xmax, n_samples)

# === Functions ===
def Cost(error, slope=0.1, gain=2):
    clipped_error = np.clip(error, -100, 100)
    log_cosh = np.log(np.cosh(clipped_error))
    return slope**np.tanh(gain * error) * log_cosh

def CostDerivative(error, slope=0.1, gain=2):
    u = np.tanh(gain * error)
    term1 = slope**u * np.tanh(error)
    term2 = np.log(np.cosh(np.clip(error, -100, 100))) * slope**u * np.log(slope) * gain * (1 - u**2)
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

def should_terminate(epoch, r, r_old, grad, grad_old, r_count, grad_count, n_epochs, n_hold, r_accuracy, label=""):
    if epoch >= n_epochs:
        print(f"Terminated {label} at epoch {epoch}: reached max epochs")
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
        print(f"Terminated {label} at epoch {epoch}: r stable for {n_hold} epochs")
        return True, r_count, grad_count
    if grad_count >= n_hold:
        print(f"Terminated {label} at epoch {epoch}: grad stable for {n_hold} epochs")
        return True, r_count, grad_count

    return False, r_count, grad_count

# === Generate Gaussian Data ===
data = np.random.normal(mean, sigma, n_samples)

# === Setup Optimization Data ===
r_values_upper = []
r_values_lower = []
cost_values_upper = []
cost_values_lower = []
epochs = []
r_upper = mean
r_lower = mean
r_upper_old = None
r_lower_old = None
grad_upper_old = None
grad_lower_old = None
r_upper_count = 0
r_lower_count = 0
grad_upper_count = 0
grad_lower_count = 0
epoch = 0
terminated_upper = False
terminated_lower = False

while not (terminated_upper and terminated_lower):
    # Upper target optimization
    if not terminated_upper:
        grad_upper = np.mean(f_derivative(r_upper, data, slope_upper, gain, mean, sigma))
        cost_upper = np.mean(f(r_upper, data, slope_upper, gain, mean, sigma))
        cost_values_upper.append(cost_upper)
        if epoch % n_steps == 0 or epoch == n_epochs - 1:
            r_values_upper.append((epoch, r_upper))

        # Check termination
        terminate_upper, r_upper_count, grad_upper_count = should_terminate(
            epoch, r_upper, r_upper_old, grad_upper, grad_upper_old,
            r_upper_count, grad_upper_count, n_epochs, n_hold, r_accuracy, label="upper"
        )
        if terminate_upper:
            terminated_upper = True
        else:
            r_upper_new = r_upper - learning_rate * grad_upper
            r_upper_old = r_upper
            grad_upper_old = grad_upper
            r_upper = r_upper_new

    # Lower target optimization
    if not terminated_lower:
        grad_lower = np.mean(f_derivative(r_lower, data, slope_lower, gain, mean, sigma))
        cost_lower = np.mean(f(r_lower, data, slope_lower, gain, mean, sigma))
        cost_values_lower.append(cost_lower)
        if epoch % n_steps == 0 or epoch == n_epochs - 1:
            r_values_lower.append((epoch, r_lower))

        # Check termination
        terminate_lower, r_lower_count, grad_lower_count = should_terminate(
            epoch, r_lower, r_lower_old, grad_lower, grad_lower_old,
            r_lower_count, grad_lower_count, n_epochs, n_hold, r_accuracy, label="lower"
        )
        if terminate_lower:
            terminated_lower = True
        else:
            r_lower_new = r_lower - learning_rate * grad_lower
            r_lower_old = r_lower
            grad_lower_old = grad_lower
            r_lower = r_lower_new

    # Store epoch
    epochs.append(epoch)
    epoch += 1
print(f"Final values:")
print(f" r_lower {r_lower:>7.4f}, true {mean-2*sigma:>7.4f}, Cost_lower = {cost_lower:6.4f}")
print(f" r_upper {r_upper:>7.4f}, true {mean+2*sigma:>7.4f}, Cost_upper = {cost_upper:6.4f}.")

# === Animation Setup ===
fig, ax = plt.subplots(figsize=(10, 6))

def update(frame):
    ax.clear()
    epoch_upper, r_upper = r_values_upper[min(frame, len(r_values_upper)-1)]
    epoch_lower, r_lower = r_values_lower[min(frame, len(r_values_lower)-1)]

    # Histogram and PDF
    ax.hist(data, bins=n_bins, density=True, alpha=0.4, color='skyblue', label='Sample Histogram')
    ax.plot(x_vals, pdf(x_vals, mean, sigma), 'k-', lw=2, label='PDF')

    # Lines for r and targets
    ax.axvline(r_upper, color='green', linestyle='-', lw=2, label=f'r_upper = {r_upper:.2f}')
    ax.axvline(r_lower, color='red', linestyle='-', lw=2, label=f'r_lower = {r_lower:.2f}')
    ax.axvline(target_upper, color='gray', linestyle='--', lw=2, label=f'Target upper (2σ) = {target_upper:.2f}')
    ax.axvline(target_lower, color='gray', linestyle='--', lw=2, label=f'Target lower (-2σ) = {target_lower:.2f}')

    # Compute f(r, x) for both
    f_upper_vals = f(r_upper, x_vals, slope_upper, gain, mean, sigma)
    f_lower_vals = f(r_lower, x_vals, slope_lower, gain, mean, sigma)
    # Scale to match PDF magnitude
    scale_factor_upper = np.max(pdf(x_vals, mean, sigma)) / np.max(f_upper_vals) * 0.5
    scale_factor_lower = np.max(pdf(x_vals, mean, sigma)) / np.max(f_lower_vals) * 0.5
    ax.plot(x_vals, f_upper_vals * scale_factor_upper, 'g--', lw=2, label='f_upper (scaled)')
    ax.plot(x_vals, f_lower_vals * scale_factor_lower, 'r--', lw=2, label='f_lower (scaled)')

    ax.set_title(f"Epoch {max(epoch_upper, epoch_lower)} | r_upper = {r_upper:.4f}, r_lower = {r_lower:.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)

max_frames = max(len(r_values_upper), len(r_values_lower))
ani = FuncAnimation(fig, update, frames=max_frames, interval=250, repeat=False)
plt.tight_layout()
plt.show()

# === Plot Cost vs. Epochs (Semilog) ===
fig_cost, ax_cost = plt.subplots(figsize=(10, 4))
ax_cost.semilogy(epochs, cost_values_upper, 'g-', lw=2, label='Cost upper (mean f(r_upper, x))')
ax_cost.semilogy(epochs, cost_values_lower, 'r-', lw=2, label='Cost lower (mean f(r_lower, x))')
ax_cost.set_title("Cost vs. Epochs (Semilog)")
ax_cost.set_xlabel("Epoch")
ax_cost.set_ylabel("Cost (log scale)")
ax_cost.legend()
ax_cost.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()