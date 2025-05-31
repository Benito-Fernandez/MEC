import numpy as np
import matplotlib.pyplot as plt
from genericNNetGPT3 import NNet  # Base NN class

# =============================================================================
# Derived Class for a Nonlinear ARMAX Model
# =============================================================================

class NonlinearARMAX(NNet):
    def __init__(self, ar_order, ma_order, exog_order, n_hidden, n_outputs, **kwargs):
        """
        Create a Nonlinear ARMAX network.

        Parameters:
          ar_order:  Number of lagged outputs (autoregressive part)
          ma_order:  Number of lagged errors (moving average part)
          exog_order:Number of exogenous input variables per timestep
          n_hidden:  List specifying hidden layer sizes.
          n_outputs: Number of outputs (typically 1)

        The total input dimension is: exog_order + ar_order + ma_order.
        Additional keyword arguments are passed along to the base NNet initializer.
        """
        # Determine input size from AR, MA, and exogenous orders.
        n_inputs = exog_order + ar_order + ma_order
        super().__init__(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs, **kwargs)
        self.ar_order = ar_order
        self.ma_order = ma_order
        self.exog_order = exog_order
        # Initialize history buffers with zeros.
        self.output_history = [0.0] * ar_order
        self.error_history  = [0.0] * ma_order

    def update_history(self, y_pred, error):
        """
        Update history buffers with the latest predicted output and prediction error.

        y_pred and error are expected to be scalars (or 1-element arrays).
        """
        # In case y_pred or error are arrays, extract the scalar value.
        y_val = y_pred.item() if isinstance(y_pred, np.ndarray) else y_pred
        err_val = error.item() if isinstance(error, np.ndarray) else error

        self.output_history.append(y_val)
        if len(self.output_history) > self.ar_order:
            self.output_history = self.output_history[-self.ar_order:]
        self.error_history.append(err_val)
        if len(self.error_history) > self.ma_order:
            self.error_history = self.error_history[-self.ma_order:]

    def prepare_input(self, exog_current):
        """
        Build the augmented input vector for the ARMAX model.

        The input vector is constructed as:
          [ current exogenous inputs, lag-outputs, lag-errors ]

        If the provided exogenous inputs or history buffers are shorter than required,
        they will be padded with zeros.
        """
        # Ensure the current exogenous inputs are in proper vector form
        exog_arr = np.array(exog_current).flatten()
        if len(exog_arr) < self.exog_order:
            exog_arr = np.pad(exog_arr, (0, self.exog_order - len(exog_arr)), 'constant')
        else:
            exog_arr = exog_arr[:self.exog_order]

        # Past outputs: if not enough history, pad at the beginning with zeros.
        ar_vals = np.array(self.output_history).flatten()
        if len(ar_vals) < self.ar_order:
            ar_vals = np.pad(ar_vals, (self.ar_order - len(ar_vals), 0), 'constant')

        # Past errors: pad if needed.
        ma_vals = np.array(self.error_history).flatten()
        if len(ma_vals) < self.ma_order:
            ma_vals = np.pad(ma_vals, (self.ma_order - len(ma_vals), 0), 'constant')

        # Concatenate into one input vector.
        input_vec = np.concatenate([exog_arr, ar_vals, ma_vals])
        return input_vec

    def forward_armax(self, exog_current):
        """
        Perform a one-step forecast by augmenting the exogenous input with the
        current AR and MA history and calling the network's forward method.
        """
        input_vec = self.prepare_input(exog_current)
        y_pred = self.forward(input_vec)
        return y_pred

    def train_armax(self, exog_series, target_series, n_epochs=1, cost_type='SE', aggregate='sum'):
        """
        Train the ARMAX model on a time-series dataset provided as a sequence.

        Parameters:
          exog_series:  A sequence (or 1D NumPy array) of exogenous inputs.
                        Each element should be an array or scalar of length exog_order.
          target_series: A sequence (or 1D NumPy array) of target outputs.
          n_epochs:     Number of passes over the dataset
          cost_type:    Cost function type; 'SE' (Squared Error) by default.
          aggregate:    Aggregation method for cost ('sum', 'max', or 'norm').

        Returns:
          loss_history: A list with the average loss for each epoch.
        """
        n_samples = len(target_series)
        loss_history = []
        for epoch in range(n_epochs):
            total_loss = 0.0
            for t in range(n_samples):
                current_exog = exog_series[t]
                current_target = np.array([target_series[t]])  # Ensure target is array-like.
                input_vec = self.prepare_input(current_exog)
                y_pred = self.forward_armax(input_vec)
                total_cost, cost, grad_output = self.compute_cost(y_pred, current_target,
                                                                  cost_type=cost_type,
                                                                  aggregate=aggregate)
                total_loss += total_cost
                self.backward(grad_output)
                self.update_weights()
                error = current_target - y_pred  # Compute error for history update.
                self.update_history(y_pred, error)
            avg_loss = total_loss / n_samples
            loss_history.append(avg_loss)
            print("Epoch {}: Loss = {}".format(epoch+1, avg_loss))
        return loss_history

    def predict_armax(self, exog_series):
        """
        Given a sequence (or array) of exogenous inputs, produce one-step-ahead predictions.
        Note that the internal history is updated as predictions are made.

        Returns:
          predictions: A NumPy array of predictions.
        """
        predictions = []
        for current_exog in exog_series:
            y_pred = self.forward_armax(current_exog)
            predictions.append(y_pred)
            # In prediction mode, you might assume zero error (or update otherwise)
            self.update_history(y_pred, 0)
        return np.array(predictions)


# =============================================================================
# Test Code for the Nonlinear ARMAX Model
# =============================================================================

# In this test, we'll simulate a simple nonlinear time series.
# Suppose the true system is:
#    y(t) = tanh(0.5 * y(t-1) - 0.3 * y(t-2) + 0.5 * u(t)) + noise
# where u(t) is an exogenous input signal.

def generate_armax_data(n_samples=200, noise_std=0.05, seed=42):
    np.random.seed(seed)
    # Generate exogenous input: a slow sine-wave.
    t = np.linspace(0, 4 * np.pi, n_samples)
    u = np.sin(t)  # shape: (n_samples,)
    y = np.zeros(n_samples)
    # Initialize first two outputs.
    y[0] = 0.0
    y[1] = 0.1
    for t_idx in range(2, n_samples):
        # Nonlinear AR component plus exogenous input.
        y[t_idx] = np.tanh(0.5 * y[t_idx - 1] - 0.3 * y[t_idx - 2] + 0.5 * u[t_idx]) \
                   + np.random.normal(0, noise_std)
    return u, y

#-------------------------------------------------------------------------------
# Sample execution
if __name__ == "__main__":

    # Generate synthetic data.
    n_samples = 200
    exog_series, target_series = generate_armax_data(n_samples=n_samples)

    # Set ARMAX orders.
    ar_order = 2   # use 2 past outputs
    ma_order = 2   # use 2 past errors
    exog_order = 1 # our exogenous input is scalar

    # Network configuration: input dimension will automatically be (exog_order + ar_order + ma_order)
    n_hidden = [10, 10]  # Two hidden layers; you may adjust as needed.
    n_outputs = 1

    # Create instance of the Nonlinear ARMAX model.
    armax_model = NonlinearARMAX(ar_order=ar_order,
                                 ma_order=ma_order,
                                 exog_order=exog_order,
                                 n_hidden=n_hidden,
                                 n_outputs=n_outputs)

    # Train the model on the synthetic time series.
    n_epochs = 50
    loss_history = armax_model.train_armax(exog_series, target_series, n_epochs=n_epochs,
                                           cost_type='SE', aggregate='sum')

    # Generate predictions.
    # For testing, we can run the predictor on the same exogenous series.
    predictions = armax_model.predict_armax(exog_series)

    # =============================================================================
    # Plotting: Loss Evolution and Predictions vs. True Signal
    # =============================================================================

    plt.figure(figsize=(10, 4))
    plt.plot(loss_history, color='purple', marker='o')
    plt.title('Training Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(target_series, label='True Output', color='red', linewidth=2)
    plt.plot(predictions, label='Predicted Output', color='green', linestyle='--')
    plt.title('Nonlinear ARMAX: Predictions vs. True Output')
    plt.xlabel('Time Step')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()
