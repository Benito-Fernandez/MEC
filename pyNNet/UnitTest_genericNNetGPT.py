import unittest
import tempfile
import os
import csv
import io
import sys

import numpy as np
import matplotlib.pyplot as plt
from genericNNetGPT3 import NNet

# If your network classes (NNet and Weights) are defined in another module (say, nnet_module),
# you could import them via:
# from nnet_module import Weights, NNet
# For these tests we assume they are in the same module.

class TestWeights(unittest.TestCase):
    def setUp(self):
        self.weights = Weights()
        self.sample_matrix = np.array([0.1, 0.2, 0.3])

    def test_set_and_get_weights(self):
        # Test that setting and retrieving weights works using both [] and callable syntax.
        self.weights.set_weights(0, 1, self.sample_matrix)
        retrieved = self.weights[0, 1]
        np.testing.assert_array_equal(retrieved, self.sample_matrix)
        retrieved_call = self.weights(0, 1)
        np.testing.assert_array_equal(retrieved_call, self.sample_matrix)

    def test_remove_weights(self):
        # Test removal of a weight matrix.
        self.weights.set_weights(0, 1, self.sample_matrix)
        self.weights.remove_weights(0, 1)
        with self.assertRaises(KeyError):
            _ = self.weights[0, 1]

    def test_nonexistent_weights(self):
        # Confirm that accessing a nonexistent weight throws a KeyError.
        with self.assertRaises(KeyError):
            _ = self.weights(99, 99)

    def test_save_and_load_csv(self):
        # Test that saving to a CSV and then re-loading reproduces the weight matrix.
        self.weights.set_weights(0, 1, self.sample_matrix)
        temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', newline='')
        temp_file.close()
        try:
            self.weights.save_to_csv(temp_file.name)
            new_weights = Weights()
            new_weights.load_from_csv(temp_file.name)
            retrieved = new_weights[0, 1]
            np.testing.assert_array_equal(retrieved, self.sample_matrix)
        finally:
            os.unlink(temp_file.name)

class TestNNet(unittest.TestCase):
    def setUp(self):
        # Create a simple network with 3 layers:
        # Input (2 neurons), Hidden (2 neurons), Output (1 neuron).
        self.nNodes = [2, 2, 1]
        self.net = NNet(nNodes=self.nNodes)
        # For sequential connectivity, the default connections are 0->1 and 1->2.
        # Override with deterministic weights to control the forward pass.
        # For connection 0->1, set weights as a 2x2 identity.
        self.net.weights.set_weights(0, 1,
                                     np.array([[1.0, 0.0],
                                               [0.0, 1.0]]))
        # For connection 1->2, set weights as a 1x2 row vector.
        self.net.weights.set_weights(1, 2,
                                     np.array([[1.0, 1.0]]))

    def test_forward(self):
        # With these weights and activations:
        # Layer0 is 'linear'; layer1 uses 'tanh'; layer2 is 'linear' (by default).
        # If we set input x = [1, 0], then:
        # Hidden net = W(0,1) * x = [1, 0]  → activation = tanh([1, 0]) = [tanh(1), 0].
        # Output net = W(1,2) * hidden = 1*tanh(1) + 1*0 = tanh(1)
        # Expected output ≈ tanh(1) (~ 0.7616)
        x = np.array([1.0, 0.0])
        yNet = self.net.forward(x)  # This also updates neuron states.
        expected_hidden = np.tanh(np.array([1.0, 0.0]))  # [tanh(1), 0]
        expected_output = expected_hidden[0]
        self.assertAlmostEqual(yNet[0], expected_output, places=4)

    def test_compute_cost_SE(self):
        # Test the squared error (SE) cost and gradient.
        yNet = np.array([0.7616])
        target = np.array([1.0])
        # Given error = (target - yNet) = 0.2384, squared error = (0.2384)**2.
        total_cost, cost, grad = self.net.compute_cost(yNet, target, cost_type='SE', aggregate='sum')
        self.assertAlmostEqual(total_cost, (0.2384)**2, places=4)
        np.testing.assert_almost_equal(grad, 2*(yNet - target), decimal=4)

    def test_compute_cost_abs(self):
        # Test absolute error cost.
        yNet = np.array([0.7616])
        target = np.array([1.0])
        total_cost, cost, grad = self.net.compute_cost(yNet, target, cost_type='abs', aggregate='sum')
        self.assertAlmostEqual(total_cost, abs(1.0 - 0.7616), places=4)
        np.testing.assert_array_equal(grad, np.array([-1.0]))

    def test_compute_cost_logcosh(self):
        # Test the logcosh cost option.
        yNet = np.array([0.7616])
        target = np.array([1.0])
        total_cost, cost, grad = self.net.compute_cost(yNet, target, cost_type='logcosh', aggregate='sum')
        expected_cost = np.log(np.cosh(target - yNet))
        expected_grad = np.tanh(yNet - target)
        self.assertAlmostEqual(total_cost, expected_cost, places=4)
        np.testing.assert_almost_equal(grad, expected_grad, decimal=4)

    def test_backward_and_weight_grads(self):
        # Run a forward pass and compute cost, then backpropagate.
        x = np.array([1.0, 0.0])
        self.net.forward(x)
        yNet = self.net.states[-1]
        target = np.array([1.0])
        _, _, grad_output = self.net.compute_cost(yNet, target, cost_type='SE', aggregate='sum')
        grad_input = self.net.backward(grad_output)

        # Confirm that gradients are computed for our two connections.
        self.assertIn((0, 1), self.net.weight_grads)
        self.assertIn((1, 2), self.net.weight_grads)
        # For connection 0->1, weight gradient shape should be (2,2); for 1->2, shape should be (1,2).
        self.assertEqual(self.net.weight_grads[(0, 1)].shape, (self.nNodes[1], self.nNodes[0]))
        self.assertEqual(self.net.weight_grads[(1, 2)].shape, (self.nNodes[2], self.nNodes[1]))
        # Also, the gradient with respect to the input must match the input layer dimension.
        self.assertEqual(grad_input.shape, (self.nNodes[0],))

    def test_update_weights(self):
        # Perform a forward pass, compute gradients, then update the weights.
        x = np.array([1.0, 0.0])
        self.net.forward(x)
        yNet = self.net.states[-1]
        target = np.array([1.0])
        _, _, grad_output = self.net.compute_cost(yNet, target, cost_type='SE', aggregate='sum')
        self.net.backward(grad_output)

        # Record weights prior to update.
        old_W01 = self.net.weights.data[(0, 1)].copy()
        old_W12 = self.net.weights.data[(1, 2)].copy()

        # Use plain gradient descent.
        self.net.optimizer = "gd"
        self.net.update_weights()

        new_W01 = self.net.weights.data[(0, 1)]
        new_W12 = self.net.weights.data[(1, 2)]
        # Verify that weights have changed.
        self.assertFalse(np.array_equal(old_W01, new_W01))
        self.assertFalse(np.array_equal(old_W12, new_W12))

    def test_train_method(self):
        # Create a dummy training dataset.
        # Assume each row is structured as: [input1, input2, target]
        dummy_training_data = np.array([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0]
        ])
        self.net.training_data = dummy_training_data

        # Capture printed output when training.
        captured_output = io.StringIO()
        sys.stdout = captured_output
        # Run for 2 epochs (display cost on every epoch).
        self.net.train(n_epochs=2, display_epoch=1, cost_type='SE', aggregate='sum')
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        # Check that cost output is printed for each epoch.
        self.assertIn("Epoch 1/2", output)
        self.assertIn("Epoch 2/2", output)

    def test_repr(self):
        rep = repr(self.net)
        # Check that the representation includes the network structure
        self.assertIn(str(self.nNodes), rep)

if __name__ == '__main__':
    unittest.main()