import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

from NARMAXNNet import NARMAXNetwork

# Testing Phase
def test_narmax(network, test_file, n_horizon, n_skip=5):
    test_data = pd.read_csv(test_file).values
    time = np.arange(len(test_data))
    inputs = test_data[:, :-2]
    targets = test_data[:, -2:]
    print("\n@test_narmax...")
    print(f"test_data.shape = {test_data.shape}")

    fig, ax = plt.subplots()
    ax.set_xlim(0, len(time))
    ax.set_ylim(-10, 10)
    actual_line, = ax.plot([], [], 'b-', label="Actual Data")
    predicted_line, = ax.plot([], [], 'r-', label="Predicted")
    ax.legend()

    def update(t_idx):
        if t_idx >= len(inputs) - n_horizon:
            return

        #print(f"@ test_narmax.update[{t_idx}]")
        current_input = inputs[t_idx:t_idx + n_horizon]
        #print(f"current_input.shape = {current_input.shape}")
        predictions = network.predict(current_input)

        #print(23*'-^-',)
        actual_line.set_data(time[:t_idx + n_horizon], targets[:t_idx + n_horizon, 0])
        #print(23*'-^-',)
        predicted_line.set_data(time[t_idx:t_idx + n_horizon], predictions[:, 0])
        return actual_line, predicted_line

    print(23*'-+-',)
    print("ani = animation.FuncAnimation(fig,update,frames=range(0,len(time),n_skip),interval=100, blit=True)")
    ani = animation.FuncAnimation(fig, update, frames=range(0, len(time),
                                  n_skip), interval=100, blit=True)
    print(23*'-+-',)
    ani.save("simulation_test.mp4", writer="ffmpeg")
    print(23*'-+-',)
    plt.show()

    # Save predictions to a CSV
    print(23*'-+-',)
    prediction_data = pd.DataFrame(outputs, columns=["Prediction"])
    print(23*'-+-',)
    prediction_data.to_csv("simulation_results.csv", index=False)
    print(23*'-+-',)

# When run as file
if __name__ == "__main__":
    # Create network
    print(69*'-')
    print("network = NARMAXNetwork(n_inputs=3, n_outputs=2, n_hidden=10,\n",
          22*" ",                 "n_horizon=7, n_history=5)")
    print(69*'-')
    network = NARMAXNetwork(n_inputs=3, n_outputs=2, n_inputs_delays=2,
                            n_outputs_delays=3, n_horizon=7)

    #network.inputs = network.inputs.reshape(len(inputs),1)
    print(f"network.inputs.shape = {network.inputs.shape}")
    print(f"network.inputs = \n{network.inputs}")
    print(f"network.outputs.shape = {network.outputs.shape}")
    print(f"network.outputs = \n{network.outputs}")
##    network.states = np.concatenate((network.inputs,
##                                  network.outputs.reshape(len(network.outputs),1)),
##                                  axis=0)
    print(f"network.states.shape = {network.states.shape}")
    print(f"network.states=\n{network.states}")

    print(network)  # Pretty-print the network's topology and parameters

    # save the network
    print(69*'-')
    print("network.save_model('trained_network.net')")
    network.save_model("trained_network.net")  # Save the model into a .net file

    # Load the trained network
    print(69*'-')
    print("network = NARMAXNetwork(n_inputs=3, n_outputs=2, n_hidden=10, n_horizon=5)")
    network = NARMAXNetwork(n_inputs=3, n_outputs=2, n_hidden=10, n_horizon=5)

    print(69*'-')
    print("network.load_model('trained_network.net')")
    network.load_model("trained_network.net")

    # Run the test
    print(69*'-')
    print("test_narmax(network, 'testing_file.csv', n_horizon=5, n_skip=5)")
    test_narmax(network, "testing_file.csv", n_horizon=5, n_skip=5)

    print(69*'-')
    print("                           DONE!")
