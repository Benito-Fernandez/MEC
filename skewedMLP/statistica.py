#-------------------------------------------------------------------------------
# Name:        statistica
# Purpose:
#
# Author:      benit
#
# Created:     26/02/2025
# Copyright:   (c) benit 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np

def statistica(dx = 0.1, dy = 0.1, nXpts = 101, nYpts = 101):
    # Initialize variables
    k = 0
    dx = 0.1
    dy = 0.1
    statisdat = []

    xx = np.linspace(0, nXpts*dx, nXpts)
    yy = np.linspace(0, nYpts*dy, nYpts)
    zz = 1 + 0.5 * np.cos(xx / 2) * np.sin(2 * yy)

    # Loop over the grid and compute the values
    for i in range(nXpts):
        for j in range(nYpts):
            k += 1
            x = (i - 1) * dx
            y = (j - 1) * dy
            z = 1 + 0.5 * np.cos(x / 2) * np.sin(2 * y)
            statisdat.append([x, y, z])

    # Convert the statisdat list into a numpy array
    statisdat = np.array(statisdat)

    return statisdat


def buckner():
    # Initialize variables
    k = 0
    dx = 0.1
    dy = 0.1
    statisdat = []

    # Loop over the grid and compute the values
    for i in range(1, 102):  # MATLAB index starts at 1, so the loop in Python will go from 1 to 101
        for j in range(1, 102):
            k += 1
            x = (i - 1) * dx
            y = (j - 1) * dy
            z = 1 + 0.5 * np.cos(x / 2) * np.sin(y / 4)
            statisdat.append([x, y, z])

    # Convert the statisdat list into a numpy array
    statisdat = np.array(statisdat)

    # Extract the x, y, and z components for plotting
    x_data = statisdat[:, 0]
    y_data = statisdat[:, 1]
    z_data = statisdat[:, 2]

    # 3D Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_data, y_data, z_data, c=z_data, cmap='viridis')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of Buckner Data')

    # Show the plot
    plt.show()

    return statisdat

if __name__ == '__main__':
    data = statistica()

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Call the function to generate the plot and data
    statisdat = buckner()

