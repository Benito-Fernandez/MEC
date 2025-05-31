#-------------------------------------------------------------------------------
# Name:        skrdn
#
# Author:      benit
#
# Created:     26/02/2025
#
#-----------------------------------------------------------------------------------
# Copyright (c) 2001-2025 Benito R. Fernandez
#                          benito.fernandez@gmail.com
# Copyright (c) 2001-2025 The Whisper Company (TWC)
#                          benito@TheWhisperCompany.com
# Copyright (c) 2001-2025 Machine Essence Corporation (MEC)
#                          benito.fernandez@MachineEssence.com
# Copyright (c) 2001-2025 Salute Physique Aesthetica Technologie, LLC (SPAT)
#                          OLiu3q@gmail.com
#-----------------------------------------------------------------------------------
#
# Purpose:
#    Generate Skew Random Signal
#
#    Sigma is a 2-column vector with the positive and negative sigmas
#
#
import numpy as np
import matplotlib.pyplot as plt

def skrdn(x, sigma):
    """
    Generate Skew Random Signal.

    Arguments:
    x : int
        Number of samples to generate.
    sigma : list or array-like, length 2
        A two-element vector containing the positive and negative sigmas.

    Returns:
    y : numpy array
        The skew random signal.
    """
    y1 =  np.abs(np.random.randn(x) * sigma[0])
    y2 = -np.abs(np.random.randn(x) * sigma[1])

    r = sigma[1] / (sigma[0] + sigma[1])

    y = np.where(np.random.rand(x) > r, y1, y2)

    return y


if __name__ == '__main__':

    # Example usage
    sigma = [1, 5]  # Positive and negative sigmas

    # Generate data
    x = np.arange(0, 100.01, 0.01)  # Generate x from 0 to 100 with step size 0.01
    y = skrdn(len(x), sigma)  # Skewed random signal
    print(f"Generated 'y = skrdn(len(x), sigma)' for sigma = {sigma} which is of size: {y.size}")

    # Plot the skewed random signal
    plt.figure(1)
    plt.clf()

    # Plotting the skewed data points
    plt.plot(x, y, 'xr', label='Skewed data')

    # Creating histogram
    n, bins = np.histogram(y, bins=25)
    # Scaling the histogram
    n_scaled = n * max(x) / len(x)

    # Plotting the histogram
    plt.plot(n_scaled, bins[:-1], label='Histogram', color='blue')

    # Adding a horizontal line at y=0
    plt.plot([0, 100], [0, 0], 'k', label='y=0 line')

    # Adding title and labels
    plt.title('Test of Asymmetric/Skewed Distribution')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Show legend
    plt.legend()

    # Display the plot
    plt.show()
