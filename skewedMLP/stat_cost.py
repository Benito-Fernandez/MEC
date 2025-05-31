#-------------------------------------------------------------------------------
# Name:        stat_cost
#
# Author:      benit
#
# Created:     26/02/2025
#
#-------------------------------------------------------------------------------
# Copyright (c) 2001-2025 Benito R. Fernandez
#                          benito.fernandez@gmail.com
# Copyright (c) 2001-2025 The Whisper Company (TWC)
#                          benito@TheWhisperCompany.com
# Copyright (c) 2001-2025 Machine Essence Corporation (MEC)
#                          benito.fernandez@MachineEssence.com
# Copyright (c) 2001-2025 Salute Physique Aesthetica Technologie, LLC (SPAT)
#                          OLiu3q@gmail.com
#-------------------------------------------------------------------------------
#
# Purpose:
#    Calculate the cost and gradient for the data based
#    on the specified center and alpha values.
#
#    Parameters:
#    - data: numpy array, data points
#    - center: float, center value (default: 0)
#    - alpha: float, scaling factor (default: 1)
#
#    Returns:
#    - cost: computed cost value
#    - grad: computed gradient
#
#-------------------------------------------------------------------------------

import numpy as np

def stat_cost(data, center=0, alpha=1):
    """
    Calculate the cost and gradient for the data based on the specified center and alpha values.

    Parameters:
    - data: numpy array, data points
    - center: float, center value (default: 0)
    - alpha: float, scaling factor (default: 1)

    Returns:
    - cost: computed cost value
    - grad: computed gradient
    """
    data_size = len(data)
    cost = 0
    grad = np.zeros_like(data)

    for i in range(data_size):
        if data[i] < center:
            cost += (center - data[i]) / alpha / data_size
        else:
            cost += (data[i] - center) * alpha / data_size

        if grad is not None:
            if data[i] < center:
                grad[i] = -1 / alpha / data_size
            else:
                grad[i] = +alpha / data_size

    return cost, grad


if __name__ == '__main__':
    main()
