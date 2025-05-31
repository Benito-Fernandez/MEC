#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      benit
#
# Created:     25/12/2024
# Copyright:   (c) benit 2024
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np

def find_nearest_bin(p, p_min, p_max, n_bins):
##    n = len(p)
##    indices = np.zeros(n, dtype=int)
##    binSize = np.zeros(n, dtype=float)
##
##    for i in range(n):
##        binSize[i] = (p_max[i] - p_min[i]) / (n_bins - 1)
##        indices[i] = int(round((p[i] - p_min[i]) / binSize[i]))
##        indices[i] = max(0, min(indices[i], n_bins - 1))  # Ensure indices are within bounds

    # Calculate bin sizes
    bin_sizes = [(max_val - min_val) / (n_bins - 1) for min_val, max_val in zip(p_min, p_max)]

    # Calculate indices
    indices = [int(round((val - min_val) / bin_size)) for val, min_val, bin_size in zip(p, p_min, bin_sizes)]

    # Ensure indices are within bounds
    indices = [max(0, min(index, n_bins - 1)) for index in indices]

    return indices

import numpy as np

def initialize_Universe(p_min, p_max, n):
    dimensions = len(p_min)
    Ushape = [n] * dimensions
    Uarray = np.zeros(Ushape + [dimensions])
    binSize = np.zeros(dimensions, dtype=float)

    for dim in range(dimensions):
        binSize[dim] = (p_max[dim] - p_min[dim]) / (n - 1)

    for index in np.ndindex(*Ushape):
        for dim in range(dimensions):
            Uarray[index][dim] = p_min[dim] + index[dim] * binSize[dim]

    return Uarray, binSize

##def initialize_ndarray1(x_min, x_max, n):
##    dimensions = len(x_min)
##    shape = [n] * dimensions
##    array = np.zeros(shape)
##
##    for index in np.ndindex(*shape):
##        for dim in range(dimensions):
##            bin_size = (x_max[dim] - x_min[dim]) / (n - 1)
##            array[index] = x_min[dim] + index[dim] * bin_size
##
##    return array
##
##def initialize_ndarray2(nDimensions, nColumns, value):
##    # Create an n-dimensional array with the specified dimensions and fill it with the given value
##    nColumns = np.full(nDimensions, nColumns)
##    array = np.full(nColumns, value)
##    return array

def main():

    #--------------------------------------------------------------------------#
    # Example data for initialize_ndarray
    p_min = np.array([0.0, 3.0, -2.0])
    p_max = np.array([5.0, 7.0,  2.0])
    n_bins = 6

    U, bin_sizes = initialize_Universe(p_min, p_max, n_bins)

    print("\n", 30*"*")
    print(" The Universe of Discourse vertices are:")
    print("  p_min: ", p_min)
    print("  p_max: ", p_max)

    print(" The bins' sizes are: ",bin_sizes)

    print("\n The resulting array has the following elements:\n",U)
    print(U)

    #--------------------------------------------------------------------------#
    # Example data for find_nearest_bin
    p = np.array([2.5, 4.5, 1.5])
    indices = find_nearest_bin(p, p_min, p_max, n_bins)

    print("\n The test vector is: ",p)
    print(" The Nearest bin indices are:", indices)

    p = np.array([3.0, 5.5, -1.5])
    indices = find_nearest_bin(p, p_min, p_max, n_bins)

    print("\n The test vector is: ",p)
    print(" The Nearest bin indices are:", indices)

    print("\n", 30*"*")



if __name__ == '__main__':
    main()

    # Example data
    p = p = np.array([3.0, 5.5, -1.5])# np.array([2.5, 3.5, 4.5])
    p_min = np.array([0.0, 3.0, -2.0])
    p_max = np.array([5.0, 7.0,  2.0])
    n_bins = 6

    # Calculate bin sizes
    bin_sizes = [(max_val - min_val) / (n_bins - 1) for min_val, max_val in zip(p_min, p_max)]

    # Calculate indices
    indices = [int(round((val - min_val) / bin_size)) for val, min_val, bin_size in zip(p, p_min, bin_sizes)]

    # Ensure indices are within bounds
    indices = [max(0, min(index, n_bins - 1)) for index in indices]

    print("Nearest bin indices:", indices)
