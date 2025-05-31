import numpy as np

def distributed_steps(t, cuando, que):
    """
    Return a step function of t over several steps

    Parameters:
    t      -- time values (scalar or array)
    cuando -- vector defining steps durations
    que    -- vector defining steps sizes

    Originally from The University of Texas at Austin
    Mechanical Engineering Department
    (c) 2003-2013 Benito R. Fernandez
    """

    # Convert t to numpy array if it's not already
    t = np.asarray(t)

    if t.size > 1:  # If t is an array
        force = np.zeros(t.shape)
        for i in range(t.size):
            # Find the indices where cuando <= current t value
            indices = np.where(np.array(cuando) <= t[i])[0]
            if indices.size > 0:
                # Get the last (max) index and use it to index into que
                force[i] = que[indices[-1]]
    else:  # If t is a scalar
        indices = np.where(np.array(cuando) <= t)[0]
        if indices.size > 0:
            force = que[indices[-1]]
        else:
            force = 0  # Default if no valid step is found

    return force



