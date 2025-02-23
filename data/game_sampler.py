import numpy as np

# Uniform sampler with equal probability for all positions
def uniform_sampler(length_game, num_to_sample):
    return np.random.choice(length_game, size=num_to_sample, replace=False)


def linear_augmentation_sampler(length_game, num_to_sample):
    """
    Samples positions with linearly increasing probability from beginning to end.
    """
    weights = np.linspace(1, length_game, length_game)
    probabilities = weights / np.sum(weights)
    return np.random.choice(length_game, size=num_to_sample, replace=False, p=probabilities)


def constant_then_linear_sampler(length_game, num_to_sample):
    """
    Constant weight for first 20 positions, then linearly increasing weight.
    """
    weights = np.concatenate([np.ones(20), np.linspace(1, length_game - 20, length_game - 20)])
    probabilities = weights / np.sum(weights)
    return np.random.choice(length_game, size=num_to_sample, replace=False, p=probabilities)



def order_and_compute_deltas(sampled_positions):
    """
    sorts elements and computes the deltas between consecutive positions.

    Returns:
    tuple: First chosen position and the array of deltas between consecutive positions.
    """

    sorted_positions = np.sort(sampled_positions)

    # Compute deltas (differences) between consecutive positions
    deltas = np.diff(sorted_positions)

    # Return the first chosen position and the deltas
    L_res = deltas.tolist()
    L_res.insert(0, int(sorted_positions[0]))
    return L_res


if __name__ == '__main__':
    print(uniform_sampler(100, 10))
    print(linear_augmentation_sampler(100, 10))
    print(constant_then_linear_sampler(100, 10))
    print(order_and_compute_deltas(linear_augmentation_sampler(100, 10)))
