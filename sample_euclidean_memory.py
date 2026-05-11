import numpy as np


def uniform_sampling_in_ball(M, dim, R, center=None, rng=None):
    random_directions = np.random.normal(size=(M, dim))
    norms = np.linalg.norm(random_directions, axis=1, keepdims=True)
    unit_vectors = random_directions / norms
    random_radii = R * np.random.uniform(0, 1, size=(M, 1))**(1/dim)
    points = unit_vectors * random_radii
    if center is not None:
        points += np.array(center)

    return points
