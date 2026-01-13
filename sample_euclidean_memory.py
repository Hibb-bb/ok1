import numpy as np

def origin_point_euclidean(dim: int):
    return np.zeros((dim,), dtype=float)

def origin_batch_euclidean(dim: int, batch: int):
    return np.zeros((batch, dim), dtype=float)

def sample_tangent_ball_at_origin_euclidean(dim: int, M: int, R: float, rng: np.random.Generator):
    u = rng.normal(size=(M, dim))
    u /= np.linalg.norm(u, axis=1, keepdims=True)

    rad = R * (rng.random(M) ** (1.0 / dim))
    u = u * rad[:, None]
    return u  # (M, dim)

def sample_euclidean_points_from_tangent_ball(M: int, dim: int, R: float, rng: np.random.Generator):
    o = origin_point_euclidean(dim)                          # (dim,)
    v = sample_tangent_ball_at_origin_euclidean(dim, M, R, rng)  # (M, dim)

    # Exp_o(v) = o + v in Euclidean space
    x = o[None, :] + v                                       # (M, dim)
    return x, v



def sample_euclidean_points_from_ball(M: int, dim: int, R: float, rng: np.random.Generator):

    v = sample_tangent_ball_at_origin_euclidean(dim, M, R, rng)  # (M, dim)

    # Exp_o(v) = o + v in Euclidean space
    x = o[None, :] + v                                       # (M, dim)
    return x, v


def uniform_sampling_in_ball(M, dim, R, center=None, rng=None):
    random_directions = np.random.normal(size=(M, dim))
    norms = np.linalg.norm(random_directions, axis=1, keepdims=True)
    unit_vectors = random_directions / norms
    random_radii = R * np.random.uniform(0, 1, size=(M, 1))**(1/dim)
    points = unit_vectors * random_radii
    if center is not None:
        points += np.array(center)

    return points