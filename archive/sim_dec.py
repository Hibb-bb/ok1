import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Simulation: Euclidean vs Lorentz (hyperbolic) MAP (nearest-prototype)
# ----------------------------

rng = np.random.default_rng(0)

# Parameters (tweak as desired)
dims = np.array([2, 3, 5, 8, 12, 16, 24, 32, 48, 64])  # feature dimensions d
M = 1000                       # number of prototypes/classes
T = 4000                      # trials per dimension
R = 2.0                       # "radius" of prototypes (Euclidean norm / hyperbolic geodesic radius)
sigma = 1.0                   # noise scale (Euclidean ambient; hyperbolic tangent)

def unit_sphere(d, n):
    x = rng.normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

# Minkowski inner product <a,b>_L = -a0 b0 + sum_{i>=1} ai bi
def minkowski_dot(a, b):
    return -a[..., 0]*b[..., 0] + np.sum(a[..., 1:]*b[..., 1:], axis=-1)

def lorentz_norm(v):
    # v is tangent (so <v,v>_L > 0 for spacelike vectors)
    val = minkowski_dot(v, v)
    return np.sqrt(np.maximum(val, 1e-12))

def arcosh(x):
    return np.log(x + np.sqrt(np.maximum(x*x - 1.0, 0.0)))

def hyp_dist(x, y):
    # d(x,y) = arcosh(-<x,y>_L), for points on hyperboloid with <x,x>_L=-1
    ip = minkowski_dot(x, y)
    return arcosh(np.maximum(-ip, 1.0 + 1e-12))

def hyp_shell_points(d, n, R):
    # Points at geodesic radius R from origin: x0=cosh(R), spatial=sinh(R)*u
    u = unit_sphere(d, n)
    x0 = np.full((n, 1), np.cosh(R))
    xs = np.sinh(R) * u
    return np.concatenate([x0, xs], axis=1)  # shape (n, d+1)

def project_to_tangent(x, g):
    # Project ambient vector g to tangent space at x: v = g + <g,x>_L x (so <v,x>_L=0)
    ip = minkowski_dot(g, x)[..., None]
    return g + ip * x

def exp_map(x, v):
    # Exp_x(v) = cosh(||v||) x + sinh(||v||) v/||v||   (curvature -1)
    nv = lorentz_norm(v)[..., None]
    # handle tiny nv
    coef1 = np.cosh(nv)
    coef2 = np.sinh(nv) / np.maximum(nv, 1e-12)
    return coef1 * x + coef2 * v

def euclid_accuracy(d):
    # prototypes on Euclidean sphere of radius R
    S = R * unit_sphere(d, M)               # (M,d)
    # pick true indices
    y = rng.integers(0, M, size=T)
    x = S[y] + sigma * rng.normal(size=(T, d))
    # decode: nearest prototype in Euclidean distance
    # compute squared distances: ||x||^2 + ||S||^2 - 2 x·S
    x2 = np.sum(x*x, axis=1, keepdims=True)        # (T,1)
    s2 = np.sum(S*S, axis=1)[None, :]              # (1,M) = R^2
    d2 = x2 + s2 - 2.0 * (x @ S.T)                 # (T,M)
    yhat = np.argmin(d2, axis=1)
    return np.mean(yhat == y)

def lorentz_accuracy(d):
    # prototypes on hyperbolic shell radius R (Lorentz model)
    S = hyp_shell_points(d, M, R)                  # (M, d+1)
    y = rng.integers(0, M, size=T)
    x0 = S[y]                                      # (T,d+1)
    # noise in tangent: sample ambient gaussian then project to tangent, scale to sigma
    g = rng.normal(size=(T, d+1))
    v = project_to_tangent(x0, g)
    # normalize direction; then scale to sigma (so typical ||v|| ~ sigma*sqrt(d))
    # If you'd rather set expected ||v|| ~ sigma, uncomment the normalization block below.
    v = sigma * v  # simplest comparable per-coordinate noise
    
    x = exp_map(x0, v)                             # noisy point on hyperboloid
    # decode: nearest in hyperbolic geodesic distance
    # compute distances to all prototypes
    # vectorize: compute Minkowski dot of x (T, d+1) with S (M, d+1)
    # ip = -x0*S0 + sum xi*si
    ip = -x[:, [0]] * S[None, :, 0] + (x[:, None, 1:] * S[None, :, 1:]).sum(axis=2)  # (T,M)
    dist = arcosh(np.maximum(-ip, 1.0 + 1e-12))
    yhat = np.argmin(dist, axis=1)
    return np.mean(yhat == y)

eu_acc = []
hy_acc = []
for d in dims:
    eu_acc.append(euclid_accuracy(int(d)))
    hy_acc.append(lorentz_accuracy(int(d)))

eu_acc = np.array(eu_acc)
hy_acc = np.array(hy_acc)

plt.figure(figsize=(7.5, 4.8))
plt.plot(dims, eu_acc, marker='o', linewidth=2, label='Euclidean MAP (NN)')
plt.plot(dims, hy_acc, marker='o', linewidth=2, label='Lorentz/Hyperbolic MAP (geodesic NN)')
plt.xlabel('Dimension d')
plt.ylabel('Accuracy')
plt.title(f'Decoding Acc. vs dimension (M={M})')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("sim_good.png")

