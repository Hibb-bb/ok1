import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Geometry helpers (hyperbolic)
# -----------------------------
def minkowski_dot(x, y):
    return -x[..., 0]*y[..., 0] + np.sum(x[..., 1:]*y[..., 1:], axis=-1)

def hyp_dist(x, y, kappa=-1.0):
    assert kappa < 0
    a = np.sqrt(-kappa)
    arg = kappa * minkowski_dot(x, y)
    arg = np.maximum(arg, 1.0 + 1e-12)
    return np.arccosh(arg) / a

def exp_map_origin(v, kappa=-1.0):
    assert kappa < 0
    a = np.sqrt(-kappa)
    r = np.linalg.norm(v, axis=-1, keepdims=True)
    r_safe = np.maximum(r, 1e-12)
    t = (1.0 / a) * np.cosh(a * r)
    s = (np.sinh(a * r) / (a * r_safe)) * v
    return np.concatenate([t, s], axis=-1)

# -----------------------------
# Experiment setup
# -----------------------------
kappa = -1.0
S = 3.0
s_grid = np.linspace(-S, S, 600)

# Preferred stimuli (same for all panels)
N_neurons = 10
u_list = np.linspace(-S, S, N_neurons)

s1 = np.linspace(-S, S, 400)
s2 = np.zeros_like(s1)   # fix one axis

v = np.stack([s1, s2], axis=1)

# neuron preferences in different directions
angles = np.linspace(0, 2*np.pi, N_neurons, endpoint=False)
r = 2.0
u_list = np.stack([r*np.cos(angles), r*np.sin(angles)], axis=1)

# Hyperbolic embeddings
z = exp_map_origin(s_grid[:, None], kappa)
z_pref = exp_map_origin(u_list[:, None], kappa)

sigmas = [1.0, 0.5]

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)

for col, sigma in enumerate(sigmas):
    # -------- Hyperbolic --------
    ax = axes[0, col]
    for zi in z_pref:
        d = hyp_dist(z, zi[None, :], kappa)
        lam = np.exp(-(d**2) / (2 * sigma**2))
        ax.plot(s_grid, lam, linewidth=2, color='lime')
    ax.set_title(f"Hyperbolic, σ={sigma}")

    # -------- Euclidean --------
    ax = axes[1, col]
    for u in u_list:
        d = s_grid - u
        lam = np.exp(-(d**2) / (2 * sigma**2))
        ax.plot(s_grid, lam, linewidth=2, color='lime')
    ax.set_title(f"Euclidean, σ={sigma}")

# Labels
axes[1, 0].set_xlabel("stimulus s")
axes[1, 1].set_xlabel("stimulus s")
axes[0, 0].set_ylabel("firing rate λ(s)")
axes[1, 0].set_ylabel("firing rate λ(s)")

plt.tight_layout()
plt.savefig("tuning_curve_2x2.png", dpi=200)



