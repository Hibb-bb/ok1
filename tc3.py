import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def minkowski_dot(x, y):
    return -x[..., 0]*y[..., 0] + np.sum(x[..., 1:]*y[..., 1:], axis=-1)

def hyp_dist(x, y, kappa=-1.0):
    a = np.sqrt(-kappa)
    arg = kappa * minkowski_dot(x, y)
    arg = np.maximum(arg, 1.0 + 1e-12)
    return np.arccosh(arg) / a

def exp_map_origin(v, kappa=-1.0):
    a = np.sqrt(-kappa)
    r = np.linalg.norm(v, axis=-1, keepdims=True)
    r_safe = np.maximum(r, 1e-12)
    t = (1.0 / a) * np.cosh(a * r)
    s = (np.sinh(a * r) / (a * r_safe)) * v
    return np.concatenate([t, s], axis=-1)

# -----------------------------
# Stimulus grid (uniform 2D)
# -----------------------------
S = 3.0
n = 150
s1 = np.linspace(-S, S, n)
s2 = np.linspace(-S, S, n)
S1, S2 = np.meshgrid(s1, s2)
stim = np.stack([S1.ravel(), S2.ravel()], axis=1)

# Preferred stimulus
u = np.array([1.0, 0.8])

sigma = 1
kappa = -3.0

# -----------------------------
# Euclidean tuning surface
# -----------------------------
dE = np.linalg.norm(stim - u[None, :], axis=1)
lamE = np.exp(-(dE**2)/(2*sigma**2)).reshape(n, n)

# -----------------------------
# Hyperbolic tuning surface
# -----------------------------
z_stim = exp_map_origin(stim, kappa)
z_u = exp_map_origin(u[None, :], kappa)[0]
dH = hyp_dist(z_stim, z_u[None, :], kappa)
lamH = np.exp(-(dH**2)/(2*sigma**2)).reshape(n, n)

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(S1, S2, lamE, cmap="viridis", linewidth=0)
ax1.set_title("Euclidean tuning surface")
ax1.set_xlabel("s₁")
ax1.set_ylabel("s₂")
ax1.set_zlabel("λ(s)")

ax2 = fig.add_subplot(122, projection="3d")
ax2.plot_surface(S1, S2, lamH, cmap="viridis", linewidth=0)
ax2.set_title("Hyperbolic tuning surface")
ax2.set_xlabel("s₁")
ax2.set_ylabel("s₂")
ax2.set_zlabel("λ(s)")

plt.tight_layout()
plt.savefig("tuning_surface_3D.png", dpi=250)
