import numpy as np
import matplotlib.pyplot as plt

def minkowski_dot(x, y):
    return -x[..., 0]*y[..., 0] + np.sum(x[..., 1:]*y[..., 1:], axis=-1)

def hyp_dist(x, y, kappa=-1.0):
    assert kappa < 0
    a = np.sqrt(-kappa)
    # Correct: arg = kappa * <x,y>_L  >= 1
    arg = kappa * minkowski_dot(x, y)
    arg = np.maximum(arg, 1.0 + 1e-12)  # numerical guard
    return np.arccosh(arg) / a

def exp_map_origin(v, kappa=-1.0):
    assert kappa < 0
    a = np.sqrt(-kappa)
    r = np.linalg.norm(v, axis=-1, keepdims=True)     # (...,1)
    r_safe = np.maximum(r, 1e-12)
    t = (1.0/a) * np.cosh(a * r)                      # (...,1)
    s = (np.sinh(a * r) / (a * r_safe)) * v           # (...,d)
    return np.concatenate([t, s], axis=-1)            # (..., d+1)

# -----------------------------
# 1D stimulus visualization
# -----------------------------
kappa = -1.0
sigma = 0.1
S = 3.0

s_grid = np.linspace(-S, S, 500)
v = s_grid[:, None]                  # (N,1)
z = exp_map_origin(v, kappa=kappa)   # (N,2)

u_list = np.array([-2.0, -0.5, 1.0, 2.2])
z_pref = exp_map_origin(u_list[:, None], kappa=kappa)  # (M,2)

lam = []
for zi in z_pref:
    d = hyp_dist(z, zi[None, :], kappa=kappa)          # (N,)
    lam.append(np.exp(-(d**2) / (2*sigma**2)))
lam = np.stack(lam, axis=0)

plt.figure(figsize=(8, 4))
for i in range(lam.shape[0]):
    plt.plot(s_grid, lam[i], color='lime')
plt.xlabel("stimulus s")
plt.ylabel(r"firing rate $\lambda_i(s)$")
plt.title(f"Hyperbolic tuning curves via Exp map (kappa={kappa}, sigma={sigma})")
# plt.legend()
plt.tight_layout()
plt.savefig("tc.png", dpi=200)
plt.close()


# import numpy as np
# import matplotlib.pyplot as plt

# def hyp_to_poincare(x, kappa=-1.0):
#     """
#     Project hyperboloid point x=(x0, x1,...,xd) to Poincaré ball coordinates.
#     For curvature kappa<0 and origin at (1/sqrt(-kappa),0,...,0):
#       p = x_space / (x0 + 1/sqrt(-kappa))
#     """
#     assert kappa < 0
#     a = np.sqrt(-kappa)
#     denom = x[..., 0:1] + 1.0/a
#     return x[..., 1:] / denom

# kappa = -1.0
# sigma = 0.1
# S = 2.5
# grid = 220

# # Uniform 2D stimuli in Euclidean square
# s1 = np.linspace(-S, S, grid)
# s2 = np.linspace(-S, S, grid)
# S1, S2 = np.meshgrid(s1, s2)
# v = np.stack([S1.ravel(), S2.ravel()], axis=1)  # (grid^2, 2)

# # Embed via Exp at origin
# z = exp_map_origin(v, kappa=kappa)  # (grid^2, 3)

# # Choose one preferred stimulus center in the same stimulus coordinates
# u = np.array([1.2, -0.6])
# z_pref = exp_map_origin(u[None, :], kappa=kappa)[0]  # (3,)

# # Compute tuning curve over the grid
# d = hyp_dist(z, z_pref[None, :], kappa=kappa)
# lam = np.exp(-(d**2)/(2*sigma**2)).reshape(grid, grid)

# # For a more "hyperbolic-looking" plot, show it in Poincaré disk coords
# p = hyp_to_poincare(z, kappa=kappa)  # (grid^2,2)
# P1 = p[:, 0].reshape(grid, grid)
# P2 = p[:, 1].reshape(grid, grid)

# plt.figure(figsize=(6, 6))
# plt.pcolormesh(P1, P2, lam, shading="auto")
# plt.gca().set_aspect("equal", "box")
# plt.xlabel("Poincaré x")
# plt.ylabel("Poincaré y")
# plt.title("Hyperbolic place field (heatmap) after Exp map embedding")
# # Draw unit circle boundary of the disk
# theta = np.linspace(0, 2*np.pi, 400)
# plt.plot(np.cos(theta), np.sin(theta), linewidth=1)
# plt.tight_layout()
# plt.savefig("tc2.png")
