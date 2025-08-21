"""
OSSE U‑Net prototype: reconstruct gridded ocean currents (u,v) from swath/scattered observations.

What this script does
---------------------
1) Generates synthetic "nature run" vector fields (u,v) from randomized Gaussian-vortex streamfunctions.
2) Samples them along configurable swaths + scattered points to create observations with noise.
3) Builds inputs: [u_obs, v_obs, obs_mask] and targets: [u_true, v_true].
4) Trains a light U‑Net to inpaint/reconstruct the full gridded field.
5) Reports RMSE and optional spectral skill.

Quick start
-----------
$ pip install torch numpy matplotlib
$ python osse_unet_swath2grid.py --epochs 5 --train-samples 2000 --val-samples 200

Notes
-----
- This is a minimal, self-contained baseline. Replace the synthetic generator with
  your model snapshot(s) to turn this into a genuine OSSE.
- You can extend the input with extra channels (e.g., per-pixel uncertainty, distance-to-nearest-obs,
  bathymetry, geostrophic constraints) by modifying ObsToGridDataset.__getitem__ and UNet(in_ch, out_ch).
- For 3D/time, treat time as channels or use a 3D U‑Net.
"""

import argparse
import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ------------------------------------------------------------
#  Utilities
# ------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_torch(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).float()


# ------------------------------------------------------------
#  Synthetic Nature Run (truth) generator
# ------------------------------------------------------------

def make_grid(ny: int, nx: int, Lx: float = 1.0, Ly: float = 1.0):
    """Create a normalized grid in [0, Lx] x [0, Ly]."""
    y = np.linspace(0.0, Ly, ny, dtype=np.float32)
    x = np.linspace(0.0, Lx, nx, dtype=np.float32)
    Y, X = np.meshgrid(y, x, indexing="ij")
    return X, Y


def gaussian_vortex_streamfunction(X, Y, cx, cy, amp, radius):
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    return amp * np.exp(-r2 / (2 * radius ** 2))


def generate_truth_field(ny: int, nx: int, n_vort: int = 5,
                         Lx: float = 1.0, Ly: float = 1.0,
                         amp_range=(0.5, 1.5), r_range=(0.05, 0.2)) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (u,v) from a sum of Gaussian vortices via a streamfunction ψ, with u = -∂ψ/∂y, v = ∂ψ/∂x."""
    X, Y = make_grid(ny, nx, Lx, Ly)
    psi = np.zeros((ny, nx), dtype=np.float32)
    for _ in range(n_vort):
        cx = np.random.uniform(0.1 * Lx, 0.9 * Lx)
        cy = np.random.uniform(0.1 * Ly, 0.9 * Ly)
        amp = np.random.uniform(*amp_range) * random.choice([-1, 1])
        rad = np.random.uniform(*r_range)
        psi += gaussian_vortex_streamfunction(X, Y, cx, cy, amp, rad)

    # Finite-diff gradients (central diff interior, fwd/bwd edges)
    dy = Ly / (ny - 1)
    dx = Lx / (nx - 1)
    dpsi_dy = np.gradient(psi, dy, axis=0)
    dpsi_dx = np.gradient(psi, dx, axis=1)

    u = -dpsi_dy
    v = dpsi_dx

    # Normalize magnitudes for numerical stability
    scale = np.max(np.sqrt(u ** 2 + v ** 2)) + 1e-6
    u /= scale
    v /= scale
    return u.astype(np.float32), v.astype(np.float32)


# ------------------------------------------------------------
#  Observation operator: build swaths + scattered samples
# ------------------------------------------------------------
@dataclass
class ObsConfig:
    swath_width: int = 8          # pixels
    n_swaths: int = 2
    swath_angle_deg: float = 35.0 # orientation of swaths
    swath_gap: int = 32           # pixel gap between parallel swaths
    scattered_frac: float = 0.02  # fraction of extra scattered points
    noise_std: float = 0.03       # observation noise (relative to normalized field)


def make_swath_mask(ny: int, nx: int, angle_deg: float, width: int, gap: int, n_swaths: int) -> np.ndarray:
    """Create multiple parallel diagonal(ish) swaths as a boolean mask."""
    Y, X = np.mgrid[0:ny, 0:nx]
    angle = math.radians(angle_deg)
    # Rotate coordinates
    Xc = X - nx / 2
    Yc = Y - ny / 2
    Xr = Xc * math.cos(angle) + Yc * math.sin(angle)

    mask = np.zeros((ny, nx), dtype=bool)
    # Swaths centered around several Xr positions
    centers = [(-n_swaths // 2 + i) * gap for i in range(n_swaths)]
    for c in centers:
        mask |= (np.abs(Xr - c) <= (width // 2))
    return mask


def add_scattered(mask: np.ndarray, frac: float) -> np.ndarray:
    ny, nx = mask.shape
    n_extra = int(frac * ny * nx)
    idx = np.random.choice(ny * nx, size=n_extra, replace=False)
    flat = mask.flatten()
    flat[idx] = True
    return flat.reshape(mask.shape)


def observe(u: np.ndarray, v: np.ndarray, cfg: ObsConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = u.shape
    swath_mask = make_swath_mask(ny, nx, cfg.swath_angle_deg, cfg.swath_width, cfg.swath_gap, cfg.n_swaths)
    full_mask = add_scattered(swath_mask, cfg.scattered_frac)

    # Add noise on observed points only
    u_obs = np.zeros_like(u)
    v_obs = np.zeros_like(v)
    u_obs[full_mask] = u[full_mask] + np.random.normal(0.0, cfg.noise_std, size=full_mask.sum()).astype(np.float32)
    v_obs[full_mask] = v[full_mask] + np.random.normal(0.0, cfg.noise_std, size=full_mask.sum()).astype(np.float32)

    obs_mask = full_mask.astype(np.float32)
    return u_obs, v_obs, obs_mask


# ------------------------------------------------------------
#  Dataset
# ------------------------------------------------------------
class ObsToGridDataset(Dataset):
    def __init__(self, n_samples: int, ny: int = 128, nx: int = 128, n_vort: int = 6,
                 obs_cfg: ObsConfig = ObsConfig()):
        self.n = n_samples
        self.ny = ny
        self.nx = nx
        self.n_vort = n_vort
        self.obs_cfg = obs_cfg

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Generate truth
        u, v = generate_truth_field(self.ny, self.nx, self.n_vort)
        # Observe
        u_obs, v_obs, mask = observe(u, v, self.obs_cfg)
        # Build input (3-ch) and target (2-ch)
        x = np.stack([u_obs, v_obs, mask], axis=0)  # [3, H, W]
        y = np.stack([u, v], axis=0)               # [2, H, W]
        return to_torch(x), to_torch(y)


# ------------------------------------------------------------
#  U-Net model (lightweight)
# ------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=2, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.outc = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.outc(d1)
        return out


# ------------------------------------------------------------
#  Training / Evaluation
# ------------------------------------------------------------

def rmse(a, b):
    return math.sqrt(float(torch.mean((a - b) ** 2)))


def train_epoch(model, loader, opt, device, lmbd_obs=0.1):
    model.train()
    total = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        # Primary loss: L2 over full field
        loss = F.mse_loss(pred, y)
        # Optional: consistency loss on observed points to anchor training
        # Extract mask from input (channel 2)
        mask = x[:, 2:3, :, :]
        loss_obs = F.mse_loss(pred * mask, y * mask)
        loss = loss + lmbd_obs * loss_obs
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss.detach().cpu()) * x.size(0)
    return total / len(loader.dataset)


def eval_epoch(model, loader, device):
    model.eval()
    tot_rmse_u, tot_rmse_v = 0.0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            tot_rmse_u += rmse(pred[:, 0], y[:, 0]) * x.size(0)
            tot_rmse_v += rmse(pred[:, 1], y[:, 1]) * x.size(0)
    n = len(loader.dataset)
    return tot_rmse_u / n, tot_rmse_v / n


def plot_swaths_and_reconstruction(u_true, v_true, u_obs, v_obs, obs_mask, u_rec=None, v_rec=None, skip=4):
    """
    Visualize the truth, observations, and optional reconstruction.

    Parameters
    ----------
    u_true, v_true : np.ndarray
        True vector field [H, W]
    u_obs, v_obs : np.ndarray
        Observed vector field (zeros outside swaths)
    obs_mask : np.ndarray
        Boolean mask of observed points
    u_rec, v_rec : np.ndarray, optional
        Reconstructed field by the model
    skip : int
        Subsampling for quiver plot for clarity
    """
    ny, nx = u_true.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

    fig, axes = plt.subplots(1, 3 if u_rec is not None else 2, figsize=(18, 6))

    # 1. Truth field
    ax = axes[0]
    ax.set_title("Truth Field")
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              u_true[::skip, ::skip], v_true[::skip, ::skip],
              scale=5, color='k')

    # 2. Observed swaths
    ax = axes[1]
    ax.set_title("Observed Swaths + Scattered Points")
    obs_y, obs_x = np.where(obs_mask)
    ax.quiver(obs_x, obs_y, u_obs[obs_mask == 1], v_obs[obs_mask == 1],
              scale=5, color='r')
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)

    # 3. Reconstructed field (optional)
    if u_rec is not None and v_rec is not None:
        ax = axes[2]
        ax.set_title("Reconstructed Field")
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                  u_rec[::skip, ::skip], v_rec[::skip, ::skip],
                  scale=5, color='b')

    for ax in axes:
        ax.set_aspect('equal')
        ax.invert_yaxis()

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
#  Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--val-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    # Observation config knobs
    parser.add_argument("--swath-width", type=int, default=8)
    parser.add_argument("--n-swaths", type=int, default=2)
    parser.add_argument("--swath-angle", type=float, default=35.0)
    parser.add_argument("--swath-gap", type=int, default=64)
    parser.add_argument("--scattered-frac", type=float, default=0)
    parser.add_argument("--noise-std", type=float, default=0.03)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    obs_cfg = ObsConfig(
        swath_width=args.swath_width,
        n_swaths=args.n_swaths,
        swath_angle_deg=args.swath_angle,
        swath_gap=args.swath_gap,
        scattered_frac=args.scattered_frac,
        noise_std=args.noise_std,
    )

    train_ds = ObsToGridDataset(args.train_samples, args.ny, args.nx, obs_cfg=obs_cfg)
    val_ds = ObsToGridDataset(args.val_samples, args.ny, args.nx, obs_cfg=obs_cfg)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = UNet(in_ch=3, out_ch=2, base=args.base_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = math.inf
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, opt, device)
        rmse_u, rmse_v = eval_epoch(model, val_loader, device)
        score = 0.5 * (rmse_u + rmse_v)
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_RMSE_u={rmse_u:.4f} | val_RMSE_v={rmse_v:.4f}")
        if score < best:
            best = score
            torch.save({
                'model': model.state_dict(),
                'cfg': vars(args),
                'obs_cfg': obs_cfg.__dict__,
            }, 'osse_unet_best.pt')
            print("Saved: osse_unet_best.pt")

    print("Training done. Loading best model for visualization...")
    # Load best model
    ckpt = torch.load('osse_unet_best.pt', map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Generate a single validation sample for plotting
    ran_num = 20
    print(f"plotting {ran_num} samples from validation set")
    for i in range(ran_num):
        ran_idx = random.randrange(args.val_samples)
        print(f"plotting validation sample at index: {ran_idx}")
        val_sample = val_ds[ran_idx]
        x_val, y_val = val_sample
        u_obs, v_obs, mask = x_val[0].numpy(), x_val[1].numpy(), x_val[2].numpy()
        u_true, v_true = y_val[0].numpy(), y_val[1].numpy()

        # Run model on this sample
        with torch.no_grad():
            x_tensor = x_val.unsqueeze(0).to(device)  # add batch dim
            pred = model(x_tensor)
            u_rec = pred[0, 0].cpu().numpy()
            v_rec = pred[0, 1].cpu().numpy()

        # Plot using the function defined earlier
        plot_swaths_and_reconstruction(u_true, v_true, u_obs, v_obs, mask, u_rec, v_rec)

    print("Done.")


if __name__ == "__main__":
    main()
