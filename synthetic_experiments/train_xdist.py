import os
import copy
import string
import datetime

import math
import random
import numpy as np
from scipy.linalg import cholesky
from scipy.stats import chi2
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import wandb
import pickle
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from distributions import (
    generate_x_distribution,
    generate_o_distribution,
    generate_l_distribution,
    generate_dash_distribution,
    generate_gauss_distribution,
    generate_dot_distribution,
    generate_x_and_gauss_distribution,
    generate_sunshine_distribution,
    generate_stretched_sunshine_distribution,
)


def _infer_type(val: str):
    if val.lower() in ["true", "false"]:
        return val.lower() == "true"
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val


def _parse_wandb_extras(unknown_args):
    """Parses unknown CLI args of the form --wandb.KEY VALUE or --wandb.KEY=VALUE into a dict."""
    extras = {}
    i = 0
    while i < len(unknown_args):
        token = unknown_args[i]
        if isinstance(token, str) and token.startswith("--wandb."):
            # strip prefix
            rest = token[len("--wandb."):]
            if "=" in rest:
                key, raw_val = rest.split("=", 1)
                extras[key] = _infer_type(raw_val)
                i += 1
            else:
                # consume next as value if present and not another flag
                if i + 1 < len(unknown_args) and not str(unknown_args[i + 1]).startswith("-"):
                    extras[rest] = _infer_type(unknown_args[i + 1])
                    i += 2
                else:
                    extras[rest] = True
                    i += 1
        else:
            i += 1
    return extras


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dim", type=int, default=2, help="Dimensionality of the data")
    parser.add_argument(
        "--synthetic_distribution_shapes",
        type=str,
        nargs="+",
        default=["X"],
        choices=["X", "O", "L", "-", "gauss", ".", "x_and_gauss", "sunshine", "stretched_sunshine"],
        help="Shapes of the synthetic data distribution. Can specify multiple.",
    )
    parser.add_argument(
        "--synthetic_distribution_samples",
        type=int,
        nargs="+",
        default=[10000],
        help="Number of samples for each distribution shape. Must have the same number of elements as synthetic_distribution_shapes.",
    )
    parser.add_argument("--x_dist_variance", type=float, default=0.0, help="Variance for the X distribution.")
    parser.add_argument("--x_weight", type=float, default=0.5, help="Mixutre weight for the X distribution.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Small constant for numerical stability")
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--base_lr", type=float, default=5e-3, help="Base learning rate")
    parser.add_argument("--schedule_type", type=str, choices=["cosine", "linear", "constant"], default="cosine", help="lr scheduler")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warm-up steps")
    parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")

    # logging
    parser.add_argument("--num_steps", type=int, default=200000, help="Number of training steps")
    parser.add_argument("--log_interval", type=int, default=1000, help="Logging interval in steps")
    parser.add_argument('--use_wandb', type=eval, choices=[True, False], default=False)
    parser.add_argument('--out_dir', type=str, default="out")

    # loss function weights
    parser.add_argument('--var_loss_weight', type=float, default=25.0)
    parser.add_argument('--cov_loss_weight', type=float, default=1.0)
    parser.add_argument('--radial_ce_loss_weight', type=float, default=100.0)
    parser.add_argument('--radial_ent_loss_weight', type=float, default=100.0)
    parser.add_argument('--embedding_sparsity_loss_weight', type=float, default=0.0)
    parser.add_argument('--e2mc_loss_weight', type=float, default=0.0)
    parser.add_argument('--use_radial_ent_sigmoid', type=eval, choices=[True, False], default=False) # this is for weather we want to squueze all the entropy points to be between 0 and 1 or not

    # radial objective selection
    parser.add_argument(
        "--radial_mode",
        type=str,
        choices=["forward_kl", "reverse_kl", "w1"],
        default="forward_kl",
        help="Choice of radial matching objective: forward_kl (CE - entropy), reverse_kl (KDE), or w1 (1D Wasserstein).",
    )
    parser.add_argument('--radial_revkl_weight', type=float, default=1.0, help="Weight for reverse-KL radial objective")
    parser.add_argument('--radial_w1_weight', type=float, default=1.0, help="Weight for Wasserstein-1 radial objective")
    parser.add_argument('--kde_bandwidth', type=float, default=-1.0, help="If > 0, fixed KDE bandwidth for reverse-KL; otherwise uses Silverman's rule")
    parser.add_argument('--radial_normalize', type=eval, choices=[True, False], default=True, help="Normalize radii by sqrt(D-1) for reverse-KL/W1 computations")
    parser.add_argument('--radial_target_num_samples', type=int, default=512, help="Number of Chi target radii used for reverse-KL/W1 computations")

    # metric calculation
    parser.add_argument("--num_samples_for_metrics", type=int, default=2048, help="Number of samples for expensive (anisotropy,uniformity) metric calculations")

    # wandb
    parser.add_argument('--wandb_project', type=str)
    parser.add_argument('--wandb_entity', type=str)
    parser.add_argument('--wandb_name', type=str, default="")

    parser.add_argument('--num_of_sub_figures', type=int, default=10, help="Number of snapshots to save for the progression plot.")

    args, unknown = parser.parse_known_args()
    # Collect any extra --wandb.KEY args into a dict for logging
    args._wandb_extra = _parse_wandb_extras(unknown)
    return args


class DataDistribution(nn.Module):
    def __init__(self, data_dim, distribution_shapes, distribution_samples, x_dist_variance=0.0, x_weight=0.5):
        super().__init__()

        # assertion of data_dim
        self.data_dim = data_dim
        assert data_dim == 2

        self.distribution_shapes = distribution_shapes
        self.distribution_samples = distribution_samples
        self.x_dist_variance = x_dist_variance
        self.x_weight = x_weight

        if len(self.distribution_shapes) != len(self.distribution_samples):
            raise ValueError(
                "The number of distribution shapes and samples must be the same."
            )

        self.distribution_map = {
            "X": generate_x_distribution,
            "O": generate_o_distribution,
            "L": generate_l_distribution,
            "-": generate_dash_distribution,
            "gauss": generate_gauss_distribution,
            ".": generate_dot_distribution,
            "x_and_gauss": generate_x_and_gauss_distribution,
            "sunshine": generate_sunshine_distribution,
            "stretched_sunshine": generate_stretched_sunshine_distribution,
        }

        for shape in self.distribution_shapes:
            if shape not in self.distribution_map:
                raise ValueError(f"Unknown distribution shape: {shape}")

    def forward(self):
        all_points = []
        for shape, num_samples in zip(
            self.distribution_shapes, self.distribution_samples
        ):
            generation_func = self.distribution_map[shape]
            if shape == "X":
                points = generation_func(num_samples, variance=self.x_dist_variance)
            elif shape == "x_and_gauss":
                points = generation_func(num_samples, x_weight=self.x_weight, variance=self.x_dist_variance)
            else:
                points = generation_func(num_samples)
            all_points.append(points)

        x = torch.cat(all_points, dim=0)
        # shuffle the points
        x = x[torch.randperm(x.size(0))]
        return x


class RadialVCRegLoss(nn.Module):
    def __init__(
        self,
        data_dim,
        var_loss_weight,
        cov_loss_weight,
        radial_ce_loss_weight,
        radial_ent_loss_weight,
        embedding_sparsity_loss_weight,
        e2mc_loss_weight,
        use_radial_ent_sigmoid,
        radial_mode,
        radial_revkl_weight,
        radial_w1_weight,
        kde_bandwidth,
        radial_normalize,
        radial_target_num_samples,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight
        self.radial_ce_loss_weight = radial_ce_loss_weight
        self.radial_ent_loss_weight = radial_ent_loss_weight
        self.embedding_sparsity_loss_weight = embedding_sparsity_loss_weight
        self.e2mc_loss_weight = e2mc_loss_weight
        self.use_radial_ent_sigmoid = use_radial_ent_sigmoid
        self.radial_mode = radial_mode
        self.radial_revkl_weight = radial_revkl_weight
        self.radial_w1_weight = radial_w1_weight
        self.kde_bandwidth = kde_bandwidth
        self.radial_normalize = radial_normalize
        self.radial_target_num_samples = radial_target_num_samples

    def chi2_radial_nll_loss(self, x, eps):
        # data shape
        N, D = x.size()

        # X: [batch_size, 2]
        r = torch.norm(x, dim=1)  # ||x||_2 for each sample, shape [batch_size]

        # Avoid log(0) by clamping small values
        r_safe = torch.clamp(r, min=eps)

        # entropy loss
        radial_ent_loss = self.m_spacing_entropy_loss(r_safe, eps)
        
        # cross entropy loss
        radial_ce_loss = torch.mean(0.5 * (r_safe**2) - (self.data_dim-1)*torch.log(r_safe)) 
        
        return radial_ce_loss, radial_ent_loss

    def m_spacing_entropy_loss(self, x, eps):
        x = x.unsqueeze(1)
        
        if self.use_radial_ent_sigmoid:
            x = torch.sigmoid(x)

        # data shape
        N, D = x.size()
        m = round(math.sqrt(N))

        x_sorted, _ = torch.sort(x, dim=0)
        spacings = x_sorted[m:] - x_sorted[: N - m]
        spacings = spacings * (N + 1) / m

        marginal_ents = torch.log(spacings + eps).sum(dim=0) / (N - m)
        return marginal_ents

    def e2mc_loss(self, x, eps):
        N, D = x.size()
        m = round(math.sqrt(N))

        # sigmoid transformation
        x = torch.sigmoid(x)

        # calculate the marginal entropies
        x_sorted, _ = torch.sort(x, dim=0)
        spacings = x_sorted[m:] - x_sorted[: N - m]
        spacings = spacings * (N + 1) / m
        marginal_ents = torch.log(spacings + eps).sum(dim=0) / (N - m)

        ent_loss = marginal_ents.mean()
        return ent_loss

    def variance_loss(self, x, eps):
        # variance loss
        std_x = torch.sqrt(x.var(axis=0) + eps)
        var_loss = torch.mean(F.relu(1 - std_x))
        return var_loss

    def covariance_loss(self, x):
        N, D = x.size()
        x = x - x.mean(dim=0)
        cov_x = (x.T @ x) / (N - 1)
        diag = torch.eye(D, device=x.device)
        cov_loss = cov_x[~diag.bool()].pow_(2).sum() / D
        return cov_loss 
    
    @staticmethod # static methods can be called without instantciating RadialVicReg loss - we need them in case we call them before using this loss as a method
    def _sample_chi_radii(num_samples: int, dim: int, device, dtype) -> torch.Tensor: # yes inded it is the case that this is a chi distribution and not a chi-squared
        z = torch.randn((num_samples, dim), device=device, dtype=dtype) # z will be a chi distribution
        return torch.norm(z, dim=1)

    def _normalize_radius(self, r: torch.Tensor) -> torch.Tensor: # normalizes the vector r that correspones of all the Radius - we need it for W1 and KL
        if self.radial_normalize:
            scale = math.sqrt(max(self.data_dim - 1, 1))
            return r / scale
        return r

    def _reverse_kl_kde(self, r: torch.Tensor, eps: float) -> torch.Tensor:
        # r: [N] radii from model
        N = r.numel() # the number of elements - r will be a torch.tensor here
        device, dtype = r.device, r.dtype

        # choose number of target samples
        M = min(N, self.radial_target_num_samples) if self.radial_target_num_samples > 0 else N
        u = self._sample_chi_radii(M, self.data_dim, device, dtype)

        # optional normalization #IMPORTANT
        r_used = self._normalize_radius(r)
        u_used = self._normalize_radius(u)

        # bandwidth selection
        if self.kde_bandwidth is not None and self.kde_bandwidth > 0:
            h = torch.tensor(float(self.kde_bandwidth), device=device, dtype=dtype)
        else:
            std = torch.std(r_used)
            std = torch.clamp(std, min=eps)
            h = 1.06 * std * (N ** (-1.0 / 5.0))
            h = torch.clamp(h, min=torch.tensor(eps, device=device, dtype=dtype))

        # Gaussian KDE with standard normal pdf kernel
        # density(u_j) = (1/(N*h)) * sum_i phi((u_j - r_i)/h)
        # where phi(t) = (1/sqrt(2π)) exp(-0.5 t^2)
        diff = (u_used.unsqueeze(1) - r_used.unsqueeze(0)) / h  # [M, N]
        kernel_vals = torch.exp(-0.5 * diff.pow(2)) / math.sqrt(2.0 * math.pi)  # [M, N]
        densities = kernel_vals.mean(dim=1) / h  # [M]
        densities = torch.clamp(densities, min=torch.tensor(eps, device=device, dtype=dtype))
        revkl = -torch.mean(torch.log(densities))
        return revkl

    def _wasserstein1_radius(self, r: torch.Tensor) -> torch.Tensor:
        # r: [N]
        N = r.numel()
        device, dtype = r.device, r.dtype
        K = min(N, self.radial_target_num_samples) if self.radial_target_num_samples > 0 else N
        # subsample if needed
        if K < N:
            idx = torch.randperm(N, device=device)[:K]
            r_used = r[idx]
        else:
            r_used = r
        u = self._sample_chi_radii(r_used.numel(), self.data_dim, device, dtype)

        # optional normalization
        r_used = self._normalize_radius(r_used)
        u = self._normalize_radius(u)

        r_sorted, _ = torch.sort(r_used)
        u_sorted, _ = torch.sort(u)
        w1 = torch.mean(torch.abs(r_sorted - u_sorted))
        return w1

    def forward(self, x, eps):
        # (num_samples, data_dim)
        N, D = x.shape

        # variance loss
        var_loss = self.variance_loss(x, eps)
        
        # covariance loss        
        cov_loss = self.covariance_loss(x)
        
        # Always compute CE and Entropy (for logging/metrics)
        radial_ce_loss, radial_ent_loss = self.chi2_radial_nll_loss(x, eps)

        # Compute radii once
        r = torch.norm(x, dim=1)

        # Choose radial objective
        radial_objective_term = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.radial_mode == 'forward_kl':
            radial_objective_term = (
                self.radial_ce_loss_weight * radial_ce_loss
                - self.radial_ent_loss_weight * radial_ent_loss
            )
        elif self.radial_mode == 'reverse_kl':
            revkl = self._reverse_kl_kde(r, eps)
            radial_objective_term = self.radial_revkl_weight * revkl
        elif self.radial_mode == 'w1':
            w1 = self._wasserstein1_radius(r)
            radial_objective_term = self.radial_w1_weight * w1
        else:
            raise ValueError(f"Unknown radial_mode: {self.radial_mode}")

        # e2mc loss
        e2mc_loss = self.e2mc_loss(x, eps)

        # embedding sparsity "loss" is the negative of the metric
        embed_sparsity_loss = -embedding_sparsity_metric(x, eps)

        # VC Loss
        loss = (
            self.var_loss_weight * var_loss
            + self.cov_loss_weight * cov_loss
            + radial_objective_term
            + self.embedding_sparsity_loss_weight * embed_sparsity_loss
            + self.e2mc_loss_weight * e2mc_loss
        )

        return loss, var_loss, cov_loss, radial_ce_loss, radial_ent_loss, embed_sparsity_loss, e2mc_loss



def distribution_distances_to_standard_normal(
    samples: torch.Tensor,
    ref_samples: torch.Tensor = None,
    gamma: float = 1.0,
):
    import ot  # Import the POT library

    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("Input samples must have shape [batch_size, 2]")

    device = samples.device
    dtype = samples.dtype
    
    # --------------------------------------------------------------------------
    # 1. Moment Matching Distance
    # --------------------------------------------------------------------------
    mu_samples = torch.mean(samples, dim=0)
    
    # torch.cov() is available from PyTorch 1.10.0+
    if hasattr(torch, 'cov'):
        sigma_samples = torch.cov(samples.T)
    else:
        samples_centered = samples - mu_samples
        sigma_samples = (samples_centered.T @ samples_centered) / (samples.shape[0] - 1)

    mu_std = torch.zeros(2, device=device, dtype=dtype)
    sigma_std = torch.eye(2, device=device, dtype=dtype)
    
    mean_dist = torch.norm(mu_samples - mu_std, p=2)
    cov_dist = torch.norm(sigma_samples - sigma_std, p='fro')
    
    moment_matching_dist = (mean_dist, cov_dist)
    
    # --------------------------------------------------------------------------
    # 2. Maximum Mean Discrepancy (MMD)
    # --------------------------------------------------------------------------
    if ref_samples is None:
        ref_samples = torch.randn(samples.shape, device=device, dtype=dtype)

    def rbf_kernel(x, y, gamma):
        dist_matrix = torch.cdist(x, y, p=2).pow(2)
        return torch.exp(-gamma * dist_matrix)

    K_XX = rbf_kernel(samples, samples, gamma=gamma)
    K_YY = rbf_kernel(ref_samples, ref_samples, gamma=gamma)
    K_XY = rbf_kernel(samples, ref_samples, gamma=gamma)

    mmd_squared = torch.mean(K_XX) + torch.mean(K_YY) - 2 * torch.mean(K_XY)
    mmd_dist = torch.sqrt(torch.clamp(mmd_squared, min=0))

    # --------------------------------------------------------------------------
    # 3. Wasserstein Distance (using POT library)
    # --------------------------------------------------------------------------
    # POT functions require NumPy arrays, so we convert the PyTorch tensors.
    samples_np = samples.detach().cpu().numpy()
    ref_samples_np = ref_samples.detach().cpu().numpy()
    
    # We assume uniform weights for both empirical distributions
    a = np.ones(len(samples_np)) / len(samples_np)
    b = np.ones(len(ref_samples_np)) / len(ref_samples_np)

    # Compute the cost matrix between samples
    M = ot.dist(samples_np, ref_samples_np, metric='sqeuclidean')

    # Compute the EMD (Earth Mover's Distance) with POT's solver
    wasserstein_dist_np = np.sqrt(ot.emd2(a, b, M))
    
    # Convert the result back to a PyTorch tensor
    wasserstein_dist = torch.tensor(wasserstein_dist_np, device=device, dtype=dtype)

    return wasserstein_dist, mmd_dist, moment_matching_dist

def generate_out_dir(base_path, run_name):
    # Get current timestamp with precision to seconds
    timestamp_date = datetime.datetime.now().strftime("%Y-%m-%d")
    timestamp_hours = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Generate a random string of 6 characters
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    
    # Combine components to form the directory
    out_dir = os.path.join(base_path, timestamp_date, f"{run_name}_checkpoint_{timestamp_hours}_{random_str}")
    
    return out_dir


def lr_schedule(step, total_steps, base_lr, schedule_type, warmup_steps, final_lr):
    if total_steps <= warmup_steps:
        raise ValueError("total_steps must be greater than warmup_steps.")
    
    if step < warmup_steps:
        # Linear warmup
        return base_lr * ((step + 1) / (warmup_steps + 1))
    
    decay_progress = (step - warmup_steps) / (total_steps - warmup_steps)
    
    if schedule_type == 'cosine':
        cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
        return final_lr + (base_lr - final_lr) * cosine_decay
    elif schedule_type == 'linear':
        return max(final_lr, base_lr * (1 - decay_progress))
    elif schedule_type == 'constant':
        return base_lr
    else:
        raise ValueError(f"Unknown schedule: {schedule_type}")


def anisotropy_loss(x: torch.Tensor) -> torch.Tensor:
    """Computes the anisotropy of a batch of features.
    0 means isotropic (average cosine similarity ~ 0).
    1 means collapsed (all vectors point the same way, average cosine similarity ~ 1).

    Args:
        x (torch.Tensor): NxD Tensor of features.

    Returns:
        torch.Tensor: Anisotropy value.
    """
    N, _ = x.shape
    if N <= 1:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    x_norm = F.normalize(x, dim=1) # we can maybe use our _noramlize function here as well if need be to help out 
    cosine_sim_matrix = x_norm @ x_norm.T

    mask = ~torch.eye(N, dtype=torch.bool, device=x.device)
    
    off_diagonal_cosine_sim = cosine_sim_matrix[mask]
    
    if off_diagonal_cosine_sim.numel() == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
    mean_cosine_sim = off_diagonal_cosine_sim.mean()
    return mean_cosine_sim


def uniform_loss(x, t=2):
    x = F.normalize(x, dim=1)  # normalize to unit sphere
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def embedding_sparsity_metric(embeddings: torch.Tensor, epsilon: float = 1e-12):
    """
    Calculates the embedding sparsity metric S = (1/D) * (||z_row||_1 / ||z_row||_2)^2 for each
    row (embedding) of a PyTorch tensor. This measures how sparsely features are
    activated within each embedding.

    Args:
        embeddings (torch.Tensor): The input tensor of shape (B, D), where B is batch size
                                   (number of embeddings) and D is the number of
                                   dimensions/features per embedding.
        epsilon (float): A small value added to the L2 norm before division to prevent
                         division by zero. Default is 1e-12.

    Returns:
        torch.Tensor: The mean sparsity calculated over the B embedding sparsity values.
    """
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")
    if embeddings.ndim != 2:
        raise ValueError("Input tensor must be 2-dimensional (B x D).")

    B, D = embeddings.shape

    l1_norm_per_row = torch.linalg.norm(embeddings, ord=1, dim=1)
    l2_norm_per_row = torch.linalg.norm(embeddings, ord=2, dim=1)
    l2_norm_per_row_stable = l2_norm_per_row + epsilon
    ratio = l1_norm_per_row / l2_norm_per_row_stable
    metric_per_embedding = (1.0 / D) * (ratio**2)

    return metric_per_embedding.mean()


def hypercovariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Computes the hypercovariance term of the E2MC loss for a single view."""
    N, D = z.size()
    x_hyper = torch.sigmoid(z)
    x_hyper_centered = x_hyper - x_hyper.mean(dim=0)
    cov_x_hyper = (x_hyper_centered.T @ x_hyper_centered) / (N - 1)
    diag = torch.eye(D, device=z.device)
    hypercov_loss = cov_x_hyper[~diag.bool()].pow_(2).sum() / D
    return hypercov_loss


def plot_func(args, original_points, optimized_points, run_name, out_dir, list_of_radial_ce_loss, list_of_radial_ent_loss, list_of_var_loss, list_of_cov_loss, list_of_e2mc_loss, list_of_w_dist, list_of_mmd_dist, list_of_mean_dist, list_of_cov_dist, list_of_lr):
    # Create side-by-side subplots
    fig, axs = plt.subplots(4, 4, figsize=(24, 16)) 
    
    title_text = ""
    if args.wandb_name == "": # if no run name is provided then we will construct one
        # Label
        title_text = run_name
    else:
        title_text = args.wandb_name

    # Orginal Data
    # Calculate the point density (skip KDE if covariance matrix is singular)
    xy_orig = np.vstack([original_points[:, 0], original_points[:, 1]])
    try:
        kde_orig = gaussian_kde(xy_orig)
        density_orig = kde_orig(xy_orig)
        sc_orig = axs[0, 0].scatter(
            original_points[:, 0],
            original_points[:, 1],
            c=density_orig,
            s=10,
            cmap="viridis",
            alpha=0.7,
        )
        fig.colorbar(sc_orig, ax=axs[0, 0]).set_label("Density")
    except np.linalg.LinAlgError:
        # Fallback for degenerate distributions (e.g. all points identical)
        sc_orig = axs[0, 0].scatter(
            original_points[:, 0], original_points[:, 1], s=10, color="blue", alpha=0.7
        )

    axs[0, 0].set_title('Density of Random Data')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')
    axs[0, 0].set_aspect('equal', adjustable='box')
    axs[0, 0].grid(True)

    # optimized data
    # Plot Covariance of original_points
    cov_orig = np.cov(original_points.T)
    axs[0, 1].imshow(cov_orig, cmap='viridis')
    axs[0, 1].set_title('Covariance for the random data points')
    axs[0, 1].set_aspect('equal', adjustable='box')

    # Annotate each cell with the value
    for i in range(cov_orig.shape[0]):
        for j in range(cov_orig.shape[1]):
            axs[0, 1].text(j, i, f'{cov_orig[i, j]:.2f}', ha='center', va='center', color='white')

    # Plot Covariance of optimized_points
    cov_opt = np.cov(optimized_points.T)
    axs[0, 2].imshow(cov_opt, cmap='viridis')
    axs[0, 2].set_title('Covariance for the optimized data points')
    axs[0, 2].set_aspect('equal', adjustable='box')

    # Annotate each cell with the value
    for i in range(cov_opt.shape[0]):
        for j in range(cov_opt.shape[1]):
            axs[0, 2].text(j, i, f'{cov_opt[i, j]:.2f}', ha='center', va='center', color='white')

    # plot learning rate
    axs[0, 3].scatter(np.arange(len(list_of_lr)), list_of_lr) #, alpha=0.5, edgecolors='k', linewidths=0.2)
    axs[0, 3].set_title('Learning Rate')
    axs[0, 3].set_xlabel('Training Steps')
    axs[0, 3].set_ylabel('Learning Rate')
    axs[0, 3].grid(True)

    # Plot optimized_points (handle singular covariance the same way)
    xy_opt = np.vstack([optimized_points[:, 0], optimized_points[:, 1]])
    try:
        # guard against NaNs/Infs
        if np.isfinite(xy_opt).all():
            kde_opt = gaussian_kde(xy_opt)
            density_opt = kde_opt(xy_opt)
            sc_opt = axs[1, 0].scatter(
                optimized_points[:, 0],
                optimized_points[:, 1],
                c=density_opt,
                s=10,
                cmap="viridis",
                alpha=0.7,
            )
            fig.colorbar(sc_opt, ax=axs[1, 0]).set_label("Density")
        else:
            raise ValueError("Non-finite values in optimized points")
    except (np.linalg.LinAlgError, ValueError):
        sc_opt = axs[1, 0].scatter(
            optimized_points[:, 0], optimized_points[:, 1], s=10, color="green", alpha=0.7
        )

    axs[1, 0].set_title(f'{title_text}') # removed the Density of Data trained with part 
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_ylabel('y')
    axs[1, 0].set_aspect('equal', adjustable='box')
    axs[1, 0].grid(True)

    # Plot Radial Loss
    axs[1, 1].plot(np.arange(len(list_of_radial_ce_loss)), list_of_radial_ce_loss) #, alpha=0.5, linewidths=0.2)
    axs[1, 1].set_title(f'Radial CE Loss')
    axs[1, 1].set_xlabel('Logging Steps')
    axs[1, 1].set_ylabel('Radial CE Loss')
    axs[1, 1].grid(True)

    # Plot Radial Entropy Loss
    axs[1, 2].plot(np.arange(len(list_of_radial_ent_loss)), list_of_radial_ent_loss) #, alpha=0.5, linewidths=0.2)
    axs[1, 2].set_title(f'Radial Entropy Loss')
    axs[1, 2].set_xlabel('Logging Steps')
    axs[1, 2].set_ylabel('Radial Entropy Loss')
    axs[1, 2].grid(True)

    # Plot Variance Loss
    axs[1, 3].plot(np.arange(len(list_of_var_loss)), list_of_var_loss) #, alpha=0.5, linewidths=0.2)
    axs[1, 3].set_title(f'Variance Loss')
    axs[1, 3].set_xlabel('Logging Steps')
    axs[1, 3].set_ylabel('Variance Loss')
    axs[1, 2].grid(True)

    # Plot Covariance Loss
    axs[2, 0].plot(np.arange(len(list_of_cov_loss)), list_of_cov_loss) #, alpha=0.5, linewidths=0.2)
    axs[2, 0].set_title(f'Covariance Loss')
    axs[2, 0].set_xlabel('Logging Steps')
    axs[2, 0].set_ylabel('Covariance Loss')
    axs[2, 0].grid(True)

    # Plot E2MC Loss
    axs[2, 1].plot(np.arange(len(list_of_e2mc_loss)), list_of_e2mc_loss) #, alpha=0.5, linewidths=0.2)
    axs[2, 1].set_title(f'E2MC Loss')
    axs[2, 1].set_xlabel('Logging Steps')
    axs[2, 1].set_ylabel('E2MC Loss')
    axs[2, 1].grid(True)

    # Plot distance to standard normal
    axs[2, 2].bar([0, 1], [list_of_w_dist[0], list_of_w_dist[1]], width=0.4)
    axs[2, 2].set_xticks([0, 1])
    axs[2, 2].set_xticklabels(['Original', 'Optimized'])
    axs[2, 2].set_title(f'Wasserstein Distance to Standard Normal')
    axs[2, 2].set_xlabel('Data')
    axs[2, 2].set_ylabel('Wasserstein Distance to Standard Normal')
    axs[2, 2].grid(True)

    # Plot MMD Distance to Standard Normal
    axs[2, 3].bar([0, 1], [list_of_mmd_dist[0], list_of_mmd_dist[1]], width=0.4)
    axs[2, 3].set_xticks([0, 1])
    axs[2, 3].set_xticklabels(['Original', 'Optimized'])
    axs[2, 3].set_title(f'MMD Distance to Standard Normal')
    axs[2, 3].set_xlabel('Data')
    axs[2, 3].set_ylabel('MMD Distance to Standard Normal')
    axs[2, 3].grid(True)

    # Plot Mean Distance to Standard Normal
    axs[3, 0].bar([0, 1], [list_of_mean_dist[0], list_of_mean_dist[1]], width=0.4)
    axs[3, 0].set_xticks([0, 1])
    axs[3, 0].set_xticklabels(['Original', 'Optimized'])
    axs[3, 0].set_title(f'Mean Distance to Standard Normal')
    axs[3, 0].set_xlabel('Data')
    axs[3, 0].set_ylabel('Mean Distance to Standard Normal')
    axs[3, 0].grid(True)

    # Plot Covariance Distance to Standard Normal
    axs[3, 1].bar([0, 1], [list_of_cov_dist[0], list_of_cov_dist[1]], width=0.4)
    axs[3, 1].set_xticks([0, 1])
    axs[3, 1].set_xticklabels(['Original', 'Optimized'])
    axs[3, 1].set_title(f'Covariance Distance to Standard Normal')
    axs[3, 1].set_xlabel('Data')
    axs[3, 1].set_ylabel('Covariance Distance to Standard Normal')
    axs[3, 1].grid(True)


    plt.tight_layout()
    plt.savefig(f"{os.path.join(out_dir,run_name)}.png", dpi=300)
    return fig


def empirical_minimal_chi_nll_without_constant(d):
    """
    Computes the minimal achievable Negative Log-Likelihood (NLL) loss
    when ||z||_2 follows a Chi distribution with `d` degrees of freedom.
    
    Args:
        d (int): Dimension of the feature vector (degrees of freedom).
        
    Returns:
        float: Minimal NLL loss value.
    """
    return -(d-1)*np.log(np.sqrt(d-1)) + (1/2)*(d-1)

def plot_progression(args, progression_snapshots, run_name, out_dir):
    # Cap the number of snapshots to the maximum we can display (num_of_sub_figures)
    snapshots = progression_snapshots[: args.num_of_sub_figures]
    num_snapshots = len(snapshots)
    
    # Create a 4x5 grid for 20 subplots
    fig, axs = plt.subplots(4, 5, figsize=(25, 20))
    plt.suptitle(f'Progression of Data Distribution for {run_name}', fontsize=16)

    for i in range(num_snapshots):
        step, points = snapshots[i]
        
        # Plotting the 2D distribution
        row, col = i // 5, i % 5
        ax_dist = axs[row, col]

        # Guard against non-finite data before attempting KDE
        if np.isfinite(points).all():
            xy = np.vstack([points[:, 0], points[:, 1]])
            try:
                # Use Silverman's rule for bandwidth if not specified, or a small value if singular.
                bw_method = args.kde_bandwidth if args.kde_bandwidth > 0 else 'silverman'
                kde = gaussian_kde(xy, bw_method=bw_method)
                density = kde(xy)
                sc = ax_dist.scatter(points[:, 0], points[:, 1], c=density, s=10, cmap="viridis", alpha=0.7)
                fig.colorbar(sc, ax=ax_dist).set_label("Density")

                # Set axis limits to [2.5, 95.5] percentiles and expand by ±1 for context
                x_low, x_high = np.percentile(points[:, 0], [2.5, 95.5])
                y_low, y_high = np.percentile(points[:, 1], [2.5, 95.5])
                ax_dist.set_xlim(x_low - 1.0, x_high + 1.0)
                ax_dist.set_ylim(y_low - 1.0, y_high + 1.0)
                ax_dist.set_title(f'Step: {step}')

            except (np.linalg.LinAlgError, ValueError):
                # Fallback scatter if KDE fails for any reason (e.g., singular matrix)
                ax_dist.scatter(points[:, 0], points[:, 1], s=10, color="blue", alpha=0.7)
                ax_dist.set_title(f'Step: {step} (KDE failed)')
        else:
            # Handle cases where data has diverged to NaN/Inf
            ax_dist.text(0.5, 0.5, 'Invalid Data (NaN/Inf)', horizontalalignment='center', verticalalignment='center', transform=ax_dist.transAxes)
            ax_dist.set_title(f'Step: {step} (Diverged)')

        ax_dist.set_xlabel('x')
        ax_dist.set_ylabel('y')
        ax_dist.set_aspect('equal', adjustable='box')
        ax_dist.grid(True)

    # Calculate radii for all snapshots and plot histograms
    for i in range(num_snapshots):
        step, points = snapshots[i]
        radii = np.linalg.norm(points, axis=1)
        finite_radii = radii[np.isfinite(radii)]

        # Histograms start from the 11th plot (index 10)
        ax_hist_row, ax_hist_col = (i + 10) // 5, (i + 10) % 5
        ax_hist = axs[ax_hist_row, ax_hist_col]

        if finite_radii.size > 0:
            # Set histogram range: start at 0, end at min(8, 97.5th percentile + 5)
            try:
                radius_cap = float(np.percentile(finite_radii, 97.5)) + 5.0
                radius_max = min(8.0, radius_cap)
            except IndexError:
                radius_max = 8.0 # Fallback
            radius_max = max(radius_max, 1e-6)
            ax_hist.hist(finite_radii, bins=50, density=True, alpha=0.6, label='Empirical', range=(0.0, radius_max))

            # Overlay Chi PDF (not Chi-squared)
            try:
                d = args.data_dim
                r = np.linspace(0.0, radius_max, 200)
                from scipy.stats import chi as _chi
                pdf = _chi.pdf(r, df=d)
                pdf = np.where(np.isfinite(pdf), pdf, 0.0)
                if np.any(np.isfinite(pdf)):
                    ax_hist.plot(r, pdf, 'r-', lw=2, label=f'Chi(df={d})')
            except Exception:
                pass # Fail silently if chi pdf fails
        else:
            ax_hist.text(0.5, 0.5, 'No Finite Radii', horizontalalignment='center', verticalalignment='center', transform=ax_hist.transAxes)

        ax_hist.set_title(f'Radii at Step: {step}')
        ax_hist.set_xlabel('Radius')
        ax_hist.set_ylabel('Density')
        ax_hist.legend(fontsize=8)
        ax_hist.grid(True)

    # Hide any unused subplots (if fewer than 20 snapshots)
    for i in range(num_snapshots, 10):
        row, col = i // 5, i % 5
        axs[row, col].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    progression_fig_path = os.path.join(out_dir, "progression_of_shapes.png")
    plt.savefig(progression_fig_path, dpi=300)
    
    if args.use_wandb:
        wandb.log({"progression_plot": wandb.Image(progression_fig_path)})
    
    plt.close(fig)


def main(args):
    # check that shapes and samples have same length
    if len(args.synthetic_distribution_shapes) != len(
        args.synthetic_distribution_samples
    ):
        raise ValueError("Must provide same number of distribution shapes and samples.")
    
    # Assert mutual exclusivity of radial objective weights
    if args.radial_mode == 'forward_kl':
        # In forward_kl, other radial weights should ideally be zero.
        if args.radial_revkl_weight != 0.0 or args.radial_w1_weight != 0.0:
            print("Warning: forward_kl mode is active, but revkl or w1 weights are non-zero. They will be ignored in the loss calculation but this may indicate a configuration error.")
    elif args.radial_mode == 'reverse_kl':
        if args.radial_ce_loss_weight != 0.0 or args.radial_ent_loss_weight != 0.0 or args.radial_w1_weight != 0.0:
            print("Warning: reverse_kl mode is active, but forward_kl (ce/ent) or w1 weights are non-zero. They will be ignored in the loss calculation but this may indicate a configuration error.")
    elif args.radial_mode == 'w1':
        if args.radial_ce_loss_weight != 0.0 or args.radial_ent_loss_weight != 0.0 or args.radial_revkl_weight != 0.0:
            print("Warning: w1 mode is active, but forward_kl (ce/ent) or revkl weights are non-zero. They will be ignored in the loss calculation but this may indicate a configuration error.")
    
    args.num_samples = sum(args.synthetic_distribution_samples)

    # get run_name
    if args.wandb_name:
        run_name = args.wandb_name
    else:
        dist_str = "_".join(
            [
                f"{shape}-{s}"
                for shape, s in zip(
                    args.synthetic_distribution_shapes, args.synthetic_distribution_samples
                )
            ]
        )

        # Base loss weights (non-radial)
        base_loss_weights = {
            "V": args.var_loss_weight,
            "CV": args.cov_loss_weight,
            "E2MC": args.e2mc_loss_weight,
            "ES": args.embedding_sparsity_loss_weight,
        }

        # Create a list of loss components for the run name
        loss_parts = [f"{name}{int(weight) if weight == float(int(weight)) else weight}" for name, weight in base_loss_weights.items() if weight > 0]
        
        # Add the active radial objective component to the name
        if args.radial_mode == 'forward_kl':
            if args.radial_ce_loss_weight > 0:
                ce_weight = args.radial_ce_loss_weight
                ent_weight = args.radial_ent_loss_weight
                loss_parts.append(f"fkl_ce{int(ce_weight) if ce_weight == float(int(ce_weight)) else ce_weight}_ent{int(ent_weight) if ent_weight == float(int(ent_weight)) else ent_weight}")
        elif args.radial_mode == 'reverse_kl':
            if args.radial_revkl_weight > 0:
                weight = args.radial_revkl_weight
                loss_parts.append(f"revkl{int(weight) if weight == float(int(weight)) else weight}")
        elif args.radial_mode == 'w1':
            if args.radial_w1_weight > 0:
                weight = args.radial_w1_weight
                loss_parts.append(f"w1{int(weight) if weight == float(int(weight)) else weight}")

        run_name_parts = [f"dist_{dist_str}"]
        if loss_parts:
            run_name_parts.append("_".join(loss_parts))
        run_name_parts.append(f"lr{args.base_lr}")
        run_name_parts.append(f"schedule_{args.schedule_type}")

        run_name = "_".join(run_name_parts)

    # output_dir
    out_dir = generate_out_dir(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # random seed
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_seed)
        print(f"Setting Random seeds to {args.random_seed}")
    
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # wandb
    if args.use_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=args)
        # Merge any --wandb.KEY extras into wandb config for easier filtering
        if hasattr(args, "_wandb_extra") and args._wandb_extra:
            wandb.config.update(args._wandb_extra, allow_val_change=True)

    # data distribution
    data_distribution = DataDistribution(
        args.data_dim,
        distribution_shapes=args.synthetic_distribution_shapes,
        distribution_samples=args.synthetic_distribution_samples,
        x_dist_variance=args.x_dist_variance,
        x_weight=args.x_weight,
    )

    optimized_points = data_distribution()
    optimized_points = optimized_points.to(device)
    optimized_points.requires_grad = True

    original_points = optimized_points.detach().cpu().clone().numpy()
    
    # logging
    list_of_radial_ce_loss = []
    list_of_radial_ent_loss = []
    list_of_var_loss = []
    list_of_cov_loss = []
    list_of_lr = []
    list_of_anisotropy = []
    list_of_uniformity = []
    list_of_ent_loss = []
    list_of_hyper_cov_loss = []
    list_of_embed_sparsity_mean = []
    list_of_embed_sparsity_loss = []
    list_of_e2mc_loss = []
    
    # For progression plot
    progression_snapshots = []
    if args.num_of_sub_figures > 0:
        snapshot_steps = set(np.round(np.linspace(0, args.num_steps - 1, args.num_of_sub_figures)).astype(int))
    else:
        snapshot_steps = set()

    # loss & optimizer
    vc_reg_loss = RadialVCRegLoss(
        data_dim=args.data_dim,
        var_loss_weight=args.var_loss_weight,
        cov_loss_weight=args.cov_loss_weight,
        radial_ce_loss_weight=args.radial_ce_loss_weight,
        radial_ent_loss_weight=args.radial_ent_loss_weight,
        embedding_sparsity_loss_weight=args.embedding_sparsity_loss_weight,
        e2mc_loss_weight=args.e2mc_loss_weight,
        use_radial_ent_sigmoid=args.use_radial_ent_sigmoid,
        radial_mode=args.radial_mode,
        radial_revkl_weight=args.radial_revkl_weight,
        radial_w1_weight=args.radial_w1_weight,
        kde_bandwidth=args.kde_bandwidth,
        radial_normalize=args.radial_normalize,
        radial_target_num_samples=args.radial_target_num_samples,
    )

    optimizer = torch.optim.SGD([optimized_points], lr=args.base_lr)
    print(f"##### var_loss_weight ={args.var_loss_weight} | cov_loss_weight={args.cov_loss_weight} | radial_mode={args.radial_mode} | radial_ce_loss_weight = {args.radial_ce_loss_weight} | radial_ent_loss_weight = {args.radial_ent_loss_weight} | radial_revkl_weight = {args.radial_revkl_weight} | radial_w1_weight = {args.radial_w1_weight} | kde_bandwidth = {args.kde_bandwidth} | radial_normalize = {args.radial_normalize} | target_M = {args.radial_target_num_samples} | e2mc_loss_weight = {args.e2mc_loss_weight} | embedding_sparsity_loss_weight = {args.embedding_sparsity_loss_weight}")


    # training loop
    for step in range(args.num_steps):
        # lr schedule
        lr = lr_schedule(step, args.num_steps, args.base_lr, args.schedule_type, args.warmup_steps, args.final_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        list_of_lr.append(lr)

        # Snapshot for progression plot
        if step in snapshot_steps:
            progression_snapshots.append((step, optimized_points.detach().cpu().clone().numpy()))

        # compute loss
        loss, var_loss, cov_loss, radial_ce_loss, radial_ent_loss, embed_sparsity_loss, e2mc_loss = vc_reg_loss(optimized_points, args.eps)

        # constant offset for logging
        radial_ce_loss_for_logging = radial_ce_loss.item()
        radial_ent_loss_for_logging = radial_ent_loss.item()
        embed_sparsity_loss_for_logging = embed_sparsity_loss.item()
        e2mc_loss_for_logging = e2mc_loss.item()
        total_loss_for_logging = loss.item()

        if step % args.log_interval == 0:
            with torch.no_grad():
                # Subsample for expensive metrics
                if args.num_samples > args.num_samples_for_metrics:
                    perm = torch.randperm(args.num_samples, device=device)
                    idx = perm[:args.num_samples_for_metrics]
                    points_subset = optimized_points[idx]
                else:
                    points_subset = optimized_points
                
                # compute additional metrics for logging
                anisotropy = anisotropy_loss(points_subset)
                uniformity = uniform_loss(points_subset)

                # these can run on all points
                hyper_cov_loss = hypercovariance_loss(optimized_points)
                embed_sparsity_mean = -embed_sparsity_loss_for_logging # The loss is the negative of the metric

                # radial metrics across all three variants
                r_all = torch.norm(optimized_points, dim=1)
                forward_kl_est = radial_ce_loss_for_logging - radial_ent_loss_for_logging
                # robustly handle reverse KL on degenerate inputs
                try:
                    if torch.isfinite(r_all).all():
                        reverse_kl_est = vc_reg_loss._reverse_kl_kde(r_all, args.eps).item()
                    else:
                        reverse_kl_est = float('nan')
                except Exception:
                    reverse_kl_est = float('nan')
                w1_radius = vc_reg_loss._wasserstein1_radius(r_all).item()

                empirical_norm = r_all.mean()
                print(f"Step {step} | lr = {lr} | Total Loss = {total_loss_for_logging} | var_loss = {var_loss.item()} | cov_loss = {cov_loss.item()} | radial_ce_loss = {radial_ce_loss_for_logging} | radial_ent_loss = {radial_ent_loss_for_logging} | forward_kl_est = {forward_kl_est:.4f} | reverse_kl_est = {reverse_kl_est} | w1_radius = {w1_radius:.4f} | norm = {empirical_norm} | anisotropy = {anisotropy.item():.4f} | uniformity = {uniformity.item():.4f} | ent_loss = {e2mc_loss_for_logging:.4f} | hyper_cov_loss = {hyper_cov_loss.item():.4f} | embed_sparsity_loss = {embed_sparsity_loss_for_logging:.4f} | embed_sparsity_mean = {embed_sparsity_mean:.4f} | e2mc_loss = {e2mc_loss_for_logging:.4f}")

                list_of_radial_ce_loss.append(radial_ce_loss_for_logging)
                list_of_radial_ent_loss.append(radial_ent_loss_for_logging)
                list_of_var_loss.append(var_loss.item())
                list_of_cov_loss.append(cov_loss.item())
                list_of_anisotropy.append(anisotropy.item())
                list_of_uniformity.append(uniformity.item())
                list_of_ent_loss.append(e2mc_loss_for_logging)
                list_of_hyper_cov_loss.append(hyper_cov_loss.item())
                list_of_embed_sparsity_mean.append(embed_sparsity_mean)
                list_of_embed_sparsity_loss.append(embed_sparsity_loss_for_logging)
                list_of_e2mc_loss.append(e2mc_loss_for_logging)

                if args.use_wandb:
                    wandb.log({
                        "train_step":step,
                        "lr":lr,
                        "empirical_norm": empirical_norm,
                        "train/total_loss":total_loss_for_logging,
                        "train/var_loss": var_loss.item(),
                        "train/cov_loss": cov_loss.item(),
                        "train/radial_ce_loss": radial_ce_loss_for_logging,
                        "train/radial_ent_loss": radial_ent_loss_for_logging,
                        "train/radial_forward_kl": forward_kl_est,
                        "train/radial_reverse_kl": reverse_kl_est,
                        "train/radial_w1": w1_radius,
                        "train/anisotropy_loss": anisotropy.item(),
                        "train/uniformity_loss": uniformity.item(),
                        "train/entropy_loss": e2mc_loss_for_logging,
                        "train/hypercovariance_loss": hyper_cov_loss.item(),
                        "train/embedding_sparsity_loss": embed_sparsity_loss_for_logging,
                        "train/embedding_sparsity_mean": embed_sparsity_mean,
                        "train/e2mc_loss": e2mc_loss_for_logging,
                    })

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    optimized_points_numpy = optimized_points.detach().cpu().numpy()

    # Plot distance to standard normal
    w_dist_orig, mmd_dist_orig, (mean_dist_orig, cov_dist_orig) = distribution_distances_to_standard_normal(torch.from_numpy(original_points))
    # Guard against non-finite values before measuring distances
    opt_pts = optimized_points_numpy
    if not np.isfinite(opt_pts).all():
        # replace non-finite with zeros to avoid crashing metrics; still useful to log
        opt_pts = np.nan_to_num(opt_pts, nan=0.0, posinf=0.0, neginf=0.0)
    w_dist_opt, mmd_dist_opt, (mean_dist_opt, cov_dist_opt) = distribution_distances_to_standard_normal(torch.from_numpy(opt_pts))

    if args.use_wandb:
        wandb.log({
            "w_dist_orig": w_dist_orig.item(),
            "w_dist_opt": w_dist_opt.item(),
            "mmd_dist_orig": mmd_dist_orig.item(),
            "mmd_dist_opt": mmd_dist_opt.item(),
            "mean_dist_orig": mean_dist_orig.item(),
            "mean_dist_opt": mean_dist_opt.item(),
            "cov_dist_orig": cov_dist_orig.item(),
            "cov_dist_opt": cov_dist_opt.item(),
        })

    list_of_w_dist = [w_dist_orig, w_dist_opt]
    list_of_mmd_dist = [mmd_dist_orig, mmd_dist_opt]
    list_of_mean_dist = [mean_dist_orig, mean_dist_opt]
    list_of_cov_dist = [cov_dist_orig, cov_dist_opt]

    # print the list of w_dist, mmd_dist, mean_dist, cov_dist
    print(f"w_dist = {list_of_w_dist} | mmd_dist = {list_of_mmd_dist} | mean_dist = {list_of_mean_dist} | cov_dist = {list_of_cov_dist}") 
    print(f"var_loss_weight = {args.var_loss_weight} | cov_loss_weight = {args.cov_loss_weight} | radial_ce_loss_weight = {args.radial_ce_loss_weight} | radial_ent_loss_weight = {args.radial_ent_loss_weight} | e2mc_loss_weight = {args.e2mc_loss_weight} | embedding_sparsity_loss_weight = {args.embedding_sparsity_loss_weight} | cov = {np.cov(optimized_points_numpy.T)}")

    # save statistics for plotting
    statistics_for_plotting = {
        "args": args,
        "list_of_radial_ce_loss": list_of_radial_ce_loss,
        "list_of_radial_ent_loss": list_of_radial_ent_loss,
        "list_of_var_loss": list_of_var_loss,
        "list_of_cov_loss": list_of_cov_loss,
        "list_of_e2mc_loss": list_of_e2mc_loss,
        "list_of_w_dist": list_of_w_dist,
        "list_of_mmd_dist": list_of_mmd_dist,
        "list_of_mean_dist": list_of_mean_dist,
        "list_of_cov_dist": list_of_cov_dist,
        "list_of_lr": list_of_lr,
        "progression_snapshots": progression_snapshots,
        "original_points": original_points,
        "optimized_points": optimized_points_numpy,
    }

    # save the statistics for plotting
    with open(os.path.join(out_dir, "statistics_for_plotting.pkl"), "wb") as f:
        pickle.dump(statistics_for_plotting, f)

    # plotting
    if args.data_dim == 2:
        fig = plot_func(args, original_points, optimized_points_numpy, run_name, out_dir, list_of_radial_ce_loss, list_of_radial_ent_loss, list_of_var_loss, list_of_cov_loss, list_of_e2mc_loss, list_of_w_dist, list_of_mmd_dist, list_of_mean_dist, list_of_cov_dist, list_of_lr)
        if args.use_wandb:
            wandb.log({"results_plot": wandb.Image(fig)})
            plt.close(fig)
        
        # Add the final snapshot before plotting progression
        if args.num_of_sub_figures > 0 and args.num_steps not in snapshot_steps and len(progression_snapshots) < args.num_of_sub_figures:
            progression_snapshots.append((args.num_steps, optimized_points_numpy))
        
        if progression_snapshots:
            plot_progression(args, progression_snapshots, run_name, out_dir)
    else:
        print("skip plotting")

if __name__ == "__main__":
    args = get_args()
    main(args)

