import os.path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from losses import reconstruction_loss

from models import *


@torch.no_grad()
def save_recon_grid(model, loader, device, path, n=8):
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(device).view(n, -1)

    out = model(x)
    x_hat = out["x_hat"].view(n, 1, 28, 28)
    x = x.view(n, 1, 28, 28)

    grid = torch.cat([x, x_hat], dim=0)
    save_image(grid, path, nrow=n)


@torch.no_grad()
def save_prior_samples(model, latent_dim, device, path, n=64):
    model.eval()
    z = torch.randn(n, latent_dim, device=device)
    x = model.decoder(z).view(n, 1, 28, 28)
    save_image(x, path, nrow=8)


@torch.no_grad()
def collect_latents(model, loader, device, max_batches=None):
    model.eval()
    mus, logvars, labels = [], [], []
    for i, (x, y) in enumerate(loader):
        x = x.to(device).view(x.size(0), -1)
        mu, logvar = model.encoder(x)
        mus.append(mu.cpu())
        logvars.append(logvar.cpu())
        labels.append(y.cpu())
        if max_batches is not None and i + 1 >= max_batches:
            break
    return torch.cat(mus), torch.cat(logvars), torch.cat(labels)


def mean_rate(mus, logvars):
    kl = 0.5 * (mus.pow(2) + logvars.exp() - 1.0 - logvars).sum(dim=1)
    return kl.mean().item()


@torch.no_grad()
def approx_mi_diag_gaussian(mus, logvars):
    var_x = logvars.exp()
    mean_mu = mus.mean(dim=0)
    agg_var = mus.var(dim=0, unbiased=False) + var_x.mean(dim=0)
    agg_var = torch.clamp(agg_var, min=1e-8)

    kl = 0.5 * (
        (var_x / agg_var).sum(dim=1)
        + ((mus - mean_mu).pow(2) / agg_var).sum(dim=1)
        - mus.size(1)
        + agg_var.log().sum()
        - logvars.sum(dim=1)
    )
    return kl.mean().item()


def latent_spread(mus):
    return mus.std(dim=0, unbiased=False).mean().item()


@torch.no_grad()
def decoder_sensitivity(model, latent_dim, device, n_pairs=256):
    z1 = torch.randn(n_pairs, latent_dim, device=device)
    z2 = torch.randn(n_pairs, latent_dim, device=device)

    p1 = model.decoder(z1)
    p2 = model.decoder(z2)

    return (p1 - p2).abs().mean().item()


@torch.no_grad()
def local_decoder_sensitivity(model, latent_dim, device, n=256, eps=0.25):
    z = torch.randn(n, latent_dim, device=device)
    dz = eps * torch.randn_like(z)

    p1 = model.decoder(z)
    p2 = model.decoder(z + dz)

    return (p1 - p2).abs().mean().item()


def sample_diag_gaussian(mu, logvar):
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


def aggregate_posterior_moments(mus, logvars):
    var_x = logvars.exp()
    mean_mu = mus.mean(dim=0)
    agg_var = mus.var(dim=0, unbiased=False) + var_x.mean(dim=0)
    agg_var = torch.clamp(agg_var, min=1e-8)
    return mean_mu, agg_var


def approx_prior_mismatch_diag_gaussian(mus, logvars):
    mean_mu, agg_var = aggregate_posterior_moments(mus, logvars)
    kl = 0.5 * (agg_var + mean_mu.pow(2) - 1.0 - agg_var.log()).sum()
    return kl.item()


def approx_rate_mi_mismatch_decomposition(mus, logvars):
    rate = mean_rate(mus, logvars)
    mi = approx_mi_diag_gaussian(mus, logvars)
    delta = approx_prior_mismatch_diag_gaussian(mus, logvars)
    return {
        "rate_proxy": rate,
        "mi_proxy": mi,
        "prior_mismatch_proxy": delta,
        "decomp_gap": rate - (mi + delta),
    }


def save_agg_posterior_vs_prior(mus, logvars, path, dims=(0, 1, 2, 3), model_name=None, beta=None, max_points=5000):
    n = min(max_points, mus.size(0))
    mus = mus[:n]
    logvars = logvars[:n]
    z = sample_diag_gaussian(mus, logvars).cpu()

    num_dims = len(dims)
    fig, axes = plt.subplots(1, num_dims, figsize=(4 * num_dims, 3))
    if num_dims == 1:
        axes = [axes]

    for ax, d in zip(axes, dims):
        ax.hist(z[:, d].numpy(), bins=50, density=True, alpha=0.7, label="agg posterior")
        prior = torch.randn(n).numpy()
        ax.hist(prior, bins=50, density=True, alpha=0.5, label="prior")
        if len(dims) > 1:
            ax.set_title(f"latent dim {d}")
        else:
            if beta==None and model_name==None:
                ax.set_title(f"Prior vs Aggregated Posterior")
            elif model_name !=None:
                ax.set_title(f"Prior vs Aggregated Posterior  ({model_name})")
            else:
                ax.set_title(f"Prior vs Aggregated Posterior  (beta={beta})")
        ax.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


@torch.no_grad()
def latent_intervention_metrics(model, loader, device, n=64, eps=0.25):
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(device).view(n, -1)

    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar)

    z_zero = torch.zeros_like(z)
    z_shuffle = z[torch.randperm(n, device=device)]
    z_perturb = z + eps * torch.randn_like(z)

    x_hat = model.decode(z)
    x_hat_zero = model.decode(z_zero)
    x_hat_shuffle = model.decode(z_shuffle)
    x_hat_perturb = model.decode(z_perturb)

    recon = reconstruction_loss(x_hat, x).mean().item()
    recon_zero = reconstruction_loss(x_hat_zero, x).mean().item()
    recon_shuffle = reconstruction_loss(x_hat_shuffle, x).mean().item()
    recon_perturb = reconstruction_loss(x_hat_perturb, x).mean().item()

    return {
        "recon_from_q": recon,
        "recon_from_zero": recon_zero,
        "recon_from_shuffle": recon_shuffle,
        "recon_from_perturb": recon_perturb,
        "delta_recon_zero": recon_zero - recon,
        "delta_recon_shuffle": recon_shuffle - recon,
        "delta_recon_perturb": recon_perturb - recon,
        "decode_change_zero": (x_hat - x_hat_zero).abs().mean().item(),
        "decode_change_shuffle": (x_hat - x_hat_shuffle).abs().mean().item(),
        "decode_change_perturb": (x_hat - x_hat_perturb).abs().mean().item(),
    }


@torch.no_grad()
def save_latent_intervention_grid(model, loader, device, path, n=8, eps=0.25):
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(device).view(n, -1)

    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar)

    z_zero = torch.zeros_like(z)
    z_shuffle = z[torch.randperm(n, device=device)]
    z_perturb = z + eps * torch.randn_like(z)

    x_hat = model.decode(z).view(n, 1, 28, 28)
    x_hat_zero = model.decode(z_zero).view(n, 1, 28, 28)
    x_hat_shuffle = model.decode(z_shuffle).view(n, 1, 28, 28)
    x_hat_perturb = model.decode(z_perturb).view(n, 1, 28, 28)
    x_img = x.view(n, 1, 28, 28)

    grid = torch.cat([x_img, x_hat, x_hat_zero, x_hat_shuffle, x_hat_perturb], dim=0)
    save_image(grid, path, nrow=n)


@torch.no_grad()
def prior_generation_metrics(model, latent_dim, device, n=128):
    model.eval()
    z_prior = torch.randn(n, latent_dim, device=device)
    x_prior = model.decode(z_prior)

    z_zero = torch.zeros_like(z_prior)
    x_zero = model.decode(z_zero)

    z_perm = z_prior[torch.randperm(n, device=device)]
    x_perm = model.decode(z_perm)

    return {
        "prior_decode_pixel_std": x_prior.std(dim=0, unbiased=False).mean().item(),
        "prior_decode_pairwise_l1": (x_prior - x_perm).abs().mean().item(),
        "prior_vs_zero_change": (x_prior - x_zero).abs().mean().item(),
    }
