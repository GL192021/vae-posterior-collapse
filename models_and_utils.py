import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import defaultdict
from collections import defaultdict
from numbers import Number


## DATA
def get_mnist_loaders(batch_size=128, root="./data"):
    transform = transforms.ToTensor()

    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


## MODELS
class Encoder(nn.Module):
    """
    Standard encoder q_phi(z|x) with diagonal Gaussian posterior.
    Input is flattened MNIST: [B, 784].
    """
    def __init__(self, x_dim=784, h1=512, h2=256, latent_dim=16):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(x_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(h2, latent_dim)
        self.logvar_layer = nn.Linear(h2, latent_dim)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar


class ConstantEncoder(nn.Module):
    """
    Controlled counterexample:
    q_phi(z|x) is independent of x.
    """
    def __init__(self, latent_dim=16, mu0=4.0, logvar0=-8.0, learnable=False):
        super().__init__()

        mu = torch.full((latent_dim,), float(mu0))
        logvar = torch.full((latent_dim,), float(logvar0))

        if learnable:
            self.mu0 = nn.Parameter(mu)
            self.logvar0 = nn.Parameter(logvar)
        else:
            self.register_buffer("mu0", mu)
            self.register_buffer("logvar0", logvar)

    def forward(self, x):
        batch_size = x.size(0)
        mu = self.mu0.unsqueeze(0).expand(batch_size, -1)
        logvar = self.logvar0.unsqueeze(0).expand(batch_size, -1)
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder parameterizing Bernoulli p_theta(x|z).
    Returns the Bernoulli mean x_hat = mu_theta(z) in (0,1)^784.
    """
    def __init__(self, latent_dim=16, h1=256, h2=512, x_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, x_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return {
            "x_hat": x_hat,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }



## LOSSES
def reconstruction_loss(x_hat, x):
    # per-sample Bernoulli negative log-likelihood
    return F.binary_cross_entropy(x_hat, x, reduction="none").sum(dim=1)


def kl_to_standard_normal(mu, logvar):
    # KL(q(z|x) || N(0, I)) for diagonal Gaussian
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1)


def beta_vae_loss(x_hat, x, mu, logvar, beta=1.0):
    recon = reconstruction_loss(x_hat, x)
    kl = kl_to_standard_normal(mu, logvar)

    loss = recon.mean() + beta * kl.mean()
    stats = {
        "loss": loss.item(),
        "distortion": recon.mean().item(),
        "rate": kl.mean().item(),
    }
    return loss, stats


def target_rate_loss(x_hat, x, mu, logvar, target_rate=4.0, lam=100.0, penalty="l1"):
    recon = reconstruction_loss(x_hat, x)
    kl = kl_to_standard_normal(mu, logvar)
    rate = kl.mean()

    if penalty == "l1":
        pen = torch.abs(rate - target_rate)
    elif penalty == "l2":
        pen = (rate - target_rate) ** 2
    elif penalty == "huber":
        pen = F.smooth_l1_loss(rate, torch.tensor(target_rate, device=rate.device))
    else:
        raise ValueError("Unknown penalty")

    loss = recon.mean() + lam * pen
    stats = {
        "loss": loss.item(),
        "distortion": recon.mean().item(),
        "rate": rate.item(),
        "rate_penalty": pen.item(),
    }
    return loss, stats



## TRAINING UTILS
def prefix_stats(stats, prefix):
    return {f"{prefix}_{k}": v for k, v in stats.items()}


def _numeric_stats_only(stats):
    cleaned = {}
    for k, v in stats.items():
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                v = v.item()
            else:
                continue
        if isinstance(v, Number):
            cleaned[k] = float(v)
    return cleaned


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    objective="beta",
    beta=1.0,
    target_rate=4.0,
    lam=100.0,
    penalty="l1",
):
    model.train()
    running = defaultdict(float)
    n_batches = 0

    for x, _ in loader:
        x = x.to(device).view(x.size(0), -1)

        optimizer.zero_grad()
        out = model(x)

        x_hat = out["x_hat"]
        mu = out["mu"]
        logvar = out["logvar"]

        if objective == "beta":
            loss, stats = beta_vae_loss(x_hat, x, mu, logvar, beta=beta)
        elif objective == "target":
            loss, stats = target_rate_loss(
                x_hat, x, mu, logvar,
                target_rate=target_rate, lam=lam, penalty=penalty
            )
        else:
            raise ValueError("Unknown objective")

        loss.backward()
        optimizer.step()

        stats = _numeric_stats_only(stats)
        for k, v in stats.items():
            running[k] += v
        n_batches += 1

    return {k: v / n_batches for k, v in running.items()}


@torch.no_grad()
def evaluate_epoch(
    model,
    loader,
    device,
    objective="beta",
    beta=1.0,
    target_rate=4.0,
    lam=100.0,
    penalty="l1",
):
    model.eval()
    running = defaultdict(float)
    n_batches = 0

    for x, _ in loader:
        x = x.to(device).view(x.size(0), -1)

        out = model(x)

        x_hat = out["x_hat"]
        mu = out["mu"]
        logvar = out["logvar"]

        if objective == "beta":
            loss, stats = beta_vae_loss(x_hat, x, mu, logvar, beta=beta)
        elif objective == "target":
            loss, stats = target_rate_loss(
                x_hat, x, mu, logvar,
                target_rate=target_rate, lam=lam, penalty=penalty
            )
        else:
            raise ValueError("Unknown objective")

        stats = _numeric_stats_only(stats)
        for k, v in stats.items():
            running[k] += v
        n_batches += 1

    return {k: v / n_batches for k, v in running.items()}

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    config,
    split_info,
    extra_metrics=None,
):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "config": config,
        "split_info": split_info,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra_metrics is not None:
        payload["extra_metrics"] = extra_metrics

    torch.save(payload, path)
