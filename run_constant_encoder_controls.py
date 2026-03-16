import os
import json
import torch
import pandas as pd

from data import get_mnist_loaders
from models import ConstantEncoder, Decoder, VAE
from training_utils import train_one_epoch, evaluate_epoch
from diagnostics2 import (
    collect_latents,
    approx_mi_diag_gaussian,
    approx_prior_mismatch_diag_gaussian,
    approx_rate_mi_mismatch_decomposition,
    latent_intervention_metrics,
    save_recon_grid,
    save_prior_samples,
    save_latent_intervention_grid,
    save_agg_posterior_vs_prior,
)

def run_case(case_name, mu0, logvar0, out_dir, latent_dim=16, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_mnist_loaders(batch_size=128, root="./data")

    encoder = ConstantEncoder(latent_dim=latent_dim, mu0=mu0, logvar0=logvar0, learnable=False)
    decoder = Decoder(latent_dim=latent_dim)
    model = VAE(encoder, decoder).to(device)

    # train decoder only
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=1e-3)

    run_dir = os.path.join(out_dir, case_name)
    os.makedirs(run_dir, exist_ok=True)

    history = []
    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device,
            objective="beta", beta=1.0
        )
        val_stats = evaluate_epoch(
            model, test_loader, device,
            objective="beta", beta=1.0
        )

        mus, logvars, _ = collect_latents(model, test_loader, device, max_batches=40)
        decomp = approx_rate_mi_mismatch_decomposition(mus, logvars)
        interv = latent_intervention_metrics(model, test_loader, device, n=64, eps=0.25)

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "train_distortion": train_stats["distortion"],
            "val_distortion": val_stats["distortion"],
            **decomp,
            **interv,
        }
        history.append(row)
        print(case_name, row)

    pd.DataFrame(history).to_csv(os.path.join(run_dir, "epoch_stats.csv"), index=False)

    save_recon_grid(model, test_loader, device, os.path.join(run_dir, "recon_grid.png"), n=8)
    save_prior_samples(model, latent_dim, device, os.path.join(run_dir, "prior_samples.png"), n=64)
    save_latent_intervention_grid(model, test_loader, device, os.path.join(run_dir, "interventions.png"), n=8)

    mus, logvars, _ = collect_latents(model, test_loader, device, max_batches=40)
    save_agg_posterior_vs_prior(mus, logvars, os.path.join(run_dir, "agg_post_vs_prior.png"))

def main():
    out_dir = "./results_constant_encoder_controls"
    os.makedirs(out_dir, exist_ok=True)

    run_case("collapse_kl_zero", mu0=0.0, logvar0=0.0, out_dir=out_dir)
    run_case("collapse_kl_large", mu0=4.0, logvar0=-8.0, out_dir=out_dir)

if __name__ == "__main__":
    main()
