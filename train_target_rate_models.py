import os
import json
import pandas as pd
import torch

from data import get_mnist_loaders
from models import Encoder, Decoder, VAE
from training_utils import train_one_epoch, evaluate_epoch
from diagnostics2 import (
    collect_latents,
    approx_rate_mi_mismatch_decomposition,
    latent_intervention_metrics,
    decoder_sensitivity,
    local_decoder_sensitivity,
    save_recon_grid,
    save_prior_samples,
    save_latent_intervention_grid,
    save_agg_posterior_vs_prior,
)

def main():
    targets = [2.0, 4.0, 8.0]
    lams = [25.0, 50.0, 100.0]
    seeds = [0, 1, 2, 3, 4]
    out_dir = "./results_mnist_target_rate"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_mnist_loaders(batch_size=128, root="./data")
    latent_dim = 16
    all_runs = []

    for target_rate in targets:
        for lam in lams:
            for seed in seeds:
                torch.manual_seed(seed)

                run_dir = os.path.join(out_dir, f"target_{target_rate}_lam_{lam}_seed_{seed}")
                os.makedirs(run_dir, exist_ok=True)

                model = VAE(Encoder(latent_dim=latent_dim), Decoder(latent_dim=latent_dim)).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                history = []
                for epoch in range(1, 31):
                    train_stats = train_one_epoch(
                        model, train_loader, optimizer, device,
                        objective="target", target_rate=target_rate, lam=lam, penalty="l1"
                    )
                    val_stats = evaluate_epoch(
                        model, test_loader, device,
                        objective="target", target_rate=target_rate, lam=lam, penalty="l1"
                    )

                    mus, logvars, _ = collect_latents(model, test_loader, device, max_batches=40)
                    decomp = approx_rate_mi_mismatch_decomposition(mus, logvars)
                    interv = latent_intervention_metrics(model, test_loader, device, n=64, eps=0.25)

                    row = {
                        "target_rate": target_rate,
                        "lam": lam,
                        "seed": seed,
                        "epoch": epoch,
                        "train_loss": train_stats["loss"],
                        "train_distortion": train_stats["distortion"],
                        "train_rate": train_stats["rate"],
                        "train_rate_penalty": train_stats["rate_penalty"],
                        "val_loss": val_stats["loss"],
                        "val_distortion": val_stats["distortion"],
                        "val_rate": val_stats["rate"],
                        "decoder_sensitivity": decoder_sensitivity(model, latent_dim, device),
                        "local_decoder_sensitivity": local_decoder_sensitivity(model, latent_dim, device),
                        **decomp,
                        **interv,
                    }
                    history.append(row)

                pd.DataFrame(history).to_csv(os.path.join(run_dir, "epoch_stats.csv"), index=False)
                save_recon_grid(model, test_loader, device, os.path.join(run_dir, "recon_grid.png"), n=8)
                save_prior_samples(model, latent_dim, device, os.path.join(run_dir, "prior_samples.png"), n=64)
                save_latent_intervention_grid(model, test_loader, device, os.path.join(run_dir, "interventions.png"), n=8)

                mus, logvars, _ = collect_latents(model, test_loader, device, max_batches=40)
                save_agg_posterior_vs_prior(mus, logvars, os.path.join(run_dir, "agg_post_vs_prior.png"))

                all_runs.append(history[-1])

    pd.DataFrame(all_runs).to_csv(os.path.join(out_dir, "summary.csv"), index=False)

if __name__ == "__main__":
    main()
