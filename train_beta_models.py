import os
import json
import pandas as pd
import torch

from data import get_mnist_loaders
from models import Encoder, Decoder, VAE
from training_utils import train_one_epoch, evaluate_epoch
from diagnostics2 import (
    collect_latents,
    latent_spread,
    decoder_sensitivity,
    local_decoder_sensitivity,
    approx_rate_mi_mismatch_decomposition,
    latent_intervention_metrics,
    prior_generation_metrics,
    save_recon_grid,
    save_prior_samples,
    save_latent_intervention_grid,
    save_agg_posterior_vs_prior,
)

def main():
    # betas = [0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 20.0, 25.0, 30.0, 35.0]
    ### In the same spitit of the Alemi et al., we search for betas with same beta-losses but different M.I.
    betas = [1, 10, 30]
    seeds = [0, 1, 2, 3, 4]

    out_dir = "./results_mnist_beta_sweep"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train_loader, test_loader = get_mnist_loaders(batch_size=128, root="./data")
    latent_dim = 16

    all_runs_summary = []

    for beta in betas:
        for seed in seeds:
            torch.manual_seed(seed)

            print(f"\n========== Training beta={beta}, seed={seed} ==========\n")

            run_dir = os.path.join(out_dir, f"beta_{beta}_seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)

            model = VAE(Encoder(latent_dim=latent_dim), Decoder(latent_dim=latent_dim)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            history = []

            for epoch in range(1, 31):
                train_stats = train_one_epoch(
                    model, train_loader, optimizer, device,
                    objective="beta", beta=beta
                )
                val_stats = evaluate_epoch(
                    model, test_loader, device,
                    objective="beta", beta=beta
                )

                mus, logvars, _ = collect_latents(model, test_loader, device, max_batches=40)
                spread = latent_spread(mus)
                dec_sens = decoder_sensitivity(model, latent_dim, device)
                local_dec_sens = local_decoder_sensitivity(model, latent_dim, device)

                decomp = approx_rate_mi_mismatch_decomposition(mus, logvars)
                interv = latent_intervention_metrics(model, test_loader, device, n=64, eps=0.25)
                prior_gen = prior_generation_metrics(model, latent_dim, device, n=128)

                row = {
                    "beta": beta,
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": train_stats["loss"],
                    "train_distortion": train_stats["distortion"],
                    "train_rate": train_stats["rate"],
                    "val_loss": val_stats["loss"],
                    "val_distortion": val_stats["distortion"],
                    "val_rate": val_stats["rate"],
                    "latent_spread": spread,
                    "decoder_sensitivity": dec_sens,
                    "local_decoder_sensitivity": local_dec_sens,
                    **decomp,
                    **interv,
                    **prior_gen,
                }
                history.append(row)

                mi_proxy = decomp["mi_proxy"]

                print(
                    f"Epoch {epoch:02d} | "
                    f"train_loss={train_stats['loss']:.4f} | "
                    f"val_loss={val_stats['loss']:.4f} || "
                    f"train_rate = {train_stats['rate']:.4f} | "
                    f"val_rate = {val_stats['rate']:.4f} || "
                    f"train_distortion = {train_stats['distortion']:.4f} | "
                    f"val_distortion = {val_stats['distortion']:.4f} || "
                    f"test_MI_proxy={mi_proxy:.4f} | "
                    f"spread={spread:.4f} | "
                    f"dec_sens={dec_sens:.4f} | "
                    f"local_dec_sens={local_dec_sens:.4f}"
                )

            pd.DataFrame(history).to_csv(os.path.join(run_dir, "epoch_stats.csv"), index=False)



            with open(os.path.join(run_dir, "epoch_stats.json"), "w") as f:
                json.dump(history, f, indent=2)

            save_recon_grid(model, test_loader, device, os.path.join(run_dir, "recon_grid.png"), n=8)
            save_prior_samples(model, latent_dim, device, os.path.join(run_dir, "prior_samples.png"), n=64)
            save_latent_intervention_grid(model, test_loader, device, os.path.join(run_dir, "latent_interventions.png"), n=8)

            mus, logvars, _ = collect_latents(model, test_loader, device, max_batches=40)
            save_agg_posterior_vs_prior(mus, logvars, os.path.join(run_dir, "agg_post_vs_prior.png"))

            torch.save(
                {
                    "beta": beta,
                    "seed": seed,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                },
                os.path.join(run_dir, "final_model.pt")
            )

            all_runs_summary.append(history[-1])

    pd.DataFrame(all_runs_summary).to_csv(
        os.path.join(out_dir, "summary_across_betas.csv"),
        index=False
    )

if __name__ == "__main__":
    main()
