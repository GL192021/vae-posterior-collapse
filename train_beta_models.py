import os
import random
import numpy as np
import pandas as pd
import torch

from data import get_mnist_loaders
from models import Encoder, Decoder, VAE
from training_utils import (
    train_one_epoch,
    evaluate_epoch,
    prefix_stats,
    save_checkpoint
)
from diagnostics import (
    collect_latents,
    approx_mi_diag_gaussian
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_csv_document(path, obj):
    if isinstance(obj, dict):
        df = pd.DataFrame([obj])
    elif isinstance(obj, list):
        if len(obj) > 0 and all(isinstance(item, dict) for item in obj):
            df = pd.DataFrame(obj)
        else:
            df = pd.DataFrame({"value": obj})
    else:
        df = pd.DataFrame({"value": [obj]})
    df.to_csv(path, index=False)


def main():
    # betas = [5.0, 10.0, 12.0, 15.0, 18.0, 25.0, 30.0, 35.0, 40.0]
    betas = [6.0, 7.0, 8.0, 9.0, 11.0, 13.0, 14.0]
    seeds = [0, 1, 2, 3, 4]
    seeds = [0]
    epochs = 30
    batch_size = 128
    latent_dim = 16
    lr = 1e-3
    split_seed = 0

    out_dir = "./results_mnist_beta_sweep"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Fixed split, reused across all beta runs
    train_loader, val_loader, test_loader, split_info = get_mnist_loaders(
        batch_size=batch_size,
        root="./data",
        val_fraction=0.1,
        split_seed=split_seed,
        num_workers=2,
    )
    print(len(train_loader))
    print(len(val_loader))
    print(len(test_loader))

    save_csv_document(os.path.join(out_dir, "split_info.csv"), split_info)

    all_histories = []

    for beta in betas:
        for seed in seeds:
            set_seed(seed)

            print(f"\n========== Training beta={beta}, seed={seed} ==========\n")

            run_dir = os.path.join(out_dir, f"beta_{beta}_seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)

            config = {
                "protocol": "beta",
                "objective": "beta",
                "beta": beta,
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "latent_dim": latent_dim,
                "learning_rate": lr,
                "split_seed": split_seed,
                "device": str(device),
            }
            save_csv_document(os.path.join(run_dir, "config.csv"), config)
            save_csv_document(os.path.join(run_dir, "split_info.csv"), split_info)

            model = VAE(Encoder(latent_dim=latent_dim), Decoder(latent_dim=latent_dim)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            history = []

            for epoch in range(1, epochs + 1):
                train_stats = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    objective="beta",
                    beta=beta,
                )

                val_stats = evaluate_epoch(
                    model,
                    val_loader,
                    device,
                    objective="beta",
                    beta=beta,
                )

                row = {
                    "protocol": "beta",
                    "beta": beta,
                    "seed": seed,
                    "epoch": epoch,
                    **prefix_stats(train_stats, "train"),
                    **prefix_stats(val_stats, "val"),
                    "tr_M.I.": np.nan,
                    "val_M.I.": np.nan
                }
                history.append(row)

                print(
                    f"Epoch {epoch:02d} | "
                    f"train_loss={row['train_loss']:.4f} | "
                    f"val_loss={row['val_loss']:.4f} || "
                    f"train_rate={row['train_rate']:.4f} | "
                    f"val_rate={row['val_rate']:.4f} || "
                    f"train_distortion={row['train_distortion']:.4f} | "
                    f"val_distortion={row['val_distortion']:.4f} || "
                )

            tr_mus, tr_logvars, _ = collect_latents(model, train_loader, device, max_batches=215)
            tr_mi_proxy = approx_mi_diag_gaussian(tr_mus, tr_logvars)
            history[-1]["tr_M.I."] = tr_mi_proxy
            tr_rate = history[-1]["train_rate"]
            print(f"\nTraining Mutual Information:  {tr_mi_proxy}")
            print(f"KL(q(z) || p(z)) = Rate - M.I. :  {tr_rate - tr_mi_proxy}\n")

            print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

            val_mus, val_logvars, _ = collect_latents(model, val_loader, device, max_batches=30)
            val_mi_proxy = approx_mi_diag_gaussian(val_mus, val_logvars)
            history[-1]["val_M.I."] = val_mi_proxy
            val_rate = history[-1]["val_rate"]
            print(f"\nValidation Mutual Information:  {val_mi_proxy}")
            print(f"KL(q(z) || p(z)) = Rate - M.I. :  {tr_rate - tr_mi_proxy}")
            print(
                "===================================================================================================\n")

            history_df = pd.DataFrame(history)
            history_df.to_csv(os.path.join(run_dir, "epoch_stats.csv"), index=False)

            # Final epoch model is the canonical model for this run
            save_checkpoint(
                os.path.join(run_dir, "final_model.pt"),
                model,
                optimizer,
                epoch=epochs,
                config=config,
                split_info=split_info,
                extra_metrics={"last_epoch": epochs},
            )



            all_histories.append(history_df)

    summary_df = pd.concat(all_histories, ignore_index=True)
    summary_df.to_csv(
        os.path.join(out_dir, "summary_across_betas.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
