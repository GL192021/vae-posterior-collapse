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
    save_checkpoint,
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
    targets = [1, 4, 8]
    lams = [0.5, 1.0, 2.0]
    # seeds = [0, 1, 2]
    seeds = [0]
    epochs = 30
    batch_size = 128
    latent_dim = 16
    lr = 1e-3
    split_seed = 0
    penalty = "l2"

    out_dir = "./results_mnist_target_rate"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Fixed split, reused across all target-rate runs
    train_loader, val_loader, test_loader, split_info = get_mnist_loaders(
        batch_size=batch_size,
        root="./data",
        val_fraction=0.1,
        split_seed=split_seed,
        num_workers=2,
    )

    save_csv_document(os.path.join(out_dir, "split_info.csv"), split_info)

    all_histories = []

    for target_rate in targets:
        for lam in lams:
            for seed in seeds:
                set_seed(seed)

                print(
                    f"\n========== Training: target_rate={target_rate}, "
                    f"lambda={lam}, seed={seed} ==========\n"
                )

                run_dir = os.path.join(out_dir, f"target_{target_rate}_lam_{lam}_seed_{seed}")
                os.makedirs(run_dir, exist_ok=True)

                config = {
                    "protocol": "target_rate",
                    "objective": "target_rate",
                    "target_rate": target_rate,
                    "lam": lam,
                    "penalty": penalty,
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
                        objective="target",
                        target_rate=target_rate,
                        lam=lam,
                        penalty=penalty,
                    )

                    val_stats = evaluate_epoch(
                        model,
                        val_loader,
                        device,
                        objective="target",
                        target_rate=target_rate,
                        lam=lam,
                        penalty=penalty,
                    )

                    row = {
                        "protocol": "target_rate",
                        "target_rate": target_rate,
                        "lam": lam,
                        "penalty": penalty,
                        "seed": seed,
                        "epoch": epoch,
                        "epochs_total": epochs,
                        "latent_dim": latent_dim,
                        "batch_size": batch_size,
                        "split_seed": split_seed,
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
                print(f"KL(q(z) || p(z)) = Rate - M.I. :  {tr_rate-tr_mi_proxy}\n")

                print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

                val_mus, val_logvars, _ = collect_latents(model, val_loader, device, max_batches=30)
                val_mi_proxy = approx_mi_diag_gaussian(val_mus, val_logvars)
                history[-1]["val_M.I."] = val_mi_proxy
                val_rate = history[-1]["val_rate"]
                print(f"\nValidation Mutual Information:  {val_mi_proxy}")
                print(f"KL(q(z) || p(z)) = Rate - M.I. :  {val_rate - val_mi_proxy}")
                print("===================================================================================================\n")

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
        os.path.join(out_dir, "summary_target_rate.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
