import os
import random
import numpy as np
import torch
import pandas as pd

from data import get_mnist_loaders
from models import ConstantEncoder, Decoder, VAE
from training_utils import (
    train_one_epoch,
    evaluate_epoch,
    prefix_stats,
    save_checkpoint,
)
from diagnostics import(
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


def run_case(
    case_name,
    mu0,
    logvar0,
    out_dir,
    train_loader,
    val_loader,
    split_info,
    latent_dim=16,
    epochs=20,
    lr=1e-3,
    device=None,
    seed=0,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = ConstantEncoder(latent_dim=latent_dim, mu0=mu0, logvar0=logvar0, learnable=False)
    decoder = Decoder(latent_dim=latent_dim)
    model = VAE(encoder, decoder).to(device)

    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=lr)

    run_dir = os.path.join(out_dir, case_name)
    os.makedirs(run_dir, exist_ok=True)

    config = {
        "protocol": "constant_encoder",
        "case_name": case_name,
        "objective": "beta",
        "beta": 1.0,
        "mu0": mu0,
        "logvar0": logvar0,
        "seed": seed,
        "latent_dim": latent_dim,
        "epochs": epochs,
        "batch_size": split_info["batch_size"],
        "split_seed": split_info["split_seed"],
        "learning_rate": lr,
        "device": str(device),
    }
    save_csv_document(os.path.join(run_dir, "config.csv"), config)
    save_csv_document(os.path.join(run_dir, "split_info.csv"), split_info)

    history = []

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            objective="beta",
            beta=1.0,
        )

        val_stats = evaluate_epoch(
            model,
            val_loader,
            device,
            objective="beta",
            beta=1.0,
        )

        row = {
            "protocol": "constant_encoder",
            "case": case_name,
            "mu0": mu0,
            "logvar0": logvar0,
            "seed": seed,
            "epoch": epoch,
            "epochs_total": epochs,
            "latent_dim": latent_dim,
            "batch_size": split_info["batch_size"],
            "split_seed": split_info["split_seed"],
            **prefix_stats(train_stats, "train"),
            **prefix_stats(val_stats, "val"),
            "tr_M.I.": np.nan,
            "val_M.I.": np.nan
        }
        history.append(row)

        print(case_name, row)

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

    return history_df


def main():
    out_dir = "./results_constant_encoder_controls"
    os.makedirs(out_dir, exist_ok=True)

    epochs = 20
    batch_size = 128
    latent_dim = 16
    lr = 1e-3
    split_seed = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same fixed split reused here too
    train_loader, val_loader, test_loader, split_info = get_mnist_loaders(
        batch_size=batch_size,
        root="./data",
        val_fraction=0.1,
        split_seed=split_seed,
        num_workers=2,
    )

    save_csv_document(os.path.join(out_dir, "split_info.csv"), split_info)

    all_histories = []

    set_seed(0)
    history_zero = run_case(
        case_name="collapse_kl_zero",
        mu0=0.0,
        logvar0=0.0,
        out_dir=out_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        split_info=split_info,
        latent_dim=latent_dim,
        epochs=epochs,
        lr=lr,
        device=device,
        seed=0,
    )
    all_histories.append(history_zero)

    set_seed(0)
    history_small = run_case(
        case_name="collapse_kl_small",
        mu0=0.02,
        logvar0=-0.25,
        out_dir=out_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        split_info=split_info,
        latent_dim=latent_dim,
        epochs=epochs,
        lr=lr,
        device=device,
        seed=0,
    )
    all_histories.append(history_small)

    set_seed(0)
    history_large = run_case(
        case_name="collapse_kl_large",
        mu0=4.0,
        logvar0=0.0,
        out_dir=out_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        split_info=split_info,
        latent_dim=latent_dim,
        epochs=epochs,
        lr=lr,
        device=device,
        seed=0,
    )
    all_histories.append(history_large)

    summary_df = pd.concat(all_histories, ignore_index=True)
    summary_df.to_csv(
        os.path.join(out_dir, "summary_constant_encoder.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
