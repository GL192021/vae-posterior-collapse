import pandas as pd

# def find_similar_elbo_diff_mi_pairs(csv_path, loss_tol=0.5, min_mi_gap=0.5):
#     df = pd.read_csv(csv_path).copy()
#     pairs = []
#
#     rows = df.to_dict("records")
#     for i in range(len(rows)):
#         for j in range(i + 1, len(rows)):
#             a, b = rows[i], rows[j]
#
#             if abs(a["val_loss"] - b["val_loss"]) <= loss_tol:
#                 mi_gap = abs(a["mi_proxy"] - b["mi_proxy"])
#                 if mi_gap >= min_mi_gap:
#                     pairs.append({
#                         "beta_a": a["beta"],
#                         "seed_a": a["seed"],
#                         "beta_b": b["beta"],
#                         "seed_b": b["seed"],
#                         "val_loss_a": a["val_loss"],
#                         "val_loss_b": b["val_loss"],
#                         "mi_a": a["mi_proxy"],
#                         "mi_b": b["mi_proxy"],
#                         "mi_gap": mi_gap,
#                     })
#
#     pairs = sorted(pairs, key=lambda d: -d["mi_gap"])
#     return pd.DataFrame(pairs)

def find_similar_elbo_diff_mi_pairs(csv_path, beta=None, loss_tol_perc=0.01, min_mi_gap_perc=0.2):
    df = pd.read_csv(csv_path).copy()

    if beta is not None:
        df = df[df["beta"] == beta].copy()

    pairs = []
    rows = df.to_dict("records")

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]

            if a["beta"] != b["beta"]:
                continue

            if abs(a["val_loss"] - b["val_loss"]) <= loss_tol_perc:
                mi_gap = abs(a["mi_proxy"] - b["mi_proxy"])
                if mi_gap >= min_mi_gap:
                    pairs.append({
                        "beta": a["beta"],
                        "seed_a": a["seed"],
                        "seed_b": b["seed"],
                        "val_loss_a": a["val_loss"],
                        "val_loss_b": b["val_loss"],
                        "mi_a": a["mi_proxy"],
                        "mi_b": b["mi_proxy"],
                        "mi_gap": mi_gap,
                    })

    pairs = sorted(pairs, key=lambda d: -d["mi_gap"])
    return pd.DataFrame(pairs)
