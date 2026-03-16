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

# def find_similar_elbo_diff_mi_pairs(csv_path, beta=None, loss_tol_perc=0.01, min_mi_gap_perc=0.2):
#     df = pd.read_csv(csv_path).copy()

#     if beta is not None:
#         df = df[df["beta"] == beta].copy()

#     pairs = []
#     rows = df.to_dict("records")

#     for i in range(len(rows)):
#         for j in range(i + 1, len(rows)):
#             a, b = rows[i], rows[j]

#             if a["beta"] != b["beta"]:
#                 continue

#             if abs(a["val_loss"] - b["val_loss"]) <= loss_tol_perc:
#                 mi_gap = abs(a["mi_proxy"] - b["mi_proxy"])
#                 if mi_gap >= min_mi_gap:
#                     pairs.append({
#                         "beta": a["beta"],
#                         "seed_a": a["seed"],
#                         "seed_b": b["seed"],
#                         "val_loss_a": a["val_loss"],
#                         "val_loss_b": b["val_loss"],
#                         "mi_a": a["mi_proxy"],
#                         "mi_b": b["mi_proxy"],
#                         "mi_gap": mi_gap,
#                     })

#     pairs = sorted(pairs, key=lambda d: -d["mi_gap"])
#     return pd.DataFrame(pairs)

def symmetric_relative_diff(a, b, eps=1e-8):
    scale = max((abs(a) + abs(b)) / 2.0, eps)
    return abs(a - b) / scale


def find_similar_elbo_diff_mi_pairs(
    csv_path,
    beta=None,
    max_loss_rel_diff=0.05,   # e.g. at most 5% loss difference
    min_mi_rel_gap=0.4,      # e.g. at least 40% MI difference
    min_mi_abs_gap=0.2        # also require some absolute MI gap: e.g. 0.01 vs 0.02 is a 67% relative difference, but both are close enough to zero
):
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

            loss_rel_diff = symmetric_relative_diff(a["val_loss"], b["val_loss"])
            mi_abs_gap = abs(a["mi_proxy"] - b["mi_proxy"])
            mi_rel_gap = symmetric_relative_diff(a["mi_proxy"], b["mi_proxy"])

            if loss_rel_diff <= max_loss_rel_diff:
                if mi_abs_gap >= min_mi_abs_gap and mi_rel_gap >= min_mi_rel_gap:
                    pairs.append({
                        "beta": a["beta"],
                        "seed_a": a["seed"],
                        "seed_b": b["seed"],
                        "val_loss_a": a["val_loss"],
                        "val_loss_b": b["val_loss"],
                        "loss_rel_diff": loss_rel_diff,
                        "mi_a": a["mi_proxy"],
                        "mi_b": b["mi_proxy"],
                        "mi_abs_gap": mi_abs_gap,
                        "mi_rel_gap": mi_rel_gap,
                    })

    pairs = sorted(
        pairs,
        key=lambda d: (-d["mi_abs_gap"], d["loss_rel_diff"])
    )
    return pd.DataFrame(pairs)
