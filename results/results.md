# Numerical Results

We evaluate posterior collapse using four complementary diagnostics:
validation loss, rate, MI proxy, and latent-intervention sensitivity.

### 1. Similar loss can hide different latent usage

![Loss vs MI](assets/results/fig1_loss_vs_mi.png)

Runs with nearly equal validation loss can still have substantially different MI proxy.
This shows that ELBO / validation loss alone is not enough to determine whether the latent variable is actually being used.

### 2. Increasing beta suppresses latent usage

![Beta trends](assets/results/fig2_beta_trends.png)

As beta increases, both rate and MI proxy decrease, and latent interventions have weaker effect on reconstruction.

### 3. Target-rate helps prevent collapse

![Target rate](assets/results/fig3_target_rate.png)

The target-rate objective keeps the model away from collapse by enforcing nonzero rate.

### 4. KL can be misleading

![Constant encoder controls](assets/results/fig4_constant_encoder_controls.png)

In constant-encoder controls, the encoder is independent of x by construction.
These runs show that KL / rate can be large even when the latent carries essentially no information about the input.

### Representative qualitative examples

| Setting | Reconstruction / intervention |
|---|---|
| Low-MI beta run | ![](assets/results/beta_low_mi_recon.png) |
| High-MI beta run | ![](assets/results/beta_high_mi_recon.png) |
| Target-rate run | ![](assets/results/target_rate_recon.png) |



---



# Posterior collapse implementation part 2: numerical results and figures

This guide is the clean way to turn your existing training outputs into paper-ready numerical evidence.

## What each existing script already gives you

`train_the_models.py` runs a beta sweep over selected `beta` values and seeds, records validation loss, distortion, rate, MI proxy, latent spread, decoder sensitivity, intervention metrics, and prior-generation diagnostics, then saves both per-run histories and a final `summary_across_betas.csv`.

`train_target_rate_models.py` runs the target-rate objective over chosen target rates, penalties, and seeds, records validation loss, distortion, rate, MI proxy, decoder sensitivities, and intervention metrics, then saves a final `summary.csv`.

`run_constant_encoder_controls.py` builds the two key control cases `collapse_kl_zero` and `collapse_kl_large` using the constant encoder, trains only the decoder, and saves per-epoch statistics plus qualitative grids.

The MI / rate / prior-mismatch decomposition and the latent-intervention diagnostics are already implemented in `diagnostics2.py`. In particular, `approx_rate_mi_mismatch_decomposition` returns `rate_proxy`, `mi_proxy`, and `prior_mismatch_proxy`, and `latent_intervention_metrics` measures the effect of zeroing, shuffling, or perturbing the latent.

Your pair-finding utility in `compare_runs.py` is already set up to identify runs with similar validation loss but meaningfully different MI, using relative loss difference plus both relative and absolute MI gaps.

## The strongest claims you can support

1. **Loss alone does not tell you whether the latent is being used.**  
   Use the beta-sweep summary together with `compare_runs.py` to identify same- or near-same-loss runs with different `mi_proxy`.

2. **Increasing `beta` pushes the model toward collapse.**  
   Show that `val_rate`, `mi_proxy`, and intervention sensitivity shrink as `beta` increases.

3. **Target-rate training keeps the model away from collapse by enforcing nonzero rate.**  
   Compare achieved `val_rate` and `mi_proxy` across target rates.

4. **KL / rate is not the same thing as information usage.**  
   The constant-encoder controls prove this cleanly: when `q(z|x)` is independent of `x`, the model can still have large rate due to prior mismatch, even though it is not transmitting information about the input.

## Exact figure set to produce

Use `make_part2_figures.py` after your three training scripts finish.

It generates:

- `fig1_val_loss_vs_mi.png`
- `fig2_beta_trends.png`
- `fig3_mi_vs_intervention.png`
- `fig4_target_rate_results.png`
- `fig5_constant_encoder_controls.png`
- compact summary tables
- a curated folder of representative qualitative grids copied from the original run directories
- a text file with suggested captions / claim mapping

## Suggested main-text narrative

A compact sequence is:

- first show `fig1_val_loss_vs_mi.png` as the core contradiction to “low loss means healthy latent usage”;
- then `fig2_beta_trends.png` to show the collapse trend under larger `beta`;
- then `fig4_target_rate_results.png` to show how target-rate avoids that degeneration;
- finally `fig5_constant_encoder_controls.png` to make the conceptual point that KL can be high even when the encoder carries no information about `x`.

## Minimal workflow

```bash
python train_the_models.py
python train_target_rate_models.py
python run_constant_encoder_controls.py
python make_part2_figures.py
