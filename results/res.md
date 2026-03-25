# Numerical Results

We evaluate posterior collapse using four complementary diagnostics: validation loss, rate, an approximate mutual-information proxy, and latent-intervention sensitivity.

The figures below summarize the main empirical claims of the repository.

---

## 1. Similar loss can hide very different latent usage

The table below demostrates how different values of $\beta$, seeminlgy producing similar models, can in fact have very different latent encoding quality; compare with the discussion in [[1]](#1).

| Comparison Table | Total Loss | Mutual Information |
|---|---:|---:|
| $\beta=11$ | 185 | 4.86 |
| $\beta=15$ | 199 | 0.629 |

The two models produce comparable total $\beta$-losses: 
- From $\beta=15$ to $\beta=11$, the total loss decreases from 199 to 185, i.e. by 14 units, or approximately 7.0\%. Equivalently, it is about 1.08 times larger at $\beta=15$.
- 
On the other hand the mututal information--and thus the latent ussage--drastically changes:
- From $\beta=15$ to $\beta=11$, the mutual information increases from 0.629 to 4.86, i.e. by about 4.23 units, or approximately 672.7\%. Equivalently, it is about 7.73 times larger at $\beta=11$.

                    


---

![MI proxy versus intervention effect](../figs/fig3_mi_vs_intervention.png)

This figure compares the MI proxy with the reconstruction penalty incurred after modifying the latent variable.

The interpretation is simple: if the decoder is genuinely using the latent code, then changing or zeroing that code should noticeably worsen reconstruction. If the decoder is largely ignoring the latent code, then the reconstruction should change much less.

Accordingly, runs with larger MI proxy also tend to show stronger intervention effects. This provides an additional empirical check that the MI proxy is tracking meaningful latent usage rather than just numerical noise.

---

## 4. Target-rate training helps prevent collapse

![Target-rate results](../figs/fig4_target_rate_results.png)

This figure summarizes the target-rate experiments.

The target-rate objective explicitly penalizes deviation from a prescribed nonzero rate. Empirically, this keeps the model away from the fully collapsed regime more effectively than a plain beta-VAE objective. The achieved validation rate remains nontrivial, and the MI proxy also stays away from zero.

This is the main positive result of the project: target-rate training provides a practical mechanism for preserving latent usage.

---

## 5. KL alone can be misleading

![Constant-encoder controls](../figs/fig5_constant_encoder_controls.png)

This figure shows the constant-encoder control experiments.

These controls are designed so that the encoder is independent of the input. Therefore, the latent variable is not actually carrying meaningful information about \(x\). Nevertheless, the KL or rate term can still become large because of mismatch between the aggregated posterior and the prior.

This is conceptually important: a large KL term is not automatically evidence that the latent representation is informative. It may reflect prior mismatch rather than true information flow from input to latent.

---

## Representative Qualitative Examples

### Low-MI beta run

![Low-MI reconstruction grid](../figs/beta_low_mi_recon_grid.png)

This reconstruction grid comes from a beta-run with relatively weak latent usage. The reconstructions may still look reasonable, but the quantitative diagnostics indicate that the latent variable is playing a limited role.

![Low-MI latent interventions](../figs/beta_low_mi_latent_interventions.png)

The intervention plot shows that modifying the latent code has a comparatively smaller effect, which is consistent with partial or near collapse.

---

### High-MI beta run

![High-MI reconstruction grid](../figs/prior_samples_11.0.png)

This reconstruction grid comes from a beta-run with stronger latent usage.

![High-MI latent interventions](../figs/beta_high_mi_latent_interventions.png)

Here the effect of intervening on the latent code is more substantial. This is consistent with the larger MI proxy and supports the interpretation that this model is using the latent variable more meaningfully.

---

### Target-rate run

![Target-rate reconstruction grid](../figs/target_rate4_recon_grid.png)

This example illustrates a representative target-rate model.

![Target-rate interventions](../figs/target_rate4_interventions.png)

The main point is that the target-rate objective preserves a visibly active latent space while maintaining good reconstructions. Qualitatively and quantitatively, this is the intended contrast with collapse-prone beta-only training.

---

## Constant-Encoder Counterexamples

### Collapse with near-zero KL

![Collapse with near-zero KL](../figs/collapse_kl_zero_recon_grid.png)

This control corresponds to a degenerate setting where the encoder is constant and the KL remains small. It is a useful sanity check: low KL is compatible with collapse, which is the usual textbook warning.

### Collapse with large KL

![Collapse with large KL](../figs/collapse_kl_large_recon_grid.png)

This control is the more interesting counterexample. The encoder is still independent of the input, so the latent code is not informative, yet the KL can be large. This shows that large KL is not sufficient evidence of meaningful latent usage.

---

## Summary

The numerical experiments support four main conclusions:

1. Similar ELBO or validation-loss values do not guarantee similar latent usage.
2. Increasing beta tends to suppress both rate and mutual information.
3. Target-rate training helps keep the model away from collapse by enforcing nonzero rate.
4. KL alone is not a reliable proxy for how much information the latent variable carries about the input.


## References
<a id="1">[1]</a> 
Alexander A. Alemi, Ben Poole, Ian Fischer, Joshua V. Dillon, Rif A. Saurous, and Kevin Murphy. 
**Fixing a Broken ELBO**. 
Proceedings of the 35th International Conference on Machine Learning (ICML), volume 80, pages 159–168, 2018.
