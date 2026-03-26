# Numerical Results

We evaluate posterior collapse using four complementary diagnostics: validation loss, rate, an approximate mutual-information proxy, and latent sensitivity.

The figures below summarize the main empirical claims of the repository.

---

## 1. Similar loss can hide very different latent usage

### Mutual Information
The table below demostrates how different values of $\beta$, seeminlgy producing similar models, can in fact have very different latent encoding quality; compare with the discussion in [[1]](#1).

| Comparison Table | Total Loss | Mutual Information |
|---|---:|---:|
| $\beta=11$ | 185 | 4.86 |
| $\beta=15$ | 199 | 0.629 |

The table above give a numerical example of this phenomenon. 

The two models produce comparable total $\beta$-losses: 
- From $\beta=15$ to $\beta=11$, the total loss decreases from 199 to 185, i.e. by 14 units, or approximately 7.0\%. Equivalently, it is about 1.08 times larger at $\beta=15$.

On the other hand the mututal information---and thus the latent ussage---drastically changes:
- From $\beta=15$ to $\beta=11$, the mutual information increases from 0.629 to 4.86, i.e. by about 4.23 units, or approximately 672.7\%. Equivalently, it is about 7.73 times larger at $\beta=11$.

### Quality of latent encoding
So similar models, in the sense of $\beta$-losses, can actually differ drastically. In the extremes, this difference can be manifested as no-collapse versus collapse. An indicator of collapsing (i.e. usage of latent space), is that the generative aspect of the model performs very poorly, as sampling from the prior is meaningless, since the model does not utilize the latent space. 

<p align="center">
  <img src="../figs/prior_samples_11.0.png" alt="Alt text" width="250"><br>
  <em>sampling from prior $\beta=11$.</em>
</p>
  
<p align="center">
  <img src="../figs/prior_samples_15.0.png" alt="Alt text" width="250"><br>
  <em>sampling from prior $\beta=15$.</em>
</p>





This of course can occure by a prior aggregated posterior mismatch (as we shall see below), therefore we will further investigate the latent usage. For starters, we notice that we do not have a case of mismatch in our examples.

| Comparison Table | KL Loss | $\beta \times$ KL loss | Total Loss |
|---|---:|---:|---:|
| $\beta=11$ | 4.88   | 53.68 | 185 |
| $\beta=15$ | 0.645 | 9.67   | 199 |

More schematically we can see indeed that the the aggregated posterior and the prior are in fact aligned in both cases.

<p align="center">
  <img src="../figs/prior_vs_post__11.png" alt="Alt text" width="300"><br>
</p>

<p align="center">
  <img src="../figs/prior_vs_post__15.png" alt="Alt text" width="300"><br>
</p>


To further illustrate the posterior collapse phenomenon, we are going to study the dependence of reconstruction of a datapoint $x$ on the the latent variable $z$. This should definitetvely show that one model is in fact encoding meaningfully the dataset in the latent space and uses is using the latent information for the reconstruction stage, while the other isn't.

In the figures below, we are seeing the follwing experiment:
- first row: real data
- second row: reconstructions
- thrid row: reconstructions with constant latent $z=0$
- fourth row: reconstructions after shuffling the latents $z_i$ of the batch we considered
- fifth row: reconstruction after randomly perturbing the latent $z$

<p align="center">
  <img src="../figs/latent_interventions__11.png" alt="Alt text" width="250"><br>
  <em>latent quality $\beta = 11$.</em>
</p>
  
<p align="center">
  <img src="../figs/latent_interventions__15.png" alt="Alt text" width="250"><br>
  <em>latent quality $\beta = 15$.</em>
</p>


The bottom line of this section is that, we want of course to make KL small, but "over-prioritizing" with large $\beta$, might hurt the model and even lead to posterior collapse.
---



## 2. KL alone can be misleading: Collapse with large KL

The above discussion might be an indicator that in order to avoid posterior collapse, we only need to keep the KL term (relatively) high. As we shall see, that does not always solve the problem.

To demostrate what could potentially go wrong, even with keeping the KL term away from zero, we will implement some controloed counter examples using a Constant-Encoder. In particular, we are going to see what o posterior collapse might look like with the KL term being (arbitrarily) large. 

This control is the more interesting counterexample. The encoder is still independent of the input, so the latent code is not informative, yet the KL can be large. This shows that large KL is not sufficient evidence of meaningful latent usage.

| Model | KL Loss | Total Loss | Mutual Information |
|---|---:|---:|---:|
| constant encoder with $\mathcal{N}(4,I)$ | 128   | 334 | 0 |

<p align="center">
  <img src="../figs/prior_samples__kl_large.png" alt="Alt text" width="250"><br>
  <em>sampling from prior $N(0,I)$.</em>
</p>

<p align="center">
  <img src="../figs/latent_interventions__kl_large.png" alt="Alt text" width="250"><br>
  <em>latent quality.</em>
</p>

The two figures above show that the quality of latent encoding is extremely low, leading to severe posterior collapse

<p align="center">
  <img src="../figs/prior_vs_post__kl_large.png" alt="Alt text" width="300"><br>
</p>

The final figure semonstrates, that the only reson the KL term is non zero, is because of the extreme misalignment of the two distributions. 


---

## 4. Target-rate run

The main point is that the target-rate objective preserves a visibly active latent space while maintaining good reconstructions. Qualitatively and quantitatively, this is the intended contrast with collapse-prone beta-only training.

---

## Summary

The numerical experiments support four main conclusions:

1. Similar ELBO or validation-loss values do not guarantee similar latent usage.
2. Target-rate training helps keep the model away from collapse.
3. KL alone is not a reliable proxy for how much information the latent variable carries about the input.


## References
<a id="1">[1]</a> 
Alexander A. Alemi, Ben Poole, Ian Fischer, Joshua V. Dillon, Rif A. Saurous, and Kevin Murphy. 
**Fixing a Broken ELBO**. 
Proceedings of the 35th International Conference on Machine Learning (ICML), volume 80, pages 159–168, 2018.
