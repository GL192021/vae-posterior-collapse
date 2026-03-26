# VAE Posterior Collapse: Rate, Distortion, and Latent Information

This repository is a practical implementation of the ideas in *Fixing a Broken ELBO (Alemi et al., ICML 2018)* [[1]](#1). It shows that ELBO / $\beta$-VAE losses alone do not tell us whether the latent variable is actually being used, and that a *target-rate objective* can force the model away from posterior collapse by selecting solutions with nonzero rate.

---

The central question is not only whether the KL term becomes small, but whether the latent variable $z$ actually carries information about the input $x$.

A VAE balances two competing goals:

- **distortion**: reconstruct the data well,
- **rate**: keep the encoder distribution close to the prior.

This gives the familiar $\beta$-VAE objective


$$
\mathcal{L}_{\beta} = D + \beta R.
$$


A key point is that the rate decomposes as

$$
R = I(X;Z) + \mathrm{KL}(q(z) \Vert p(z)),
$$

where $I(X;Z)$ is the *mutual information*, $p(z)$ is the *prior distribution* and $q(z)$ is the *aggregated posterior*. This decomposition shows that a large KL term is not automatically evidence of a meaningful latent representation. It may come from true information \$I(X;Z)$, or simply from mismatch between the aggregated posterior $q(z)$ and the prior $p(z)$.

In this project, I train MNIST VAEs across a sweep of $\beta$ values and compare them using:

- reconstruction loss,
- KL / rate,
- an approximate mutual-information proxy,
- latent spread,
- decoder sensitivity to the latent variable $z$,
- reconstructions and prior samples.

I also discuss a target-rate objective of the form

$$
L = D + \lambda |R - R^*|,
$$

which penalizes collapse toward zero rate and helps preserve nontrivial latent usage. In addition, I include a **“fake large KL”** counterexample showing that high KL alone does not guarantee an informative latent space.



---



## Mathematical explanation
The `notes/` folder contains the main conceptual contribution of the repository: a concrete mathematical explanation of why the target-rate objective helps discourage posterior collapse, together with a more explicit mechanism-level account of how it favors genuinely informative latent representations over merely inflated KL values for the typical $\beta$-objective case. This point is not developed explicitly in [[1]](#1), and it is one of the main motivations for writing these notes.

---



## Numerical Results and Figures
The file `results/numerical_results.md` contains the numerical results of the current implementation together with figures demonstrating the main claims.


---




## References
<a id="1">[1]</a> 
Alexander A. Alemi, Ben Poole, Ian Fischer, Joshua V. Dillon, Rif A. Saurous, and Kevin Murphy. 
**Fixing a Broken ELBO**. 
Proceedings of the 35th International Conference on Machine Learning (ICML), volume 80, pages 159–168, 2018.
