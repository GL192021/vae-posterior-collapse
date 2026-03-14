# VAE Posterior Collapse: Rate, Distortion, and Latent Information

This project studies posterior collapse in variational autoencoders through the lens of the paper:

```
**Alexander A. Alemi, Ben Poole, Ian Fischer, Joshua V. Dillon, Rif A. Saurous, and Kevin Murphy.**  
*Fixing a Broken ELBO.*  
Proceedings of the 35th International Conference on Machine Learning (ICML), volume 80, pages 159–168, 2018.
```

The central question is not only whether the KL term becomes small, but whether the latent variable \(z\) actually carries information about the input \(x\).

A VAE balances two competing goals:

- **distortion**: reconstruct the data well,
- **rate**: keep the encoder distribution close to the prior.

This gives the familiar \(\beta\)-VAE objective

```
\[
\mathcal{L}_{\beta} = D + \beta R.
\]
```

A key point is that the rate decomposes as

\[
R = I(X;Z) + \mathrm{KL}(q(z)\|p(z)),
\]

so a large KL term is not automatically evidence of a meaningful latent representation. It may come from true information \(I(X;Z)\), or simply from mismatch between the aggregated posterior \(q(z)\) and the prior \(p(z)\).

In this project, I train MNIST VAEs across a sweep of \(\beta\) values and compare them using:

- reconstruction loss,
- KL / rate,
- an approximate mutual-information proxy,
- latent spread,
- decoder sensitivity to the latent variable \(z\),
- reconstructions and prior samples.

I also discuss a target-rate objective of the form

\[
L = D + \lambda |R - R^*|,
\]

which penalizes collapse toward zero rate and helps preserve nontrivial latent usage. In addition, I include a **“fake large KL”** counterexample showing that high KL alone does not guarantee an informative latent space.

## Numerical Results
