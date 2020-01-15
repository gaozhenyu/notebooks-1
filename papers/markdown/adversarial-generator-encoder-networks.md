### Paper Information
* Title:   [It Takes (Only) Two: Adversarial Generator-Encoder Networks](https://arxiv.org/pdf/1704.02304.pdf) <br>
* Authors: Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky (2017)


### Main Ideas

* While autoencoders learn a bidirectional mapping from the latent space to the data space, GANs lack inference models. Hybrid approaches add one or more discriminators on top of the encoder-decoder network, increasing complexity of the model.

* The idea is to simply take the encoder-decoder network and make the encoder act a bit like a discriminator. The encoder tries to match the real data encoded in the latent space to some prior distribution (uniform or Gaussian). Conversely, the generator maximizes the difference between between the fake data encoded in the latent space and that prior.

* It's hypothesized that once the generator matches the real data and the fake data in the latent space, so does it in the data space.

* Reconstruction losses are added to encourage invertibility both in the latent space and in the data space.


### Adversarial Generator-Encoder Networks
* An encoder $e_\psi(x)$ maps the data space $\mathcal{X}$ to the latent space $\mathcal{Z}$ while a decoder $g_\theta(z)$ does the reverse. We're interested in $\Delta(e(X) \, || \, e(g(Z))$, which denotes some divergence between the encoded real data and the encoded fake data.

* It's hard to compute $\Delta(e(X), \, e(g(Z))$ because both $e(X)$ and $e(g(Z))$ are empirical distributions. The adversarial game is therefore written in terms of a reference distribution $Y$ (we can let $Y$ be the prior distribution $Z$):

    $$\max_{e} \min_g V(g, e) = \Delta(e(g(Z)) \, || \, Y) - \Delta(e(X) \, || \, Y).$$

    If the encoder is fixed, the first term helps avoid collapsing $g(Z)$. If the generator is fixed, however, it can lead to $+\infty$ as some divergences are not bounded. In the experiments, the authors compute $\Delta(e(\cdot) \, || \, Y)$ by (1) letting $Y = \mathcal{N}(0, I)$ and (2) fitting an isotropic Gaussian using $e(\cdot)$.

* We also use reconstruction errors to encourage encoder and decoder to be reciprocal, both in the latent space and in the data space:

    \begin{align}
    \mathcal{L}_{\mathcal{X}}(g_\theta, e_\psi) &= \mathbb{E}_{x \sim X} ||x - g_\theta(e_\psi(x)) ||_1,\\
    \mathcal{L}_{\mathcal{Z}}(g_\theta, e_\psi) &= \mathbb{E}_{z \sim Z} ||z - e_\psi(g_\theta(z)) ||_2^2.
    \end{align}

* Combining the divergence and the reconstruction errors, we obtain the objectives for the generator and the decoder:

    \begin{align}
    \hat{\theta} &= \arg \min_{\theta} (V(g_\theta, e_\psi) + \lambda \mathcal{L}_{\mathcal{Z}}(g_\theta, e_\psi)), \\
    \hat{\psi} &= \arg \max_{\psi} (V(g_\theta, e_\psi) - \mu \mathcal{L}_{\mathcal{X}}(g_\theta, e_\psi)),
    \end{align}

    Interestingly, adding $\mathcal{L}_{\mathcal{X}}(g_\theta, e_\psi)$ to the objective of the generator leads to blurry images. In the experiments, $\mu = 10$ and $\lambda \in \{500, 1000, 2000\}$.


### Experiments
* The first experiment is for unconditional image generation. Generator and discriminator are taken from DCGAN, with the discriminator modified to project points onto a sphere. The authors perform 2 generator updates per on encoder update. Inception score on CIFAR-10 is $5.90 \pm 0.04$ (while that for ALI is $5.34 \pm 0.05$).

* The second experiment is for image colorization on Stanford Cars dataset, where they use ResNet-like architectures.
<img src="images/adversarial-generator-encoder-networks/figure-3.png" alt="Drawing" style="width: 70%;"/>
