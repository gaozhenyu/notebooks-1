### Paper Information
* Title: [IntroVAE: Introspective Variational Autoencoders for Photographic Image Synthesis](https://arxiv.org/pdf/1807.06358.pdf)
* Authors: Huaibo Huang, Zhihang Li, Ran He∗, Zhenan Sun, Tieniu Tan (2018)


### Main Ideas

* The goal is to combine VAE with adversarial training in such a way that works for high-resolution image generation. The adversarial game in the latent space bears a resemblance to another in [AGE](https://arxiv.org/pdf/1704.02304.pdf), where the encoder tries to make the encoded real data more Gaussian and the encoded fake data less so while the decoder wants the opposite.

* The difference between IntroVAE and AGE is that the former lends itself better to the variational inference framework. The benefits include (1) the distributions of encoded data are Gaussian by design and are also less dependent on batch statistics, and (2) the stochastic decoder can make use of the added noise in inputs and becomes more robust.

<img src="images/introspective-variational-autoencoders/figure-2.png" alt="Drawing" style="width: 70%;"/>


### Introspective Variational Inference (IntroVAE)

* We start with a vanilla variational autoencoder that consists of an encoder $e_\psi(x)$ and a decoder $g_\theta(z)$ and minimizes the variational lower bound:

    $$\log p_\theta(x) \geq \mathbb{E}_{q_\psi(z | x)}[ \log p_\theta (x | z)] - \text{KL}[q_\psi(z | x) \, \| \, p(z)] = \mathcal{F}(\theta, \psi).$$

* The reconstruction term $\mathbb{E}_{q_\psi(z | x)}[ \log p_\theta (x | z)]$ is simply made the pixel-wise mean squared error. This choice is different from [AGE](https://arxiv.org/pdf/1704.02304.pdf) in that (1) no reconstruction error in the latent space is considered and (2) $L_2$ loss is preferred to $L_1$ loss in the data space.

    $$\mathbb{E}_{q_\psi(z | x)}[ \log p_\theta (x | z)] = \mathbb{E}_{x \sim p(x)} ||x - g_\theta(e_\psi(x)) ||_2$$

* Similar to [AGE](https://arxiv.org/pdf/1704.02304.pdf), we design an adversarial game in the latent space that involves the KL term of ELBO, namely $\text{KL}[q_\psi(z | x) \, \| \, p(z)]$:

    $$\max_{e} \min_g V(g, e) = \text{KL}[\mathcal{N}(z; e_\psi(g_\theta(z))) \, || \, p(z)] - \text{KL}[\mathcal{N}(z; e_\psi(x)) \, || \, p(z)].$$

    Here, we write $\mathcal{N}(z; \cdot)$ to denote the typical choice of the approximate posterior $q(z | x)$. The authors modify the objective of the encoder a bit by preventing the KL on $e_\psi(g_\theta(z))$ from getting too positive, simply ignoring it above some positive threshold $m$. Note that the decoder here is stochastic as in vanilla VAE while that of [AGE](https://arxiv.org/pdf/1704.02304.pdf) is deterministic. Also, variational inference conveniently provides explicit means and variances, so there's no need to fit isotropic Gaussian with batch statistics as done in [AGE](https://arxiv.org/pdf/1704.02304.pdf).

* Putting everything together, the loss functions for the encoder and decoder are as follows:

    \begin{align}
    \mathcal{L}_\psi &= \alpha \max(0, m - \text{KL}[\mathcal{N}(z; e_\psi(g_\theta(z))) \, || \, p(z)]) \, + \, \text{KL}(\mathcal{N}(z; e_\psi(x)) \, || \, p(z)) \, + \, \beta \, \mathbb{E}_{x \sim p(x)} \| x - g_\theta(e_\psi(x)) \|_2 \\
    \mathcal{L}_\theta &= \alpha \, \text{KL}[\mathcal{N}(z; e_\psi(g_\theta(z))) \, || \, p(z)]  \, + \, \beta \, \mathbb{E}_{x \sim p(x)} \| x - g_\theta(e_\psi(x)) \|_2,
    \end{align}

    where $m$, $\alpha$ and $\beta$ are hyperparameters. In the experiments, $m \approx 100$, $\alpha = 0.25$, and $\beta \in \{0.0025, 0.05, 0.5\}$. Setting $\alpha = 0$ makes it almost identical to vanilla VAE, which is suggested at the beginning of training.


### Experiments

<img src="images/introspective-variational-autoencoders/figure-3.png" alt="Drawing" style="width: 70%;"/>


The generated samples are pretty high-quality, although they are arguably less impressive than those in [ProGAN](https://arxiv.org/pdf/1710.10196.pdf). The FID score on CelebA-HQ is $5.19$, while the FID score for [ProGAN](https://arxiv.org/pdf/1710.10196.pdf) is $7.30$.
