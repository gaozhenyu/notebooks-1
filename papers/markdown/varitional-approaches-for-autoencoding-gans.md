### Paper Information

* Title: [Variational Approaches for Auto-Encoding Generative Adversarial Networks](https://arxiv.org/pdf/1706.04987.pdf)
* Authors: Mihaela Rosca, Balaji Lakshminarayanan, David Warde-Farley, Shakir Mohamed (2017)


### Main Ideas

* VAEs have no mode collapse problem but often generate blurry samples. GANs generate sharp samples but suffer from mode collapse. Marrying these two approaches also allow for effective inference and reconstruction.

* GANs use a discriminative network to compute the density ratio between the empirical distribution and the model distribution. We can use the same trick to enrich both terms of ELBO by (1) replacing the explicit likelihood $q(x | z)$ with a synthetic one implemented by a discriminator in the data space and (2) implementing the KL term $\text{KL}[q_\eta(z | x) \, || \, p(z)]$ with another discriminator in the latent space.
<img src="images/varitional-approaches-for-autoencoding-gans/figure-1.png" alt="Drawing" style="width: 60%;"/>


### Density Ratio Trick

* The density ratio between the true distribution $p^*(x)$ and the model distribution $p_\theta(x)$ can be written in terms of a classifer $D_\phi(x)$:

    $$r_\phi(x) = \frac{p^*(x)}{p_\theta(x)} = \frac{p(x | y = 1)}{p(x | y = 0)} = \frac{p(y = 1 | x)}{p(y = 0 | x)} \frac{p(y = 0)}{p(y = 1)} = \frac{p(y = 1 | x)}{p(y = 0 | x)}  = \frac{\mathcal{D}(x)}{1 - \mathcal{D}(x)}. $$

    Here we apply Bayes' rule and assume $p(y = 1) = p(y = 0)$. The trick implies that we don't need access to the analytical forms of $p^*(x)$ and $p_\theta(x)$ to compute their ratio; a classifer that discriminate their samples suffices.


### Fusion of Variational and Adversarial Learning ($\alpha$-GAN)

* We start with a vanilla variational autoencoder comprised of an encoder $\mathcal{E}_\theta(x)$ and a generator $\mathcal{G}_\eta(z)$, which maximizes the variational lower bound (ELBO) denoted by $\mathcal{F}(\theta, \eta)$:

    $$\log p_\theta(x) \geq \mathbb{E}_{q_\eta(z | x)}[ \log p_\theta (x | z)] - \text{KL}[q_\eta(z | x) \, || \, p(z)] = \mathcal{F}(\theta, \eta).$$

* If $p_\theta(x | z)$ is a zero-mean Laplace distribution with scale parameter $\lambda$, i.e. $p_\theta(x | z) \propto \exp(-\lambda \|x - \mathcal{G}_\theta(z)\|_1)$, the first term of ELBO is precisely the $L_1$ reconstruction error. We can  add complexity by using the density ratio trick with a discriminator that discriminates real data from fake data:

    $$\mathbb{E}_{q_\eta(z | x)}[ \log p_\theta (x | z)] =  \mathbb{E}_{q_\eta(z | x)} \left[ \log p^* (x) + \log \left( \frac{p_\theta (x | z)}{ p^*(z)}\right)\right] = \mathbb{E}_{q_\eta(z | x)} \left[ -\lambda \|x - \mathcal{G}_\theta(z)\|_1 + \log \left( \frac{\mathcal{D}_\phi(\mathcal{G}_\theta(z))}{1 - \mathcal{D}_\phi(\mathcal{G}_\theta(z))} \right) \right].$$

    The $L_1$ reconstruction loss prevents mode collapse while the added discriminator allows adversarial training.

* As to the regularization term of ELBO, namely $-\text{KL}[q_\eta(z | x) \, || \, p(z)]$, we make use of the density ratio trick with another discriminator in the latent space as in [Adversarial Autoencoders](https://arxiv.org/pdf/1511.05644.pdf) and [Adversarial Variational Bayes](https://arxiv.org/pdf/1701.04722.pdf):

    $$-\text{KL}[q_\eta(z | x) \, || \, p(z)] = \mathbb{E}_{q_\eta(z | x)} \left[ \log \left( \frac{p_\theta (z)}{q_\eta(z | x)} \right) \right] = \mathbb{E}_{q_\eta(z | x)} \left[ \log\left( \frac{\mathcal{C}_\omega(z)}{1 - \mathcal{C}_\omega(z)} \right) \right].$$

* Putting everything together, we obtain a hybrid objective function:

    $$\mathcal{L}(\theta, \eta) = \mathbb{E}_{q_\eta(z | x)} \left[ -\lambda \|x - \mathcal{G}_\theta(z)\|_1 + \log \left( \frac{\mathcal{D}_\phi(\mathcal{G}_\theta(z))}{1 - \mathcal{D}_\phi(\mathcal{G}_\theta(z))} \right) + \log\left( \frac{\mathcal{C}_\omega(z)}{1 - \mathcal{C}_\omega(z)} \right) \right].$$

    For stronger gradients, the generator loss is modified by replacing $-\log(1 - \mathcal{D}_\phi)$ with $\log(\mathcal{D}_\phi) -\log(1 - \mathcal{D}_\phi)$. Another advice is to let $\mathcal{D}_\phi$ receive both generated samples and reconstructions (as opposed to reconstructions only).


### Experiments

<figure>
    <img src="images/varitional-approaches-for-autoencoding-gans/figure-7.png" alt="Elephant at sunset">
    <figcaption>Generated Samples on CelebA. Top Left: <a href="https://arxiv.org/pdf/1511.06434.pdf">DCGAN</a>. Top Right: <a href="https://arxiv.org/abs/1704.00028">WGAN-GP</a>. Bottom Left: <a href="https://arxiv.org/pdf/1704.02304.pdf">AGE</a>. Bottom Right: <a href="https://arxiv.org/pdf/1706.04987.pdf">$\alpha$-GAN</a></figcaption>
</figure>
