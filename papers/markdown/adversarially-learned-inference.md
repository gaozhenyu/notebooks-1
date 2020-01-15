### Paper Information
* Title: [Adversarially Learned Inference](https://arxiv.org/pdf/1606.00704.pdf)
* Authors: Vincent Dumoulin, Ishmael Belghazi, Ben Poole, Olivier Mastropietro, Alex Lamb, Martin Arjovsky, Aaron Courville (2017)


### Main Ideas

* The goal is to joinly learn an inference network (encoder) and a generation network (decoder) using an adversarial process. A discriminator is added on top of an encoder-decoder network to discriminate real joint data-latent samples from fake ones.

* The idea is that once the joint data-latent distributions of the encoder and the decoder match up, so do their marginal distributions. The consequences include (1) fake samples are similar to real samples and (2) latent representations follow desired priors such as isotrophic Gaussians.

* The idea is exactly identical to that of [Donahue et al.](https://arxiv.org/pdf/1605.09782.pdf) (BiGAN). The only difference is that the encoder in [Donahue et al.](https://arxiv.org/pdf/1605.09782.pdf) is deterministic.

<img src="images/adversarially-learned-inference/figure-1.png" alt="Drawing" style="width: 60%;"/>


### Adversarially Learned Inference (ALI)

* The generator has an encoder $G_z(x)$ with joint distribution $q(x, z) = q(x) \, q(z |x)$ and a decoder $G_x(z)$ with joint distribution $p(x, z) = p(z) \, p(x | z)$. Sampling from the joint distributions is done by (1) getting samples from the marginal distributions ($q(x)$ is the empirical distribution, $p(z)$ can be made Gaussian) and (2) passing them through the encoder and decoder to get conditional samples from $q(z | x)$ and $p(x | z)$. No densities are involved, as only samples are required.

* The discriminator distinguishes joint pairs $(x, G_z(x))$ and $(G_x(z), z)$ (vanilla GANs distinguish samples from $q(x)$ and $p(x)$).

    $$\min_G \max_D V(D, G) = \mathbb{E}_{q(x)} [\log(D(x, G_z(x)))] + \mathbb{E}_{p(z)}[\log(1 - D(G_x(z), z))].$$

* To fight against gradient vanishing when the the discriminator gets too far ahead, the generator is trained to maximize:

    $$V'(D, G) = \mathbb{E}_{q(x)} [\log(1 - D(x, G_z(x)))] + \mathbb{E}_{p(z)}[\log(D(G_x(z), z))].$$

* Unlike typical autoencoders, no reconstruction loss is added to the objective. The only difference between ALI and vanilla GANs is that the discriminator of ALI also takes latent representations as inputs.

* Training proceeds by alternately updating the discriminator and the generator. As in VAEs, the reparametrization trick is used to compute the gradient of the generator loss with respect to the parameters of $p(z)$.


### Experiments

* For conditional image generation with CIFAR10, SVHN, CelebA, and Tiny ImageNet, there seem to be mismatches during reconstruction.

<img src="images/adversarially-learned-inference/figure-3.png" alt="Drawing" style="width: 80%;"/>

* After training, the encoder $G_z(x)$ can be used to extract features, which can be helpful for classification tasks (semi-supervised learning). The authors train SVM using these features on SVHN and CIFAR10, achieving a misclassification rate of $3.00 \pm 0.50\%$. No feature matching is needed as in [Salimans et al. (2016)](https://arxiv.org/pdf/1606.03498.pdf).

* For conditional image generation, the joint pairs $(x, z)$ is replaced with $(x, z, y)$ where $y$ represents class variables.
<img src="images/adversarially-learned-inference/figure-7.png" alt="Drawing" style="width: 80%;"/>
