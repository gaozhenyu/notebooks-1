### Minimax Games

Consider a two-player game $\min_{\mathbf{x}} \max_{\mathbf{y}} f(\mathbf{x}, \mathbf{y})$ where $f: \mathbb{R}^m \times \mathbb{R}^n \rightarrow \mathbb{R}$ is twice continuously differentiable. An interesting example is generative adversarial networks (GANs):

$$\min_{G} \max_{D} \, \mathbb{E}_{x \sim p_{\text{data}}} \left[ \log(D(x)) \right] + \mathbb{E}_{z \sim p_{\text{latent}}} \left[ \log(1 - D(G(z))) \right].$$

Here $\mathbf{x}$ represents the parameters of the generator $G$ which aims to transform latent vectors $z \sim p_{\text{latent}}$ into samples $G(z)$ that mimic those from a given data distribution $p_{\text{data}}$, and $\mathbf{y}$ represents the parameters of the discriminator $D$ which minimizes a [log loss](http://wiki.fast.ai/index.php/Log_Loss) to differentiate between $G(z)$ and $x \sim p_{\text{data}}$. As in the case of GANs, the players of minimax games are often parametrized by neural networks, and the loss functions $f(\mathbf{x}, \mathbf{y})$ are not necessarily convex in $\mathbf{x}$ or concave in $\mathbf{y}$. Hence, Nash equilibria do not necessarily exist, and even if they do, finding a global Nash equilibrium is hopelessly impractical. We therefore restrict to gradient-based methods that hopefully converge to local solutions.

For convenience, let $\mathbf{w} = (\mathbf{x}, \mathbf{y}) \in \mathbb{R}^{m + n}$ denote the vector of combined parameters and $\xi(\mathbf{w}) = \xi(\mathbf{x}, \mathbf{y}) =(\nabla_{\mathbf{x}} f(\mathbf{w}), -\nabla_{\mathbf{y}}f(\mathbf{w})) \in \mathbb{R}^{m + n}$ denote the signed vector of partial derivatives, often known as the simultaneous gradient. The Hessian of the game is a $(m+ n) \times (m + n)$-matrix of second-order derivaties, which is not necessarily symmetric:

$$\mathbf{H}(\mathbf{w}) := \nabla_{\mathbf{w}} \cdot \xi(\mathbf{w})^{\mathsf{T}} = \begin{bmatrix}\nabla^2_{\mathbf{x}\mathbf{x}} f(\mathbf{w}) & \nabla^2_{\mathbf{x}\mathbf{y}} f(\mathbf{w}) \\  -\nabla^2_{\mathbf{y}\mathbf{x}} f(\mathbf{w}) & -\nabla^2_{\mathbf{y}\mathbf{y}} f(\mathbf{w}) \end{bmatrix}.$$

**Definition 1:** A point $\mathbf{w}^\star = (\mathbf{x}^\star, \mathbf{y}^\star)$ is a **local minimax** (or Nash equilibirum) if there is a neighborhood $U$ of $(\mathbf{x}^\star, \mathbf{y}^\star)$ such that $f(\mathbf{x}^\star, \mathbf{y}) \leq f(\mathbf{x}^\star, \mathbf{y}^\star) \leq f(\mathbf{x}, \mathbf{y}^\star)$ for all $(\mathbf{x}, \mathbf{y}) \in U$. These conditions are equivalent to $\xi(\mathbf{w}^\star) = 0$ and $\nabla^2_{\mathbf{x}\mathbf{x}} f(\mathbf{w}^\star) \succeq 0$ and $\nabla^2_{\mathbf{y}\mathbf{y}} f(\mathbf{w}^\star) \preceq 0$.

**Definition 2:** A point $\mathbf{w}$ in a discrete time dynamical system with update rule $\mathbf{w}_{t + 1} = \omega(\mathbf{w}_{t})$ is called a **fixed point** if $\omega(\mathbf{w}) = \mathbf{w}$. A fixed point $\mathbf{w}$ is **stable** if the spectral radius $\rho(\mathbf{J}(\mathbf{w}))$ is at most 1, where $\mathbf{J(\mathbf{w})}$ is the Jacobian of $\omega$ computed at $\mathbf{w}$.

The reason we're interested in spectral analysis of the Jacobian of the fixed points is the following well-known fact: if a fixed point $\mathbf{w}$ is stable and hyperbolic (i.e. $\mathbf{J}(\mathbf{w})$ has no eigenvalues with absolute value 1), there is a small neighborhood around $\mathbf{w}$ such that all initializations in that neighborhood results in convergence to $\mathbf{w}$.


### Gradient Descent Ascent (GDA)
A straightforward optimization routine to solve $\min_{\mathbf{x}} \max_{\mathbf{y}} f(\mathbf{x}, \mathbf{y})$ is <em>gradient descent-ascent</em> (GDA), where both players take a gradient update simultaneously $\mathbf{w}_{t + 1} = \mathbf{w}_t - \eta \xi(\mathbf{w})$, i.e.:

$$\begin{bmatrix}\mathbf{x}_{t + 1} \\ \mathbf{y}_{t + 1} \end{bmatrix} = \begin{bmatrix}\mathbf{x}_{t} \\ \mathbf{y}_{t} \end{bmatrix} - \eta \begin{bmatrix}\nabla_{\mathbf{x}} f( \mathbf{x}_t, \mathbf{y}_t)  \\ -\nabla_{\mathbf{y}} f( \mathbf{x}_t, \mathbf{y}_t) \end{bmatrix}.$$

**Theorem 1** ([Mescheder et al., 2017](http://papers.nips.cc/paper/6779-the-numerics-of-gans.pdf), [Daskalakis and Panageas (2018](https://papers.nips.cc/paper/8136-the-limit-points-of-optimistic-gradient-descent-in-min-max-optimization.pdf)): If the Hessian computed at a local minimax has no purely imaginary eigenvalue, then the local minimax is a stable fixed point with small enough learning rate.

**Proof of Theorem 1**:
Before we prove Theorem 1, it's worth noting that stability does not guarantee local minimaxity. For example, $(0, 0)$ is not a local minimax of $f(x, y) = 3x^2 + 4xy + y^2$ due to $\nabla_{\mathbf{y}\mathbf{y}}((0, 0)) = 2 > 0$, though it is a stable fixed point of GDA for all $\eta < 1$ (the Jacobian of GDA has eigenvalues $1 - 2\eta$).

More generally, the Jacobian of GDA at a local minimax $\mathbf{w}^\star$ is $\mathbf{J}(\mathbf{w}^\star) = \mathbf{I} - \eta \mathbf{H}(\mathbf{w}^\star)$, which has eigenvalues $1 - \eta \lambda(\mathbf{H})$ where $\lambda(\mathbf{H})$ are eigenvalues of the Hessian evaluated at $\mathbf{w}^\star$. Since $\mathbf{w}^\star$ is a local minimax, $\nabla^2_{\mathbf{x}\mathbf{x}}(\mathbf{w}^\star)$ and $\nabla^2_{\mathbf{y}\mathbf{y}}(\mathbf{w}^\star)$ are positive semidefinite and thus by Ky Fan inequality, $\text{Re}(\lambda(\mathbf{H})) \geq \frac{1}{2} \lambda_\min(\mathbf{H} + \mathbf{H}^\mathsf{T}) \geq 0$. By choosing $\eta < 2\min_{\lambda(\mathbf{H})} \left\{\text{Re}(\lambda(\mathbf{H})) / |\lambda(\mathbf{H})|^2  \right\}$, we have $|1 - \eta \lambda(\mathbf{H})| = 1 - \eta (2\text{Re}(\lambda(\mathbf{H}) - \eta |\lambda(\mathbf{H})|^2) < 1$. In other words, any local minimax $\mathbf{w}^\star$ of GDA is stable if the learning rate $\eta$ is small enough. However, if the Hessian has an eigenvalue with a small real part but a large imaginary part ($\text{Re}(\lambda(\mathbf{H})) / |\lambda(\mathbf{H})|^2$ is small), the learning rate has to be very small, which implies extremely slow convergence.

In case the Hessians contain purely imaginary eigenvalues, Theorem 1 does not guarantee convergence of GDA. In fact, recent works ([Mertikopoulos et al., 2018](https://arxiv.org/abs/1709.02738), [Balduzzi et al., 2018](https://arxiv.org/abs/1802.05642)) show that GDA exhibits strong rotation around fixed points and sometimes diverges. In the simple bilinear setting where $f(\mathbf{x}, \mathbf{y}) = \mathbf{x}^{\mathsf{T}} \mathbf{A} \mathbf{y}$ for some matrix $\mathbf{A}$, for example, the simultaneous gradient $\xi(\mathbf{x}, \mathbf{y}) = (\mathbf{A} \mathbf{y}, - \mathbf{A}^{\mathsf{T}} \mathbf{x})$ implies

$$\mathbf{w}_{t + 1} = \begin{bmatrix}\mathbf{I} & -\mathbf{A} \\ \mathbf{A}^{\mathsf{T}} & \mathbf{I}  \end{bmatrix}\mathbf{w}_{t} = \det(\mathbf{I} + \mathbf{A} \mathbf{A}^{\mathsf{T}}) \, (\mathbf{R} \mathbf{w}) \quad\qquad \text{where} \quad\qquad \mathbf{R} = \frac{1}{\det(\mathbf{I} + \mathbf{A} \mathbf{A}^{\mathsf{T}})}  \begin{bmatrix}\mathbf{I} & -\mathbf{A} \\ \mathbf{A}^{\mathsf{T}} & \mathbf{I}  \end{bmatrix} \in \text{SO}(m + n).$$

is a rotation matrix. As a result, GDA updates show cyclic behavior around $(\mathbf{0}, \mathbf{0})$ and can easily diverge when $\det(\mathbf{I} + \mathbf{A} \mathbf{A}^{\mathsf{T}}) > 1$. The rotation phenomenon is not limited to the bilinear setting (see Figure 1); it is commonly observed that the generator in GANs often cycles through a subset of modes and fails to capture the diversity of the data distribution (see Figure 2), a problem often known as mode collapsing.

<figure>
  <img src="images/mechanics-of-differentiable-games/softplus.png" style='margin: 10px auto' alt="my alt text"/>
  <figcaption>Figure 1: The paths taken by various gradient-based methods next to a loss surface and its contour for solving $\min_x \max_y f(x, y)$ where $f(x, y) = \log(1 + e^x) + 3xy - \log(1 + e^y)$.</figcaption>
</figure>

<figure>
  <img src="images/mechanics-of-differentiable-games/gda.gif" style='margin: 10px auto' alt="my alt text"/>
  <figcaption>Figure 2: Training GAN on a mixture of 16 Gaussians with gradient descent ascent (GDA). Left: Kernel density plot of samples generated by the generator. Middle: Scatter plots of generated samples in orange and true samples in blue together with contours of the discriminator. Right: Training loss values of the generator and the discrimnator. </figcaption>
</figure>


### Consensus Optimization (CO)

[Mescheder et al. (2017)](http://papers.nips.cc/paper/6779-the-numerics-of-gans.pdf) observed that when the rotation phenomenon happens, simultaneous gradients $\xi(\mathbf{x}, \mathbf{y})$ decrease slowly in norm (consider the bilinear setting where $\mathbf{x}, \mathbf{y} \in \mathbb{R}$ and $\mathbf{A} = 1$). Since $\xi(\mathbf{x}, \mathbf{y})$ at fixed points are 0, one way to mitigate rotation is to directly penalize $\|\xi(\mathbf{x}, \mathbf{y})\|^2$, i.e. solving $\min_{\mathbf{x}} \ell_1(\mathbf{x}, \mathbf{y})$ and $\min_{\mathbf{y}} \ell_2(\mathbf{x}, \mathbf{y})$ simultaneously where

$$\ell_1(\mathbf{x}, \mathbf{y}) = f(\mathbf{x}, \mathbf{y}) + \frac{1}{2}\gamma \|\xi(\mathbf{x}, \mathbf{y})\|^2, \qquad \ell_2(\mathbf{x}, \mathbf{y}) = -f(\mathbf{x}, \mathbf{y}) + \frac{1}{2}\gamma \|\xi(\mathbf{x}, \mathbf{y})\|^2$$

and $\gamma > 0$ is a hyperparameter. The new optimization method is called consensus optimization (CO), whose gradient update has the form $\mathbf{w}_{t + 1} = \mathbf{w}_t - \eta \xi(\mathbf{w}) - \eta \gamma \mathbf{H}^{\mathsf{T}}(\mathbf{w}) \xi(\mathbf{w})$, i.e.

$$\begin{bmatrix}\mathbf{x}_{t + 1} \\ \mathbf{y}_{t + 1} \end{bmatrix} = \begin{bmatrix}\mathbf{x}_{t} \\ \mathbf{y}_{t} \end{bmatrix} - \eta \begin{bmatrix}\nabla_{\mathbf{x}} f( \mathbf{x}_t, \mathbf{y}_t)  \\ -\nabla_{\mathbf{y}} f( \mathbf{x}_t, \mathbf{y}_t) \end{bmatrix} - \eta \gamma \begin{bmatrix}\nabla^2_{\mathbf{x}\mathbf{x}} f(\mathbf{x}, \mathbf{y}) & \nabla^2_{\mathbf{x}\mathbf{y}} f(\mathbf{x}, \mathbf{y}) \\  -\nabla^2_{\mathbf{y}\mathbf{x}} f(\mathbf{x}, \mathbf{y}) & -\nabla^2_{\mathbf{y}\mathbf{y}} f(\mathbf{x}, \mathbf{y}) \end{bmatrix}^\mathbf{\mathsf{T}} \begin{bmatrix}\nabla_{\mathbf{x}} f( \mathbf{x}_t, \mathbf{y}_t)  \\ -\nabla_{\mathbf{y}} f( \mathbf{x}_t, \mathbf{y}_t) \end{bmatrix}.$$

Here, we can view CO as GDA on the new loss functions where the simultaneous gradient is $\xi_\gamma(\mathbf{w}) = \xi(\mathbf{w}) + \gamma \mathbf{H}^{\mathsf{T}}(\mathbf{w}) \xi(\mathbf{w}) = (\mathbf{I} + \gamma \mathbf{H}^{\mathsf{T}} (\mathbf{w}))\xi(\mathbf{w})$ and the Hessian is $\mathbf{H}_\gamma(\mathbf{w}) = \nabla_{\mathbf{w}} \cdot \xi_\gamma(\mathbf{w})^{\mathsf{T}} = \mathbf{H}(\mathbf{w}) + \gamma \mathbf{H}^{\mathsf{T}}(\mathbf{w}) \mathbf{H}(\mathbf{w}) + (\nabla_{\mathbf{w}} \cdot \mathbf{H}(\mathbf{w})) \xi(\mathbf{w})$. Since the objective functions have changed, a natural question to ask is whether the local minimaxes of the original game $\min_{\mathbf{x}} \max_{\mathbf{y}} f(\mathbf{x}, \mathbf{y})$ are still retained. The answer is yes, if $\gamma$ is small enough, for two following reasons:
* $\xi_\gamma(\mathbf{w}^\star) = 0$ implies $\xi(\mathbf{w}^\star) = 0$, if we pick $\gamma$ such that $-\gamma^{-1}$ is not an eigenvalue of $\mathbf{H}(\mathbf{w}^\star)$, i.e. $\mathbf{I} + \gamma \mathbf{H}^{\mathsf{T}} (\mathbf{w}^\star)$ is invertible.
* At any local minimax $\mathbf{w}^\star$ of the original game, $\xi(\mathbf{w}^\star) = 0$ implies that $\mathbf{H}_\gamma(\mathbf{w}^\star) = \mathbf{H}(\mathbf{w}^\star) + \gamma \mathbf{H}^{\mathsf{T}}(\mathbf{w}^\star) \mathbf{H}(\mathbf{w}^\star)$. Since $\nabla^2_{\mathbf{x}\mathbf{x}} f(\mathbf{w}^\star) \succeq 0$ and $\nabla^2_{\mathbf{y}\mathbf{y}} f(\mathbf{w}^\star) \preceq 0$ and $\mathbf{H}^{\mathsf{T}}(\mathbf{w}^\star) \mathbf{H}(\mathbf{w}^\star)$ is positive semi definite, it is clear that $\nabla^2_{\mathbf{x}\mathbf{x}} \ell_1 (\mathbf{w}^\star) \succeq 0$ for all $\gamma$ and $\nabla^2_{\mathbf{y}\mathbf{y}} \ell_2 (\mathbf{w}^\star) \preceq 0$ for small enough $\gamma$. Note that $\nabla^2_{\mathbf{y}\mathbf{y}} \ell_2 (\mathbf{w}^\star)$ is actually "less positive definite" than $\nabla^2_{\mathbf{y}\mathbf{y}} f(\mathbf{w}^\star)$ because of $\mathbf{H}^{\mathsf{T}}(\mathbf{w}^\star) \mathbf{H}(\mathbf{w}^\star)$.

By an argument similar to the proof of Theorem 1, we can easily show that any eigenvalue $\lambda(\mathbf{H}_\gamma)$ of $\mathbf{H}_\gamma(\mathbf{w}^\star)$ has nonnegative real part: $\text{Re}(\lambda(\mathbf{H}_\gamma)) \geq \frac{1}{2} \lambda_{\min} (\mathbf{H}_\gamma + \mathbf{H}_\gamma^{\mathsf{T}}) \geq 0$. [Mescheder et al. (2017)](http://papers.nips.cc/paper/6779-the-numerics-of-gans.pdf) also came up with some upper bound for the imaginary-to-real ratio, showing that that convergence of CO is potentially faster and more stable, although the bound is not intuitive. Empirically, consensus optimization works quite well in settings where GDA struggles with rotational forces (see Figure 1), but its performance is very sensitive to the choice of $\gamma$.

<!-- #region jupyter={"source_hidden": true} -->
<figure>
  <img src="images/mechanics-of-differentiable-games/co.gif" style='margin: 10px auto' alt="my alt text"/>
  <figcaption>Figure 2: Training GAN on a mixture of 16 Gaussians with consensus optimization (CO). Left: Kernel density plot of samples generated by the generator. Middle: Scatter plots of generated samples in orange and true samples in blue together with contours of the discriminator. Right: Training loss values of the generator and the discrimnator. </figcaption>
</figure>
<!-- #endregion -->

### Symplectic Gradient Adjustment (SGA)
To be continued...
