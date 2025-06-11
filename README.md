# Flow of granular materials in a rotary kiln.

**Authors:** L. Plato, M. Sliwinski, D. Peschka

**Language:** Python + FEniCS

<img src="media/kiln.gif" width="50%">

**Description:** This documentation describes a coupled phase-field and flow model, combining the Allen–Cahn equation with incompressible Stokes flow. This is a first step towards the simulation of a rotary kiln (the granular flow aspect of it) as a problem considered in a  European Study Groups with Industry at NTU Trondheim (**ESGI 186 – the Norwegian Study Group with Industry**).

## Allen–Cahn–Stokes model for rotary kiln

We consider a domain $\Omega \subset \mathbb{R}^d$ (rotary kiln $d=3$ or a cross-section $d=2$) and a phase-field variable $\phi : \Omega \times [0, T] \to [0, 1]$, where $\phi \approx 1$ denotes the granular phase and $\phi \approx 0$ denotes the gas phase.

### 1. Allen–Cahn equation (with advection)

The phase field evolves according to:

$$
\frac{\phi^{n+1} - \phi^n}{\tau} + \mathbf{v} \cdot \nabla \phi^{n+1} = m\left[-\varepsilon \Delta \phi^{n+1} + \frac{2}{\varepsilon} \phi^{n+1} (2(\phi^{n+1})^2 - 3\phi^{n+1} + 1)\right],
$$

where:

* $\tau$ is the time step,
* $m$ is the mobility,
* $\varepsilon$ is the interface width,
* $\mathbf{v}$ is the velocity field from the Stokes problem.

### 2. Stokes flow with $\phi$-dependent viscosity

The velocity $\mathbf{v}$ and pressure $p$ satisfy:

$$
-\nabla \cdot \sigma(\phi,\nabla\mathbf{v},p) = \mathbf{f}_{\text{grav}}(\phi), \quad \nabla \cdot \mathbf{v} = 0,
$$

with:

* $\sigma = -p\mathbb{I}+2 \mu(\phi) \boldsymbol{\varepsilon}(\mathbf{v})$ (Cauchy stress),
* $\boldsymbol{\varepsilon}(\mathbf{v}) = \tfrac{1}{2}(\nabla \mathbf{v} + \nabla \mathbf{v}^T)$ (symmetric gradient),
* $\mu(\phi) = \mu_s |\phi| + \mu_g |1 - \phi|$ (interpolated viscosity),
* $\mathbf{f}_{\text{grav}}(\phi) = \mathbf{g} \phi$ (gravity acting in one granular phase).

Boundary terms include slip and penalty constraints

$$
t\cdot\sigma n = \beta_{\text{slip}}(v-v_{\text{rot}})\cdot t,\qquad n\cdot\sigma n = \beta_{\text{pen}}v\cdot n
$$

which for $\beta_{\text{pen}}\gg 1$ should lead to $v\cdot n=0$ and for $\beta_{\text{slip}}\gg 1$ to $(v-v_{\text{rot}})\cdot t=0$.


**Reference:**
* granular rheology (Gray, J. M. N. T. (2001). *Granular flow in partially filled slowly rotating drums*. JFM)

