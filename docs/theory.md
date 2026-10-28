---
icon: lucide/orbit
---

# Theory: Overview

Raytrax solves the ray equations of microwave rays employed for electron cyclotron resonance heating of magnetic confinement fusion plasmas. With peak magnetic flux densities of several Teslas, electron cyclotron resonance frequencies are in the range of hundreds of GHz and corresponding wave lengths of the order of millimeters. As the length scales of variation of density and temperature are typically much larger, the use of the **geometrical optics** (GO) approximation (sometimes called WKB approximation in analogy with quantum mechanics) is justified.


## Ray Representation

In the GO approximation, the monochromatic wave field of a single ray is represented as

$$\boldsymbol{E}(\boldsymbol{r}, t) = \boldsymbol{E}_0(\boldsymbol{r}) e^{i \left(S(\boldsymbol{r}) - \omega t\right)}$$

where $\boldsymbol{E}_0(\boldsymbol{r})$ is the slowly varying amplitude and $S(\boldsymbol{r})$ is the rapidly varying eikonal (scalar phase function). Here and in the following, boldface symbols denote vector or tensor quantities, and we use SI units.

The most relevant quantities describing wave propagation in the plasma are:

- **$\boldsymbol{r}$**: Position vector along the ray trajectory
- **$\boldsymbol{k}(\boldsymbol{r})$**: Local complex wave vector, defined as $\boldsymbol{k} = \nabla S$, which is tangential to the ray direction $\hat{\boldsymbol{s}}$, $\boldsymbol{k} = k_r \hat{\boldsymbol{s}} + i k_i \hat{\boldsymbol{s}}$
- **$\boldsymbol{n}(\boldsymbol{r})$**: Vectorial index of refraction, defined as $\boldsymbol{n} = (c/\omega) \boldsymbol{k}$ with magnitude $n = |\boldsymbol{n}|$
- **$\alpha$**: Absorption coefficient, given by $\alpha = 2\,\text{Im}(\boldsymbol{k} \cdot \hat{\boldsymbol{s}}) = 2 k_i$, leading to exponential decay $e^{-\alpha s}$ of the wave power (and $e^{-\alpha s/2}$ of the wave amplitude) along the ray

See [absorption](theory/absorption.md) for more details on the absorption coefficient and how it is computed from the plasma parameters.

## Dispersion Relation

Treating the plasma as a dielectric medium which is homogeneous (in a volume that is large compared to the wavelength), but *anisotropic* due to the presence of a strong external magnetic field $\boldsymbol{B}_0$, the plasma response to the electromagnetic wave is described by the complex dielectric tensor $\boldsymbol{\varepsilon_r}$, 

$$\boldsymbol{D}=\varepsilon_0 \boldsymbol{\varepsilon_r}\boldsymbol{E}$$

and Maxwell's equations read $\nabla\times\boldsymbol{E}= -\frac{\partial \boldsymbol{B}}{\partial t}$ and $\nabla\times\boldsymbol{H}= \frac{\partial \boldsymbol{D}}{\partial t}$ in the absence of free currents. From this, the wave equation in the frequency domain becomes:

$$\nabla \times (\nabla \times \boldsymbol{E}) + \frac{\omega^2}{c^2} \boldsymbol{\varepsilon_r} \cdot \boldsymbol{E} = 0$$

Substituting the eikonal ansatz in the locally homogeneous approximation and retaining leading-order terms yields an eigenvalue problem for the wave field:

$$\boldsymbol{\mathsf{D}} \cdot \boldsymbol{E}_0 = 0$$

Here, $\boldsymbol{\mathsf{D}}$ (not to be confused with $\boldsymbol{D}$) is the dispersion tensor:

$$\boldsymbol{\mathsf{D}} = \boldsymbol{\varepsilon_r} - n^2 \boldsymbol{I} + \boldsymbol{n}\boldsymbol{n}$$

where $\boldsymbol{I}$ is the identity tensor and $\boldsymbol{n}\boldsymbol{n}$ is the dyadic product. For a nontrivial wave field to exist, the determinant must vanish:

$$\det \boldsymbol{\mathsf{D}} = 0$$

This dispersion relation connects the wave vector $\boldsymbol{k}$ to the frequency $\omega$ and the local plasma parameters through $\boldsymbol{\varepsilon_r}(\boldsymbol{B}, n_e, T)$. See [ray tracing](theory/ray-tracing.md) for how this dispersion relation is used to derive the ray equations.

## Raytrax Inputs and Outputs

The main outputs of [`trace`][raytrax.api.trace] are the **ray trajectory** $\boldsymbol{r}(s)$ and the **linear deposition power density** $dP/ds$, both available in the [`BeamProfile`][raytrax.types.BeamProfile] inside the returned [`TraceResult`][raytrax.types.TraceResult]. The volumetric deposition profile as a function of $\rho$ is in [`RadialProfile`][raytrax.types.RadialProfile].

The inputs are the **magnetic field** $\boldsymbol{B}(\boldsymbol{r})$, **electron density** $n_e(\boldsymbol{r})$, and **temperature** $T(\boldsymbol{r})$ (provided via [`MagneticConfiguration`][raytrax.equilibrium.interpolate.MagneticConfiguration] and [`RadialProfiles`][raytrax.types.RadialProfiles]), which together determine the dielectric tensor $\boldsymbol{\varepsilon}_r$.

Additionally, the initial conditions for the ray (initial position $\boldsymbol{r}_0$, direction $\hat{\boldsymbol{s}}_0$, and wave frequency $\omega$) are specified in the [`Beam`][raytrax.types.Beam].

## Conventions

Throughout the documentation and code, we use the **normalized effective (minor) radius** $\rho$ defined as the square root of the normalized toroidal flux, $\rho = \sqrt{\psi/\psi_\text{edge}}$, where $\psi$ is the toroidal flux and $\psi_\text{edge}$ is its value at the plasma edge.