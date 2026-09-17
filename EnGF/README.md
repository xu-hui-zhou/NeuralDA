# Ensemble Generative Filter (EnGF)

The **Ensemble Generative Filter (EnGF)** is a non-Gaussian filtering framework for sequential data assimilation in dynamical systems.

The key idea is to fit a lightweight generative model to the forecast ensemble at each data assimilation cycle and exploit inexpensive sampling from the learned distribution to generate a much larger particle population for Bayesian analysis, without requiring additional model forecasts.

In the current implementation, Gaussian mixture models (GMMs) are used as lightweight generative models that can be efficiently fitted to moderately large forecast ensembles.

## Method

At each data assimilation cycle, the EnGF consists of four main steps:

1. **Forecast:** Propagate the ensemble through the dynamical model.
2. **Generative modeling:** Fit a generative model to the forecast ensemble.
3. **Bayesian analysis:** Draw a large number of inexpensive particles from the learned forecast distribution and update their importance weights using the observation likelihood.
4. **Resampling:** Resample the weighted particles to construct the analysis ensemble for the next forecast cycle.

This separates the number of computationally expensive model forecasts from the number of particles used during the Bayesian analysis.

Two extensions are also considered:

- **Tempered EnGF:** Uses likelihood tempering to mitigate particle degeneracy under informative observations.
- **Latent EnGF:** Performs generative modeling and Bayesian analysis in a reduced latent space for high-dimensional systems.

## Numerical Examples

The repository includes numerical experiments for:

- Doubling map
- Lorenz-63 system
- Lorenz-96 system
- Toro shock tube problem

These examples examine the performance of the EnGF across increasingly complex nonlinear and non-Gaussian filtering problems.

## Paper

X.-H. Zhou and J. Han,  
[*"Ensemble generative filtering for sequential data assimilation in dynamical systems."*](https://arxiv.org/abs/2609.14078),  
*arXiv:2609.14078*, 2026.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{zhou2026ensemble,
  title   = {Ensemble generative filtering for sequential data assimilation in dynamical systems},
  author  = {Zhou, Xu-Hui and Han, Jiequn},
  journal = {arXiv preprint arXiv:2609.14078},
  year    = {2026}
}
