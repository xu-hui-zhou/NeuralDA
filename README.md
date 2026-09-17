# NeuralDA

**NeuralDA** is a collection of machine-learning-integrated data assimilation methods for complex dynamical systems.

## Neural Ensemble Kalman Filter (Neural EnKF)

The **Neural EnKF** performs the ensemble Kalman filter (EnKF) update in neural-network parameter space. It is designed for data assimilation in systems containing sharp or discontinuous features, with a particular focus on compressible flows with shocks.

Code, numerical examples, and documentation are available in [`NeuralEnKF/`](./NeuralEnKF).

## Ensemble Generative Filter (EnGF)

The **EnGF** is a non-Gaussian filtering framework that fits a generative model to the forecast ensemble at each data assimilation cycle. The learned distribution enables inexpensive generation of a much larger particle population for Bayesian analysis without requiring additional model forecasts.

Code, numerical examples, and documentation will be available in [`EnGF/`](./EnGF).

## Future Developments

Additional machine-learning-integrated data assimilation methods will be added to this repository as they are developed.

## Related Papers

- X.-H. Zhou, L. Beronilla, M. K. Sleeman, H. Hu, M. Morzfeld, A. M. Stuart, and T. A. Zaki,  
  [*"Neural ensemble Kalman filter: Data assimilation for compressible flows with shocks."*](https://doi.org/10.1016/j.jcp.2026.115136),  
  *Journal of Computational Physics*, 2026.

- X.-H. Zhou and J. Han,  
  [*"Ensemble generative filtering for sequential data assimilation in dynamical systems."*](https://arxiv.org/abs/2609.14078),  
  *arXiv:2609.14078*, 2026.

## License

Please refer to the license information provided with each method.
