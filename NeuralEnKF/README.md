# Neural EnKF

This repository provides the implementation of the Neural Ensemble Kalman Filter (Neural EnKF) for data assimilation in compressible flows with shocks, together with benchmark cases and data-assimilation examples.

## Repository Structure

~~~text
NeuralEnKF/
├── benchmarks/
├── core/
└── examples/
~~~

- **`benchmarks/`**  
  Benchmark cases for compressible-flow simulations, including both initial and restart workflows. These cases are provided for standalone flow simulations and do not involve data assimilation.

- **`core/`**  
  Core implementation of the Neural EnKF, including neural state representations, ensemble Kalman analysis, chain-based training, and simulation utilities.

- **`examples/`**  
  Data-assimilation examples demonstrating the application of the Neural EnKF to benchmark problems. Each example includes its configuration, simulation setup, data-assimilation workflow, postprocessing scripts, and reference results.

See the README in each directory for additional details.

## Requirements

### M2C (Multiphysics Modeling and Computation)

The numerical simulations require the M2C multiphysics solver:

- Developer's repository: https://github.com/kevinwgy/m2c
- Article: https://doi.org/10.1016/j.cpc.2026.110023
- arXiv: https://arxiv.org/abs/2508.16387

Please install and configure M2C before running the benchmark cases or data-assimilation examples.

### Python

The Neural EnKF implementation and postprocessing scripts require Python with the following main packages:

- NumPy
- PyTorch
- PyVista
- PyYAML
- Matplotlib

The one-dimensional examples are designed to run on CPUs and do not require GPU acceleration. For two/three-dimensional examples, a CUDA-enabled GPU is strongly recommended to accelerate neural-network training.

## Citation

The Neural EnKF method is described in:

Zhou, X.-H., Beronilla, L., Sleeman, M. K., Hu, H., Morzfeld, M., Stuart, A. M., and Zaki, T. A. (2026).  
**Neural ensemble Kalman filter: Data assimilation for compressible flows with shocks.**  
*Journal of Computational Physics*, 115136.  
https://doi.org/10.1016/j.jcp.2026.115136

If you use this code in your research, please cite:

~~~bibtex
@article{zhou2026neural,
  title={Neural ensemble Kalman filter: Data assimilation for compressible flows with shocks},
  author={Zhou, Xu-Hui and Beronilla, Lorenzo and Sleeman, Michael K and Hu, Hangchuan and Morzfeld, Matthias and Stuart, Andrew M and Zaki, Tamer A},
  journal={Journal of Computational Physics},
  pages={115136},
  year={2026},
  publisher={Elsevier}
}
~~~

## License and third-party code

This is a mixed-license repository:

- The Neural EnKF code authored by Xuhui Zhou is available under the [MIT License](LICENSE).

- Source and input files from or derived from the Multiphysics Modeling and Computation (M2C) code are licensed separately under the GNU General Public License, version 3 only.

M2C itself is an external dependency and is not distributed as part of this repository. Its own license applies when it is obtained or used.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the affected files and attribution details. The MIT License does not apply to, replace, or override the GPLv3 terms for the M2C-derived files.

## Contact

For questions regarding the Neural EnKF code, please contact Xuhui Zhou:

- xuz067@ucsd.edu or xzhou517@caltech.edu
