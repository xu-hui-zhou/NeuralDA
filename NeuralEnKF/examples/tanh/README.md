# Hyperbolic tangent function

This example provides a minimal, solver-free demonstration of the Neural Ensemble Kalman Filter (Neural EnKF) using a one-dimensional hyperbolic-tangent profile.

Unlike the compressible-flow examples, this case does not require a forward numerical solver. It is intended to illustrate the core Neural EnKF workflow in a simple setting: the forecast ensemble, consisting of hyperbolic-tangent profiles, is represented by neural networks, the EnKF analysis is performed in neural-parameter space, and the analyzed neural parameters are mapped back to physical space.

## Files

- **`main.py`**  
  Generates the truth, forecast ensemble, and observations; constructs the training chain; performs the Neural EnKF analysis; and saves the results to `da_results/`.

- **`neural_scalar.py`**  
  Provides the scalar neural representation $x \mapsto h(x)$ and the chain-based neural-network training used specifically for this example. The general chain construction and ensemble Kalman analysis are reused from `core/one_d/`.

- **`postprocess/plot_analysis.py`**  
  Plots the forecast and analysis ensembles together with the truth and observations.

- **`figures/`**  
  Contains the reference figure for this example.

## Running the Example

From the `tanh/` directory, run

```bash
python main.py
```

The Neural EnKF results are written to

```text
da_results/
├── truth.npy
├── observations.npy
├── forecast.npy
└── analysis.npy
```

Then generate the analysis figure with

```bash
python postprocess/plot_analysis.py
```

The figure is saved as

```text
figures/tanh_analysis.png
```
