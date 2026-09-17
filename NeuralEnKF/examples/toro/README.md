# Toro Shock Tube

This example demonstrates the Neural EnKF for the one-dimensional Toro shock tube problem.

## Configuration

The experiment settings are specified in `input.yaml`, including the initial ensemble, data-assimilation settings, observation noise, neural-network architecture, and training parameters.

## Running the example

First, generate the reference truth solution:

```bash
cd toro.truth
./Allrun input.st
cd ..
```

Then run the Neural EnKF workflow:

```bash
python main.py
```

The script generates the initial ensemble, performs the forecast and data-assimilation cycles, and writes the DA results to `da_results/`.

## Postprocessing

Generate the analysis figure with

```bash
python postprocess/plot_analysis.py
```

Generate the RMSE and ensemble-spread figure with

```bash
python postprocess/plot_rmse_spread.py
```

The generated figures are written to `figures/`. The figures included in this repository provide reference results for this example.

## Cleaning

Remove generated ensemble directories and DA results with

```bash
./Allclean
```

The reference figures in `figures/` are preserved.
