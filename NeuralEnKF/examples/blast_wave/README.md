# 2D Blast Wave

This example demonstrates the Neural EnKF for a two-dimensional blast-wave problem.

## Configuration

The experiment settings are specified in `input.yaml`, including the initial ensemble, data-assimilation settings, observation noise, neural-network architecture, training parameters, and physical constraints.

## Running the example

A CUDA-enabled GPU is strongly recommended for neural-network training in this two-dimensional example.

First, generate the reference truth solution:

~~~bash
cd bw.truth
./Allrun input.st
cd ..
~~~

Then run the Neural EnKF workflow:

~~~bash
python main.py
~~~

The script generates the initial ensemble and performs the forecast and data-assimilation cycles. The resulting ensemble solutions are stored in the corresponding `bw_###/` directories, while observation-related DA results are written to `da_results/`.

## Postprocessing

First, compute the ensemble-mean solutions:

~~~bash
python postprocess/make_mean.py
~~~

The ensemble-mean VTR files are written to `bw.mean/results/`.

Generate the analysis flow-field figures with

~~~bash
python postprocess/plot_analysis.py
~~~

Generate the RMSE and ensemble-spread figure with

~~~bash
python postprocess/plot_rmse_spread.py
~~~

The generated figures are written to `figures/`. The figures included in this repository provide reference results for this example.

## Cleaning

Remove generated ensemble directories, ensemble-mean solutions, DA results, and figures with

~~~bash
./Allclean
~~~
