[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10406144.svg)](https://doi.org/10.5281/zenodo.10406144)

# sEMG innervation zone estimation

The algorithm in this python software package is able to identify innervation
zone clusters within continuous surface electromyography (sEMG) recordings
based on linear electrode arrays. It is an implementation of the algorithm
described in the conference publication [1].

This implementation is published under Apache 2.0 license.

## installation

As installation dependencies you need a version of `Python >= 3.12.2`
and a recent version of pip installed. You also need the rust build system.
This you can install via [rustup](https://rustup.rs/)

If you have these requiremnts set, you can install this python package by executing

```bash
pip install -e .
```

in the project root. Note If you ave difficulties installing it in your
default python environemnt create a virtual environemnt (e.g. with
[conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html)
or [venv](https://docs.python.org/3/library/venv.html))
with a version of `Python >= 3.12.2`.

## usage

For the basic usage please refer to the following pseudo code.
A more complete documentation will follow in later releases.

```python
import semg_iz_estimation

# Set the parameters
ize = semg_iz_estimation.IzEstimation(
    window_width_s,
    window_step_s,
    lam,
    epsilon,
    inter_electrode_distance,
    expected_v_conduction
)

# a list of size N_electrodes
electrode_position = [...]

# a list of size N_Samples
time_s = [...]

# a nested list of size N_samples X N_electrodes
emg_array = [
    [...], #<- at sample 0 the electrode potential vector of size N_electrodes
    [...],
      .
      .
      .
    [...]
]

r_single_trhead = ize.find_IPs(
    time_s
    emg_array,
    electrode_position)

r_n_threads = ize.find_IPs_parallel(
    time_s,
    emg_array,
    electrode_position,
    n_worker
)
```

For a complete usage example please refer to the integration test
`./test/integration.py`.

# References

[1] Mechtenberg, M, Grimmelsmann, N and Schneider, A (2024)
"A new Algorithm for Innervation Zone Estimation Using Surface
Electromyography – a Simulation Study Based on a Simulator for
Continuous sEMGs" Biosignals 2024. Conference still pending.
