# Generalized Toffoli Gate Simulation Code and Data

This repository contains numerical simulation code and data for the paper:

**Generalized Toffoli Gates with Customizable Single-Step Multiple-Qubit Control**

The code studies generalized Toffoli gates based on Ising-type interactions, including single-step Hamming-control gates, multiple-designated-configuration gates, sequential circuit baselines, decoherence effects, and control-error effects.

## Repository Contents

```text
.
├── Data.xlsx
├── README.md
├── BuildingHammingCircuit.py
├── SimulationOfToffoli.py
├── SimulationOfCircuitByHammingToffoli.py
├── SimulationOfHammingCircuitsBy2Toffoli.py
├── SimulationOfToffoliControlError.py
└── SimulationOfCircuitControlError.py
```

## File Description

### `Data.xlsx`

Numerical data used to generate the plots in the paper.

The workbook contains average-error data for different generalized Toffoli gates, physical platforms, and parameter settings.

### `SimulationOfToffoli.py`

Simulates the single-step generalized Toffoli gate based on the Ising-type Hamiltonian.

This script is used for simulations of gates such as:

- Hamming-control Toffoli gates
- weighted-Hamming-control Toffoli gates
- multiple-designated-configuration Toffoli gates

It supports both decoherence-free and decoherence-included simulations.

### `SimulationOfCircuitByHammingToffoli.py`

Simulates a sequential implementation of a multiple-designated-configuration gate using several Hamming-control Toffoli gates.

This is used as a baseline for comparison with the proposed simultaneous multi-frequency single-step implementation.

### `SimulationOfHammingCircuitsBy2Toffoli.py`

Simulates a multi-step circuit implementation of a Hamming-control Toffoli gate using ordinary two-control-qubit Toffoli gates.

This corresponds to the baseline dynamical simulation for the Hamming-control comparison. In this simulation, X gates are treated as ideal gates, and decoherence is modeled during the two-control-qubit Toffoli evolutions.

### `SimulationOfToffoliControlError.py`

Simulates the proposed single-step generalized Toffoli gate under control error.

The control-error model applies independent Gaussian fluctuations to nonzero designed parameters, including:

- target-control Ising couplings
- driving amplitudes
- driving frequencies

The noisy parameters are resampled as piecewise-constant values during the gate execution.

### `SimulationOfCircuitControlError.py`

Simulates the sequential circuit implementation under the same type of control-error model.

This allows comparison between the proposed simultaneous-driving implementation and the sequential multi-step implementation.

### `BuildingHammingCircuit.py`

Builds a Qiskit-based reference decomposition of a Hamming-weight-two oracle on four control qubits.

The generated oracle flips the target qubit if and only if the Hamming weight of the four control qubits is equal to two. The script enumerates the six weight-two control patterns, uses X gates to convert open controls into closed controls, applies Qiskit's multi-controlled-X decomposition, searches over supported MCX modes and transpiler seeds, prints gate counts, and exports the best circuit to QASM.

This script is a Qiskit decomposition helper for checking alternative circuit-synthesis baselines. It is not the QuTiP dynamical simulation script for the average-fidelity plots.

## Requirements

The simulation scripts are written in Python and use QuTiP for quantum dynamics simulation.

Required packages for the QuTiP simulations:

```bash
pip install numpy qutip
```

Optional packages for inspecting or processing the data file:

```bash
pip install pandas openpyxl
```

`BuildingHammingCircuit.py` additionally requires Qiskit:

```bash
pip install qiskit
```

The scripts use Python multiprocessing. On shared servers, it is recommended to limit the number of workers or run with fewer CPU cores to avoid overloading the machine.

## Basic Usage

Each simulation script is self-contained. Before running a simulation, edit the parameters near the top of the corresponding file.

Typical parameters include:

```python
N = 5
w = np.array([1, 1, 1, 1])
plateform = "cir"   # "cir" or "ion"
deco = "on"         # "on" or "off"
number_drive = 2
n_min = 8
n_range = 9
dis_n = np.arange(n_min, n_range, 4)
```

Then run the script directly:

```bash
python3 SimulationOfToffoli.py
```

For control-error simulations:

```bash
python3 SimulationOfToffoliControlError.py
python3 SimulationOfCircuitControlError.py
```

For the Qiskit Hamming-weight-two oracle decomposition:

```bash
python3 BuildingHammingCircuit.py
```

To use two ancilla qubits, matching the common paper-level comparison setting:

```bash
python3 BuildingHammingCircuit.py --anc 2 --seeds 0,1,2 --opt 3
```

To force CCX gates to be decomposed away:

```bash
python3 BuildingHammingCircuit.py --no-ccx --anc 2 --seeds 0,1,2 --opt 3
```

## Physical Platforms

The QuTiP simulation scripts currently support two parameter sets.

### Superconducting circuit

```python
plateform = "cir"
J = 2 * np.pi * 40 * 10**6
T1 = 30 * 10**(-6)
T2 = 30 * 10**(-6)
```

### Trapped ion

```python
plateform = "ion"
J = 2 * np.pi * 2 * 10**3
T1 = np.inf
T2 = 50
```

The variable name `plateform` is kept because it is used in the current scripts.

## Decoherence Model

The simulations use independent per-qubit amplitude damping and pure dephasing.

For each qubit, the collapse operators are:

```text
sqrt(1 / T1) * sigma_minus
sqrt(gamma_phi / 2) * Z
```

where

```text
gamma_phi = 1 / T2 - 1 / (2 * T1)
```

The factor `1/2` in `sqrt(gamma_phi / 2)` is required because a Pauli-Z Lindblad operator causes off-diagonal coherence to decay at twice the coefficient used in the dissipator.

## Control-Error Model

The control-error scripts use independent relative Gaussian noise on the designed nonzero control parameters.

For each resampling interval, the noisy parameters are modeled as:

```text
J_ij  -> J_ij  * (1 + delta_J)
Omega -> Omega * (1 + delta_Omega)
omega -> omega * (1 + delta_omega)
```

where each noise variable is independently sampled from a zero-mean Gaussian distribution.

The number of resampling intervals is controlled by:

```python
N_RESAMPLE_LIST
```

For example:

```python
N_RESAMPLE_LIST = [10, 20, 30, 40, 50, 60, 70, 80, 100, 1000, 10000]
```

The relative noise strengths are controlled by:

```python
sigma_J_rel
sigma_Omaga_rel
sigma_omaga_rel
```

For example:

```python
sigma_J_rel = 0.001
sigma_Omaga_rel = 0.001
sigma_omaga_rel = 0.001
```



## Citation

If you use this code or data, please cite the associated paper:

```text
Generalized Toffoli Gates with Customizable Single-Step Multiple-Qubit Control
```
