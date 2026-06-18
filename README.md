# Nerual ODEs
This package implements Nerual-ODEs driven non-adiabatic molecular dynamics simulations workflow. It currently supports two models: CNF (Continuous Normalizing Flow) and HNN (Hamiltonian Nerual Network).
## Features

### 1. CNF Training

### 2. CNF propagation

### 3. HNN Training

### 4. HNN propagation

## Directory Structure

A typical project directory for HNN propagation looks like:
```text
TRAJ_xxxx/                            # trajectory Folder
├── SHARC_HNN_interface.py            # SHARC-HNN interface
├── QM.in                             # structural input
├── QM.out                            # energy output
├── submit.sh                         # job submission script
├── output_data                       # all outputs
├── input                             # main simulation input

Each TRAJ_xxxx directory corresponds to one independent trajectory.

## Installation / Requirements
### 1. Python
```python
Python ≥ 3.8

### 2. Required interfaced software
SHARC ≥ 2.1
The SHARC program serves as a required interface for non-adiabatic transition calculations using local diabatization (LD). The official website (https://github.com/sharc-md/sharc.git) and installation guide (http://www.sharc-md.org) are provided here

### 2. Required Python packages
numpy ≥ 1.21.0
torch ≥ 1.14.1
argcomplete ≥ 3.5.3   
fastparquet ≥ 2024.11.0 
fonttools ≥ 4.60.0 
fsspec ≥ 2025.9.0 
idna ≥ 3.10 
joblib ≥ 1.5.2 
kiwisolver ≥ 1.4.7 
matplotlib ≥ 3.9.4
packaging ≥ 25.0 
pandas ≥ 2.2.2
pipx ≥ 1.7.1 
pyparsing ≥ 3.2.4 
scikit-learn ≥ 1.6.1 
scipy ≥ 1.13.1 
setuptools ≥ 78.1.1 
six ≥ 1.17.0 
wheel ≥ 0.45.1 


