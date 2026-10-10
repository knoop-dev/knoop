# knoop

[![release](https://github.com/lvthnn/amsim/actions/workflows/release.yml/badge.svg)](https://github.com/lvthnn/amsim/actions/workflows/release.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22741101.svg)](https://doi.org/10.5281/zenodo.22741101)

knoop is a CLI tool for efficient, flexible forward-time simulation of
assortative mating and participation/ascertainment effects. It supports arbitary
multivariate assortment regimes and a highly configurable sampling suite for
simulating various study designs.

knoop is free software, licensed under the GNU General Public License v3.

This software is under development and may contain bugs. Please report bugs
via [issues](https://github.com/knoop/amsim/issues) on GitHub or by
[email](mailto:kh626@imperial.ac.uk).

## Installation

### Option A: download a prebuilt binary (recommended)

Prebuilt binaries for Linux (x86_64, arm64) and macOS (x86_64, arm64) are
attached to every [release](https://github.com/lvthnn/amsim/releases).
Dependencies are linked statically or built directly into the binary, so no
installation aside from the binary is required.

To install amsim, grab the appropriate filename for your platform from the
[latest release page](https://github.com/lvthnn/amsim/releases/latest).

### Option B: build from source

### Prerequisites

- CMake >= 3.20
- A C++23 compiler (tested with recent Clang and GCC)
- BLAS and LAPACK (on macOS, Apple's Accelerate framework is used automatically;
  on Linux, install e.g. OpenBLAS: `apt install libopenblas-dev liblapack-dev`
  or equivalent)
- HDF5 with C bindings (`apt install libhdf5-dev`, `brew install hdf5`, or
  equivalent)

Linux and macOS are supported natively. Windows is not currently natively
supported, although you can use WSL.

### Building

```sh
git clone https://github.com/lvthnn/amsim.git
cd amsim
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

This may take some time.

## Usage

amsim is driven entirely through command-line flags. Every flag has detailed,
example-driven documentation built in:

```sh
amsim --help                  # display help message + exit
amsim --help --n-individuals  # usage string for specific option(s)
```
