# knoop[^1] <picture><source media="(prefers-color-scheme: dark)" srcset=".github/assets/knoop_dark.svg"><img src=".github/assets/knoop_light.svg" width="250" align="right" hspace="6"></picture>

[![release](https://github.com/lvthnn/amsim/actions/workflows/release.yml/badge.svg)](https://github.com/lvthnn/amsim/actions/workflows/release.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22741101.svg)](https://doi.org/10.5281/zenodo.22741101)

[^1]: **K**nown **N**uisances **o**f **O**bservational **P**henogenetics. Also
a [microhardness test](https://en.wikipedia.org/wiki/Knoop_hardness_test). Or
just **knoop**, whichever you like.

**knoop** (formerly **amsim**) is a command-line tool for quantitative genetics
simulation. It lets you run sensitivity analyses of study designs and estimation
methods under assortative mating and selection bias at biobank scale.

**knoop** is free software, licensed under the [GNU General Public License
v3](LICENSE).

This software is under development and may contain bugs. Please report bugs
via [issues](https://github.com/knoop/issues) on GitHub or via
[email](mailto:kh626@imperial.ac.uk).

## Installation

### Prebuilt binaries (recommended)

Prebuilt binaries for Linux (x86_64, arm64) and macOS (x86_64, arm64) are
attached to every [release](https://github.com/lvthnn/amsim/releases). To
install **knoop**, grab the appropriate filename for your platform from the
[latest release page](https://github.com/lvthnn/amsim/releases/latest).

### Building from source via CMake

The following dependencies are required to build from source:
- CMake 3.20+
- A C++23 compiler
- BLAS and LAPACK
- OpenMP

macOS's Accelerate library provides BLAS and LAPACK natively. On Linux, these
can be installed, e.g., via `apt` as `libopenblas-dev` and `liblapack-dev` on
Debian-based distributions or equivalents. Notice that the OpenBLAS vendor
specifically is required. Currently, Windows is not supported, although you can
use the [Windows Subsystem for Linux (WSL)](https://github.com/microsoft/WSL).
With dependencies sorted, the following can be used to build the software:

```sh
git clone https://github.com/lvthnn/knoop.git
cmake -S knoop -B knoop/build -DCMAKE_BUILD_TYPE=Release
cmake --build knoop/build --target knoop --parallel
```

This will create the `knoop` binary inside of `knoop/build`.

## Usage

**knoop** simulations are configured and run via Lua scripts. Small simulations
are relatively easy to configure:
```lua
local genome = genome {
    n_loci = 1000,
    locus_freq = 0.5,
    locus_rec = 0.5,
    locus_mut = 0.0
}

local height = phenotype {
    name = "height",
    n_loci = 500,
    effects = normal{},
    var_genetic = 0.45,
    var_environmental = 0.55
}

local estimators = { estimator.heritability }

simulation {
    n_individuals = 5000,
    n_generations = 20,
    n_replicates = 20,
    genome = genome,
    phenotypes = phenotypes{ phenotypes = {height} },
    estimators = estimators,
    random_seed = 22112026
}
```
A configuration can be run like so:
```sh
knoop simulation.lua
```
Running the above in the command line will produce two files: `results.h5`, a
HDF5 file with parameters, raw and summarised data, and a log file named
`results.log`.

## Contributing

**knoop** is open source software under active development. Befittingly,
contributions are encouraged and welcome. Consult the
[contribution guide](CONTRIBUTING.md) for tips on where to start and how to
contribute effectively.
