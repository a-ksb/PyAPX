</div>
<div align="center">
  <img src="logo/logo_banner.png" alt="PyAPX Logo" width="450">
</div>

[![arXiv](https://img.shields.io/badge/arXiv-2511.17972-b31b1b.svg)](http://arxiv.org/abs/2511.17972)
[![DOI](https://zenodo.org/badge/1092716137.svg)](https://doi.org/10.5281/zenodo.21351586)

# PyAPX - Python Toolkit for Atomic Configuration Pattern Exploration

A toolkit that integrates DFT codes with Bayesian optimization to explore stable atomic configurations in materials <sup>[1]</sup>.

## Requirements

- Python 3.9.6 (tested)
- numpy == 1.26.4
- pandas == 2.2.3
- [physbo](https://github.com/issp-center-dev/PHYSBO) == 3.1.0
- scikit-learn == 1.6.1
- scipy == 1.13.1

### DFT code

- [Quantum ESPRESSO](https://www.quantum-espresso.org/) v7.3.1 (tested)

## Installation

```bash
git clone https://github.com/a-ksb/PyAPX.git
cd PyAPX
pip install -r requirements.txt
```

## Quick Start

### Example Using an Ising Model (Instead of a DFT Code)

For details about this material system, please refer to articles <sup>[2,3]</sup>.

```bash
cd PyAPX/examples/H_GaN0001_6x6
python gen_candidates.py
PYTHONPATH=../.. python -m pyapx.cli
python visualize_results.py
```

## How to Use

### Example Using Quantum ESPRESSO

For details about this material system, please refer to the article <sup>[4]</sup>.

```bash
cd PyAPX/examples/h-BCN_3x3
```

Users need to prepare the following three files:

- `apx.in`: PyAPX input file
- `qe_template.in`: Quantum ESPRESSO input template
- `candidates.csv`: List of candidate atomic configurations

In `apx.in`, users can set parameters as follows:

```ini
ENCODING = True
RANDOM_SAMPLING = 2    # the number of random sampling
BAYES_SAMPLING = 2    # the number of Bayesian sampling
ENERGY_EVALUATOR = qe    # "qe" or custom:module.function
#PARALLEL_COMMAND = mpirun -np 8    # mpi setting for dft, Default: "mpiexec"
OPTIMIZER = physbo
```

When ENCODING = True, `encoded_candidates.pkl` is generated, which is required for Bayesian sampling.
Then, schedule the number of random sampling and Bayesian sampling iterations. At least two initial data points are required for Bayesian sampling.
PyAPX (v1.0.0) currently supports only Quantum ESPRESSO as a DFT code.
As an energy evaluator, you can also specify user-defined functions instead of DFT codes (see Quick Start example).

```ini
# settings for encoding
ENCODE_TYPE = NAmod    # encoding method: "OH", "NA" or "NAmod"
WEIGHT = 0.3    # parameter for "NA" and "NAmod"
```

You can choose from the following encoding methods: one-hot (OH) encoding, neighbor-atom (NA) encoding, and modified neighbor-atom (NAmod) encoding. For details, please refer to the article <sup>[1]</sup>.

```ini
# settings for physbo
SCORE = TS    # acquisition function: "TS", "EI" or "PI"
NUM_RAND_BASIS = 3000    # the number of basis functions
```

You can choose from the following acquisition functions: Thompson Sampling (TS), Expected Improvement (EI), and Probability of Improvement (PI). For details, please refer to the [PHYSBO documentation](https://issp-center-dev.github.io/PHYSBO/manual/master/en/index.html).

```ini
# settings for "NA", "NAmod"
NEIGHBOR_SITES
10   12   18    # site_1's neighbors are site_10, site_12 and site_18
10   11   16    # site_2's neighbors are site_10, site_11 and site_16
11   12   17    # site_3's neighbors are site_11, site_12 and site_17
      .
      .
      .
```

When using NA or NAmod encoding, list the neighboring sites in order from site_1 to define the site network.

In `qe_template.in`, users need to include atomic coordinates in the [Quantum ESPRESSO input file](https://www.quantum-espresso.org/Doc/INPUT_PW.html) as follows:

```ini
&control
calculation = 'vc-relax'    # DFT calc settings
      .
      .
      .

ATOMIC_SPECIES
   C   12.01060   C_paw.UPF
   B   10.81350   B_paw.UPF
   N   14.00650   N_paw.UPF
ATOMIC_POSITIONS {crystal}
X   0.0000   0.0000   0.0000    # site_1
X   0.0000   0.3333   0.0000    # site_2
X   0.0000   0.6667   0.0000    # site_3
      .
      .
      .
```

Here, sites that are subject to atomic rearrangement should be set with the element as a wildcard "X".

In this example, `candidates.csv` is generated using the following enumeration code, which contains as many as 5,717,712 candidate configurations.

```bash
julia gen_candidates.jl    # julia 1.10.3 (tested)
```

```ini
structure_id,site_1,site_2,site_3,...,site_18
0,C,B,B,B,B,B,B,C,C,C,C,C,N,N,N,N,N,N
1,C,B,B,B,B,B,B,C,C,C,C,N,C,N,N,N,N,N
2,C,B,B,B,B,B,B,C,C,C,C,N,N,C,N,N,N,N
      .
      .
      .
5717711,C,N,N,N,N,N,N,C,C,C,C,C,B,B,B,B,B,B
```

Users are required to prepare `candidates.csv` in the format shown above.
The sites in `candidates.csv` correspond to the sites specified by "X" in `qe_template.in` in order.

Finally, ensure that the three files `apx.in`, `qe_template.in`, and `candidates.csv` are present in the current directory, and run the following command with the PYTHONPATH set to the project root directory where the `pyapx` directory is located.

```bash
PYTHONPATH=../.. python -m pyapx.cli
```

Output files:

- `samples.csv`: Sampling results and energy values
- `dft_calc/qe_sample_[ID].in`: Generated QE input files
- `dft_calc/qe_sample_[ID].out`: QE calculation output files

## References

When publishing the results using PyAPX, we hope that you cite the following article <sup>[1]</sup>. Additionally, please follow the guidelines for [PHYSBO](https://github.com/issp-center-dev/PHYSBO?tab=readme-ov-file#license) and [Quantum ESPRESSO](https://www.quantum-espresso.org/quote/) to acknowledge their usage and cite the relevant references.

[1] A. Kusaba et al., "PyAPX: Python toolkit for atomic configuration pattern exploration", arXiv:2511.17972 [cond-mat.mtrl-sci].

PyAPX originates from following our previous studies <sup>[2,3,4]</sup>.

[2] A. Kusaba et al., "Exploration of a large-scale reconstructed structure on GaN(0001) surface by Bayesian optimization", *Applied Physics Letters* **120**, 021602 (2022).

[3] K. Kawka et al., "Augmentation of the Electron Counting Rule with Ising Model", *Journal of Applied Physics* **135**, 225302 (2024).

[4] T. Hara et al., "Exploration of Stable Atomic Configurations in Graphene-like BCN Systems by Density Functional Theory and Bayesian Optimization", *Crystal Growth & Design* **25**, 6719-6726 (2025).
