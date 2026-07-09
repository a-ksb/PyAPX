# h-BCN 3x3 Example with CHGNet

This example demonstrates PyAPX's custom energy-evaluator interface, where a
user-defined Python function relaxes h-BCN 3x3 candidate structures with CHGNet
and returns the relaxed total energy.

## Files

- `apx.in`: PyAPX input file.
- `gen_candidates.jl`: Julia script to generate `candidates.csv`.
- `qe_template.in`: Minimal structure template with `CELL_PARAMETERS` and
  `ATOMIC_POSITIONS {crystal}`.
- `my_evaluator.py`: User-defined energy-evaluator function using ASE and CHGNet.

## Tested Environment

This example was tested with the following environment:

- Python 3.12.3
- chgnet 0.4.2
- ase 3.29.0
- numpy 2.4.6
- pandas 2.2.3
- physbo 3.1.0
- scipy 1.18.0

## Preparation

Install CHGNet (the pretrained model weights are bundled with the package):

```bash
pip install chgnet
```

Then generate the candidate structures:

```bash
julia gen_candidates.jl
```

## Run

From this directory, run PyAPX with the repository root on `PYTHONPATH`:

```bash
PYTHONPATH=../.. python -m pyapx.cli
```

Relaxed structures are written to `relaxed_structures/`.
