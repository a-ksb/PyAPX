"""Custom energy evaluator using UMA."""

import csv
import time
from functools import lru_cache
from pathlib import Path
from ase import Atoms
from ase.io import write
from ase.filters import UnitCellFilter
from ase.optimize import BFGS
from fairchem.core import FAIRChemCalculator

SCRIPT_DIR = Path(__file__).resolve().parent
CANDIDATES_PATH = SCRIPT_DIR / "candidates.csv"
TEMPLATE_PATH = SCRIPT_DIR / "qe_template.in"
MODEL_PATH = SCRIPT_DIR / "uma-m-1p1.pt"
OUTPUT_DIR = SCRIPT_DIR / "relaxed_structures"
TASK_NAME = "omat"
DEVICE = "cpu"
FMAX = 0.05  # Force convergence criterion for BFGS relaxation (eV/Angstrom).
STEPS = 200  # Maximum number of BFGS steps.
RELAX_CELL = True  # True: optimize cell and atoms. False: optimize atoms only.
OUTPUT_FORMAT = "cif"  # "cif" or "vasp"

def read_qe_template(template_file):
    """Read cell, coordinate mode, symbols, and coordinates from qe_template.in."""
    cell = []
    symbols = []
    coordinates = []
    coordinate_mode = None
    section = None

    with open(template_file) as file:
        for raw_line in file:
            line = raw_line.split("!", 1)[0].split("#", 1)[0].strip()
            if not line:
                continue

            upper = line.upper()
            if upper.startswith("CELL_PARAMETERS"):
                section = "cell"
                continue
            if upper.startswith("ATOMIC_POSITIONS"):
                section = "positions"
                if "{" in line and "}" in line:
                    coordinate_mode = line.split("{", 1)[1].split("}", 1)[0].strip().lower()
                continue

            if section == "cell" and len(cell) < 3:
                cell.append([float(value) for value in line.split()[:3]])
            elif section == "positions":
                parts = line.split()
                symbols.append(parts[0])
                coordinates.append([float(value) for value in parts[1:4]])

    return cell, coordinate_mode, symbols, coordinates


def build_atoms(structure_id):
    """Build an ASE Atoms object from qe_template.in and candidates.csv."""
    # Read atomic configuration from candidates.csv.
    with CANDIDATES_PATH.open(newline="") as file:
        reader = csv.reader(file)
        next(reader)  # header
        for row in reader:
            if int(row[0]) == int(structure_id):
                atomic_config = row[1:]
                break

    cell, coordinate_mode, template_symbols, coordinates = read_qe_template(TEMPLATE_PATH)

    # Assign elements to X sites in template order.
    symbols = []
    config_index = 0
    for template_symbol in template_symbols:
        if template_symbol.upper() == "X":
            symbols.append(atomic_config[config_index])
            config_index += 1
        else:
            symbols.append(template_symbol)

    if coordinate_mode == "crystal":
        atoms = Atoms(symbols=symbols, scaled_positions=coordinates, cell=cell, pbc=[True, True, True])
        return atoms, atomic_config

    raise ValueError("Use ATOMIC_POSITIONS {crystal} in qe_template.in.")


@lru_cache(maxsize=1)
def get_uma_calculator():
    """Load the UMA calculator once and reuse it for all samples."""
    return FAIRChemCalculator.from_model_checkpoint(
        name_or_path=str(MODEL_PATH),
        task_name=TASK_NAME,
        device=DEVICE,
    )


def run_uma_calculation(sample_id, structure_id):
    """
    PyAPX calls this function for each sampled structure.
    Build the structure, relax it with UMA, and return its energy.
    Returns:
        tuple: (success, total_energy, atomic_config, error_message)
    """
    try:
        atoms, atomic_config = build_atoms(structure_id)
        atoms.calc = get_uma_calculator()

        target = UnitCellFilter(atoms) if RELAX_CELL else atoms
        opt = BFGS(target)
        start_time = time.perf_counter()
        opt.run(fmax=FMAX, steps=STEPS)
        elapsed_time = time.perf_counter() - start_time

        energy = float(atoms.get_potential_energy())
        print(
            "UMA relaxation - "
            f"Sample {sample_id}, Structure {structure_id}: "
            f"{energy:.6f} eV, {elapsed_time:.2f} s"
        )
        print()

        OUTPUT_DIR.mkdir(exist_ok=True)
        output_path = OUTPUT_DIR / f"sample_{sample_id}.{OUTPUT_FORMAT}"
        write(str(output_path), atoms, format=OUTPUT_FORMAT)

        return True, energy, atomic_config, None
    except Exception as exc:
        return False, None, None, str(exc)
