"""Custom energy evaluator for the h-BCN (3x3) toy system.

Ising-like nearest-neighbour bond counting E = sum_edges J[s_i, s_j]
mimicking h-BCN chemistry (ground state: B-N / C-C stripes). The bond
graph is read from the NEIGHBOR_SITES card in apx.in, so the same file
defines both the encoding neighbourhood and the energy model.
"""
import numpy as np
import pandas as pd

SPECIES = {"B": 0, "C": 1, "N": 2}

# J(B,N) = -1.0, J(C,C) = -0.9, J(B,C) = J(C,N) = -0.4, J(B,B) = J(N,N) = +0.6
J = np.zeros((3, 3))
J[0, 2] = J[2, 0] = -1.0
J[1, 1] = -0.9
J[0, 1] = J[1, 0] = -0.4
J[1, 2] = J[2, 1] = -0.4
J[0, 0] = +0.6
J[2, 2] = +0.6

_edges_cache = None


def _read_edges():
    """Unique undirected edges from the NEIGHBOR_SITES card in apx.in."""
    global _edges_cache
    if _edges_cache is not None:
        return _edges_cache
    card = []
    with open("apx.in") as f:
        inside = False
        for line in f:
            s = line.strip()
            if s.startswith("NEIGHBOR_SITES"):
                inside = True
                continue
            if inside and (not s or (s.isupper() and not s.startswith("#"))):
                break
            if inside:
                card.append(s)
    edges = set()
    for i, line in enumerate(card):
        for j in map(int, line.split()):
            edges.add((min(i, j - 1), max(i, j - 1)))
    _edges_cache = np.array(sorted(edges), dtype=np.int64)
    return _edges_cache


def calculate_bond_energy(atomic_config):
    """E = sum over edges of J[s_i, s_j] for one configuration."""
    s = np.array([SPECIES[a] for a in atomic_config], dtype=np.int64)
    e = _read_edges()
    return float(J[s[e[:, 0]], s[e[:, 1]]].sum())


def run_ising_calculation(sample_id, structure_id):
    """
    Custom energy calculation function (PyAPX evaluator interface).

    Args:
        sample_id (int): Sample ID
        structure_id (int): Structure ID

    Returns:
        tuple: (success, total_energy, atomic_config, error_message)
    """
    try:
        df = pd.read_csv("candidates.csv")
        row = df[df.iloc[:, 0] == structure_id]
        if row.empty:
            raise ValueError(f"Structure ID {structure_id} not found in candidates.csv")
        atomic_config = row.iloc[0, 1:].values.tolist()

        energy = calculate_bond_energy(atomic_config)
        print(f"Ising energy calculation - Sample {sample_id}, "
              f"Structure {structure_id}: {energy:.6f}")
        print()
        return True, energy, atomic_config, None

    except Exception as e:
        return False, None, None, str(e)
