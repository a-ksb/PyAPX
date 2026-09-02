#!/usr/bin/env python3
"""Generate candidates.csv for the h-BCN (3x3) toy Ising example.

18 sites on a (3x3) periodic honeycomb, composition 6B + 6C + 6N,
site_1 fixed to C (translation-equivalence removal). A fixed-seed random
subset of the full configuration space is used as the candidate pool.
"""
import numpy as np
import pandas as pd

NUM_POOL = 2000
SEED = 42
SPECIES = np.array(["B", "C", "N"])


def generate_candidates(num_pool=NUM_POOL, seed=SEED,
                        output_file="./candidates.csv"):
    rng = np.random.default_rng(seed)
    base = np.array([0] * 6 + [1] * 5 + [2] * 6, dtype=np.int8)  # site_1 = C fixed
    seen = {}
    while len(seen) < num_pool:
        batch = rng.permuted(np.tile(base, (num_pool, 1)), axis=1)
        full = np.concatenate(
            [np.full((batch.shape[0], 1), 1, dtype=np.int8), batch], axis=1)
        for row in full:
            if len(seen) >= num_pool:
                break
            seen.setdefault(row.tobytes(), row)
    pool = np.array(list(seen.values()), dtype=np.int8)
    assert pool.shape == (num_pool, 18)
    assert (pool[:, 0] == 1).all()

    df = pd.DataFrame(SPECIES[pool],
                      columns=[f"site_{j + 1}" for j in range(18)])
    df.insert(0, "structure_id", np.arange(num_pool))
    df.to_csv(output_file, index=False)
    print(f"Candidates saved to {output_file} ({num_pool} configurations)")
    return df


if __name__ == "__main__":
    df = generate_candidates()
    print(df.head().to_string(index=False))
