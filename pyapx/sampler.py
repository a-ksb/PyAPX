"""
PyAPX - Sampling Module
Provides random sampling and Bayesian optimization sampling functionality
"""

import random
import os
import pandas as pd
import numpy as np
import physbo
from .encoder import encode_options
_rng = random.Random()

def run_random_sampling(max_structure_id):
    """
    Execute random sampling function
    
    Args:
        max_structure_id (int): Maximum structure ID
    
    Returns:
        int: Selected structure ID
    """
    # Random sampling
    sampled_id = _rng.randint(0, max_structure_id)
    
    return sampled_id

def run_physbo_sampling(current_sample_id):
    """
    Execute PHYSBO sampling function
    
    Args:
        current_sample_id (int): Current sample ID to use as random seed
    
    Returns:
        int: Selected structure ID
    """
    # Read PHYSBO settings from apx.in
    from .utils import read_physbo_setting, read_physbo_centering_setting, load_encoded_data_from_cache
    score, num_rand_basis = read_physbo_setting()
    
    # Load pre-encoded data from cache
    success, X = load_encoded_data_from_cache()
    if not success:
        return None
    
    X = X.astype(np.float32)
    if read_physbo_centering_setting():
        # Column-wise standardization (z-score) of the feature matrix
        X = physbo.misc.centering(X)
    # else: pass the encoded features to PHYSBO as-is

    # Load initial data from samples.csv
    samples_df = pd.read_csv("samples.csv")
    t_initial = -samples_df['energy'].values  # Convert minimization problem to maximization problem
    calculated_ids = samples_df['structure_id'].values

    # Create policy with initial data
    policy = physbo.search.discrete.policy(test_X=X, initial_data=[calculated_ids, t_initial])
    policy.set_seed(_rng.randint(0, 2**31 - 1))

    # Perform Bayesian search using settings from apx.in
    if score == "TS":
        # Use num_rand_basis only for Thompson Sampling
        actions = policy.bayes_search(
            max_num_probes=1, 
            simulator=None, 
            score=score, 
            interval=0, 
            num_rand_basis=num_rand_basis
        )
    else:
        # For EI, PI, etc., don't use num_rand_basis
        actions = policy.bayes_search(
            max_num_probes=1, 
            simulator=None, 
            score=score, 
            interval=0
        )

    return actions[0]