"""
PyAPX - Utility Module
Provides common utility functions
"""

import os
import pandas as pd

def apx_print(message):
    """
    Print message with apx prefix
    """
    print(f"(apx) {message}", flush=True)

def get_atom_types_and_num_sites_from_candidates():
    """
    Function to get atom types and number of sites from structure ID 0 in candidates.csv
    
    Returns:
        tuple: (atom_types, num_sites)
            - atom_types: List of atom types
            - num_sites: Number of sites
    """
    # Read only the row with structure ID 0
    df = pd.read_csv("candidates.csv", nrows=1)
    first_row = df.iloc[0, 1:].values  # Atomic configuration of structure ID 0 (excluding structure ID column)
    
    # Extract unique atom types and sort
    atom_types = list(set(first_row))
    atom_types.sort()
    
    # Number of sites is the length of atomic configuration
    num_sites = len(first_row)
    
    return atom_types, num_sites

def read_card(file_path, card_name):
    """
    Function to read a specific card from input file
    
    Args:
        file_path (str): Path to input file
        card_name (str): Card name
    
    Returns:
        list: List of card content lines
    """
    card_data = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        inside_card = False

        for line in lines:
            stripped_line = line.strip()

            # Detect start of specified card
            if stripped_line.startswith(card_name):
                inside_card = True
                continue

            # Stop reading when encountering another card or empty line
            # Detect uppercase card names (but exclude lowercase data values)
            if inside_card and (not stripped_line or 
                              (stripped_line.isupper() and not stripped_line.startswith('#'))):
                break

            # Read data lines
            if inside_card:
                card_data.append(stripped_line)

    return card_data

def get_max_structure_id():
    """
    Function to get maximum structure ID from candidates.csv
    
    Returns:
        int: Maximum structure ID (None if failed)
    """
    try:
        candidates_path = "candidates.csv"
        with open(candidates_path, "r") as file:
            # Skip header
            next(file)
            # Get last line
            for line in file:
                pass
            max_structure_id = int(line.strip().split(',')[0])
        return max_structure_id
    except FileNotFoundError:
        apx_print("candidates.csv not found")
        return None
    except (ValueError, IndexError):
        apx_print("candidates.csv format is incorrect")
        return None

def read_card_value(file_path, card_name):
    """
    Read card value from input file (format with "=")
    
    Args:
        file_path (str): Path to input file
        card_name (str): Card name
    
    Returns:
        str: Card value (None if not found)
    """
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith(f"{card_name} ="):
                # Extract value after "="
                value = stripped_line.split("=", 1)[1].strip()
                return value
        return None
    except Exception as e:
        apx_print(f"Error reading card value {card_name}: {e}")
        return None

def read_energy_evaluator():
    """
    Read energy evaluator settings from apx.in
    
    Returns:
        str: Energy evaluator name
            - "qe": Quantum ESPRESSO
            - "vasp": VASP
            - "custom:module.function": User-defined function
              Example: "custom:my_evaluator.run_calculation"
    """
    try:
        evaluator = read_card_value("apx.in", "ENERGY_EVALUATOR")
        if evaluator:
            return evaluator
        else:
            return "qe"  # Default
    except Exception as e:
        apx_print(f"Error reading energy evaluator settings: {e}")
        return "qe"  # Default

def read_sampling_sequence():
    """
    Read sampling settings from apx.in
    
    Returns:
        list: List of sampling tasks
    """
    try:
        tasks = []
        
        # Read random sampling settings
        random_sampling = read_card_value("apx.in", "RANDOM_SAMPLING")
        if random_sampling:
            try:
                num_iterations = int(random_sampling)
                tasks.append(("RANDOM_SAMPLING", num_iterations))
            except (ValueError, IndexError):
                apx_print("RANDOM_SAMPLING value is incorrect")
        
        # Read Bayesian sampling settings
        bayes_sampling = read_card_value("apx.in", "BAYES_SAMPLING")
        if bayes_sampling:
            try:
                num_iterations = int(bayes_sampling)
                tasks.append(("BAYES_SAMPLING", num_iterations))
            except (ValueError, IndexError):
                apx_print("BAYES_SAMPLING value is incorrect")
        
        return tasks
    
    except Exception as e:
        apx_print(f"Error reading sampling settings: {e}")
        return []

def get_current_sample_id():
    """
    Function to get current sample ID from samples.csv (for continuation)
    
    Returns:
        int: Next sample ID (0 if file doesn't exist)
    """
    samples_path = "samples.csv"
    
    if not os.path.exists(samples_path):
        return 0
    
    try:
        with open(samples_path, "r") as file:
            lines = file.readlines()
            if len(lines) > 1:  # If there is data other than header
                # Get sample ID from last line and add 1
                last_line = lines[-1].strip()
                if last_line:
                    last_sample_id = int(last_line.split(',')[0])
                    return last_sample_id + 1
        return 0
    except Exception as e:
        apx_print(f"Error reading existing samples.csv: {e}")
        return 0


def ensure_samples_csv_header():
    """
    Function to create samples.csv header (only if file doesn't exist)
    """
    samples_csv_path = "samples.csv"
    
    if not os.path.exists(samples_csv_path):
        # Get number of sites from structure ID 0
        _, num_sites = get_atom_types_and_num_sites_from_candidates()
        
        header_parts = ["sample_id", "structure_id"]
        for i in range(num_sites):
            header_parts.append(f"site_{i+1}")
        header_parts.append("energy")
        
        with open(samples_csv_path, "w") as samples_file:
            samples_file.write(','.join(header_parts) + '\n')

def read_encode_type_setting():
    """
    Read encode type setting from apx.in
    
    Returns:
        tuple: (encode_type, weight)
            - encode_type (str): Encoding method ("OH", "NA", or "NAmod")
            - weight (float): Weight for neighbor atom encoding
    """
    try:
        encode_type = read_card_value("apx.in", "ENCODE_TYPE")
        if not encode_type:
            encode_type = "OH"  # Default
        
        weight_str = read_card_value("apx.in", "WEIGHT")
        weight = float(weight_str) if weight_str else 0.0
        
        return encode_type, weight
    except Exception as e:
        apx_print(f"Error reading encode type setting: {e}")
        return "OH", 0.0  # Default values

def read_wl_setting():
    """
    Read Weisfeiler-Lehman (WL) encoding setting from apx.in

    Returns:
        int: Number of WL refinement iterations (WL_H card, default 1)
    """
    try:
        wl_h_str = read_card_value("apx.in", "WL_H")
        wl_h = int(wl_h_str) if wl_h_str else 1
        if wl_h < 0:
            apx_print("WL_H must be >= 0, falling back to default (1)")
            return 1
        return wl_h
    except Exception as e:
        apx_print(f"Error reading WL setting: {e}")
        return 1  # Default value

def read_encoding_setting():
    """
    Read encoding setting from apx.in
    
    Returns:
        bool: Whether to perform encoding preprocessing before sampling
    """
    try:
        encoding_str = read_card_value("apx.in", "ENCODING")
        encoding = encoding_str.lower() == "true" if encoding_str else False
        return encoding
    except Exception as e:
        apx_print(f"Error reading encoding setting: {e}")
        return False  # Default value

def read_dimension_reduction_setting():
    """
    Read dimension reduction settings from apx.in
    
    Returns:
        tuple: (use_dimension_reduction, method, params)
            - use_dimension_reduction (bool): Whether to use dimension reduction
            - method (str): Method ("PCA" or "AUTOENCODER")
            - params (dict): Parameters for the selected method
    """
    try:
        use_dimension_reduction_str = read_card_value("apx.in", "USE_DIMENSION_REDUCTION")
        use_dimension_reduction = use_dimension_reduction_str.lower() == "true" if use_dimension_reduction_str else False
        
        if use_dimension_reduction:
            # Read method
            method = read_card_value("apx.in", "DIMENSION_REDUCTION_METHOD")
            if not method:
                method = "PCA"  # Default
            
            random_state_str = read_card_value("apx.in", "AUTOENCODER_RANDOM_STATE")
            random_state = int(random_state_str) if random_state_str else 42
            
            if method.upper() == "PCA":
                # Read PCA-specific parameters
                pca_n_components_str = read_card_value("apx.in", "PCA_N_COMPONENTS")
                if pca_n_components_str:
                    if pca_n_components_str.lower() == "none":
                        pca_n_components = None  # sklearn default
                    else:
                        pca_n_components = int(pca_n_components_str)
                else:
                    pca_n_components = None  # sklearn default
                
                # For PCA, random_state is optional (only needed for randomized SVD)
                # Use sklearn default (None) unless explicitly set
                pca_random_state_str = read_card_value("apx.in", "PCA_RANDOM_STATE")
                pca_random_state = int(pca_random_state_str) if pca_random_state_str else None
                
                params = {
                    'n_components': pca_n_components,
                    'random_state': pca_random_state
                }
            elif method.upper() == "AUTOENCODER":
                # Read Auto Encoder specific parameters
                latent_dim_str = read_card_value("apx.in", "AUTOENCODER_LATENT_DIM")
                latent_dim = int(latent_dim_str) if latent_dim_str else 20
                
                hidden_layers_str = read_card_value("apx.in", "AUTOENCODER_HIDDEN_LAYERS")
                if hidden_layers_str:
                    hidden_layers = tuple(int(x.strip()) for x in hidden_layers_str.split(','))
                else:
                    hidden_layers = (128, 64)
                
                learning_rate_str = read_card_value("apx.in", "AUTOENCODER_LEARNING_RATE")
                learning_rate = float(learning_rate_str) if learning_rate_str else 0.001
                
                epochs_str = read_card_value("apx.in", "AUTOENCODER_EPOCHS")
                epochs = int(epochs_str) if epochs_str else 100
                
                batch_size_str = read_card_value("apx.in", "AUTOENCODER_BATCH_SIZE")
                batch_size = int(batch_size_str) if batch_size_str else 1024
                
                params = {
                    'latent_dim': latent_dim,
                    'hidden_layers': hidden_layers,
                    'learning_rate': learning_rate,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'random_state': random_state
                }
            else:
                apx_print(f"Unknown dimension reduction method: {method}")
                return False, None, None
        else:
            method = None
            params = None
        
        return use_dimension_reduction, method, params
    except Exception as e:
        apx_print(f"Error reading dimension reduction settings: {e}")
        return False, None, None  # Default values

def read_autoencoder_setting():
    """
    Read auto encoder settings from apx.in (for backward compatibility)
    
    Returns:
        tuple: (use_autoencoder, autoencoder_params)
            - use_autoencoder (bool): Whether to use auto encoder
            - autoencoder_params (dict): Auto encoder parameters
    """
    use_dimension_reduction, method, params = read_dimension_reduction_setting()
    
    if use_dimension_reduction and method and method.upper() == "AUTOENCODER":
        return True, params
    else:
        return False, None

def read_physbo_setting():
    """
    Read PHYSBO settings from apx.in
    
    Returns:
        tuple: (score, num_rand_basis)
            - score (str): Score function ("TS", "EI", "PI")
            - num_rand_basis (int): Number of random basis (only used for "TS")
    """
    try:
        score = read_card_value("apx.in", "SCORE")
        if score:
            # Read num_rand_basis only if score is "TS"
            if score == "TS":
                num_rand_basis_str = read_card_value("apx.in", "NUM_RAND_BASIS")
                num_rand_basis = int(num_rand_basis_str) if num_rand_basis_str else 3000
            else:
                num_rand_basis = 3000  # Default value for TS, not used for other scores
            return score, num_rand_basis
        else:
            return "TS", 3000  # Default values
    except Exception as e:
        apx_print(f"Error reading PHYSBO settings: {e}")
        return "TS", 3000  # Default values

def read_optimizer():
    """
    Read optimizer settings from apx.in
    
    Returns:
        str: Optimizer name ("physbo", etc.)
    """
    try:
        optimizer = read_card_value("apx.in", "OPTIMIZER")
        if optimizer:
            return optimizer
        else:
            return "physbo"  # Default
    except Exception as e:
        apx_print(f"Error reading optimizer settings: {e}")
        return "physbo"  # Default

def read_parallel_command():
    """
    Read parallel execution command prefix from apx.in (part before DFT code executable)
    
    Returns:
        str: Parallel execution command prefix (e.g., "mpiexec", "mpirun -np 8")
             Default: "mpiexec"
    """
    try:
        parallel_command = read_card_value("apx.in", "PARALLEL_COMMAND")
        if parallel_command:
            return parallel_command.strip()
        else:
            # Default command prefix
            return "mpiexec"
    except Exception as e:
        apx_print(f"Error reading parallel command settings: {e}")
        # Default command prefix
        return "mpiexec"

def load_encoded_data_from_cache():
    """
    Load pre-encoded data from cache
    
    Returns:
        tuple: (success, data)
            - success (bool): True if successful, False otherwise
            - data (np.ndarray): Encoded data if successful, None otherwise
    """
    import os
    import pickle
    import numpy as np
    
    cache_file = "encoded_candidates.pkl"
    if not os.path.exists(cache_file):
        apx_print(f"Error: Cache file {cache_file} not found.")
        apx_print("Please run encoding first by setting ENCODING = True")
        return False, None
    
    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return True, data
    except Exception as e:
        apx_print(f"Error loading cache file: {e}")
        return False, None

def read_use_initial_density():
    """
    Read USE_INITIAL_DENSITY setting from apx.in
    
    Returns:
        bool: Whether to use initial density (True if USE_INITIAL_DENSITY = True, False otherwise)
    """
    try:
        use_initial_density_str = read_card_value("apx.in", "USE_INITIAL_DENSITY")
        use_initial_density = use_initial_density_str.lower() == "true" if use_initial_density_str else False
        return use_initial_density
    except Exception as e:
        apx_print(f"Error reading USE_INITIAL_DENSITY setting: {e}")
        return False  # Default value

def read_qe_prefix_and_outdir():
    """
    Read prefix and outdir from qe_template.in
    
    Returns:
        tuple: (prefix, outdir)
            - prefix (str): Prefix value from qe_template.in (None if not found)
            - outdir (str): Outdir value from qe_template.in (None if not found)
    """
    try:
        qe_template_path = "qe_template.in"
        if not os.path.exists(qe_template_path):
            return None, None
        
        prefix = None
        outdir = None
        
        with open(qe_template_path, "r") as file:
            lines = file.readlines()
            inside_control = False
            
            for line in lines:
                stripped_line = line.strip()
                
                # Skip empty lines and comments
                if not stripped_line or stripped_line.startswith("#"):
                    continue
                
                # Detect start of &control section
                if stripped_line.startswith("&control") or stripped_line.startswith("&CONTROL"):
                    inside_control = True
                    continue
                
                # Stop reading when encountering another section or end of control
                if inside_control:
                    if stripped_line.startswith("/"):
                        break
                    if stripped_line.startswith("&"):
                        break
                    
                    # Parse prefix (handle both 'prefix = ...' and 'prefix=' formats)
                    if "prefix" in stripped_line.lower() and "=" in stripped_line:
                        # Remove comment if present
                        line_without_comment = stripped_line.split("#")[0].strip()
                        if "=" in line_without_comment:
                            value = line_without_comment.split("=", 1)[1].strip()
                            # Remove quotes if present
                            prefix = value.strip("'\"")
                    
                    # Parse outdir (handle both 'outdir = ...' and 'outdir=' formats)
                    if "outdir" in stripped_line.lower() and "=" in stripped_line:
                        # Remove comment if present
                        line_without_comment = stripped_line.split("#")[0].strip()
                        if "=" in line_without_comment:
                            value = line_without_comment.split("=", 1)[1].strip()
                            # Remove quotes if present
                            outdir = value.strip("'\"")
        
        return prefix, outdir
    except Exception as e:
        apx_print(f"Error reading prefix and outdir from qe_template.in: {e}")
        return None, None
