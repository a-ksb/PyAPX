"""
PyAPX - Encoding Module
Provides feature encoding functionality for atomic configurations
"""

import numpy as np
import pandas as pd
import os
import pickle
from .utils import get_atom_types_and_num_sites_from_candidates, read_card

# Conditional imports for dimension reduction
SKLEARN_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# PyTorch imports for Auto Encoder
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    
    class AutoEncoder(nn.Module):
        """
        PyTorch Auto Encoder for dimension reduction
        """
        def __init__(self, input_dim, latent_dim, hidden_layers):
            super(AutoEncoder, self).__init__()
            
            # Build encoder
            encoder_layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_layers:
                encoder_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                prev_dim = hidden_dim
            
            # Final encoder layer (to latent space)
            encoder_layers.append(nn.Linear(prev_dim, latent_dim))
            
            # Build decoder
            decoder_layers = []
            prev_dim = latent_dim
            for hidden_dim in reversed(hidden_layers):
                decoder_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                prev_dim = hidden_dim
            
            # Final decoder layer (back to input dimension)
            decoder_layers.append(nn.Linear(prev_dim, input_dim))
            
            self.encoder = nn.Sequential(*encoder_layers[:-1])  # Exclude final linear layer
            self.latent_layer = encoder_layers[-1]  # Final encoder layer
            self.decoder = nn.Sequential(*decoder_layers)
            
        def forward(self, x):
            # Encode
            encoded = self.encoder(x)
            latent = self.latent_layer(encoded)
            
            # Decode
            decoded = self.decoder(latent)
            
            return decoded, latent
        
        def encode(self, x):
            """Extract latent representation"""
            encoded = self.encoder(x)
            latent = self.latent_layer(encoded)
            return latent
            
except ImportError:
    TORCH_AVAILABLE = False

def encode_options(encode="OH", weight=0, wl_h=1, use_dimension_reduction=False, reduction_method=None, reduction_params=None):
    """
    Perform feature encoding for atomic configurations
    
    Parameters:
    - encode (str): Encoding method ('OH', 'NA', 'NAmod', or 'WL').
      - 'OH': One-hot encoding.
      - 'NA': Neighbor atom encoding.
      - 'NAmod': Modified neighbor atom encoding.
      - 'WL': Weisfeiler-Lehman subtree encoding (concatenated per-level
        label histograms, L2-normalised). Requires the NEIGHBOR_SITES card.
    - weight (float, optional): Weight to use in (modified) neighbor atom encoding (default is 0).
    - wl_h (int, optional): Number of WL label-refinement iterations for 'WL' encoding (default is 1).
    - use_dimension_reduction (bool): Whether to apply dimension reduction after encoding (default is False).
    - reduction_method (str): Method for dimension reduction ('PCA' or 'AUTOENCODER').
    - reduction_params (dict): Parameters for dimension reduction (default is None).

    Returns:
    - np.ndarray: Encoded input data.
    """

    # Try to load from cache
    cache_filename = "encoded_candidates.pkl"
    
    # Use cache if available
    if os.path.exists(cache_filename):
        try:
            with open(cache_filename, 'rb') as f:
                cached_result = pickle.load(f)
            print(f"(apx) Loaded encoded data from cache: {cache_filename}")
            return cached_result
        except Exception as e:
            print(f"(apx) Cache loading failed: {e}, recalculating...")
    
    # Load candidates of atomic configuration
    # Read all data for encoding
    df = pd.read_csv("candidates.csv")
    x_candidates = df.values
    # Exclude first column (structure ID) and get only atomic configuration data
    x_candidates = x_candidates[:, 1:]  # Exclude structure ID column
    num_candidates = x_candidates.shape[0]
    num_sites = x_candidates.shape[1]
    
    # Get atom types from structure ID 0
    atom_types, _ = get_atom_types_and_num_sites_from_candidates()
    num_atom_types = len(atom_types)
    
    # Define neighbor_sites only if needed
    if encode in ["NA", "NAmod", "WL"]:
        neighbor_sites_data = read_card("apx.in", "NEIGHBOR_SITES")
        neighbor_sites = [list(map(int, line.split())) for line in neighbor_sites_data]

    def onehot():
        """Perform one-hot encoding."""
        print(f"(apx) Performing one-hot encoding...", flush=True)
        phi_candidates = np.zeros((num_candidates, num_sites * num_atom_types), dtype=float)
        for i in range(num_sites):
            for j, atom_type in enumerate(atom_types):
                phi_candidates[:, i * num_atom_types + j] = (x_candidates[:, i] == atom_type).astype(int)
        print(f"(apx) One-hot encoding completed.", flush=True)
        return phi_candidates

    def neighbor_atom():
        """Perform neighbor atom encoding."""
        print(f"(apx) Performing neighbor atom encoding...", flush=True)
        phi_candidates = onehot()
        for i in range(num_candidates):
            for j in range(num_sites):
                for k, neighbor_site in enumerate(neighbor_sites[j]):
                    for l, atom_type in enumerate(atom_types):
                        if x_candidates[i, neighbor_site - 1] == atom_type:
                            phi_candidates[i, (j * num_atom_types) + l] += weight
        print(f"(apx) Neighbor atom encoding completed.", flush=True)
        return phi_candidates

    def modified_neighbor_atom():
        """Perform modified neighbor atom encoding."""
        print(f"(apx) Performing modified neighbor atom encoding...", flush=True)
        phi_candidates = neighbor_atom()
        phi_candidates_mod = np.zeros((num_candidates, num_sites * (num_atom_types + 1)), dtype=float)
        for i in range(num_candidates):
            for j in range(num_sites):
                phi_candidates_mod[i][j * (num_atom_types + 1):j * (num_atom_types + 1) + num_atom_types] = phi_candidates[i][j * num_atom_types:j * num_atom_types + num_atom_types]

        for i in range(num_candidates):
            for j in range(num_sites):
                neighbor_mean = np.zeros(num_atom_types)
                for k, neighbor_site in enumerate(neighbor_sites[j]):
                    neighbor_mean += phi_candidates[i][(neighbor_site - 1) * num_atom_types:(neighbor_site - 1) * num_atom_types + num_atom_types]
                neighbor_mean /= len(neighbor_sites[j])
                sigma = 0
                for k, neighbor_site in enumerate(neighbor_sites[j]):
                    sigma += np.linalg.norm(
                        phi_candidates[i][(neighbor_site - 1) * num_atom_types:(neighbor_site - 1) * num_atom_types + num_atom_types] - neighbor_mean
                    ) ** 2
                sigma = np.sqrt(sigma / len(neighbor_sites[j]))
                phi_candidates_mod[i][(j + 1) * (num_atom_types + 1) - 1] = sigma
        print(f"(apx) Modified neighbor atom encoding completed.", flush=True)
        return phi_candidates_mod

    def weisfeiler_lehman():
        """Perform Weisfeiler-Lehman (WL) subtree encoding.

        Initial labels are the atom species; each WL iteration replaces the
        label of a site by a compressed label of (own label, sorted multiset
        of neighbour labels). The feature vector is the concatenation of the
        per-level label histograms (levels 0..wl_h), L2-normalised per row,
        so that the linear kernel <phi_x, phi_y> equals the cosine-normalised
        WL subtree kernel. The encoding is invariant under lattice symmetry
        operations (translations, rotations, mirrors) because it only counts
        labelled local patterns, not site indices.

        Fully vectorised over candidates; complexity O(num_candidates *
        num_sites * wl_h) label updates.
        """
        print(f"(apx) Performing Weisfeiler-Lehman encoding (h={wl_h})...", flush=True)
        # Species -> integer initial labels
        labels = np.empty((num_candidates, num_sites), dtype=np.int64)
        for j, atom_type in enumerate(atom_types):
            labels[x_candidates == atom_type] = j

        # Neighbour index table (0-based); ragged lists padded with -1
        max_deg = max(len(nb) for nb in neighbor_sites)
        nbr_idx = np.full((num_sites, max_deg), -1, dtype=np.int64)
        for j, nb in enumerate(neighbor_sites):
            nbr_idx[j, :len(nb)] = np.asarray(nb, dtype=np.int64) - 1
        pad_mask = nbr_idx < 0
        nbr_safe = np.where(pad_mask, 0, nbr_idx)

        blocks = []
        labs = labels
        n_labels = num_atom_types
        for level in range(wl_h + 1):
            # Per-candidate histogram of current labels
            flat = np.arange(num_candidates, dtype=np.int64)[:, None] * n_labels + labs
            hist = np.bincount(flat.ravel(), minlength=num_candidates * n_labels)
            blocks.append(hist.reshape(num_candidates, n_labels).astype(float))
            if level == wl_h:
                break
            # Label refinement: (own label, sorted neighbour labels) -> new label
            nb_labs = labs[:, nbr_safe]                       # (n, num_sites, max_deg)
            if pad_mask.any():
                nb_labs = np.where(pad_mask[None, :, :], np.int64(-1), nb_labs)
            nb_labs = np.sort(nb_labs, axis=2)
            keys = np.concatenate([labs[..., None], nb_labs], axis=2)
            keys = keys.reshape(num_candidates * num_sites, 1 + max_deg)
            uniq, inv = np.unique(keys, axis=0, return_inverse=True)
            labs = inv.reshape(num_candidates, num_sites)
            n_labels = uniq.shape[0]

        phi_candidates = np.concatenate(blocks, axis=1)
        # L2 normalisation (linear kernel == cosine WL subtree kernel)
        norms = np.sqrt((phi_candidates ** 2).sum(axis=1))
        norms[norms == 0] = 1.0
        phi_candidates /= norms[:, None]
        print(f"(apx) WL encoding completed. Feature dimension: {phi_candidates.shape[1]} "
              f"(levels: {[b.shape[1] for b in blocks]})", flush=True)
        return phi_candidates

    def apply_dimension_reduction(features, method, params):
        """Apply dimension reduction to features."""
        input_dim = features.shape[1]
        
        if method.upper() == "PCA":
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn is required for PCA but not available")
            
            from sklearn.decomposition import PCA
            
            # Use PCA for dimension reduction
            n_components = params.get('n_components', None)
            random_state = params.get('random_state', None)
            
            pca = PCA(n_components=n_components, random_state=random_state)
            
            # Fit and transform
            print(f"(apx) Training PCA for dimension reduction...", flush=True)
            latent_features = pca.fit_transform(features)
            
            return latent_features, pca, None
            
        elif method.upper() == "AUTOENCODER":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is required for Auto Encoder but not available")
            
            # Prepare data for PyTorch
            features_tensor = torch.FloatTensor(features)
            
            # Create auto encoder
            latent_dim = params['latent_dim']
            hidden_layers = params['hidden_layers']
            learning_rate = params['learning_rate']
            epochs = params['epochs']
            batch_size = params.get('batch_size', 1024)
            
            model = AutoEncoder(input_dim, latent_dim, hidden_layers)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Create DataLoader for batch processing
            dataset = TensorDataset(features_tensor, features_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Training with batch processing
            print(f"(apx) Training Auto Encoder for dimension reduction...", flush=True)
            model.train()
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                for batch_features, batch_targets in dataloader:
                    optimizer.zero_grad()
                    decoded, latent = model(batch_features)
                    loss = criterion(decoded, batch_targets)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches
                
                if (epoch + 1) % 20 == 0:
                    print(f"(apx) Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.6f}", flush=True)
            
            # Extract latent features
            model.eval()
            with torch.no_grad():
                # Process in batches to handle large datasets
                latent_features_list = []
                for batch_features, _ in dataloader:
                    batch_latent = model.encode(batch_features).numpy()
                    latent_features_list.append(batch_latent)
                
                latent_features = np.vstack(latent_features_list)
            
            return latent_features, model, None
        else:
            raise ValueError(f"Unknown dimension reduction method: {method}")

    # Perform encoding
    if encode == "OH":
        result = onehot()
    elif encode == "NA":
        result = neighbor_atom()
    elif encode == "NAmod":
        result = modified_neighbor_atom()
    elif encode == "WL":
        result = weisfeiler_lehman()
    else:
        raise ValueError("Invalid option. Choose 'OH', 'NA', 'NAmod', or 'WL'.")

    # Apply dimension reduction if requested
    if use_dimension_reduction and reduction_method and reduction_params:
        try:
            # Store original dimension before reduction
            original_dim = result.shape[1]
            result, reduction_model, encoder_model = apply_dimension_reduction(result, reduction_method, reduction_params)
            
            # Print dimension reduction results
            output_dim = result.shape[1]
            print(f"(apx) Dimension reduction completed.", flush=True)
            print(f"(apx) Input dimension: {original_dim} -> Latent dimension: {output_dim}", flush=True)
            
            if reduction_method.upper() == "PCA" and reduction_model is not None:
                explained_variance_ratio = reduction_model.explained_variance_ratio_.sum()
                print(f"(apx) Explained variance ratio: {explained_variance_ratio:.3f}", flush=True)
                
        except ImportError as e:
            print(f"(apx) Error: {e}", flush=True)
            print(f"(apx) Program terminated due to missing required library.", flush=True)
            raise SystemExit(1)
        except Exception as e:
            print(f"(apx) Error during dimension reduction: {e}", flush=True)
            print(f"(apx) Program terminated due to dimension reduction error.", flush=True)
            raise SystemExit(1)
    
    # Save to cache
    try:
        print(f"(apx) Saving encoded data to cache: {cache_filename}", flush=True)
        
        with open(cache_filename, 'wb') as f:
            pickle.dump(result, f)
        
        print(f"(apx) Saved encoded data to cache: {cache_filename}", flush=True)
    except Exception as e:
        print(f"(apx) Cache saving failed: {e}", flush=True)
    
    return result
