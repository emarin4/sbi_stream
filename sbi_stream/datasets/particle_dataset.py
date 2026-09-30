
import os
import pickle
from pathlib import Path
from typing import List, Union, Optional
import h5py

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric import transforms as T
from tqdm import tqdm

from . import io_utils, preprocess_utils


LABEL_RENAME_MAP = {
    "time_impact": "impact_time",
    "v_rel_para": "v_para",
    "v_rel_perp": "v_perp",
    "angle_pos_at_impact": "angle_pos",
    "angle_vel_at_impact": "angle_vel",
    # direct matches (included for clarity, not strictly required)
    "phi1_impact_today": "phi1_impact_today",
    "impact_parameter": "impact_param",
}
 
# Fixed constant per spec -- broadcast to all sims in a file.
DELTA_PHI1_CONSTANT = 5.0
 
 
def _derive_labels(f):
    """
    Build the derived-quantity labels (log_mass, log_scale_radius,
    log_impact_parameter) by reusing preprocess_utils.calculate_derived_properties,
    so this stays in sync with the CSV-based pipeline instead of duplicating
    the log10 logic. Also computes delta_angle and delta_phi1, which are not
    covered by that function.
    """
    n_sims = f["mass"].shape[0]
 
    # calculate_derived_properties expects an 'impact_parameter' column name,
    # but the h5 file stores it as 'impact_param' -- rename on the way in.
    table = pd.DataFrame({
        "mass": f["mass"][:],
        "scale_radius": f["scale_radius"][:],
        "impact_parameter": f["impact_param"][:],
    })
    table = preprocess_utils.calculate_derived_properties(table)
 
    derived = {
        "log_mass": table["log_mass"].to_numpy(),
        "log_scale_radius": table["log_scale_radius"].to_numpy(),
        #"log_impact_parameter": table["log_impact_parameter"].to_numpy(),
        "delta_angle": f["angle_vel"][:] - f["angle_pos"][:],
        "delta_phi1": np.full(n_sims, DELTA_PHI1_CONSTANT, dtype=np.float64),
    }
    return derived
 
 
def read_raw_particle_datasets_h5(
    data_dir: Union[str, Path],
    features: List[str],
    labels: List[str],
    h5_files: Optional[List[str]] = None,
    num_subsamples: int = 1,
    num_per_subsample: int = None,
    num_per_subsample_min: int = None,
    num_per_subsample_max: int = None,
    phi1_min: Optional[float] = None,
    phi1_max: Optional[float] = None,
    uncertainty_model: Optional[str] = None,
    include_uncertainty: bool = False,
):
    """
    Read and process particle-level stream datasets stored as flat HDF5 files
    (no separate labels.csv -- labels are top-level per-sim datasets in the
    same file, and particle data is sliced using an explicit `ptr` dataset).
 
    Parameters
    ----------
    data_dir : str or Path
        Directory containing the perturbers_batch_*.h5 files.
    features : list of str
        Feature names to extract (must be top-level, per-particle datasets
        in the h5 file, e.g. ['phi1', 'phi2', 'pm1', 'pm2', 'vr', 'dist']).
    labels : list of str
        Label names as used downstream (e.g. config.labels). These are
        resolved via LABEL_RENAME_MAP / _derive_labels against the actual
        dataset names stored in the h5 file.
    h5_files : list of str, optional
        Explicit list of h5 filenames (relative to data_dir) to read. If
        None, reads every *.h5 file in data_dir.
    (remaining args identical in meaning to read_raw_particle_datasets)
 
    Returns
    -------
    list of Data
    """
    phi1_min = phi1_min if phi1_min is not None else -np.inf
    phi1_max = phi1_max if phi1_max is not None else np.inf
 
    data_dir = Path(data_dir)
    if h5_files is None:
        h5_files = sorted(data_dir.glob("*.h5"))
    else:
        h5_files = [data_dir / fn for fn in h5_files]
 
    graph_list = []
 
    for h5_path in h5_files:
        print(f"Reading in data from {h5_path}")
        with h5py.File(h5_path, "r") as f:
            ptr = f["ptr"][:]
            n_sims = len(ptr) - 1
 
            # Pull per-particle feature arrays once per file (not per sim)
            feat_arrays = {feat_name: f[feat_name][:] for feat_name in features}
 
            # Resolve labels: renamed direct reads + derived quantities
            derived = _derive_labels(f)
            label_arrays = {}
            for lbl in labels:
                if lbl in derived:
                    label_arrays[lbl] = derived[lbl]
                elif lbl in LABEL_RENAME_MAP:
                    label_arrays[lbl] = f[LABEL_RENAME_MAP[lbl]][:]
                elif lbl in f.keys():
                    label_arrays[lbl] = f[lbl][:]
                else:
                    raise KeyError(
                        f"Label '{lbl}' not found as a direct dataset, rename, "
                        f"or derived quantity. Available top-level keys: {list(f.keys())}"
                    )
 
            for j in tqdm(range(n_sims), desc=f"Processing streams in {h5_path.name}"):
                sl = slice(ptr[j], ptr[j + 1])
 
                phi1 = feat_arrays["phi1"][sl]
                phi2 = feat_arrays["phi2"][sl]
                feat = np.stack([feat_arrays[feat_name][sl] for feat_name in features], axis=1)
                label = np.array([label_arrays[lbl][j] for lbl in labels])
 
                mask = (phi1_min <= phi1) & (phi1 < phi1_max)
                phi1 = phi1[mask]
                phi2 = phi2[mask]
                feat = feat[mask]
 
                for _ in range(num_subsamples):
                    if num_per_subsample is not None:
                        phi1_ppr, phi2_ppr, feat_ppr = preprocess_utils.subsample_arrays(
                            [phi1, phi2, feat], num_per_subsample=num_per_subsample)
                    elif (num_per_subsample_min is not None) and (num_per_subsample_max is not None):
                        N = np.random.randint(num_per_subsample_min, num_per_subsample_max)
                        phi1_ppr, phi2_ppr, feat_ppr = preprocess_utils.subsample_arrays(
                            [phi1, phi2, feat], num_per_subsample=N)
                    else:
                        phi1_ppr, phi2_ppr, feat_ppr = phi1, phi2, feat
 
                    phi1_ppr, phi2_ppr, feat_ppr, _, feat_unc_ppr = preprocess_utils.add_uncertainty(
                        phi1_ppr, phi2_ppr, feat_ppr, features, uncertainty_model=uncertainty_model)
 
                    pos = np.stack([phi1_ppr, phi2_ppr], axis=1)
                    if uncertainty_model is not None and include_uncertainty:
                        feat_ppr = np.concatenate([feat_ppr, feat_unc_ppr], axis=1)
 
                    graph_data = Data(
                        x=torch.tensor(feat_ppr, dtype=torch.float32),
                        y=torch.tensor(label, dtype=torch.float32).unsqueeze(0),
                        pos=torch.tensor(pos, dtype=torch.float32),
                    )
                    graph_list.append(graph_data)
 
    print(f"Total number of graphs: {len(graph_list)}")
    return graph_list



def read_raw_particle_datasets(
    data_dir: Union[str, Path],
    features: List[str],
    labels: List[str],
    num_datasets: int = 1,
    start_dataset: int = 0,
    num_subsamples: int = 1,
    num_per_subsample: int = None,
    num_per_subsample_min: int = None,
    num_per_subsample_max: int = None,
    phi1_min: Optional[float] = None,
    phi1_max: Optional[float] = None,
    uncertainty_model: Optional[str] = None,
    include_uncertainty: bool = False,
):
    """
    Read and process particle-level stream datasets as PyTorch Geometric graphs.

    Parameters
    ----------
    data_dir : str or Path
        Path to the directory containing the stream data.
    features : list of str, optional
        List of feature names to extract.
    labels : list of str, optional
        List of labels to use for the regression.
    num_datasets : int, optional
        Number of datasets to read in. Default is 1.
    start_dataset : int, optional
        Index to start reading the dataset. Default is 0.
    num_subsamples : int, optional
        Number of subsamples to use. Default is 1.
    num_per_subsample : int, optional
        Number of particles per subsample. Default is None (use all particles).
    phi1_min : float, optional
        Minimum phi1 value to filter data.
    phi1_max : float, optional
        Maximum phi1 value to filter data.
    uncertainty_model : str, optional
        If not None, include measurement uncertainty. Either "present" or "future".
    include_uncertainty : bool, optional
        If True, include uncertainty features in the node features. This is
        only applicable if uncertainty_model is not None.

    Returns
    -------
    list of Data
        List of PyTorch Geometric Data objects, one per stream. Each Data object has:
        - x: node features (excluding phi1, phi2, and dist if dist is used in pos)
        - y: labels tensor
        - pos: position coordinates (phi1, phi2, dist) or (phi1, phi2) if dist not in features
    """
    # default args
    #phi1_min = phi1_min or -np.inf
    #phi1_max = phi1_max or np.inf

    phi1_min = phi1_min if phi1_min is not None else -np.inf
    phi1_max = phi1_max if phi1_max is not None else np.inf

    graph_list = []

    for i in range(start_dataset, start_dataset + num_datasets):
        label_fn = os.path.join(data_dir, f'labels.{i}.csv')
        data_fn = os.path.join(data_dir, f'data.{i}.hdf5')

        if os.path.exists(label_fn) & os.path.exists(data_fn):
            print('Reading in data from {}'.format(data_fn))
        else:
            print('Dataset {} not found. Skipping...'.format(i))
            continue

        # read in the data and label
        table = pd.read_csv(label_fn)
        table = preprocess_utils.calculate_derived_properties(table)
        print(table.columns.tolist())
        print(table[labels].head())
        data, ptr = io_utils.read_dataset(data_fn, unpack=False)

        for j in tqdm(range(len(table)), desc='Processing streams'):
            phi1 = data['phi1'][ptr[j]:ptr[j+1]]
            phi2 = data['phi2'][ptr[j]:ptr[j+1]]
            feat = np.stack([data[f][ptr[j]:ptr[j+1]] for f in features], axis=1)
            label = table[labels].iloc[j].values

            print("RAW phi1 size:", len(phi1)) #ADDED CHECK

            mask = (phi1_min <= phi1) & (phi1 < phi1_max)
            phi1 = phi1[mask]
            phi2 = phi2[mask]
            feat = feat[mask]

            print("AFTER mask:", np.sum(mask)) #ADDED CHECK


            for _ in range(num_subsamples):
                # Subsample particles if specified
                if num_per_subsample is not None:
                    phi1_ppr, phi2_ppr, feat_ppr = preprocess_utils.subsample_arrays(
                        [phi1, phi2, feat], num_per_subsample=num_per_subsample)

                elif (num_per_subsample_min is not None) and (num_per_subsample_max is not None):
                    N = np.random.randint(num_per_subsample_min, num_per_subsample_max)
                    phi1_ppr, phi2_ppr, feat_ppr = preprocess_utils.subsample_arrays(
                        [phi1, phi2, feat], num_per_subsample=N)

                # Add uncertainty if specified
                phi1_ppr, phi2_ppr, feat_ppr, _, feat_unc_ppr = preprocess_utils.add_uncertainty(
                    phi1_ppr, phi2_ppr, feat_ppr, features, uncertainty_model=uncertainty_model)

                # Create PyTorch Geometric Data object
                pos = np.stack([phi1_ppr, phi2_ppr], axis=1)
                if uncertainty_model is not None and include_uncertainty:
                    feat_ppr = np.concatenate([feat_ppr, feat_unc_ppr], axis=1)

                graph_data = Data(
                    x=torch.tensor(feat_ppr, dtype=torch.float32),
                    y=torch.tensor(label, dtype=torch.float32).unsqueeze(0),
                    pos=torch.tensor(pos, dtype=torch.float32),
                )
                graph_list.append(graph_data)

    print('Total number of graphs: {}'.format(len(graph_list)))

    return graph_list


def read_processed(
    data_dir: Union[str, Path],
    num_datasets: int = 1,
    start_dataset: int = 0,
):
    """
    Read preprocessed particle-level stream datasets from pickle files as PyTorch Geometric graphs.

    Parameters
    ----------
    data_dir : str or Path
        Path to the directory containing the processed data files.
    num_datasets : int, optional
        Number of datasets to read in. Default is 1.
    start_dataset : int, optional
        Index to start reading the dataset. Default is 0.

    Returns
    -------
    list of Data
        List of PyTorch Geometric Data objects loaded from pickle files.
    """
    graph_list = []

    for i in tqdm(range(start_dataset, start_dataset + num_datasets)):
        data_path = os.path.join(data_dir, f'data.{i}.pkl')
        if not os.path.exists(data_path):
            continue

        with open(data_path, "rb") as f:
            graphs = pickle.load(f)

        # If the pickle file contains a list of Data objects, extend graph_list
        # Otherwise, if it's a single Data object, append it
        if isinstance(graphs, list):
            graph_list.extend(graphs)
        else:
            graph_list.append(graphs)

    print('Total number of graphs loaded: {}'.format(len(graph_list)))

    return graph_list

#Added by Ella 8/15
# @lru_cache(maxsize=None)
# def _load_track_poly(path="/expanse/lustre/projects/upa160/lmarin/aau_sbi_project/run_sims/stream_track_phi2_poly.pkl"):
#     """
#     Loading in the spline fit to the unperturbed stream cut at phi1 = [-20, 16]
#     """
#     with open(path, "rb") as f:
#         info = pickle.load(f)
#     track = np.poly1d(info["coeffs"])
#     return track, info


# def subtract_track_from_data(data: List[Data], feature_names: List[str],
#                                track_feature="phi2", phi1_feature="phi1",
#                                track_path="/expanse/lustre/projects/upa160/lmarin/aau_sbi_project/run_sims/stream_track_phi2_poly.pkl"):
#     """
#     For every stream, replace each particle's phi2 value with phi2 - the spline fit at that particle's phi1 
#     location.
#     """
#     track, info = _load_track_poly(track_path)

#     phi1_idx = feature_names.index(phi1_feature)
#     feat_idx = feature_names.index(track_feature)

#     n_out_of_range = 0
#     for d in data:
#         phi1_vals = d.x[:, phi1_idx].numpy()

#         # flag (don't silently extrapolate) if particles fall outside the
#         # phi1 range the track was actually fit on
#         out_of_range = (phi1_vals < info["phi1_min"]) | (phi1_vals >= info["phi1_max"])
#         n_out_of_range += out_of_range.sum()

#         trend = track(phi1_vals)
#         d.x[:, feat_idx] = d.x[:, feat_idx] - torch.tensor(trend, dtype=d.x.dtype)

#     if n_out_of_range > 0:
#         print(f"Warning: {n_out_of_range} particles had phi1 outside "
#               f"[{info['phi1_min']}, {info['phi1_max']}) — track was extrapolated for these.")

#     return data

# def prepare_particle_dataloaders_detrended(
#     data: List[Data], 
#     feature_names: List[str], 
#     norm_dict: dict = None, 
#     train_frac: float = 0.8, 
#     train_batch_size: int = 32,
#     eval_batch_size: int = 32,
#     num_workers: int = 0,
#     seed: int = 42,
#     num_subsamples: int = 1,
#     track_feature="phi2",
#     track_path="/expanse/lustre/projects/upa160/lmarin/aau_sbi_project/run_sims/stream_track_phi2_poly.pkl",
# ):
#     """
#     Same as prepare_particle_dataloaders, but subtracts the fitted stream
#     track from `track_feature` before computing normalization stats.
#     """

#     rng = np.random.default_rng(seed)
#     num_total = len(data)

#     if num_subsamples > 1:
#         # Special case if subsampling is enabled
#         # This is required to prevent data leakage - keep subsamples from the same stream together
#         assert num_total % num_subsamples == 0, \
#             f"Data size {num_total} must be divisible by num_subsamples {num_subsamples}"

#         num_total_subsample = num_total // num_subsamples

#         # Reshape to group subsamples together
#         data_grouped = [data[i:i+num_subsamples] for i in range(0, num_total, num_subsamples)]

#         # Shuffle the groups
#         shuffle_indices = rng.permutation(num_total_subsample)
#         data_grouped = [data_grouped[i] for i in shuffle_indices]

#         # Split into train/val
#         num_train_groups = int(train_frac * num_total_subsample)
#         train_data_grouped = data_grouped[:num_train_groups]
#         val_data_grouped = data_grouped[num_train_groups:]

#         # Flatten back to lists
#         train_data = [item for group in train_data_grouped for item in group]
#         val_data = [item for group in val_data_grouped for item in group]
#     else:
#         # Standard shuffle and split
#         indices = rng.permutation(num_total)
#         num_train = int(train_frac * num_total)
#         train_indices = indices[:num_train]
#         val_indices = indices[num_train:]

#         train_data = [data[i] for i in train_indices]
#         val_data = [data[i] for i in val_indices]
    
    
#     #Subtract spline from both training and validation data, these are the only new lines
#     train_data = subtract_track_from_data(train_data, feature_names,
#                                             track_feature=track_feature, track_path=track_path)
#     val_data = subtract_track_from_data(val_data, feature_names,
#                                           track_feature=track_feature, track_path=track_path)

#     if norm_dict is None:
#         # Collect all node features and labels from training data
#         all_x = torch.cat([d.x for d in train_data], dim=0)
#         all_y = torch.stack([d.y for d in train_data], dim=0)

#         # Compute normalization for node features
#         x_loc = all_x.mean(dim=0)
#         x_scale = all_x.std(dim=0)

#         # Compute normalization for labels (min-max scaling to [-1, 1])
#         y_min = all_y.min(dim=0)[0]
#         y_max = all_y.max(dim=0)[0]
#         y_loc = (y_min + y_max) / 2
#         y_scale = (y_max - y_min) / 2

#         norm_dict = {
#             "x_loc": x_loc,
#             "x_scale": x_scale,
#             "y_loc": y_loc,
#             "y_scale": y_scale,
#         }
#     else:
#         x_loc = norm_dict["x_loc"]
#         x_scale = norm_dict["x_scale"]
#         y_loc = norm_dict["y_loc"]
#         y_scale = norm_dict["y_scale"]

#     # Normalize training data
#     for d in train_data:
#         d.x = (d.x - x_loc) / x_scale
#         d.y = (d.y - y_loc) / y_scale

#     # Normalize validation data
#     for d in val_data:
#         d.x = (d.x - x_loc) / x_scale
#         d.y = (d.y - y_loc) / y_scale

#     # Create PyTorch Geometric DataLoaders
#     train_loader = DataLoader(
#         train_data,
#         batch_size=train_batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=torch.cuda.is_available()
#     )

#     val_loader = DataLoader(
#         val_data,
#         batch_size=eval_batch_size,
#         shuffle=True,   # enable for callbacks, which take random subsets of val
#         num_workers=num_workers,
#         pin_memory=torch.cuda.is_available()
#     )

#     return train_loader, val_loader, norm_dict

def prepare_dataloaders(
    data: List[Data],
    norm_dict: dict = None,
    train_frac: float = 0.8,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    num_subsamples: int = 1,
):
    """
    Create PyTorch Geometric dataloaders for training and evaluation of particle-level stream datasets.

    Parameters
    ----------
    data : list of Data
        List of PyTorch Geometric Data objects.
    norm_dict : dict, optional
        Dictionary containing normalization parameters. If None, will be computed from training data.
        Expected keys: 'x_loc', 'x_scale', 'y_loc', 'y_scale'
    train_frac : float, optional
        Fraction of data to use for training. Default is 0.8.
    train_batch_size : int, optional
        Batch size for training. Default is 32.
    eval_batch_size : int, optional
        Batch size for evaluation. Default is 32.
    num_workers : int, optional
        Number of workers for data loading. Default is 0.
    seed : int, optional
        Random seed for shuffling. Default is 42.
    num_subsamples : int, optional
        Number of subsamples per stream. Default is 1.

    Returns
    -------
    tuple
        (train_loader, val_loader, norm_dict)
    """
    rng = np.random.default_rng(seed)
    num_total = len(data)

    # Shuffle and split data accounting for subsamples
    # TODO: Move this logic into a separate function
    if num_subsamples > 1:
        # Special case if subsampling is enabled
        # This is required to prevent data leakage - keep subsamples from the same stream together
        assert num_total % num_subsamples == 0, \
            f"Data size {num_total} must be divisible by num_subsamples {num_subsamples}"

        num_total_subsample = num_total // num_subsamples

        # Reshape to group subsamples together
        data_grouped = [data[i:i+num_subsamples] for i in range(0, num_total, num_subsamples)]

        # Shuffle the groups
        shuffle_indices = rng.permutation(num_total_subsample)
        data_grouped = [data_grouped[i] for i in shuffle_indices]

        # Split into train/val
        num_train_groups = int(train_frac * num_total_subsample)
        train_data_grouped = data_grouped[:num_train_groups]
        val_data_grouped = data_grouped[num_train_groups:]

        # Flatten back to lists
        train_data = [item for group in train_data_grouped for item in group]
        val_data = [item for group in val_data_grouped for item in group]
    else:
        # Standard shuffle and split
        indices = rng.permutation(num_total)
        num_train = int(train_frac * num_total)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]

        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]

    # Compute normalization statistics if not provided
    if norm_dict is None:
        # Collect all node features and labels from training data
        all_x = torch.cat([d.x for d in train_data], dim=0)
        all_y = torch.stack([d.y for d in train_data], dim=0)

        # Compute normalization for node features
        x_loc = all_x.mean(dim=0)
        x_scale = all_x.std(dim=0)

        # Compute normalization for labels (min-max scaling to [-1, 1])
        y_min = all_y.min(dim=0)[0]
        y_max = all_y.max(dim=0)[0]
        y_loc = (y_min + y_max) / 2
        y_scale = (y_max - y_min) / 2

        norm_dict = {
            "x_loc": x_loc,
            "x_scale": x_scale,
            "y_loc": y_loc,
            "y_scale": y_scale,
        }
    else:
        x_loc = norm_dict["x_loc"]
        x_scale = norm_dict["x_scale"]
        y_loc = norm_dict["y_loc"]
        y_scale = norm_dict["y_scale"]

    # Normalize training data
    for d in train_data:
        d.x = (d.x - x_loc) / x_scale
        d.y = (d.y - y_loc) / y_scale

    # Normalize validation data
    for d in val_data:
        d.x = (d.x - x_loc) / x_scale
        d.y = (d.y - y_loc) / y_scale

    # Create PyTorch Geometric DataLoaders
    train_loader = DataLoader(
        train_data,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_data,
        batch_size=eval_batch_size,
        shuffle=True,   # enable for callbacks, which take random subsets of val
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, norm_dict


def prepare_test_dataloader(
    data: List[Data],
    norm_dict: dict,
    test_batch_size: int = 32,
    num_workers: int = 0,
):
    """
    Create PyTorch Geometric dataloader for testing of particle-level stream datasets.

    Parameters
    ----------
    data : list of Data
        List of PyTorch Geometric Data objects.
    norm_dict : dict
        Dictionary containing normalization parameters.
        Expected keys: 'x_loc', 'x_scale', 'y_loc', 'y_scale'
    test_batch_size : int, optional
        Batch size for testing. Default is 32.
    num_workers : int, optional
        Number of workers for data loading. Default is 0.

    Returns
    -------
    DataLoader
        PyTorch Geometric DataLoader for the test dataset.
    """
    x_loc = norm_dict["x_loc"]
    x_scale = norm_dict["x_scale"]
    y_loc = norm_dict["y_loc"]
    y_scale = norm_dict["y_scale"]

    # Normalize test data
    for d in data:
        d.x = (d.x - x_loc) / x_scale
        d.y = (d.y - y_loc) / y_scale

    # Create PyTorch Geometric DataLoader
    test_loader = DataLoader(
        data,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return test_loader
