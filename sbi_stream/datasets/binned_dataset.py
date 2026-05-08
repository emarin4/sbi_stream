
import os
import pickle
from pathlib import Path
from typing import List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import torch
from absl import logging
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from . import io_utils, preprocess_utils


def read_and_process_raw(
    data_dir: Union[str, Path],
    features: List[str],
    labels: List[str],
    binning_fn: str = 'bin_stream',
    binning_args: dict = None,
    num_datasets: int = 1,
    start_dataset: int = 0,
    num_subsamples: int = 1,
    num_per_subsample: int = None,
    phi1_min: Optional[float] = None,
    phi1_max: Optional[float] = None,
    bounds: dict = None,
    use_density: bool = True,
    use_width: bool = True,
    uncertainty: Optional[str] = None,
):
    """
    Read and process raw binned stream datasets from HDF5 files.

    Parameters
    ----------
    data_dir : str or Path
        Path to the directory containing the stream data.
    features : list of str, optional
        List of feature names to extract.
    labels : list of str, optional
        List of labels to use for the regression.
    binning_fn : str, optional
        The binning function to use ('bin_stream', 'bin_stream_spline').
        Default is 'bin_stream'.
    binning_args : dict, optional
        Arguments for the binning function.
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
    use_density : bool, optional
        If True, include the fraction of stars in each bin. Default is True.
    use_width : bool, optional
        If True, include the width of the stream in each bin. Default is True.
    uncertainty : str, optional
        If not None, include uncertainty. Either "present" or "future".

    Returns
    -------
    tuple
        (x, y, t, padding_mask) where:
        - x: features array (num_samples, max_len, num_features)
        - y: labels array (num_samples, num_labels)
        - t: time/bin_centers array (num_samples, max_len, 1)
        - padding_mask: mask for padded entries (num_samples, max_len)
    """
    # default args
    binning_args = binning_args or {}
    phi1_min = phi1_min or -np.inf
    phi1_max = phi1_max or np.inf

    if binning_fn not in ['bin_stream', 'bin_stream_spline']:
        raise ValueError(
            f"Invalid binning_fn: {binning_fn}. Must be 'bin_stream' or 'bin_stream_spline'")

    x, y, t = [], [], []

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
        data, ptr = io_utils.read_dataset(data_fn, unpack=True)

        # compute some derived labels
        table = preprocess_utils.calculate_derived_properties(table)

        loop = tqdm(range(len(table)), desc='Processing streams')

        for pid in loop:
            phi1 = data['phi1'][pid]
            phi2 = data['phi2'][pid]
            feat = np.stack([data[f][pid] for f in features], axis=1)
            label = table[labels].iloc[pid].values

            mask = (phi1_min <= phi1) & (phi1 < phi1_max)
            phi1 = phi1[mask]
            phi2 = phi2[mask]
            feat = feat[mask]

            for _ in range(num_subsamples):
                # Subsample particles if specified
                if num_per_subsample is not None:
                    phi1, phi2, feat = preprocess_utils.subsample_arrays(
                        [phi1, phi2, feat], num_per_subsample=num_per_subsample)

                # Add uncertainty if specified
                phi1, phi2, feat, feat_err = preprocess_utils.add_uncertainty(
                    phi1, phi2, feat, features, uncertainty_model=uncertainty)


                # bin the stream
                if binning_fn == 'bin_stream':
                    bin_centers, feat_mean, feat_stdv, feat_count = preprocess_utils.bin_stream(
                        phi1, feat, **binning_args)
                elif binning_fn == 'bin_stream_spline':
                    bin_centers, feat_mean, feat_stdv, feat_count = preprocess_utils.bin_stream_spline(
                        phi1, phi2, feat, **binning_args)

                if len(bin_centers) == 0:
                    continue

                all_feats = []
                all_feats.append(feat_mean)
                if use_width:
                    all_feats.append(feat_stdv)
                if use_density:
                    all_feats.append(feat_count / np.sum(feat_count))
                all_feats = np.concatenate(all_feats, axis=1)
                x.append(all_feats)
                y.append(label)
                t.append(bin_centers.reshape(-1, 1))

    logging.info('Total number of samples: {}'.format(len(x)))

    x, padding_mask = preprocess_utils.pad_and_create_mask(x)
    t, _ = preprocess_utils.pad_and_create_mask(t)
    y = np.stack(y, axis=0)

    return x, y, t, padding_mask


def read_processed(
    data_dir: Union[str, Path],
    num_datasets: int = 1,
    start_dataset: int = 0,
):
    """
    Read preprocessed binned stream datasets from pickle files.

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
    tuple
        (x, y, t, padding_mask) where:
        - x: features array (num_samples, max_len, num_features)
        - y: labels array (num_samples, num_labels)
        - t: time/bin_centers array (num_samples, max_len, 1)
        - padding_mask: mask for padded entries (num_samples, max_len)
    """
    x, y, t, padding_mask = [], [], [], []
    xlen = []
    for i in tqdm(range(start_dataset, start_dataset + num_datasets)):
        data_path = os.path.join(data_dir, f'data.{i}.pkl')
        if not os.path.exists(data_path):
            continue
        with open(data_path, "rb") as f:
            d = pickle.load(f)
        x.append(d[0])
        y.append(d[1])
        t.append(d[2])
        padding_mask.append(d[3])
        xlen.append(d[0].shape[1])

    # Pad sequences to the maximum length
    # check if the lengths of sequences are the same
    if len(set(xlen)) != 1:
        max_xlen = max(xlen)
        for i in range(len(x)):
            pad_len = max_xlen - xlen[i]
            x[i] = np.pad(x[i], ((0, 0), (0, pad_len), (0, 0)), mode='constant', constant_values=0)
            t[i] = np.pad(t[i], ((0, 0), (0, pad_len), (0, 0)), mode='constant', constant_values=0)
            padding_mask[i] = np.pad(padding_mask[i], ((0, 0), (0, pad_len)), mode='constant', constant_values=True)

    x = np.concatenate(x)
    y = np.concatenate(y)
    t = np.concatenate(t)
    padding_mask = np.concatenate(padding_mask)

    return x, y, t, padding_mask


def prepare_dataloaders(
    data: Tuple,
    norm_dict: dict = None,
    train_frac: float = 0.8,
    train_batch_size: int = 128,
    eval_batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
    num_subsamples: int = 1,
    subsample_shuffle: bool = True,
):
        """
        Create PyTorch dataloaders for training and evaluation of binned stream datasets.

        Parameters
        ----------
        data : tuple
            Tuple containing (x, y, t, padding_mask).
        norm_dict : dict, optional
            Dictionary containing normalization parameters. If None, will be computed from training data.
        train_frac : float, optional
            Fraction of data to use for training. Default is 0.8.
        train_batch_size : int, optional
            Batch size for training. Default is 128.
        eval_batch_size : int, optional
            Batch size for evaluation. Default is 128.
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

        # unpack the data
        x, y, t, padding_mask = data
        num_total = len(x)

        if num_subsamples > 1:
            # special case if subsampling is enabled
            # this is required to prevent data leakage
            assert num_total % num_subsamples == 0, f"Data size {len(x)} must be divisible by num_subsamples {num_subsamples}"
            num_total_subsample = num_total // num_subsamples

            x = x.reshape(num_total_subsample, num_subsamples, x.shape[1], x.shape[2])
            y = y.reshape(num_total_subsample, num_subsamples, y.shape[1])
            t = t.reshape(num_total_subsample, num_subsamples, t.shape[1], t.shape[2])
            padding_mask = padding_mask.reshape(num_total_subsample, num_subsamples, padding_mask.shape[1])

            shuffle = rng.permutation(num_total_subsample)
            x = x[shuffle]
            y = y[shuffle]
            t = t[shuffle]
            padding_mask = padding_mask[shuffle]

            # split the data into training and validation sets
            num_train = int(train_frac * num_total_subsample)
            x_train, x_val = x[:num_train], x[num_train:]
            y_train, y_val = y[:num_train], y[num_train:]
            t_train, t_val = t[:num_train], t[num_train:]
            padding_mask_train, padding_mask_val = padding_mask[:num_train], padding_mask[num_train:]

            # flatten the data back to original shape
            x_train = x_train.reshape(-1, x_train.shape[2], x_train.shape[3])
            y_train = y_train.reshape(-1, y_train.shape[2])
            t_train = t_train.reshape(-1, t_train.shape[2], t_train.shape[3])
            padding_mask_train = padding_mask_train.reshape(-1, padding_mask_train.shape[2])
            x_val = x_val.reshape(-1, x_val.shape[2], x_val.shape[3])
            y_val = y_val.reshape(-1, y_val.shape[2])
            t_val = t_val.reshape(-1, t_val.shape[2], t_val.shape[3])
            padding_mask_val = padding_mask_val.reshape(-1, padding_mask_val.shape[2])
        else:
            shuffle = rng.permutation(num_total)
            x = x[shuffle]
            y = y[shuffle]
            t = t[shuffle]
            padding_mask = padding_mask[shuffle]

            num_train = int(train_frac * len(x))
            x_train, x_val = x[:num_train], x[num_train:]
            y_train, y_val = y[:num_train], y[num_train:]
            t_train, t_val = t[:num_train], t[num_train:]
            padding_mask_train, padding_mask_val = padding_mask[:num_train], padding_mask[num_train:]

        # normalize the data
        if norm_dict is None:
            # norm mask for x
            mask = np.repeat(~padding_mask_train[..., None], x_train.shape[-1], axis=-1)
            x_loc = x_train.mean(axis=(0, 1), where=mask)
            x_scale = x_train.std(axis=(0, 1), where=mask)

            # normalize y such that it is in range [-1, 1]
            # this is required for NSF
            y_min = np.min(y_train, axis=0)
            y_max = np.max(y_train, axis=0)
            y_loc = (y_min + y_max) / 2
            y_scale = (y_max - y_min) / 2

            # normalize time by min-max scaling
            t_loc = t_train.min()
            t_scale = t_train.max() - t_loc
            norm_dict = {
                "x_loc": x_loc, "x_scale": x_scale,
                "y_loc": y_loc, "y_scale": y_scale,
                "t_loc": t_loc, "t_scale": t_scale
            }
        else:
            x_loc, x_scale = norm_dict["x_loc"], norm_dict["x_scale"]
            y_loc, y_scale = norm_dict["y_loc"], norm_dict["y_scale"]
            t_loc, t_scale = norm_dict["t_loc"], norm_dict["t_scale"]

        # normalize the data
        x_train = (x_train - x_loc) / x_scale
        y_train = (y_train - y_loc) / y_scale
        t_train = (t_train - t_loc) / t_scale
        x_val = (x_val - x_loc) / x_scale
        y_val = (y_val - y_loc) / y_scale
        t_val = (t_val - t_loc) / t_scale

        # convert to tensor
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        t_train = torch.tensor(t_train, dtype=torch.float32)
        padding_mask_train = torch.tensor(padding_mask_train, dtype=torch.bool)
        x_val = torch.tensor(x_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        t_val = torch.tensor(t_val, dtype=torch.float32)
        padding_mask_val = torch.tensor(padding_mask_val, dtype=torch.bool)

        # create data loader
        train_loader = DataLoader(
            TensorDataset(x_train, y_train, t_train, padding_mask_train),
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            TensorDataset(x_val, y_val, t_val, padding_mask_val),
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        return train_loader, val_loader, norm_dict


def prepare_test_dataloader(data, norm_dict, **kwargs):
    """Create a test dataloader.

    Args:
        data: Test dataset
        norm_dict: Normalization dictionary
        **kwargs: Additional arguments for DataLoader

    Returns:
        DataLoader for the test dataset
    """
    x_test, y_test, t_test, padding_mask_test = data

    # normalize the data
    x_test = (x_test - norm_dict["x_loc"]) / norm_dict["x_scale"]
    y_test = (y_test - norm_dict["y_loc"]) / norm_dict["y_scale"]
    t_test = (t_test - norm_dict["t_loc"]) / norm_dict["t_scale"]

    # convert to tensor
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    t_test = torch.tensor(t_test, dtype=torch.float32)
    padding_mask_test = torch.tensor(padding_mask_test, dtype=torch.bool)

    # create data loader
    test_loader = DataLoader(
        TensorDataset(x_test, y_test, t_test, padding_mask_test),
        batch_size=kwargs.get("batch_size", 32),
        shuffle=False,
        num_workers=kwargs.get("num_workers", 0),
        pin_memory=torch.cuda.is_available()
    )

    return test_loader