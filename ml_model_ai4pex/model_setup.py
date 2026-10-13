#!/usr/bin/env python3
"""Setup functions for configuring model scenarios and loading preprocessed data."""

from ml_model_ai4pex.cnn import CNN, Scenario
from ml_model_ai4pex.unet import UNet
from ml_model_ai4pex import preprocess_data
import numpy as np
import xarray as xr


def _get_split_indices_from_nt(nt: int, args):
    """
    Compute train/val/test indices along the `t` dimension.

    Note: this mirrors the logic in `get_data_split`, including `train_stride`.
    """
    # Match existing default behavior in get_data_split
    n_train = args.train_ratio if args.train_ratio is not None else 0.7
    n_val = args.val_ratio if args.val_ratio is not None else 0.15
    n_test = args.test_ratio if args.test_ratio is not None else 0.15
    train_stride = args.train_stride if args.train_stride is not None else 1

    total_ratio = n_train + n_val + n_test
    if not abs(total_ratio - 1.0) < 1e-6:
        raise ValueError(
            f"Train, val, and test ratios must sum to 1.0, got {total_ratio}"
        )

    n_train_t = int(nt * n_train)
    n_val_t = int(nt * n_val)
    n_test_t = int(nt * n_test)

    test_idx = np.arange(nt - n_test_t, nt)
    val_idx = np.arange(nt - n_test_t - n_val_t, nt - n_test_t)
    train_idx_full = np.arange(0, nt - n_test_t - n_val_t)

    train_idx = train_idx_full[::train_stride]

    return train_idx, val_idx, test_idx


def _compute_norm_time_indices(args, logger=None):
    """
    Determine which `t` indices should be used to compute mean/std.

    For leakage prevention we compute normalization statistics on the same
    training time subset that will be used for model training.
    """
    if not args.domain:
        raise ValueError("args.domain must be provided as a list.")
    if not args.data_filenames:
        raise ValueError("args.data_filenames must be provided as a list.")

    region0 = args.domain[0]
    directory_region = args.data_dir.format(domain=region0)
    fn0 = args.data_filenames[0].format(domain=region0)
    sample_path = directory_region + fn0

    if logger and args.verbose:
        logger.info(f"Computing normalization indices from: {sample_path}")

    ds_meta = xr.open_dataset(sample_path, decode_times=True)
    try:
        if "t" not in ds_meta.sizes:
            raise KeyError("Expected dimension 't' in dataset.")
        nt = int(ds_meta.sizes["t"])
    finally:
        ds_meta.close()

    train_idx, _, _ = _get_split_indices_from_nt(nt, args)
    return train_idx

def setup_scenario(args, logger=None):

    if logger and args.verbose:
        logger.info(
            f"Setting up scenario with features: {args.features}, "
            f"target: {args.target}, model: {args.model}"
        )

    if args.model == "unet":
        scenario = Scenario(
            input_var=args.features,
            target=args.target,
            base_filters=args.base_filters,
            depth=args.depth,
        )
    else:
        scenario = Scenario(
            input_var=args.features, 
            target=args.target, 
            filters=args.filters, 
            kernels=args.kernels,
            padding=args.padding,
            dilation_rates=args.dilation_rates,
        )

    logger.info(f"Scenario setup complete: {scenario}")

    return scenario

def get_data(scenario, args, logger):

    if logger and args.verbose:
        logger.info(
            f"Getting data with features: {args.features}, \
                     target: {args.target}, data_dir: {args.data_dir}, \
                          data_filenames: {args.data_filenames}, \
                              domain: {args.domain}"
        )

    if args.local_norm:
        norm_time_indices = None
        # If we don't supply explicit norm_stats, compute mean/std using only
        # training time indices to avoid leakage from val/test periods.
        if args.norm_stats is None:
            norm_time_indices = _compute_norm_time_indices(args, logger=logger)

        ds, sc = preprocess_data.open_and_process_data(
            scenario,
            args.data_dir,
            args.data_filenames,
            args.domain,
            norm_stats=args.norm_stats,
            norm_time_indices=norm_time_indices,
        )
    else:
        raise ValueError("local_norm must be true: local normalisation is the only kind supported.")

    logger.info(
        f"Data loading and processing complete. \
                 Dataset: {ds}, Scenario: {sc}"
    )

    return ds, sc

def get_data_split(ds, args, logger): 

    # Set default data split if not provided in the config file
    n_train = args.train_ratio if args.train_ratio is not None else 0.7
    n_val = args.val_ratio if args.val_ratio is not None else 0.15
    n_test = args.test_ratio if args.test_ratio is not None else 0.15

    # set the stride for training samling
    train_stride = args.train_stride if args.train_stride is not None else 1

    if logger and args.verbose:
        logger.info(
            f"Splitting data with train_ratio: {n_train}, \
                     val_ratio: {n_val}, test_ratio: {n_test}, \
                          train_stride: {train_stride}"
        )

    # Validate that ratios sum to 1.0
    total_ratio = n_train + n_val + n_test
    if not abs(total_ratio - 1.0) < 1e-6:
        raise ValueError(f"Train, val, and test ratios must sum to 1.0, got {total_ratio}")

    nt = int(ds.sizes["t"])
    train_idx, val_idx, test_idx = _get_split_indices_from_nt(nt, args)

    # get the data splits and return
    if args.train:
        train_out = ds.isel(t=train_idx).stack(sample=("r", "t"))
        val_out   = ds.isel(t=val_idx).stack(sample=("r", "t"))
        logger.info(
            f"Datasets split complete for training. Train set: {train_out}, Val set: {val_out}"
        )
        return train_out, val_out
    else:
        ds_val   = ds.isel(t=val_idx)
        ds_test  = ds.isel(t=test_idx)
        logger.info(
            f"Datasets split complete. Val set: {ds_val}, Test set: {ds_test}"
        )
        return ds_val, ds_test

def get_data_shuffle(ds, args, logger):

    if logger and args.verbose:
        if args.shuffle_seed is not None:
            logger.info(
                f"Shuffling data with seed: {args.shuffle_seed}.\
                      Fixed shuffling."
            )
        else:
            logger.info(
                f"Shuffling data with no seed. Random shuffling."
            )

    # get the total number of samples in the dataset
    n_samples = ds.sample.size
    idx = np.arange(n_samples)

    # create a random number generator with the specified seed
    rng = np.random.default_rng(args.shuffle_seed)

    # generate a random permutation of the sample indices
    rng.shuffle(idx)

    # shuffle the dataset across samples using the shuffled indices
    ds_shuffled = ds.isel(sample=idx)

    logger.info(
        f"Data shuffling complete. Shuffled dataset: {ds_shuffled}"
    )

    return ds_shuffled
