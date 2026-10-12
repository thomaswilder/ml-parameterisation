#!/usr/bin/env python3
"""
Build the provenance record stored inside saved .keras models
(see model_components.ProvenanceMixin).

Record layout:
    run_id            timestamp of the run (used in the default model filename)
    run_settings      resolved args after CLI overrides (data paths left out)
    data_split        split ratios, shuffle seed, stride, omitted months,
                      sample counts
    normalisation     mean/std applied to each input variable
    code_version      git commit hash
    environment       package versions, host, SLURM job, GPUs
    training_history  per-epoch metrics and which epoch's weights were saved

Epoch numbers are 1-based, matching Keras's printed output.
"""

import os
import platform
import socket
import subprocess

import numpy as np
import xarray as xr
import keras
import tensorflow as tf

from ml_model_ai4pex.model_setup import _get_omit_months_from_args

# args pointing at the input data; left out of the record
_EXCLUDED_ARGS = ("data_dir", "data_filenames")


def build_training_history(history, early_stopping, start_time, end_time,
                           steps_per_epoch, validation_steps):
    """Summarise a keras History object (and EarlyStopping callback, if used)."""

    epochs_completed = len(history.epoch)

    # EarlyStopping restores the best weights at the end of training
    # whenever it saw an epoch past start_from_epoch, even if it never stopped
    restored_best_weights = (
        early_stopping is not None
        and early_stopping.restore_best_weights
        and early_stopping.best_weights is not None
    )
    stopped_early = (
        early_stopping is not None and early_stopping.stopped_epoch > 0
    )

    return {
        "start_time": start_time.isoformat(timespec="seconds"),
        "end_time": end_time.isoformat(timespec="seconds"),
        "epochs_completed": epochs_completed,
        "steps_per_epoch": steps_per_epoch,
        "validation_steps": validation_steps,
        "early_stopped_at_epoch": (
            early_stopping.stopped_epoch + 1 if stopped_early else None
        ),
        "restored_best_weights": restored_best_weights,
        "weights_from_epoch": (
            early_stopping.best_epoch + 1 if restored_best_weights
            else epochs_completed
        ),
        "metrics": {
            name: [float(v) for v in values]
            for name, values in history.history.items()
        },
    }


def _get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _get_environment():
    gpus = [
        tf.config.experimental.get_device_details(gpu).get("device_name", gpu.name)
        for gpu in tf.config.list_physical_devices("GPU")
    ]
    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
        "xarray": xr.__version__,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpus": gpus,
    }


def _get_normalisation(args, ds, sc):
    """
    Stats come from the config if given, otherwise they were computed per
    region by preprocess_data and stored in ds as {var}_norm_mean/_norm_std.
    """
    if args.norm_stats is not None:
        return {
            "local_norm": args.local_norm,
            "source": "config",
            "stats": args.norm_stats,
        }

    stats = {}
    for i, region in enumerate(args.domain):
        stats[region] = {}
        for var in sc.input_var:
            if f"{var}_norm_mean" not in ds:
                continue
            mean = ds[f"{var}_norm_mean"]
            std = ds[f"{var}_norm_std"]
            if "r" in mean.dims:
                mean, std = mean.isel(r=i), std.isel(r=i)
            stats[region][var] = {"mean": float(mean), "std": float(std)}

    return {
        "local_norm": args.local_norm,
        "source": "computed from training split",
        "stats_per_region": stats,
    }


def build_provenance(args, ds, sc, run_id, n_train_samples, n_val_samples,
                     training_history):
    """Assemble the full record; assign it to model.provenance before saving."""

    return {
        "run_id": run_id,
        "run_settings": {
            k: v for k, v in vars(args).items() if k not in _EXCLUDED_ARGS
        },
        "data_split": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "train_stride": args.train_stride,
            "shuffle_seed": args.shuffle_seed,
            "train_omit_months": _get_omit_months_from_args(args),
            "n_train_samples": n_train_samples,
            "n_val_samples": n_val_samples,
        },
        "normalisation": _get_normalisation(args, ds, sc),
        "code_version": {"git_commit": _get_git_commit()},
        "environment": _get_environment(),
        "training_history": training_history,
    }
