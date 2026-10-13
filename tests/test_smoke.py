#!/usr/bin/env python3
"""
Description: 
    Basic tests for ml_model_ai4pex.
    Using synthetic data, the CNN and UNet approaches are shown to work end-to-end.

Run using pytest:
    pytest tests/ -v
    pytest tests/ --log-cli-level=WARNING

"""

import sys
import logging
import argparse

import numpy as np
import xarray as xr
import tensorflow as tf


# ── helpers ─────────────────────────────────────────────────────────────────────

def _null_logger():
    """Logger that silently discards all output."""
    logger = logging.getLogger("smoke_test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def _unet_args():
    return argparse.Namespace(
        model="unet",
        features=["vor", "coarse_ke", "sa"],
        target=["fine_ke"],
        base_filters=4,
        depth=1,
        verbose=False,
        train=True,
        predict=False,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        train_stride=1,
        shuffle_seed=42,
        # CNN-specific (unused)
        filters=None, kernels=None, padding=None, dilation_rates=None,
    )


def _cnn_args():
    return argparse.Namespace(
        model="cnn",
        features=["vor", "coarse_ke", "sa"],
        target=["fine_ke"],
        filters=[8, 1],
        kernels=[(3, 3), (1, 1)],
        padding=[(1, 1), (0, 0)],
        dilation_rates=[1, 1],
        verbose=False,
        train=True,
        predict=False,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        train_stride=1,
        shuffle_seed=42,
        # UNet-specific (unused)
        # base_filters=None, depth=None,
    )


def _synthetic_dataset(n_r=2, n_t=20, n_y=16, n_x=16):
    """Build a minimal xarray Dataset with the same structure as the real data."""
    rng = np.random.default_rng(0)
    dims = ("r", "t", "y_c", "x_c")
    shape = (n_r, n_t, n_y, n_x)
    return xr.Dataset(
        {
            "vor":       (dims, rng.standard_normal(shape).astype(np.float32)),
            "coarse_ke": (dims, np.abs(rng.standard_normal(shape)).astype(np.float32)),
            "sa":        (dims, rng.standard_normal(shape).astype(np.float32)),
            "fine_ke":   (dims, np.abs(rng.standard_normal(shape)).astype(np.float32)),
        },
        coords={
            "r":   np.arange(n_r),
            "t":   np.arange(n_t),
            "y_c": np.arange(n_y),
            "x_c": np.arange(n_x),
        },
    )


# ── individual tests ─────────────────────────────────────────────────────────────

def test_import():
    """Package can be imported."""
    import ml_model_ai4pex 
    print("  [PASS] import ml_model_ai4pex")

def test_data_split():
    """get_data_split returns stacked train/val datasets with the expected sizes."""
    from ml_model_ai4pex.model_setup import get_data_split

    args = _unet_args()
    ds = _synthetic_dataset(n_r=2, n_t=20)
    ds_train, ds_val = get_data_split(ds, args, _null_logger())

    assert "sample" in ds_train.dims, "Expected 'sample' dim after stack"
    assert ds_train.sizes["sample"] > 0
    assert ds_val.sizes["sample"] > 0
    print(
        f"  [PASS] get_data_split  "
        f"train samples={ds_train.sizes['sample']}  "
        f"val samples={ds_val.sizes['sample']}"
    )

def test_data_shuffle():
    """get_data_shuffle reorders samples while preserving the total count."""
    from ml_model_ai4pex.model_setup import get_data_split, get_data_shuffle

    args = _unet_args()
    ds = _synthetic_dataset(n_r=2, n_t=20)
    ds_train, _ = get_data_split(ds, args, _null_logger())
    ds_shuffled = get_data_shuffle(ds_train, args, _null_logger())

    assert ds_shuffled.sizes["sample"] == ds_train.sizes["sample"]

    original_idx = ds_train.coords["sample"].values
    shuffled_idx = ds_shuffled.coords["sample"].values
    assert not np.array_equal(original_idx, shuffled_idx), \
        "Shuffle did not reorder samples (same order before and after)"

    print(
        f"  [PASS] get_data_shuffle  "
        f"samples={ds_shuffled.sizes['sample']} (order changed ✓)"
    )


def test_scenario_unet():
    """setup_scenario builds a valid Scenario for UNet."""
    from ml_model_ai4pex.model_setup import setup_scenario
    sc = setup_scenario(_unet_args(), _null_logger())
    assert sc.base_filters == 4
    assert sc.depth == 1
    assert sc.input_var == ["vor", "coarse_ke", "sa"]
    assert sc.target == ["fine_ke"]
    print("  [PASS] setup_scenario (unet)")


def test_scenario_cnn():
    """setup_scenario builds a valid Scenario for CNN."""
    from ml_model_ai4pex.model_setup import setup_scenario
    sc = setup_scenario(_cnn_args(), _null_logger())
    assert sc.filters == [8, 1]
    assert sc.kernels == [(3, 3), (1, 1)]
    print("  [PASS] setup_scenario (cnn)")


def test_unet_forward_pass():
    """UNet produces correctly shaped output for a random input batch."""
    from ml_model_ai4pex.model_setup import setup_scenario
    from ml_model_ai4pex.unet import UNet

    args = _unet_args()
    sc = setup_scenario(args, _null_logger())
    H, W, C = 16, 16, len(args.features)

    model = UNet(sc, input_shape=(H, W, C), dropout_rate=0.0)
    x = tf.zeros((2, H, W, C))
    y = model(x, training=False)

    expected = (2, H, W, len(args.target))
    assert tuple(y.shape) == expected, f"Got {tuple(y.shape)}, expected {expected}"
    print(f"  [PASS] UNet forward pass  input={tuple(x.shape)} → output={tuple(y.shape)}")


def test_cnn_forward_pass():
    """CNN produces correctly shaped output for a random input batch."""
    from ml_model_ai4pex.model_setup import setup_scenario
    from ml_model_ai4pex.cnn import CNN

    args = _cnn_args()
    sc = setup_scenario(args, _null_logger())
    H, W, C = 16, 16, len(args.features)

    model = CNN(sc, input_shape=(H, W, C), dropout_rate=0.0)
    x = tf.zeros((2, H, W, C))
    y = model(x, training=False)

    expected = (2, H, W, 1)
    assert tuple(y.shape) == expected, f"Got {tuple(y.shape)}, expected {expected}"
    print(f"  [PASS] CNN forward pass   input={tuple(x.shape)} → output={tuple(y.shape)}")





# ── runner ───────────────────────────────────────────────────────────────────────

TESTS = [
    test_import,
    test_scenario_unet,
    test_scenario_cnn,
    test_data_split,
    test_data_shuffle,
    test_unet_forward_pass,
    test_cnn_forward_pass,
]

if __name__ == "__main__":
    print("Running smoke tests for ml_model_ai4pex ...\n")
    failed = []
    for test_fn in TESTS:
        try:
            test_fn()
        except Exception as exc:
            print(f"  [FAIL] {test_fn.__name__}: {exc}")
            failed.append(test_fn.__name__)
        print()

    if failed:
        print(f"FAILED: {len(failed)}/{len(TESTS)} tests failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All tests passed!")
