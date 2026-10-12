#!/usr/bin/env python3

"""Prediction functions for loading trained models and generating predictions on test data."""

import numpy as np
import xarray as xr
import keras
from pathlib import Path


def predict_model(scenario, ds_test, args, logger=None):
    """
    Load a trained model, run predictions on test data, build an xarray
    Dataset of predicted vs true values, and optionally save to file.

    Parameters
    ----------
    scenario : cnn.Scenario
        The scenario object with input_var and target attributes.
    ds_test : xr.Dataset
        Test dataset (already split, with dimensions including r, t, y_c, x_c).
    args : argparse.Namespace
        Parsed arguments including model_dir, model_filename,
        predict_save_filename, domain, etc.
    logger : logging.Logger, optional

    Returns
    -------
    pred_ds : xr.Dataset
        Dataset containing predicted and true target values.
    """

    if logger and args.verbose:
        logger.info(
            f"Starting prediction with scenario: {scenario}"
        )

    model_path = args.model_dir + args.model_filename
    if logger and args.verbose:
        logger.info(f"Loading model from {model_path}")

    model = keras.saving.load_model(model_path, compile=True)

    if logger and args.verbose:
        import io
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        logger.info('Model summary is:\n%s', stream.getvalue())

    # iterate over each region (r dimension) and predict
    n_regions = ds_test.sizes.get("r", 1)
    pred_list = []

    n_lags = getattr(args, 'n_lags', 0) or 0

    for r_idx in range(n_regions):
        ds_region = ds_test.isel(r=r_idx)

        # build input / target arrays
        if n_lags == 0:
            batch_input = xr.merge(
                [ds_region[x] for x in scenario.input_var]
            ).to_array('var').transpose('t', 'y_c', 'x_c','var')

            batch_target = xr.merge(
                [ds_region[x] for x in scenario.target]
            ).to_array('var').transpose('t', 'y_c', 'x_c','var')

            time_values = ds_region.t.values
        else:
            # Lagged path: drop first n_lags timesteps (no lag history).
            n_t = ds_region.sizes['t']
            t_anchor = np.arange(n_lags, n_t)
            t_cur = xr.DataArray(t_anchor, dims='sample')
            t_lag = xr.DataArray(t_anchor - n_lags, dims='sample')

            cur_slice = ds_region.isel(t=t_cur)
            lag_slice = ds_region.isel(t=t_lag)

            # Group-by-variable channels: [var0_t, var0_{t-n_lags}, var1_t, ...]
            input_arrays = []
            for var in scenario.input_var:
                input_arrays.append(cur_slice[var])
                input_arrays.append(lag_slice[var])
            batch_input = xr.concat(input_arrays, dim='var') \
                            .transpose('sample', 'y_c', 'x_c', 'var')

            target_arrays = [cur_slice[v] for v in scenario.target]
            batch_target = xr.concat(target_arrays, dim='var') \
                              .transpose('sample', 'y_c', 'x_c', 'var')

            time_values = ds_region.t.values[n_lags:]

        prediction = model.predict(batch_input.to_numpy())

        # hacking this to transpose time index to be first
        # prediction = np.transpose(prediction, (2, 0, 1, 3))

        if logger and args.verbose:
            logger.info(
                f"Prediction complete for region index {r_idx}. "
                f"Output shape: {prediction.shape}"
            )

        # build per-region dataset
        target_name = scenario.target[0]
        region_ds = xr.Dataset(
            data_vars={
                "fine_ke_pred": (
                    ['time_counter', 'y', 'x', 'var'],
                    np.exp(prediction),
                ),
                "fine_ke_true": (
                    ['time_counter', 'y', 'x', 'var'],
                    np.exp(batch_target.to_numpy()),
                ),
            },
            coords={
                "time_counter": (
                    ["time_counter"],
                    time_values,
                    ds_region.t.attrs if hasattr(ds_region.t, 'attrs') else {},
                ),
                "gphit": (
                    ["y", "x"],
                    ds_region.gphit.values,
                    {"standard_name": "Latitude", "units": "degrees_north"},
                ),
                "glamt": (
                    ["y", "x"],
                    ds_region.glamt.values,
                    {"standard_name": "Longitude", "units": "degrees_east"},
                ),
                "var": scenario.target,
            },
            attrs={
                "Title": f"Fine kinetic energy - predicted and truth",
                "Description": (
                    f"Predicted fine kinetic energy from coarse-grained data using CNN"
                ),
                "Units": ["m$^2$/s$^2$"], 
                "Source": model_path,
            },
        )
        pred_list.append(region_ds)

    # concatenate regions if more than one
    if len(pred_list) > 1:
        pred_ds = xr.concat(pred_list, dim='r')
    else:
        pred_ds = pred_list[0]

    # save to file if requested
    if args.predict_save_filename:
        if args.predict_save_filename.endswith('.nc'):
            save_path = args.predict_save_dir + args.predict_save_filename
        else:
            stem = Path(args.model_filename).stem
            datetime_str = stem.split('_')[1]

            save_path = args.predict_save_dir \
                + args.predict_save_filename +\
                f"_{datetime_str}.nc"
            
        pred_ds.to_netcdf(save_path)
        if logger:
            logger.info(f"Predictions saved to {save_path}")

    return pred_ds
