#!/usr/bin/python3
"""Data preprocessing utilities for opening, slicing, and normalizing datasets."""

import numpy as np
import xarray as xr
import copy
import logging

# data_preparation is not handed the run's logger, so warnings go to this
# module's logger; run_model.py's logging.basicConfig sends them to the run's log file
logger = logging.getLogger(__name__)

#--------------------------------------------------------
# Set of functions

def add_dimension_and_slice(ds, x_slice, y_slice):
    '''
        Adds a dimension to allow all regional datasets to 
            be combined.
        Slice the data based on input x and y.
    '''
    ds = ds.expand_dims({"r": 1})
    for coord in ds.coords:
        if coord != "t": #* omit t dimension
            ds[coord] = ds[coord].expand_dims({"r": 1})

    ds_tmp = ds.isel(x_c=x_slice, y_c=y_slice)

    return ds_tmp


def slice_data(ds, x_slice, y_slice):
    '''
        Slice the data based on input x and y.
    '''

    ds_tmp = ds.isel(x_c=x_slice, y_c=y_slice)

    return ds_tmp


def open_and_process_data(scenario, 
                          directory, 
                          filenames, 
                          domain, 
                          norm_stats=None,
                          norm_time_indices=None):
    '''
        Opens and preprocesses the data by feeding in directory and filenames
            into the preprocessing step.
            n.b. this only provides local normalization related to each sub domain.
                use open_and_combine_data then feed in datasets for global 
                normalization.
    '''
    ds = {}

    sc = copy.deepcopy(scenario) # make a copy of the original (untransformed) scenario
    sc_transformed = None       # will hold the transformed scenario after processing
    print(sc)

    for region in domain:

        directory_region = directory.format(domain=region)
        fnames_region = [f.format(domain=region) for f in filenames]

        processor = data_preparation(scenario, 
                                    directory=directory_region, 
                                    filenames=fnames_region,
                                    parallel=False,
                                    norm_stats=norm_stats,
                                    norm_time_indices=norm_time_indices,
                                    )
        ds[region] = processor()
        sc_transformed = copy.deepcopy(processor.sc)  # capture transformed scenario (e.g. coarse_ke_log, fine_ke_log)

        scenario = copy.deepcopy(sc)  # reset for next region
        print(scenario)

    dom_slice_dict = {
        'dDP': (slice(2, 37), slice(15, 50)),
        'uDP': (slice(2, 37), slice(10, 45)),
        'SP': (slice(2, 37), slice(15, 50)),
        'IO': (slice(2, 37), slice(15, 50)),
        'SO_JET': (slice(3, 43), slice(7, 47)),
        'SO_EAST': (slice(10, 50), slice(10, 50)),
        'SO_GYRE': (slice(10, 50), slice(10, 50)),
        'SO_TROP': (slice(10, 50), slice(7, 47)),
        }
    
    for region in domain:
        ds_region = slice_data(ds[region], 
                                 dom_slice_dict[region][0],
                                 dom_slice_dict[region][1],
                                )

        ds[region] = ds_region

    dataset_list = [ds[region] for region in domain]

    ds_combined = xr.concat(dataset_list, dim='r')

    # Drop z_c coordinate that emerges from the mke .nc files
    if 'z_c' in ds_combined.dims:
        ds_combined = ds_combined.drop_dims('z_c')

    return ds_combined, sc_transformed


def open_and_combine_data(scenario, directory, filenames, mask_fn, domain):
    '''
        Opens and combines the data to input directly as an xarray dataset into the
            preprocessing step.
    '''
    ds = {}
    mask = {}

    # sc = copy.deepcopy(scenario) # make a copy
    # # print(sc)

    for region in domain: 

        directory_region = directory.format(domain=region)
        print(directory_region)
        fnames_region = [f.format(domain=region) for f in filenames]
        print(fnames_region)
        mask_fn_region = mask_fn.format(domain=region)
        print(mask_fn_region)

        ds_tmp = xr.Dataset()
        for filename in fnames_region:
            tmp = xr.open_dataset(directory_region + filename)
            ds_tmp = xr.merge([ds_tmp, tmp])
        ds[region] = ds_tmp
        mask[region] = xr.open_dataset(directory_region + mask_fn_region)

    dom_slice_dict = {
        'dDP': (slice(2, 37), slice(15, 50)),
        'uDP': (slice(2, 37), slice(10, 45)),
        'SP': (slice(2, 37), slice(15, 50)),
        'IO': (slice(2, 37), slice(15, 50)),
        'SO_JET': (slice(3, 43), slice(7, 47)),
        'SO_EAST': (slice(10, 50), slice(10, 50)),
        }
    
    for region in domain:
        ds_region = add_dimension_and_slice(ds[region],
                                    dom_slice_dict[region][0],
                                    dom_slice_dict[region][1],
                                  )

        mask_region = add_dimension_and_slice(mask[region],
                                    dom_slice_dict[region][0],
                                    dom_slice_dict[region][1],
                                  )

        ds[region] = ds_region
        mask[region] = mask_region

        # #! a hack to drop coords that prevent concat.
        # coords_to_drop = ['z_c', 'gdept_1d', 'gdept_0', 'nav_lev',]
        # for coords in coords_to_drop:
        #     if coords in ds[region].coords:
        #         ds[region] = ds[region].reset_coords(coords, drop=True)

    dataset_list = [ds[region] for region in domain]
    mask_list = [mask[region] for region in domain]
    
    ds_combined = xr.concat(dataset_list, dim='r')
    mask_combined = xr.concat(mask_list, dim='r')

    return ds_combined, mask_combined, scenario

#--------------------------------------------------------

class data_preparation:
    '''
    A class for the preparation of data:
        - merge datasets using variables from scenario
        - find mean and std
        - normalize data
    '''
    def __init__(self, sc, dataset=None, mask=None,
                 directory=None, filenames=None, 
                 chunks=None, parallel=False,
                 norm_stats=None,
                 norm_time_indices=None,):
        '''
            If parallel is True, then set the chunks.
        '''

        self.sc = sc
        self.chunks = chunks
        self.parallel = parallel
        self.norm_stats = norm_stats
        self.norm_time_indices = norm_time_indices

        # --- Handle input mode ---
        if dataset is not None:
            # Direct dataset input mode
            self.ds = dataset
            self.mask = mask

        elif directory is not None and filenames is not None:
            # Load from directory and filenames
            self.mask = xr.Dataset()
            self.ds = xr.Dataset()
            self.directory = directory
            self.filenames = filenames
            self.ds, self.mask = self._get_data()

        else:
            raise ValueError(
                "Must provide either (dataset, mask) or (directory, filenames)."
            )


        # if parallel:
        #     pass #TODO implement dask.distributed

    def _add_dimension(self, ds):
        '''
            Adds a dimension to the dataset.
        '''
        ds = ds.expand_dims({"r": 1})
        for coord in ds.coords:
            if coord != "t": #* omit t dimension
                ds[coord] = ds[coord].expand_dims({"r": 1})

        return ds

    def _get_data(self):
        for filename in self.filenames:
            if 'mask' in filename:
                if self.parallel:
                    self.mask = xr.open_dataset(self.directory + filename, 
                                                chunks=self.chunks)
                else:
                    self.mask = xr.open_dataset(self.directory + filename)

            else:
                if self.parallel:
                    tmp = ( 
                        xr.open_dataset(self.directory + filename, 
                                        chunks=self.chunks)
                    )
                else:
                    tmp = ( 
                        xr.open_dataset(self.directory + filename)
                    )

                self.ds = xr.merge([self.ds, tmp])

        self.mask = self._add_dimension(self.mask)
        self.ds = self._add_dimension(self.ds)

        return self.ds, self.mask

    def transform_input(self):
        for variable in self.sc.input_var:
            # if variable=="coarse_ke_f" or variable=="slope":
            if variable == 'coarse_ke' or variable == 'coarse_ke_shift' or variable == 'fine_ke_shift':
                values = self.ds[variable].compute()
                # zero or negative values have no log: treat them as missing, so
                # they become NaN (never -inf) and are filled with 0 below
                not_positive = values <= 0
                n_not_positive = int(not_positive.sum())
                if n_not_positive > 0:
                    logger.warning(
                        "%s: %d value(s) that are zero or negative are treated "
                        "as missing before the log is taken",
                        variable, n_not_positive,
                    )
                    values = values.where(~not_positive)
                self.ds[variable + "_log"] = xr.apply_ufunc(
                    np.log, values,
                    input_core_dims=[['t', 'y_c', 'x_c']],
                    output_core_dims=[['t', 'y_c', 'x_c']],
                )
                self.ds[variable + "_log"] = self.ds[variable + "_log"].fillna(0) 
                self.sc.input_var[self.sc.input_var.index(variable)] = variable + "_log"
            # apply a asinh transform to vorticity
            # if variable == 'vor':
            #     asinh_scale = 1e-4
            #     self.ds[variable + "_asinh"] = xr.apply_ufunc(
            #         np.arcsinh, (self.ds[variable] / asinh_scale).compute(),
            #         input_core_dims=[['t', 'y_c', 'x_c']],
            #         output_core_dims=[['t', 'y_c', 'x_c']],
            #     )
            #     self.ds[variable + "_asinh"] = self.ds[variable + "_asinh"].fillna(0) 
            #     self.sc.input_var[self.sc.input_var.index(variable)] = variable + "_asinh"
            #     self.ds[variable + "_asinh"] = self.ds[variable + "_asinh"].assign_attrs({"asinh_scale": asinh_scale})


    def normalize_data(self):
        if 't' in self.mask.tmask.dims:
            cell_area = self.mask.e1t.isel(t=0)*self.mask.e2t.isel(t=0)
            ff_t = self.mask.ff_t.isel(t=0)
        else:
            cell_area = self.mask.e1t*self.mask.e2t
            ff_t = self.mask.ff_t
        for variable in (self.sc.input_var): 
            # normalize the data
            if variable == 'vor':
                self.ds[variable] = (self.ds[variable] / ff_t).compute()
                self.ds[variable] = self.ds[variable].fillna(0)
            if self.norm_stats is not None:
                # dictionary format
                mean = self.norm_stats[variable]['mean']
                std = self.norm_stats[variable]['std']
            else:
                var_for_stats = self.ds[variable]
                if self.norm_time_indices is not None:
                    var_for_stats = var_for_stats.isel(t=self.norm_time_indices)

                # Denominator must match the subset used to compute mean/std.
                t_len = int(var_for_stats.sizes["t"])

                mean = (
                    (var_for_stats * cell_area)
                    .sum(("t", "y_c", "x_c"))
                    / ((cell_area).sum(("y_c", "x_c")) * t_len)
                ).mean(("r"))

                std = np.sqrt(
                    (cell_area * (var_for_stats - mean) ** 2)
                    .sum(("r", "t", "x_c", "y_c"))
                    / (cell_area.sum(("r", "y_c", "x_c")) * t_len)
                )

                # store mean and std for introspection/debugging
                self.ds[f"{variable}_norm_mean"] = mean
                self.ds[f"{variable}_norm_std"] = std

            self.ds[variable] = (( self.ds[variable] - mean ) / std).compute()
            self.ds[variable] = self.ds[variable].fillna(0)

    def transform_target(self):
        for variable in self.sc.target:
            # only log transform fine_ke
            if variable == 'fine_ke':
                self.ds[variable + "_log"] = xr.apply_ufunc(
                    np.log, self.ds[variable].compute(),
                    input_core_dims=[['t', 'y_c', 'x_c']],
                    output_core_dims=[['t', 'y_c', 'x_c']],
                )
                self.ds[variable + "_log"] = self.ds[variable + "_log"].fillna(0) 
                self.sc.target = [variable + "_log"]
            # apply a asinh transform to eke tendency
            if variable == 'eke_tendency':
                asinh_scale = 1e-5
                self.ds[variable + "_asinh"] = xr.apply_ufunc(
                    np.arcsinh, (self.ds[variable] / asinh_scale).compute(),
                    input_core_dims=[['t', 'y_c', 'x_c']],
                    output_core_dims=[['t', 'y_c', 'x_c']],
                )
                self.ds[variable + "_asinh"] = self.ds[variable + "_asinh"].fillna(0) 
                self.sc.target = [variable + "_asinh"]
                self.ds[variable + "_asinh"] = self.ds[variable + "_asinh"].assign_attrs({"asinh_scale": asinh_scale})

    def mask_data(self):
        for variable in self.sc.input_var + self.sc.target:
            if variable in self.ds:
                if 't' in self.mask.tmask.dims: #* ORCA36
                    self.ds[variable] = (self.ds[variable] * \
                        self.mask.tmask.isel(t=0, nav_lev=0)).compute()
                else: #* DINO mesh mask
                    self.ds[variable] = (self.ds[variable] * \
                        self.mask.tmask.isel(z_c=0)).compute()
            else:
                print(f"Variable {variable} not found in dataset.")
 
    def __call__(self):
        self.transform_input()
        self.normalize_data()
        self.transform_target()
        self.mask_data()
        return self.ds

    