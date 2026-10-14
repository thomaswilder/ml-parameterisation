# A Neural Network for the prediction of Eddy Kinetic Energy in GEOMETRIC

CNN and U-Net models for a Machine Learning eddy parameterisation for [NEMO](https://www.nemo-ocean.eu/) developed as part of the [AI4PEX](https://ai4pex.eu/) project. The models learn to predict eddy kinetic energy from eddy-permitting resolution variables. The NN will form an ML-GEOMETRIC to replace the eddy energy budget in [Mak et al., 2023](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS003886).

## Installation

Clone the repository and install in editable mode into your Python environment:

```bash
git clone https://github.com/thomaswilder/ml_model_ai4pex.git
cd ml_model_ai4pex
pip install -e .
```

This installs the `ml_model_ai4pex` package and all required dependencies. To also install what the tests need (`pytest`), ask for the `test` extra:

```bash
pip install -e ".[test]"
```

`pip install -e .` installs the plain `tensorflow` package, without the NVIDIA CUDA libraries. `pip install "tensorflow[and-cuda]"` adds them; they are also listed in `requirements-lock.txt` (see below).

## Dependencies

- Python >= 3.9
- TensorFlow / Keras
- Tensorboard
- xarray[io]
- xbatcher
- numpy
- PyYAML
- pytest (for the tests only, through the `test` extra)

`pyproject.toml` does not pin versions. The released models were trained with Python 3.9, TensorFlow 2.20 and Keras 3.10; the exact version of every package in that environment is recorded in `requirements-lock.txt`. To recreate that environment, with Python 3.9:

```bash
python -m venv venv
venv/bin/python -m pip install -r requirements-lock.txt
venv/bin/python -m pip install --no-deps -e .
```

## Usage

Copy and edit the example configuration file, then run the model script:

```bash
cp run/config_model.yml my_config.yml
# Edit my_config.yml to point to your data files and set hyperparameters
python run/run_model.py --config my_config.yml
```

You can also override options at runtime
```bash
python run/run_model.py --config my_config.yml --epochs=200
```

An option given on the command line takes precedence over the same key in the YAML file, and an option that is not in the YAML file can be given on the command line alone.

### Training

With `mode: train`, the script reads the data of every region in `domain`, trains the model and saves it in the Keras format to `model_dir`, as `model_<date>-<time>.keras` unless `model_save_filename` names the file. `model_dir` must exist before the run starts: the model is saved after training, and the save fails if the directory is missing. The run's messages go to `verbose_output_filename` (`./verbose_output.log` if it is not set) and the TensorBoard logs to `tensboard_log_dir`.

The time axis is split in order: the training period first, then the validation period, then the test period. Training uses every `train_stride`-th time of the training period, shuffled with `shuffle_seed`.

### Prediction

With `mode: predict`, the script loads the model `model_filename` from `model_dir` and predicts the test period of every region in `domain`. `model_filename` is required in this mode. The data options (`features`, `target`, the split and the normalisation) should be those the model was trained with. Without `predict_save_filename` no prediction file is written.

```bash
# my_config.yml has `mode: predict`
python run/run_model.py --config my_config.yml \
    --model_filename model_20260928-111329.keras \
    --predict_save_dir ./predictions/ \
    --predict_save_filename predictions_SO_JET
```

The predictions are written as a NetCDF file in `predict_save_dir` (the directory must exist). If `predict_save_filename` ends in `.nc` it is used as it is; otherwise the date and time in the model file name are appended (the text between the first underscore and the next underscore or the extension), so the command above writes `./predictions/predictions_SO_JET_20260928-111329.nc`. The file holds `fine_ke_pred` and `fine_ke_true`, both transformed back from the logarithm the model works with.

### Configuration

`run/config_model.yml` controls all aspects of the run. The section names in the file (`data`, `training`, ...) are only for reading: the keys inside them are what counts.

| Key | Description |
|-----|-------------|
| `mode` | `train` or `predict` (required) |
| `domain` | One or more region names, separated by spaces. Each must be one of the regions listed in `dom_slice_dict` in `ml_model_ai4pex/preprocess_data.py`, which gives the window that is cut from each region's files |
| `data_dir` | Directory with the input files; `{domain}` in it is replaced by the region name |
| `data_filenames` | NetCDF files with the inputs, the target and the mesh mask, separated by spaces; `{domain}` is replaced by the region name. The file with `mask` in its name is read as the mesh mask |
| `model_dir` | Directory for saving and loading the model |
| `model_save_filename` | File name for the trained model; it must end in `.keras` (optional, `train`) |
| `model_filename` | Model file to load (`predict`) |
| `predict_save_dir`, `predict_save_filename` | Where to write the predictions (`predict`) |
| `features` | Input variables, separated by spaces, in the order of the input channels. `coarse_ke` enters as its natural logarithm and is called `coarse_ke_log` from then on (also in `norm_stats`); `vor` is divided by the Coriolis parameter `ff_t` of the mesh mask |
| `target` | Target variable. `fine_ke` is predicted as its natural logarithm |
| `train_ratio`, `val_ratio`, `test_ratio` | Fractions of the time axis for training, validation and test; they must sum to 1 |
| `train_stride` | Use every n-th time of the training period |
| `shuffle_seed` | Seed for shuffling the training samples |
| `model` | Architecture to use: `unet` or `cnn` |
| `base_filters`, `depth` | U-Net: filters of the first level, and number of levels |
| `filters`, `kernels`, `padding`, `dilation_rates` | CNN: one entry per layer (see the commented example in the file) |
| `epochs` | Number of training epochs |
| `batch_size` | Batch size |
| `dropout_rate` | Rate of the spatial dropout layers |
| `learning_rate` | Initial learning rate |
| `use_learning_rate_scheduler`, `learning_rate_decay_steps_multiplier`, `learning_rate_decay_rate` | Exponential decay of the learning rate |
| `use_early_stopping`, `early_stopping_monitor`, `early_stopping_patience`, `early_stopping_min_delta`, `early_stopping_start_from_epoch` | Early stopping; the weights of the best epoch are restored |
| `use_ema`, `ema_momentum` | Exponential moving average of the weights. The average has no effect on the saved model: with early stopping on, the weights of the best epoch are restored after it is applied, so the saved weights are those of that epoch, not the average |
| `local_norm` | Must be `true`: each input is normalised (mean removed, divided by the standard deviation) |
| `norm_stats` | Mean and standard deviation of each input. If it is left out, they are computed for each region from the times used for training. It must stay inside one of the YAML sections (`data`, `training`, ...), like every other key |
| `verbose`, `verbose_output_filename`, `tensboard_log_dir` | Logging |

Options that are true or false (`use_ema`, `local_norm`, `use_early_stopping`, ...) should be set in the YAML file. A value given on the command line arrives as text, and any text, including `false`, counts as true, so the command line can switch such an option on but never off.

Directories are joined to file names as plain text, so `data_dir`, `model_dir` and `predict_save_dir` must end with `/`.

A key in the YAML file that is not an option stops the run with an error. This includes options that earlier versions of this code had and that have been removed: `n_lags`, `train_omit_seasons`, `train_omit_months`, `season_hemisphere`, `use_attention`, `k_folds`, `global_norm` and `pickup`.

Do `python run/run_model.py -h` for the full list of options.

### Loading a trained model

A script that loads a model itself must import the model classes first, so that Keras knows the names stored in the file. Importing `ml_model_ai4pex.model_setup` does this (importing `ml_model_ai4pex.unet` for a U-Net, or `ml_model_ai4pex.cnn` for a CNN, is enough too):

```python
import keras
import ml_model_ai4pex.model_setup  # registers UNet, CNN and Scenario with Keras

model = keras.saving.load_model("examples/sample_model/sample_model.keras")
```

`run/run_model.py` and the example notebook already do so.

## Project Structure

```
ml_model_ai4pex/
├── ml_model_ai4pex/                    # Installable Python package
│   ├── __init__.py
│   ├── cnn.py                          # Fully convolutional architecture
│   ├── unet.py                         # U-Net encoder-decoder architecture
│   ├── model_components.py             # Scenario dataclass, padding layer, provenance storage
│   ├── model_setup.py                  # Scenario setup, data loading, split and shuffle
│   ├── preprocess_data.py              # Data loading and preprocessing
│   ├── train_model.py                  # Training options and training loop
│   ├── predict_model.py                # Inference pipeline
│   ├── provenance.py                   # Training record saved with each model
│   └── parsing_args.py                 # YAML config and CLI argument parser
├── run/
│   ├── run_model.py                    # Main entry point script
│   ├── config_model.yml                # Example configuration file
│   └── submit_model.sh                 # Example SLURM submission script
├── examples/
│   ├── sample_dataset_for_github/
│   │   └── sample*                     # Sample datasets including input and target, and mask*, the mesh mask
│   ├── sample_model/
│   │   └── sample*                     # Sample pre-trained keras model
│   └── sample_offline_analysis.ipynb   # Sample notebook
├── tests/
│   ├── __init__.py
│   └── test_smoke.py                   # Smoke tests on synthetic data
├── requirements-lock.txt               # Exact versions of the environment used for the released models
└── pyproject.toml
```

## Running on a Cluster (SLURM)

An example SLURM submission script is provided at `run/submit_model.sh`. Edit the paths and resource requirements to match your cluster configuration before submitting. Currently spec'd for [JASMIN](https://jasmin.ac.uk/). It calls `run_model.py` and `config_model.yml` by relative path, so submit it from the `run/` directory.

## Running example notebook

A sample testing dataset and simple pre-trained keras model are found in directories `examples/sample_*`.

You can run this notebook yourself by installing `ml_model_ai4pex` and installing the following pip packages: `matplotlib`, `cmocean`, `scikit-image` and `scikit-learn`, plus Jupyter (`notebook`). The notebook reads the sample files by relative path, so start it from the `examples/` directory.

## Tests

```bash
pip install -e ".[test]"
pytest
```

This runs the smoke tests in `tests/test_smoke.py`. They use synthetic data and need none of the data files.

## Data & Reproducibility

### Data

The U-Net model is trained on high-resolution data from the [DINO configuration ](https://github.com/vopikamm/DINO/tree/DINO_4.2.1). The model uses coarse-grained fields of: $\zeta/f$ (vorticity over Coriolis parameter, `vor`), coarse kinetic energy (`coarse_ke`), $L_d/\Delta s$ (baroclinic deformation radius over grid scale, `sa`), to predict fine kinetic energy, or eddy kinetic energy (`fine_ke`).

The full dataset will be uploaded to a Zenodo repository on publication.

### Provenance

A model trained with `run/run_model.py` carries a record of its training run inside the `.keras` file: the options of the run (without the data paths), the data split and sample counts, the normalisation statistics, the git commit of the code, the versions of Python, TensorFlow, Keras, numpy and xarray, the machine the run was made on, and the training history. After loading, it is the dictionary `model.provenance`; it can also be read without loading the model:

```bash
unzip -p model.keras assets/provenance.json
```

A model saved before this record existed, such as the sample model in `examples/`, loads with `model.provenance` equal to `None`.

The commit in the record is that of the git repository holding the installed package files. It is the commit of this code when the package is installed in editable mode (`pip install -e .`) from a clone.

### Apply your own data

You can of course do this. Our workflow uses a combination of [xnemogcm](https://github.com/rcaneill/xnemogcm), [CDFTOOLS](https://github.com/meom-group/CDFTOOLS/tree/master), [xesmf](https://github.com/pangeo-data/xESMF), and [gcm-filters](https://github.com/ocean-eddy-cpt/gcm-filters).

The code expects, for each region, NetCDF files whose variables have the dimensions `t`, `y_c` and `x_c` and carry the coordinates `gphit` and `glamt`, and a mesh mask file with `tmask`, `e1t`, `e2t` and `ff_t`. The files in `examples/sample_dataset_for_github/` show the layout. A new region needs an entry in `dom_slice_dict` (see `domain` above).

## License

The code in this repository is licensed under the MIT License. See the `LICENSE` file for details. The full dataset used for the project is not included and is subject to separate access and use restrictions.
