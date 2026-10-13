#!/usr/bin/python3
"""Command-line argument parsing for model training and prediction."""


import argparse
import yaml

def tuple2(k):
    k = [tuple(map(int, k.strip('()').split(','))) \
         for k in k.split()]
    return k

def list1(k): 
    k = [int(i) for i in k.split()]
    return k

def list2(k):
    k = [k]
    return k

def split1(k: str):
    k = k.split()
    return k

def parse_args(argv=None):

    parser = argparse.ArgumentParser(description="Train a NN model on preprocessed data.")

    # NEW: YAML config
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")

    # filenames and directories
    parser.add_argument("--data_dir", type=str, help="Directory containing the preprocessed data files.")
    parser.add_argument("--data_filenames", type=split1, help="List of filenames to load the data from.")
    parser.add_argument("--domain", type=split1, default=None, help="Domain identifier for data files.")
    parser.add_argument("--model_dir", type=str, help="Directory to save/load the trained model.")
    parser.add_argument("--model_filename", type=str, default=None, help="Name of the model file to load.")
    parser.add_argument("--model_save_filename", type=str, default=None, help="Name of the model file to save to.")
    parser.add_argument("--predict_save_dir", type=str, default=None, help="Directory to save prediction output to.")
    parser.add_argument("--predict_save_filename", type=str, default=None, help="Filename to save prediction output to (NetCDF). If not set, predictions are not saved.")

    # data slicing
    parser.add_argument("--train_ratio", type=float, default=None, help="Ratio of data to use for training.")
    parser.add_argument("--val_ratio", type=float, default=None, help="Ratio of data to use for validation.")
    parser.add_argument("--test_ratio", type=float, default=None, help="Ratio of data to use for testing.")
    parser.add_argument("--train_stride", type=int, default=None, help="Stride to use when sampling the training data.")
    parser.add_argument("--shuffle_seed", type=int, default=None, help="Random seed for shuffling the data. Omit for no shuffling.")

    # choice of model architecture
    parser.add_argument("--model", type=str, default=None, help="Model architecture to use (e.g. 'unet' or 'cnn').")

    # model features and parameters
    # CNN-specific parameters
    parser.add_argument("--kernels", type=tuple2, default=None, help="Size of the convolutional kernel.")
    parser.add_argument("--padding", type=tuple2, default=None, help="Convolutional padding.")
    parser.add_argument("--dilation_rates", type=list1, default=None, help="Dilation rates for the convolutional layers.")
    parser.add_argument("--filters", type=list1, default=None, help="Number of filters in the convolutional layers.")

    # U-NET specific parameters
    parser.add_argument("--base_filters", type=int, default=None,
                    help="Base filter count for U-Net.")
    parser.add_argument("--depth", type=int, default=None,
                    help="Encoder depth for U-Net.")

    # General training parameters
    parser.add_argument("--features", type=split1, help="List of feature names to use for training.")
    parser.add_argument("--target", type=list2, help="Target.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train the model.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer. \
                        If --use_learning_rate_scheduler is set, this is the initial learning rate.")
    parser.add_argument("--use_learning_rate_scheduler", default=None, \
                            help="Flag to indicate whether to use a learning rate scheduler. \
                            This is exponential decay.") 
    parser.add_argument("--learning_rate_decay_steps_multiplier", type=int, default=None, help="Multiplier for the number of steps before applying decay to the learning rate. \
                        The number of steps is calculated as learning_rate_decay_steps_multiplier * (number of training samples / batch size).")
    parser.add_argument("--learning_rate_decay_rate", type=float, default=None, help="Decay rate for the learning rate scheduler.")
    parser.add_argument("--dropout_rate", type=float, default=None, help="Dropout rate for the dropout layers.")
    parser.add_argument("--use_ema", default=None, help="Flag to use Exponential Moving Average of model weights.")
    parser.add_argument("--ema_momentum", type=float, default=None, help="Momentum for EMA (e.g. 0.99).")
    parser.add_argument("--use_early_stopping", default=None, help="Flag to use early stopping during training.")
    parser.add_argument("--early_stopping_patience", type=int, default=None, help="Number of epochs with no improvement after which training will be stopped when using early stopping.")
    parser.add_argument("--early_stopping_min_delta", type=float, default=None, help="Minimum change in the monitored metric to qualify as an improvement.")
    parser.add_argument("--early_stopping_monitor", type=str, default=None, help="Metric to monitor for early stopping (e.g. 'val_loss').")
    parser.add_argument("--early_stopping_start_from_epoch", type=int, default=None, help="Epoch number to start monitoring for early stopping. This allows the model some time to start learning before early stopping can kick in.")

    # key flags associated with training, evaluation, prediction
    parser.add_argument("--train", default=None, help="Flag to indicate whether to train the model.")
    # parser.add_argument("--evaluate", action="store_true", help="Flag to indicate whether to evaluate the model.")
    parser.add_argument("--predict", default=None, help="Flag to indicate whether to predict using the model.")


    parser.add_argument("--verbose", default=None, help="Verbose output during training.")
    parser.add_argument("--local_norm", default=None, help="Normalize training data within local domain.")

    # logging and monitoring
    # # parser.add_argument("--tensboard_log_dir", type=str, default=None, help="Directory to save TensorBoard logs.")
    parser.add_argument("--tensboard_log_dir", type=str, default=None, help="Name of the TensorBoard log directory.")
    # # parser.add_argument("--verbose_output_dir", type=str, default=None, help="Directory to save verbose output logs.")
    parser.add_argument("--verbose_output_filename", type=str, default=None, help="Name of the verbose output log file.")
    parser.add_argument("--norm_stats", type=str, default=None, help="Normalization statistics dict (set via YAML config, not CLI).")
 
    # arguments that come in via the sbatch script
    pre_args, remaining = parser.parse_known_args(argv)

    # print("Pre args:", pre_args)
    # print("Remaining args:", remaining)

    # load in arguments from yaml config file
    cfg_nested = {}
    if pre_args.config:
        cfg_nested = yaml.safe_load(open(pre_args.config, 'r'))
        
        mode = cfg_nested.pop("mode")
        if mode == "train":
            cfg_nested["train"] = True
        elif mode == "predict":
            cfg_nested["predict"] = True
        else:
            raise ValueError(f"No other modes supported yet: {mode}")
        
        # retrieve the nested config values
        cfg = {}
        for parent_keys, parent_values in cfg_nested.items():
            # print(parent_values)
            if isinstance(parent_values, dict):
                # print(parent_values)
                for child_key, child_values in parent_values.items():
                    # print(child_key, child_values)
                    cfg[child_key] = child_values
            #         # print(f"{child_key}")
            else:
                cfg[parent_keys] = parent_values
        
        # override arguments from yaml config file with command line arguments (if provided)
        # an argument still at its parser default is taken as not given on the command
        # line, so the yaml value is kept (otherwise learning_rate in the yaml is ignored);
        # a flag given with its default value still wins in the final parse below
        for k, v in cfg.items():
            if getattr(pre_args, k) is not None and getattr(pre_args, k) != parser.get_default(k):
                # print(f"Overriding config value for {k} with command line argument: {getattr(pre_args, k)}")
                cfg[k] = getattr(pre_args, k)
        
        # print("cfg.items() is:", cfg.items())
        
        # gets the valid dict keys from the parser
        valid_dests = set()
        for action in parser._actions:
            valid_dests.add(action.dest)

        # print("Valid dests:", valid_dests)


        kept = {}
        for k, v in cfg.items():
            if k in valid_dests:
                kept[k] = v

        # print("Kept:", kept)

        parser.set_defaults(**kept)

    # re-parse the full command line with the yaml values as defaults, so any
    # flag given on the command line wins, including ones absent from the yaml
    # (parsing only `remaining` silently dropped those, e.g. --model_filename)
    args = parser.parse_args(argv)

    # check if any required arguments are missing
    required_args = [
        'data_dir',
        'data_filenames',
        'domain',
        'model_dir',
        'features',
        'target',
    ]
    if args.train:
        if args.model == "unet":
            required_args += [
                'epochs',
                'batch_size',
                'base_filters',
                'depth',
            ]
        else:
            required_args += [
                'epochs',
                'batch_size',
                'filters',
                'kernels',
                'padding',
            ]
    if args.predict:
        required_args += [
            'model_filename',
        ]

    missing_args = [arg for arg in required_args if getattr(args, arg) is None]
    
    if missing_args:
        parser.error(f"Missing required arguments: {', '.join(missing_args)}")

    if args.local_norm is None:
        parser.error("Must specify local_norm.")

    return args