#!/usr/bin/env python3

'''
    Description: 


'''


from ml_model_ai4pex.parsing_args import parse_args
from ml_model_ai4pex.model_setup import setup_scenario, \
      get_data, get_data_split, \
      get_data_shuffle
from ml_model_ai4pex.train_model import TrainingConfig, train_model
from ml_model_ai4pex.predict_model import predict_model
from ml_model_ai4pex.provenance import build_provenance
import tensorflow as tf

from datetime import datetime

import logging



if __name__ == "__main__":

    # parsing the arguments
    args = parse_args(None)

    # check if verbose output filename is provided, if not, set a default one
    if args.verbose_output_filename is None:
        args.verbose_output_filename = "./verbose_output.log"

    # setting up the logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
                filename=args.verbose_output_filename, 
                format="%(asctime)s %(levelname)s %(message)s",
                level=logging.INFO, 
                filemode='w'
    )

    logger.info("Starting the routine ...")

    # GPU Configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if args.verbose:
                logger.info(f"GPU(s) detected: {len(gpus)}")
        except RuntimeError as e:
            if args.verbose:
                logger.error(f"GPU configuration error: {e}")
    else:
        if args.verbose:
            logger.info("No GPUs detected. Training will use CPU.")

    # get the scenario for the CNN model
    scenario = setup_scenario(args, logger)

    # get the data for the scenario
    #! only using local normalisation for now
    ds, sc = get_data(scenario, args, logger)

    if args.train:

        # get the data splits for training, validation, and testing
        ds_train, ds_val = get_data_split(ds, args, logger)

        # shuffle the training data
        ds_train_shuffled = get_data_shuffle(ds_train, args, logger)

        print("ds_train_shuffled: ", ds_train_shuffled)

        # configure training options
        training_config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            dropout_rate=args.dropout_rate,
            use_learning_rate_scheduler=args.use_learning_rate_scheduler,
            learning_rate_decay_steps_multiplier=args.learning_rate_decay_steps_multiplier,
            learning_rate_decay_rate=args.learning_rate_decay_rate,
            use_early_stopping=args.use_early_stopping,
            early_stopping_monitor=args.early_stopping_monitor,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            early_stopping_start_from_epoch=args.early_stopping_start_from_epoch,
            use_ema=args.use_ema,
            ema_momentum=args.ema_momentum,
            model_type=args.model,
            model_dir=args.model_dir,
            model_filename=args.model_filename,
            tensboard_log_dir=args.tensboard_log_dir,
            verbose=args.verbose,
            current_period=datetime.now().strftime("%Y%m%d-%H%M%S"),
        )

        # train the model
        model = train_model(
            sc, 
            ds_train_shuffled, 
            ds_val, 
            training_config, 
            logger=logger
        )

        n_train_samples = ds_train_shuffled.sizes['sample']
        n_val_samples = ds_val.sizes['sample']

        # training record, saved inside the .keras file (assets/provenance.json)
        model.provenance = build_provenance(
            args, ds, sc,
            run_id=training_config.current_period,
            n_train_samples=n_train_samples,
            n_val_samples=n_val_samples,
            training_history=model.provenance['training_history'],
        )

        if args.model_save_filename:
            model_filename = args.model_dir + args.model_save_filename
            model.save(model_filename)
            logger.info(f"Model saved to {model_filename}")
        else:
            model_filename = args.model_dir + \
                f"model_{training_config.current_period}.keras"
            model.save(model_filename)
            logger.info(f"Model saved to {model_filename}")

    if args.predict:
        
        # get the data splits for testing
        ds_val, ds_test = get_data_split(ds, args, logger)

        # run predictions
        pred_ds = predict_model(sc, ds_test, args, logger=logger)

    logger.info("Model routine completed.")

