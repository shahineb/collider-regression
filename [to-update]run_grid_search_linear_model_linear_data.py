"""
Description : Runs hyperparameter search for linear regression regression experiment

Usage: run_grid_search_linear_model_linear_data.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
"""
import os
import yaml
import logging
from joblib import Parallel, delayed
from docopt import docopt
from tqdm import tqdm
from src.generate_data import make_data, linear
from utils import product_dict, flatten_dict_as_str
from run_linear_model_linear_data import make_model, fit, evaluate


def main(args, cfg):
    # Create cartesian product of grid search parameters
    hyperparams_grid = list(product_dict(**cfg['search']['grid']))

    # Create a single iteration function
    def iteration(hyperparams):
        # Flatten out hyperparameters into string to name output directory
        dirname = flatten_dict_as_str(hyperparams)
        output_dir = os.path.join(args['--o'], dirname)

        # Create directory and dump current set of hyperparameters
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'hyperparams.yaml'), 'w') as f:
            yaml.dump(hyperparams, f)

        # Run linear model
        scores = run(cfg, hyperparams)

        # Dump scores
        dump_path = os.path.join(output_dir, 'scores.metrics')
        with open(dump_path, 'w') as f:
            yaml.dump(scores, f)

    # Parallelise grid search
    Parallel(n_jobs=cfg['search']['n_jobs'])(delayed(iteration)(hyperparams)
                                             for hyperparams in tqdm(hyperparams_grid))


def run(cfg, hyperparams):
    # Create dataset
    data = make_data(cfg={'data': hyperparams}, builder=linear.build_data_generator)

    # Instantiate model
    baseline, collider = make_model(cfg)

    # Fit model
    baseline, collider = fit(baseline=baseline, collider=collider, data=data, cfg=cfg)

    # Run evaluation
    scores = evaluate(baseline=baseline, collider=collider, data=data, cfg=cfg)
    return scores


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Load config file
    with open(args['--cfg'], "r") as f:
        cfg = yaml.safe_load(f)

    # Setup logging
    logging.basicConfig(level=logging.ERROR)
    logging.info(f'Arguments: {args}\n')
    logging.info(f'Configuration file: {cfg}\n')

    # Create output directory if doesn't exists
    os.makedirs(args['--o'], exist_ok=True)
    with open(os.path.join(args['--o'], 'cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    # Run session
    main(args, cfg)
