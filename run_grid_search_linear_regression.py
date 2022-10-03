"""
Description : Runs hyperparameter search for linear regression regression experiment

Usage: run_grid_search_linear_regression.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
"""
import os
import yaml
import logging
from docopt import docopt
from progress.bar import Bar
from utils import product_dict, flatten_dict_as_str
from run_linear_regression import run


def main(args, cfg):
    # Create cartesian product of grid search parameters
    hyperparams_grid = list(product_dict(**cfg['search']['grid']))
    n_grid_points = len(hyperparams_grid)
    search_bar = Bar("Grid Search", max=n_grid_points)

    # Iterate over combinations of hyperparameters
    for j, hyperparams in enumerate(hyperparams_grid):

        # Flatten out hyperparameters into string to name output directory
        dirname = flatten_dict_as_str(hyperparams)
        output_dir = os.path.join(args['--o'], dirname)

        # Create directory and dump current set of hyperparameters
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'hyperparams.yaml'), 'w') as f:
            yaml.dump(hyperparams, f)

        # Run linear model
        scores = run(**hyperparams)

        # Dump scores
        dump_path = os.path.join(output_dir, 'scores.metrics')
        with open(dump_path, 'w') as f:
            yaml.dump(scores, f)

        # Update progress bar
        search_bar.suffix = f"{j + 1}/{n_grid_points} | {hyperparams}"
        search_bar.next()


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Load config file
    with open(args['--cfg'], "r") as f:
        cfg = yaml.safe_load(f)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Arguments: {args}\n')
    logging.info(f'Configuration file: {cfg}\n')

    # Create output directory if doesn't exists
    os.makedirs(args['--o'], exist_ok=True)
    with open(os.path.join(args['--o'], 'cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    # Run session
    main(args, cfg)
