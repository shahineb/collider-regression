"""
Description : Aggregate together in dataframe outputs from grid search run.

Usage: aggregate_hyperparams_search_results.py  [options] --i=<input_dir> --o=<output_dir>

Options:
  --i=<input_dir>                  Path to directory where grid search outputs are saved.
  --o=<output_dir>                 Output directory.
"""
import os
import yaml
import logging
from docopt import docopt
from progress.bar import Bar
import xarray as xr
import numpy as np
from utils import product_dict, flatten_dict, flatten_dict_as_str


def main(args):
    # Load configuration file corresponding to grid search
    with open(os.path.join(args['--i'], 'cfg.yaml'), "r") as f:
        cfg = yaml.safe_load(f)

    # Load all scores in xarray dataset with hyperparameters as dimensions
    logging.info("Loading grid search scores... (can take a while)")
    scores_dataset = open_scores_as_xarray(dirpath=args['--i'], cfg=cfg)
    logging.info(f"Loaded grid search scores \n {scores_dataset}")

    # Dump entire scores dataset and best candidates
    dump_results(scores_dataset=scores_dataset,
                 output_dir=args['--o'])
    logging.info(f"Dumped results in {args['--o']}")


def open_scores_as_xarray(dirpath, cfg):
    """Loads k-fold grid search scores (as saved by execution of grid search script)
    into xarray with dimensions (hyperparam_1, hyperparam_2, ..., hyperparam_n, fold)
    and with variables the different metrics.
    Args:
        dirpath (str): path to directory where grid search scores are saved.
        cfg (dict): configuration file used to perform grid search (typically saved in dirpath)
    Returns:
        type: xarray.Dataset
    """
    # Initialize hyperparameters grid
    hyperparams_grid = list(product_dict(**cfg['search']['grid']))

    # Initialize xarray dataset to record scores
    scores_dataset = init_scores_dataset(cfg=cfg)

    # Setup progress bar
    n_grid_points = len(hyperparams_grid)
    search_bar = Bar("Hyperparams", max=n_grid_points)

    # Iterate over sets of hyperparameters
    for j, hyperparams in enumerate(hyperparams_grid):

        # Look up corresponding directory
        hyperparams_dirname = flatten_dict_as_str(hyperparams)
        hyperparams_dirpath = os.path.join(dirpath, hyperparams_dirname)

        # Load corresponding scores
        scores_path = os.path.join(hyperparams_dirpath, 'scores.metrics')
        if os.path.isfile(scores_path):
            with open(scores_path, "r") as f:
                scores = flatten_dict(yaml.safe_load(f))

                # Initialize metrics dataarrays in scores dataset
                if not scores_dataset.data_vars.variables:
                    scores_dataset = init_metrics_dataarrays(scores_dataset=scores_dataset, metrics=list(scores.keys()))

                # Record in dataset the value of each metric
                for metric, value in scores.items():
                    scores_dataset[metric].loc[hyperparams] = value

        # Update progress bar
        search_bar.suffix = f"{j + 1}/{n_grid_points} | {hyperparams}"
        search_bar.next()
    return scores_dataset


def init_scores_dataset(cfg):
    """Initializes empty xarray dataset with dimensions (hyperparam_1, hyperparam_2, ..., hyperparam_n, fold)
    and no variables.
    Args:
        cfg (dict): configuration file used to perform grid search
    Returns:
        type: xarray.Dataset
    """
    cv_search_grid = cfg['search']['grid'].copy()
    scores_dataset = xr.Dataset(coords=cv_search_grid)
    return scores_dataset


def init_metrics_dataarrays(scores_dataset, metrics):
    """Initializes empty datarrays for each metric as variables of the scores dataset.
    Args:
        scores_dataset (xarray.Dataset): Empty xarray dataset with no variables.
        metrics (list[str]): list of metrics names.
    Returns:
        type: xarray.Dataset
    """
    dims = list(scores_dataset.dims.keys())
    shape = list(scores_dataset.dims.values())
    for metric in metrics:
        scores_dataset[metric] = (dims, np.full(shape, fill_value=np.nan))
    scores_dataset = scores_dataset.assign_attrs(fill_value=np.nan)
    return scores_dataset


def dump_results(scores_dataset, output_dir):
    dataset_path = os.path.join(output_dir, 'cv-search-scores.nc')
    scores_dataset.to_netcdf(path=dataset_path)


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Arguments: {args}\n')

    # Create output directory if doesn't exists
    os.makedirs(args['--o'], exist_ok=True)

    # Run
    main(args)
