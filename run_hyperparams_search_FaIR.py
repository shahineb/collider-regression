"""
Description : Runs hyperparameter search for kernel ridge regression model with FaIR data generating process

Usage: run_hyperparams_search_FaIR.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
  --device=<device_index>          Device to use [default: cpu]
"""
import os
import yaml
import logging
from joblib import Parallel, delayed
from docopt import docopt
from tqdm import tqdm
import torch
from gpytorch import kernels
import linear_operator
from src.models import KRR
from src.kernels import ProjectedKernel, ConstantKernel
from src.evaluation.metrics import spearman_correlation
from run_FaIR_experiment import make_data
from utils import product_dict, flatten_dict_as_str


def main(args, cfg):
    # Create cartesian product of grid search parameters
    hyperparams_baseline_grid = list(product_dict(**cfg['search']['baseline_grid']))
    hyperparams_before_grid = list(product_dict(**cfg['search']['before_grid']))
    hyperparams_after_grid = list(product_dict(**cfg['search']['after_grid']))

    # Create a single iteration functional
    def build_iteration(run_fn, model_dir):
        os.makedirs(os.path.join(args['--o'], model_dir), exist_ok=True)
        with open(os.path.join(args['--o'], model_dir, 'cfg.yaml'), 'w') as f:
            buffer = cfg.copy()
            buffer['search']['grid'] = buffer['search'].pop(f'{model_dir}_grid')
            yaml.dump(buffer, f)

        def iteration(hyperparams):
            # Flatten out hyperparameters into string to name output directory
            dirname = flatten_dict_as_str(hyperparams)
            output_dir = os.path.join(args['--o'], model_dir, dirname)

            # Create directory and dump current set of hyperparameters
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'hyperparams.yaml'), 'w') as f:
                yaml.dump(hyperparams, f)

            # Run model
            try:
                scores = run_fn(cfg, hyperparams)
            except linear_operator.utils.errors.NotPSDError:
                return

            # Dump scores
            dump_path = os.path.join(output_dir, 'scores.metrics')
            with open(dump_path, 'w') as f:
                yaml.dump(scores, f)
        return iteration

    # Parallelise grid search
    baseline_iteration = build_iteration(run_baseline, 'baseline')
    Parallel(n_jobs=cfg['search']['n_jobs'])(delayed(baseline_iteration)(hyperparams)
                                             for hyperparams in tqdm(hyperparams_baseline_grid))

    before_iteration = build_iteration(run_before, 'before')
    Parallel(n_jobs=cfg['search']['n_jobs'])(delayed(before_iteration)(hyperparams)
                                             for hyperparams in tqdm(hyperparams_before_grid))

    after_iteration = build_iteration(run_after, 'after')
    Parallel(n_jobs=cfg['search']['n_jobs'])(delayed(after_iteration)(hyperparams)
                                             for hyperparams in tqdm(hyperparams_after_grid))


def run_baseline(cfg, hyperparams):
    # Create dataset
    cfg['data']['seed'] = hyperparams['seed']
    data = make_data(cfg=cfg)

    # Instantiate base kernels
    k1 = kernels.RBFKernel(active_dims=list(range(data.d_X1))) + ConstantKernel()
    k2 = kernels.RBFKernel(active_dims=list(range(data.d_X1, data.Xtrain.size(1))))
    k = k1 * k2
    k1.kernels[0].lengthscale = hyperparams['k1_lengthscale']
    k2.lengthscale = hyperparams['k2_lengthscale']

    # Instantiate regressors
    baseline = KRR(kernel=k, λ=hyperparams['lbda_krr'])

    # Fit model
    Xtrain = (data.Xtrain - data.mu_X) / data.sigma_X
    Ytrain = (data.Ytrain - data.mu_Y) / data.sigma_Y
    baseline.fit(Xtrain, Ytrain)

    # Load samples to evaluate over
    X = torch.load(cfg['evaluation']['Xval_path'])
    Y = torch.load(cfg['evaluation']['Yval_path'])
    X = (X - data.mu_X) / data.sigma_X
    Y = (Y - data.mu_Y) / data.sigma_Y

    # Run prediction
    with torch.no_grad():
        pred_baseline = baseline(X)

    # Compute scores
    Y = Y.squeeze()
    baseline_rmse = torch.sqrt(torch.square(Y - pred_baseline).mean()).item()
    baseline_corr = spearman_correlation(Y, pred_baseline)

    # Make output dict
    scores = {'rmse': baseline_rmse,
              'corr': baseline_corr}
    return scores


def run_before(cfg, hyperparams):
    # Create dataset
    cfg['data']['seed'] = hyperparams['seed']
    data = make_data(cfg=cfg)

    # Instantiate base kernels
    k1 = kernels.RBFKernel(active_dims=list(range(data.d_X1))) + ConstantKernel()
    k2 = kernels.RBFKernel(active_dims=list(range(data.d_X1, data.Xtrain.size(1))))
    k = k1 * k2
    l = kernels.RBFKernel(active_dims=list(range(data.d_X1, data.Xtrain.size(1))))
    k1.kernels[0].lengthscale = hyperparams['k1_lengthscale']
    k2.lengthscale = hyperparams['k2_lengthscale']
    l.lengthscale = hyperparams['l_lengthscale']

    # Precompute kernel matrices
    Xsemitrain = (data.Xsemitrain - data.mu_X) / data.sigma_X
    K = k(Xsemitrain, Xsemitrain).evaluate()
    L = l(Xsemitrain, Xsemitrain)
    Lλ = L.add_diag(hyperparams['lbda_cme'] * torch.ones(L.shape[0]))
    chol = torch.linalg.cholesky(Lλ.evaluate())
    Lλ_inv = torch.cholesky_inverse(chol)

    # Instantiate projected kernel
    kP = ProjectedKernel(k, l, Xsemitrain, K, Lλ_inv)

    # Instantiate regressors
    project_before = KRR(kernel=kP, λ=hyperparams['lbda_krr'])

    # Fit model
    Xtrain = (data.Xtrain - data.mu_X) / data.sigma_X
    Ytrain = (data.Ytrain - data.mu_Y) / data.sigma_Y
    project_before.fit(Xtrain, Ytrain)

    # Load samples to evaluate over
    X = torch.load(cfg['evaluation']['Xval_path'])
    Y = torch.load(cfg['evaluation']['Yval_path'])
    X = (X - data.mu_X) / data.sigma_X
    Y = (Y - data.mu_Y) / data.sigma_Y

    # Run prediction
    with torch.no_grad():
        pred_before = project_before(X)

    # Compute scores
    Y = Y.squeeze()
    before_rmse = torch.sqrt(torch.square(Y - pred_before).mean()).item()
    before_corr = spearman_correlation(Y, pred_before)

    # Make output dict
    scores = {'rmse': before_rmse,
              'corr': before_corr}
    return scores


def run_after(cfg, hyperparams):
    # Create dataset
    cfg['data']['seed'] = hyperparams['seed']
    data = make_data(cfg=cfg)

    # Instantiate base kernels
    k1 = kernels.RBFKernel(active_dims=list(range(data.d_X1))) + ConstantKernel()
    k2 = kernels.RBFKernel(active_dims=list(range(data.d_X1, data.Xtrain.size(1))))
    k = k1 * k2
    l = kernels.RBFKernel(active_dims=list(range(data.d_X1, data.Xtrain.size(1))))
    k1.kernels[0].lengthscale = hyperparams['k1_lengthscale']
    k2.lengthscale = hyperparams['k2_lengthscale']
    l.lengthscale = hyperparams['l_lengthscale']

    # Precompute kernel matrices
    Xsemitrain = (data.Xsemitrain - data.mu_X) / data.sigma_X
    L = l(Xsemitrain, Xsemitrain)
    Lλ = L.add_diag(hyperparams['lbda_cme'] * torch.ones(L.shape[0]))
    chol = torch.linalg.cholesky(Lλ.evaluate())

    # Instantiate regressors
    baseline = KRR(kernel=k, λ=hyperparams['lbda_krr'])

    # Fit model
    Xtrain = (data.Xtrain - data.mu_X) / data.sigma_X
    Ytrain = (data.Ytrain - data.mu_Y) / data.sigma_Y
    baseline.fit(Xtrain, Ytrain)

    # Load samples to evaluate over
    X = torch.load(cfg['evaluation']['Xval_path'])
    Y = torch.load(cfg['evaluation']['Yval_path'])
    X = (X - data.mu_X) / data.sigma_X
    Y = (Y - data.mu_Y) / data.sigma_Y

    # Run prediction
    with torch.no_grad():
        pred_baseline = baseline(X)
        cme = torch.cholesky_solve(l(data.Xsemitrain, X).evaluate(), chol)
        pred_after = pred_baseline - baseline(data.Xsemitrain) @ cme

    # Compute scores
    Y = Y.squeeze()
    after_rmse = torch.sqrt(torch.square(Y - pred_after).mean()).item()
    after_corr = spearman_correlation(Y, pred_after)

    # Make output dict
    scores = {'rmse': after_rmse,
              'corr': after_corr}
    return scores


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
    logging.info("Grid search completed")
