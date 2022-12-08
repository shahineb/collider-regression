"""
Description : Runs kernel ridge regression model with FaIR emulator data

Usage: run_kernel_model_FaIR_data.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
  --seed=<seed>                    Random seed.
  --semi_prop=<semi_prop>          Proportion of semi-supervised samples.
  --plot                           Outputs plots.
"""
import os
import yaml
import logging
from docopt import docopt
import torch
from gpytorch import kernels
from src.models import KRR
from src.kernels import ProjectedKernel, ConstantKernel
from src.generate_data import make_data, FaIR


def main(args, cfg):
    # Create dataset
    logging.info("Generating dataset")
    data = make_data(cfg=cfg, builder=FaIR.build_data_generator)

    # Instantiate model
    baseline, project_before = make_model(cfg=cfg, data=data)
    logging.info(f"{baseline, project_before}")

    # Fit model
    logging.info("\n Fitting model")
    baseline, project_before = fit(baseline=baseline,
                                   project_before=project_before,
                                   data=data,
                                   cfg=cfg)

    # Run evaluation
    scores = evaluate(baseline=baseline,
                      project_before=project_before,
                      data=data,
                      cfg=cfg)

    # Dump scores
    dump_path = os.path.join(args['--o'], 'scores.metrics')
    with open(dump_path, 'w') as f:
        yaml.dump(scores, f)
    logging.info(f"\n Dumped scores at {dump_path}")


def make_model(cfg, data):
    # Instantiate base kernels
    k1 = kernels.RBFKernel(active_dims=list(range(data.d_X1))) + ConstantKernel()
    k2 = kernels.RBFKernel(active_dims=list(range(data.d_X1, data.Xtrain.size(1))))
    k = k1 * k2
    l = kernels.RBFKernel(active_dims=list(range(data.d_X1, data.Xtrain.size(1))))
    k1.kernels[0].lengthscale = cfg['model']['k1']['lengthscale']
    k2.lengthscale = cfg['model']['k2']['lengthscale']
    l.lengthscale = cfg['model']['l']['lengthscale']

    # Precompute kernel matrices
    with torch.no_grad():
        Xsemitrain = (data.Xsemitrain - data.mu_X) / data.sigma_X
        K = k(Xsemitrain, Xsemitrain).evaluate()
        L = l(Xsemitrain, Xsemitrain)
        Lλ = L.add_diag(cfg['model']['cme']['lbda'] * torch.ones(L.shape[0]))
        chol = torch.linalg.cholesky(Lλ.evaluate())
        Lλ_inv = torch.cholesky_inverse(chol)

    # Instantiate projected kernel
    kP = ProjectedKernel(k, l, Xsemitrain, K, Lλ_inv)

    # Instantiate regressors
    baseline = KRR(kernel=k, λ=cfg['model']['baseline']['lbda'])
    project_before = KRR(kernel=kP, λ=cfg['model']['project_before']['lbda'])
    return baseline, project_before


def fit(baseline, project_before, data, cfg):
    # Fit baseline and "project before" model
    Xtrain = (data.Xtrain - data.mu_X) / data.sigma_X
    Ytrain = (data.Ytrain - data.mu_Y) / data.sigma_Y
    baseline.fit(Xtrain, Ytrain)
    project_before.fit(Xtrain, Ytrain)
    return baseline, project_before


def evaluate(baseline, project_before, data, cfg):
    # Generate samples to evaluate over
    logging.info("\n Generating testing set")
    X, Y = data.generate(n=cfg['evaluation']['n_test'],
                         seed=cfg['evaluation']['seed'])
    X = (X - data.mu_X) / data.sigma_X
    Y = (Y - data.mu_Y) / data.sigma_Y
    Xsemitrain = (data.Xsemitrain - data.mu_X) / data.sigma_X

    # Run prediction
    with torch.no_grad():
        pred_baseline = baseline(X)
        pred_before = project_before(X)

        # Compute CMEs on test set
        Lλ_inv = project_before.kernel.Lλ_inv
        cme = Lλ_inv @ project_before.kernel.l(Xsemitrain, X).evaluate()

        # Project baseline model
        pred_after = pred_baseline - baseline(Xsemitrain) @ cme

    # Compute MSEs
    baseline_mse = torch.square(Y.squeeze() - pred_baseline).mean()
    before_mse = torch.square(Y.squeeze() - pred_before).mean()
    after_mse = torch.square(Y.squeeze() - pred_after).mean()

    # Make output dict
    output = {'baseline': baseline_mse.item(),
              'before': before_mse.item(),
              'after': after_mse.item()}
    return output


def update_cfg(cfg, args):
    if args['--seed']:
        cfg['data']['seed'] = int(args['--seed'])
        cfg['evaluation']['seed'] = int(args['--seed'])
    if args['--semi_prop']:
        cfg['data']['semi_prop'] = float(args['--semi_prop'])
    return cfg


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Load config file
    with open(args['--cfg'], "r") as f:
        cfg = yaml.safe_load(f)
    cfg = update_cfg(cfg, args)

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
