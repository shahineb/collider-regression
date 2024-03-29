"""
Description : Runs kernel ridge regression model with mvn data generating process

Usage: run_mvn_experiment.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
  --seed=<seed>                    Random seed.
  --n=<n_train>                    Number of training samples.
  --semi_prop=<semi_prop>          Proportion of semi-supervised samples.
  --d_X2=<d_X2>                    Dimension of X2.
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
from src.generate_data import make_data, mvn


def main(args, cfg):
    # Create dataset
    logging.info("Loading dataset")
    data = make_data(cfg=cfg, builder=mvn.build_data_generator)

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
    r_plus = kernels.RBFKernel(active_dims=list(range(data.d_X1))) + ConstantKernel()
    l = kernels.RBFKernel(active_dims=list(range(data.d_X1, data.Xtrain.size(1))))
    k = r_plus * l
    r_plus.kernels[0].lengthscale = cfg['model']['r']['lengthscale']
    l.lengthscale = cfg['model']['l']['lengthscale']

    # Precompute kernel matrices
    with torch.no_grad():
        R_plus = r_plus(data.Xsemitrain, data.Xsemitrain).evaluate()
        L = l(data.Xsemitrain, data.Xsemitrain)
        Lλ = L.add_diag(cfg['model']['cme']['lbda'] * torch.ones(L.shape[0]))
        chol = torch.linalg.cholesky(Lλ.evaluate())
        Lλ_inv = torch.cholesky_inverse(chol)

    # Instantiate projected kernel
    kP = ProjectedKernel(r_plus, l, data.Xsemitrain, R_plus, Lλ_inv)

    # Instantiate regressors
    baseline = KRR(kernel=k, λ=cfg['model']['baseline']['lbda'])
    project_before = KRR(kernel=kP, λ=cfg['model']['project_before']['lbda'])
    return baseline, project_before


def fit(baseline, project_before, data, cfg):
    # Fit baseline and "project before" model
    baseline.fit(data.Xtrain, data.Ytrain)
    project_before.fit(data.Xtrain, data.Ytrain)
    return baseline, project_before


def evaluate(baseline, project_before, data, cfg):
    # Generate samples to evaluate over
    X, Y = data.generate(n=cfg['evaluation']['n_test'],
                         seed=cfg['evaluation']['seed'])

    # Run prediction
    with torch.no_grad():
        pred_baseline = baseline(X)
        pred_before = project_before(X)

        # Compute conditional expectation of baseline on test set
        Lλ_inv = project_before.kernel.Lλ_inv
        Rplus = project_before.kernel.r_plus(data.Xtrain, data.Xsemitrain).evaluate()
        Λ = project_before.kernel.l(X, data.Xtrain).evaluate()
        Λ_Rplus = Λ.view(-1, len(data.Xtrain), 1).mul(Rplus)
        CE_pred_baseline = (Lλ_inv @ project_before.kernel.l(data.Xsemitrain, X).evaluate()).T
        CE_pred_baseline = torch.bmm(Λ_Rplus, CE_pred_baseline.unsqueeze(-1)).squeeze()
        CE_pred_baseline = CE_pred_baseline @ baseline.α

        # Project baseline model
        pred_after = pred_baseline - CE_pred_baseline

    # Compute MSEs
    baseline_mse = torch.square(Y.squeeze() - pred_baseline).mean()
    before_mse = torch.square(Y.squeeze() - pred_before).mean()
    after_mse = torch.square(Y.squeeze() - pred_after).mean()

    # New most gain
    d = cfg["evaluation"]["n_gain"]
    X, _ = data.generate(n=cfg['evaluation']['n_test'],
                         seed=cfg['evaluation']['seed'],
                         most_gain=True,
                         most_gain_samples=d)
    pred_baseline_avg = torch.zeros(X.size(1))
    for i in range(d):
        with torch.no_grad():
            pred_slice = baseline(X[i])
        pred_baseline_avg += pred_slice
    pred_baseline_avg = 1 / d * pred_baseline_avg
    most_gain = torch.square(pred_baseline_avg).mean()

    # Make output dict
    output = {'baseline': baseline_mse.item(),
              'before': before_mse.item(),
              'after': after_mse.item(),
              'most_gain': most_gain.item()}
    return output


def update_cfg(cfg, args):
    if args['--seed']:
        cfg['data']['seed'] = int(args['--seed'])
        cfg['evaluation']['seed'] = int(args['--seed'])
    if args['--n']:
        cfg['data']['n'] = int(args['--n'])
    if args['--semi_prop']:
        cfg['data']['semi_prop'] = int(args['--semi_prop'])
    if args['--d_X2']:
        cfg['data']['d_X2'] = int(args['--d_X2'])
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
