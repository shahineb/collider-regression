"""
Description : Runs kernel ridge regression model with polynomial data generating process

Usage: run_kernel_model_polynomial_data.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
  --seed=<seed>                    Random seed.
  --plot                           Outputs plots.
"""
import os
import yaml
import logging
from docopt import docopt
import torch
from gpytorch import kernels
from src.models import KRR, CME_mat
from src.kernels import ProjectedKernel
from src.generate_data import make_data, rotation


def main(args, cfg):
    # Create dataset
    logging.info("Loading dataset")
    data = make_data(cfg=cfg, builder=rotation.build_data_generator)

    # Instantiate model
    baseline, project_before, project_after = make_model(cfg=cfg, data=data)
    logging.info(f"{baseline, project_before, project_after}")

    # Fit model
    logging.info("\n Fitting model")
    baseline, project_before, project_after = fit(baseline=baseline,
                                                  project_before=project_before,
                                                  project_after=project_after,
                                                  data=data,
                                                  cfg=cfg)

    # Run evaluation
    scores = evaluate(baseline=baseline,
                      project_before=project_before,
                      project_after=project_after,
                      data=data,
                      cfg=cfg)

    # Dump scores
    dump_path = os.path.join(args['--o'], 'scores.metrics')
    with open(dump_path, 'w') as f:
        yaml.dump(scores, f)
    logging.info(f"\n Dumped scores at {dump_path}")


def make_model(cfg, data):
    # Instantiate base kernels
    k = kernels.RBFKernel()
    l = kernels.RBFKernel()
    k.lengthscale = 10.
    l.lengthscale = 10.

    # Precompute kernel matrices
    K = k(data.Xsemitrain, data.Xsemitrain).evaluate()
    L = l(data.Xsemitrain, data.Xsemitrain)
    Lλ = L.add_diag(cfg['model']['cme']['lbda'] * torch.ones(L.shape[0]))
    Lλ_inv = torch.cholesky_inverse(Lλ.evaluate())

    # Instantiate projected kernel
    kP = ProjectedKernel(k, l, data.Xsemitrain, K, Lλ_inv)

    # Instantiate regressors
    baseline = KRR(kernel=k, λ=cfg['model']['baseline']['lbda'])
    project_before = KRR(kernel=kP, λ=cfg['model']['project_before']['lbda'])
    project_after = KRR(kernel=k, λ=cfg['model']['project_after']['lbda'])
    return baseline, project_before, project_after


def fit(baseline, project_before, project_after, data, cfg):
    # Fit baseline and "project before" model
    baseline.fit(data.Xtrain, data.Ytrain)
    project_before.fit(data.Xtrain, data.Ytrain)
    # Use weights of baseline model to mimic a "project after" behavior
    project_after_weight = baseline.α 
    project_after.register_buffer('α', project_after_weight)
    project_after.register_buffer('X', baseline.X)
    return baseline, project_before, project_after


def evaluate(baseline, project_before, project_after, data, cfg):
    # Generate samples to evaluate over
    X, Y = data.generate(n=cfg['evaluation']['n_test'],
                         seed=cfg['evaluation']['seed'])

    # Run prediction
    with torch.no_grad():
        pred_baseline = baseline(X)
        pred_before = project_before(X)
        #pred_after = project_after(X)
        pred_after = baseline(X)- project_before.kernel(X, project_before.kernel.X) @ project_before.kernel.Lλ_inv @ baseline(project_before.kernel.X)
    
    # Compute MSEs
    baseline_mse = torch.square(Y - pred_baseline).mean()
    before_mse = torch.square(Y - pred_before).mean()
    after_mse = torch.square(Y - pred_after).mean()
    
    # Compute CME Fit:

    if cfg["evaluation"]["most_gain"]:
        d = cfg["evaluation"]["n_gain"]
        X,Y=data.generate(n=cfg['evaluation']['n_test'],
                         seed=cfg['evaluation']['seed'],most_gain=True, most_gain_samples=d)
        pred_baseline_avg=torch.zeros_like(Y)
        for i in range(d):
            with torch.no_grad():             
                pred_slice = baseline(X[:,:,i])
            pred_baseline_avg += pred_slice
        pred_baseline_avg=1/d*pred_baseline_avg
        most_gain = torch.square(Y - pred_baseline_avg).mean()

    # Make output dict
    if cfg["evaluation"]["most_gain"]:
        output = {'baseline': baseline_mse.item(),
                  'before': before_mse.item(),
                  'after': after_mse.item(),
                  "most_gain":most_gain.item()} 
    else:   
        output = {'baseline': baseline_mse.item(),
                'before': before_mse.item(),
                'after': after_mse.item()}
    return output


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
