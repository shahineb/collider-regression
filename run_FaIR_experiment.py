"""
Description : Runs kernel ridge regression model with FaIR emulator data

Usage: run_FaIR_experiment.py  [options] --cfg=<path_to_config> --o=<output_dir>

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
from src import generate_data


def main(args, cfg):
    # Create dataset
    logging.info("Generating dataset")
    data = make_data(cfg=cfg)

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


def make_data(cfg):
    # Define data generator builder that loads pre-generated datasets and shuffles them
    def build_data_generator(Xtrain_path, Ytrain_path, **kwargs):
        def generate(n, seed=None):
            if seed:
                torch.random.manual_seed(seed)
            X = torch.load(Xtrain_path)
            Y = torch.load(Ytrain_path)
            rdm_idx = torch.randperm(len(X))[:n]
            X = X[rdm_idx]
            Y = Y[rdm_idx]
            return X, Y
        return generate
    data = generate_data.make_data(cfg=cfg, builder=build_data_generator)
    return data


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
        Xsemitrain = (data.Xsemitrain - data.mu_X)
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
    Xtrain = (data.Xtrain - data.mu_X)
    Ytrain = (data.Ytrain - data.mu_Y)
    baseline.fit(Xtrain, Ytrain)
    project_before.fit(Xtrain, Ytrain)
    return baseline, project_before


def evaluate(baseline, project_before, data, cfg):
    # Load samples to evaluate over
    logging.info("\n Loading testing set")
    X = torch.load(cfg['evaluation']['Xtest_path'])
    Y = torch.load(cfg['evaluation']['Ytest_path'])
    X = (X - data.mu_X)
    Y = (Y - data.mu_Y)
    Xsemitrain = (data.Xsemitrain - data.mu_X)

    # Run prediction
    with torch.no_grad():
        pred_baseline = baseline(X)
        pred_before = project_before(X)

        # Compute CMEs on test set
        Lλ_inv = project_before.kernel.Lλ_inv
        cme = Lλ_inv @ project_before.kernel.l(Xsemitrain, X).evaluate()

        # Project baseline model
        pred_after = pred_baseline - baseline(Xsemitrain) @ cme

        # Unstandardize predictions
        # pred_baseline = data.sigma_Y * pred_baseline + data.mu_Y
        # pred_before = data.sigma_Y * pred_before + data.mu_Y
        # pred_after = data.sigma_Y * pred_after + data.mu_Y

    # Compute MSEs
    zero_mse = torch.square(Y).mean()
    baseline_mse = torch.square(Y.squeeze() - pred_baseline).mean() / zero_mse
    before_mse = torch.square(Y.squeeze() - pred_before).mean() / zero_mse
    after_mse = torch.square(Y.squeeze() - pred_after).mean() / zero_mse

    # Compute correlations
    baseline_corr = torch.corrcoef(torch.stack([Y.squeeze(), pred_baseline]))[0, 1].item()
    before_corr = torch.corrcoef(torch.stack([Y.squeeze(), pred_before]))[0, 1].item()
    after_corr = torch.corrcoef(torch.stack([Y.squeeze(), pred_after]))[0, 1].item()

    # Compute SNR
    baseline_snr = 10 * torch.log10(torch.square(Y).sum() / torch.square(Y.squeeze() - pred_baseline).sum()).item()
    before_snr = 10 * torch.log10(torch.square(Y).sum() / torch.square(Y.squeeze() - pred_before).sum()).item()
    after_snr = 10 * torch.log10(torch.square(Y).sum() / torch.square(Y.squeeze() - pred_after).sum()).item()

    # Make output dict
    output = {'mse_baseline': baseline_mse.item(),
              'mse_before': before_mse.item(),
              'mse_after': after_mse.item(),
              'corr_baseline': baseline_corr,
              'corr_before': before_corr,
              'corr_after': after_corr,
              'snr_baseline': baseline_snr,
              'snr_before': before_snr,
              'snr_after': after_snr}
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
