"""
Description : Runs linear regression model with polynomial data generating process

Usage: run_linear_model_polynomial_data.py  [options] --cfg=<path_to_config> --o=<output_dir>

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
from sklearn.linear_model import LinearRegression
from src.generate_data import make_data, polynomial


def main(args, cfg):
    # Create dataset
    logging.info("Loading dataset")
    data = make_data(cfg=cfg, builder=polynomial.build_data_generator)

    # Instantiate model
    baseline, collider = make_model(cfg=cfg)
    logging.info(f"{baseline, collider}")

    # Fit model
    logging.info("\n Fitting model")
    baseline, collider = fit(baseline=baseline, collider=collider, data=data, cfg=cfg)

    # Run evaluation
    scores = evaluate(baseline=baseline, collider=collider, data=data, cfg=cfg)

    # Dump scores
    dump_path = os.path.join(args['--o'], 'scores.metrics')
    with open(dump_path, 'w') as f:
        yaml.dump(scores, f)
    logging.info(f"\n Dumped scores at {dump_path}")


def make_model(cfg):
    # Instantiate models
    baseline = LinearRegression()
    collider = LinearRegression()
    return baseline, collider


def fit(baseline, collider, data, cfg):
    # Fit model
    baseline.fit(data.Xtrain, data.Ytrain)
    collider.fit(data.Xsemitrain[:, data.d_X1:], baseline.predict(data.Xsemitrain))
    return baseline, collider


def evaluate(baseline, collider, data, cfg):
    # Generate samples to evaluate over
    X, Y = data.generate(n=cfg['evaluation']['n_test'],
                         seed=cfg['evaluation']['seed'])

    # Run prediction
    pred_baseline = torch.from_numpy(baseline.predict(X))
    pred_collider = pred_baseline - torch.from_numpy(collider.predict(X[:, data.d_X1:]))

    # Compute MSEs
    baseline_mse = torch.square(Y - pred_baseline).mean()
    collider_mse = torch.square(Y - pred_collider).mean()

    # Evaluate maximum possible gain
    X, Y = data.generate(n=cfg['evaluation']['n_test'],
                         seed=cfg['evaluation']['seed'],
                         most_gain=True)
    pred_most_gain = torch.from_numpy(baseline.predict(X))
    most_gain = torch.square(Y - pred_most_gain).mean()

    # Make output dict
    output = {'baseline': baseline_mse.item(),
              'collider': collider_mse.item(),
              'difference': baseline_mse.item() - collider_mse.item(),
              'most_gain': most_gain.item()}
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
