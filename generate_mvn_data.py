"""
Description : Generates large mvn training and testing set

Usage: generate_mvn_data.py  [options] --cfg=<path_to_config>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory [default: data/mvn/].
  --seed=<seed>                    Random seed.
  --val                            If True, also generates a validation set.
 """
import os
import yaml
import logging
from docopt import docopt
import torch
from src.generate_data import mvnposterior


def main(args, cfg):
    # Instantiate data generator
    data_generator = mvnposterior.build_data_generator(**cfg['train'])

    # Create training dataset
    logging.info("Generating training set")
    Xtrain, Ytrain = data_generator(n=cfg['train']['size'],
                                    seed=cfg['train']['seed'])

    # Dump training set
    torch.save(Xtrain, os.path.join(args['--o'], 'Xtrain.pt'))
    torch.save(Ytrain, os.path.join(args['--o'], 'Ytrain.pt'))
    logging.info(f"\n Dumped training set under {args['--o']}")

    if args['--val']:
        # Create validation dataset
        logging.info("Generating validation set")
        Xtest, Ytest = data_generator(n=cfg['val']['size'],
                                      seed=cfg['val']['seed'])

        # Dump validation set
        torch.save(Xtest, os.path.join(args['--o'], 'Xval.pt'))
        torch.save(Ytest, os.path.join(args['--o'], 'Yval.pt'))
        logging.info(f"\n Dumped validation set under {args['--o']}")

    # Create test dataset
    logging.info("Generating testing set")
    Xtest, Ytest = data_generator(n=cfg['test']['size'],
                                  seed=cfg['test']['seed'])

    # Dump test set
    torch.save(Xtest, os.path.join(args['--o'], 'Xtest.pt'))
    torch.save(Ytest, os.path.join(args['--o'], 'Ytest.pt'))
    logging.info(f"\n Dumped testing set under {args['--o']}")


def update_cfg(cfg, args):
    if args['--seed']:
        cfg['train']['seed'] = int(args['--seed'])
        cfg['val']['seed'] = int(args['--seed']) + 1
        cfg['test']['seed'] = int(args['--seed']) + 2
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
