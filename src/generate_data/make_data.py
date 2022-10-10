from collections import namedtuple
from math import floor


field_names = ['generate',
               'Xtrain',
               'Ytrain',
               'Xsemitrain',
               'd_X1',
               'd_X2',
               'd_Y',
               'seed']
Data = namedtuple(typename='Data', field_names=field_names, defaults=(None,) * len(field_names))


def make_data(cfg, builder):
    """Prepares and formats data to be used for training and testing.
    Returns all data objects needed to run experiment encapsulated in a namedtuple.
    Returned elements are not comprehensive and minimially needed to run experiments,
        they can be subject to change depending on needs.

    Args:
        cfg (dict): configuration file
        builder (callable): builder to instantiate data generator

    Returns:
        type: Data
    """
    # Instantiate synthetic data generator
    data_generator = builder(d_X1=cfg['data']['d_X1'],
                             d_X2=cfg['data']['d_X2'],
                             d_Y=cfg['data']['d_Y'],
                             noise=cfg['data']['noise'])

    # Extract useful variables from config
    n = cfg['data']['n']
    semi_prop = cfg['data']['semi_prop']
    seed = cfg['data']['seed']

    # Augment number of samples by semi-supervised proporition
    m = floor(n * (1 + semi_prop))

    # Generate dataset and split into supervised and semi-supervised
    X, Y = data_generator(n=m, seed=seed)
    Xtrain, Ytrain = X[:n], Y[:n]
    Xsemitrain = X

    # Encapsulate into named tuple object
    kwargs = {'generate': data_generator,
              'Xtrain': Xtrain,
              'Ytrain': Ytrain,
              'Xsemitrain': Xsemitrain,
              'd_X1': cfg['data']['d_X1'],
              'd_X2': cfg['data']['d_X2'],
              'd_Y': cfg['data']['d_Y'],
              'seed': seed,
              }
    data = Data(**kwargs)
    return data
