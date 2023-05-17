import torch
import numpy as np
import tqdm
from fair.RCPs import rcp45
import src.generate_data.fair as fair


def build_data_generator(noise, **kwargs):
    # Initialise FaIR
    base_kwargs = fair.get_params()
    base_kwargs.update(forcing_noise=noise * np.array([1., 0., 1., 1.]), f2=base_kwargs['f2'])
    years = rcp45.Emissions.year
    emissions = rcp45.Emissions.emissions[:, [1, 3, 5, 9]].T
    emissions[2] = emissions[2] * 2  # correct for underestimated SO2 emissions

    # Define sampling method
    def sample():
        res = fair.run(time=years,
                       emission=emissions,
                       base_kwargs=base_kwargs)
        year_idx = 255  # year = 2020
        Fco2 = res['RF'][0, year_idx]
        Faer = res['RF'][2:, year_idx].sum(axis=0)
        T = res['T'][year_idx]
        return Fco2, Faer, T

    # Define utility that generates n samples
    def generate_data(n):
        Fco2, Faer, T = [], [], []
        for i in tqdm.tqdm(range(n)):
            draw = sample()
            Fco2.append(draw[0])
            Faer.append(draw[1])
            T.append(draw[2])
        Fco2 = torch.tensor(Fco2)
        Faer = torch.tensor(Faer)
        T = torch.tensor(T)
        X = torch.stack([T, Fco2], dim=1)
        Y = Faer
        return X, Y

    # Define utility that generates samples for most gain evaluation
    def generate_most_gain_data(n, most_gain_sample):
        raise NotImplementedError

    # Define utility that wraps up above
    def generate(n, seed=None, most_gain=False, most_gain_samples=0):
        if seed:
            torch.random.manual_seed(seed)
            np.random.seed(seed)
        if most_gain:
            raise NotImplementedError
        else:
            X, Y = generate_data(n)
        return X, Y

    # Return utility for usage
    return generate
