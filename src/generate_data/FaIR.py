import torch
import numpy as np
import tqdm
from fair.RCPs import rcp45
import src.generate_data.fair as fair


def build_data_generator(noise, **kwargs):
    # Initialise FaIR
    base_kwargs = fair.get_params()
    # forcings_stddevs = np.array([1.46985971, 0.17249708, 0.01918899, 0.00458906])
    forcings_stddevs = np.array([14.6985971, 0.17249708, 19.18899, 0.00458906])
    f2 = np.array([0., 0., -0.126, 0.00302])
    base_kwargs.update(forcing_noise=noise * forcings_stddevs, f2=f2)
    years = rcp45.Emissions.year
    emissions = rcp45.Emissions.emissions[:, [1, 3, 5, 9]].T

    # Define sampling method
    def sample():
        res = fair.run(time=years,
                       emission=emissions,
                       base_kwargs=base_kwargs)
        year_idx = 258  # year = 2023
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
