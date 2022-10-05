# rkhs-collider


## Running a grid search experiment

- Setup configuration file of the experiment + search grid under `config/search/my_config.yaml`

- Run grid search script. For example, KRR with polynomial data generating process would be run as
```bash
$ python run_grid_search_kernel_model_polynomial_data.py --cfg=config/search/my_config.yaml --o=my/output/directory/
```

- Aggregate search results together into an xarray dataset
```bash
$ python aggregate_grid_search_results.py --i=my/output/directory --o=another/output/directory
```

This last script will dump a `cv-search-scores.nc` file which can then be loaded in the notebook `notebooks/score-analysis.ipynb` to visualise metrics against searched hyperparameters values.
