"""
This script demonstrates how to retrieve data from wandb using the stable-SSL library.
"""

import stable_ssl as ssl

config, df = ssl.utils.reader.wandb_run(
    "excap", "single_dataset_sequential", "p67ng6bq"
)
print(df)
configs, dfs = ssl.utils.reader.wandb_project("excap", "single_dataset_sequential")
print(dfs)
