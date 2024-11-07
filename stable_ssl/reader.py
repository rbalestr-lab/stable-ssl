# -*- coding: utf-8 -*-
"""Reader for logs."""
#
# Author: Randall Balestriero <randallbalestriero@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

try:
    import wandb
except ModuleNotFoundError:
    logging.warning(
        "Wandb module is not installed, make sure to not use wandb for logging "
        "or an error will be thrown."
    )
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import jsonlines
from pathlib import Path
import omegaconf
import logging
from tqdm.contrib.logging import logging_redirect_tqdm

# yaml.add_constructor("tag:yaml.org,2002:python/object/apply:pathlib.PosixPath", Path)


def jsonl_project(folder, num_workers=8):
    """Load configs and values from runs in a folder."""
    if not Path(folder).is_dir():
        raise ValueError(f"The provided folder ({folder}) is not a directory!")
    runs = list(Path(folder).rglob("*/hparams.yaml"))
    configs = []
    values = []
    logging.basicConfig(level=logging.INFO)
    with logging_redirect_tqdm():
        args = [run.parent for run in runs]
        with Pool(num_workers) as p:
            results = list(tqdm(p.imap(jsonl_run, args), total=len(runs)))
        for c, v in results:
            configs.append(flatten_config(c))
            values.append(v)
    config = pd.DataFrame(configs)
    return config, values


def jsonl_run(path):
    """Load config and values from a single run directory."""
    _path = Path(path)
    if not _path.is_dir():
        raise ValueError(f"The provided path ({path}) is not a directory!")
    # Load the config file.
    if not (_path / "hparams.yaml").is_file():
        raise ValueError(
            f"The provided path ({path}) must at least contain a `hparams.yaml` file."
        )
    config = omegaconf.OmegaConf.load(_path / "hparams.yaml")
    values = []
    # Load the values from logs_rank_* files.
    logs_files = list(_path.glob("logs_rank_*"))
    for log_file in logs_files:
        # Extract rank from the filename.
        rank = int(log_file.name.split("_")[-1])
        for obj in jsonlines.open(log_file).iter(type=dict, skip_invalid=True):
            obj["rank"] = rank  # Add rank field to each dict.
            values.append(obj)
    return config, values


def wandb_project(
    entity, project, max_steps=-1, keys=None, num_workers=10, state="finished"
):
    """Download configs and data from a wandb project."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    runs = [r for r in runs if r.state == state]

    configs = []
    dfs = []
    with Pool(num_workers) as p:
        results = list(
            tqdm(
                p.imap(
                    _wandb_run_packed,
                    [(entity, project, r.id, max_steps, keys) for r in runs],
                ),
                total=len(runs),
                desc=f"Downloading project: {project}",
            )
        )
    configs, dfs = zip(*results)
    config = pd.DataFrame(configs)
    return config, dfs


def _wandb_run_packed(args):
    """Help function to unpack arguments for wandb_run."""
    return wandb_run(*args)


def wandb_run(entity, project, run_id, max_steps=-1, keys=None):
    """Download config and data for a single wandb run."""
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    if max_steps == -1:
        max_steps = run.lastHistoryStep
        # min_step = 0
    # else:
    # min_step = run.lastHistoryStep - max_steps

    summary = run.summary
    # extract names that are not hidden
    columns = [k for k, v in summary.items() if k[0] != "_" and np.isscalar(v)]
    # add back the runtime and timestamp and this is useful to users
    columns += ["_runtime", "_timestamp"]
    df = pd.DataFrame(index=range(max_steps), columns=columns)
    df.index.name = "step"
    for row_idx, row in tqdm(
        enumerate(run.scan_history(page_size=10000, keys=keys)),
        total=max_steps,
        desc=f"Downloading run: {run.name}",
    ):
        df.update(pd.DataFrame([row], index=[row_idx]))
    config = flatten_config(run.config)
    return config, df


def flatten_config(config):
    """Flatten nested config dictionaries into a single level."""
    for name in ["log", "data", "model", "optim", "hardware"]:
        for k, v in config[name].items():
            config[f"{name}.{k}"] = v
        del config[name]
    return config


def tabulate_runs(configs, runs, value, ignore=["hardware.port"]):
    """Create a pivot table from configs and runs for a specific value."""
    res = configs
    for col in configs.columns:
        if len(configs[col].unique()) == 1 or col in ignore:
            res = res.drop(col, axis=1)
    variables = res.columns
    print("Remaining columns:", variables)
    res["_index"] = res.index
    rows = input("Which to use as rows?").split(",")
    table = pd.pivot_table(
        res,
        index=rows,
        columns=[v for v in variables if v not in rows],
        values="_index",
    )

    def fn(i):
        try:
            i = int(i)
            return runs[i][value][-1]
        except ValueError:
            print(i)

    table = table.map(fn)
    return table
