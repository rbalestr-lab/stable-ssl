"""Reader for logs."""

#
# Author: Randall Balestriero <randallbalestriero@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

try:
    import wandb as wandbapi
except ModuleNotFoundError:
    logging.warning(
        "Wandb module is not installed, make sure to not use wandb for logging "
        "or an error will be thrown."
    )
import logging
import re
from multiprocessing import Pool
from pathlib import Path

import jsonlines
import numpy as np
import omegaconf
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import Optional, Dict, Any


def alphanum_key(key):
    return [int(c) if c.isdigit() else c.lower() for c in re.split("([0-9]+)", key)]


def natural_sort(values):
    return sorted(values, key=alphanum_key)


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
            results = list(tqdm(p.imap(jsonl, args), total=len(runs)))
        for c, v in results:
            configs.append(flatten_config(c))
            values.append(v)
    config = pd.DataFrame(configs)
    return config, values


def jsonl(path):
    """Load config and values from a single run directory."""
    _path = Path(path)
    if not _path.is_dir():
        raise ValueError(f"The provided path ({path}) is not a directory!")

    values = []
    # Load the values from logs_rank_* files.
    logs_files = list(_path.glob("logs_rank_*.jsonl"))
    logging.info(f"Reading .jsonl files from {_path}")
    logging.info(f"\t=> {len(logs_files)} ranks detected")
    for log_file in logs_files:
        # Extract rank from the filename.
        rank = int(log_file.stem.split("rank_")[1])
        for obj in jsonlines.open(log_file).iter(type=dict, skip_invalid=True):
            obj["rank"] = rank  # Add rank field to each dict.
            values.append(obj)
    logging.info(f"\t=> total length of logs: {len(values)}")
    return values


def config(path):
    """Load config and values from a single run directory."""
    _path = Path(path)
    if not _path.is_dir():
        raise ValueError(f"The provided path ({path}) is not a directory!")
    # Load the config file.
    config = omegaconf.OmegaConf.load(_path / ".hydra" / "config.yaml")
    return config


def wandb_project_to_table(
    dfs: dict[str : pd.DataFrame],
    configs: dict[str : pd.DataFrame],
    value: str,
    row: str,
    column: str,
    agg: callable,
) -> pd.DataFrame:
    """Format a pandas DataFrame as a table given the user args.

    Args:
        dfs (dict): first output of wandb_project
        configs (dict): second output of wandb_project
        value (str): name of the column in dfs to use as values
        row (str): name of the column in configs to use as row
        column (str): name of the column in configs to use as column
        agg (callable): aggregator if many values are present

    Returns
    -------
        DataFrame: the formatted table
    """
    df = pd.DataFrame(configs).T
    rows = natural_sort(df[row].astype(str).unique())
    columns = natural_sort(df[column].astype(str).unique())
    output = pd.DataFrame(columns=columns, index=rows)
    for r in rows:
        for c in columns:
            ids = df[
                (df[row].astype(str) == r) & (df[column].astype(str) == c)
            ].index.values
            samples = []
            for id in ids:
                samples.append(dfs[id][value])
            output.loc[r, c] = agg(np.stack(samples))
    return output


def wandb_project(
    entity: str,
    project: str,
    filters: Optional[Dict[str, Any]] = None,
    order: str = "+created_at",
    per_page: int = 50,
    include_sweeps: bool = True,
    min_step: int = 0,
    max_step: int = -1,
    keys: list = None,
    num_workers: int = 10,
):
    """Download configs and data from a wandb project."""
    api = wandbapi.Api()
    runs = api.runs(
        f"{entity}/{project}",
        filters=filters,
        order=order,
        per_page=per_page,
        include_sweeps=include_sweeps,
    )
    logging.info(f"Found {len(runs)} runs for project {project}")
    with Pool(num_workers) as p:
        results = list(
            tqdm(
                p.imap(
                    _wandb_run_packed,
                    [(entity, project, r.id, min_step, max_step, keys) for r in runs],
                ),
                total=len(runs),
                desc=f"Downloading project: {project}",
            )
        )
    dfs = {}
    configs = {}
    for r, (result, config) in zip(runs, results):
        dfs[f"{entity}/{project}/{r.id}"] = result
        configs[f"{entity}/{project}/{r.id}"] = config
    return dfs, configs


def _wandb_run_packed(args):
    return wandb(*args, _tqdm_disable=True)


def wandb(
    entity, project, run_id, min_step=0, max_step=-1, keys=None, _tqdm_disable=False
):
    """Download data for a single wandb run."""
    api = wandbapi.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    if max_step == -1:
        max_step = run.lastHistoryStep + 1
    if min_step < 0:
        min_step = max_step + min_step

    if keys is None:
        summary = run.summary
        # extract names that are not hidden
        keys = [k for k, v in summary.items() if k[0] != "_" and np.isscalar(v)]
        # add back the runtime and timestamp and this is useful to users
        keys += ["_runtime", "_timestamp", "_step"]
    else:
        if "_step" not in keys:
            keys.append("_step")

    data = []
    for row_idx, row in tqdm(
        enumerate(
            run.scan_history(
                page_size=10000, keys=keys, min_step=min_step, max_step=max_step
            )
        ),
        total=max_step,
        desc=f"Downloading run: {run.name}",
        disable=_tqdm_disable,
    ):
        data.append(row)
    df = pd.DataFrame(data)
    df.set_index("_step", inplace=True)
    # config = flatten_config(run.config)
    return df, run.config


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
