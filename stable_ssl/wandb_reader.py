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
from typing import Any, Dict, Optional

import numpy as np
import omegaconf
import pandas as pd
from tqdm import tqdm


def alphanum_key(key):
    return [int(c) if c.isdigit() else c.lower() for c in re.split("([0-9]+)", key)]


def natural_sort(values):
    return sorted(values, key=alphanum_key)


def offline_config(path):
    """Load config and values from a single run directory."""
    _path = Path(path)
    if not _path.is_dir():
        raise ValueError(f"The provided path ({path}) is not a directory!")
    # Load the config file.
    config = omegaconf.OmegaConf.load(_path / ".hydra" / "config.yaml")
    return config


def project_to_table(
    dfs: dict[str : pd.DataFrame],
    configs: dict[str : pd.DataFrame],
    value: str,
    row: str,
    column: str,
    agg: callable,
    filters=None,
) -> pd.DataFrame:
    """Format a pandas DataFrame as a table given the user args.

    Args:
        dfs (dict): first output of project
        configs (dict): second output of project
        value (str): name of the column in dfs to use as values
        row (str): name of the column in configs to use as row
        column (str): name of the column in configs to use as column
        agg (callable): aggregator if many values are present
        filters: filters to apply to the data

    Returns:
        DataFrame: the formatted table
    """
    logging.info(f"Creating table from {len(configs)} runs.")
    filters = filters or {}
    df = pd.DataFrame(configs).T
    for id in df.index.values:
        assert id in dfs
    for k, v in filters.items():
        if type(v) not in [tuple, list]:
            v = [v]
        s = df[k].isin(v)
        df = df.loc[s]
        logging.info(f"After filtering {k}, {len(df)} runs are left.")

    rows = natural_sort(df[row].astype(str).unique())
    logging.info(f"Found rows: {rows}")
    columns = natural_sort(df[column].astype(str).unique())
    logging.info(f"Found columns: {columns}")
    output = pd.DataFrame(columns=columns, index=rows)
    for r in rows:
        for c in columns:
            cell_runs = (df[row].astype(str) == r) & (df[column].astype(str) == c)
            n = np.count_nonzero(cell_runs)
            samples = []
            logging.info(f"Number of runs for cell ({r}, {c}): {n}")
            for id in df[cell_runs].index.values:
                if value not in dfs[id].columns:
                    logging.info(f"Run {id} missing {value}, skipping....")
                    continue
                samples.append(dfs[id][value].values.reshape(-1))
            if len(samples) == 0:
                output.loc[r, c] = np.nan
            else:
                output.loc[r, c] = agg(np.concatenate(samples))
    return output


def project(
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
    data = []
    for run in tqdm(runs):
        run_data = dict()
        run_data.update(run.summary._json_dict)
        run_data.update(run.config)
        run_data.update({"tags": run.tags})
        run_data.update({"name": run.name})
        run_data.update({"created_at": run.created_at})
        run_data.update({"id": run.id})
        data.append(run_data)

    runs_df = pd.DataFrame.from_records(data)
    return runs_df
    with Pool(num_workers) as p:
        results = list(
            tqdm(
                p.imap(
                    _run_packed,
                    [(entity, project, r.id, min_step, max_step, keys) for r in runs],
                ),
                total=len(runs),
                desc=f"Downloading project: {project}",
            )
        )
    dfs = {}
    for r, df in zip(runs, results):
        dfs[f"{entity}/{project}/{r.id}"] = df
    return dfs


def _run_packed(args):
    return run(*args, _tqdm_disable=True)


def run(
    entity, project, run_id, min_step=0, max_step=-1, keys=None, _tqdm_disable=False
):
    """Download data for a single wandb run."""
    api = wandbapi.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    if max_step == -1:
        max_step = run.lastHistoryStep + 1
    if min_step < 0:
        min_step = max_step + min_step

    # if keys is None:
    #     summary = run.summary
    #     # extract names that are not hidden
    #     keys = [k for k, v in summary.items() if k[0] != "_" and np.isscalar(v)]
    #     # add back the runtime and timestamp and this is useful to users
    #     keys += ["_runtime", "_timestamp", "_step"]
    # else:
    if keys is not None and "_step" not in keys:
        keys.append("_step")

    data = []
    for row in tqdm(
        run.scan_history(keys=keys, min_step=min_step, max_step=max_step),
        total=max_step,
        desc=f"Downloading run: {run.name}",
        disable=_tqdm_disable,
    ):
        data.append(row)
    df = pd.DataFrame(data)
    for key in run.config:
        df[key] = run.config[key]
    names = list(run.config.keys())
    assert "_step" in df.columns
    df.set_index(names + ["_step"], inplace=True)
    return df


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
