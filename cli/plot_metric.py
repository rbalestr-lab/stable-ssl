import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import stable_ssl as ssl


def parse_rules(v):
    rules = []
    for rule in v.split("&"):
        rules.append(rule.split("="))
    return rules


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--savefig", type=Path, default=None)
    parser.add_argument("--filters", type=parse_rules, default=[])
    parser.add_argument("--hparams", type=lambda x: x.split(","), default=None)
    parser.add_argument("--legend", action="store_true")
    parser.add_argument("--minimize", action="store_true")
    parser.add_argument("--metric", type=str, required=True)
    args = parser.parse_args()

    configs, values = ssl.reader.jsonl_project(args.path)
    if args.hparams is None:
        args.hparams = []
        for name, series in configs.items():
            if len(np.unique(series.astype(str))) > 1 and ("port" not in name):
                args.hparams.append(name)
        logging.info("No hparams was given...")
        logging.info(f"We automatically detected the following ones: {args.hparams}")

    results = []
    for (index, conf), ts in zip(configs.iterrows(), values):
        for rule in args.filters:
            if conf[rule[0]] != rule[1]:
                continue
        ts = [v[args.metric] for v in ts if args.metric in v]
        p = {name: conf[name] for name in args.hparams}
        label = [f"{k}: {v}" for k, v in p.items()]
        plt.plot(ts, label=", ".join(p))
        if args.minimize:
            p["_value"] = np.min(ts)
        else:
            p["_value"] = np.max(ts)
        results.append(p)
    results = pd.DataFrame(results).sort_values("_value")
    print(results)

    if args.legend:
        plt.legend()
    plt.tight_layout()
    if args.savefig is not None:
        plt.savefig(args.savefig)
        plt.close()
    else:
        plt.show()
