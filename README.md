# stable-ssl

[![Documentation](https://img.shields.io/badge/Documentation-blue.svg)](https://rbalestr-lab.github.io/stable-ssl.github.io/dev/)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-blue.svg)](https://github.com/rbalestr-lab/stable-ssl/tree/main/benchmarks)
[![Test Status](https://github.com/rbalestr-lab/stable-ssl/actions/workflows/testing.yml/badge.svg)](https://github.com/rbalestr-lab/stable-ssl/actions/workflows/testing.yml)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/rbalestr-lab/stable-ssl/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/rbalestr-lab/stable-ssl/tree/main)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![WandB](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/site)

## Doing One Thing, But Doing It Right... Ours is Self Supervised Learning

Self Supervised Learning (SSL) is the last frontier of AI. But quick experimentation is not possible today as no library offers stable and modular key-in-hand solutions. Existing libraries are either static (lightly-ai, solo-learn) or not concerned with SSL--justifying our existence. *Our goal is to provide a flexible, full-fledge, optimized framework to conduct rapid SSL research and scale as needed.*

## How?

To reach flexibility, scalability and stability, we rely on battle-tested third party libraries: `PyTorch`, `Lightning`, `HuggingFace`, `TorchMetrics` amongst a few others. Those dependencies allow us to focus on one thing: assembling everything into a powerful SSL research framework. ``stable-ssl`` adopts a flexible and modular design for seamless integration of components from external libraries, including architectures, loss functions, evaluation metrics, and augmentations.

## Log log log, monitor monitor monitor!

The key to SSL research is to log and monitor everything. This is what we bring to a new level with `stable-ssl` by providing extremely rich logging and numerous callbacks that can be added and combined in any way you like within your trainer such as `stable_ssl.callbacks.OnlineProbe`, `stable_ssl.callbacks.OnlineKNN`, `stable_ssl.callbacks.RankMe`, `stable_ssl.callbacks.LiDAR`, `stable_ssl.callbacks.OnlineWriter`, and so on.

## Core Structure

`stable-ssl` only requires you to get familiar with 3 components:

1. **data**: the dataset should be a huggingface dataset e.g.
    ```
    import stable_ssl as ssl
    train_dataset = ssl.data.HFDataset(
        path="frgfm/imagenette",
        name="160px",
        split="train",
        transform=train_transform,
    )
    ```
    if it already exists on the Hub, otherwise you can wrap your own dataset into a HF dataset. **Why?** Imposing that format ensures consistent behavior (each sample is a dictionary) and leverage powerful utilities from the `datasets` package. Once datasets (train et al.) are created, they can be used as-is with `torch.utils.data.DataLoader`. However we recommend putting them into our `DataModule` e.g.
    ```
    datamodule = ssl.data.DataModule(train=train_dataset, val=val_dataset, ...)
    ```
    to ensure precise logging and easy debugging.
2. **module, models, forward**: the overall orchestration leverages `ssl.Module` which inherits from `lightning.LightningModule`. We provide all the basic required utilities (optimizer/scheduler creation etc). So the only required implementation for the user is the `forward` method, for example a supervised learning run would define
    ```
    def forward(self, batch, stage):
        batch["embedding"] = self.backbone(batch["image"])["logits"]
        if self.training:
            preds = self.classifier(batch["embedding"])
            batch["loss"] = torch.nn.functional.cross_entropy(preds, batch["label"])
        return batch
    ```
    the `forward` method takes in a dictionary (`batch` from the data loader) and should return a dictionary. If any module has to be trained, then a `loss` key must be present. Further customization can be done (see the `examples`) ensuring that any desired behavior can be achieved. The `self` is a LightningModule with any attribute passed during module creation:
    ```
    config = AutoConfig.from_pretrained("microsoft/resnet-18")
    backbone = ViT(512)
    projector = torch.nn.Linear(512, 128)
    module = ssl.Module(
        backbone=backbone,
        projector=projector,
        forward=forward,
        simclr_loss=ssl.losses.NTXEntLoss(temperature=0.1),
    )
    ```
    any `kwarg` passed to `stable_ssl.Module` is automatically set, the only reserved `kwarg` is `forward`
3. **trainer**: the final step is to describe how training will happen! This is done with the `lightning.Trainer` module, for example
    ```
    trainer = pl.Trainer(
            max_epochs=10,
            num_sanity_val_steps=1,
            callbacks=[linear_probe, knn_probe, rankme],
            precision="16-mixed",
            logger=False,
            enable_checkpointing=False,
        )
    manager = ssl.Manager(trainer=trainer, module=module, data=data)
    manager()
    ```
    once this is specified, simply pipe everything into our manager class that will connect everything and launch fitting! This extra wrapper is needed to produce as precise logging as possible.

<details>
  <summary>Minimal Example : SimCLR INET10</summary>
    ```
    import optimalssl as ssl
    import torch
    from transformers import AutoModelForImageClassification, AutoConfig
    import lightning as pl
    from optimalssl.data import transforms
    import torchmetrics

    # without transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=(5, 5), p=1.0),
        transforms.ToImage(mean=mean, std=std),
    )
    train_dataset = ssl.data.HFDataset(
        path="frgfm/imagenette",
        name="160px",
        split="train",
        transform=train_transform,
    )
    train = torch.utils.data.DataLoader(
        dataset=train_dataset,
        sampler=ssl.data.sampler.RepeatedRandomSampler(train_dataset, n_views=2),
        batch_size=64,
        num_workers=20,
        drop_last=True,
    )
    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToImage(mean=mean, std=std),
    )
    val = torch.utils.data.DataLoader(
        dataset=ssl.data.HFDataset(
            path="frgfm/imagenette",
            name="160px",
            split="validation",
            transform=val_transform,
        ),
        batch_size=128,
        num_workers=10,
    )
    data = ssl.data.DataModule(train=train, val=val)

    def forward(self, batch, stage):
        batch["embedding"] = self.backbone(batch["image"])["logits"]
        if self.training:
            proj = self.projector(batch["embedding"])
            views = ssl.data.fold_views(proj, batch["sample_idx"])
            batch["loss"] = self.simclr_loss(views[0], views[1])
        return batch

    config = AutoConfig.from_pretrained("microsoft/resnet-18")
    backbone = AutoModelForImageClassification.from_config(config)
    projector = torch.nn.Linear(512, 128)
    backbone.classifier[1] = torch.nn.Identity()
    module = ssl.Module(
        backbone=backbone,
        projector=projector,
        forward=forward,
        simclr_loss=ssl.losses.NTXEntLoss(temperature=0.1),
    )
    linear_probe = ssl.callbacks.OnlineProbe(
        "linear_probe",
        module,
        "embedding",
        "label",
        probe=torch.nn.Linear(512, 10),
        loss_fn=torch.nn.CrossEntropyLoss(),
        metrics={
            "top1": torchmetrics.classification.MulticlassAccuracy(10),
            "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
        },
    )
    knn_probe = ssl.callbacks.OnlineKNN(
        module,
        "knn_probe",
        "embedding",
        "label",
        20000,
        metrics=torchmetrics.classification.MulticlassAccuracy(10),
        k=10,
        features_dim=512,
    )

    trainer = pl.Trainer(
        max_epochs=6,
        num_sanity_val_steps=1,
        callbacks=[linear_probe, knn_probe],
        precision="16-mixed",
        logger=False,
        enable_checkpointing=False,
    )
    manager = ssl.Manager(trainer=trainer, module=module, data=data)
    manager()
    ```
</details>


## Installation

The library is not yet available on PyPI. You can install it from the source code, as follows.

1. <details><summary>conda (optional)</summary>

    First use your favorite environment manager and install your favorite pytorch version, we provide an example with conda
    ```
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    ```
    follow installation instructions... once completed, create your environment
    ```
    conda create -n my_env python=3.11
    ```
    with your environment name (here `my_env`) and your favorite Python version (here, `3.11`). Once completed, make sure to activate your environment (`conda activate my_env`) before proceeding to the next steps!
  </details>

2. Pytorch and our library (we recommend using `uv` for quicker package management):
    ```
    pip3 install uv
    uv pip install torch torchvision torchaudio
    uv pip install -e .
    ```
    if you do not want to use uv, simply remove it from the above commands.

3. API login (optional)
    ```
    wandb login
    huggingface-cli login
    ```
4. LATEX support in Matplotlib (optional)

    1.  <details>
        <summary>Install the LaTex font (Computer Modern)</summary>

        - we provide the ttf files [in the repo](assets/cm-unicode-0.7.0%202/) to make things simple
        - create your local folder (if not present) and copy the ttf files there
          - `mkdir -p ~/.local/share/fonts `
          - `cp assets/cm-unicode-0.7.0\ 2/*ttf ~/.local/share/fonts/`
        - refresh the font cache with `fc-cache -f -v`
        - validate that the fonts are listed in your system with `fc-list | grep cmu`
        - refresh matplotlib cache
          ```
          import shutil
          import matplotlib

          shutil.rmtree(matplotlib.get_cachedir())
          ```
        </details>


    2. <details>
        <summary>Install the Tex compiler (optional, if not available on your system)</summary>

        - install texlive locally following https://tug.org/texlive/quickinstall.html#running where you can use `-texdir your_path` to install to a local path (so you don't need sudo privileges)
        - follow the instructions at the end of the installation to edit the PATH variables, you can edit that variable for a conda environment with `conda env config vars set PATH=$PATH`
        - make sure inside the conde environment that you point to the right binaries e.g. `whereis latex` and `whereis mktexfmt`
        - If at some point there is an error that the file `latex.fmt` is not found. You can generate it with
          - `pdftex -ini   -jobname=latex -progname=latex -translate-file=cp227.tcx *latex.ini`
          - or (unsure) `fmtutil-sys --all`
        </details>

    3. <details>
        <summary>rc config (optional)</summary>

        ```
        font.family: serif
        font.serif: cmr10
        font.sans-serif: cmss10
        font.monospace: cmtt10

        text.usetex: True
        text.latex.preamble: \usepackage{amssymb} \usepackage{amsmath} \usepackage{bm}

        xtick.labelsize: 14
        ytick.labelsize: 14
        legend.fontsize: 14
        axes.labelsize: 16
        axes.titlesize: 16
        axes.formatter.use_mathtext: True
        ```
        which can be written to a file, e.g., `~/.config/matplotlib/matplotlibrc` or set via `rc` in your script directly. See here for more details.
        </details>

    4. <details>
        <summary>Example of matplotlib script to run for a quick test (optional)</summary>

        ```
        from matplotlib import rc
        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        rc('text', usetex=True)
        import numpy as np
        import matplotlib.pyplot as plt


        t = np.arange(0.0, 1.0 + 0.01, 0.01)
        s = np.cos(4 * np.pi * t) + 2

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.plot(t, s)

        plt.xlabel(r'\textbf{time} (s)')
        plt.ylabel(r'\textit{voltage} (mV)',fontsize=16)
        plt.title(r"\TeX\ is Number "
                  r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!",
                  fontsize=16, color='gray')
        # Make room for the ridiculously large title.
        plt.subplots_adjust(top=0.8)

        plt.savefig('tex_demo')
        plt.show()
        ```
      </details>

## Ways You Can Contribute:

- If you'd like to contribute new features, bug fixes, or improvements to the documentation, please refer to our [contributing guide](https://rbalestr-lab.github.io/stable-ssl.github.io/dev/contributing.html) for detailed instructions on how to get started.

- You can also contribute by adding new methods, datasets, or configurations that improve the current performance of a method in the [benchmark section](https://github.com/rbalestr-lab/stable-ssl/tree/main/benchmarks).

## Contributors

`stable-ssl` was started by `Randall Balestriero` circa 2020 for internal research projects. After numerous refactorings and simplifications, it became practical for external use circa 2024 at which point `Hugues Van Assel` and `Lucas Maes` joined as core contributors.
