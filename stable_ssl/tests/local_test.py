import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

from stable_ssl.reader import jsonl

OmegaConf.register_new_resolver("eval", eval)


def test_base_trainer(tmp_path):
    with initialize(version_base="1.2", config_path="configs"):
        cfg = compose(config_name="simclr_mnist")

        # Write out the config.yaml to simulate Hydra’s usual job folder
        hydra_dir = tmp_path / ".hydra"
        hydra_dir.mkdir(parents=True, exist_ok=True)
        config_file = hydra_dir / "config.yaml"
        OmegaConf.save(config=cfg, f=config_file)

        # Convert to a dictionary to easily add missing keys
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict["trainer"]["logger"]["dump_path"] = tmp_path
        cfg = OmegaConf.create(cfg_dict)

        trainer = hydra.utils.instantiate(
            cfg.trainer, _convert_="object", _recursive_=False
        )
        trainer.setup()
        trainer.launch()

        logs = jsonl(path=tmp_path)
        assert logs[-1]["test/acc1"] == 0.8585619926452637
