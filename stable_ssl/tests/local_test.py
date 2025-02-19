from hydra import compose, initialize
from omegaconf import OmegaConf


def test_base_trainer(tmp_path):
    with initialize(version_base="1.2", config_path="configs"):
        cfg = compose(config_name="tiny_mnist")

        # Save out the config to simulate Hydra’s normal behavior
        hydra_dir = tmp_path / ".hydra"
        hydra_dir.mkdir(parents=True, exist_ok=True)
        config_file = hydra_dir / "config.yaml"
        OmegaConf.save(config=cfg, f=config_file)

        # Convert to a dict to add dump_path
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict["trainer"]["logger"]["dump_path"] = tmp_path
        cfg = OmegaConf.create(cfg_dict)

        # trainer = hydra.utils.instantiate(
        #     cfg.trainer, _convert_="object", _recursive_=False
        # )
        # trainer.setup()
        # trainer.launch()

        # logs = jsonl(path=tmp_path)
        # assert logs[-1]["test/acc1"] == 0.10000000149011612
