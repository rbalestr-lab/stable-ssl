import tempfile
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from stable_ssl.reader import jsonl


@hydra.main(version_base="1.2", config_path="configs", config_name="simclr_mnist")
def local_test(cfg):
    # Create a temporary directory that will be cleaned up after the script ends
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Simulate Hydra’s usual job folder structure
        hydra_dir = tmp_path / ".hydra"
        hydra_dir.mkdir(parents=True, exist_ok=True)
        config_file = hydra_dir / "config.yaml"
        OmegaConf.save(config=cfg, f=config_file)

        # Convert to a dict to add any missing keys
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict["trainer"]["logger"]["dump_path"] = str(tmp_path)
        updated_cfg = OmegaConf.create(cfg_dict)

        # Instantiate and run trainer
        trainer = hydra.utils.instantiate(
            updated_cfg.trainer, _convert_="object", _recursive_=False
        )
        trainer.setup()
        trainer.launch()

        # Read logs and check final metric
        logs = jsonl(path=tmp_path)
        final_acc = logs[-1]["test/acc1"]
        print(f"Final test/acc1: {final_acc}")

        # Example assertion (or handle however you like)
        assert final_acc == 0.8585619926452637, "Unexpected final test accuracy!"


if __name__ == "__main__":
    local_test()
