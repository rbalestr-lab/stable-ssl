import sys
from unittest.mock import MagicMock, patch

from stable_ssl.cli import entry


def test_cli_with_config_file(config_file_path):
    """Ensure Hydra sees the config, and a trainer is instantiated."""
    test_args = [
        "stable-ssl",
        "--config-path",
        config_file_path,
        "--config-name",
        "tiny_mnist",
    ]

    # Patch sys.argv so the CLI sees our arguments
    with (
        patch.object(sys, "argv", test_args),
        patch("stable_ssl.cli.hydra.utils.instantiate") as mock_instantiate,
    ):
        # Mock the returned trainer
        mock_trainer = MagicMock(name="MockTrainerObject")
        mock_instantiate.return_value = mock_trainer

        # Call the CLI entry point
        entry()

        # Check that hydra.utils.instantiate was indeed called
        mock_instantiate.assert_called()
        # Check that the trainer got called
        mock_trainer.assert_called_once()
