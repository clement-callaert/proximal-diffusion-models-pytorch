"""Hydra entry point for the ProxDM pipeline. Implement orchestration here."""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    raise NotImplementedError("Wire pipeline phases to src/ modules.")


if __name__ == "__main__":
    main()
