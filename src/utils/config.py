from omegaconf import OmegaConf, DictConfig
from pathlib import Path

def load_config(path: str):
    # load YAML into a config object/dict
    #cfg_path = Path(path)
    cfg = load_config(str(Path(PROJECT_ROOT) / 'configs/config.yaml'))
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    cfg = OmegaConf.load(str(cfg_path))
    return cfg  # DictConfig
