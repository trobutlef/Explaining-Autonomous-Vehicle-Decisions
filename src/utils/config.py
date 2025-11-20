from omegaconf import OmegaConf, DictConfig
from pathlib import Path

def load_config(path: str):
    # load YAML into a config object/dict
    cfg_path = Path(path)

    # If the given path is not found relative to CWD, try relative to project root
    if not cfg_path.exists():
        project_root = Path(__file__).resolve().parents[2]
        alt_path = project_root / path
        if alt_path.exists():
            cfg_path = alt_path
        else:
            raise FileNotFoundError(f"Config file not found: {cfg_path} or {alt_path}")

    cfg: DictConfig = OmegaConf.load(str(cfg_path))
    return cfg  # DictConfig
