import tomllib
from pathlib import Path
from pydantic import BaseModel


class Config(BaseModel):
    splatflow_data_root: str


def load_config() -> Config:
    """Load configuration from config.toml."""
    config_path = Path(__file__).parent.parent.parent / "config.toml"

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    return Config.model_validate(data)
