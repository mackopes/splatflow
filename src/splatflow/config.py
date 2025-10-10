import tomllib
from pathlib import Path
from pydantic import BaseModel, field_validator


class Config(BaseModel):
    splatflow_data_root: Path

    @field_validator("splatflow_data_root", mode="before")
    @classmethod
    def expand_path(cls, v):
        """Expand user path (e.g., ~) in splatflow_data_root."""
        return Path(v).expanduser()


def load_config() -> Config:
    """Load configuration from config.toml."""
    config_path = Path(__file__).parent.parent.parent / "config.toml"

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    return Config.model_validate(data)
