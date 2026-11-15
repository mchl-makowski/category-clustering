"""README documents every env var and every YAML section/key the code accepts."""

from pathlib import Path

from pydantic import BaseModel

from problem_clustering import Config
from problem_clustering.backend.azure import FoundrySettings

README = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")


def test_readme_lists_every_foundry_env_var():
    for field in FoundrySettings.model_fields:
        assert f"FOUNDRY_{field.upper()}" in README


def test_readme_lists_every_yaml_section_and_key():
    for section, info in Config.model_fields.items():
        assert f"`{section}`" in README, section
        if isinstance(info.default, BaseModel):
            for key in type(info.default).model_fields:
                assert f"`{key}`" in README, f"{section}.{key}"
