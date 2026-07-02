"""
project.py
----------
Repo root discovery and config loading for fus-predict.
"""

from pathlib import Path

import yaml


def find_repo_root(start: str | Path | None = None) -> Path:
    """
    Walk up from start (default: cwd) until a directory containing
    both a config/ folder and a pyproject.toml is found.
    """
    start_path = Path.cwd().resolve() if start is None else Path(start).resolve()
    return next(
        (
            path
            for path in [start_path, *start_path.parents]
            if (path / "config").is_dir() and (path / "pyproject.toml").is_file()
        ),
        start_path,
    )


def load_project_config(
    repo_root: str | Path | None = None,
    config_name: str = "config.yml",
) -> dict:
    """Load a config YAML file relative to repo_root (or auto-detected root).

    Parameters
    ----------
    repo_root : str or Path or None
        Repo root directory. Auto-detected from cwd if None.
    config_name : str
        Filename inside ``config/`` to load. Defaults to ``"config.yml"``.
        Pass ``"config_mouse.yml"`` to load the mouse experiment config.
    """
    root = find_repo_root() if repo_root is None else Path(repo_root).resolve()
    config_path = root / "config" / config_name
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}