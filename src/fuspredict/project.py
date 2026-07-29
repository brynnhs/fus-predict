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


def get_excluded_sessions(
    config: dict,
    subject: str,
    override: list[str] | None = None,
) -> list[str]:
    """Resolve the excluded-session-id list for a subject.

    Looks up ``config["subjects"]["sessions_to_exclude"][subject]``.
    If ``override`` is given (e.g. from a CLI flag), it takes precedence
    over the config value entirely.
    """
    default_exclude = config["subjects"].get("sessions_to_exclude", {}).get(subject, [])
    return override if override is not None else default_exclude


def resolve_standardized_and_mask_dirs(
    repo_root: str | Path,
    project_cfg: dict,
    subject: str,
    standardized_dir: str | Path | None = None,
    mask_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve the standardized-sessions and tissue-mask directories for a subject.

    Falls back to the standard preprocessing layout under
    ``project_cfg["paths"]["preprocessing"]/<subject>/`` when
    ``standardized_dir``/``mask_dir`` overrides are not given.
    """
    preproc_root = Path(repo_root) / project_cfg["paths"]["preprocessing"] / subject
    resolved_standardized_dir = (
        Path(standardized_dir) if standardized_dir else preproc_root / "baseline_only_standardized"
    )
    resolved_mask_dir = Path(mask_dir) if mask_dir else preproc_root / "tissue_masks"
    return resolved_standardized_dir, resolved_mask_dir