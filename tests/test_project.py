from pathlib import Path

from fuspredict.project import get_excluded_sessions, resolve_standardized_and_mask_dirs


def _config(exclude=None):
    return {"subjects": {"sessions_to_exclude": exclude or {}}}


def test_get_excluded_sessions_returns_configured_list():
    config = _config({"subj1": ["sess_a", "sess_b"]})
    assert get_excluded_sessions(config, "subj1") == ["sess_a", "sess_b"]


def test_get_excluded_sessions_defaults_to_empty_list():
    config = _config()
    assert get_excluded_sessions(config, "subj1") == []


def test_get_excluded_sessions_unrelated_subject_not_affected():
    config = _config({"subj1": ["sess_a"]})
    assert get_excluded_sessions(config, "subj2") == []


def test_get_excluded_sessions_override_takes_precedence():
    config = _config({"subj1": ["sess_a"]})
    assert get_excluded_sessions(config, "subj1", override=["sess_z"]) == ["sess_z"]


def test_get_excluded_sessions_override_empty_list_is_respected():
    config = _config({"subj1": ["sess_a"]})
    assert get_excluded_sessions(config, "subj1", override=[]) == []


def test_get_excluded_sessions_override_none_falls_back_to_config():
    config = _config({"subj1": ["sess_a"]})
    assert get_excluded_sessions(config, "subj1", override=None) == ["sess_a"]


def _project_cfg():
    return {"paths": {"preprocessing": "derivatives/preprocessing"}}


def test_resolve_dirs_defaults_to_standard_layout():
    standardized_dir, mask_dir = resolve_standardized_and_mask_dirs(
        Path("/repo"), _project_cfg(), "subj1"
    )
    assert standardized_dir == Path("/repo/derivatives/preprocessing/subj1/baseline_only_standardized")
    assert mask_dir == Path("/repo/derivatives/preprocessing/subj1/tissue_masks")


def test_resolve_dirs_honors_standardized_dir_override():
    standardized_dir, mask_dir = resolve_standardized_and_mask_dirs(
        Path("/repo"), _project_cfg(), "subj1", standardized_dir="/custom/std"
    )
    assert standardized_dir == Path("/custom/std")
    assert mask_dir == Path("/repo/derivatives/preprocessing/subj1/tissue_masks")


def test_resolve_dirs_honors_mask_dir_override():
    standardized_dir, mask_dir = resolve_standardized_and_mask_dirs(
        Path("/repo"), _project_cfg(), "subj1", mask_dir="/custom/masks"
    )
    assert standardized_dir == Path("/repo/derivatives/preprocessing/subj1/baseline_only_standardized")
    assert mask_dir == Path("/custom/masks")
