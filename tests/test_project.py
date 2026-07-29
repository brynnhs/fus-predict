from fuspredict.project import get_excluded_sessions


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
