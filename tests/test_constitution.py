import pytest
from canon.constitution import constitution_from_file, constitution_from_dict
from canon.errors import ConfigError


def test_from_yaml(tmp_path):
    p = tmp_path / "constitution.yaml"
    p.write_text("mission: Serve borrowers well\nprinciples:\n  - Be fair\n  - Be transparent\nversion: v1\n")
    c = constitution_from_file(p)
    assert c.mission == "Serve borrowers well"
    assert c.principles == ("Be fair", "Be transparent")
    assert c.version == "v1"


def test_missing_mission_raises():
    with pytest.raises(ConfigError):
        constitution_from_dict({"principles": ["x"]})


def test_from_json_file():
    """A .json constitution file loads through the YAML parser (a JSON
    superset); the json.loads fallback is unreached for well-formed input."""
    import json
    import tempfile as _tempfile
    from pathlib import Path as _Path
    d = _Path(_tempfile.mkdtemp())
    p = d / "constitution.json"
    p.write_text(json.dumps({"mission": "Serve well", "principles": ["Be fair"], "version": 2}))
    c = constitution_from_file(p)
    assert c.mission == "Serve well"
    assert c.principles == ("Be fair",)
    assert c.version == "2"


def test_empty_principles_list_raises():
    with pytest.raises(ConfigError, match="principles"):
        constitution_from_dict({"mission": "m", "principles": []})


def test_non_mapping_document_raises(tmp_path):
    p = tmp_path / "constitution.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ConfigError, match="must be a mapping"):
        constitution_from_file(p)


def test_integer_version_is_coerced_to_string():
    c = constitution_from_dict({"mission": "m", "principles": ["p"], "version": 3})
    assert c.version == "3" and isinstance(c.version, str)
