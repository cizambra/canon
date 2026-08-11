import pytest
import yaml

from canon.errors import ConfigError
from canon.runner import run_suite


class _NullMetric:
    def score(self, artifact, task=""):
        raise AssertionError("should not be reached")


def test_missing_artifacts_key_raises_config_error(tmp_path):
    p = tmp_path / "suite.yaml"
    p.write_text("task: decide\n")
    with pytest.raises(ConfigError, match="suite has no artifacts"):
        run_suite(p, _NullMetric())


def test_empty_artifacts_list_raises_config_error(tmp_path):
    p = tmp_path / "suite.yaml"
    p.write_text(yaml.safe_dump({"task": "decide", "artifacts": []}))
    with pytest.raises(ConfigError, match="suite has no artifacts"):
        run_suite(p, _NullMetric())


def test_artifacts_must_be_a_list_of_strings(tmp_path):
    """A bare string would otherwise be iterated character by character."""
    p = tmp_path / "suite.yaml"
    p.write_text("task: decide\nartifacts: one big string\n")
    with pytest.raises(ConfigError, match="list of strings"):
        run_suite(p, _NullMetric())
