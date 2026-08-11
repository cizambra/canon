from importlib import resources


def test_default_rubric_is_packaged_data():
    # Resolves via the installed package, not a hard-coded src/ path.
    p = resources.files("canon.data").joinpath("default_rubric.yaml")
    assert p.is_file()
    assert "version" in p.read_text()
