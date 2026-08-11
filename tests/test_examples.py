import os
import subprocess
import sys


def test_example_pytest_test_passes_with_mock(tmp_path):
    env = {**os.environ, "CANON_MOCK_JUDGE": "yes"}
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "examples/tests/coherence/test_lending.py", "-q"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0, r.stdout + r.stderr
