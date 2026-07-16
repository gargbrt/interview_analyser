"""Minimal pytest-compatible test runner for environments without pytest
installed (this sandbox has no network access to pip install it). Real
contributors should just use `pytest` directly -- this script exists only
so the test suite could be executed end-to-end here.

Supports: a `tmp_path` fixture (real temp dir per test) and a fake `pytest`
module providing `pytest.raises` as a context manager, since test_analyzer.py
uses it.
"""
from __future__ import annotations

import contextlib
import importlib.util
import inspect
import pathlib
import shutil
import sys
import tempfile
import traceback
import types

TESTS_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(TESTS_DIR.parent / "src"))
sys.path.insert(0, str(TESTS_DIR))


# --- fake pytest shim (only what our tests use) ---
fake_pytest = types.ModuleType("pytest")


@contextlib.contextmanager
def _raises(exc_type, match=None):
    try:
        yield
    except exc_type as e:
        if match and match not in str(e):
            raise AssertionError(f"Exception message {e!r} did not match {match!r}") from e
        return
    else:
        raise AssertionError(f"Expected {exc_type} to be raised, but nothing was.")


fake_pytest.raises = _raises
sys.modules["pytest"] = fake_pytest


def run_all() -> tuple[int, int]:
    passed, failed = 0, 0
    test_files = sorted(TESTS_DIR.glob("test_*.py"))

    for test_file in test_files:
        module_name = test_file.stem
        spec = importlib.util.spec_from_file_location(module_name, test_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        test_funcs = [
            (name, obj) for name, obj in vars(module).items()
            if name.startswith("test_") and callable(obj)
        ]

        for name, func in test_funcs:
            sig = inspect.signature(func)
            kwargs = {}
            tmp_dir_ctx = None
            if "tmp_path" in sig.parameters:
                tmp_dir_ctx = tempfile.mkdtemp(prefix="interview_analyzer_test_")
                kwargs["tmp_path"] = pathlib.Path(tmp_dir_ctx)

            try:
                func(**kwargs)
                print(f"PASS  {module_name}.py::{name}")
                passed += 1
            except Exception:
                print(f"FAIL  {module_name}.py::{name}")
                traceback.print_exc()
                failed += 1
            finally:
                if tmp_dir_ctx:
                    shutil.rmtree(tmp_dir_ctx, ignore_errors=True)

    return passed, failed


if __name__ == "__main__":
    passed, failed = run_all()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
