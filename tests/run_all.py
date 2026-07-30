#!/usr/bin/env python3
"""Run both unittest-style and function-style tests without extra tooling."""

from __future__ import annotations

import importlib
import inspect
import sys
import traceback
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "tests"
sys.path[:0] = [str(ROOT), str(TEST_DIR)]


def main() -> None:
    module_names = [path.stem for path in sorted(TEST_DIR.glob("test_*.py"))]
    modules = [importlib.import_module(name) for name in module_names]

    unittest_suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromModule(module) for module in modules
    )
    unittest_result = unittest.TextTestRunner(verbosity=2).run(unittest_suite)

    function_failures: list[str] = []
    function_count = 0
    for module in modules:
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_") or function.__module__ != module.__name__:
                continue
            function_count += 1
            qualified_name = f"{module.__name__}.{name}"
            print(f"[RUN] {qualified_name}", flush=True)
            try:
                function()
            except Exception:
                function_failures.append(qualified_name)
                traceback.print_exc()

    if not unittest_result.wasSuccessful() or function_failures:
        if function_failures:
            print("[FAIL] function tests: " + ", ".join(function_failures), file=sys.stderr)
        raise SystemExit(1)
    print(
        f"[OK] {unittest_result.testsRun} unittest tests and "
        f"{function_count} function tests passed"
    )


if __name__ == "__main__":
    main()
