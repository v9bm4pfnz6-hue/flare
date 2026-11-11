# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Run tests of Triton programs on CPU tensors in Triton's interpreter.

Triton chooses the interpreter when a program is declared, so the tests of a module that launch
Triton programs run in a child process that sets TRITON_INTERPRET before it imports them. The
interpreter evaluates a program with NumPy: it checks indexing, masks, the order of operations and
IEEE arithmetic. It does not model the thread layout of a reduction or fused multiply-adds, and it
does not round to FP8 as the device does.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
interpreted = pytest.mark.skipif(not INTERPRETED, reason="runs in the Triton interpreter child process")


def run_interpreted(module, passed, *names):
    """Run the named tests of the test file `module` in an interpreter child and require `passed` passes."""
    if INTERPRETED:
        pytest.skip("this process runs the interpreted tests")
    environment = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(module), "-k", " or ".join(names)],
        env=environment, capture_output=True, text=True, timeout=1800, cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0 and f"{passed} passed" in result.stdout, (
        result.stdout[-3000:] + result.stderr[-2000:])
