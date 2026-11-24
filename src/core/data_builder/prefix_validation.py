from __future__ import annotations

import ast
from enum import Enum
import logging
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import signal
import sys
import threading
from typing import List, Optional

from .invalid_prefix_reason import InvalidPrefixReason

logger = logging.getLogger(__name__)


def check_codes_are_equal(reference_code: str, transformed_code: str) -> bool:
    """Compare Python syntax trees without changing source or comparing locations."""
    return ast.dump(ast.parse(reference_code)) == ast.dump(ast.parse(transformed_code))


class TestOutcome(Enum):
    __test__ = False
    PASSED = "passed"
    PROGRAM_FAILURE = "program_failure"
    TIMEOUT = "timeout"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


class _ValidationDeadline(BaseException):
    """Distinguish the harness alarm from a program raising TimeoutError."""


def _execute_tests(code, tests) -> tuple[TestOutcome, str | None]:
    namespace = {}
    try:
        exec(code, namespace)
        for test in tests:
            exec(test, namespace)
    except (Exception, SystemExit) as exc:
        return TestOutcome.PROGRAM_FAILURE, f"{type(exc).__name__}: {exc}"
    return TestOutcome.PASSED, None


def _child_run(code_str: str, tests: list[str], result) -> None:
    with result:
        code = compile(code_str, "<candidate>", "exec")
        compiled_tests = [compile(test, "<test>", "exec") for test in tests]
        result.send(_execute_tests(code, compiled_tests))


def _run_with_signal_timeout(code, tests, timeout: float) -> tuple[TestOutcome, str | None]:
    expired = False
    finished = False

    def on_alarm(signum, frame):
        nonlocal expired
        if not finished:
            expired = True
            raise _ValidationDeadline

    previous_handler = signal.signal(signal.SIGALRM, on_alarm)
    try:
        # Generated code can contain bare except blocks that swallow one alarm.
        signal.setitimer(signal.ITIMER_REAL, timeout, 0.01)
        outcome, error = _execute_tests(code, tests)
    except _ValidationDeadline:
        pass
    finally:
        finished = True
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    if expired:
        return TestOutcome.TIMEOUT, f"Validation exceeded {timeout:g} seconds."
    return outcome, error


def _run_in_child(code_str: str, tests: list[str], timeout: float) -> tuple[TestOutcome, str | None]:
    ctx = mp.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    child = ctx.Process(target=_child_run, args=(code_str, tests, send), daemon=True)
    with receive, send:
        child.start()
        try:
            send.close()
            ready = wait([receive, child.sentinel], timeout)
            if not ready:
                return TestOutcome.TIMEOUT, f"Validation exceeded {timeout:g} seconds."
            if receive in ready:
                try:
                    return receive.recv()
                except EOFError:
                    pass
            return TestOutcome.INFRASTRUCTURE_FAILURE, "Validation worker exited without a result."
        finally:
            child.join(0.1)
            if child.is_alive():
                child.kill()
                child.join()
            child.close()


def run_validation_tests(
    code_str: str, tests: list[str] | str, *, timeout: float
) -> tuple[TestOutcome, str | None]:
    if isinstance(tests, str):
        try:
            tests = ast.literal_eval(tests)
        except (SyntaxError, ValueError) as exc:
            return TestOutcome.INFRASTRUCTURE_FAILURE, f"Cannot parse tests: {exc}"
    if not isinstance(tests, list) or not tests or not all(isinstance(test, str) for test in tests):
        return TestOutcome.INFRASTRUCTURE_FAILURE, "Expected a nonempty list of test source strings."
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Validation timeout must be finite and positive.")

    use_signal = os.name != "nt" and threading.current_thread() is threading.main_thread()
    if not use_signal and mp.current_process().daemon:
        return TestOutcome.INFRASTRUCTURE_FAILURE, "Validation worker cannot enforce a timeout."

    from core.data_builder.cache_manager.cache_test_result import CacheTestResult

    settings = {
        "timeout": float(timeout),
        "backend": "signal" if use_signal else "spawn",
        "runtime": sys.version,
        "platform": sys.platform,
        "optimize": sys.flags.optimize,
        "hash_seed": os.getenv("PYTHONHASHSEED"),
    }
    cached = CacheTestResult.get(code_str, tests, settings)
    if cached is not None:
        passed, error = cached
        return (TestOutcome.PASSED if passed else TestOutcome.PROGRAM_FAILURE), error

    try:
        compiled_tests = [compile(test, "<test>", "exec") for test in tests]
    except (SyntaxError, ValueError) as exc:
        return TestOutcome.INFRASTRUCTURE_FAILURE, f"Cannot compile tests: {exc}"
    code = compile(code_str, "<candidate>", "exec")
    if use_signal:
        outcome, error = _run_with_signal_timeout(code, compiled_tests, timeout)
    else:
        outcome, error = _run_in_child(code_str, tests, timeout)

    if outcome in (TestOutcome.PASSED, TestOutcome.PROGRAM_FAILURE):
        CacheTestResult.set(code_str, tests, settings, outcome is TestOutcome.PASSED, error)
    return outcome, error


def tests_pass(
    code_str: str,
    tests: list[str] | str,
    *,
    timeout: float = 5.0,
    return_error: bool = False,
):
    """Run validation tests and return a boolean, optionally with the failure detail."""
    outcome, error = run_validation_tests(code_str, tests, timeout=timeout)
    passed = outcome is TestOutcome.PASSED
    return (passed, error) if return_error else passed


def get_transformation_invalid_reason(
    old_code: str,
    new_code: str,
    altering_applied: bool,
    tests: List[str],
    *,
    timeout: float = 30.0,
) -> Optional[InvalidPrefixReason]:
    if check_codes_are_equal(old_code, new_code):
        return InvalidPrefixReason.SAME

    outcome, error = run_validation_tests(new_code, tests, timeout=timeout)
    if outcome is TestOutcome.INFRASTRUCTURE_FAILURE:
        logger.warning("Validation infrastructure failure: %s", error)
        return InvalidPrefixReason.VALIDATION_ERROR
    if outcome is TestOutcome.TIMEOUT:
        return InvalidPrefixReason.TIMEOUT
    if altering_applied and outcome is TestOutcome.PASSED:
        return InvalidPrefixReason.TESTS_PASSED_WITH_SEMANTIC_CHANGE
    if not altering_applied and outcome is TestOutcome.PROGRAM_FAILURE:
        return InvalidPrefixReason.TESTS_FAILED_NO_SEMANTIC_CHANGE
    return None
