from __future__ import annotations

import logging
import ast
import json
from typing import List, Optional, Tuple, Union
from core.data_builder.not_valid_reason import NotValidReason
from core.data_builder.transformations import ALTERING_TRANSFORMATION_IDS
logger = logging.getLogger(__name__)



import multiprocessing as mp
import queue
import traceback
import threading
import os

import signal
import hashlib
import pandas as pd



def df_checksum(df: pd.DataFrame) -> str:
    data_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(data_bytes).hexdigest()


def _preprocess_code(code: str) -> str:
    """
    Preprocess code string to normalize whitespace and formatting.
    - Strip leading/trailing whitespace
    - Normalize line endings
    - Remove trailing whitespace from each line
    """
    # Normalize line endings to \n
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    # Split into lines, strip trailing whitespace, and rejoin
    lines = [line.rstrip() for line in code.split('\n')]
    # Remove leading empty lines
    while lines and not lines[0].strip():
        lines.pop(0)
    # Remove trailing empty lines
    while lines and not lines[-1].strip():
        lines.pop()
    # Rejoin with normalized line endings
    return '\n'.join(lines)


def check_codes_are_equal(reference_code: str, transformed_code: str) -> bool:
    """
    Return True if the two source strings parse to the same AST (ignoring 
    line numbers, column offsets, and other non‑semantic details).
    
    Codes are preprocessed to normalize whitespace and formatting before comparison.
    """
    # Preprocess both codes before parsing
    reference_code = _preprocess_code(reference_code)
    transformed_code = _preprocess_code(transformed_code)
    
    reference_tree = ast.parse(reference_code)
    try:
        transformed_tree = ast.parse(transformed_code)
    except Exception as e:
        raise e
    dump_kwargs = dict(include_attributes=False, annotate_fields=False)
    return ast.dump(reference_tree, **dump_kwargs) == ast.dump(transformed_tree, **dump_kwargs)


def extract_ground_truth_altering_id(paths: list) -> int | None:
    """
    Extract the semantic-altering transformation ID from paths.
    Each path has exactly one altering transformation (IDs 1-4).
    """
    if not paths or not isinstance(paths[0], (list, tuple)):
        raise ValueError(f"Invalid path format: expected list of lists/tuples, got: {paths}")
   
    for tid in paths[0]:
        t = int(tid)
        if t in ALTERING_TRANSFORMATION_IDS:
            return t
    raise ValueError(f"No altering transformation ID found in first path: {paths[0]}")


def parse_paths(paths_raw):
    """Parse paths from string to list[list[int]]."""
    if isinstance(paths_raw, (list, tuple)):
        return paths_raw
    if pd.isna(paths_raw) or paths_raw == "":
        return []
    try:
        parsed = json.loads(paths_raw)
        if isinstance(parsed, list) and all(isinstance(p, list) for p in parsed):
            return [[int(x) for x in path] for path in parsed]
        return []
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def _child_run(code_str: str, tests: List[str], out_q: "mp.Queue[Tuple[bool, Optional[str]]]") -> None:
    """
    Child process: define code once, then run all tests sequentially.
    No per-test timeout here; parent enforces a single overall timeout.
    """
    try:
        g = {}
        exec(code_str, g)
        for i, t in enumerate(tests):
            try:
                exec(t, g)
            except AssertionError as e:
                # Provide more context about which test failed
                error_msg = f"AssertionError in test {i+1}: {t}\n{str(e)}"
                out_q.put((False, error_msg))
                return
            except Exception as e:
                # Provide more context about which test had an error
                error_msg = f"Error in test {i+1}: {t}\n{traceback.format_exc()}"
                out_q.put((False, error_msg))
                return
        out_q.put((True, None))
    except BaseException:
        out_q.put((False, traceback.format_exc()))


def _run_in_current_process(code_str: str, tests: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Execute code + tests in the current process and return (ok, error_msg).
    This is used by the signal-based timeout path (daemon processes can't spawn children).
    """
    try:
        g = {}
        exec(code_str, g)
        for i, t in enumerate(tests):
            try:
                exec(t, g)
            except TimeoutError as e:
                return (False, str(e))
            except AssertionError as e:
                return (False, f"AssertionError in test {i+1}: {t}\n{str(e)}")
            except Exception:
                return (False, f"Error in test {i+1}: {t}\n{traceback.format_exc()}")
        return (True, None)
    except TimeoutError as e:
        return (False, str(e))
    except BaseException:
        return (False, traceback.format_exc())


def _run_with_signal_timeout(code_str: str, tests: List[str], timeout: float) -> Tuple[bool, Optional[str]]:
    """
    Run code+tests in-process with a Unix signal-based timeout.
    This works in daemon multiprocessing workers (which cannot spawn child processes).
    NOTE: Must run on the main thread.
    """
    if timeout is None or timeout <= 0:
        return _run_in_current_process(code_str, tests)

    if threading.current_thread() is not threading.main_thread():
        # Can't reliably enforce signal timeouts off the main thread.
        return (False, "Timeout enforcement unavailable (not on main thread).")

    def _alarm_handler(signum, frame):
        # Include the word "timeout" so downstream logic can reliably detect it.
        raise TimeoutError(f"Timeout: overall timeout exceeded ({timeout:.2f}s).")

    prev_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _alarm_handler)
        # setitimer gives sub-second precision.
        signal.setitimer(signal.ITIMER_REAL, timeout)
        return _run_in_current_process(code_str, tests)
    except TimeoutError as e:
        return (False, str(e))
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except Exception:
            pass
        try:
            signal.signal(signal.SIGALRM, prev_handler)
        except Exception:
            pass


def tests_pass(
    code_str: str,
    tests: Union[List[str], str],
    *,
    timeout: float = 5.0,          # overall timeout for code + all tests
    return_error: bool = False,
):

    # TODO: Refactor this. _CacheTestResult is unnecessary. If Redis is not available, raise an error (stop processing and check for valid states).

    # Optional Redis cache (best-effort). Keep `tests_pass` usable even if redis isn't installed
    # or Redis isn't reachable in a given environment.
    CacheTestResult = None
    try:
        from core.data_builder.cache_manager.cache_test_result import CacheTestResult as _CacheTestResult
        CacheTestResult = _CacheTestResult
    except Exception:
        CacheTestResult = None

    if CacheTestResult is not None:
        cached_result = CacheTestResult.get(code_str, tests)
        if cached_result is not None:
            success, error_msg = cached_result
            if return_error:
                return (success, error_msg)
            return success
    
    # Handle case where tests is a string (e.g., from CSV) that needs to be parsed
    if isinstance(tests, str):
        try:
            tests = ast.literal_eval(tests)
        except (ValueError, SyntaxError) as e:
            error_msg = f"Failed to parse tests string: {e}"
            if CacheTestResult is not None:
                CacheTestResult.set(code_str, tests, False, error_msg)
            if return_error:
                return (False, error_msg)
            return False
    
    # Ensure tests is a list
    if not isinstance(tests, list):
        error_msg = f"Tests must be a list or parseable string, got {type(tests)}"
        if CacheTestResult is not None:
            CacheTestResult.set(code_str, tests, False, error_msg)
        if return_error:
            return (False, error_msg)
        return False
    
    # Ensure tests list is not empty
    if len(tests) == 0:
        error_msg = "Tests list is empty"
        if CacheTestResult is not None:
            CacheTestResult.set(code_str, tests, False, error_msg)
        if return_error:
            return (False, error_msg)
        return False
    
    # IMPORTANT: untrusted generated code can contain infinite loops.
    # A thread-based "timeout" cannot stop a stuck thread, so we must avoid threads here.
    current_process = mp.current_process()

    # Fast path on Unix: signal-based timeout (no process spawning, and works in notebooks),
    # but only available from the main thread.
    if os.name != "nt" and threading.current_thread() is threading.main_thread():
        ok, err = _run_with_signal_timeout(code_str, tests, timeout)
        if CacheTestResult is not None:
            CacheTestResult.set(code_str, tests, ok, err)
        return (ok, err) if return_error else ok

    # If we're in a daemon worker and not on the main thread, we cannot safely enforce a timeout.
    # Returning a failure is safer than risking an unkillable hang.
    if current_process.daemon:
        msg = "Timeout enforcement unavailable (daemon process and not on main thread)."
        if CacheTestResult is not None:
            CacheTestResult.set(code_str, tests, False, msg)
        return (False, msg) if return_error else False

    # Fallback: run in a child process and enforce a hard timeout we can kill.
    # macOS/Jupyter-safe: use spawn explicitly.
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue(maxsize=1)

    p = ctx.Process(target=_child_run, args=(code_str, tests, out_q))
    p.daemon = True
    p.start()
    p.join(timeout)

    if p.is_alive():
        try:
            p.terminate()
            p.join(0.2)
            if p.is_alive():
                p.kill()
            p.join()
        except Exception:
            # Best-effort: if we can't kill (restricted env), still treat as timeout.
            pass
        msg = f"Timeout: overall timeout exceeded ({timeout:.2f}s)."
        if CacheTestResult is not None:
            CacheTestResult.set(code_str, tests, False, msg)
        return (False, msg) if return_error else False

    try:
        ok, err = out_q.get_nowait()
    except queue.Empty:
        msg = "No result from child process."
        if CacheTestResult is not None:
            CacheTestResult.set(code_str, tests, False, msg)
        return (False, msg) if return_error else False

    if CacheTestResult is not None:
        CacheTestResult.set(code_str, tests, ok, err)
    return (ok, err) if return_error else ok




# TODO: refactor this
def get_transformation_invalid_reason(
    old_code: str,
    new_code: str,
    altering_applied: bool,
    tests: List[str],
    *,
    timeout: float = 30.0,
) -> Optional[NotValidReason]:
    """
    Validate a transformation and determine if it's invalid.
    
    Checks if the transformed code is valid according to transformation rules:
    - Codes must be different
    - If altering_applied=False (no altering applied yet), tests must pass
    - If altering_applied=True (altering applied), tests must fail
    
    Args:
        old_code: The original code before transformation
        new_code: The code after transformation
        altering_applied: Whether semantic altering transformation has been applied
        tests: List of test cases to run against the transformed code
        
    Returns:
        None if transformation is valid, otherwise returns NotValidReason enum value:
        - NotValidReason.SAME: codes are equal (no change)
        - NotValidReason.TESTS_FAILED_NO_SEMANTIC_CHANGE: tests failed when no semantic change occurred
        - NotValidReason.TESTS_PASSED_WITH_SEMANTIC_CHANGE: tests passed when semantic change occurred
    """
    # Check if codes are the same
    if check_codes_are_equal(old_code, new_code):
        return NotValidReason.SAME
    
    # Run tests with timeout for transformation validation
    timeout_occurred = False
    try:
        # Use return_error=True to detect timeouts specifically
        result = tests_pass(new_code, tests, timeout=timeout, return_error=True)
        if isinstance(result, tuple):
            passed, error_msg = result
            # Check if error message indicates timeout
            if error_msg and 'timeout' in error_msg.lower():
                timeout_occurred = True
        else:
            passed = result
    except TimeoutError:
        # Explicit timeout exception
        timeout_occurred = True
        passed = False
    except Exception as e:
        # Other errors during test execution
        passed = False
    
    # If timeout occurred, return timeout reason
    if timeout_occurred:
        return NotValidReason.TIMEOUT
    
    if not altering_applied:
        # No altering applied yet - tests must pass
        if not passed:
            return NotValidReason.TESTS_FAILED_NO_SEMANTIC_CHANGE
    else:
        # Altering applied - tests must fail
        if passed:
            return NotValidReason.TESTS_PASSED_WITH_SEMANTIC_CHANGE
    
    return None





def get_data_detail(k: int, alt_pos: int) -> pd.DataFrame:
    import pandas as pd
    path_format = 'data/1_k_and_pos/k-{k}_als-pos-{alt_pos}.csv'
    file_path = path_format.format(k=k, alt_pos=alt_pos)
    return pd.read_csv(file_path)


def creat_transformations_description(transformations: List[Transformer]):
    format_instruction_helper = "format: <transformation_id>: <transformation_description>"
    return format_instruction_helper + '\n' + '\n'.join([
        f"{t.id}: {t.long_description}" for t in transformations
    ])


def get_data_k(k: int) -> pd.DataFrame:
    import pandas as pd
    from core.data_builder.transformations import ID_TO_TRANSFORMER
    path_format = f'data/2_aggregate_k/k_{k}.csv'
    file_path = path_format.format(k=k)
    df = pd.read_csv(file_path)
    df["path_ids"] = df["path"].apply(ast.literal_eval)
    df['path_names'] = df['path_ids'].apply(lambda x: [ [ID_TO_TRANSFORMER[p] for p in path] for path in x])
    df['unique_path'] = df['path'].apply(count_unique_inner_sets)
    df['len_path'] = df["path"].apply(len)
    return df

def get_result_k(k: int) -> pd.DataFrame:
    import pandas as pd
    path_format = f'results/aggregate_k/k_{k}.csv'
    file_path = path_format.format(k=k)
    return pd.read_csv(file_path)
    

def save_aggregate_k(df: pd.DataFrame, k: int):
    print(f'k = {k}, data shape = {df.shape}')
    path_format = 'tmp_aggregate/k_{k}.csv'
    file_path = path_format.format(k=k)
    df.to_csv(file_path, index=False)







def draw_conf_matrix_for_k(k: int):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    df = get_result_k(k)
    if (df['predicted_class'] == 'Error').any():
        raise ValueError("Data contains 'Error' in predicted_class column.")
    pred = df['predicted_class']
    real = df['real_class']

    cm = confusion_matrix(real, pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['False', 'True'],
        yticklabels=['False', 'True']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def count_unique_inner_sets(row_path):
    # e.g. [[1,2], [2,1], [3]] -> unique = {frozenset({1,2}), frozenset({3})} -> 2
    return {frozenset(inner) for inner in row_path} 




def parse_predicted_path(raw) -> list | None:
    """Parse predicted_path from CSV (can be string like '[5, 7, 1]' or list)."""
    if pd.isna(raw) or raw == "":
        raise ValueError("predicted_path is missing or NaN")
   
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw if str(x).strip().isdigit()]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [int(x) for x in parsed if isinstance(x, (int, float))]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    raise ValueError(f"Unable to parse predicted_path: {raw!r}")