import subprocess
import time
import signal
import os
from tqdm import tqdm
from threading import Thread, Event
from typing import Any
import queue
import json
import tempfile
import shutil
from pathlib import Path
import re
import glob
import select
import threading
import fcntl  # for file locking
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import sys 
import random
import datetime
import string
from math import floor
import multiprocessing as mp
import sqlite3
import psycopg2
import psycopg2.extras
import pytz
import redis
from redis.exceptions import ConnectionError, TimeoutError

# Define the Toronto timezone
toronto_tz = pytz.timezone("America/Toronto")
# Auto-release stale locks (seconds)
LOCK_TIMEOUT_S = int(os.environ.get("LOCK_TIMEOUT_S", "960"))

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from LLMs.plan_and_code_query import plan_and_code_query
from LLMs.utils import compile_prompt_to_md

from typing import Any, Callable, cast
import nbformat
from nbformat.v4 import new_notebook, new_code_cell

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
INVALID_RE = re.compile(r"Invalid submission:\s*(.*)")

manual_versions = {"osic-pulmonary-fibrosis-progression_pradyut23_pulmonary-fibrosis-eda": "3.8",
"alaska2-image-steganalysis_dzz1th_kernel3de146856b": "3.9",
"spooky-author-identification_ttetls_a-gentle-mathematical-approach-to-spooky-en-fr": "3.6",
"statoil-iceberg-classifier-challenge_brassmonkey381_viewing-leak-and-machine-images": "3.6",
"rsna-2022-cervical-spine-fracture-detection_lsl000ud_rsna2022-7th-place-inference": "3.10",
"petfinder-pawpularity-score_guillaumes_petfinder-ensemble-xgboost-tabnet": "3.10",
"cassava-leaf-disease-classification_surayuthpintawong_inference-cassava": "3.5",
"cassava-leaf-disease-classification_saurabh2mishra_cassava-leaf-disease-inference-label-smoothing": "3.8",
"aerial-cactus-identification_visali_cactus-classification-using-fastai": "3.9",
"AI4Code_valentinaliferov_ai4code-submit": "3.11",
"AI4Code_vaaliferov_ai4code-submit": "3.11"
}

with open("./upgrade/competition_score_rank.json", "r", encoding="utf-8") as f:
    lower_better = json.load(f)

JSON_PATH = "apiDowngrade/kernel_w_pyVersion.json"
SUBMISSION_NOAPI_PATH = "apiDowngrade/submission_noAPI.json"

patterns = [
    "will ", 
    "'ll ", "’ll", 
    "'s going to ", "'re going to ", 
    "’s going to ", "’re going to ", 
    "is going to ", "are going to ",
    "'s about to ", "'re about to ", 
    "’s about to ", "’re about to ", 
    "is about to ", "are about to "
]

def _is_lock_stale(locked_at) -> bool:
    try:
        t = float(locked_at or 0)
    except Exception:
        t = 0.0
    return t > 0 and (time.time() - t) > LOCK_TIMEOUT_S

class KernelStore:
    """
    File-backed kernel state with cross-process locking.

    Expected kernel JSON format:
      {
        "some_script_name": {"fix": 0, "response": "...", "ready": 0},
        ...
      }

    We add:
      - locked: 0/1
      - locked_by: "llm"|"exec"|None
      - executed: 0/1 (prevents re-executing the same ready=1 item repeatedly)
      - last_exec: dict (optional)
      - last_err: str (optional)
      - failed: 0/1 (terminal state; do not retry/claim again)
      - failure: str (human-readable failure reason / exception summary)
    """

    def __init__(self, kernel_path: Path, lock_path: Path | None = None):
        self.kernel_path = Path(kernel_path)
        self.lock_path = Path(lock_path) if lock_path else self.kernel_path.with_suffix(
            self.kernel_path.suffix + ".lock"
        )
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.kernel_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure files exist
        if not self.lock_path.exists():
            self.lock_path.write_text("", encoding="utf-8")
        if not self.kernel_path.exists():
            self.kernel_path.write_text("{}", encoding="utf-8")

    def _locked(self, shared: bool = False):
        f = open(self.lock_path, "a+", encoding="utf-8")
        fcntl.flock(f.fileno(), fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        return f

    def _load_unlocked(self) -> dict:
        try:
            with open(self.kernel_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if not isinstance(data, dict):
                return {}
            return data
        except Exception:
            return {}

    def _save_unlocked(self, data: dict) -> None:
        tmp = self.kernel_path.with_suffix(self.kernel_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False)
        os.replace(tmp, self.kernel_path)

    def _load_state(self, key: str) -> dict:
        lockf = self._locked(shared=True)
        try:
            data = self._load_unlocked()
            v = data.get(key, {})
            if not isinstance(v, dict):
                v = {}
            return v
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
            lockf.close()

    def _latest_score(self, v: dict) -> float | None:
        up = v.get("upgrade", {})
        if isinstance(up, dict) and up:
            try:
                max_key = max(up.keys(), key=lambda k: int(k))
            except Exception:
                max_key = sorted(up.keys())[-1]
            s = up.get(max_key, {}).get("score", None)
            return s if isinstance(s, (int, float)) else None
        s0 = v.get("score", None)
        return s0 if isinstance(s0, (int, float)) else None
    
    def _is_item_done(self, v: dict, max_fix: int) -> bool:
        # Terminal failure (e.g., prompt too long / context limit): treat as done
        if int(v.get("failed", 0) or 0) == 1:
            return True

        target = v.get("target", None)
        current = self._latest_score(v)

        # Stop early if replicable
        if is_replicable(target, current) or int(v.get("fix", 0)) > max_fix:
            return True
        
        # Otherwise must finish max_fix, and the last fix has been executed
        fix = int(v.get("fix", 0))
        ready = int(v.get("ready", 0))
        executed = int(v.get("executed", 0))
        if fix >= max_fix:
            return (executed == 1) and (ready == 0)
        
        return False
    
    def all_done(self, max_fix: int) -> bool:
        """
        Done means every item is either:
          - terminal failed (failed==1), OR
          - _is_item_done(v, max_fix) == True

        - every item reached fix >= max_fix
        - and there is no pending execution of the last produced fix (ready must be 0)
        - and the last produced fix has been executed at least once (executed==1)

        Note: execution failure at fix==max_fix is still considered "executed".
        """
        # lockf = self._locked(shared=True)
        # try:
        data = self._load_unlocked()
        # return all(self._is_item_done(v, max_fix) for v in data.values())
        for v in data.values():
            if not isinstance(v, dict):
                continue
            # Explicit check requested: only evaluate normal completion when failed != 1
            if int(v.get("failed", 0) or 0) != 1 and not self._is_item_done(v, max_fix):
                return False
        return True
        # finally:
        #     fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
        #     lockf.close()

    def get_progress(self, max_fix: int) -> tuple[int, int]:
        """Returns (num_done, total_items)"""
        lockf = self._locked(shared=True)
        try:
            data = self._load_unlocked()
            total = len(data)
            done = sum(1 for v in data.values() if self._is_item_done(v, max_fix))
            return done, total
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
            lockf.close()
    
    def get_step_progress(self, max_fix: int) -> tuple[int, int, int, int]:
        """
        Returns (llm_steps_done, exec_steps_done, items_done, total_items).

        Rule: if _is_item_done(v,max_fix) is True (including replicable early-stop),
        count the item as fully completed (== max_fix steps) for both llm+exec bars.
        """
        lockf = self._locked(shared=True)
        try:
            data = self._load_unlocked()
            total = len(data)

            llm_steps = 0
            exec_steps = 0
            items_done = 0

            for v in data.values():
                if not isinstance(v, dict):
                    continue

                done = self._is_item_done(v, max_fix)
                fix = int(v.get("fix", 0) or 0)
                ready = int(v.get("ready", 0) or 0)
                executed = int(v.get("executed", 0) or 0)

                if done:
                    items_done += 1
                    llm_steps += max_fix
                    exec_steps += max_fix
                    continue

                # LLM step progress = number of fixes produced so far (cap at max_fix)
                llm_steps += min(max_fix, max(0, fix))

                # Exec step progress: each produced fix should be executed once.
                # If current fix is not executed yet (executed==0), last executed is fix-1.
                if fix <= 0:
                    exec_i = 0
                else:
                    exec_i = fix if (executed == 1 and ready == 0) else max(0, fix - 1)
                exec_steps += min(max_fix, exec_i)

            return llm_steps, exec_steps, items_done, total
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
            lockf.close()


    def claim_for_llm(self, max_fix: int = 10) -> tuple[str, dict] | None:
        """
        Claim for LLM if:
          - fix < max_fix
          - ready == 0
          - not locked
          - and (fix == 0 OR executed == 1)  <-- prevents LLM making fix#2 before fix#1 ran
          - and failed != 1
        """
        lockf = self._locked()
        try:
            data = self._load_unlocked()
            for k, v in data.items():
                if int(v.get("failed", 0) or 0) == 1:
                    continue
                if self._is_item_done(v, max_fix):
                    continue

                fix = int(v.get("fix", 0))
                ready = int(v.get("ready", 0))
                locked = int(v.get("locked", 0))
                executed = int(v.get("executed", 0))
                
                if fix >= max_fix: # reached max fix
                    continue
                if ready != 0 or locked != 0: # not ready for execution or not locked
                    continue
                if fix > 0 and executed != 1: # previous fix not executed yet
                    continue
                if locked != 0 and _is_lock_stale(v.get("locked_at", 0)):
                    if v['locked_by'] == 'llm':
                        v['ready'] = 0
                        v['executed'] = 1
                    else:
                        v['ready'] = 1
                        v['executed'] = 0
                    v["locked"] = 0
                    v["locked_by"] = None
                    v.pop("locked_at", None)
                    data[k] = v
                    self._save_unlocked(data)
                    continue
                if locked == 1:
                    continue

                v["locked"] = 1
                v["locked_by"] = "llm"
                v["locked_at"] = time.time()
                v["ready"] = 0
                data[k] = v

                self._save_unlocked(data)
                return k, v
            
            return None
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
            lockf.close()

    def claim_for_exec(self, max_fix: int = 10) -> tuple[str, dict] | None:
        """
        Execute fixes up to and including max_fix (fix==10 must run).
        Claim if: fix <= max_fix AND ready == 1 AND executed == 0 AND not locked AND failed != 1.
        """
        lockf = self._locked()
        try:
            data = self._load_unlocked()
            for k, v in data.items():
                if int(v.get("failed", 0) or 0) == 1:
                    continue
                if self._is_item_done(v, max_fix):
                    continue

                fix = int(v.get("fix", 0))
                ready = int(v.get("ready", 0))
                executed = int(v.get("executed", 0))
                locked = int(v.get("locked", 0))

                if locked != 0 and _is_lock_stale(v.get("locked_at", 0)):
                    if v['locked_by'] == 'llm':
                        v['ready'] = 0
                        v['executed'] = 1
                    else:
                        v['ready'] = 1
                        v['executed'] = 0
                    v["locked"] = 0
                    v["locked_by"] = None
                    v.pop("locked_at", None)
                    data[k] = v
                    self._save_unlocked(data)
                    continue
                if locked == 1:
                    continue

                if fix <= max_fix and ready == 1 and executed == 0 and locked == 0:
                    v["locked"] = 1
                    v["locked_by"] = "exec"
                    v["locked_at"] = time.time()
                    v["ready"] = 0
                    data[k] = v
                    self._save_unlocked(data)
                    return k, v
            return None
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
            lockf.close()
    
    def mark_failed(self, key: str, err_text: str, LLM_res=None) -> None:
        """Terminally fail an item so it won't be retried/claimed again."""
        lockf = self._locked()
        try:
            data = self._load_unlocked()
            v = data.get(key, {})
            if not isinstance(v, dict):
                v = {}

            v["failed"] = 1
            v["failure"] = err_text
            v["failure_LLM_response"] = _to_jsonable(LLM_res)

            # Ensure it's not stuck locked/ready
            v["ready"] = 0
            v["locked"] = 0
            v["locked_by"] = None

            data[key] = v
            self._save_unlocked(data)
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
            lockf.close()
        
    def mark_retryable_error(self, key: str) -> None:
        """Non-deterministic error: unlock item, allow retry."""
        lockf = self._locked()
        try:
            data = self._load_unlocked()
            v = data.get(key, {})
            if not isinstance(v, dict):
                v = {}

            # Unlock, so it can be claimed again
            v["locked"] = 0
            v["locked_by"] = None
            v.pop("locked_at", None)

            # Keep it eligible for LLM retry (ready must be 0 for claim_for_llm)
            v["ready"] = 0

            data[key] = v
            self._save_unlocked(data)
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
            lockf.close()

    def complete_llm(self, key: str, new_fix: int, nl_plan: str, llm_code: str, req_time, in_tok_count, out_tok_count, cached_tokens, model_name, completion, err_cell_num: int | None = None, raw: str | None = None, max_fix: int = 10) -> None:
        """After fixing: 
            - fix = new_fix
            - ready = 1 (eligible for execution)
            - executed = 0 (must execute this fix)
            - upgrade[str(new_fix)] gets response
        """
        lockf = self._locked()
        try:
            data = self._load_unlocked()
            v = data.get(key, {})

            v["fix"] = int(new_fix)
            v["ready"] = 1
            v["executed"] = 0

            up = v.get("upgrade")
            if not isinstance(up, dict):
                up = {}
            entry = up.get(str(new_fix), {})
            if not isinstance(entry, dict):
                entry = {}
            
            entry["plan"] = nl_plan
            entry["code"] = llm_code
            entry["request_time"] = req_time
            entry["in_tokens"] = in_tok_count
            entry["out_tokens"] = out_tok_count

            entry["cached_tokens"] = cached_tokens
            entry["model"] = model_name
        
            entry["full_message"] = _to_jsonable(completion)
            if raw is not None:
                entry["fixed_path"] = raw

            up[str(new_fix)] = entry
            v["upgrade"] = up


            v["locked"] = 0
            v["locked_by"] = None
            data[key] = v
            self._save_unlocked(data)
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
            lockf.close()

    def complete_exec(self, key: str, exec_result: dict) -> None:
        """
        After execution:
          - store results into upgrade[str(fix)]
          - ready = 0 (eligible for next LLM fix, if fix<max_fix)
          - executed = 1
        Always unlock.
        """
        lockf = self._locked()
        try:
            data = self._load_unlocked()
            v = data.get(key, {})
            fix = int(v.get("fix", 0))

            up = v.get("upgrade")
            if not isinstance(up, dict):
                up = {}
            entry = up.get(str(fix), {})
            if not isinstance(entry, dict):
                entry = {}

            # exec_result is currently {filename: {...}} in your code; normalize
            payload = exec_result.get(key) if isinstance(exec_result, dict) and key in exec_result else exec_result
            if isinstance(payload, dict):
                entry.update(payload)

            up[str(fix)] = entry
            v["upgrade"] = up

            # replicable detection
            if is_replicable(v.get("target", None), payload.get("score", None)) and "output" in payload:
                shutil.copy2(payload.get("output"), work_dir / 'csv_output')
                v["replicable"] = True

            v["ready"] = 0
            v["executed"] = 1

            v["locked"] = 0
            v["locked_by"] = None
            data[key] = v
            self._save_unlocked(data)
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
            lockf.close()

class NotebookRunner:
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds

    def monitor_execution(self, process, timeout_event, start_event, execution_queue, compt, filename):
        """Monitor Docker process output and detect execution start"""
        execution_started = False
        start_time = None

        try:
            while process.poll() is None and not timeout_event.is_set():
                # Use select to check if there's output available without blocking indefinitely
                ready, _, _ = select.select([process.stdout], [], [], 0.1)
                
                if ready:
                    # Reads stdout line-by-line from the subprocess
                    line = process.stdout.readline()
                    if line:
                        line = line.strip().replace('\r', '').replace('\x1b[K', '')
                        execution_queue.put(('output', line))

                        # print(f"GPU {compt} DOCKER: {line}")

                        # Detect when notebook execution starts (usually the message starts with '[NbClientApp] Executing notebook with kernel:')
                        if not execution_started and any(keyword in line.lower() for keyword in [
                            'converting notebook', 'executing notebook', 'executing cell', 'executing:', 'running cell', "debugging will proceed", #filename.lower(),
                        ]):
                            execution_started = True
                            start_time = time.monotonic()
                            start_event.set()
                            execution_queue.put(('status', 'Execution started'))
                    
                # Check timeout only after execution has started
                if execution_started and start_time:
                    elapsed_time = time.monotonic() - start_time

                    # Use carriage return to overwrite the same line
                    # print(f"\rElapsed {compt} time: {elapsed_time:.1f}s", end='', flush=True)

                    if elapsed_time > self.timeout_seconds:
                        # print(f"\nTimeout after {elapsed_time:.1f}s")
                        execution_queue.put(('status', f'Timeout after {elapsed_time:.1f}s'))
                        timeout_event.set()
                        break
               
                # time.sleep(0.1)
                
        except Exception as e:
            execution_queue.put(('error', str(e)))
    
    def run_single_notebook(self, docker_command, compt, filename):
        """Run a single notebook with timeout monitoring"""

        # Start a subprocess
        process = subprocess.Popen(
            docker_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            text=True,
            shell=True,
            preexec_fn=os.setsid
        )
        
        # Signaling timeout/start
        timeout_event = Event()
        start_event = Event()
        # Receive output/status messages from the monitor thread
        execution_queue = queue.Queue()
        
        # Launch monitoring thread
        monitor_thread = Thread(
            target=self.monitor_execution,
            args=(process, timeout_event, start_event, execution_queue, compt, filename)
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        
        result = {}
        start_time = None

        # Capture ALL output lines here (since monitor consumes stdout)
        all_lines = []
        abs_start_time = time.monotonic()

        try:
            # If the process is still running
            while process.poll() is None:
                # Add a fallback hard timeout (e.g. 10m limit + 5m buffer) in case execution start isn't detected
                if time.monotonic() - abs_start_time > (self.timeout_seconds + 300):
                    timeout_event.set()
                    try:
                        execution_queue.put(('status', f'Hard timeout after {(time.monotonic() - abs_start_time):.1f}s'))
                    except queue.Full:
                        pass

                # Watch for a timeout signal
                if timeout_event.is_set():
                    # Terminate the entire process group
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    # Wait for graceful shutdown with timeout
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        process.wait()

                    # Update results accordingly
                    result['timeout'] = True
                    result['execution_time'] = (time.monotonic() - start_time) if start_time else self.timeout_seconds

                    # Drain any queued output before returning
                    while True:
                        try:
                            msg_type, msg_data = execution_queue.get(timeout=0.1)
                        except queue.Empty:
                            break
                        if msg_type == "output":
                            all_lines.append(msg_data)
                        elif msg_type == "status" and "Execution started" in msg_data:
                            start_time = time.monotonic()

                    result["detail"] = "\n".join(all_lines)
                    if process.returncode not in (0, None):
                        result["error"] = result["detail"]

                    return result
                    
                # Drain queue messages
                while True:
                    # Check fallback hard timeout inside the drain loop to prevent starvation during heavy logging
                    if time.monotonic() - abs_start_time > (self.timeout_seconds + 300):
                        timeout_event.set()
                        try:
                            execution_queue.put(('status', f'Hard timeout after {(time.monotonic() - abs_start_time):.1f}s'))
                        except queue.Full:
                            pass
                        break

                    try:
                        msg_type, msg_data = execution_queue.get(timeout=0.3)
                    except queue.Empty:
                        break

                    if msg_type == "output":
                        all_lines.append(msg_data)
                    elif msg_type == "status" and "Execution started" in msg_data:
                        start_time = time.monotonic()

                time.sleep(0.1)
            
            # Process completed: drain remaining queue
            while True:
                if time.monotonic() - abs_start_time > (self.timeout_seconds + 300):
                    timeout_event.set()
                    try:
                        execution_queue.put(('status', f'Hard timeout after {(time.monotonic() - abs_start_time):.1f}s'))
                    except queue.Full:
                        pass
                    break
                
                try:
                    msg_type, msg_data = execution_queue.get(timeout=0.3)
                except queue.Empty:
                    break
                if msg_type == "output":
                    all_lines.append(msg_data)
                elif msg_type == "status" and "Execution started" in msg_data and start_time is None:
                    start_time = time.monotonic()

            if start_time:
                result['execution_time'] = time.monotonic() - start_time
            # else:
            #     result['execution_time'] = time.monotonic() - abs_start_time

            # result['success'] = process.returncode == 0

            result['detail'] = "\n".join(ANSI_RE.sub("", line) for line in all_lines)
            
            if process.returncode != 0:
                result["error"] = "\n".join(all_lines)

        except KeyboardInterrupt:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            result['error'] = 'Interrupted by user'
            result["detail"] = "\n".join(all_lines)
        except Exception as e:
            result['error'] = str(e)
            result["detail"] = "\n".join(all_lines)
        
        return result

def _to_jsonable(x: Any) -> Any:
    """Best-effort conversion of SDK objects (OpenAI/Gemini/etc.) into JSON-serializable data."""
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]

    # Pydantic v2 (OpenAI responses)
    if hasattr(x, "model_dump"):
        try:
            return _to_jsonable(x.model_dump())
        except Exception:
            pass

    # Some SDKs
    if hasattr(x, "to_dict"):
        try:
            return _to_jsonable(x.to_dict())
        except Exception:
            pass

    # Pydantic v1
    if hasattr(x, "dict"):
        try:
            return _to_jsonable(x.dict())
        except Exception:
            pass

    # Protobuf-like
    if hasattr(x, "__dict__"):
        try:
            return _to_jsonable(vars(x))
        except Exception:
            pass

    # Fallback: string
    try:
        json.dumps(x)
        return x
    except Exception:
        return str(x)

def progress_monitor(store_path: str, max_fix: int, q: "mp.queues.SimpleQueue", state_backend: str) -> None:
    """
    Single process that owns all tqdm rendering.
    Overall progress == count of items where KernelStore._is_item_done(v, max_fix) is True.
    Workers push events to q for real-time updates.
    """
    store = get_store(state_backend, store_path)

    llm_steps, exec_steps, items_done, total = store.get_step_progress(max_fix)

    # Upper bounds; may exceed due to early-stop (replicable) but still useful as a gauge.
    llm_total = max(1, total * max_fix)
    exec_total = max(1, total * max_fix)

    done_bar = tqdm(total=max(1, total), initial=items_done, desc="Items done (_is_item_done)", position=0, leave=True)
    llm_bar = tqdm(total=llm_total, initial=llm_steps, desc="LLM fixes completed", position=1, leave=True)
    exec_bar = tqdm(total=exec_total, initial=exec_steps, desc="Executions completed", position=2, leave=True)

    last_items_done = items_done
    last_llm_steps = llm_steps
    last_exec_steps = exec_steps
    last_total = total

    done_bar.set_postfix({"done": items_done, "total": total, "remaining": max(0, total - items_done)})

    while True:
        try:
            msg = q.get()  # blocks: updates happen on events => "true real-time"
            if not msg:
                continue
            kind = msg[0]
            if kind == "STOP":
                break
            if kind == "log":
                # msg = ("log", "text...")
                tqdm.write(str(msg[1]))
                continue
            # if kind == "llm_done":
            #     llm_done_cnt += 1
            #     llm_bar.update(1)
            # elif kind == "exec_done":
            #     exec_done_cnt += 1
            #     exec_bar.update(1)
            
            if kind == "llm_fail":
                # keep running; show failures in postfix
                cur = llm_bar.postfix if isinstance(llm_bar.postfix, dict) else {}
                llm_bar.set_postfix({"fails": int(cur.get("fails", 0)) + 1})
            elif kind == "exec_fail":
                cur = exec_bar.postfix if isinstance(exec_bar.postfix, dict) else {}
                exec_bar.set_postfix({"fails": int(cur.get("fails", 0)) + 1})
        except queue.Empty:
            pass

        llm_steps, exec_steps, items_done, total = store.get_step_progress(max_fix)

        # Recompute overall done after every event (truth source is kernel-state + _is_item_done)
        if total != last_total:
            last_total = total
            done_bar.total = max(1, total)
            llm_bar.total = max(1, total * max_fix)
            exec_bar.total = max(1, total * max_fix)

            # If total shrank (e.g., redis delete), clamp bar positions to current counts
            done_bar.n = min(done_bar.n, items_done)
            llm_bar.n = min(llm_bar.n, llm_steps)
            exec_bar.n = min(exec_bar.n, exec_steps)

            done_bar.refresh()
            llm_bar.refresh()
            exec_bar.refresh()

        if items_done > last_items_done:
            done_bar.update(items_done - last_items_done)
            last_items_done = items_done
        elif items_done < last_items_done:
            # handle decreases from deletions
            last_items_done = items_done
            done_bar.n = items_done
            done_bar.refresh()

        if llm_steps > last_llm_steps:
            llm_bar.update(llm_steps - last_llm_steps)
            last_llm_steps = llm_steps
        elif llm_steps < last_llm_steps:
            last_llm_steps = llm_steps
            llm_bar.n = min(llm_bar.n, llm_steps)
            llm_bar.refresh()

        if exec_steps > last_exec_steps:
            exec_bar.update(exec_steps - last_exec_steps)
            last_exec_steps = exec_steps
        elif exec_steps < last_exec_steps:
            last_exec_steps = exec_steps
            exec_bar.n = min(exec_bar.n, exec_steps)
            exec_bar.refresh()
        done_bar.set_postfix({"done": items_done, "total": total, "remaining": max(0, total - items_done)})


        # Auto-finish when store reports all done
        if total > 0 and items_done >= total and store.all_done(max_fix):
            break

    llm_bar.close()
    exec_bar.close()
    done_bar.close()

class PostgresKernelStore:
    """
    Postgres-backed kernel state store (safe for multi-host concurrency).
    DSN via env var PG_DSN, e.g. postgresql://user:pass@host:5432/db
    """
    def __init__(self):
        self.dsn = postgres_dsn or os.environ.get("PG_DSN", "")
        if not self.dsn:
            raise ValueError("PG_DSN is required for postgres backend.")
        self.table = f"{upgrade_type}_{run_on}_{server_node}_{llm_model}" if run_on == 'cpu' else f"{upgrade_type}_{run_on}_{llm_model}"
        self._init_db()

    def _connect(self):
        conn = psycopg2.connect(self.dsn)
        conn.autocommit = False
        return conn

    def _init_db(self):
        with self._connect() as conn:
            with conn.cursor() as cur:
                # Serialize DDL to avoid pg_type duplicate errors
                cur.execute("SELECT pg_advisory_lock(hashtext(%s))", (f"init:{self.table}",))
                try:
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS "{self.table}" (
                            key TEXT PRIMARY KEY,
                            state TEXT NOT NULL,
                            fix INTEGER,
                            ready INTEGER,
                            executed INTEGER,
                            failed INTEGER,
                            failure TEXT,
                            replicable INTEGER,
                            locked INTEGER,
                            locked_by TEXT,
                            locked_at REAL
                        )
                    """)
                finally:
                        cur.execute("SELECT pg_advisory_unlock(hashtext(%s))", (f"init:{self.table}",))
            conn.commit()

    def _state_to_cols(self, v: dict) -> dict:
        return {
            "fix": int(v.get("fix", 0) or 0),
            "ready": int(v.get("ready", 0) or 0),
            "executed": int(v.get("executed", 0) or 0),
            "failed": int(v.get("failed", 0) or 0),
            "failure": v.get("failure", None),
            "replicable": int(v.get("replicable", 0) or 0),
            "locked": int(v.get("locked", 0) or 0),
            "locked_by": v.get("locked_by", None),
            "locked_at": v.get("locked_at", None),
        }

    def _latest_score(self, v: dict) -> float | None:
        up = v.get("upgrade", {})
        if isinstance(up, dict) and up:
            try:
                max_key = max(up.keys(), key=lambda k: int(k))
            except Exception:
                max_key = sorted(up.keys())[-1]
            s = up.get(max_key, {}).get("score", None)
            return s if isinstance(s, (int, float)) else None
        s0 = v.get("score", None)
        return s0 if isinstance(s0, (int, float)) else None

    def _is_item_done(self, v: dict, max_fix: int) -> bool:
        # Terminal failure (e.g., prompt too long / context limit): treat as done
        if int(v.get("failed", 0) or 0) == 1:
            return True

        target = v.get("target", None)
        current = self._latest_score(v)

        # Stop early if replicable
        if is_replicable(target, current) or int(v.get("fix", 0)) > max_fix:
            return True
        
        # Otherwise must finish max_fix, and the last fix has been executed
        fix = int(v.get("fix", 0))
        ready = int(v.get("ready", 0))
        executed = int(v.get("executed", 0))
        if fix >= max_fix:
            return (executed == 1) and (ready == 0)
        return False

    def _load_state(self, key: str) -> dict:
        """Load full JSON state for a given key."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT state FROM \"{self.table}\" WHERE key = %s", (key,))
                row = cur.fetchone()
            if not row:
                return {}
            try:
                return json.loads(row[0])
            except Exception:
                return {}

    def _save_state(self, key: str, v: dict, conn) -> None:
        """Save full JSON state for a given key.
                CREATE TABLE IF NOT EXISTS file_gpu_gpt5.2 (
                    key TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    fix INTEGER,
                    ready INTEGER,
                    executed INTEGER,
                    failed INTEGER,
                    failure TEXT,
                    replicable INTEGER,
                    locked INTEGER,
                    locked_by TEXT,
                    locked_at REAL
                )
        """
        cols = self._state_to_cols(v)
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO "{self.table}"(key, state, fix, ready, executed, failed, failure, replicable, locked, locked_by, locked_at)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (key) DO UPDATE SET
                    state=EXCLUDED.state,
                    fix=EXCLUDED.fix,
                    ready=EXCLUDED.ready,
                    executed=EXCLUDED.executed,
                    failed=EXCLUDED.failed,
                    failure=EXCLUDED.failure,
                    replicable=EXCLUDED.replicable,
                    locked=EXCLUDED.locked,
                    locked_by=EXCLUDED.locked_by,
                    locked_at=EXCLUDED.locked_at
            """, (
                key,
                json.dumps(v, ensure_ascii=False),
                cols["fix"], cols["ready"], cols["executed"], cols["failed"], cols["failure"], cols["replicable"],
                cols["locked"], cols["locked_by"], cols["locked_at"]
            ))

    def _iter_states(self, conn, itersize: int = 500):
        """Stream state rows to avoid fetchall() for large tables."""
        with conn.cursor(name="state_stream") as cur:
            cur.itersize = itersize
            cur.execute(f'SELECT state FROM "{self.table}"')
            for (state,) in cur:
                yield state

    def all_done(self, max_fix: int) -> bool:
        """
        Done means every item is either:
          - terminal failed (failed==1), OR
          - _is_item_done(v, max_fix) == True

        - every item reached fix >= max_fix
        - and there is no pending execution of the last produced fix (ready must be 0)
        - and the last produced fix has been executed at least once (executed==1)

        Note: execution failure at fix==max_fix is still considered "executed".
        """
        with self._connect() as conn:
            for state in self._iter_states(conn):
                try:
                    v = json.loads(state)
                except Exception:
                    continue
                if int(v.get("failed", 0) or 0) != 1 and not self._is_item_done(v, max_fix):
                    return False
            return True

    def get_step_progress(self, max_fix: int) -> tuple[int, int, int, int]:
        """
        Returns (llm_steps_done, exec_steps_done, items_done, total_items).

        Rule: if _is_item_done(v,max_fix) is True (including replicable early-stop),
        count the item as fully completed (== max_fix steps) for both llm+exec bars.
        """
        with self._connect() as conn:
            total = 0
            llm_steps = 0
            exec_steps = 0
            items_done = 0

            for state in self._iter_states(conn):
                total += 1
                try:
                    v = json.loads(state)
                except Exception:
                    continue

                done = self._is_item_done(v, max_fix)
                fix = int(v.get("fix", 0) or 0)
                ready = int(v.get("ready", 0) or 0)
                executed = int(v.get("executed", 0) or 0)

                if done:
                    items_done += 1
                    llm_steps += max_fix
                    exec_steps += max_fix
                    continue

                # LLM step progress = number of fixes produced so far (cap at max_fix)
                llm_steps += min(max_fix, max(0, fix))

                # Exec step progress: each produced fix should be executed once.
                # If current fix is not executed yet (executed==0), last executed is fix-1.
                if fix <= 0:
                    exec_i = 0
                else:
                    exec_i = fix if (executed == 1 and ready == 0) else max(0, fix - 1)
                exec_steps += min(max_fix, exec_i)
            return llm_steps, exec_steps, items_done, total

    def claim_for_llm(self, max_fix: int = 16) -> tuple[str, dict] | None:
        """
        Claim for LLM if:
          - fix < max_fix
          - ready == 0
          - not locked
          - and (fix == 0 OR executed == 1)  <-- prevents LLM making fix#2 before fix#1 ran
          - and failed != 1
          - replicable = False or missing
        """
        while True:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT key, state FROM "{self.table}"
                        WHERE failed=0 AND locked=0 AND ready=0
                        AND (replicable=0 OR replicable IS NULL) AND fix < %s
                        AND (fix=0 OR executed=1)
                        ORDER BY fix ASC
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    """, (max_fix,))
                    row = cur.fetchone()

                    if not row:
                        conn.commit()
                        return None

                    key, state = row
                    v = json.loads(state)

                    if self._is_item_done(v, max_fix):
                        v["replicable"] = True
                        self._save_state(key, v, conn)
                        conn.commit()
                        continue

                    v["locked"] = 1
                    v["locked_by"] = "llm"
                    v["locked_at"] = time.time()
                    v["ready"] = 0

                    self._save_state(key, v, conn)
                    conn.commit()
                    return key, v

    def claim_for_exec(self, max_fix: int = 16) -> tuple[str, dict] | None:
        """
        Execute fixes up to and including max_fix (fix==16 must run).
        Claim if: fix <= max_fix AND ready == 1 AND executed == 0 AND not locked AND failed != 1.
        """
        while True:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT key, state FROM "{self.table}"
                        WHERE failed=0 AND locked=0 AND ready=1 AND executed=0
                        AND (replicable=0 OR replicable IS NULL) AND fix <= %s
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    """, (max_fix,))
                    row = cur.fetchone()

                    if not row:
                        conn.commit()
                        return None

                    key, state = row
                    v = json.loads(state)

                    if self._is_item_done(v, max_fix):
                        v["replicable"] = True
                        self._save_state(key, v, conn)
                        conn.commit()
                        continue

                    v["locked"] = 1
                    v["locked_by"] = "exec"
                    v["locked_at"] = time.time()
                    v["ready"] = 0

                    self._save_state(key, v, conn)
                    conn.commit()
                    return key, v

    def mark_failed(self, key: str, err_text: str, LLM_res=None) -> None:
        """Terminally fail an item so it won't be retried/claimed again."""
        with self._connect() as conn:
            v = self._load_state(key) or {}

            v["failed"] = 1
            v["failure"] = err_text
            v["failure_LLM_response"] = _to_jsonable(LLM_res)

            v["ready"] = 0
            v["locked"] = 0
            v["locked_by"] = None

            self._save_state(key, v, conn)
            conn.commit()

    def mark_retryable_error(self, key: str) -> None:
        """Non-deterministic error: unlock item, allow retry."""
        with self._connect() as conn:
            v = self._load_state(key) or {}

            # Unlock, so it can be claimed again
            v["locked"] = 0
            v["locked_by"] = None
            v.pop("locked_at", None)

            # Keep it eligible for LLM retry (ready must be 0 for claim_for_llm)
            v["ready"] = 0
            self._save_state(key, v, conn)
            conn.commit()

    def complete_llm(self, key: str, new_fix: int, nl_plan: str, llm_code: str, req_time, in_tok_count, out_tok_count, cached_tokens, model_name, completion, err_cell_num: int | None = None, raw: str | None = None, max_fix: int = 10) -> None:
        """After fixing: 
            - fix = new_fix
            - ready = 1 (eligible for execution)
            - executed = 0 (must execute this fix)
            - upgrade[str(new_fix)] gets response
        """
        with self._connect() as conn:
            v = self._load_state(key) or {}
            v["fix"] = int(new_fix)
            v["ready"] = 1
            v["executed"] = 0

            up = v.get("upgrade")
            if not isinstance(up, dict):
                up = {}
            entry = up.get(str(new_fix), {})
            if not isinstance(entry, dict):
                entry = {}

            entry["plan"] = nl_plan
            entry["code"] = llm_code
            entry["request_time"] = req_time
            entry["in_tokens"] = in_tok_count
            entry["out_tokens"] = out_tok_count
            entry["cached_tokens"] = cached_tokens

            entry["model"] = model_name
            entry["full_message"] = _to_jsonable(completion)
            if raw is not None:
                entry["fixed_path"] = raw
                
            up[str(new_fix)] = entry
            v["upgrade"] = up

            v["locked"] = 0
            v["locked_by"] = None

            self._save_state(key, v, conn)
            conn.commit()

    def complete_exec(self, key: str, exec_result: dict) -> None:
        """
        After execution:
          - store results into upgrade[str(fix)]
          - ready = 0 (eligible for next LLM fix, if fix<max_fix)
          - executed = 1
        Always unlock.
        """
        with self._connect() as conn:
            v = self._load_state(key) or {}
            fix = int(v.get("fix", 0))

            up = v.get("upgrade")
            if not isinstance(up, dict):
                up = {}
            entry = up.get(str(fix), {})
            if not isinstance(entry, dict):
                entry = {}

            payload = exec_result.get(key) if isinstance(exec_result, dict) and key in exec_result else exec_result
            if isinstance(payload, dict):
                entry.update(payload)

            up[str(fix)] = entry
            v["upgrade"] = up

            if is_replicable(v.get("target", None), payload.get("score", None)) and "output" in payload:
                shutil.copy2(payload.get("output"), work_dir / 'csv_output')
                v["replicable"] = True

            v["ready"] = 0
            v["executed"] = 1
            v["locked"] = 0
            v["locked_by"] = None

            self._save_state(key, v, conn)
            conn.commit()

class RedisKernelStore:
    """
    Redis-backed kernel state store (fast for high-frequency point reads/writes).
    Requires redis-py and a REDIS_URL env var, e.g. redis://localhost:6379/0
    """
    def __init__(self):
        self.url = postgres_dsn or os.environ.get("REDIS_URL", "")
        self.r = redis.Redis.from_url(
            self.url, 
            decode_responses=True,
            socket_keepalive=True,
            health_check_interval=30,
            retry_on_timeout=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        self.prefix = f"kernel:{upgrade_type}:{run_on}:{server_node}:{llm_model}"
        self.keys_set = f"{self.prefix}:keys"

    def _reconnect(self):
        self.r = redis.Redis.from_url(
            self.url,
            decode_responses=True,
            socket_keepalive=True,
            health_check_interval=30,
            retry_on_timeout=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )

    def _retry(self, fn, *args, **kwargs):
        last_err = None
        for attempt in range(8):
            try:
                return fn(*args, **kwargs)
            except (ConnectionError, TimeoutError) as e:
                last_err = e
                self._reconnect()
                time.sleep(min(3.25, 0.5 * (1.25 ** attempt))) 
        raise last_err

    def _key(self, k: str) -> str:
        return f"{self.prefix}:{k}"

    def _load_state(self, key: str) -> dict:
        # raw =  self._retry(self.r.hget, self._key(key), "state")
        raw = self.r.hget(self._key(key), "state")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _save_state(self, key: str, v: dict) -> None:
        self._retry(self.r.sadd, self.keys_set, key)
        # if not self.r.exists(self._key(key)):
        #     return
        mapping = {
            "state": json.dumps(v, ensure_ascii=False),
            "fix": int(v.get("fix", 0) or 0),
            "ready": int(v.get("ready", 0) or 0),
            "executed": int(v.get("executed", 0) or 0),
            "failed": int(v.get("failed", 0) or 0),
            "replicable": int(v.get("replicable", 0) or 0),
            "locked": int(v.get("locked", 0) or 0),
            "locked_by": v.get("locked_by", "") or "",
            "locked_at": v.get("locked_at", "") or "",
            "failure": v.get("failure", "") or "",
        }
        self._retry(self.r.hset, self._key(key), mapping=mapping)

    def _iter_keys(self):
        # keys = list(self.r.smembers(self.keys_set))
        # if keys:
        #     return keys
        pattern = f"{self.prefix}:*"
        found = []
        for k in self.r.scan_iter(match=pattern):
            if self.r.type(k) != "hash":
                continue
            # k is full key; strip prefix to get the logical key
            if k.startswith(self.prefix + ":"):
                found.append(k[len(self.prefix) + 1 :])
        return found

    def _latest_score(self, v: dict) -> float | None:
        up = v.get("upgrade", {})
        if isinstance(up, dict) and up:
            try:
                max_key = max(up.keys(), key=lambda k: int(k))
            except Exception:
                max_key = sorted(up.keys())[-1]
            s = up.get(max_key, {}).get("score", None)
            return s if isinstance(s, (int, float)) else None
        s0 = v.get("score", None)
        return s0 if isinstance(s0, (int, float)) else None

    def _is_item_done(self, v: dict, max_fix: int) -> bool:
        if int(v.get("failed", 0) or 0) == 1:
            return True

        target = v.get("target", None)
        current = self._latest_score(v)

        if is_replicable(target, current) or int(v.get("fix", 0)) > max_fix:
            return True

        fix = int(v.get("fix", 0))
        ready = int(v.get("ready", 0))
        executed = int(v.get("executed", 0))
        if fix >= max_fix:
            return (executed == 1) and (ready == 0)

        return False

    def all_done(self, max_fix: int) -> bool:
        for k in self._iter_keys():
            v = self._load_state(k)
            
            if int(v.get("failed", 0) or 0) != 1 and not self._is_item_done(v, max_fix):
                return False
        return True

    def get_step_progress(self, max_fix: int) -> tuple[int, int, int, int]:
        total = 0
        llm_steps = 0
        exec_steps = 0
        items_done = 0
        for k in self._iter_keys():
            total += 1
            v = self._load_state(k)

            done = self._is_item_done(v, max_fix)
            fix = int(v.get("fix", 0) or 0)
            ready = int(v.get("ready", 0) or 0)
            executed = int(v.get("executed", 0) or 0)

            if done:
                items_done += 1
                llm_steps += max_fix
                exec_steps += max_fix
                continue

            llm_steps += min(max_fix, max(0, fix))

            if fix <= 0:
                exec_i = 0
            else:
                exec_i = fix if (executed == 1 and ready == 0) else max(0, fix - 1)
            exec_steps += min(max_fix, exec_i)

        return llm_steps, exec_steps, items_done, total

    def _claim(self, kind: str, max_fix: int) -> tuple[str, dict] | None:
        # kind: "llm" or "exec"
        for key in self._iter_keys():
            rkey = self._key(key)
            with self.r.pipeline() as pipe:
                try:
                    pipe.watch(rkey)
                    v = self._load_state(key)
                    if not v:
                        pipe.unwatch()
                        continue

                    locked = int(v.get("locked", 0))
                    # if locked != 0 and _is_lock_stale(v.get("locked_at", 0)):
                    #     if v['locked_by'] == 'llm':
                    #         v['ready'] = 0
                    #         v['executed'] = 1
                    #     else:
                    #         v['ready'] = 1
                    #         v['executed'] = 0
                    #     v["locked"] = 0
                    #     v["locked_by"] = None
                    #     v.pop("locked_at", None)
                    #     pipe.multi()
                    #     pipe.hset(rkey, mapping={
                    #         "state": json.dumps(v, ensure_ascii=False),
                    #         "fix": int(v.get("fix", 0) or 0),
                    #         "ready": int(v.get("ready", 0) or 0),
                    #         "executed": int(v.get("executed", 0) or 0),
                    #         "failed": int(v.get("failed", 0) or 0),
                    #         "replicable": int(v.get("replicable", 0) or 0),
                    #         "locked": 0,
                    #         "locked_by": "",
                    #         "locked_at": "",
                    #         "failure": v.get("failure", "") or "",
                    #     })
                    #     pipe.execute()
                    #     continue
                    if locked == 1:
                        pipe.unwatch()
                        continue
                    if int(v.get("failed", 0) or 0) == 1:
                        pipe.unwatch()
                        continue
                    if self._is_item_done(v, max_fix):
                        pipe.unwatch()
                        continue

                    fix = int(v.get("fix", 0))
                    ready = int(v.get("ready", 0))
                    executed = int(v.get("executed", 0))

                    if kind == "llm":
                        if fix >= max_fix:
                            pipe.unwatch()
                            continue
                        if ready != 0 or locked != 0:
                            pipe.unwatch()
                            continue
                        if fix > 0 and executed != 1:
                            pipe.unwatch()
                            continue

                        v["locked"] = 1
                        v["locked_by"] = "llm"
                        v["locked_at"] = time.time()
                        v["ready"] = 0
                    else:
                        if not (fix <= max_fix and ready == 1 and executed == 0 and locked == 0):
                            pipe.unwatch()
                            continue
                        if fix == 0:
                            continue

                        v["locked"] = 1
                        v["locked_by"] = "exec"
                        v["locked_at"] = time.time()
                        v["ready"] = 0

                    pipe.multi()
                    pipe.hset(rkey, mapping={
                        "state": json.dumps(v, ensure_ascii=False),
                        "fix": int(v.get("fix", 0) or 0),
                        "ready": int(v.get("ready", 0) or 0),
                        "executed": int(v.get("executed", 0) or 0),
                        "failed": int(v.get("failed", 0) or 0),
                        "replicable": int(v.get("replicable", 0) or 0),
                        "locked": int(v.get("locked", 0) or 0),
                        "locked_by": v.get("locked_by", "") or "",
                        "locked_at": v.get("locked_at", "") or "",
                        "failure": v.get("failure", "") or "",
                    })
                    pipe.execute()
                    return key, v
                except redis.WatchError:
                    print(f"WatchError: concurrent modification on {key}")
                    continue
        return None

    def claim_for_llm(self, max_fix: int = 16) -> tuple[str, dict] | None:
        return self._claim("llm", max_fix)

    def claim_for_exec(self, max_fix: int = 16) -> tuple[str, dict] | None:
        return self._claim("exec", max_fix)

    def mark_failed(self, key: str, err_text: str, LLM_res=None) -> None:
        v = self._load_state(key) or {}
        v["failed"] = 1
        v["failure"] = err_text
        v["failure_LLM_response"] = _to_jsonable(LLM_res)
        v["ready"] = 0
        v["locked"] = 0
        v["locked_by"] = None
        self._save_state(key, v)

    def mark_retryable_error(self, key: str) -> None:
        v = self._load_state(key) or {}
        v["locked"] = 0
        v["locked_by"] = None
        v.pop("locked_at", None)
        v["ready"] = 0
        self._save_state(key, v)
    
    def mark_reexec_error(self, key: str) -> None:
        v = self._load_state(key) or {}
        # up = v['upgrade']
        # fix_v = v['fix']
        # code = up[str(fix_v)]['code']
        # nb = _llm_code_to_ipynb(code)
        # with open(f"/home/b27jin/CodeModernization/results/upgrade/{upgrade_type}/script_out_allINone/fix_{fix_v}/{key}", "w", encoding="utf-8") as file:
        #     nbformat.write(nb, file)

        v['ready'] = 0
        v['executed'] = 1
        v['fix'] = v['fix'] - 1
        up = v.get('upgrade', {})
        keys_to_remove = [k for k in up if int(k) > v['fix']]
        for k in keys_to_remove:
            del up[k]
        
        if "error" in up[str(v['fix'])]:
            del up[str(v['fix'])]["error"]
        v['upgrade'] = up

        v["locked"] = 0
        v["locked_by"] = None
        v.pop("locked_at", None)
        self._save_state(key, v)

    def complete_llm(self, key: str, new_fix: int, nl_plan: str, llm_code: str, req_time, in_tok_count, out_tok_count, cached_tokens, model_name, completion, err_cell_num: int | None = None, raw: str | None = None, max_fix: int = 10) -> None:
        v = self._load_state(key) or {}

        v["fix"] = int(new_fix)
        v["ready"] = 1
        v["executed"] = 0

        up = v.get("upgrade")
        if not isinstance(up, dict):
            up = {}
        entry = up.get(str(new_fix), {})
        if not isinstance(entry, dict):
            entry = {}

        entry["plan"] = nl_plan
        entry["code"] = llm_code
        entry["request_time"] = req_time
        entry["in_tokens"] = in_tok_count
        entry["out_tokens"] = out_tok_count

        entry["cached_tokens"] = cached_tokens
        entry["model"] = model_name

        entry["full_message"] = _to_jsonable(completion)
        if raw is not None:
            entry["fixed_path"] = raw

        up[str(new_fix)] = entry
        v["upgrade"] = up

        v["locked"] = 0
        v["locked_by"] = None
        self._save_state(key, v)

    def complete_exec(self, key: str, exec_result: dict) -> None:
        v = self._load_state(key) or {}
        fix = int(v.get("fix", 0))

        up = v.get("upgrade")
        if not isinstance(up, dict):
            up = {}
        entry = up.get(str(fix), {})
        if not isinstance(entry, dict):
            entry = {}

        payload = exec_result.get(key) if isinstance(exec_result, dict) and key in exec_result else exec_result
        if isinstance(payload, dict):
            entry.update(payload)

        up[str(fix)] = entry
        v["upgrade"] = up

        if is_replicable(v.get("target", None), payload.get("score", None)) and "output" in payload:
            shutil.copy2(payload.get("output"), work_dir / 'csv_output')
            v["replicable"] = True

        v["ready"] = 0
        v["executed"] = 1

        v["locked"] = 0
        v["locked_by"] = None
        self._save_state(key, v)

def get_store(backend: str, store_path: str):
    if backend == "redis":
        return RedisKernelStore()
    if backend == "postgres":
        return PostgresKernelStore()
    return KernelStore(Path(store_path))

def init_kernel_state(state_path: Path, candidates: list[str], kernel_meta: dict | None = None) -> None:
    """
    Create/extend the kernel state JSON so KernelStore can manage it:
    
    Minimal per item:
      {
        "fix": 0,
        "ready": 0,
        "executed": 0,
        "upgrade": {},
        "failed": 0,
        "failure": null
      }
    
    We also preserve any existing fields like "score", "target", "api".
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        try:
            existing = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except Exception:
            existing = {}
    else:
        existing = {}

    for k in candidates:
        v = existing.get(k, {})
        if not isinstance(v, dict):
            v = {}
        v.setdefault("fix", 0)
        v.setdefault("ready", 0)      # queued for LLM
        v.setdefault("executed", 0)   # executed current fix?
        v.setdefault("upgrade", {})   # per-fix history
        v.setdefault("failed", 0)    # terminal failure
        v.setdefault("failure", None) # failure reason
        existing[k] = v

    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, state_path)

def init_kernel_state_postgres(candidates: list[str]) -> None:
    """
    Create/extend the kernel state JSON so KernelStore can manage it:
    
    Minimal per item:
      {
        "fix": 0,
        "ready": 0,
        "executed": 0,
        "upgrade": {},
        "failed": 0,
        "failure": null
      }
    
    We also preserve any existing fields like "score", "target", "api".
    """
    store = PostgresKernelStore()
    with store._connect() as conn:
        for k in candidates:
            with conn.cursor() as cur:
                cur.execute(f"SELECT 1 FROM \"{store.table}\" WHERE key = %s", (k,))
                if cur.fetchone():
                    continue
            v = {}
            v.setdefault("fix", 0)
            v.setdefault("ready", 0)
            v.setdefault("executed", 0)
            v.setdefault("upgrade", {})
            v.setdefault("failed", 0)
            v.setdefault("failure", None)
            v.setdefault("replicable", None)

            meta = candidates.get(k, {})
            if isinstance(meta, dict):
                for mk, mv in meta.items():
                    v.setdefault(mk, mv)

            store._save_state(k, v, conn)
        conn.commit()

def init_kernel_state_redis(candidates: dict) -> None:
    store = RedisKernelStore()
    for k, meta in candidates.items():
        v = {}
        v.setdefault("fix", 0)
        v.setdefault("ready", 0)
        v.setdefault("executed", 0)
        v.setdefault("upgrade", {})
        v.setdefault("failed", 0)
        v.setdefault("failure", None)
        v.setdefault("replicable", None)
        if isinstance(meta, dict):
            for mk, mv in meta.items():
                v.setdefault(mk, mv)
        
        store._save_state(k, v)
        # mapping = {
        #     "state": json.dumps(v, ensure_ascii=False),
        #     "fix": int(v.get("fix", 0) or 0),
        #     "ready": int(v.get("ready", 0) or 0),
        #     "executed": int(v.get("executed", 0) or 0),
        #     "failed": int(v.get("failed", 0) or 0),
        #     "replicable": int(v.get("replicable", 0) or 0),
        #     "locked": int(v.get("locked", 0) or 0),
        #     "locked_by": v.get("locked_by", "") or "",
        #     "locked_at": v.get("locked_at", "") or "",
        #     "failure": v.get("failure", "") or "",
        # }
        # store.r.hset(store._key(k), mapping=mapping)

def migrate_json_to_postgres(json_path: Path) -> None:
    if not json_path.exists():
        return
    data = json.loads(json_path.read_text(encoding="utf-8"))
    store = PostgresKernelStore()

    now = time.time()
    print(f"Migrating kernel state from {json_path} to Postgres...")
    with store._connect() as conn:
        for k, v in data.items():
            if not isinstance(v, dict):
                continue
            store._save_state(k, v, conn)
        conn.commit()
    print("Migration complete in {:.2f}s.".format(time.time() - now))

def migrate_json_to_redis(json_path: Path) -> None:
    if not json_path.exists():
        return
    data = json.loads(json_path.read_text(encoding="utf-8"))
    store = RedisKernelStore()

    now = time.time()
    print(f"Migrating kernel state from {json_path} to Redis...")
    for k, v in data.items():
        if not isinstance(v, dict):
            continue
        store._save_state(k, v)
    print("Migration complete in {:.2f}s.".format(time.time() - now))

def _has_cell_headers(llm_code: str) -> bool:
    """Return True if llm_code contains at least one '## === cell n' header."""
    code = llm_code.strip()
    if code.startswith("```") and code.endswith("```"):
        code = code[3:-3].strip()

    header_re = re.compile(r'^\s*##\s*===\s*cell\s*\d+\s*$', re.IGNORECASE)
    return any(header_re.match(line.strip()) for line in code.splitlines(True))

def _llm_code_to_ipynb(llm_code: str) -> dict:
    """
    Convert LLM code with sections:
      ## === cell 1
      <code>
      ## === cell 2
      <code>
    into a Jupyter notebook with one code cell per section.

    If no '## === cell n' markers are present, create a single cell with all code.
    """
    # Strip optional fenced code blocks
    code = llm_code.strip()
    if code.startswith("```") and code.endswith("```"):
        code = code[3:-3].strip()

    lines = code.splitlines(True)
    # Match lines like '## === cell 1' (case-insensitive, allow spaces)
    header_re = re.compile(r'^\s*##\s*===\s*cell\s*\d+\s*$', re.IGNORECASE)

    # if upgrade_type == "cell":
    #     current: list[str] = []
    #     fix_return = 0

    #     for line in lines:
    #         if header_re.match(line.strip()):
    #             fix_return += 1
    #             if current:
    #                 current = []
    #             continue  # do not include the header line
    #         current.append(line)
    #     if fix_return > 1:
    #         return ""
    #     else:
    #         return "".join(current)
    # else:
    current: list[str] = []
    nb = new_notebook()

    for line in lines:
        if header_re.match(line.strip()):
            if current:
                nb.cells.append(new_code_cell("".join(current)))
                current = []
            continue  # do not include the header line
        current.append(line)

    if current:
        nb.cells.append(new_code_cell("".join(current)))

    return nb

def _strip_cell_header_lines(text: str) -> str:
    """Remove lines like '## === cell 3' from LLM output."""
    kept: list[str] = []
    # Match lines like '## === cell 1' (case-insensitive, allow spaces)
    header_re = re.compile(r'^\s*##\s*===\s*cell\s*\d+\s*$', re.IGNORECASE)
    for line in text.splitlines(True):  # keep newlines
        if header_re.match(line.strip()):
            continue
        kept.append(line)
    return "".join(kept)

def _drop_hash_comment_lines(src: str) -> str:
    """
    Remove lines that are commented out, e.g. starting with '#'
    (also removes ones with leading whitespace before '#').
    Keeps inline comments like: x = 1  # keep
    """
    kept: list[str] = []
    in_triple = None  # None, "'''", or '"""'

    for line in src.splitlines(True):  # keep newlines
        # naive triple-quote toggle (won't handle all edge cases, but avoids the common ones)
        if in_triple is None:
            if "'''" in line or '"""' in line or '```' in line:
                # enter triple-quote mode if it starts an odd-count triple quote on this line
                if line.count("'''") % 2 == 1:
                    in_triple = "'''"
                elif line.count('"""') % 2 == 1:
                    in_triple = '"""'
        else:
            if line.count(in_triple) % 2 == 1:
                in_triple = None

        if in_triple is None and line.lstrip().startswith("#"):
            continue
        kept.append(line)

    return "".join(kept)

def write_fixed_artifact(filename: str, llm_model: str, sys_prompt: str, usr_prompt: str, fixed_root: Path, fix_num: int, upgrade_type: str, err_cell_num: int, timeout: int, has_err: bool):
    """
    Save fixed version under fix_{fix_num}/.
    """
    out_path = fixed_root / 'script_out_allINone'/ f"fix_{fix_num}"
    out_path.mkdir(parents=True, exist_ok=True)

    for attempt in range(10):  # Retry up to 10 times if LLM output is invalid
        # print(f"Attempt {attempt+1}")
        try:
            nl_plan, llm_code, req_time, in_tok_count, out_tok_count, cached_tokens, model_name, completion = plan_and_code_query(llm_model, sys_prompt, usr_prompt)
        except TypeError:
            continue
        # print(f"LLM request time: {req_time:.2f}s for {filename} fix #{fix_num}")

        if not _has_cell_headers(llm_code):
            continue

        if upgrade_type == "file" or timeout > 600 or not has_err:
            # if progress_q is not None:
            #     progress_q.put(("log", f"fix #{fix_num} {filename}: {timeout=} and {has_err=} "))
            nb = _llm_code_to_ipynb(llm_code)
            # Type check: must be a notebook-like object
            cells = nb["cells"] if isinstance(nb, dict) else nb.cells
            if len(cells) == 1 and filename not in file_w_one_cell:  # If it is invalid response
                continue

            with open(out_path/filename, "w", encoding="utf-8") as file:
                nbformat.write(nb, file)

            return nl_plan, llm_code, req_time, in_tok_count, out_tok_count, cached_tokens, completion, (out_path/filename), model_name

        if upgrade_type == "cell":
            if fix_num == 1:
                file_path = Path("./results/baseline/script_out_allINone") / filename if run_type == "baseline" else Path("./results/downgrade/baseline/script_out_allINone") / filename
            else:
                file_path = fixed_root / 'script_out_allINone' / f"fix_{fix_num-1}" / filename

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                        nb = json.load(f)
                
                for cell in nb['cells']:
                    if cell.get('cell_type') == 'code':
                        cell['outputs'] = []
                        cell['execution_count'] = None

                # Remove any "## === cell N" lines from LLM output before inserting into a single cell
                llm_code_clean = _strip_cell_header_lines(llm_code)
                # Type check: one cell fix
                if len(llm_code_clean) == 0:
                    continue
                nb["cells"][err_cell_num]["source"] = llm_code_clean.splitlines(True)
                        
                (out_path/filename).write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")

                return nl_plan, llm_code, req_time, in_tok_count, out_tok_count, cached_tokens, completion, (out_path/filename), model_name
            except Exception:
                pass
    raise ValueError(f"Failed to get valid LLM output for {filename} after 10 attempts.")

def shift_md_headings_down_one_level(md: str) -> str:
    """
    Turn:
      # H1 -> ## H1
      ## H2 -> ### H2
    (caps at ######), and does not modify fenced code blocks.
    """
    out_lines: list[str] = []
    in_fence = False

    fence_re = re.compile(r"^\s*```")                # ``` or ```python etc.
    heading_re = re.compile(r"^(\s*)(#{1,6})(\s+.*)$")  # headings with a space + text

    for line in md.splitlines(True):  # keep line endings
        if fence_re.match(line):
            in_fence = not in_fence
            out_lines.append(line)
            continue

        if not in_fence:
            m = heading_re.match(line)
            if m:
                indent, hashes, rest = m.groups()
                new_hashes = "#" * min(6, len(hashes) + 1)
                out_lines.append(f"{indent}{new_hashes}{rest}")
                continue

        out_lines.append(line)

    return "".join(out_lines)

def is_replicable(reported_score,measured_score):

    if isinstance(reported_score, float) and isinstance(measured_score, float) and reported_score != 0.0:
        thrus = abs(measured_score-reported_score)/abs(reported_score) * 100
    else:
        thrus = None

    return (thrus is not None) and (thrus <= float(args.threshold))

def _make_llm_prompts(filename: str, state: dict, upgrade_type: str, fix_num: int) -> tuple[str, str]:
    
    if fix_num == 1:
        file_path = Path("./results/baseline/script_out_allINone") / filename if run_type == "baseline" else Path("./results/downgrade/baseline/script_out_allINone") / filename
    else:
        file_path = work_dir / 'script_out_allINone' / f"fix_{fix_num-1}" / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Previous fix file not found: {file_path}")
    # Extract competition name from filename
    compt = filename.split("_")[0]

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load notebook from {file_path}: {e}")

    # Read task description
    desc_path = Path(f"./upgrade/competitions/{compt}.md")
    task_desc = desc_path.read_text(encoding="utf-8")
    task_desc = shift_md_headings_down_one_level(task_desc)

    # Get installed packages in the environment from state
    installed_packages = state.get("installed_packages", "No external packages required in the script and installed.")
    installed_packages = "No external packages required in the script and installed." if not installed_packages else installed_packages

    # Get data preview from state
    data_preview_path = Path(f"./upgrade/data_previews/{compt}.txt")
    data_preview = data_preview_path.read_text(encoding="utf-8")

    # Get target and current accuracy from state
    target_score = state.get("target", None)
    current_score = "Not yielded"
    grading_error = None
    up = state.get("upgrade", {})
    if isinstance(up, dict) and fix_num > 1:
        last_fix = up.get(str(fix_num - 1), {})
        if isinstance(last_fix, dict):
            if 'error' in last_fix and "bash: line 1:     9 Killed" in last_fix['error']:
                raise ValueError(f"Failed to load notebook from {file_path}")
            current_score = last_fix.get("score", "Not yielded")
            if isinstance(current_score, str) and current_score != "Not yielded":
                grading_error = current_score
                current_score = "Not yielded"
    else:
        if "score" in state:
            if isinstance(state["score"], float):
                current_score = state["score"]
            if isinstance(state["score"], str):
                grading_error = state["score"]

    # Get last error
    last_err = ""
    err_cell_num = None
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            for output in cell.get('outputs', []):
                if output.get('output_type') == 'error':
                    last_err = "\n".join(output.get('traceback', []))
                    err_cell_num = nb['cells'].index(cell)
                    break
            if last_err or err_cell_num is not None:
                break

    # Is the score higher the better?
    higher_better = not lower_better.get(compt, False)

    # Previous plans
    plans = []
    for fix in sorted(up.keys(), key=lambda k: int(k)):
        achieved_score = up[fix].get("score", None)
        plan = up[fix].get('plan', None)
        if isinstance(achieved_score, float) and plan:
            pattern = '|'.join(re.escape(p) for p in patterns)
            improve_plan = re.sub(pattern, '', plan)

            plans.append(
                f"- What this solution (achieved {achieved_score}) has done: '{improve_plan}'"
            )
    
    python_version = state.get("python_version", ">=3.5")

    usr_prompt: Any = {}
    code_blocks = []

    # timeout case
    timeout = 0
    if isinstance(up, dict) and fix_num > 1:
        last_fix = up.get(str(fix_num - 1), {})
        if isinstance(last_fix, dict):
            timeout = last_fix.get("execution_time", 0)

    if timeout>600:
        sys_prompt = """You are an expert performance engineer for a Kaggle competition code solution. 

# Goal
- Refactor the user-provided multi-cell Python script to run within a hard 600-second timeout while preserving result accuracy and the algorithm's core logic.

# Behavior constraints
- STRICT: Do NOT rewrite the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics.
- STRICT: Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep I/O paths unchanged; do not assume new data sources.
- Avoid non-determinism: keep seeds and determinism settings (or add them if missing) so results remain stable.
- The environment timeout is 10 minutes: prioritize optimizations that reduce asymptotic cost and unnecessary work.
- When unsure, favor changes that are provably equivalent (e.g., reordering computations, caching pure functions, vectorized equivalents, reducing repeated work) rather than risky heuristics.

# Inputs you will receive from the user
- 1. Kaggle task description (problem statement + evaluation metric + dataset description).
- 2. Python version (in the target environment).
- 3. Installed packages (in the target environment).
- 4. Data file paths (available in the environment).
- 5. Code solution (in a .py-like "cells" format, including any error tracebacks), formatted as:
   - ## === cell k for each cell
   - and optionally: ## --- ERROR in cell k, traceback: followed by the exception text

# Workflow (follow in this order every time)
- 1) Diagnose precisely the most likely timeouts/bottlenecks based on the provided code and traceback (if any).
- 2) Produce an optimized replacement script in the SAME cell-delimited format, preserving cell numbers and overall structure. Only change what is needed.
- 3) For each changed block, add a short explanation immediately above it describing why it improves runtime and why it preserves correctness.
- 4) Where relevant, propose micro-optimizations (vectorization, avoiding Python loops, caching, preallocation, avoiding repeated disk I/O, using efficient pandas/numpy ops, batch operations, avoiding repeated model fits, avoiding repeated tokenization, etc.) AND higher-level optimizations (algorithmic constant factors, pruning redundant computation, using appropriate data structures).
- 5) If the code uses multiprocessing/threads/GPU, optimize within the same paradigm; do not remove parallelism, but you may tune chunk sizes, worker counts, or data transfer patterns.
- 6) If you cannot guarantee correctness preservation for a proposed optimization, do not apply it.

# Output requirements (must follow exactly)
## Return exactly two parts in this order
- 1) A short natural-language outline (3-5 sentences) describing what you will change and why, followed by 
- 2) A single Markdown code block (wrapped in ```) containing the complete implementation in a newline.
## Hard constraints
- Do NOT include any additional headings, titles, bullet points, numbered lists, or extra commentary outside the two parts above. Just natural language text followed by a newline and then the markdown code block.
- Use exactly ONE Markdown code block total.
## Code block constraints
- The code block must contain the full solution in a "cells" format.
- Each cell header must be on its own line and must match exactly: `## === cell N`.
- Start at N = 1 and increment by 1 with no gaps (1, 2, 3, ...).
- Preserve the original cell order and include all required code needed to run end-to-end.
- Do not add any additional Markdown outside the single code block.
## Cells template (must match exactly in structure)
```
## === cell 1
{{code}}
## === cell 2
{{code}}
...
## === cell N
{{code}}
```

You must be careful, systematic, and pragmatic.
"""
        usr_prompt['Goal'] = "Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic."

        usr_prompt['Requirements'] = (
            "Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.\n",
            "Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.\n",
            "Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.\n",
            "Keep file paths unchanged.",
        )

        usr_prompt["1. Kaggle task description"] = task_desc
        usr_prompt["2. Python version"] = python_version
        usr_prompt["3. Installed packages"] = installed_packages
        usr_prompt["4. Data file paths"] = data_preview

        for i, cell in enumerate(nb.get('cells', [])):
            if cell.get('cell_type') != 'code':
                continue

            source = ''.join(cell.get('source', []))
            source = _drop_hash_comment_lines(source)
            if source.strip() == "":
                continue
            code_blocks.append(f"## === cell {i}\n{source}")

            for output in cell.get('outputs', []) or []:
                if output.get('output_type') == 'error':
                    err_msg = "\n".join(output.get('traceback', []) or [])
                    err_msg = ANSI_RE.sub("", err_msg)
                    code_blocks.append(f"## --- ERROR in cell {i}, traceback:\n{err_msg}")
                    break
        
        if isinstance(grading_error, str):
            code_blocks.append(f"## --- ERROR in outputing the csv:\n{grading_error}")

        usr_prompt["5. Code solution"] = "\n\n".join(code_blocks)

        return sys_prompt, usr_prompt, err_cell_num, timeout, True if len(last_err.strip())> 0 else False

    # no errors in the notebook or script
    if not last_err or len(last_err.strip()) == 0:
        sys_prompt = f"""You are an expert Kaggle competition performance engineer.

# Goal
- Improve the user's provided Kaggle code solution so that it moves the evaluation score toward the target score (i.e., minimize |current_score - target_score|), NOT to achieve the best possible score, while respecting Kaggle constraints.
- If the current score is already close to the target, prioritize stability and minimal changes over further optimization.

# What "toward the target score" means (score-matching objective)
- Define: gap = current_score - target_score (if current_score is provided).
- Allow a ±{args.threshold}% of tolerance band.
- If higher-is-better:
  - If gap < 0 (worse than target): increase score/performance cautiously toward target.
  - If gap > 0 (better than target): allow score/performance to decrease toward target.
- If lower-is-better:
  - If gap > 0 (worse than target): decrease score/performance toward target.
  - If gap < 0 (better than target): allow score/performance to increase toward target.
- Prefer the smallest change that is expected to reduce |gap|.

# Behavior constraints
- STRICT: Do NOT rewrite the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function unless the |current_score - target_score|/|target_score| > 30%. Maintain identical core logic and evaluation semantics.
- STRICT: Every change must be directly relevant to (a) producing a valid submission .csv when missing AND (b) moving the score closer to the target; avoid unrelated refactors or stylistic edits.
- STRICT: Do NOT optimize for the best score. Optimize for closeness to the target score (minimize absolute gap), even if that means the score decreases when the current score is better than the target.
- Treat the target score (target_score) as a destination, not a ceiling to break.
- If the current score is worse than target: make the smallest legitimate changes to improve toward the target band.
- If the current score is better than target: do NOT pursue further improvements; decrease the performance toward the target instead.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep I/O paths unchanged; do not assume new data sources.
- If information is missing, make the best safe assumption, state it explicitly, and proceed.
- Never assume unavailable files, internet access, hidden columns, or extra packages; only use the installed packages provided by the user and only the data paths the user provides.
- Do NOT manipulate the output results (e.g., hardcoding answers, leaking labels, copying public submissions); all improvements must come from legitimate code changes only.
- Do NOT include long generic tutorials; be specific to the provided code and metric.
- Do NOT output placeholders like "..." inside the final code blocks; output complete code.
- Make the code finish within a 600-second timeout.

# When adjusting score toward the target (NOT best score)
- Align the training objective and prediction post-processing with the evaluation metric, but only with minimal changes that preserve core logic.
- Avoid target leakage and ensure train/validation splitting is correct.
- Confirm submission format and sorting/index alignment.
- Prefer adjustments that change calibration/regularization/determinism/thresholding (if metric allows) over changing the modeling approach.

# When multiple bugs exist
- Fix in the order that unblocks execution earliest.
- After it runs, ensure it produces a valid submission file with suffix .csv (this is mandatory).
- After that, address correctness (metric, labels, leakage, split, format).
- Then adjust performance only insofar as it reduces |current_score - target_score| (not "best score").

# Inputs you will receive from the user
- 1. Kaggle task description (problem statement + evaluation metric + dataset description).
- 2. Python version (in the target environment).
- 3. Installed packages (in the target environment).
- 4. Data file paths (available in the environment).
- 5. Target score (target_score).
- 6. Current score (current_score where the current code solution achieved, may indicate "not yielded" because no valid submission .csv was generated).
- 7. Whether higher score is better.
- 8. Previous improvement plans (may be empty).
- 9. Code solution (in a .py-like "cells" format, including any error tracebacks), formatted as:
   - ## === cell k for each cell
   - and optionally: ## --- ERROR in cell k, traceback: followed by the exception text

# Workflow (follow in this order every time)
- 1) Parse the user message into: task/metric, data paths, used python version, installed packages, target score, current score, existing plan(s), and the full script split into cells.
- 2) Parse the cell format:
   - Cells are labeled `## === cell k`.
- 3) Determine the branch:
   - If the user provides a current Kaggle score for an existing submission:
      - Analyze what changes are likely to minimize |current_score - target_score|.
      - Use prior improvement plans (if any) to avoid repeating work.
      - Explicitly reason about direction: whether to increase/decrease score to get closer to target given "higher is better" or "lower is better".
      - Stop once the score is plausibly within the target tolerance band (do not keep improving).
   - If the user cannot provide a score because no valid .csv submission was generated (or the file is missing/invalid):
      - Diagnose why the pipeline did not produce a .csv (wrong filename/extension, wrong path, exception before write, empty dataframe, wrong variable name, permission/path issues, etc.).
      - Ensure the script writes a valid submission file with suffix .csv and correct columns.
- 4) Propose minimal, safe, incremental changes that are expected to move the score toward the target (reduce absolute gap), consistent with the competition metric and constraints.
- 5) Propose the smallest viable patch:
   - Return a minimal but sufficient diff patch that (a) makes the code run end-to-end and (b) produces a valid submission .csv.
   - Only refactor if it removes the root cause (e.g., missing CSV) or clearly reduces |gap| with minimal risk.
- 6) Validate logically:
   - Ensure the fix is consistent with the Python version and installed packages.
   - Ensure submission schema is correct: filename ends with .csv, columns match, row count matches, alignment is correct.
- 7) Implement changes:
   - Add a short explanation immediately above each change describing why it is expected to move the score closer to the target (not maximize).
   - Provide the final updated script in the "## === cell i" format.
   - Ensure it writes the submission CSV to the expected location and uses the exact required column names.

# Output requirements (must follow exactly)
## Return exactly two parts in this order
- 1) A short natural-language outline (3-5 sentences) describing what you will change and why, followed by 
- 2) A single Markdown code block (wrapped in ```) containing the complete implementation in a newline.
## Hard constraints
- Do NOT include any additional headings, titles, bullet points, numbered lists, or extra commentary outside the two parts above. Just natural language text followed by a newline and then the markdown code block.
- Use exactly ONE Markdown code block total.
## Code block constraints
- The code block must contain the full solution in a "cells" format.
- Each cell header must be on its own line and must match exactly: `## === cell N`.
- Start at N = 1 and increment by 1 with no gaps (1, 2, 3, ...).
- Preserve the original cell order and include all required code needed to run end-to-end.
- Do not add any additional Markdown outside the single code block.
## Cells template (must match exactly in structure)
```
## === cell 1
{{code}}
## === cell 2
{{code}}
...
## === cell N
{{code}}
```

You must be careful, systematic, and pragmatic.
"""
        usr_prompt['Goal'] = "I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need."

        usr_prompt["Requirements"] = (
            "Keep changes minimal unless necessary.\n",
            "Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.\n",
            "Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.\n",
            "Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.\n",
            "Ensure it runs end-to-end and produces a valid submission file.",
        )

        usr_prompt["1. Kaggle task description"] = task_desc
        usr_prompt["2. Python version"] = python_version
        usr_prompt["3. Installed packages"] = installed_packages
        usr_prompt["4. Data file paths"] = data_preview
        usr_prompt["5. Target score"] = target_score
        usr_prompt["6. Current score"] = current_score
        usr_prompt["7. Whether higher score is better"] = "Higher is better." if higher_better else "Lower is better."
        # usr_prompt["8. Previous improvement plans"] = "N/A"
        if len(plans) > 1:
            usr_prompt["8. Previous improvement plans"] = "\n".join(plans)
        elif len(plans) == 1:
            usr_prompt["8. Previous improvement plan"] = plans[0]
        else:
            usr_prompt["8. Previous improvement plan"] = "N/A"


        for i, cell in enumerate(nb.get('cells', [])):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                source = _drop_hash_comment_lines(source)
                if source.strip() == "":
                    continue
                code_blocks.append(f"## === cell {i}\n{source}")
        
        usr_prompt["9. Code solution"] = "\n\n".join(code_blocks)

    else:
        # If script/notebook has errors
        sys_prompt = """You are a Python debugger and patching assistant.

# Goal
- Fix the crash by modifying only what is necessary, focusing on the buggy cell k and, only if strictly required, earlier cells that the buggy cell depends on.
- Prioritize correctness, determinism, and minimal necessary edits. Do not redesign the entire solution unless required to make it run.

# Behavior constraints
- STRICT: Do NOT rewrite the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics.
- STRICT: Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- STRICT: Do NOT complete missing cells or add new modeling logic beyond what is required to fix the error.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep I/O paths unchanged; do not assume new data sources.
- You may read cell k+1 to ensure compatibility, but you must not implement or complete logic for cells k, k+1, or later cells.
- Keep the patch minimal and localized: only changes inside cell k.
- If information is missing, make the best safe assumption, state it explicitly, and proceed.
- Do NOT output anything unrelated to the bug fix goal.
- Do NOT include long generic tutorials; be specific to the provided code and metric.
- Never assume unavailable files, internet access, hidden columns, or extra packages; only use the installed packages provided by the user and only the data paths the user provides.
- Do NOT output placeholders like "..." inside the final code blocks; output complete code.

# When multiple bugs exist
- Fix in the order that unblocks execution earliest.
- After it runs, address correctness (metric, labels, leakage, split).
- Then optimize performance and score.

# Inputs you will receive from the user
- 1. Python version (in the target environment).
- 2. Installed packages (in the target environment).
- 3. Data file paths (available in the environment).
- 4. Code solution (in a .py-like "cells" format, including any error tracebacks), formatted as:
   - ## === cell k for each cell
   - ## --- ERROR in cell k, traceback: followed by the exception text
   - Only cells 1..k (up to the failing cell) and cell k+1 (the next cell) are provided. Assume later cells exist but are unknown.

# Workflow (follow in this order every time)
- 1) Parse the cell format:
   - Identify failing cell index k, the traceback, and any referenced lines/variables.
- 2) Diagnose precisely:
   - Identify the root cause (e.g., API method deprecation, API updates, wrong paths, and incorrect submission format, etc.), not just the symptom.
   - If multiple issues exist, fix in dependency order (imports → data IO → shapes/types → training → inference).
   - Plan each issue with: location (cell/line), cause, and fix.
- 4) Propose the smallest viable patch that fit the competition metric:
   - Return a minimal but sufficient diff patch that resolves the root cause and keeps compatibility with cell k+1.
   - Only refactor if it removes the root cause.
- 5) Validate logically:
   - Ensure the fix is consistent with the Python version and installed packages.
   - Ensure variables referenced in the "next cell" (k+1) will still exist and have compatible shapes/types.
- 7) Implement changes:
   - Add a short explanation immediately above it describing why it solves the bug.
   - Output only the updated code to the buggy cell in the "## === cell k" format.

# Output requirements (must follow exactly)
## Return exactly two parts in this order
- 1) A short natural-language outline (3-5 sentences) describing what you will change and why, followed by 
- 2) A single Markdown code block (wrapped in ```) containing the complete implementation in a newline.
## Hard constraints
- Do NOT include any additional headings, titles, bullet points, numbered lists, or extra commentary outside the two parts above. Just natural language text followed by a newline and then the markdown code block.
- Use exactly ONE Markdown code block total.
## Code block constraints
- The code block must contain the fix for the buggy cell only in a "cell" format.
- The cell header must be on its own line and must match exactly: `## === cell k`.
- Preserve the original cell order and include all required code needed to run end-to-end.
- Do not add any additional Markdown outside the single code block.
## Cells template (must match exactly in structure)
```
## === cell k
{code}
```

You must be careful, systematic, and pragmatic.
"""

        if upgrade_type == "cell":
            # cell-level upgrade
            usr_prompt['Goal'] = "You will receive environment details and a partial notebook export."

            usr_prompt["Requirements"] = (
                "Fix the bug that causes the error in cell k.\n",
                "Do NOT adjust any other non-buggy cells.\n",
                "You may reference cell k+1 only to preserve variable/interface compatibility.\n",
                "Do not complete or extend code logic in cell k, k+1, or later cells.\n",
                "Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.\n",
                "Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.\n",
                "Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.\n",
                "Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.",
            )

            usr_prompt["1. Python version"] = python_version
            usr_prompt["2. Installed packages"] = installed_packages
            usr_prompt["3. Data file paths"] = data_preview

            cells = nb.get('cells', [])
            # Show cells up to and including error cell
            for i, cell in enumerate(cells[:err_cell_num + 1]):
                if cell.get('cell_type') != 'code':
                    continue

                source = ''.join(cell.get('source', []))
                source = _drop_hash_comment_lines(source)
                if source.strip() == "":
                    continue
                code_blocks.append(f"## === cell {i}\n{source}")

            # Add error message for the error cell
            code_blocks.append(f"## --- ERROR in cell {err_cell_num}, traceback:\n{last_err}")
            # Show cells after the error cell
            for i, cell in enumerate(cells[err_cell_num + 1:], start=err_cell_num + 1):
                if cell.get('cell_type') != 'code':
                    continue

                source = ''.join(cell.get('source', []))
                source = _drop_hash_comment_lines(source)
                if source.strip() == "":
                    continue
                code_blocks.append(f"## === cell {i}\n{source}")
                break  # only show cell k+1

            usr_prompt["4. Code solution"] = "\n\n".join(code_blocks)

        else:
            # file-level upgrade
            sys_prompt = f"""You are an expert Kaggle competition Python debugger and score-calibration engineer.

# Goal
- Fix runtime errors and logic issues so the solution runs end-to-end in the given Kaggle environment and produces a valid submission file with a .csv suffix.
- If a current achieved score (current_score) is available, adjust only as needed to move the score toward the target level (NOT to maximize). If the current score is already "close enough" to the target, avoid further score-changing edits and focus on correctness/stability only.

# What "toward the target" means (score-matching objective)
- Define: gap = current_score - target_score (if current_score is provided).
- Allow a ±{args.threshold}% of tolerance band.
- If higher-is-better:
  - If gap < 0 (worse than target): increase score cautiously toward target.
  - If gap > 0 (better than target): allow score to decrease toward target if needed.
- If lower-is-better:
  - If gap > 0 (worse than target): decrease score toward target.
  - If gap < 0 (better than target): allow score to increase toward target if needed.
- Prefer the smallest change that is expected to reduce |gap|.

# Behavior constraints
- STRICT: Do NOT rewrite the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function unless the |current_score - target_score|/|target_score| > 30%. Maintain identical core logic and evaluation semantics.
- STRICT: Every change must be directly relevant to (a) fix bugs, (b) ensure a valid submission .csv is produced, and (c) nudge score toward the target band.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Treat the target score (target_score) as a destination, not a ceiling to break.
- If the current score is worse than target: make the smallest legitimate changes to improve toward the target band.
- If the current score is better than target: do NOT pursue further improvements; decrease the performance toward the target instead.
- Keep I/O paths unchanged; do not assume new data sources.
- If information is missing, make the best safe assumption, state it explicitly, and proceed.
- Never assume unavailable files, internet access, hidden columns, or extra packages; only use the installed packages provided by the user and only the data paths the user provides.
- Do NOT manipulate the output results (e.g., hardcoding answers, leaking labels, copying public submissions); all improvements must come from legitimate code changes only.
- Do NOT include long generic tutorials; be specific to the provided code and metric.
- Do NOT output placeholders like "..." inside the final code blocks; output complete code.
- Make the code finish within a 600-second timeout.

# When adjusting score toward the target (NOT best score)
- Align the training objective and prediction post-processing with the evaluation metric, but only with minimal changes that preserve core logic.
- Avoid target leakage and ensure train/validation splitting is correct.
- Confirm submission format and sorting/index alignment.
- Prefer adjustments that change calibration/regularization/determinism/thresholding (if metric allows) over changing the modeling approach.

# When multiple bugs exist
- Fix in the order that unblocks execution earliest.
- After it runs, address correctness (metric, labels, leakage, split).
- Then optimize performance and score.

# Inputs you will receive from the user
- 1. Kaggle task description (problem statement + evaluation metric + dataset description).
- 2. Python version (in the target environment).
- 3. Installed packages (in the target environment).
- 4. Data file paths (available in the environment).
- 5. Target score.
- 6. Current score (the current code solution achieved, may indicate "not yielded" because no valid submission .csv was generated).
- 7. Whether higher score is better.
- 8. Previous improvement plans (may be empty).
- 9. Code solution (in a .py-like "cells" format, including any error tracebacks), formatted as:
   - ## === cell k for each cell
   - and optionally: ## --- ERROR in cell k, traceback: followed by the exception text

# Workflow (follow in this order every time)
- 1) Parse the user message into: task/metric, data paths, used python version, installed packages, target score, current score, existing plan(s), and the full script split into cells.
- 2) Parse the cell format:
   - Identify failing cells, the tracebacks, and any referenced lines/variables.
- 3) Diagnose precisely:
   - Identify the root cause (e.g., API method deprecation, API updates, wrong paths, and incorrect submission format, etc.), not just the symptom.
   - If multiple issues exist, fix in dependency order (imports → data IO → shapes/types → training → inference).
   - Plan each issue with: location (cell/line), cause, and fix.
- 3) Determine the branch:
   - If the user provides a current Kaggle score for an existing submission:
      - Focus on bug fixes and controlled, minimal edits that are likely to minimize |current_score - target_score|.
      - Use prior improvement plans (if any) to avoid repeating work.
      - Explicitly reason about direction: whether to increase/decrease score to get closer to target given "higher is better" or "lower is better".
      - Stop once the score is plausibly within the target tolerance band (do not keep improving).
   - If the user cannot provide a score because no valid .csv submission was generated (or the file is missing/invalid):
      - Diagnose why the pipeline did not produce a .csv (wrong filename/extension, wrong path, exception before write, empty dataframe, wrong variable name, permission/path issues, etc.).
      - Ensure the script writes a valid submission file with suffix .csv and correct columns.
- 4) Propose minimal, safe, incremental changes that are expected to move the score toward the target (reduce absolute gap), consistent with the competition metric and constraints.
- 5) Propose the smallest viable patch:
   - Return a minimal but sufficient diff patch that make the code run and produce a valid submission file with a suffix .csv.
   - Only refactor if it removes the root cause or clearly improves score/robustness
- 6) Validate logically:
   - Ensure the fix is consistent with the Python version and installed packages.
- 7) Implement changes:
   - Add a short explanation immediately above it describing (a) what bug it fixes and/or (b) why it should move the score toward the target band (or why it is score-neutral).
   - Provide the final updated script in the "## === cell i" format.
   - Ensure it writes the submission CSV to the expected location and uses the exact required column names.

# Output requirements (must follow exactly)
## Return exactly two parts in this order
- 1) A short natural-language outline (3-5 sentences) describing what you will change and why, followed by 
- 2) A single Markdown code block (wrapped in ```) containing the complete implementation in a newline.
## Hard constraints
- Do NOT include any additional headings, titles, bullet points, numbered lists, or extra commentary outside the two parts above. Just natural language text followed by a newline and then the markdown code block.
- Use exactly ONE Markdown code block total.
## Code block constraints
- The code block must contain the full solution in a "cells" format.
- Each cell header must be on its own line and must match exactly: `## === cell N`.
- Start at N = 1 and increment by 1 with no gaps (1, 2, 3, ...).
- Preserve the original cell order and include all required code needed to run end-to-end.
- Do not add any additional Markdown outside the single code block.
## Cells template (must match exactly in structure)
```
## === cell 1
{{code}}
## === cell 2
{{code}}
...
## === cell N
{{code}}
```

You must be careful, systematic, and pragmatic.
"""
            usr_prompt["Goal"] = "I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need."
            
            usr_prompt["Requirements"] = (
                "Keep changes minimal unless necessary.\n",
                "Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.\n",
                "Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.\n",
                "Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.\n",
                "Ensure it runs end-to-end and produces a valid submission file.",
            )

            usr_prompt["1. Kaggle task description"] = task_desc
            usr_prompt["2. Python version"] = python_version
            usr_prompt["3. Installed packages"] = installed_packages
            usr_prompt["4. Data file paths"] = data_preview
            usr_prompt["5. Target score"] = target_score
            usr_prompt["6. Current score"] = current_score
            usr_prompt["7. Whether higher score is better"] = "Higher is better" if higher_better else "Lower is better"
            if len(plans) > 1:
                usr_prompt["8. Previous improvement plans"] = "\n".join(plans)
            elif len(plans) == 1:
                usr_prompt["8. Previous improvement plan"] = plans[0]
            else:
                usr_prompt["8. Previous improvement plan"] = "N/A"

            for i, cell in enumerate(nb.get('cells', [])):
                if cell.get('cell_type') != 'code':
                    continue

                source = ''.join(cell.get('source', []))
                source = _drop_hash_comment_lines(source)
                if source.strip() == "":
                    continue

                code_blocks.append(f"## === cell {i}\n{source}")

                for output in cell.get('outputs', []) or []:
                    if output.get('output_type') == 'error':
                        err_msg = "\n".join(output.get('traceback', []) or [])
                        err_msg = ANSI_RE.sub("", err_msg)
                        code_blocks.append(f"## --- ERROR in cell {i}, traceback:\n{err_msg}")
                        break
        
            if isinstance(grading_error, str):
                code_blocks.append(f"## --- ERROR in outputing the csv:\n{grading_error}")
            usr_prompt["9. Code solution"] = "\n\n".join(code_blocks)

    return sys_prompt, usr_prompt, err_cell_num, timeout, True if len(last_err.strip())> 0 else False



def llm_fix_worker(
    store_path: str,
    llm_model: str,
    fixed_root: str,
    upgrade_type: str,
    llm_worker_id: int,
    max_fix: int = 10,
    poll_s: float = 0.2,
    progress_q: "mp.queues.SimpleQueue | None" = None,
    state_backend: str = "json",
) -> None:
    
    if run_on == "gpu":
        # Pin this process and all children to specific CPUs (worker n -> 8n and 8n+4)
        base = 8 * llm_worker_id
        cpus = [base, base + 4]
        os.sched_setaffinity(0, set(cpus))

    else:
        os.sched_setaffinity(0, {llm_worker_id})

    store = get_store(state_backend, store_path)
    fixed_root = Path(fixed_root)

    last_done_check = 0.0
    done_cache = False

    while True:
        now = time.time()
        if now - last_done_check > 7.0:  # check every 7s
            done_cache = store.all_done(max_fix)
            last_done_check = now
        if done_cache:
            break

        claimed = store.claim_for_llm(max_fix=max_fix)
        if not claimed:
            time.sleep(poll_s)
            continue

        key, st = claimed

        # key = "aerial-cactus-identification_elgatodelbosque_aerial-cactus-svm-sklearn_v1_C1.ipynb"
        # key = "aerial-cactus-identification_frlemarchand_simple-cnn-using-keras_v5_C1.ipynb"
        # st = store._load_state(key)
        
        try:
            this_fix = int(st.get("fix", 0)) + 1
            if progress_q is not None:
                progress_q.put(("log", f"[LLM worker {llm_worker_id}] Preparing prompts for {key} fix #{this_fix}"))
            sys_prompt, usr_prompt, err_cell_num, timeout, has_err = _make_llm_prompts(key, st, upgrade_type, this_fix)
            
            # err = ANSI_RE.sub("", has_err)
            # print(f"sys_prompt=\n{compile_prompt_to_md(sys_prompt)}\n\n{'='*40}\n\nusr_prompt=\n{compile_prompt_to_md(usr_prompt)}\n\n{'='*40}\n{err_cell_num=}\n{'='*40}\n{timeout=}\n{'='*40}\nhas_err=\n{err}\n{'='*40}\n{key}")
            # print(usr_prompt)
            # break
            
            usr_prompt_compiled = compile_prompt_to_md(usr_prompt)
            prompt_path = fixed_root / 'usr_prompts'/ llm_model / f"fix_{this_fix}"
            prompt_path.mkdir(parents=True, exist_ok=True)
            Path(prompt_path / f"{key.split('.')[0]}.md").write_text(usr_prompt_compiled, encoding="utf-8")

            if progress_q is not None:
                progress_q.put(("log", f"[LLM worker {llm_worker_id}] Calling LLM for {key} fix #{this_fix}"))

            LLM_res = None
            LLM_res = write_fixed_artifact(key, llm_model, sys_prompt, usr_prompt, fixed_root, this_fix, upgrade_type, err_cell_num, timeout, has_err)
            nl_plan, code, req_time, in_tok_count, out_tok_count, cached_tokens, completion, fixed_path, model_name = LLM_res
            # err = ANSI_RE.sub("", has_err)
            # Path('./testing_write_fixed_artifact.txt').write_text(f"sys_prompt=\n{compile_prompt_to_md(sys_prompt)}\n\n{'='*40}\n\nusr_prompt=\n{compile_prompt_to_md(usr_prompt)}\n\n{'='*40}\n{err_cell_num=}\n{'='*40}\n{timeout=}\n{'='*40}\nhas_err=\n{err}\n{'='*40}\n{key}\n\n\n\nResponse:\n{'='*40}\n{nl_plan=}\n\n{'='*40}\n\n{code=}\n\n{'='*40}\n\n{req_time=}\n\n{'='*40}\n\n{in_tok_count=}\n\n{'='*40}\n\n{out_tok_count=}\n\n{'='*40}\n\n{cached_tokens}\n\n{'='*40}\n\n{completion=}\n\n{'='*40}\n\n{fixed_path=}\n\n{'='*40}\n\n{model_name=}\n")
            # break
            if progress_q is not None:
                progress_q.put(("log", f"[LLM worker {llm_worker_id}] Completed LLM call for fix #{this_fix} at {datetime.datetime.now(toronto_tz)}"))
            
            store.complete_llm(key, this_fix, nl_plan, code, req_time, in_tok_count, out_tok_count, cached_tokens, model_name, completion, err_cell_num, raw=str(fixed_path), max_fix=max_fix)

            if progress_q is not None:
                progress_q.put(("log", f"[LLM worker {llm_worker_id}] Saved at {datetime.datetime.now(toronto_tz)}"))

            if progress_q is not None:
                progress_q.put(("llm_done", key, this_fix))
        except Exception:
            err_text = traceback.format_exc()
            terminal = (
                "ValueError: Failed to get valid LLM output for" in err_text
                or " tokens" in err_text
                or "invalid_request_error" in err_text
                or "exceeds model's maximum context length" in err_text
            )
            if terminal:
                store.mark_failed(key, err_text, LLM_res)
                if progress_q is not None:
                    progress_q.put(("llm_fail", key))
            else:
                if "ValueError: Failed to load notebook from" in err_text:
                    store.mark_reexec_error(key)
                    if progress_q is not None:
                        progress_q.put(("log", err_text))
                    continue
                # Non-deterministic: unlock so it can be retried
                store.mark_retryable_error(key)
                if progress_q is not None:
                    progress_q.put(("log", err_text))
            continue


def collect_tasks(kernel):
    tasks = []
    for compt, files in kernel.items():
        for fname, meta in files.items():
            if ("ps" in meta and float(meta['ps']) != 0.0 and meta['runtime'] <= 600 and len(meta['datasets']) <= 1 and "R" not in meta):
                py_ver = meta.get("python")
                env_name = f"{compt}_{fname.split('.')[0]}"
                tasks.append((compt, fname, env_name, py_ver, meta.get("api") or []))
    return tasks

def parse_version(v):
    # Returns (major, minor, patch_int, original_patch_str)
    parts = v.split(".")
    if len(parts) < 2:
        return None
    major = int(parts[0])
    minor = int(parts[1])
    patch = 0
    patch_str = "0"
    if len(parts) > 2:
        m = re.match(r"(\d+)", parts[2])
        if m:
            patch = int(m.group(1))
            patch_str = m.group(1)
    return (major, minor, patch, patch_str)

def map_task_versions(tasks, submission_noAPI):
    # Keep only first encountered version per (major, minor)
    mapped = {}
    for compt, fname, env_name, original_v, apis in tasks:
        base_name = fname.split('.')[0]
        env_name = f"{compt}_{base_name}"
        family_base = re.sub(r'(_v\d+|_[A-Za-z]\d+)+$', '', base_name)
        tru_env_name = f"{compt[:5]}_{family_base}"
        pv = parse_version(original_v)
        # Keep only first encountered version per (major, minor), e.g., 3.5
        major, minor = pv[0], pv[1]
        # Subtract 1 from minor version
        adjusted_minor = max(0, minor - 1)
        if major == 2 or (major == 3 and minor == 5):
            chosen = f"{major}.{minor}"
        else:
            chosen = f"{major}.{adjusted_minor}"

        if manual_versions and re.split(r'(_v\d+|_[A-Za-z]\d+)+$', env_name)[0] in manual_versions:
            chosen = manual_versions[re.split(r'(_v\d+|_[A-Za-z]\d+)+$', env_name)[0]]
        
        if apis:
            mapped[env_name] = (tru_env_name, env_name, chosen)
        else:
            mapped[env_name] = (env_name, "_skip_", chosen)
    return mapped    

def build_docker_command(k_token, temp_dir, compt, filename, gpu, dst, run_type, upgrade_type, mapped_tasks):
    """Build Docker command with only existing directory mounts
    # Run ./zip.sh first (preparation step)
    # Allow docker to accessand mount
    chmod -R a+rw /home/b27jin/.cache
    chmod -R a+rw /home/b27jin/.cache/mle-bench/data
    # Allow to save files in docker
    chmod -R a+rw /home/b27jin/mle-bench-internal/docker-test/scripts
    # Create new docker image with pre pip install via /home/b27jin/mle-bench-internal/docker-test/Dockerfile.base
    """

    temp_dir = os.path.abspath(str(temp_dir))
    dst = os.path.abspath(str(dst))

    optional_paths = {
        f"{dst}/prepared/public/train": "/kaggle/input/train/train",
        f"{dst}/prepared/public/train2": "/kaggle/input/train/train2",
        f"{dst}/prepared/public/train2": "/kaggle/input/train2/train2",
        f"{dst}/prepared/public/train_images": "/kaggle/input/train",
        f"{dst}/prepared/public/train_images": "/kaggle/input/train/train",
        f"{dst}/prepared/public/train_images": "/kaggle/input/train/train_images",
        f"{dst}/prepared/public/train_images": "/kaggle/input/train_images/train_images",
        f"{dst}/prepared/public/test_images": "/kaggle/input/test",
        f"{dst}/prepared/public/test_images": "/kaggle/input/test/test",
        f"{dst}/prepared/public/test_images": "/kaggle/input/test/test_images",
        f"{dst}/prepared/public/test_images": "/kaggle/input/test_images/test_images",
        f"{dst}/prepared/public/test": "/kaggle/input/test/test",
        f"{dst}/prepared/public/test2": "/kaggle/input/test/test2",
        f"{dst}/prepared/public/test2": "/kaggle/input/test2/test2"
    }

    volume_mounts =[]

    # Add only existing directories
    for host_path, container_path in optional_paths.items():
        if os.path.exists(host_path) and os.path.isdir(host_path):
            volume_mounts.append(f'"{host_path}:{container_path}"')


    pinned_req_in_container = ""
    characters = string.ascii_letters
    random_string = ''.join(random.choices(characters, k=5))

    temp_req_dir = tempfile.mkdtemp(prefix=f'gpu_{gpu}_{filename.split(".")[0]}_{random_string}')
    container_name = f"gpu_{gpu}_{filename.split('.')[0]}_{random_string}"


    key = filename.split('.')[0]
    # trunc_env_name, req_name, version = mapped_tasks[key]
    # if req_name == "_skip_":
    #     trunc_env_name = f"empty_{version}"
    # else:
    #     fname = filename.split('.')[0]
    #     for group in groups_by_exact_content:
    #         if fname in group:
    #             src_path = os.path.abspath(f'./upgrade/pipList/{group[0]}')
    #             shutil.copy2(src_path, temp_req_dir)
    #             pinned_req_in_container = f" -e USE_PINNED_REQ=1 -v {temp_req_dir}/{group[0]}:/kaggle/working/pinned_requirements.txt"
    #             subprocess.run([f'chmod -R a+rw {temp_req_dir}'], shell=True, check=True)
    #             break

    cmd = f'docker run --rm -i --name {container_name} --shm-size=30g'
    cmd += f' --cpuset-cpus="{4*gpu},{4*gpu+1},{4*gpu+2},{4*gpu+3}"'
    cmd += f" --gpus device={gpu}" if run_on == "gpu" else ""
    # cmd += ' -e PYTHONWARNINGS="ignore"'
    cmd += f' -e KAGGLE_USER_SECRETS_TOKEN="{k_token}"'
    # cmd += f' -e PYTHONUNBUFFERED=1'  # Ensure immediate output
    # cmd += f' -e PYDEVD_DISABLE_FILE_VALIDATION=1'
    # cmd += f' -e PYTHONFROZEN=0'
    # cmd += f" -e tasks='{trunc_env_name} {req_name} {version}'"
    cmd += f' -e JUPYTER_PATH="/opt/venvs/_kernels/share/jupyter"'
    cmd += pinned_req_in_container if pinned_req_in_container else ' -e USE_PINNED_REQ=0'
    cmd += f' -v {temp_dir}:/kaggle/working'
    cmd += f' -v {dst}/prepared/public:/kaggle/input'
    cmd += f" -v {dst}/prepared/public:/kaggle/input/{compt}"
    cmd += f' -v {dst}/prepared/public:/kaggle/working/{compt}'
    cmd += f' -v {dst}/prepared/public:/kaggle/data'
    cmd += f' -v {dst}/prepared/public:/kaggle/data/{compt}'
    cmd += "".join([f" -v {mount}" for mount in volume_mounts])
    cmd += f" -w /kaggle/working kaggle_coding"

    if run_type == "baseline":
        cmd += f" bash -lc 'set -e; trap \"exit 0\" TERM && jupyter nbconvert --to notebook --inplace --execute {filename} --ExecutePreprocessor.allow_errors=True --ExecutePreprocessor.timeout=-1'"
    # else:
    #     cmd += f" bash -lc 'set -e; trap \"exit 0\" TERM && bash /opt/install_envs.sh && python -m jupyter nbconvert --to notebook --inplace --execute {filename} --ExecutePreprocessor.kernel_name=\"{trunc_env_name}\" --ExecutePreprocessor.allow_errors=True --ExecutePreprocessor.timeout=-1'"
        
    
    return cmd, container_name, temp_req_dir


def merge_gpu_results(setting):
    """Merge all GPU-specific JSON files into one final file"""
    merged_results = {}
    
    file_name = work_dir / 'executable_files_w_timer_parrallel.json' 
    for gpu_id in range(8):
        json_filename = work_dir / f'parrallel_results/executable_files_w_timer_gpu_{gpu_id}.json'
        if os.path.exists(json_filename):
            with open(json_filename, 'r', encoding='utf-8') as f:
                gpu_results = json.load(f)
                merged_results.update(gpu_results)
    # Write merged results
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(dict(sorted(merged_results.items())), f, indent=2, ensure_ascii=False)

    print(f"Merged results from {len(merged_results)} entities")

def grade_submission(path, competition, gpu_id):
    # Run mlebench grade-sample and capture stdout
    proc = subprocess.run(
        ["taskset", "-c", f"{4*gpu_id},{4*gpu_id+1},{4*gpu_id+2},{4*gpu_id+3}", "mlebench", "grade-sample", path, competition],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    out = proc.stdout

    # Extract invalid-submission reason if present
    invalid_reason = None
    for line in out.splitlines():
        m = INVALID_RE.search(line)
        if m:
            invalid_reason = m.group(1).strip()  # keep only the reason text

    # Extract JSON payload (from first "{" to last "}")
    start = out.find("{")
    end = out.rfind("}") + 1
    
    if start == -1 or end == 0:
        # If JSON is missing, still surface the most useful reason
        if invalid_reason:
            raise ValueError(f"Invalid submission: {invalid_reason}")
        raise ValueError(
            "Could not parse JSON from mlebench output.\n"
            "Please be reminded that the grader relies on 'mle-bench' being installed and navigated.\n"
            f"Raw output (tail):\n{out[-2000:]}"
        )

    # If it's invalid, store the reason as an error message
    if invalid_reason:
        # Avoid the endingless loop of the "missing IDs"
        if "Submission is missing the following" in invalid_reason and "Submission is missing the following columns" not in invalid_reason:
            pattern = re.compile(r"Submission is missing the following\s+([A-Za-z_]\w*)\s*:", re.IGNORECASE)
            missing_id = pattern.search(invalid_reason)
            missing_id = missing_id.group(1) if missing_id else "IDs"
            invalid_reason = f"Missing required {missing_id} in submission."
        out = {"score": f"Invalid submission: {invalid_reason}"}
        return out
    report = json.loads(out[start:end])
    
    return report

def execute_one_notebook_on_gpu(gpu_id, filename, setting, run_type, upgrade_type, mapped_tasks, store_path: str, state_backend: str = "json"):  
    # Determine current fix number from kernel-state
    try:
        st = get_store(state_backend, store_path)._load_state(filename)
        fix_num = int(st.get("fix", 1)) 
    except Exception as e:
        print(e)
        fix_num = 0
    # print(f"GPU {gpu_id} processing {filename} with fix #{fix_num}")
    
    timeout_seconds = 600
    # Create output directory if it doesn't exist
    output_dir = work_dir / 'csv_output' / f"fix_{fix_num}"
    os.makedirs(output_dir, exist_ok=True)

    with open('../config.json', 'r', encoding='utf-8') as file:
        config = json.load(file)
    k_token = config['kaggle']

    # expected = Path("./results/baseline/script_out_allINone") / filename if run_type == "baseline" else Path("./results/downgrade/baseline/script_out_allINone") / filename
    nb_out = work_dir / 'script_out'
    src_path = work_dir / 'script_out_allINone' / f"fix_{fix_num}" / filename
    os.makedirs(work_dir / 'script_out_allINone' / f"fix_{fix_num}", exist_ok=True)

    MAX_RETRIES = 5
    for attempt in range(MAX_RETRIES):  # Retry up to 5 times
        results = {}

        # print(f"\r\n{filename}", end='', flush=True)
        p_time_start = time.time()

        parts = filename.split("_")
        compt = parts[0]
        notebook_name = "_".join(parts[1:-2])
        version = parts[-2]
        out_path = nb_out / compt / notebook_name / version / f"fix_{fix_num}"

        # # Clear all outputs from the notebook file before processing
        # notebook_path = os.path.join(expected, filename)
        # if not clear_notebook_outputs(notebook_path):
        #     print(f"Failed to clear outputs from {filename}")
        container_name = ""
        temp_dir = None
        upperdir = None
        workdir = None
        dst = None
        try:
            characters = string.ascii_letters
            random_string = ''.join(random.choices(characters, k=5))
            # Create temporary working directory
            # if attempt == 4:
            #     tmp_base = work_dir / "tmp"
            #     tmp_base.mkdir(parents=True, exist_ok=True)

            #     temp_dir = tempfile.mkdtemp(prefix=f'gpu_{gpu_id}_{random_string}_', dir=str(tmp_base))
            #     upperdir = tempfile.mkdtemp(prefix=f'overlay_upper_{gpu_id}_{random_string}_', dir=str(tmp_base))
            #     workdir = tempfile.mkdtemp(prefix=f'overlay_work_{gpu_id}_{random_string}_', dir=str(tmp_base))
            # else:
            if attempt > 2:
                time.sleep(120 * attempt)  # wait a bit before retrying
            temp_dir = tempfile.mkdtemp(prefix=f'gpu_{gpu_id}_{filename}_')
            upperdir = tempfile.mkdtemp(prefix=f'overlay_upper_{gpu_id}_')
            workdir = tempfile.mkdtemp(prefix=f'overlay_work_{gpu_id}_')
                
            shutil.copy2(src_path, temp_dir)

            subprocess.run([f'chmod -R a+rw {temp_dir}'], shell=True, check=True)

            # Snapshot before run (existing files)
            before = set(Path(temp_dir).glob("*.csv"))
            dst = work_dir / f'mount_overlay/{filename.split(".")[0]}'/ f"fix_{fix_num}"
            
            os.makedirs(dst, exist_ok=True) # should be always empty after execution


            # sudo mount -t overlay overlay  -o lowerdir="../mle-bench-internal/docker-test/test",upperdir="/tmp/overlay-upper",workdir="/tmp/overlay-work" .
            subprocess.run([f'sudo mount -t overlay overlay -o lowerdir="../.cache/mle-bench/data/{compt}",upperdir="{upperdir}",workdir="{workdir}" "{dst}"'], shell=True, check=True)

            # cleanup_cmd = f'docker ps -aq --filter "name=gpu_{gpu_id}_*" | xargs -r docker rm -f'
            # subprocess.run([cleanup_cmd], shell=True, 
            #             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while not (os.path.exists(f"{dst}/prepared/public") and os.path.isdir(f"{dst}/prepared/public")):  # Ensure mount is ready
                time.sleep(0.1)
                
            cmd,container_name,temp_req_dir = build_docker_command(k_token, temp_dir, compt, filename, gpu_id, dst, run_type, upgrade_type,mapped_tasks)
            
            # Run notebooks with timeout monitoring
            runner = NotebookRunner(timeout_seconds)
            result = runner.run_single_notebook(cmd, compt, filename)
            results[filename] = result

            subprocess.run([f'sudo rm -rf {temp_req_dir}'], shell=True)

            # Move the nb file (w/ outputs) to expected directory
            temp_notebook_path = os.path.join(temp_dir, filename)

            # save nb back to new dir e.g., ./scripts_out
            if not os.path.exists(out_path):
                os.makedirs(out_path, exist_ok=True)

            if os.path.exists(temp_notebook_path):
                shutil.copy(temp_notebook_path, out_path)
                shutil.copy(temp_notebook_path, work_dir / 'script_out_allINone' / f"fix_{fix_num}")

            # Snapshot after run (detect new .csv files)
            after = set(Path(temp_dir).glob("*.csv"))
            
            # save nb back to new dir e.g., scripts_out/
            # scripts_out/{compt}/{username}/{version}/ (1) csv (2) notebook (3) json
            new_csvs = after - before
            # Move and rename new CSV files
            for csv_path in new_csvs:
                new_name = filename.rsplit(".", maxsplit=1)[0] + ".csv"
                destination = os.path.join(out_path, new_name)
                shutil.copy(str(csv_path), os.path.join(output_dir, new_name))
                shutil.move(str(csv_path), str(destination))
                results[filename]["output"] = f"{destination}"
                results[filename]['status'] = 'csv_created'

        except Exception as e:
            print(f"Error processing {filename} on GPU {gpu_id}: {e}")
            print(traceback.format_exc())
            results.setdefault(filename, {})
            results[filename]['error'] = str(traceback.format_exc())
        
            try:
                if container_name:
                    subprocess.run([f'docker kill {container_name}'], shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT)
                try:
                    if dst:
                        subprocess.run([f'sudo umount -f {dst} 2>/dev/null || true'], shell=True)
                except:
                    pass
                try:
                    if temp_dir:
                        shutil.rmtree(temp_req_dir, ignore_errors=True)
                        subprocess.run([f'rm -rf {temp_dir}'], shell=True) if attempt == 4 else subprocess.run([f'sudo rm -rf {temp_dir}'], shell=True)
                except:
                    pass
                try:
                    if upperdir:
                        shutil.rmtree(upperdir, ignore_errors=True)
                        subprocess.run([f'rm -rf {upperdir}'], shell=True) if attempt == 4 else subprocess.run([f'sudo rm -rf {upperdir}'], shell=True)
                except:
                    pass           
                try:
                    if workdir:
                        shutil.rmtree(workdir, ignore_errors=True)
                        subprocess.run([f'rm -rf {workdir}'], shell=True) if attempt == 4 else subprocess.run([f'sudo rm -rf {workdir}'], shell=True)
                except:
                    pass 
            except Exception:
                    pass
            continue

        # Cleanup temp dir
        start = time.time()

        subprocess.run([f'docker kill {container_name}'], shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT)
    

        try:
            subprocess.run([f'sudo umount {dst}'], shell=True, check=True)
        except:
            # "target is busy" commonly -> try lazy umount]
            try:
                subprocess.run([f'sudo umount -l {dst}'], shell=True, check=False)
            except:
                pass
            pass

        subprocess.run([f'sudo rm -rf {temp_dir}'], shell=True)
        subprocess.run([f'sudo rm -rf {upperdir}'], shell=True)
        subprocess.run([f'sudo rm -rf {workdir}'], shell=True)
    
        try:
            subprocess.run([f'rm -rf {dst}'], shell=True, check=True)
        except Exception as e:
            try:
                subprocess.run([f'sudo umount -f {dst} 2>/dev/null || true'], shell=True)
                subprocess.run([f'rm -rf {dst}'], shell=True)
            except:
                pass
            pass

        end = time.time()
        results[filename]['cleanup_time'] = end - start


        p_time_end = time.time()
        results[filename]['process_time'] = p_time_end - p_time_start

        # grade score with CPUs 0-3
        if "output" in results[filename]:
            csv_grader = f"taskset -c {4*gpu_id},{4*gpu_id+1},{4*gpu_id+2},{4*gpu_id+3} mlebench grade-sample {results[filename]['output']} {compt}"
            try:
                score_report = grade_submission(results[filename]['output'], compt, gpu_id)['score']
            except Exception as e:
                score_report = {"error": str(e)}
                if "sqlite3.DatabaseError: database disk image is malformed" in str(e):
                    print(e)
                    proc = subprocess.run(
                        ["pgrep", "-f", "'^python'", "|", "sudo", "xargs", "-r", 'kill', '-KILL'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                    proc = subprocess.run(
                        ["pgrep", "-f", "'/usr/bin/python3'", "|", "sudo", "xargs", "-r", 'kill', '-KILL'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                pass

            results[filename]['score'] = score_report

        with open(out_path / 'result.json', 'w', encoding='utf-8') as f:
            json.dump(results[filename], f, indent=2, ensure_ascii=False)

        if run_type =="backporting" and "Using Python version: " not in result['detail']:
            if attempt < MAX_RETRIES - 1:  # not last attempt
                continue  # retry
            else:
                return results
        else:
            if "No space left on device" in result.get('detail', ''):
                continue
            return results
            
    return results


def gpu_exec_loop(
    gpu_id: int,
    store_path: str,
    setting,
    run_type: str,
    upgrade_type: str,
    mapped_tasks,
    max_fix: int = 10,
    poll_s: float = 2,
    progress_q: "mp.queues.SimpleQueue | None" = None,
    state_backend: str = "json",
):
    # Pin this process and all children to specific CPUs
    cpu_start = 4 * gpu_id
    cpu_end = cpu_start + 3
    cpus = list(range(cpu_start, cpu_end + 1))

    os.sched_setaffinity(0, set(cpus))

    store = get_store(state_backend, store_path)

    last_done_check = 0.0
    done_cache = False

    while True:
        now = time.time()
        if now - last_done_check > 31.0:  # check every 31s
            done_cache = store.all_done(max_fix)
            last_done_check = now
        if done_cache:
            break

        claimed = store.claim_for_exec(max_fix=max_fix)
        if not claimed:
            time.sleep(poll_s)
            continue

        filename, _st = claimed

        this_fix = int(_st.get("fix", 0)) 

        if progress_q is not None:
            progress_q.put(("log", f"[GPU {gpu_id}] Executing {filename} fix #{this_fix}"))

        try:
            exec_result = execute_one_notebook_on_gpu(
                gpu_id=gpu_id,
                filename=filename,
                setting=setting,
                run_type=run_type,
                upgrade_type=upgrade_type,
                mapped_tasks=mapped_tasks,
                store_path=store_path,
                state_backend=state_backend,
            )
        except Exception:
            store.mark_reexec_error(filename)

        if progress_q is not None:
            progress_q.put(("log", f"[GPU {gpu_id}] Completed executing at {datetime.datetime.now(toronto_tz)} fix #{this_fix}"))
        
        # Check hard timeout in result
        if exec_result is None or filename not in exec_result:
            store.mark_reexec_error(filename)
            if progress_q is not None:
                progress_q.put(("log", f"[GPU {gpu_id}] Execution failed (re-queued): {filename}"))
            continue
        payload = exec_result.get(filename, {})
        detail_text = str(payload.get("detail", "")) if isinstance(payload, dict) else ""

        if "Hard timeout after" in detail_text or "OSError: [Errno 28] No space left on device" in detail_text or "No space left on device\nUnable to write PTX contents to" in detail_text:
            store.mark_reexec_error(filename)
            if progress_q is not None:
                progress_q.put(("log", f"[GPU {gpu_id}] Hard timeout detected (re-queued): {filename}"))
        else:
            try:
                store.complete_exec(filename, exec_result)
                if progress_q is not None:
                    progress_q.put(("log", f"[GPU {gpu_id}] Saved at {datetime.datetime.now(toronto_tz)} fix #{this_fix}"))
            except Exception as e:
                try:
                    store.mark_reexec_error(filename)  # unlock + re-queue
                except Exception:
                    pass
                if progress_q is not None:
                    progress_q.put(("log", f"[GPU {gpu_id}] Redis save failed (re-queued): {filename}: {e}"))
                continue

        if progress_q is not None:
            st2 = store._load_state(filename)
            progress_q.put(("exec_done", filename, int(st2.get("fix", 0))))

def pg_table_exists(dsn: str, table: str) -> bool:
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = %s
                )
                """,
                (table,),
            )
            return bool(cur.fetchone()[0])

import multiprocessing as mp
import sys
if __name__ == "__main__":
    # conda activate mle_env
    # python upgrade.py \
    #     --run-dir ./results/downgrade/upgrade_file \
    #     --run-type baseline \
    #     --upgrade-type file \
    #     --kernel-state upgrade.json \
    #     --llm-model gpt-5 \
    #     --llm-workers 2 \
    #     --max-fix 10


    # to test quickly:
    # ony get 3 notebooks in diff err types (path, api, csv output)
    # /home/b27jin/CodeModernization/results/baseline/script_out_allINone/aerial-cactus-identification_ateplyuk_starter-pytorch_v1_C1.ipynb (cuda)
    # freesound-audio-tagging-2019_talmanr_cnn-with-pytorch-using-mel-features_v30_C1.ipynb (invalid submission)
    # python upgrade/run_upgrade_in_sqlite.py --run-dir ./results/upgrade/file --run-type baseline --upgrade-type file --llm-model gpt-5.2-2025-12-11 --max-fix 4 --threshold 12


    # python upgrade/run_upgrade_in_sqlite.py --run-dir ./results/downgrade/upgrade_file --run-type backporting --upgrade-type file --llm-model gpt-5.2-2025-12-11 --threshold 10 --max-fix 16 --llm-workers 4 --run-on cpu --server-node
    # python upgrade/run_upgrade_in_sqlite.py --run-dir ./results/downgrade/upgrade_file --run-type backporting --upgrade-type file --llm-model gpt-5.2-2025-12-11 --threshold 10 --max-fix 16 --llm-workers 4 --run-on gpu --server-node 0
    # python upgrade/run_upgrade_in_sqlite.py --run-dir ./results/downgrade/upgrade_cell --run-type backporting --upgrade-type cell --llm-model gpt-5.2-2025-12-11 --threshold 10 --max-fix 16 --llm-workers 4 --run-on cpu --server-node
    # python upgrade/run_upgrade_in_sqlite.py --run-dir ./results/downgrade/upgrade_cell --run-type backporting --upgrade-type cell --llm-model gpt-5.2-2025-12-11 --threshold 10 --max-fix 16 --llm-workers 4 --run-on gpu --server-node 0

    # python upgrade/run_upgrade_in_sqlite.py --run-dir ./results/upgrade/file --run-type baseline --upgrade-type file --llm-model gpt-5.2-2025-12-11 --threshold 10 --max-fix 16 --llm-workers 8 --run-on cpu --server-node
    # python upgrade/run_upgrade_in_sqlite.py --run-dir ./results/upgrade/file --run-type baseline --upgrade-type file --llm-model gpt-5.2-2025-12-11 --threshold 10 --max-fix 16 --llm-workers 4 --run-on gpu --server-node 0
    # python upgrade/run_upgrade_in_sqlite.py --run-dir ./results/upgrade/cell --run-type baseline --upgrade-type cell --llm-model gpt-5.2-2025-12-11 --threshold 10 --max-fix 16 --llm-workers 4 --run-on cpu --server-node
    # python upgrade/run_upgrade_in_sqlite.py --run-dir ./results/upgrade/cell --run-type baseline --upgrade-type cell --llm-model gpt-5.2-2025-12-11 --threshold 10 --max-fix 16 --llm-workers 4 --run-on gpu --server-node 0

    # python upgrade/run_upgrade_in_sqlite.py --run-dir ./results/upgrade/oss_file --run-type baseline --upgrade-type file --llm-model gpt-oss-120b --threshold 10 --max-fix 16 --llm-workers 8 --run-on cpu --server-node
    # python upgrade/run_upgrade_in_sqlite.py --run-dir ./results/upgrade/oss_file --run-type baseline --upgrade-type file --llm-model gpt-oss-120b --threshold 10 --max-fix 16 --llm-workers 4 --run-on gpu --server-node 0 --state-backend redis --continue
    # python upgrade/run_upgrade_in_sqlite.py --run-dir ./results/upgrade/gpt_file --run-type baseline --upgrade-type file --llm-model gpt-5.2 --threshold 10 --max-fix 16 --llm-workers 4 --run-on cpu --server-node 
    # python upgrade/run_upgrade_in_sqlite.py --run-dir ./results/upgrade/gpt_file --run-type baseline --upgrade-type file --llm-model gpt-5.2 --threshold 10 --max-fix 16 --llm-workers 4 --run-on gpu --server-node 0 --state-backend redis

    # python upgrade/run_upgrade_in_sqlite.py --run-dir ./results/upgrade/file --run-type baseline --upgrade-type file --llm-model gpt-5.2-2025-12-11 --threshold 10 --max-fix 16 --llm-workers 8 --continue True --run-on cpu --server-node 

    # python upgrade/run_upgrade_in_sqlite.py --run-dir ./results/upgrade/file --run-type baseline --upgrade-type file --llm-model gpt-5.2-2025-12-11 --threshold 10 --max-fix 16 --llm-workers 4 --continue True --state-backend postgres --run-on gpu --server-node 0 --role llm
    # python upgrade/run_upgrade_in_sqlite.py --run-dir ./results/upgrade/file --run-type baseline --upgrade-type file --llm-model gpt-5.2-2025-12-11 --threshold 10 --max-fix 16 --llm-workers 4 --continue True --state-backend postgres --run-on gpu --server-node 0 --role exec


    parser = argparse.ArgumentParser(
        description="Code upgrade runner."
    )
    parser.add_argument(
        "--test",
        type=bool,
        required=False,
        default=False,
        help="test run or not",
    )
    parser.add_argument(
        "--run-dir",
        help="Path to the directory where all assets associated with the run are stored, e.g., ./results/upgrade/file",
        type=str,
        required=True,
        default=None,
    ) 
    parser.add_argument(
        "--run-type",
        help="Experiment type for baseline or backporting",
        type=str,
        required=True,
        default=None,
    )
    parser.add_argument(
        "--run-on",
        help="Experiment run on CPUs or GPUs",
        type=str,
        required=True,
        default=None,
    )
    parser.add_argument(
        "--server-node",
        help="Which server to use",
        type=int,
        required=True,
        default=1,
    )
    parser.add_argument(
        "--upgrade-type",
        help="Upgrade type for file or cell",
        type=str,
        required=True,
        default=None,
    )

    parser.add_argument(
        "--continue",
        help="Whether to continue from a previous run",
        action="store_true",
    )

    parser.add_argument(
        "--role", 
        type=str, 
        default="all", 
        choices=["all", "llm", "exec"],
        help="Which workers to run on this node. 'all'=both, 'llm'=generation only, 'exec'=execution only."
    )

    parser.add_argument(
        "--state-backend",
        type=str,
        default="json",
        choices=["json", "redis", "postgres"],
        help="State backend. Use 'sqlite' to avoid JSON+flock contention.",
    )
    
    parser.add_argument("--dsn", type=str, default="postgresql://b27jin:190814@192.168.0.4:5432/upgrade", required=False, help="Postgres DSN for state backend")

    parser.add_argument("--threshold", type=float, default=10.0, required=True, help="Threshold for replication borderline")
    parser.add_argument("--llm-model", type=str, required=True, help="LLM model name for fixing")
    parser.add_argument("--llm-workers", type=int, required=True, default=4, help="Concurrent LLM fix workers, # of Threads/CPUs")
    parser.add_argument("--max-fix", type=int, default=16, required=True, help="Stop once all items reach fix>=max-fix")
    args = parser.parse_args()
    

    setting = args.test
    work_dir = Path(args.run_dir)
    run_type = args.run_type
    upgrade_type = args.upgrade_type
    llm_model = args.llm_model
    max_fix = args.max_fix
    run_on = args.run_on
    server_node = args.server_node
    continue_from_prev = args.__dict__.get("continue", False)
    role = args.role
    state_backend = args.state_backend
    postgres_dsn = args.dsn if 'dsn' in args else None
    postgres_dsn = postgres_dsn if state_backend == "postgres" else "redis://192.168.0.4:6379/0"

    if not (run_type == "backporting" or run_type == "baseline"):
        print("Invalid run type. Please specify 'backporting' or 'baseline'.")
        sys.exit(1)

    print("Results to save: " , work_dir)
    print("Experiment type: " , run_type)
    print("Upgrade type: " , upgrade_type)
    print("LLM model: ", llm_model)
    print("Max fix: ", max_fix)
    print("Run on: ", run_on)
    print("Server: ", server_node)
    print("Continue from previous run: ", continue_from_prev)

    with open(SUBMISSION_NOAPI_PATH, "r", encoding="utf-8") as f:
        submission_noAPI = json.load(f)
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        kernel_meta = json.load(f)

    # if run_on == "cpu":
    #     group_env =  "upgrade/groups_by_exact_content_cpu.json"
    # else:
    #     group_env =  "upgrade/groups_by_exact_content_gpu.json"

    # with open(group_env, "r", encoding="utf-8") as f:
    #     groups_by_exact_content = json.load(f)

    submissions = collect_tasks(kernel_meta)
    print(f"Collected {len(submissions)} tasks from kernel metadata")
    mapped_tasks = map_task_versions(submissions, submission_noAPI)

    all_files = glob.glob("./baseline/notebooks/*.ipynb")
    file_w_one_cell = []
    for f in all_files:
        a = nbformat.read(f, as_version=4)
        if len(a.cells) == 1:
            file_w_one_cell.append(Path(f).name)

    Path(work_dir / 'script_out_allINone').mkdir(parents=True, exist_ok=True)
    os.makedirs(work_dir / 'script_out', exist_ok=True)
    os.makedirs(work_dir / 'csv_output', exist_ok=True)

    # (1) locate candidates -> init kernel-state (flat JSON keyed by filename)
    if state_backend == "postgres":
        # Use DSN from PG_DSN; no local sqlite/json files here
        dst_kernel_state_path = Path("postgres")  # placeholder for progress_monitor
        kernel_state_path = Path(f"./upgrade/{run_type}_{run_on}-{server_node}_candidates.json") if upgrade_type=="file" else Path(f"./upgrade/{run_type}_{run_on}-{server_node}_{upgrade_type}_candidates.json")

        if not continue_from_prev:
            print("Initializing Postgres kernel-state from candidates...")
            with open(kernel_state_path, "r", encoding="utf-8") as f:
                candidates = json.load(f)
                print(f"Existing kernel-state has {len(candidates)} entries")
            init_kernel_state_postgres(candidates)

        table_name = f"{upgrade_type}_{run_on}_{server_node}_{llm_model}" if run_on == 'cpu' else f"{upgrade_type}_{run_on}_{llm_model}"
        # if pg_table_exists(postgres_dsn, table_name) and continue_from_prev:
        #     kernel_state_path =  Path(f"./{work_dir}/{run_type}_{run_on}-{server_node}_{upgrade_type}_{llm_model}.json")
        #     migrate_json_to_postgres(kernel_state_path)
    elif state_backend == "redis":
        dst_kernel_state_path = Path("redis")  # placeholder for progress_monitor
        kernel_state_path = Path(f"./upgrade/{run_type}_{run_on}-{server_node}_candidates.json") if upgrade_type=="file" else Path(f"./upgrade/{run_type}_{run_on}-{server_node}_{upgrade_type}_candidates.json")
        if not continue_from_prev:
            print("Initializing Redis kernel-state from candidates...")
            with open(kernel_state_path, "r", encoding="utf-8") as f:
                candidates = json.load(f)
                print(f"Existing kernel-state has {len(candidates)} entries")
            init_kernel_state_redis(candidates)
        # if continue_from_prev:
        #     kernel_state_path =  Path(f"./{work_dir}/{run_type}_{run_on}-{server_node}_{upgrade_type}_{llm_model}.json")
        #     migrate_json_to_redis(kernel_state_path)
    else:
        dst_kernel_state_path = Path(f"./{work_dir}/{run_type}_{run_on}-{server_node}_{upgrade_type}_{llm_model}.json")
        if not continue_from_prev:
            kernel_state_path = Path(f"./upgrade/{run_type}_{run_on}-{server_node}_candidates.json") if upgrade_type=="file" else Path(f"./upgrade/{run_type}_{run_on}-{server_node}_{upgrade_type}_candidates.json")
            with open(kernel_state_path, "r", encoding="utf-8") as f:
                candidates = json.load(f)
                print(f"Existing kernel-state has {len(candidates)} entries")
            shutil.copy2(kernel_state_path, dst_kernel_state_path)
            init_kernel_state(dst_kernel_state_path, candidates, kernel_meta=kernel_meta)
        else:
            kernel_state_path = dst_kernel_state_path
            with open(kernel_state_path, "r", encoding="utf-8") as f:
                candidates = json.load(f)
                print(f"Existing kernel-state has {len(candidates)} entries")

    total_threads = mp.cpu_count()
    print(f"Total CPU threads available: {total_threads}")

    if args.role == "exec":
        # If we are strictly executing, we don't need to reserve CPUs for local LLM workers
        available_for_execution = floor(total_threads / 4)
        start_gpu_idx = 0
    else:
        # 'all' or 'llm' mode
        available_for_execution = floor((total_threads - args.llm_workers)/4) 
        start_gpu_idx = int(args.llm_workers/4)

    print(f"Available CPU threads for LLM {args.llm_workers} GPUs, for execution: {available_for_execution*4} GPUs ({available_for_execution} threads)")
    available_for_execution = floor(total_threads/4) 

    # llm_fix_worker(str(dst_kernel_state_path), llm_model, str(work_dir), upgrade_type, 0, max_fix)
    # print(int(args.llm_workers/4))

    # Start a single progress monitor process
    progress_q: mp.SimpleQueue = mp.SimpleQueue()
    mon = mp.Process(target=progress_monitor, args=(str(dst_kernel_state_path), max_fix, progress_q, state_backend), daemon=True)
    mon.start()
    
    # (2) Start LLM fix workers (threads)
    llm_processes = []
    if args.role in ["all", "llm"]:
        for gpu_id in range(args.llm_workers):  # GPUs 0-3 for LLM workers
            p = mp.Process(
                target=llm_fix_worker,
                args=(str(dst_kernel_state_path), llm_model, str(work_dir), upgrade_type, gpu_id, max_fix, 2, progress_q, state_backend)
            )
            llm_processes.append(p)
            p.start()
    

    # (3) Start GPU execution processes (claim from kernel-state)
    exec_processes = []
    if args.role in ["all", "exec"]:
        if run_on == "gpu":
            start_gpu_idx = 0
        # for gpu_id in range(0, int(available_for_execution)):  # GPUs 1-7
        for gpu_id in range(start_gpu_idx, int(available_for_execution)):  # GPUs 1-7
            p = mp.Process(
                target=gpu_exec_loop,
                args=(gpu_id, str(dst_kernel_state_path), setting, run_type, upgrade_type, mapped_tasks, max_fix, 2, progress_q, state_backend)
            )
            exec_processes.append(p)
            p.start()

    for p in llm_processes:
        p.join()
    for i, p in enumerate(exec_processes):
        p.join()  # This still waits, but now you see all started first
        print(f"Exec process {i} completed")

    # Stop monitor last
    progress_q.put(("STOP",))
    mon.join(timeout=5)

    if args.role in ["all", "exec"]:
        subprocess.run([f'docker kill $(docker ps -q)'], shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)