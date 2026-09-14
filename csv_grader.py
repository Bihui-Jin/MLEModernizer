import json
import subprocess
import os
import re
from tqdm import tqdm
from pathlib import Path
import fcntl  # for file locking
import argparse
import multiprocessing as mp
from typing import Any, Dict, Tuple, List

INVALID_RE = re.compile(r"Invalid submission:\s*(.*)")

def grade_submission(path, competition, gpu_id):
    # Run mlebench grade-sample and capture stdout
    proc = subprocess.run(
        ["taskset", "-c", f"{2*gpu_id},{2*gpu_id+1}", "mlebench", "grade-sample", path, competition],
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

    # Extract JSON payload (from first “{” to last “}”)
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
        if "Submission is missing the following" in invalid_reason and "Submission is missing the following columns" not in invalid_reason:
            pattern = re.compile(r"Submission is missing the following\s+([A-Za-z_]\w*)\s*:", re.IGNORECASE)
            missing_id = pattern.search(invalid_reason)
            missing_id = missing_id.group(1) if missing_id else "IDs"
            invalid_reason = f"Missing required {missing_id} in submission."
        out = {"error": f"Invalid submission: {invalid_reason}"}
        return out
    report = json.loads(out[start:end])
    
    return report

def _write_json_locked(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    os.replace(tmp_path, path)


def _worker_run(worker_id: int, chunk: List[Tuple[str, str]], run_dir: str, out_q: mp.Queue) -> None:
    """Run a chunk of jobs on a single worker_id (4 CPUs via taskset inside grade_submission)."""
    for nb_name, sub_rel in tqdm(chunk, desc=f"Worker {worker_id}", position=worker_id+1):
        competition = nb_name.split("/")[-1].split("_")[0]
        sub_path = Path(run_dir) / sub_rel
        try:
            report = grade_submission(sub_path, competition, worker_id)
        except Exception as e:
            report = {"error": str(e)}
        out_q.put((nb_name, report))
    out_q.put(None)
        
if __name__ == "__main__":
    # Run at mle-bench root
    # conda activate mle_env
    # cd mle-bench
    # python ../CodeModernization/csv_grader.py --workers 24 --save-dir ../CodeModernization/results/baseline
    # python ../CodeModernization/csv_grader.py --workers 24 --save-dir ../CodeModernization/results/downgrade/baseline
    parser = argparse.ArgumentParser(
        description="Grading CSV submissions."
    )
    parser.add_argument(
        "--workers",
        type=int,
        required=False,
        default=2,
        help="Number of workers to run in parallel",
    )
    parser.add_argument(
        "--run-dir",
        help="Path to the directory where all assets associated with the run are stored.",
        type=str,
        required=False,
        default="../CodeModernization",
    )
    parser.add_argument(
        "--save-dir",
        help="Path to the directory where all assets associated with the run are stored.",
        type=str,
        required=True,
        default=None,
    )
    args = parser.parse_args()
    print("# of workers: " , args.workers)
    print("Data path: " , args.run_dir)
    print("Save path: " , args.save_dir)

    json_path = Path(args.save_dir) / "executable_files_w_timer_parrallel.json"
    with open(json_path, "r") as f:
        exec_results = json.load(f)

    # prepare list of (nb_name, submission_path)
    jobs = []
    for nb_name, meta in exec_results.items():
        sub = meta.get("output")
        if "output" in meta and meta["execution_time"] <= 600:
            jobs.append((nb_name, sub))
    print(len(jobs), "notebooks to grade")

    save_path = Path(args.save_dir) / "csv_score.json"
    num_workers = max(1, min(args.workers, len(jobs) or 1))

    # Split jobs into num_workers chunks (round-robin)
    chunks: List[List[Tuple[str, str]]] = [[] for _ in range(num_workers)]
    for i, (nb_name, sub) in enumerate(jobs):
        chunks[i % num_workers].append((nb_name, sub))

    # grade in parallel
    result_q: mp.Queue = mp.Queue()
    procs: List[mp.Process] = []
    for wid in range(num_workers):
        if not chunks[wid]:
            continue
        p = mp.Process(target=_worker_run, args=(wid, chunks[wid], args.run_dir, result_q))
        p.start()
        procs.append(p)

    # Collect results with progress and flush after each result (locked writes)
    completed = 0
    workers_done = 0
    total = len(jobs)
    with tqdm(total=total, desc="Grading", position=0) as pbar:
        while workers_done < num_workers:
            item = result_q.get()
            if item is None:
                workers_done += 1
                continue
            nb_name, report = item
            exec_results.setdefault(nb_name, {})
            exec_results[nb_name]["score"] = report
            completed += 1
            pbar.update(1)
            if completed % 25 == 0:  # Flush periodically
                _write_json_locked(save_path, exec_results)

    for p in procs:
        p.join()
    
    # Final write
    _write_json_locked(save_path, exec_results)
    print(f"Wrote results to: {save_path}")