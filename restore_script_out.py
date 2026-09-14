# restore /result/../csv_output/*
import json
import os
import shutil
from multiprocessing import Process
from pathlib import Path
from typing import Any
from tqdm import tqdm
# Directory containing baseline results (parent of script_out, csv_output, etc.)
ROOT = Path("./results/downgrade/baseline").resolve()
ROOT = Path("./results/baseline").resolve()
ROOT = Path("./results/upgrade/gpt_file").resolve()
ROOT = Path("./results/upgrade/oss_file").resolve()
ROOT = Path("./results/downgrade/cell").resolve()

EXEC_JSON = ROOT / "executable_files_w_timer_parrallel.json"
SCRIPT_FLAT = ROOT / "script_out_allINone"
CSV_DIR = ROOT / "csv_output"
SCRIPT_OUT = ROOT / "script_out"


N_WORKERS = 24
CPUS_PER_WORKER = 2

def parse_script_key(script_key: str):
    parts = script_key.split("_")
    compt = parts[0]
    notebook_name = "_".join(parts[1:-2])
    version = parts[-2]
    return compt, notebook_name, version

def split_evenly(items: list, n: int) -> list[list]:
    """Split items into n chunks; sizes differ by at most one."""
    if n <= 0:
        raise ValueError("n must be positive")
    q, r = divmod(len(items), n)
    out, start = [], 0
    for i in range(n):
        end = start + q + (1 if i < r else 0)
        out.append(items[start:end])
        start = end
    return out

def process_records(worker_id: int, records: list[tuple[str, dict[str, Any]]]) -> None:
    for script_key, payload in tqdm(records, position=worker_id+1, leave=False):
        compt, notebook_name, version = parse_script_key(script_key)
        dest_dir = SCRIPT_OUT / compt / notebook_name / version
        dest_dir.mkdir(parents=True, exist_ok=True)

        src_nb = SCRIPT_FLAT / script_key
        if not src_nb.is_file():
            print(f"skip (missing notebook): {src_nb}")
            continue
        shutil.copy2(src_nb, dest_dir / script_key)

        csv_name = script_key.replace(".ipynb", ".csv")
        src_csv = CSV_DIR / csv_name
        if src_csv.is_file():
            shutil.copy2(src_csv, dest_dir / csv_name)

        result_path = dest_dir / "result.json"
        with result_path.open("w", encoding="utf-8") as out:
            json.dump(payload, out, indent=2, ensure_ascii=False)


def worker(worker_id: int, records: list[tuple[str, dict[str, Any]]]) -> None:
    cpus = {worker_id * CPUS_PER_WORKER + j for j in range(CPUS_PER_WORKER)}
    try:
        os.sched_setaffinity(0, cpus)
    except OSError as e:
        print(f"worker {worker_id}: sched_setaffinity {cpus} failed: {e}")
    process_records(worker_id, records)


def main() -> None:
    ncpu = os.cpu_count() or 0
    need = N_WORKERS * CPUS_PER_WORKER
    if ncpu < need:
        print(f"warning: {ncpu=} < {need} (24x2); affinity may still run but CPUs can be oversubscribed")

    with EXEC_JSON.open(encoding="utf-8") as f:
        data = json.load(f)

    items = list(data.items())
    chunks = split_evenly(items, N_WORKERS)
    procs: list[Process] = []
    for i, chunk in enumerate(chunks):
        p = Process(target=worker, args=(i, chunk))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

if __name__ == "__main__":
    main()