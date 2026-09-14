import subprocess
import time
import signal
import os
from tqdm import tqdm
from threading import Thread, Event
import queue
import json
import tempfile
import shutil
from pathlib import Path
import re
import select
import threading
import fcntl  # for file locking
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from collections import defaultdict
import heapq

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
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
JSON_PATH = "apiDowngrade/kernel_w_pyVersion.json"
SUBMISSION_NOAPI_PATH = "apiDowngrade/submission_noAPI.json"

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
                            start_time = time.time()
                            start_event.set()
                            execution_queue.put(('status', 'Execution started'))
                    
                # Check timeout only after execution has started
                if execution_started and start_time:
                    elapsed_time = time.time() - start_time

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

        try:
            # If the process is still running
            while process.poll() is None:
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
                    result['execution_time'] = (time.time() - start_time) if start_time else self.timeout_seconds

                    # Drain any queued output before returning
                    while True:
                        try:
                            msg_type, msg_data = execution_queue.get(timeout=0.1)
                        except queue.Empty:
                            break
                        if msg_type == "output":
                            all_lines.append(msg_data)
                        elif msg_type == "status" and "Execution started" in msg_data:
                            start_time = time.time()

                    result["detail"] = "\n".join(all_lines)
                    if process.returncode not in (0, None):
                        result["error"] = result["detail"]

                    return result
                    
                # Drain queue messages
                while True:
                    try:
                        msg_type, msg_data = execution_queue.get(timeout=0.1)
                    except queue.Empty:
                        break

                    if msg_type == "output":
                        all_lines.append(msg_data)
                    elif msg_type == "status" and "Execution started" in msg_data:
                        start_time = time.time()

                time.sleep(0.1)
            
            # Process completed: drain remaining queue
            while True:
                try:
                    msg_type, msg_data = execution_queue.get(timeout=0.1)
                except queue.Empty:
                    break
                if msg_type == "output":
                    all_lines.append(msg_data)
                elif msg_type == "status" and "Execution started" in msg_data and start_time is None:
                    start_time = time.time()

            if start_time:
                result['execution_time'] = time.time() - start_time
            # else:
            #     result['execution_time'] = time.time() - abs_start_time

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

def collect_tasks(kernel):
    tasks = []
    for compt, files in kernel.items():
        for fname, meta in files.items():
            if ("ps" in meta and float(meta["ps"]) != 0.0 and meta['runtime'] <= 600 and len(meta['datasets']) <= 1 and "R" not in meta):
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

def build_docker_command(k_token, temp_dir, compt, filename, gpu, dst, run_type, mapped_tasks):
    """Build Docker command with only existing directory mounts
    # Run ./zip.sh first (preparation step)
    # Allow docker to accessand mount
    chmod -R a+rw {home_dir}/.cache
    chmod -R a+rw {home_dir}/.cache/mle-bench/data
    # Allow to save files in docker
    chmod -R a+rw {home_dir}/mle-bench-internal/docker-test/scripts
    # Create new docker image with pre pip install via {home_dir}/mle-bench-internal/docker-test/Dockerfile.base
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

    container_name = f"gpu_{gpu}_{filename.split('.')[0]}"

    key = filename.split('.')[0]
    trunc_env_name, req_name, version = mapped_tasks[key]
    if req_name == "_skip_":
        trunc_env_name = f"empty_{version}"

    cmd = f'docker run --rm -i --name {container_name} --shm-size=30g'
    cmd += f' --cpuset-cpus="{4*gpu},{4*gpu+1},{4*gpu+2},{4*gpu+3}"'
    cmd += f" --gpus device={gpu}"
    cmd += ' -e PYTHONWARNINGS="ignore"'
    cmd += f' -e KAGGLE_USER_SECRETS_TOKEN="{k_token}"'
    # cmd += f' -e PYTHONUNBUFFERED=1'  # Ensure immediate output
    # cmd += f' -e PYDEVD_DISABLE_FILE_VALIDATION=1'
    # cmd += f' -e PYTHONFROZEN=0'
    cmd += f" -e tasks='{trunc_env_name} {req_name} {version}'"
    cmd += f' -e JUPYTER_PATH="/opt/venvs/_kernels/share/jupyter"'
    cmd += f' -v {temp_dir}:/kaggle/working'
    cmd += f' -v {dst}/prepared/public:/kaggle/input'
    cmd += f" -v {dst}/prepared/public:/kaggle/input/{compt}"
    cmd += f' -v {dst}/prepared/public:/kaggle/working/{compt}'
    cmd += f' -v {dst}/prepared/public:/kaggle/data'
    cmd += f' -v {dst}/prepared/public:/kaggle/data/{compt}'
    cmd += "".join([f" -v {mount}" for mount in volume_mounts])
    cmd += f" -w /kaggle/working kaggle_coding"
    # cmd += " -lc 'set -euxo pipefail; ls -la; cd ../input/rsna-2022-cervical-spine-fracture-detection; ls'"
    # cmd += f" jupyter nbconvert --to notebook --inplace --execute {filename} --ExecutePreprocessor.allow_errors=True --ExecutePreprocessor.timeout=-1"
    # cmd += f" timeout {timeout_seconds*1.1} jupyter nbconvert --to notebook --inplace --execute {filename} --ExecutePreprocessor.allow_errors=True --ExecutePreprocessor.timeout=-1"
    # cmd += f" python -Xfrozen_modules=off -m jupyter nbconvert --to notebook --stdout --execute {filename} --ExecutePreprocessor.allow_errors=True --ExecutePreprocessor.timeout=-1"
    if run_type == "baseline":
        cmd += f" bash -lc 'set -e; trap \"exit 0\" TERM && jupyter nbconvert --to notebook --inplace --execute {filename} --ExecutePreprocessor.allow_errors=True --ExecutePreprocessor.timeout=-1'"
    else:
        cmd += f" bash -lc 'set -e; trap \"exit 0\" TERM && bash /opt/install_envs.sh && python -m jupyter nbconvert --to notebook --inplace --execute {filename} --ExecutePreprocessor.kernel_name=\"{trunc_env_name}\" --ExecutePreprocessor.allow_errors=True --ExecutePreprocessor.timeout=-1'"
    
    return cmd, container_name

def clear_notebook_outputs(notebook_path):
    """Clear all outputs from a Jupyter notebook file"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # 1) Clear outputs and execution counts
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                cell['outputs'] = []
                cell['execution_count'] = None
        
        # 2) Remove first cell if it is exactly our pip‐install stub
        cells = notebook.get('cells', [])
        if cells and cells[0].get('cell_type') == 'code':
            src = cells[0].get('source', '')
            first_src = ''.join(src) if isinstance(src, list) else src
            if '%pip install Unidecode monai ttach optuna optuna-integration' in first_src.strip():
                cells.pop(0)
        notebook['cells'] = cells

        # Save the cleared notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error clearing notebook outputs at {notebook_path}: \n{e}")
        return False


def merge_gpu_results(setting):
    """Merge all GPU-specific JSON files into one final file"""
    merged_results = {}
    
    # if setting == "test":
    file_name = work_dir / 'executable_files_w_timer_parrallel.json' 
    for gpu_id in range(8):
        json_filename = work_dir / f'parrallel_results/executable_files_w_timer_gpu_{gpu_id}.json'
        if os.path.exists(json_filename):
            with open(json_filename, 'r', encoding='utf-8') as f:
                gpu_results = json.load(f)
                merged_results.update(gpu_results)
    # else:
    #     file_name = 'executable_files_w_timer_parrallel_full.json'
    #     for gpu_id in range(8):
    #         json_filename = f'./baseline/parrallel_results/executable_files_w_timer_gpu_{gpu_id}_full.json'
    #         if os.path.exists(json_filename):
    #             with open(json_filename, 'r', encoding='utf-8') as f:
    #                 gpu_results = json.load(f)
    #                 merged_results.update(gpu_results)

    #     with open("executable_files_w_timer_parrallel.json", 'r', encoding='utf-8') as f:
    #             gpu_results = json.load(f)
    #             merged_results.update(gpu_results)
    
    # Write merged results
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(dict(sorted(merged_results.items())), f, indent=2, ensure_ascii=False)

    print(f"Merged results from {len(merged_results)} entities")



def split_competitions_balanced(b, c, num_groups=8):
    """
    More balanced splitting using dynamic programming approach
    """
    # Calculate total times
    competitions = []
    for comp in b.keys():
        if comp in c:
            total_time = b[comp] * c[comp]
            competitions.append((comp, total_time))
    
    # Sort by total time
    competitions.sort(key=lambda x: x[1], reverse=True)
    
    # Initialize groups
    groups = [[] for _ in range(num_groups)]
    group_totals = [0.0] * num_groups
    
    # For better balance, consider multiple assignment options for each competition
    for comp, total_time in competitions:
        # Find the group that would result in the most balanced distribution
        best_group = 0
        min_max_difference = float('inf')
        
        for g in range(num_groups):
            # Calculate what the max difference would be if we add to this group
            temp_totals = group_totals.copy()
            temp_totals[g] += total_time
            max_diff = max(temp_totals) - min(temp_totals)
            
            if max_diff < min_max_difference:
                min_max_difference = max_diff
                best_group = g
        
        # Add to best group
        groups[best_group].append((comp, total_time))
        group_totals[best_group] += total_time
    
    return groups, group_totals

def assign_files_to_groups_balanced(
    sample_files: List[str],
    per_file_stats: Dict[str, Any],
    comp_avg_time: Dict[str, float],
    comp_scripts_per_group: Dict[str, Dict[int, int]],
    num_groups: int,
) -> Dict[int, List[str]]:
    """
    Assign notebooks to groups while:
      1) respecting required per-group per-competition counts (comp_scripts_per_group)
      2) mixing variants by distributing slow notebooks across groups (LPT + min-loaded bin among eligible groups)

    per_file_stats: your `data` dict loaded from executable_files_w_timer_parrallel.json
                   used to read per-file 'process_time' if present
    """
    # needs[gid][comp] = remaining required notebooks for this (gid, comp)
    needs: List[Dict[str, int]] = [defaultdict(int) for _ in range(num_groups)]
    for comp, gmap in comp_scripts_per_group.items():
        for gid, cnt in (gmap or {}).items():
            if 0 <= gid < num_groups and cnt:
                needs[gid][comp] += int(cnt)

    # quick sanity: total required should match total files (usually true)
    total_needed = sum(sum(m.values()) for m in needs)

    def est_time(fname: str) -> float:
        info = per_file_stats.get(fname, {})
        t = info.get("process_time", None)
        try:
            if t is not None:
                return float(t)
        except Exception:
            pass
        comp = fname.split("_", 1)[0]
        return float(comp_avg_time.get(comp, 0.0))

    # Build jobs (time, comp, filename)
    jobs = []
    for f in sample_files:
        comp = f.split("_", 1)[0]
        jobs.append((est_time(f), comp, f))
    jobs.sort(key=lambda x: x[0], reverse=True)  # LPT

    group_load = [0.0] * num_groups
    assigned: Dict[int, List[str]] = {gid: [] for gid in range(num_groups)}
    unassigned = []

    for t, comp, fname in jobs:
        # eligible groups are those that still need this competition
        elig = [gid for gid in range(num_groups) if needs[gid].get(comp, 0) > 0]
        if not elig:
            unassigned.append((t, comp, fname))
            continue

        # pick least-loaded eligible group
        gid = min(elig, key=lambda g: group_load[g])
        assigned[gid].append(fname)
        group_load[gid] += t
        needs[gid][comp] -= 1
        if needs[gid][comp] <= 0:
            del needs[gid][comp]

    # If anything left unassigned (should be rare), place into least loaded groups (best-effort)
    # keeps the pipeline moving even if comp_scripts_per_group doesn't perfectly match sample_files.
    for t, comp, fname in unassigned:
        gid = min(range(num_groups), key=lambda g: group_load[g])
        assigned[gid].append(fname)
        group_load[gid] += t

    # warn if the plan didn’t match counts
    if total_needed != len(sample_files):
        print(f"[assign_files_to_groups_balanced] warning: total_needed={total_needed} != total_files={len(sample_files)}")
    if unassigned:
        print(f"[assign_files_to_groups_balanced] warning: {len(unassigned)} files had no remaining 'need' slot; assigned best-effort.")

    return assigned

def split_competitions_balanced_multiple(b, c, num_groups=12):
    """
    Balance workload across `num_groups` by treating each script as an independent job
    with estimated runtime `b[comp]` and scheduling using LPT onto `num_groups` bins.

    Returns:
      - groups: List[List[Tuple[str, float]]] where each inner list contains (comp, time_for_comp_in_group)
      - group_totals: total estimated time per group
      - comp_usage_count: how many scripts of each comp were assigned in total (should equal c[comp])
      - comp_scripts_per_group: {comp: {group_id: script_count}}
    """
    if num_groups <= 0:
        raise ValueError("num_groups must be > 0")

    # Fill missing comps similarly to your previous behavior
    avg_exec_time = (sum(b.values()) / len(b)) if b else 0.0
    avg_scripts = (sum(c.values()) / len(c)) if c else 0.0

    all_comps = set(b.keys()) | set(c.keys())
    for comp in all_comps:
        if comp not in b:
            b[comp] = avg_exec_time
        if comp not in c:
            c[comp] = int(avg_scripts)

    # Build per-script job list: each script is one job with cost b[comp]
    jobs = []
    for comp, n_scripts in c.items():
        n = int(n_scripts) if n_scripts is not None else 0
        if n <= 0:
            continue
        cost = float(b.get(comp, avg_exec_time))
        jobs.extend([(cost, comp)] * n)

    # Sort by descending runtime (LPT)
    jobs.sort(key=lambda x: x[0], reverse=True)

    # Min-heap of bins: (current_total, group_id)
    heap = [(0.0, gid) for gid in range(num_groups)]
    heapq.heapify(heap)

    group_totals = [0.0] * num_groups
    comp_scripts_per_group = {comp: {} for comp in all_comps}
    comp_usage_count = {comp: 0 for comp in all_comps}

    for cost, comp in jobs:
        cur_total, gid = heapq.heappop(heap)

        # assign this script to gid
        new_total = cur_total + cost
        group_totals[gid] = new_total

        d = comp_scripts_per_group[comp]
        d[gid] = d.get(gid, 0) + 1
        comp_usage_count[comp] += 1

        heapq.heappush(heap, (new_total, gid))

    # Build `groups` in the format your downstream code expects:
    # groups[gid] = [(comp, b[comp]*count_in_gid), ...]
    groups = [[] for _ in range(num_groups)]
    for comp, per_group in comp_scripts_per_group.items():
        for gid, count in per_group.items():
            if count <= 0:
                continue
            groups[gid].append((comp, float(b[comp]) * count))

    # Optional: sort each group by descending contributed time (nice for debugging/printing)
    for gid in range(num_groups):
        groups[gid].sort(key=lambda x: x[1], reverse=True)

    return groups, group_totals, comp_usage_count, comp_scripts_per_group


def process_gpu_files_separate(gpu_id, parrallel_groups, setting, run_type, mapped_tasks):  
    # print(f"GPU {gpu_id}: Processing {len(parrallel_groups)} files")
    # Pin this process and all children to specific CPUs
    cpu_start = 4 * gpu_id
    cpu_end = cpu_start + 3
    cpus = list(range(cpu_start, cpu_end + 1))

    os.sched_setaffinity(0, set(cpus))

    timeout_seconds = 600
    # Create output directory if it doesn't exist
    output_dir = work_dir / 'csv_output'
    os.makedirs(output_dir, exist_ok=True)

    with open('../config.json', 'r', encoding='utf-8') as file:
        config = json.load(file)
    k_token = config['kaggle']

    expected = "./baseline/notebooks"
    nb_out = work_dir / 'script_out'
    results = {}

    # Use separate JSON file for each GPU
    json_filename = work_dir / f'parrallel_results/executable_files_w_timer_gpu_{gpu_id}.json'
    os.makedirs(work_dir / 'parrallel_results', exist_ok=True)

    if setting:
        all_files = [f for f in os.listdir(expected) if f.endswith('.ipynb')][:32]
        parrallel_groups = [f for f in all_files if f.split("_")[0] in parrallel_groups]

    # for filename in tqdm(os.listdir(expected)):
    for filename in tqdm(parrallel_groups, desc=f"GPU {gpu_id}",
                        position=gpu_id,  # Each GPU gets its own line
                        leave=True):      # Keep bar visible after completion
        # print(f"\r\n{filename}", end='', flush=True)
        p_time_start = time.time()

        parts = filename.split("_")
        compt = parts[0]
        notebook_name = "_".join(parts[1:-2])
        version = parts[-2]
        out_path = nb_out / compt / notebook_name / version

        # # Clear all outputs from the notebook file before processing
        # notebook_path = os.path.join(expected, filename)
        # if not clear_notebook_outputs(notebook_path):
        #     print(f"Failed to clear outputs from {filename}")
        
        # Create temporary working directory
        temp_dir = tempfile.mkdtemp(prefix=f'gpu_{gpu_id}_')
        shutil.copy2(os.path.join(expected, filename), temp_dir)

        subprocess.run([f'chmod -R a+rw {temp_dir}'], shell=True, check=True)

        # Snapshot before run (existing files)
        before = set(Path(temp_dir).glob("*.csv"))


        upperdir = tempfile.mkdtemp(prefix=f'overlay_upper_{gpu_id}_')
        workdir = tempfile.mkdtemp(prefix=f'overlay_work_{gpu_id}_')
        dst = work_dir / f'mount_overlay/{filename.split(".")[0]}'
        os.makedirs(dst, exist_ok=True) # should be always empty after execution


        try:
            # sudo mount -t overlay overlay  -o lowerdir="../mle-bench-internal/docker-test/test",upperdir="/tmp/overlay-upper",workdir="/tmp/overlay-work" .
            subprocess.run([f'sudo mount -t overlay overlay -o lowerdir="../.cache/mle-bench/data/{compt}",upperdir="{upperdir}",workdir="{workdir}" "{dst}"'], shell=True, check=True)

            # cleanup_cmd = f'docker ps -aq --filter "name=gpu_{gpu_id}_*" | xargs -r docker rm -f'
            # subprocess.run([cleanup_cmd], shell=True, 
            #             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while not (os.path.exists(f"{dst}/prepared/public") and os.path.isdir(f"{dst}/prepared/public")):  # Ensure mount is ready
                time.sleep(0.1)
                
            cmd,container_name = build_docker_command(k_token, temp_dir, compt, filename, gpu_id, dst, run_type, mapped_tasks)
            
            # Run notebooks with timeout monitoring
            runner = NotebookRunner(timeout_seconds)
            result = runner.run_single_notebook(cmd, compt, filename)
            results[filename] = result

            # Move the nb file (w/ outputs) to expected directory
            temp_notebook_path = os.path.join(temp_dir, filename)

            # save nb back to new dir e.g., ./scripts_out
            if not os.path.exists(out_path):
                os.makedirs(out_path, exist_ok=True)

            if os.path.exists(temp_notebook_path):
                shutil.copy(temp_notebook_path, out_path)
                shutil.copy(temp_notebook_path, work_dir / 'script_out_allINone')

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
            results.setdefault(filename, {})
            results[filename]['error'] = str(traceback.format_exc())

        # Cleanup temp dir
        start = time.time()

        subprocess.run([f'docker kill {container_name}'], shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT)

        try:
            subprocess.run([f'sudo umount {dst}'], shell=True, check=True)
        except:
            # "target is busy" commonly -> try lazy umount
            subprocess.run([f'sudo umount -l {dst}'], shell=True, check=False)

        subprocess.run([f'sudo rm -rf {temp_dir}'], shell=True, check=True)
        subprocess.run([f'sudo rm -rf {upperdir}'], shell=True, check=True)
        subprocess.run([f'sudo rm -rf {workdir}'], shell=True, check=True)

        try:
            subprocess.run([f'rm -rf {dst}'], shell=True, check=True)
        except Exception as e:
            subprocess.run([f'sudo umount -f {dst} 2>/dev/null || true'], shell=True)
            subprocess.run([f'rm -rf {dst}'], shell=True)

        end = time.time()
        results[filename]['cleanup_time'] = end - start
    

        p_time_end = time.time()
        results[filename]['process_time'] = p_time_end - p_time_start

        with open(out_path / 'result.json', 'w', encoding='utf-8') as f:
            json.dump(results[filename], f, indent=2, ensure_ascii=False)

        with open(json_filename, 'w', encoding='utf-8') as file:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            json.dump(results, file, indent=2, ensure_ascii=False)
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)

import multiprocessing as mp
import sys
if __name__ == "__main__":
    # conda activate mle_env
    # python baseline/run_baseline_w_timer_parrallel.py --run-dir ./results/downgrade/baseline --type backporting
    # python baseline/run_baseline_w_timer_parrallel.py --run-dir ./results/baseline --type baseline
    parser = argparse.ArgumentParser(
        description="Run the CSV submissions."
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
        help="Path to the directory where all assets associated with the run are stored, e.g., ./results/baseline",
        type=str,
        required=True,
        default=None,
    ) 
    parser.add_argument(
        "--type",
        help="Experiment type for baseline or backporting",
        type=str,
        required=True,
        default=None,
    )
    args = parser.parse_args()
    

    setting = args.test
    work_dir = Path(args.run_dir)
    run_type = args.type
    if not (run_type == "backporting" or run_type == "baseline"):
        print("Invalid run type. Please specify 'backporting' or 'baseline'.")
        sys.exit(1)

    print("Results to save: " , work_dir)
    print("Experiment type: " , run_type)

    with open(SUBMISSION_NOAPI_PATH, "r", encoding="utf-8") as f:
        submission_noAPI = json.load(f)
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        kernel = json.load(f)

    submissions = collect_tasks(kernel)
    mapped_tasks = map_task_versions(submissions, submission_noAPI)



    os.makedirs(work_dir / 'script_out_allINone', exist_ok=True)
    os.makedirs(work_dir / 'script_out', exist_ok=True)
    os.makedirs(work_dir / 'csv_output', exist_ok=True)

    compt = []

    with open('./results/downgrade/baseline2/executable_files_w_timer_parrallel.json', 'r') as f:
        data = json.load(f)
    for entity, info in data.items():
        compt.append(entity.split("_")[0])

    a = list(set(compt))
    b = {value: [] for value in a}
    c = {value: 0 for value in a}

    for entity, info in data.items():
        if "process_time" in info:
            b[entity.split("_")[0]].append(info["process_time"])

    for key, val in b.items():
        b[key] = sum(val) / len(val) if val else 0

    full_files = [f for f in os.listdir('./baseline/notebooks') if f.endswith('.ipynb')]
    expected = work_dir / 'script_out_allINone'
    # Remove sample scripts
    done_already = [f for f in os.listdir(expected) if f.endswith('.ipynb')]
    sample_files = [x for x in full_files if x not in done_already]

    print(len(sample_files), setting)
    for entity in sample_files:
        if entity.split("_")[0] in c:
            c[entity.split("_")[0]] += 1
        else:
            c[entity.split("_")[0]] = 1

    if setting:
        del b['spaceship-titanic']
        del c['spaceship-titanic']

        # Use the function
        groups, group_totals = split_competitions_balanced(b, c, 7)

        parrallel_groups = {0: ["spaceship-titanic"]}

        # Display results
        for i, (group, total) in enumerate(zip(groups, group_totals)):
            parrallel_groups[i+1] = [comp for comp, time in group]
    else:
        # Use the function with multi-group assignment
        groups, group_totals, comp_usage, comp_scripts_per_group = split_competitions_balanced_multiple(b, c, 8)

        # Sssignment that respects per-group per-comp counts and mixes slow variants
        parrallel_groups = assign_files_to_groups_balanced(
            sample_files=sample_files,
            per_file_stats=data,          # loaded earlier from executable_files_w_timer_parrallel.json
            comp_avg_time=b,              # avg per-comp runtime
            comp_scripts_per_group=comp_scripts_per_group,
            num_groups=12,
        )


    total_assigned = sum(len(files) for files in parrallel_groups.values())
    print(f"Total assigned files: {total_assigned} out of {len(sample_files)}")
    for group_id in range(len(parrallel_groups)):
        print(f"Group {group_id}: {len(parrallel_groups[group_id])}")

    # parrallel_groups[0] = parrallel_groups[0][:2]
    # process_gpu_files_separate(0, parrallel_groups[0], setting, run_type, mapped_tasks)
    # # Create processes for each GPU
    processes = []
    for gpu_id in range(8):  # GPUs 0-7
        p = mp.Process(
            target=process_gpu_files_separate,
        args=(gpu_id, parrallel_groups[gpu_id], setting, run_type, mapped_tasks)
        )
        processes.append(p)
        p.start()

    for i, p in enumerate(processes):
        p.join()  # This still waits, but now you see all started first
        print(f"GPU {i} process completed")

    subprocess.run([f'docker kill $(docker ps -q)'], shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT)

    os.sync() 
    time.sleep(2)
    # Merge all GPU results into final file
    merge_gpu_results(setting)
    