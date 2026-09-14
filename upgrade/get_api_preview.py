import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import os
import re
import multiprocessing as mp
from tqdm import tqdm
import argparse
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from LLMs.discern_files import no_gpu_required

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

def build_docker(filename: str, compt: str, gpu: int):
    
    key = filename.split('.')[0]
    trunc_env_name, req_name, version = mapped_tasks[key]
    if req_name == "_skip_":
        trunc_env_name = f"empty_{version}"

    cmd = f'docker run --rm -i --shm-size=30g'
    cmd += f' --cpuset-cpus="{gpu}"'
    if run_on == 'gpu':
        cmd += f' --gpus device={gpu % 8}' 
    # cmd += ' -e PYTHONWARNINGS="ignore"'
    cmd += ' -e USE_PINNED_REQ=0'
    cmd += f" -e tasks='{trunc_env_name} {req_name} {version}'"
    cmd += " kaggle_coding"
    cmd += f"  bash -lc 'bash /opt/install_envs.sh && /opt/venvs/{trunc_env_name}/bin/python -m pip list --format=freeze'"

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            # timeout=1800,  # was 600
        )
        
        raw = result.stdout or ""
        cleaned = ANSI_RE.sub("", raw)

        # Extract pip freeze output (after "Environment processed." line)
        freeze_section = ""
        in_freeze = False
        for line in cleaned.splitlines():
            if "Environment processed." in line:
                in_freeze = True
                continue
            if in_freeze:
                freeze_section += line + "\n"

        # Keep only lines that look like pip freeze entries
        freeze_lines = []
        for line in freeze_section.splitlines():
            line = line.strip()
            if re.match(r"^[A-Za-z0-9_.+\-]+==.+$", line):
                freeze_lines.append(line)

        pip_freeze = "\n".join(freeze_lines)
        if pip_freeze:
            pip_freeze += "\n"

        # Save human-readable output
        if req_name == "_skip_":
            out_path = Path(output_txt_dir) / f"empty_{version}.txt"
        else:
            out_path = Path(output_txt_dir) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(pip_freeze, encoding="utf-8")
        return pip_freeze or {compt: "No pip output captured"}

    except subprocess.TimeoutExpired:
        return {compt: "Error: Container execution timeout"}
    except Exception as e:
        return {compt: f"Error: {str(e)}"}


# def _worker_process(group_chunk: list, gpu_id: int) -> None:
#     """Process a chunk of groups, each group is a list of filenames."""
#     for group in tqdm(group_chunk, desc=f"cpu {gpu_id}:", position=gpu_id+1, leave=False):
#         filename = group[0]  # representative filename
#         compt = filename.split('_')[0]
#         build_docker(filename, compt, gpu=gpu_id)

def _worker_process(wid: int, jobs: List[str], gpu_id: int, result_q: "mp.Queue") -> None:
    """Each job is a representative filename (e.g., group[0])."""
    try:
        for filename in tqdm(jobs, desc=f"worker {wid}", position=wid + 1, leave=False):
            compt = filename.split("_")[0]
            report = build_docker(filename, compt, gpu=gpu_id)
            result_q.put((filename, report))
    finally:
        result_q.put(None)

if __name__ == "__main__":
    # python upgrade/get_api_preview.py --run-on cpu --num-workers 48
    # python upgrade/get_api_preview.py --run-on gpu --num-workers 32
    parser = argparse.ArgumentParser(
        description="Generate API list for each environment."
    )
    parser.add_argument(
        "--run-on",
        type=str,
        required=True,
        default=None,
        help="Experiment running on: cpu or gpu",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        required=True,
        default=32,
        help="Number of parallel workers",
    )

    args = parser.parse_args()
    run_on = args.run_on
    num_workers = args.num_workers

    pip_result = subprocess.run(
    ["docker", "run", "--rm", "kaggle_coding", "pip", "list", "--format=freeze"],
    capture_output=True,
    text=True,
    check=True
    )
    pip_list_path = Path("./upgrade/pip_list.txt")
    pip_list_path.write_text(pip_result.stdout, encoding="utf-8")
    print(f"Pip list saved to: {pip_list_path}")

    JSON_PATH = "apiDowngrade/kernel_w_pyVersion.json"
    SUBMISSION_NOAPI_PATH = "apiDowngrade/submission_noAPI.json"
    RESULT_PATH = "results/downgrade/baseline/executable_files_w_timer_parrallel.json"
    output_txt_dir = "upgrade/pipList"
    os.makedirs(output_txt_dir, exist_ok=True)
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

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        kernel_meta = json.load(f)
    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        executable_files = json.load(f)

    no_gpu_required_list = no_gpu_required(kernel_meta)
    executable_files = [k for k,v in executable_files.items() if "Backporting Failed. Aborting." not in v['detail'] and "[NbConvertApp] Writing "  in v['detail'] and v['execution_time'] <= 600]

    filtered_meta = {}
    for compt, submissions in kernel_meta.items():
        filtered_subs = {}
        for sub_html, meta in submissions.items():
            sub_ipynb = sub_html.replace(".html", ".ipynb")
            full_key = f"{compt}_{sub_ipynb}"

            if run_on == "cpu":
                if full_key in no_gpu_required_list and full_key in executable_files:
                    filtered_subs[sub_html] = meta
                group_env = "upgrade/groups_by_exact_content_cpu.json"
            else:
                if full_key not in no_gpu_required_list and full_key in executable_files:
                    filtered_subs[sub_html] = meta
                group_env = "upgrade/groups_by_exact_content_gpu.json"

        if filtered_subs:
            filtered_meta[compt] = filtered_subs
    kernel_meta = filtered_meta
    print(f"Total {run_on}-required environments: {sum(len(v) for v in kernel_meta.values())}")

    with open(SUBMISSION_NOAPI_PATH, "r", encoding="utf-8") as f:
        submission_noAPI = json.load(f)

    with open(group_env, "r", encoding="utf-8") as f:
        groups_by_exact_content = json.load(f)


    submissions = collect_tasks(kernel_meta)
    mapped_tasks = map_task_versions(submissions, submission_noAPI)

    # Append one env per unique version with req_name == "_skip_"
    seen_skip_versions = set()
    for key, (trunc_env_name, req_name, version) in mapped_tasks.items():
        if req_name == "_skip_" and version not in seen_skip_versions:
            groups_by_exact_content.append([trunc_env_name])
            seen_skip_versions.add(version)

    # group = ['aerial-cactus-identification_nuldiego_baseline-cactus-cnn-with-keras_v10_C1.txt']
    # print(group)
    # filename = group[0]  # representative filename
    # compt = filename.split('_')[0]
    # build_docker(filename, compt, gpu=0)

    # Flatten "groups" into representative jobs (same as before: group[0])
    jobs: List[str] = [group[0] for group in groups_by_exact_content if group]

    # Split jobs into num_workers chunks (round-robin)
    chunks: List[List[str]] = [[] for _ in range(num_workers)]
    for i, filename in enumerate(jobs):
        chunks[i % num_workers].append(filename)
    chunks = [c for c in chunks if c]  # drop empties
    print(f"Total jobs: {len(jobs)}, total workers used: {len(chunks)}")

    # Run workers
    result_q: mp.Queue = mp.Queue()
    procs: List[mp.Process] = []
    for wid, chunk in enumerate(chunks):
        gpu_id = wid % num_workers
        p = mp.Process(target=_worker_process, args=(wid, chunk, gpu_id, result_q))
        p.start()
        procs.append(p)

    # Collect results with progress + periodic flush
    workers_done = 0
    expected_done = len(procs)
    total = len(jobs)

    with tqdm(total=total, desc="Processing", position=0) as pbar:
        while workers_done < expected_done:
            item = result_q.get()
            if item is None:
                workers_done += 1
                continue

            pbar.update(1)

    for p in procs:
        p.join()

    print("All groups processed!")
