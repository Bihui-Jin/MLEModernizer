from pathlib import Path
import hashlib
import json
from tqdm import tqdm
import re
import argparse
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from LLMs.discern_files import no_gpu_required

base_dir = Path("apiDowngrade/apiDowngradeList")
JSON_PATH = "apiDowngrade/kernel_w_pyVersion.json"
SUBMISSION_NOAPI_PATH = "apiDowngrade/submission_noAPI.json"
RESULT_PATH = "results/downgrade/baseline/executable_files_w_timer_parrallel.json"
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

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk_size), b""):
            h.update(b)
    return h.hexdigest()

if __name__ == "__main__":
    # python upgrade/group_api.py --run-on cpu
    # python upgrade/group_api.py --run-on gpu
    parser = argparse.ArgumentParser(
        description="Group API usage for all environments."
    )
    parser.add_argument(
        "--run-on",
        type=str,
        required=True,
        default=None,
        help="Experiment running on: cpu or gpu",
    )
    args = parser.parse_args()
    run_on = args.run_on

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        kernel_meta = json.load(f)

    with open(SUBMISSION_NOAPI_PATH, "r", encoding="utf-8") as f:
        submission_noAPI = json.load(f)

    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        executable_files = json.load(f)

    executable_files = [k for k,v in executable_files.items() if "Backporting Failed. Aborting." not in v.get('detail',"") and "[NbConvertApp] Writing "  in v.get('detail',"") and v.get('execution_time', 0) <= 600]
    no_gpu_required_list = no_gpu_required(kernel_meta)
    print(f"Total {len(no_gpu_required_list)} no-gpu-required submissions")
    print(f"Total {len(executable_files)} executable files found in the result")

    filtered_meta = {}
    for compt, submissions in kernel_meta.items():
        filtered_subs = {}
        for sub_html, meta in submissions.items():
            sub_ipynb = sub_html
            full_key = f"{compt}_{sub_ipynb}"

            if run_on == "cpu":
                if full_key in no_gpu_required_list and full_key in executable_files:
                    filtered_subs[sub_html] = meta
                out_path =  "upgrade/groups_by_exact_content_cpu.json"
            else:
                if full_key not in no_gpu_required_list and full_key in executable_files:
                    filtered_subs[sub_html] = meta
                out_path =  "upgrade/groups_by_exact_content_gpu.json"

        if filtered_subs:
            filtered_meta[compt] = filtered_subs
    kernel_meta = filtered_meta
    print(f"Total {run_on}-required environments: {sum(len(v) for v in kernel_meta.values())}")

    submissions = collect_tasks(kernel_meta)
    print(f"Total {len(submissions)} submissions to group")
    mapped_tasks = map_task_versions(submissions, submission_noAPI)
    print(f"Total {len(mapped_tasks)} mapped tasks")


    # Group by (size, sha256, version) to avoid reading entire files into memory
    groups = {}
    i, j = 0,0
    for f in tqdm(sorted(base_dir.glob("*.txt")), desc="Grouping by content + version"):
        try:
            # Extract key from filename (e.g., "competiton_kernelname.txt" -> "competiton_kernelname")
            key = f.stem  # filename without .txt
            
            # Get version from mapped_tasks
            if key not in mapped_tasks:
                i+=1
                # print(f"Warning: {key} not in mapped_tasks, skipping")
                continue
            
            trunc_env_name, req_name, version = mapped_tasks[key]
            
            # Create composite key: (file_size, sha256, version)
            file_key = (f.stat().st_size, sha256_file(f), version)
            j+=1
        except Exception as e:
            print(f"Skip {f.name}: {e}")
            continue
        
        groups.setdefault(file_key, []).append(f.name)

    print(f"Total {i} files skipped due to missing mapped_tasks, processed {j} files.")
    # Format like: [{"a.txt","d.txt"},{"c.txt"}]
    result = [set(names) for names in groups.values()]
    print(f"Total groups: {len(result)}")
    for i, group in enumerate(result[:5]):
        print(f"  Group {i}: {list(group)[:2] if len(group) > 2 else group}")

    # save
    Path(out_path).write_text(json.dumps([sorted(s) for s in result], indent=2), encoding="utf-8")
    print(f"Saved to: {out_path}")