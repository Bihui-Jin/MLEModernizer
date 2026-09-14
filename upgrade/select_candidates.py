"""
Select candidate scripts for modernization (not replicable + not runnable).
"""
import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import tiktoken

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from LLMs.discern_files import no_gpu_required

JSON_PATH = "apiDowngrade/kernel_w_pyVersion.json"
SUBMISSION_NOAPI_PATH = "apiDowngrade/submission_noAPI.json"
PIP_LIST_PATH = "./upgrade/pipList"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
BASELINE_NON_REPLICABLE = "./verification/baseline_non_reproducible.json"

with open(BASELINE_NON_REPLICABLE, "r") as f:
    baseline_non_reproducible = json.load(f)
baseline_non_reproducible = list(set(baseline_non_reproducible.keys()))
print(f"Total baseline non-reproducible: {len(baseline_non_reproducible)}")

# lower_is_better
with open("./upgrade/competition_score_rank.json", "r", encoding="utf-8") as f:
    lower_better = json.load(f)

# python_version
with open(JSON_PATH, 'r') as f:
    pyVersion_data = json.load(f)

# installed_packages
with open(SUBMISSION_NOAPI_PATH, 'r') as f:
    submission_noAPI = json.load(f)


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

def get_score(score_data, file):
    key = Path(file).stem + ".ipynb"
    passed = True if "status" in score_data[key] else False
    if passed:
        try:
            score = score_data[key]["score"]
        except:
            return None
        if "error" in score:
            return score['error']
        else:
            return score['score']
    else:
        return None

# def is_replicable(reported_score, measured_score):
#     if measured_score is str:
#         return False
#     if isinstance(reported_score, float) and isinstance(measured_score, float) and reported_score != 0.0:
#         thrus = abs(measured_score-reported_score)/abs(reported_score) * 100
#         # thrus = 2 * abs(measured_score-reported_score)/ (abs(reported_score)+abs(measured_score)) * 100
#     else:
#         thrus = None
#     return (thrus is not None) and (thrus <= float(threshold))

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

def filter_packages_by_apis(installed_packages_str: str, used_apis) -> str:
    """
    Keep only lines from pip freeze output that relate to any api in used_apis.
    Matching is fuzzy: compare alphanumeric-normalized names (so tensorflow-cloud,
    tensorflow-io-gcs-filesystem, torchvision, torchaudio, etc. match).
    If 'tensorflow' is in used_apis, also include protobuf version line.
    """
    if not installed_packages_str:
        return ""
    lines = [ln.strip() for ln in installed_packages_str.splitlines() if ln.strip()]
    api_norms = [re.sub(r'[^a-z0-9]', '', str(a).lower()) for a in used_apis or []]
    include_protobuf = "tensorflow" in api_norms
    selected = []
    for line in lines:
        # typical form: package==version (keep raw line)
        pkg = line.split("==", 1)[0].lower()
        pkg_norm = re.sub(r'[^a-z0-9]', '', pkg)
        matched = False
        for api_norm in api_norms:
            if api_norm and api_norm in pkg_norm:
                selected.append(line)
                matched = True
                break
        if not matched and include_protobuf and pkg_norm.startswith("protobuf"):
            selected.append(line)
    return "\n".join(selected)

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

def get_script_contents(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    code_blocks = []

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
    
    return "\n\n".join(code_blocks)

def safe_token_count(entity: str, text: str, encoding, chunk_chars: int = 20_000) -> int:
    """
    tiktoken can raise ValueError on some pathological inputs due to regex backtracking.
    Chunking avoids huge-regex workloads. If it still fails, fall back to a rough estimate.
    """
    if not text:
        return 0

    try:
        return len(encoding.encode(text))
    except ValueError:
        total = 0
        # chunk by characters to keep regex work bounded
        for i in range(0, len(text), chunk_chars):
            chunk = text[i : i + chunk_chars]
            try:
                total += len(encoding.encode(chunk))
            except ValueError:
                print("[WARN] tiktoken ValueError on chunk; using approximate token count for", entity)
                # last resort: approximate tokens (~4 chars/token is a common heuristic)
                total += max(1, len(chunk) // 4)
        return total

def select_candidates(score_data, token_threshold, run_type, upgrade_type):
    candidate_info = {}

    i = 0
    j = 0

    for entity, info in tqdm(score_data.items()):
        # if "[NbConvertApp] Writing " in info['detail'] and "Backporting Failed. Aborting." not in info['detail'] and info['execution_time'] <= 600:
        if entity not in baseline_non_reproducible:
            continue
        
        with open(Path("./results/baseline/script_out_allINone") / entity, "r", encoding="utf-8") as f:
            nb = json.load(f)

        has_err = False
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                for output in cell.get('outputs', []):
                    if output.get('output_type') == 'error':
                        has_err = True
                        break
                if has_err:
                    break

        if upgrade_type == "cell" and not has_err:
            continue

        script_in_prompt = get_script_contents(Path("./results/baseline/script_out_allINone") / entity)
        encoding = tiktoken.encoding_for_model("gpt-5")
        sys_tokens = safe_token_count(entity, script_in_prompt, encoding)

        if sys_tokens > token_threshold:
            continue

        compt = entity.split("_")[0]
        submission_id = "_".join(Path(entity).stem.split("_")[1:]) + ".ipynb"
        reported_score = float(pyVersion_data[compt][submission_id]['ps']) 

        # if reported_score == 0.0:
        #     continue

        env_name = re.split(r'(_v\d+|_[A-Za-z]\d+)+$', entity)[0]
        if env_name in manual_versions:
            python_version = manual_versions[env_name]
        else:
            pv = parse_version(pyVersion_data[compt][submission_id]['python'])
            python_version = f"{pv[0]}.{pv[1]}"

        # installed_packages
        used_apis = pyVersion_data[compt][submission_id]['api']
        if run_type == "baseline":
            with open("./upgrade/pip_list.txt", "r") as f:
                installed_packages_raw = f.read()
            installed_packages = filter_packages_by_apis(installed_packages_raw, used_apis)
        # else:
        #     if env_name in submission_noAPI:
        #         base_name = entity.split('.')[0]
        #         key = f"{compt}_{base_name}"
        #         _,_,version = mapped_tasks[key]
        #         with open(os.path.join(PIP_LIST_PATH, f'empty_{version}.txt'), "r") as f:
        #             raw_packages = f.read()
        #         installed_packages = filter_packages_by_apis(raw_packages, used_apis)
        #     else:
        #         for group in group_data:
        #             if entity.replace(".ipynb", ".txt") in group:
        #                 with open(os.path.join(PIP_LIST_PATH, group[0]), "r") as f:
        #                     raw_packages = f.read()
        #                 installed_packages = filter_packages_by_apis(raw_packages, used_apis)
        #                 break
        
        measured_score = get_score(score_data, entity)
        if measured_score is not None:
            candidate_info.update(
                {
                    entity: {
                        "score": measured_score,
                        "target": reported_score,
                        "python_version": python_version,
                        "lower_is_better": lower_better[compt],
                        "installed_packages": installed_packages
                    }
                }
            )
        else:
            candidate_info.update(
                {
                    entity: {
                        "target": reported_score,
                        "python_version": python_version,
                        "lower_is_better": lower_better[compt],
                        "installed_packages": installed_packages
                    }
                }
            )

        # measured_score = get_score(score_data, entity)
        # if measured_score==None:
        #     i+=1
        #     candidate_info.update(
        #         {
        #             entity: {
        #                 "target": reported_score,
        #                 "python_version": python_version,
        #                 "lower_is_better": lower_better[compt],
        #                 "installed_packages": installed_packages
        #             }
        #         }
        #     )
        # else:
        #     replicable = is_replicable(reported_score, measured_score)
        #     if not replicable:
        #         j+=1
        #         candidate_info.update(
        #             {
        #                 entity: {
        #                     "score": measured_score,
        #                     "target": reported_score,
        #                     "python_version": python_version,
        #                     "lower_is_better": lower_better[compt],
        #                     "installed_packages": installed_packages
        #                 }
        #             }
        #         )
    # print(f"Measured_score==None:{i}\nNot replicable:{j}\nTotal:{len(candidate_info)}")
    print(f"Total:{len(candidate_info)}")
    candidate_info = dict(sorted(candidate_info.items()))

    # Split the dict into num_servers chunks (round-robin by key order)
    keys = list(candidate_info.keys())
    chunks = [dict() for _ in range(num_servers)]
    for i, k in enumerate(sorted(keys)):
        chunks[i % num_servers][k] = candidate_info[k]

    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx}: {len(chunk)} files")
        fw = f'./upgrade/{run_type}_{run_on}-{idx}_candidates.json' if upgrade_type == "file" else f'./upgrade/{run_type}_{run_on}-{idx}_{upgrade_type}_candidates.json'
        with open(fw, 'w') as f:
            json.dump(chunk, f, indent=2)

if __name__ == "__main__":
    # python upgrade/select_candidates.py --run-type baseline --token_threshold 16337 --run-on cpu --num-servers 4 --upgrade-type cell
    # python upgrade/select_candidates.py --run-type baseline --token_threshold 16337 --run-on gpu --num-servers 1 --upgrade-type cell

    # python upgrade/select_candidates.py --run-type backporting --token_threshold 16337 --run-on cpu --num-servers 4
    # python upgrade/select_candidates.py --run-type backporting --token_threshold 16337 --run-on gpu --num-servers 1
    # --upgrade-type cell
    parser = argparse.ArgumentParser(
        description="Generate candidates for API upgrade."
    )
    parser.add_argument(
        "--run-type",
        type=str,
        required=True,
        default=None,
        help="Experiment type for baseline or backporting",
    )

    parser.add_argument(
        "--token_threshold",
        type=int,
        required=True,
        default=None,
        help="Token threshold in selecting baseline candidates for upgrade",
    )

    parser.add_argument(
        "--run-on",
        type=str,
        required=True,
        default=None,
        help="Experiment running on: cpu or gpu",
    )

    parser.add_argument(
        "--num-servers",
        type=int,
        required=True,
        default=1,
        help="Number of servers used.",
    )

    parser.add_argument(
        "--upgrade-type",
        help="Upgrade type for file or cell",
        type=str,
        required=False,
        default="file",
    )

    args = parser.parse_args()

    token_threshold = args.token_threshold
    run_type = args.run_type
    run_on = args.run_on
    num_servers = args.num_servers
    upgrade_type = args.upgrade_type

    # PIP_GROUPS_PATH = f"./upgrade/groups_by_exact_content_{run_on}.json"
    # with open(PIP_GROUPS_PATH, "r") as f:
    #     group_data = json.load(f)

    # score
    score_path = './results/baseline/csv_score.json' if run_type == "baseline" else './results/downgrade/baseline/csv_score.json'
    with open(score_path, 'r') as f:
        score_data = json.load(f)

    no_gpu_required_list = no_gpu_required(pyVersion_data)
    score_data = {
        k: v for k, v in score_data.items()
        if (k in no_gpu_required_list) == (run_on == "cpu")
    }

    # full_files = [f for f in os.listdir('./baseline/notebooks') if f.endswith('.ipynb')]
    # done_already = [f for f in os.listdir("./baseline/notebooks_2") if f.endswith('.ipynb')]
    # sample_files = [x for x in full_files if x not in done_already]
    # score_data = {k: v for k, v in score_data.items() if k not in no_gpu_required_list and k in sample_files}

    print(f"Total score_data: {len(score_data)}")
    
    print(f"Total score_data {run_on} non-reproducible: {len([k for k in score_data.keys() if k in baseline_non_reproducible])}")
    
    # filtered_meta = {}
    # for compt, submissions in tqdm(pyVersion_data.items()):
    #     filtered_subs = {}
    #     for sub_html, meta in submissions.items():
    #         sub_ipynb = sub_html.replace(".html", ".ipynb")
    #         full_key = f"{compt}_{sub_ipynb}"

    #         if run_on == "cpu":
    #             if full_key in no_gpu_required_list:
    #                 filtered_subs[sub_html] = meta
    #         else:
    #             if full_key not in no_gpu_required_list:
    #                 filtered_subs[sub_html] = meta

    #     if filtered_subs:
    #         filtered_meta[compt] = filtered_subs
    # pyVersion_data = filtered_meta
    # print(f"Total {run_on}-required environments: {sum(len(v) for v in pyVersion_data.values())}")

    # submissions = collect_tasks(pyVersion_data)
    # mapped_tasks = map_task_versions(submissions, submission_noAPI)

    select_candidates(score_data, token_threshold, run_type, upgrade_type)