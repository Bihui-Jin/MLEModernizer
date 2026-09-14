import json
from pathlib import Path
import sys
from tqdm import tqdm
from glob import glob
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import scipy
import argparse
import os
from run_upgrade_in_sqlite import main as run_upgrade_main
from run_split_baseline_w_timer_parrallel import main as run_baseline_main
import redis
import tiktoken
import re
import pandas as pd
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from LLMs.discern_files import no_gpu_required

compare = {'jigsaw-toxic-comment-classification-challenge': False,
            'google-quest-challenge': False,
            'detecting-insults-in-social-commentary': False,
            'tabular-playground-series-may-2022': False,
            'denoising-dirty-documents': True,
            'aerial-cactus-identification': False,
            'tweet-sentiment-extraction': False,
            'cassava-leaf-disease-classification': False,
            'aptos2019-blindness-detection': False,
            'random-acts-of-pizza': False,
            'new-york-city-taxi-fare-prediction': True,
            'nomad2018-predict-transparent-conductors': True,
            'spooky-author-identification': True,
            'mlsp-2013-birds': False,
            'plant-pathology-2020-fgvc7': False,
            'champs-scalar-coupling': True,
            'uw-madison-gi-tract-image-segmentation': False,
            'histopathologic-cancer-detection': False,
            'bms-molecular-translation': True,
            'predict-volcanic-eruptions-ingv-oe': True,
            'h-and-m-personalized-fashion-recommendations': False,
            'smartphone-decimeter-2022': True,
            'hubmap-kidney-segmentation': False,
            'whale-categorization-playground': False,
            'text-normalization-challenge-russian-language': False,
            'nfl-player-contact-detection': False,
            'hms-harmful-brain-activity-classification': True,
            'tensorflow2-question-answering': False,
            'osic-pulmonary-fibrosis-progression': False,
            'plant-pathology-2021-fgvc8': False,
            'alaska2-image-steganalysis': False,
            'hotel-id-2021-fgvc8': False,
            'multi-modal-gesture-recognition': True,
            'herbarium-2020-fgvc7': False,
            'vesuvius-challenge-ink-detection': False,
            '3d-object-detection-for-autonomous-vehicles': False,
            'tabular-playground-series-dec-2021': False,
            'inaturalist-2019-fgvc6': True,
            'iwildcam-2020-fgvc7': False,
            'seti-breakthrough-listen': False,
            'icecube-neutrinos-in-deep-ice': True,
            'herbarium-2022-fgvc9': False,
            'herbarium-2021-fgvc8': False,
            'vinbigdata-chest-xray-abnormalities-detection': False,
            'rsna-breast-cancer-detection': False,
            'us-patent-phrase-to-phrase-matching': False,
            'chaii-hindi-and-tamil-question-answering': False,
            'leaf-classification': True,
            'statoil-iceberg-classifier-challenge': True,
            'tgs-salt-identification-challenge': False,
            'dog-breed-identification': True,
            'lmsys-chatbot-arena': True,
            'learning-agency-lab-automated-essay-scoring-2': False,
            'ventilator-pressure-prediction': True,
            'dogs-vs-cats-redux-kernels-edition': True,
            'facebook-recruiting-iii-keyword-extraction': False,
            'jigsaw-unintended-bias-in-toxicity-classification': False,
            'ranzcr-clip-catheter-line-classification': False,
            'text-normalization-challenge-english-language': False,
            'billion-word-imputation': True,
            'freesound-audio-tagging-2019': False,
            'the-icml-2013-whale-challenge-right-whale-redux': False,
            'petfinder-pawpularity-score': True,
            'kuzushiji-recognition': False,
            'iwildcam-2019-fgvc6': False,
            'imet-2020-fgvc7': False,
            'siim-isic-melanoma-classification': False,
            'rsna-miccai-brain-tumor-radiogenomic-classification': False,
            'siim-covid19-detection': False,
            'rsna-2022-cervical-spine-fracture-detection': True,
            'google-research-identify-contrails-reduce-global-warming': False,
            'stanford-covid-vaccine': True,
            'tensorflow-speech-recognition-challenge': False,
            'AI4Code': False,
            'cdiscount-image-classification-challenge': False
        }
def is_close(reported_score,measured_score):
    if isinstance(reported_score, float) and isinstance(measured_score, float) and reported_score != 0.0:
        thrus = abs(measured_score-reported_score)/abs(reported_score) * 100
    else:
        thrus = None

    return (thrus is not None) and (thrus <= float(10))

def get_script_contents(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

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
    code_blocks = []
    ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


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

def get_timeout(n):
    cpu_out, gpu_out = {}, {}

    last_run_scr = Path.cwd() / f"verification/{upgrade_type}/run_{n}/{run_on}_rerun.json"
    with open(last_run_scr) as f:
        data = json.load(f)
    data = {k:v for k,v in data.items() if v.get("failed", 0) == 0}

    glob_notebooks = glob(str(Path.cwd() / f"verification/file/notebooks/*.ipynb"))
    glob_notebooks = [Path(k).stem for k in glob_notebooks]

    for k,v in data.items():
        if Path(k).stem not in glob_notebooks:
            continue
        parts = k.split("_")
        compt = parts[0]
        notebook_name = "_".join(parts[1:-2])
        version = parts[-2]
        out_path = Path.cwd() / f"verification/{upgrade_type}/run_{n}/script_out" / compt / notebook_name / version
        with open (out_path / "result.json") as f:
            result = json.load(f)

        if result.get("timeout", False): 
            if compute_resource[k] == "cpu": cpu_out[k] = file_details[k]
            if compute_resource[k] == "gpu": gpu_out[k] = file_details[k]
    
    if run_on == "cpu":
        print(f"#timeout scripts on {run_on}: {len(cpu_out)}")
        with open(f"./verification/{upgrade_type}/run_{n}/cpu_rerun_timeout.json", "w") as f:
            json.dump(cpu_out, f, indent=2) 
    else:
        print(f"#timeout scripts on {run_on}: {len(gpu_out)}")
        with open(f"./verification/{upgrade_type}/run_{n}/gpu_rerun_timeout.json", "w") as f:
            json.dump(gpu_out, f, indent=2)

def stats_not_passed(n, target_meta, compute_resource, file_details):
    comp_pairs = defaultdict(list)
    cpu_2out, gpu_2out = {}, {}
    ne = {}
    model_name = "gpt" if "gpt-5.2" in llm_model else "oss"
    if n<=2:
        last_run_scr = Path.cwd() / f"verification/{upgrade_type}/run_{n}/{run_on}_rerun.json"
    else:
        if upgrade_type in ["cell", "file"]:
            last_run_scr = Path.cwd() / f"results/upgrade/{model_name}_{upgrade_type}/csv_score.json"
        else:
            last_run_scr = Path.cwd() / f"results/{upgrade_type}/csv_score.json" if upgrade_type in ["baseline"] else Path.cwd() / f"results/downgrade/baseline/csv_score.json"
    with open(last_run_scr) as f:
        data = json.load(f)
    data = {k:v for k,v in data.items() if v.get("failed", 0) == 0}

    if upgrade_type in ["baseline"]:
        glob_notebooks = glob(str(Path.cwd() / f"baseline/notebooks/*.ipynb"))
        glob_notebooks = [Path(k).stem for k in glob_notebooks]
    else:
        with open("./verification/baseline_non_reproducible.json", "r", encoding="utf-8") as f:
            glob_notebooks = json.load(f)
        glob_notebooks = [Path(k).stem for k in glob_notebooks.keys()]

    for k,v in data.items():
        if Path(k).stem not in glob_notebooks:
            continue
        parts = k.split("_")
        compt = parts[0]
        notebook_name = "_".join(parts[1:-2])
        version = parts[-2]
        out_path = Path.cwd() / f"verification/{upgrade_type}/run_{n}/script_out" / compt / notebook_name / version
        
        if not (out_path / "result.json").exists():
            continue

        with open(out_path / "result.json") as f:
            result = json.load(f)

        if n>2:
            last_run_path = Path.cwd() / f"verification/{upgrade_type}/run_{n-1}/script_out" / compt / notebook_name / version 
            with open(last_run_path / "result.json") as f:
                last_run_path = json.load(f)
        else:
            last_run_path = root_scores[k]

        score_first = last_run_path.get("score", None)
        score_target = target_meta[k].get("target", None)
        score_rerun = result.get("score", None)

        if (
            isinstance(score_target, (int, float))
            and isinstance(score_first, (int, float))
            and not isinstance(score_rerun, (int, float))
        ):
            
            if compute_resource[k] == "cpu": 
                cpu_2out[k] = file_details[k] 
            else: gpu_2out[k] = file_details[k]

        if (
            isinstance(score_target, (int, float))
            and not isinstance(score_first, (int, float))
            and isinstance(score_rerun, (int, float))
            and n<=2
        ):
            first_rerun_path = Path(f"./verification/{upgrade_type}/run_1/script_out") / compt / notebook_name / version / "result.json"
            if first_rerun_path.exists():
                with open(first_rerun_path) as f:
                    score_first_rerun = json.load(f).get("score", None)
                if isinstance(score_first_rerun, (int, float)): score_first = score_first_rerun
        
        if (
            isinstance(score_target, (int, float))
            and isinstance(score_first, (int, float))
            and isinstance(score_rerun, (int, float))
            and n==2
        ):
            diff = score_rerun - score_first
            if np.isfinite(diff) and diff != 0:
                ne[k] = 0
            else:
                continue
            

        scores = [score_first] if isinstance(score_first, (int, float)) else []
        for i in range(2,n+1):
            score_path = Path.cwd() / f"verification/{upgrade_type}/run_{i}/script_out" / compt / notebook_name / version
            with open(score_path / "result.json") as f:
                cur_n_result = json.load(f)
            cur_n_score = cur_n_result.get("score", None)

            if isinstance(cur_n_score, (int, float)):
                scores.append(cur_n_score)

        # if len(scores) < 2:
        #     continue

        if scores:
            comp_pairs[compt].append(
                {
                    "scores": scores,
                    "is_lower_better": compare.get(compt, False),
                    "target": target_meta[k].get("target", None),
                    "key": k,
                }
            )

    ls10 = 0     
    # 2) Normalize per competition in the same style
    for compt, rows in comp_pairs.items():
        if not rows:
            continue
        # rel_tolerance = 0.1
        target_arr = np.array([r["target"] for r in rows], dtype=float)
        # same transform idea: log1p(abs(.)) on combined values
        target_log = np.log1p(np.maximum(0, np.abs(target_arr)))

        cmin = target_log.min()
        cmax = target_log.max()

        # normalize ALL scores for this key (k)
        def to_norm(arr: np.ndarray) -> np.ndarray:
            arr_log = np.log1p(np.maximum(0, np.abs(arr)))
            if cmax > cmin:
                out = (arr_log - cmin) / (cmax - cmin)
            else:
                out = arr_log.copy()
            if rows[0]["is_lower_better"]:
                out = -out
            return out
        
        target_norm = to_norm(target_arr)
        # low_raw = target_arr * (1 - rel_tolerance)
        # high_raw = target_arr * (1 + rel_tolerance)
        # low_norm = to_norm(low_raw)
        # high_norm = to_norm(high_raw)
        
        # update back to corresponding row position
        for idx, row in enumerate(rows):
            s = np.asarray(row["scores"], dtype=float)
            row["score_norm"] = to_norm(s).tolist()
            row["target_norm"] = float(target_norm[idx])
            # row["target_low_norm"] = float(min(low_norm[idx], high_norm[idx]))
            # row["target_high_norm"] = float(max(low_norm[idx], high_norm[idx]))

            # print(target_norm[idx], to_norm(s).tolist())
    #         if len(to_norm(s).tolist()) < 10:
    #             ls10 += 1
    # print(f"#scripts with less than 10 runs: {ls10}")


    cpu = {}
    gpu = {}
    total_rerun = 0
    zero_variance = 0
    all_script = 0
    # run per row
    for compt, rows in comp_pairs.items():
        for row in rows:
            all_script += 1
            k = row["key"]

            repro_scores = np.asarray(row["score_norm"], dtype=float)
            ttest_result = scipy.stats.ttest_1samp(repro_scores, row['target_norm'])
            if not np.isnan(ttest_result.pvalue) and ttest_result.pvalue < 0.05:
                continue
            
            total_rerun += 1
            if compute_resource[k] == "cpu": cpu[k] = file_details[k]
            if compute_resource[k] == "gpu": gpu[k] = file_details[k]

    print(f"Total scripts with significant differences: {total_rerun} out of {len(cpu_2out|ne|gpu_2out)}")
    print(f"#CPU: {len(cpu)}") if run_on == "cpu" else ""
    print(f"#GPU: {len(gpu)}") if run_on == "gpu" else ""
    print(f"#zero variance cases: {zero_variance}")
    os.makedirs(f"./verification/{upgrade_type}/run_{n+1}", exist_ok=True)
    if run_on == "cpu":
        with open(f"./verification/{upgrade_type}/run_{n+1}/cpu_rerun.json", "w") as f:
            json.dump(cpu, f, indent=2) 
    else:
        with open(f"./verification/{upgrade_type}/run_{n+1}/gpu_rerun.json", "w") as f:
            json.dump(gpu, f, indent=2)

def print_stats(n, target_meta, compute_resource, file_details, token_threshold, llm_model):
    comp_pairs = defaultdict(list)
    no_score_items = []

    # Use all executed baseline items as candidates.
    data = {k: v for k, v in root_scores.items() if v.get("failed", 0) == 0}
    print(f"Total baseline items: {len(data)}")


    run2_mismatch = set()

    model_name = "gpt" if "gpt-5.2" in llm_model else "oss"
    if upgrade_type in ["baseline", "backporting"]:
        glob_notebooks = glob(str(Path.cwd() / f"baseline/notebooks/*.ipynb"))
        glob_notebooks = [Path(k).stem for k in glob_notebooks]
    else:
        with open("./verification/baseline_non_reproducible.json", "r", encoding="utf-8") as f:
            glob_notebooks = json.load(f)
        glob_notebooks = [Path(k).stem for k in glob_notebooks.keys()]

    for k,v in tqdm(data.items()):
        if k not in target_meta or k not in compute_resource:
            continue
        if Path(k).stem not in glob_notebooks:
            continue

        parts = k.split("_")
        compt = parts[0]
        notebook_name = "_".join(parts[1:-2])
        version = parts[-2]

        if upgrade_type in ["baseline", "backporting"]:
            run2_path = Path.cwd() / f"verification/{upgrade_type}/run_2/script_out" / compt / notebook_name / version / "result.json"
        else:
            run2_path = Path.cwd() / f"verification/{model_name}_{upgrade_type}/run_2/script_out" / compt / notebook_name / version / "result.json"
        if not run2_path.exists():
            continue

        with open(run2_path) as f:
            run2_result = json.load(f)

        score_first = v.get("score", None)
        score_run2 = run2_result.get("score", None)
        score_target = target_meta[k].get("target", None)

        # run2_mismatch.add(k)
        if not isinstance(score_target, (int, float)):
            continue
        if (isinstance(score_first, (int, float)) and not isinstance(score_run2, (int, float))) \
            or (isinstance(score_run2, (int, float)) and not isinstance(score_first, (int, float))) \
            or (not isinstance(score_run2, (int, float)) and not isinstance(score_first, (int, float))):
            run2_mismatch.add(k)
            continue

        diff = score_run2 - score_first
        if upgrade_type == 'baseline':
            if np.isfinite(diff) and diff != 0:
                run2_mismatch.add(k)
                continue
            continue
        if score_run2 == score_first and is_close(score_target, score_run2):
            continue
        if np.isfinite(diff) and (not is_close(score_target, score_run2) or not is_close(score_target, score_first)):
            run2_mismatch.add(k)
            continue
        if score_run2 == score_first and not is_close(score_target, score_run2):
            run2_mismatch.add(k)
            continue
    print(f"Run-2 mismatch candidates: {len(run2_mismatch)}")

    if upgrade_type in ["baseline"] and token_threshold is not None:
        run2_mismatch = set([k for k in tqdm(run2_mismatch) if 
                    # root_scores[k].get("execution_time",0)<=600 
                    # and "[NbConvertApp] Writing " in root_scores[k].get('detail', "[NbConvertApp] Writing ")
                    safe_token_count(k, get_script_contents(Path("./results/baseline/script_out_allINone") / k), encoding) <= token_threshold])
    
    non_reproducible_candidates = {k: {"exceed_execution_time": 0, "not_writing": 0, "backporting_failed": 0, "runs": 0} for k in run2_mismatch}
    for k in run2_mismatch:
        parts = k.split("_")
        compt = parts[0]
        notebook_name = "_".join(parts[1:-2])
        version = parts[-2]

        scores = []
        for i in range(1, n + 1):
            if upgrade_type in ["baseline", "backporting"]:
                score_path = Path.cwd() / f"verification/{upgrade_type}/run_{i}/script_out" / compt / notebook_name / version / "result.json"
            else:
                score_path = Path.cwd() / f"verification/{model_name}_{upgrade_type}/run_{i}/script_out" / compt / notebook_name / version / "result.json"
            if not score_path.exists():
                if i == 1:
                    cur_score = data[k].get("score", None)
                else:
                    continue
            else:
                with open(score_path) as f:
                    cur_result = json.load(f)
                
                if cur_result.get("execution_time", 0) > 600:
                    non_reproducible_candidates[k]["exceed_execution_time"] += 1
                if "[NbConvertApp] Writing " not in cur_result.get('detail',  ""):
                    non_reproducible_candidates[k]["not_writing"] += 1
                if upgrade_type == "backporting" and "Environment processed." not in cur_result.get("detail", False):
                    non_reproducible_candidates[k]["backporting_failed"] += 1
                non_reproducible_candidates[k]["runs"] += 1

                cur_score = cur_result.get("score", None)

            if isinstance(cur_score, (int, float)) and np.isfinite(cur_score):
                scores.append(float(cur_score))


        if not scores:
            no_score_items.append(k)
            continue

        comp_pairs[compt].append(
            {
                "scores": scores,
                "is_lower_better": compare.get(compt, False),
                "target": target_meta[k].get("target", None),
                "key": k,
            }
        )

    

    # Normalize per competition.
    for compt, rows in comp_pairs.items():
        if not rows:
            continue

        target_arr = np.array([r["target"] for r in rows], dtype=float)
        target_log = np.log1p(np.maximum(0, np.abs(target_arr)))
        cmin = target_log.min()
        cmax = target_log.max()

        def to_norm(arr: np.ndarray) -> np.ndarray:
            arr_log = np.log1p(np.maximum(0, np.abs(arr)))
            if cmax > cmin:
                out = (arr_log - cmin) / (cmax - cmin)
            else:
                out = arr_log.copy()
            if rows[0]["is_lower_better"]:
                out = -out
            return out

        target_norm = to_norm(target_arr)
        for idx, row in enumerate(rows):
            s = np.asarray(row["scores"], dtype=float)
            row["score_norm"] = to_norm(s)
            row["target_norm"] = float(target_norm[idx])

    cpu = {}
    gpu = {}
    significant_count = []
    insufficient_scores = []
    nan_pvalue = []

    for _, rows in comp_pairs.items():
        for row in rows:
            k = row["key"]
            repro_scores = np.asarray(row["score_norm"], dtype=float)
            repro_scores = repro_scores[np.isfinite(repro_scores)]

            # Keep the test stable: one-sample t-test is unreliable for <2 points.
            if repro_scores.size < 2:
                insufficient_scores.append(k)
                continue

            ttest_result = scipy.stats.ttest_1samp(repro_scores, row["target_norm"], nan_policy="omit")
            if np.isnan(ttest_result.pvalue):
                nan_pvalue.append(k)
                continue

            if ttest_result.pvalue <= 0.05:
                significant_count.append(k)
                if compute_resource[k] == "cpu":
                    cpu[k] = file_details.get(k, {})
                if compute_resource[k] == "gpu":
                    gpu[k] = file_details.get(k, {})

    print(f"Run-2 mismatch candidates: {len(run2_mismatch)}")
    print(f"Candidates with no usable rerun score in run 1 - run {n}: {len(no_score_items)}")
    # tested_items = sum(len(rows) for rows in comp_pairs.values())
    # print(f"Items with at least one usable rerun score: {tested_items}")
    print(f"Items skipped (fewer than 2 rerun scores): {len(insufficient_scores)}")
    print(f"Items skipped (NaN p-value): {len(nan_pvalue)}")
    print(f"Total scripts significantly different from target after run_1..run_{n}: {len(significant_count)}")
    print(f" - CPU: {len(cpu)}")
    print(f" - GPU: {len(gpu)}")

    model_name = "gpt" if "gpt-5.2" in llm_model else "oss"
    root_score_path = Path(f"./results/upgrade/{model_name}_{upgrade_type}/csv_score.json") if upgrade_type not in ["baseline", "backporting"] else Path(f"./results/{upgrade_type}/csv_score.json") if upgrade_type == "baseline" else Path(f"./results/downgrade/baseline/csv_score.json")
    with open(root_score_path) as f:
        current = json.load(f)
    current = set([k for k,v in current.items()])
    # current = set([k for k,v in root_scores.items() if 
    #                v.get("execution_time")<=600 
    #                and "[NbConvertApp] Writing " in v['detail']
    #                and not is_replicable(target_meta[k].get("target", None), v.get("score", None))
    #                and safe_token_count(k, get_script_contents(Path("./results/baseline/script_out_allINone") / k), encoding) <= 13_485])
    significant_diff = set(insufficient_scores)|set(no_score_items)|set(nan_pvalue)|set(significant_count)
    
    print(f"Significant_diff before filtering: {len(significant_diff)}")
    print("Timeout: ", len(set([k for k in tqdm(significant_diff) if 
        non_reproducible_candidates[k].get("exceed_execution_time",0) == non_reproducible_candidates[k].get("runs",0) ]
        )))
    print("Not writing: ", len(set([k for k in tqdm(significant_diff) if 
        non_reproducible_candidates[k].get('not_writing', 0) == non_reproducible_candidates[k].get("runs",0)
        and non_reproducible_candidates[k].get("exceed_execution_time",0) != non_reproducible_candidates[k].get("runs",0)
        ])))
    if upgrade_type == "backporting":
        print("Backporting failed: ", len(set([k for k in tqdm(significant_diff) if 
            non_reproducible_candidates[k].get("backporting_failed", 0) == non_reproducible_candidates[k].get("runs",0)
            ])))
    
    if upgrade_type in ["baseline", "backporting"]:
        # print(f"Timeout: {len([k for k in significant_diff if root_scores[k].get('execution_time',0)>600])}")
        # print(f"Not saving: {len([k for k in significant_diff if not is_replicable(target_meta[k].get("target", None), root_scores[k].get("score", None))])}")
        significant_diff = set([
            k for k in tqdm(significant_diff) 
            if (
                non_reproducible_candidates[k].get("exceed_execution_time",0) != non_reproducible_candidates[k].get("runs",0) 
                and non_reproducible_candidates[k].get('not_writing', 0) != non_reproducible_candidates[k].get("runs",0)
                and non_reproducible_candidates[k].get("backporting_failed", 0) != non_reproducible_candidates[k].get("runs",0)
                and (
                    token_threshold is None or 
                    safe_token_count(k, get_script_contents(Path("./results/baseline/script_out_allINone") / k), encoding) <= token_threshold
                )
            )
        ])



    print(f"Non-Reproducible: {len(significant_diff)}")
    if upgrade_type =="file":
        with open(f"./verification/{model_name}_{upgrade_type}_non_reproducible.json", "w") as f:
            json.dump({k: root_scores[k] for k in significant_diff}, f, indent=2)
    else:
        with open(f"./verification/{upgrade_type}_non_reproducible.json", "w") as f:
            json.dump({k: root_scores[k] for k in significant_diff}, f, indent=2)

    print(f"Non-Reproducible in current dataset: {len(current)}")
    print(f"Diff with curr dataset: {len(significant_diff) - len(current)}")
    
    plus_num = len(significant_diff - current)   # in significant_diff, not in current
    minus_num = len(current - significant_diff)  # in current, not in significant_diff
    print(f"+ {plus_num}")
    print(f"- {minus_num}")
    

if __name__ == "__main__":
    # python verification/entrypoint.py --llm-model gpt-5.2 --threshold 10 --max-fix 16 --llm-workers 7 --run-type backporting --upgrade-type backporting --run-on cpu --server-node 0 --token-threshold 16337
    # python verification/entrypoint.py --llm-model gpt-5.2 --threshold 10 --max-fix 16 --llm-workers 7 --run-type baseline --upgrade-type baseline --run-on cpu --server-node 0 --token-threshold 16337
    # python verification/entrypoint.py --llm-model gpt-5.2 --threshold 10 --max-fix 16 --llm-workers 7 --run-type baseline --upgrade-type baseline --run-on gpu --server-node 0 
    # python verification/entrypoint.py --llm-model gpt-oss --threshold 10 --max-fix 16 --llm-workers 7 --run-type baseline --upgrade-type file --run-on gpu --server-node 0 --token-threshold 16337
    # python verification/entrypoint.py --llm-model gpt-5.2 --threshold 10 --max-fix 16 --llm-workers 7 --run-type baseline --upgrade-type cell --run-on gpu --server-node 0 --token-threshold 16337

    # python verification/entrypoint.py --llm-model gpt-5.2 --threshold 10 --max-fix 16 --llm-workers 7 --run-type backporting --upgrade-type backporting --run-on cpu --server-node 0 --token-threshold 16337 --yield-results
    # python verification/entrypoint.py --llm-model gpt-5.2 --threshold 10 --max-fix 16 --llm-workers 7 --run-type baseline --upgrade-type baseline --run-on cpu --server-node 0 --token-threshold16337 --yield-results
    # python verification/entrypoint.py --llm-model gpt-5.2 --threshold 10 --max-fix 16 --llm-workers 7 --run-type baseline --upgrade-type file --run-on gpu --server-node 0 --token-threshold 16337 --yield-results
    # python verification/entrypoint.py --llm-model gpt-oss --threshold 10 --max-fix 16 --llm-workers 7 --run-type baseline --upgrade-type file --run-on gpu --server-node 0 --token-threshold 16337 --yield-results

    parser = argparse.ArgumentParser(
        description="Code upgrade runner."
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
        help="Upgrade type for file or cell or baseline",
        type=str,
        required=True,
        default=None,
    )
    parser.add_argument(
        "--rerun-times",
        help="# of MAX times to rerun",
        type=int,
        required=False,
        default=10,
    )
    parser.add_argument("--threshold", type=float, default=10.0, required=True, help="Threshold for replication borderline")
    parser.add_argument("--token-threshold", type=int, required=False, help="Token threshold for baseline candidates for upgrade")
    parser.add_argument("--llm-model", type=str, required=True, help="LLM model name for fixing")
    parser.add_argument("--llm-workers", type=int, required=True, default=4, help="Concurrent LLM fix workers, # of Threads/CPUs")
    parser.add_argument("--max-fix", type=int, default=16, required=True, help="Stop once all items reach fix>=max-fix")
    parser.add_argument(
        "--yield-results",
        help="Whether to continue from a previous run",
        action="store_true",
    )
    args = parser.parse_args()
    

    run_type = args.run_type
    upgrade_type = args.upgrade_type
    llm_model = args.llm_model
    max_fix = args.max_fix
    run_on = args.run_on
    server_node = args.server_node
    rerun_times = args.rerun_times
    token_threshold = args.token_threshold
    yield_result_mode = args.yield_results
    print(f"{token_threshold=}")
    encoding = tiktoken.encoding_for_model("gpt-5")



    compute_resource = {}
    file_details = {}
    print(f"{upgrade_type=}")

    target_meta = {}
    with open(f"./kernel.json") as f:
        kernel_content = json.load(f)
    for comp, submissions in kernel_content.items():
        for submission_id, meta in submissions.items():
            if 'ps' in meta and float(meta['ps']) != 0.0 and meta['runtime'] <= 600 and "R" not in meta:
                k = f"{comp}_{submission_id.replace('.html','.ipynb')}"
                target_meta[k] = {"target":float(meta['ps'])}

    Path(f'./verification/{upgrade_type}/run_2').mkdir(parents=True, exist_ok=True)
    if (upgrade_type =="cell" or upgrade_type =="file"):
        cpu_files = glob("./upgrade/baseline_cpu-[0-9]_candidates.json")
        gpu_files = glob("./upgrade/baseline_gpu-[0-1]_candidates.json")

        print(f"CPU files: {len(cpu_files)}\n{cpu_files}")
        print(f"GPU files: {len(gpu_files)}\n{gpu_files}")

        
        for f in cpu_files + gpu_files:
            with open(f) as f:
                data = json.load(f)
            for k,v in data.items():
                compute_resource[k] = "cpu" if "cpu" in f.name else "gpu"
                file_details[k] = v
        print(f"Total compute resource entries: {len(compute_resource)}")

        model_name = "gpt" if "gpt-5.2" in llm_model else "oss"
        root_score_path = Path(f"./results/upgrade/{model_name}_{upgrade_type}/csv_score.json")
        with open(root_score_path) as f:
            root_scores = json.load(f)
        compute_resource = {k: v for k, v in compute_resource.items() if k in root_scores and root_scores[k].get("fix", 0) > 0}

        if yield_result_mode:
            print_stats(10, target_meta, compute_resource, file_details, token_threshold, llm_model)
            exit(0)

        if run_on == "cpu":
            with open(f'./verification/{upgrade_type}/run_2/cpu_rerun.json', 'w') as output_file:
                json.dump({itm: target_meta[itm] for itm in [k for k, v in compute_resource.items() if v == "cpu"]}, output_file, indent=2)
        else:
            with open(f'./verification/{upgrade_type}/run_2/gpu_rerun.json', 'w') as output_file:
                json.dump({itm: target_meta[itm] for itm in [k for k, v in compute_resource.items() if v == "gpu"]}, output_file, indent=2)
    else:
        with open("./apiDowngrade/kernel_w_pyVersion.json", "r", encoding="utf-8") as f:
            kernel = json.load(f)
        no_gpu_required_list = no_gpu_required(kernel)

        if run_type == "backporting":
            with open(f"./results/downgrade/baseline/csv_score.json") as f:
                root_scores = json.load(f)
        else:
            with open(f"./results/{upgrade_type}/csv_score.json") as f:
                root_scores = json.load(f)
        for k, v in root_scores.items():
            compute_resource[k] = "cpu" if k in no_gpu_required_list else "gpu"
            if isinstance(v, dict) and "score" in v and isinstance(v["score"], dict) and "score" in v["score"]:
                v["score"] = v["score"]["score"]
            v["target"] = target_meta[k].get("target", None)
            v["lower_is_better"] = compare.get(k.split("_")[0], False)
            file_details[k] = v
        
        if yield_result_mode:
            print_stats(10, target_meta, compute_resource, file_details, token_threshold, llm_model)
            exit(0)
            
        sample_files = [f for f in os.listdir('./baseline/notebooks') if f.endswith('.ipynb')]
        if run_on == "cpu":
            with open(f'./verification/{run_type}/run_2/cpu_rerun.json', 'w') as output_file:
                json.dump({itm: target_meta[itm] for itm in list(no_gpu_required_list)}, output_file, indent=2)
        else:
            with open(f'./verification/{run_type}/run_2/gpu_rerun.json', 'w') as output_file:
                json.dump({itm: target_meta[itm] for itm in list(set(sample_files) - set(no_gpu_required_list))}, output_file, indent=2)
        


    if upgrade_type in ["cell", "file"]:
        db_path = "redis://192.168.0.4:6379/0"
        conn = redis.Redis.from_url(db_path)
        prefix = f"kernel:{upgrade_type}:{run_on}:{server_node}:{llm_model}"
        keys_set = f"{prefix}:keys"

    for n in tqdm(range(2, rerun_times+1)):
        print("Rerun n =", n)
        os.makedirs(f"./verification/{upgrade_type}/run_{n}", exist_ok=True)
        if upgrade_type == "file" or upgrade_type == 'cell':
            run_upgrade_main([
                "--run-dir", f"./verification/{upgrade_type}/run_{n}",
                "--run-type", run_type,
                "--upgrade-type", upgrade_type,
                "--llm-model", llm_model,
                "--threshold", str(args.threshold),
                "--max-fix", "16",
                "--llm-workers", "8",
                "--state-backend", "redis",
                "--run-on", run_on,
                "--server-node", str(server_node),
                "--role", "exec",
                "--rerun",
            ])
        else:
            run_baseline_main([
                "--run-dir", f"./verification/{run_type}/run_{n}",
                "--run-type", run_type,
                "--split", str(n),
                "--run-on", run_on,
                "--rerun",
            ])

        print("Checking timeout scripts in the current run...")
        if n >=2:
            time_out = get_timeout(n)
            if upgrade_type == "file" or upgrade_type == 'cell':
                run_upgrade_main([
                    "--run-dir", f"./verification/{upgrade_type}/run_{n}",
                    "--run-type", run_type,
                    "--upgrade-type", upgrade_type,
                    "--llm-model", llm_model,
                    "--threshold", str(args.threshold),
                    "--max-fix", "16",
                    "--llm-workers", "8",
                    "--state-backend", "redis",
                    "--run-on", run_on,
                    "--server-node", str(server_node),
                    "--role", "exec",
                    "--run-timeout",
                ])
            else:
                run_baseline_main([
                    "--run-dir", f"./verification/{upgrade_type}/run_{n}",
                    "--run-type", upgrade_type,
                    "--split", str(n),
                    "--run-on", run_on,
                    "--run-timeout",
                ])
        
        stats_not_passed(n, target_meta, compute_resource, file_details)

        if upgrade_type in ["cell", "file"]:
            for key in tqdm(conn.scan_iter(f"{prefix}:*")):
                if conn.type(key) != b"hash":
                    continue
                k = key.decode('utf-8').split(":")[-1]
                conn.delete(f"{prefix}:{k}")
    
        
