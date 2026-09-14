import argparse
import json
from glob import glob
import os
import re
import string
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from tqdm import tqdm
import nbformat
import time
import multiprocessing as mp

NON_DESCRIPTIVE_EVALUES = {"", "none", "null", "ignored", "nan", "na", "n/a", "<null>", "<none>"}

ML_PACKAGE_HINTS = {
    "numpy", "pandas", "scipy", "statsmodels",
    "matplotlib", "seaborn", "plotly",
    "sklearn", "scikit-learn",
    "tensorflow", "keras", "torch", "torchvision",
    "xgboost", "lightgbm", "catboost", "imblearn",
    "cv2", "skimage",
    "nltk", "transformers",
    "optuna", "datasets",
}

IMPORTANT_TOKENS_ALLOWLIST = {
    "shape", "shapes", "ndim", "dimension", "dims", "axis",
    "expected", "found", "received", "incompatible", "broadcast", "reshape",
    "tensor", "tensors", "layer", "model", "weights", "dtype",
    "index", "indices", "key", "keys", "column", "columns", "feature", "features",
    "nan", "inf", "finite",
    "file", "directory", "path", "permission", "denied", "no", "such",
    "connect", "connection", "timeout", "ssl", "certificate", "http", "https",
    "module", "import", "attribute", "argument", "positional", "keyword",
    "loss", "compile", "fit", "train", "predict",
}


@dataclass(frozen=True)
class CrashRecord:
    notebook_path: str
    cell_index: int
    execution_count: Optional[int]
    ename: str
    evalue: str
    traceback_text: str
    error_value_norm: str
    tokens: Tuple[str, ...]
    crash_type: str
    root_cause: str
    is_ml_bug: bool
    packages: Tuple[str, ...]
    cluster_id: int = -1


def iter_ipynb_paths(root: Path) -> Iterable[Path]:
    if root.is_file() and root.suffix == ".ipynb":
        yield root
        return
    for p in root.rglob("*.ipynb"):
        yield p


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def extract_error_outputs(nb: nbformat.NotebookNode) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for o in cell.get("outputs", []) or []:
            if o.get("output_type") == "error":
                out.append({
                    "cell_index": i,
                    "execution_count": cell.get("execution_count"),
                    "ename": _safe_str(o.get("ename")),
                    "evalue": _safe_str(o.get("evalue")),
                    "traceback": o.get("traceback") or [],
                })
    return out


_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_PATH_RE = re.compile(r"(?:(?:[A-Za-z]:\\)|/)[^\s\"']+")
_URL_RE = re.compile(r"https?://\S+")
_HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
_NUM_RE = re.compile(r"\b\d+(\.\d+)?\b")
_QUOTED_RE = re.compile(r"(['\"])(?:(?=(\\?))\2.)*?\1")
_IDENT_RE = re.compile(r"\b[a-zA-Z_]\w*\b")
_PUNCT_RE = re.compile(r"[^\w\s]+")
_TRACEBACK_ERROR_LINE_RE = re.compile(r"^([A-Za-z_][\w]+)\s*:\s*(.*)$")
_TRACEBACK_FRAME_HINT_RE = re.compile(r"(<ipython-input-|\.ipynb|/notebooks/|/home/jovyan/)")
_SITE_PACKAGES_HINT_RE = re.compile(r"site-packages|dist-packages")


def _parse_traceback_line(line: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", line or "").strip()


def _extract_evalue_from_traceback(traceback_lines: List[str], ename: str) -> Optional[str]:
    target = (ename or "").strip().lower()
    for line in reversed(traceback_lines or []):
        for sub in reversed(_parse_traceback_line(line).split("\n")):
            m = _TRACEBACK_ERROR_LINE_RE.match(sub.strip())
            if not m:
                continue
            if m.group(1).strip().lower() == target:
                return m.group(2).strip()
    return None


def _process_noisy_evalue(evalue: str, ename: str, max_len: int = 400) -> str:
    if not evalue:
        return ""
    lines = str(evalue).split("\n")
    if len(lines) <= 1:
        return evalue.strip()
    first = lines[0].strip()
    if not first or "in user code" in first.lower():
        target = (ename or "") + ":"
        for line in reversed(lines):
            if target.lower() in line.lower():
                return re.sub(re.escape(target), "", line, flags=re.IGNORECASE).strip()[:max_len]
        for line in reversed(lines):
            if line.strip():
                return line.strip()[:max_len]
    return first[:max_len]


def preprocess_text_similarity(text: str) -> str:
    cleaned_text = re.sub(r"\r\n|\r|\n|\\n", " ", str(text))
    cleaned_text = re.sub(r"\S+\.\S+", " ", cleaned_text)
    cleaned_text = re.sub(r"\b\w*[\d_]\w*\b", " ", cleaned_text)
    cleaned_text = re.sub(r"'.*?'", "", cleaned_text)
    cleaned_text = cleaned_text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    return cleaned_text.strip().lower()


def normalize_error_value(evalue: str, traceback_lines: List[str], ename: str) -> str:
    ev = (evalue or "").strip()
    if ev.lower() in NON_DESCRIPTIVE_EVALUES:
        inferred = _extract_evalue_from_traceback(traceback_lines, ename)
        if inferred is not None:
            ev = inferred
        else:
            tb = "\n".join([_safe_str(x) for x in traceback_lines if x is not None]).strip()
            if tb:
                last = tb.splitlines()[-1].strip()
                ev = last if last else ev
    ev = _process_noisy_evalue(ev, ename)
    ev = ev.lower()
    if ename == "KeyError":
        ev = "key" if _QUOTED_RE.search(ev) else ev
    ev = _URL_RE.sub(" url ", ev)
    ev = _PATH_RE.sub(" path ", ev)
    ev = _HEX_RE.sub(" hex ", ev)
    ev = _QUOTED_RE.sub(" str ", ev)
    ev = _NUM_RE.sub(" num ", ev)
    ev = _PUNCT_RE.sub(" ", ev)
    ev = re.sub(r"\s+", " ", ev).strip()
    words = ev.split()
    kept: List[str] = []
    for w in words:
        if w in IMPORTANT_TOKENS_ALLOWLIST:
            kept.append(w)
            continue
        if _IDENT_RE.fullmatch(w) and len(w) >= 3:
            kept.append("id")
        else:
            kept.append(w)
    ev = " ".join(kept)
    ev = re.sub(r"\b(id\s+){3,}id\b", "id id", ev)
    ev = re.sub(r"\s+", " ", ev).strip()
    return ev


def tokenize_norm(ev_norm: str) -> Tuple[str, ...]:
    toks = [t for t in ev_norm.split() if t and t not in {"the", "a", "an", "to", "of", "in"}]
    return tuple(sorted(set(toks)))


def tokens_for_clustering(evalue: str, ename: str, traceback_lines: List[str]) -> Tuple[str, ...]:
    base = preprocess_text_similarity(evalue) if evalue else ""
    if not base:
        inferred = _extract_evalue_from_traceback(traceback_lines, ename)
        base = preprocess_text_similarity(inferred or "")
    if not base:
        base = (ename or "unknown").lower()
    return tokenize_norm(base)


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def cluster_by_jaccard(token_sets: List[Set[str]], threshold: float = 0.7) -> List[int]:
    """Cluster indices by Jaccard similarity using Union-Find."""
    n = len(token_sets)
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1

    buckets: Dict[Tuple[str, ...], List[int]] = {}
    for i, s in enumerate(token_sets):
        sig = tuple(sorted(list(s))[:6])
        buckets.setdefault(sig, []).append(i)
    print( f"Created {len(buckets)} buckets for candidate pairs" )

    candidates: Set[Tuple[int, int]] = set()
    for idxs in tqdm(buckets.values(), desc="Clustering: bucket candidates", unit="bucket"):
        if len(idxs) <= 1:
            continue
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                candidates.add((idxs[i], idxs[j]))

    for i, si in tqdm(enumerate(token_sets), total=n, desc="Clustering: add candidates", unit="i"):
        for j in range(i + 1, n):
            if (i, j) in candidates:
                continue
            sj = token_sets[j]
            if abs(len(si) - len(sj)) >= max(5, int(0.5 * max(len(si), len(sj)))):
                continue
            candidates.add((i, j))

    for i, j in tqdm(candidates, desc="Clustering: union candidates", unit="pair"):
        if jaccard(token_sets[i], token_sets[j]) >= threshold:
            union(i, j)

    roots = [find(i) for i in range(n)]
    remap: Dict[int, int] = {}
    next_id = 0
    out = [0] * n
    print( f"Formed {len(set(roots))} clusters" )
    for i, r in enumerate(roots):
        if r not in remap:
            remap[r] = next_id
            next_id += 1
        out[i] = remap[r]
    print( f"Assigned cluster IDs from 0 to {next_id - 1}" )
    return out


def infer_packages(traceback_lines: List[str]) -> Tuple[str, ...]:
    pkgs: List[str] = []
    for line in traceback_lines:
        s = _safe_str(line)
        m = re.search(r"site-packages[/\\]([A-Za-z0-9_\-]+)", s)
        if m:
            pkgs.append(m.group(1))
            continue
        m2 = re.search(r"dist-packages[/\\]([A-Za-z0-9_\-]+)", s)
        if m2:
            pkgs.append(m2.group(1))
            continue
    uniq = sorted(set([p.lower() for p in pkgs if p]))
    return tuple(uniq)


def infer_library_cause(traceback_lines: List[str]) -> bool:
    joined = "\n".join(traceback_lines or [])
    has_site_pkgs = bool(_SITE_PACKAGES_HINT_RE.search(joined))
    has_user_frame = bool(_TRACEBACK_FRAME_HINT_RE.search(joined))
    return has_site_pkgs and not has_user_frame


def infer_is_ml_bug(pkgs: Tuple[str, ...], ename: str, evalue: str, traceback_lines: List[str]) -> bool:
    if any(p in ML_PACKAGE_HINTS for p in pkgs):
        return True
    text = (" ".join(traceback_lines) + " " + ename + " " + (evalue or "")).lower()
    return any(re.search(rf"\b{re.escape(p)}\b", text) for p in ML_PACKAGE_HINTS)


def classify_crash_type(ename: str, evalue_norm: str, traceback_lines: List[str]) -> str:
    en = (ename or "").strip()
    ev = evalue_norm

    if en in {"ModuleNotFoundError"} or ("no module named" in ev):
        return "module not found"
    if en in {"ImportError"}:
        return "environment error"
    if en in {"NameError"} or ("name id is not defined" in ev) or ("is not defined" in ev and "name" in ev):
        return "variable not found"

    if en in {"MemoryError"} or ("out of memory" in ev) or ("resource exhausted" in ev):
        return "OOM"

    if en in {"FileNotFoundError", "PermissionError", "IOError", "OSError"}:
        return "io error"
    if "no such file" in ev or ("permission" in ev and "denied" in ev):
        return "io error"

    if "connection" in ev or "timeout" in ev or "ssl" in ev or "certificate" in ev:
        return "request error"

    if "operands could not be broadcast" in ev or ("broadcast" in ev and "shapes" in ev):
        return "unsupported broadcast"

    if "incompatible" in ev and ("shape" in ev or "ndim" in ev):
        return "tensor shape mismatch"
    if "expected" in ev and "shape" in ev and "found" in ev:
        return "tensor shape mismatch"
    if "cannot reshape" in ev and "into shape" in ev:
        return "tensor shape mismatch"

    if "feature" in ev and ("mismatch" in ev or "names" in ev or "name" in ev):
        return "feature name mismatch"
    if "column" in ev and ("not in" in ev or "missing" in ev):
        return "feature name mismatch"

    if "metrics can t handle a mix" in ev or ("mix" in ev and "targets" in ev):
        return "data value violation"
    if "input contains nan" in ev or "contains inf" in ev or "not finite" in ev:
        return "data value violation"

    if en in {"KeyError"}:
        return "key error"
    if en in {"IndexError"}:
        return "index error"

    if en in {"AttributeError"}:
        return "attribute error"
    if en in {"TypeError"}:
        return "type error"
    if en in {"ValueError"}:
        return "value error"

    if "invalid argument" in ev or ("unexpected keyword argument" in ev) or ("takes" in ev and "positional" in ev):
        return "invalid argument"

    if en in {"RuntimeError"}:
        return "runtime error"

    return "other"


def infer_root_cause(
    crash_type: str,
    ename: str,
    evalue_norm: str,
    notebook_exec_out_of_order: bool,
    has_prior_error: bool,
    pkgs: Tuple[str, ...],
    library_cause: bool,
) -> str:
    ev = evalue_norm
    en = (ename or "")

    if crash_type in {"module not found", "io error", "request error", "environment error"}:
        return "environment setting"
    if crash_type == "OOM":
        return "insufficient resource"

    if crash_type in {"variable not found", "name error"}:
        return "NB specific" if (notebook_exec_out_of_order or has_prior_error) else "implementation error"

    if crash_type in {"key error", "index error", "feature name mismatch"}:
        return "data confusion"

    if crash_type in {"tensor shape mismatch", "unsupported broadcast", "data value violation"}:
        if any(p in {"tensorflow", "keras", "torch", "sklearn", "xgboost", "lightgbm", "catboost"} for p in pkgs):
            if any(k in ev for k in {"expected", "incompatible", "received", "compile", "fit", "model", "layer"}):
                return "API misuse"
            return "ML model confusion"
        return "data confusion"

    if crash_type in {"invalid argument"}:
        return "API misuse"

    if crash_type == "attribute error":
        return "API misuse" if ("has no attribute" in ev or "no attribute" in ev) else "implementation error"

    if crash_type in {"type error", "value error"}:
        api_markers = [
            "unexpected keyword argument", "positional arguments", "got multiple values",
            "takes num", "missing required", "not supported between instances",
        ]
        if any(m in ev for m in api_markers):
            return "API misuse"
        data_markers = ["shape", "shapes", "broadcast", "nan", "inf", "column", "columns", "feature", "key", "index"]
        if any(m in ev for m in data_markers):
            return "data confusion"
        return "implementation error"

    if library_cause:
        return "library cause"

    if notebook_exec_out_of_order or has_prior_error:
        return "NB specific"

    return "implementation error"


def notebook_out_of_order(nb: nbformat.NotebookNode) -> bool:
    exec_counts: List[Tuple[int, int]] = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        ec = cell.get("execution_count")
        if isinstance(ec, int):
            exec_counts.append((i, ec))
    if len(exec_counts) < 3:
        return False
    inversions = 0
    total = 0
    for k in range(1, len(exec_counts)):
        total += 1
        if exec_counts[k][1] < exec_counts[k - 1][1]:
            inversions += 1
    return inversions / max(1, total) >= 0.15


def analyze(root, jaccard_threshold: float = 0.7, last_fix: bool = False) -> Dict[str, Any]:
    crashes: List[CrashRecord] = []

    if last_fix:
        paths = root
    else:
        paths = glob(str(root / "*.ipynb"))
    print( f"Found {len(paths)} notebooks" )
    for nb_path in tqdm(paths):
        try:
            nb = nbformat.read(str(nb_path), as_version=4)
        except Exception:
            continue

        err_outputs = extract_error_outputs(nb)

        if not err_outputs or len(err_outputs) == 0:
            continue

        oo = notebook_out_of_order(nb)


        # prior_error_cells: Set[int] = set()
        # for e in err_outputs:
        #     prior_error_cells.add(e["cell_index"])

        seen_error_cells: Set[int] = set()
        for e in err_outputs:
            cell_index = int(e["cell_index"])
            # has_prior_error = any(ci < cell_index for ci in prior_error_cells)

            has_prior_error = any(ci < cell_index for ci in seen_error_cells)
            seen_error_cells.add(cell_index)

            ename = e["ename"]
            evalue = e["evalue"]
            tb_lines = [_parse_traceback_line(str(x)) for x in (e["traceback"] or []) if x is not None]
            tb_text = "\n".join(tb_lines)

            ev_norm = normalize_error_value(evalue, tb_lines, ename)
            toks = tokens_for_clustering(evalue, ename, tb_lines)
            tok_set = set(toks)

            pkgs = infer_packages(tb_lines)
            lib_cause = infer_library_cause(tb_lines)
            is_ml = infer_is_ml_bug(pkgs, ename, evalue, tb_lines)

            crash_type = classify_crash_type(ename, ev_norm, tb_lines)
            root_cause = infer_root_cause(
                crash_type=crash_type,
                ename=ename,
                evalue_norm=ev_norm,
                notebook_exec_out_of_order=oo,
                has_prior_error=has_prior_error,
                pkgs=pkgs,
                library_cause=lib_cause,
            )

            crashes.append(CrashRecord(
                notebook_path=str(nb_path),
                cell_index=cell_index,
                execution_count=e.get("execution_count"),
                ename=ename,
                evalue=evalue,
                traceback_text=tb_text,
                error_value_norm=ev_norm,
                tokens=tuple(toks),
                crash_type=crash_type,
                root_cause=root_cause,
                is_ml_bug=is_ml,
                packages=pkgs,
            ))

    token_sets = [set(c.tokens) for c in crashes]
    print( f"Clustering {len(token_sets)} crashes with Jaccard threshold {jaccard_threshold}" )
    cluster_ids = cluster_by_jaccard(token_sets, threshold=jaccard_threshold) if crashes else []
    crashes2: List[CrashRecord] = []
    for c, cid in tqdm(zip(crashes, cluster_ids), total=len(crashes), desc="Assigning cluster IDs", unit="crash"):
        crashes2.append(CrashRecord(**{**asdict(c), "tokens": c.tokens, "packages": c.packages, "cluster_id": cid}))  # type: ignore

    rows = [asdict(c) for c in crashes2]

    def count_by(key: str) -> Dict[str, int]:
        d: Dict[str, int] = {}
        for r in rows:
            v = r.get(key, "Unknown")
            d[v] = d.get(v, 0) + 1
        return dict(sorted(d.items(), key=lambda x: (-x[1], x[0])))

    def co_occurrence(a: str, b: str) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for r in rows:
            va = r.get(a, "Unknown")
            vb = r.get(b, "Unknown")
            out.setdefault(va, {})
            out[va][vb] = out[va].get(vb, 0) + 1
        for va in list(out.keys()):
            out[va] = dict(sorted(out[va].items(), key=lambda x: (-x[1], x[0])))
        return dict(sorted(out.items(), key=lambda x: (-sum(x[1].values()), x[0])))

    clusters: Dict[int, List[int]] = {}
    for i, r in enumerate(rows):
        cid = int(r.get("cluster_id", -1))
        clusters.setdefault(cid, []).append(i)
    print( f"Formed {len(clusters)} clusters" )

    top_clusters = sorted(clusters.items(), key=lambda x: -len(x[1]))[:101] if len(clusters) >= 100 else sorted(clusters.items(), key=lambda x: -len(x[1]))
    cluster_summaries: List[Dict[str, Any]] = []
    for cid, idxs in top_clusters:
        reps = [rows[i]["error_value_norm"] for i in idxs[:5]]
        cluster_summaries.append({
            "cluster_id": cid,
            "size": len(idxs),
            "representative_error_value_norm": reps,
            "top_crash_types": count_by_in_rows([rows[i] for i in idxs], "crash_type", top_k=5),
            "top_root_causes": count_by_in_rows([rows[i] for i in idxs], "root_cause", top_k=5),
        })
    print( f"Prepared {len(cluster_summaries)} cluster summaries" )

    summary = {
        "n_crashes": len(rows),
        "jaccard_threshold": jaccard_threshold,
        "counts": {
            "exception_type": count_by("ename"),
            "crash_type": count_by("crash_type"),
            "root_cause": count_by("root_cause"),
            "is_ml_bug": {
                "ML": sum(1 for r in rows if r.get("is_ml_bug")),
                "Python": sum(1 for r in rows if not r.get("is_ml_bug")),
            },
        },
        "co_occurrence": {
            "crash_type_x_root_cause": co_occurrence("crash_type", "root_cause"),
        },
        "top_clusters": cluster_summaries,
        "rows": rows,
    }
    return summary


def count_by_in_rows(rows: List[Dict[str, Any]], key: str, top_k: int = 10) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for r in rows:
        v = r.get(key, "Unknown")
        d[v] = d.get(v, 0) + 1
    items = sorted(d.items(), key=lambda x: (-x[1], x[0]))
    return dict(items[:top_k])


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    import csv
    if not rows:
        out_path.write_text("")
        return
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def _analyze_fix_worker(fix_num: int, upgrade_type: str, run_type: str, jaccard_threshold: float, model_name: str, q: "mp.Queue", last_fix: bool):
    
    target = f"./results/upgrade/{model_name}_{upgrade_type}/script_out_allINone/fix_{fix_num}"


    start = time.monotonic()
    root = Path(target).expanduser().resolve()
    rep = analyze(root, jaccard_threshold=jaccard_threshold, last_fix=last_fix)
    end = time.monotonic()

    out_json = f"./crash_report/crash_report_{run_type}_{model_name}_{upgrade_type}-{fix_num}.json"
    out_csv = f"./crash_report/crashes_{run_type}_{model_name}_{upgrade_type}-{fix_num}.csv"
    
    Path(out_json).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(rep["rows"], Path(out_csv))

    q.put((fix_num, (end - start) / 60.0))

def _analyze_last_fix_worker(root, analyze_type: str, upgrade_type: str, run_type: str, jaccard_threshold: float, model_name: str, q: "mp.Queue", last_fix: bool):

    start = time.monotonic()
    rep = analyze(root, jaccard_threshold=jaccard_threshold, last_fix=last_fix)
    end = time.monotonic()

    out_json = f"./crash_report/crash_report_{run_type}_{model_name}_{upgrade_type}-{analyze_type}.json"
    out_csv = f"./crash_report/crashes_{run_type}_{model_name}_{upgrade_type}-{analyze_type}.csv"
    
    Path(out_json).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(rep["rows"], Path(out_csv))

    q.put((analyze_type, (end - start) / 60.0))

def _has_error(nb: str) -> bool:
    for i, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            for out in cell.get('outputs', []):
                if out.get('output_type') == 'error':
                    return True
    
    return False

if __name__ == "__main__":

    baseline = "./results/baseline/script_out_allINone"

    parser = argparse.ArgumentParser(
        description="Code upgrade runner."
    )
    parser.add_argument(
        "--run-type",
        type=str,
        required=True,
        default="downgrade",
        choices=["upgrade", "downgrade", "baseline"],
        help="run type: upgrade or downgrade or baseline",
    )

    parser.add_argument(
        "--upgrade-type",
        help="Upgrade type for file or cell",
        type=str,
        required=False,
        default=None,
    )

    parser.add_argument(
        "--model-name",
        help="Model name for GPT-5.2 (gpt) or GPT-OSS-120b (oss)",
        type=str,
        required=False,
        default=None,
    )

    parser.add_argument(
        "--last-fix",
        help="Inspect the crash report of the last fix only",
        action="store_true",
    )
    parser.add_argument(
        "--jaccard-threshold",
        help="Jaccard threshold for clustering",
        type=float,
        required=False,
        default=0.7,
    )

    args = parser.parse_args()
    run_type = args.run_type
    upgrade_type = args.upgrade_type
    last_fix = args.last_fix
    jaccard_threshold = args.jaccard_threshold
    model_name = args.model_name

    os.makedirs('./crash_report', exist_ok=True)

    if run_type == "upgrade":
        ctx = mp.get_context("spawn")
        q: mp.Queue = ctx.Queue()
        procs: List[mp.Process] = []

        if last_fix:
            with open(f"./results/upgrade/{model_name}_{upgrade_type}/executable_files_w_timer_parrallel.json", "r", encoding="utf-8") as f:
                executed_files = json.load(f)
            er = []
            enr = []
            for k,v in tqdm(executed_files.items()):
                if v.get("failed", 0) == 1:
                    continue
                if v['fix'] == 0:
                    continue

                fixes = v["fix"]
                up = v["upgrade"][str(fixes)]
                final_fix_path = f"./results/upgrade/{model_name}_{upgrade_type}/script_out_allINone/fix_{fixes}/{k}"
                with open(final_fix_path, "r") as f:
                    pred_nb = json.load(f)
                error_nb = _has_error(pred_nb)
                if not error_nb:
                    continue

                repli = v.get("replicable", False)
                if repli:
                    er.append(os.path.abspath(final_fix_path))
                else:
                    enr.append(os.path.abspath(final_fix_path))
                
            print("#ER: ", len(er), "#ENR: ", len(enr))

            for analyze_type in ["ER","ENR"]:
                p = ctx.Process(
                    target=_analyze_last_fix_worker,
                    args=(er if analyze_type == "ER" else enr, analyze_type, upgrade_type, run_type, jaccard_threshold, model_name, q, last_fix),
                )
                p.start()
                procs.append(p)

            with tqdm(total=2, desc="Analyze fixes", unit="fix") as pbar:
                done = 0
                while done < 2:
                    fix_num, mins = q.get()
                    pbar.update(1)
                    pbar.set_postfix({"last_fix": fix_num, "mins": f"{mins:.2f}"})
                    done += 1    

        else:
            for fix_num in range(1, 17):
                p = ctx.Process(
                    target=_analyze_fix_worker,
                    args=(fix_num, upgrade_type, run_type, jaccard_threshold, model_name, q, last_fix),
                )
                p.start()
                procs.append(p)

            with tqdm(total=16, desc="Analyze fixes", unit="fix") as pbar:
                done = 0
                while done < 16:
                    fix_num, mins = q.get()
                    pbar.update(1)
                    pbar.set_postfix({"last_fix": fix_num, "mins": f"{mins:.2f}"})
                    done += 1    
                    
        for p in procs:
            p.join()

            # print(json.dumps({
            #     "n_crashes": rep["n_crashes"],
            #     "jaccard_threshold": rep["jaccard_threshold"],
            #     "top_crash_types": list(rep["counts"]["crash_type"].items())[:10],
            #     "top_root_causes": list(rep["counts"]["root_cause"].items())[:10],
            #     "top_exception_types": list(rep["counts"]["exception_type"].items())[:10],
            # }, indent=2, ensure_ascii=False))
    else:
        target = f"./results/{run_type}/script_out_allINone"
        out_json = f"./crash_report/crash_report_{model_name}_{run_type}.json"
        out_csv = f"./crash_report/crashes_{model_name}_{run_type}.csv"

        start = time.monotonic()

        root = Path(target).expanduser().resolve()
        rep = analyze(root, jaccard_threshold=jaccard_threshold)

        end = time.monotonic()

        Path(out_json).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        write_csv(rep["rows"], Path(out_csv))

        print(json.dumps({
            "n_crashes": rep["n_crashes"],
            "jaccard_threshold": rep["jaccard_threshold"],
            "top_crash_types": list(rep["counts"]["crash_type"].items())[:10],
            "top_root_causes": list(rep["counts"]["root_cause"].items())[:10],
            "top_exception_types": list(rep["counts"]["exception_type"].items())[:10],
        }, indent=2, ensure_ascii=False))
        print( f"Analysis completed in {(end - start) / 60:.2f} mins" )
