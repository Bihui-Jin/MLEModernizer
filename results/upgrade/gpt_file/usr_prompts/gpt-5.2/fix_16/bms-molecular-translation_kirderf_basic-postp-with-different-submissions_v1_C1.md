# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Provided with images of chemicals, predict the corresponding International Chemical Identifier (InChI) text string of the image.

## Metric
Mean [Levenshtein distance](http://en.wikipedia.org/wiki/Levenshtein_distance) between the InChi strings you submit and the ground truth InChi values.

## Submission Format
For each `image_id` in the test set, you must predict the InChi string of the molecule in the corresponding image. The file should contain a header and have the following format:

```
image_id,InChI
00000d2a601c,InChI=1S/H2O/h1H2
00001f7fc849,InChI=1S/H2O/h1H2
000037687605,InChI=1S/H2O/h1H2
etc.
```

## Dataset
- **train/** - the training images, arranged in a 3-level folder structure by `image_id`
- **test/** - the test images, arranged in the same folder structure as `train/`
- **train_labels.csv** - ground truth InChi labels for the training images
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (68 lines)
            extra_approved_InChIs.csv (9998712 lines)
            extra_approved_InChIs.csv.zip (428.7 MB)
            sample_submission.csv (484839 lines)
            sample_submission.csv.zip (4.1 MB)
            test.zip (883.2 MB)
            train.zip (3.5 GB)
            train_labels.csv (1939349 lines)
            train_labels.csv.zip (102.5 MB)
            bms-molecular-translation/
                description.md (68 lines)
                extra_approved_InChIs.csv (9998712 lines)
                ... and 7 other files
                bms-molecular-translation/
                test/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
                train/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
            test/
                0/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                1/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                ... and 15 other folders
            train/
                0/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                1/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                ... and 15 other folders
        input/
            description.md (68 lines)
            extra_approved_InChIs.csv (9998712 lines)
            extra_approved_InChIs.csv.zip (428.7 MB)
            sample_submission.csv (484839 lines)
            sample_submission.csv.zip (4.1 MB)
            test.zip (883.2 MB)
            train.zip (3.5 GB)
            train_labels.csv (1939349 lines)
            train_labels.csv.zip (102.5 MB)
            bms-molecular-translation/
                description.md (68 lines)
                extra_approved_InChIs.csv (9998712 lines)
                ... and 7 other files
                bms-molecular-translation/
                test/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
                train/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
            test/
                0/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                1/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                ... and 15 other folders
            train/
                0/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                1/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                ... and 15 other folders
        working/
            bms-molecular-translation/
                description.md (68 lines)
                extra_approved_InChIs.csv (9998712 lines)
                ... and 7 other files
                bms-molecular-translation/
                test/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
                train/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
```

-> data/bms-molecular-translation/extra_approved_InChIs.csv has 9998711 rows and 1 columns.
The columns are: InChI

-> data/bms-molecular-translation/sample_submission.csv has 484838 rows and 2 columns.
The columns are: image_id, InChI

-> data/bms-molecular-translation/train_labels.csv has 1939348 rows and 2 columns.
The columns are: image_id, InChI

-> data/extra_approved_InChIs.csv has 9998711 rows and 1 columns.
The columns are: InChI

-> data/sample_submission.csv has 484838 rows and 2 columns.
The columns are: image_id, InChI

-> data/train_labels.csv has 1939348 rows and 2 columns.
The columns are: image_id, InChI

-> input/bms-molecular-translation/extra_approved_InChIs.csv has 9998711 rows and 1 columns.
The columns are: InChI

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import glob
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

os.environ.setdefault("PYTHONHASHSEED", "0")

subs = [
    "../input/inchi-efficientnetb7-25-epochs-inference/submission.csv",
    "../input/efficientnet-multi-layer-lstm-inference/submission.csv",
    "../input/batched-beam-search-inference-changed-parameters/submission.csv",
    "../input/batched-beam-search-inference/submission.csv",
    "../input/subbeem/submission.789.csv",
]

DISCOVERY_GLOBS = [
    "/kaggle/input/*/submission*.csv",
    "/kaggle/data/*/submission*.csv",
    "/kaggle/input/*/*/submission*.csv",
    "/kaggle/data/*/*/submission*.csv",
]

discovered = []
for pat in DISCOVERY_GLOBS:
    discovered.extend(glob.glob(pat))

all_candidates = subs + discovered
seen = set()
all_candidates = [p for p in all_candidates if not (p in seen or seen.add(p))]

existing_subs = [p for p in all_candidates if os.path.exists(p) and os.path.isfile(p)]

SAMPLE_CANDIDATES = [
    "/kaggle/input/bms-molecular-translation/sample_submission.csv",
    "/kaggle/data/bms-molecular-translation/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]
sample_path = next((p for p in SAMPLE_CANDIDATES if os.path.exists(p)), None)

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/bms-molecular-translation",
    "/kaggle/data/bms-molecular-translation",
]
data_root = next((p for p in DATA_ROOT_CANDIDATES if os.path.exists(p)), None)
if data_root is None:
    raise FileNotFoundError(
        "Could not find bms-molecular-translation data root in expected locations."
    )

train_labels_path = os.path.join(data_root, "train_labels.csv")
test_dir = os.path.join(data_root, "test")
train_dir = os.path.join(data_root, "train")

extra_inchi_path = None
for p in [
    os.path.join(data_root, "extra_approved_InChIs.csv"),
    "/kaggle/input/bms-molecular-translation/extra_approved_InChIs.csv",
    "/kaggle/data/bms-molecular-translation/extra_approved_InChIs.csv",
    "/kaggle/input/extra_approved_InChIs.csv",
    "/kaggle/data/extra_approved_InChIs.csv",
]:
    if os.path.exists(p) and os.path.isfile(p):
        extra_inchi_path = p
        break

print(f"Discovered {len(existing_subs)} existing submission candidate files.")
print(
    "extra_approved_InChIs.csv found:",
    bool(extra_inchi_path),
    "path:",
    extra_inchi_path,
)




## === cell 1
def nn_retrieval_fallback_submission(
    sample_submission_path: str,
    train_labels_csv: str,
    train_root: str,
    test_root: str,
    n_train_candidates: int = 20000,
    n_test_to_process: int | None = None,
    thumb_size=(32, 32),
    chunk_test: int = 256,
    seed: int = 123,
    only_test_ids: list[str] | None = None,
    cache_dir: str = "/kaggle/working",
) -> pd.DataFrame:
    """
    Build a valid submission by copying the InChI from the most visually similar training image
    (nearest neighbor in thumbnail pixel space).
    """
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    try:
        import cv2  # type: ignore

        _HAS_CV2 = True
    except Exception:
        cv2 = None
        _HAS_CV2 = False

    def image_path_from_id(root_dir: str, image_id: str) -> str:
        return os.path.join(
            root_dir, image_id[0], image_id[1], image_id[2], f"{image_id}.png"
        )

    def _load_thumb_gray_pil(image_path: str, size=(32, 32)) -> np.ndarray:
        with Image.open(image_path) as im:
            im = im.convert("L").resize(size, resample=Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        return arr

    def _load_thumb_gray_cv2(image_path: str, size=(32, 32)) -> np.ndarray:
        im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(image_path)
        im = cv2.resize(im, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
        return im.astype(np.float32) * (1.0 / 255.0)

    load_thumb_gray = _load_thumb_gray_cv2 if _HAS_CV2 else _load_thumb_gray_pil

    def _feats_from_ids(
        root_dir: str, ids: np.ndarray, thumb_size, max_workers: int
    ) -> np.ndarray:
        n = len(ids)
        out = np.empty((n, thumb_size[0] * thumb_size[1]), dtype=np.float32)

        def _one(i_id):
            i, image_id = i_id
            p = image_path_from_id(root_dir, image_id)
            return i, load_thumb_gray(p, size=thumb_size).ravel()

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, vec in ex.map(_one, enumerate(ids), chunksize=64):
                out[i] = vec
        return out

    rng = np.random.default_rng(seed)

    sample_sub = pd.read_csv(sample_submission_path, usecols=["image_id"])
    if "image_id" not in sample_sub.columns:
        raise ValueError(
            f"sample_submission at {sample_submission_path} missing image_id column"
        )

    if only_test_ids is None:
        test_ids = sample_sub["image_id"].astype(str).to_numpy()
        if n_test_to_process is not None:
            test_ids = test_ids[:n_test_to_process]
    else:
        test_ids = np.asarray([str(x) for x in only_test_ids], dtype=object)
        if n_test_to_process is not None:
            test_ids = test_ids[:n_test_to_process]

    labels = pd.read_csv(train_labels_csv, usecols=["image_id", "InChI"])
    labels["image_id"] = labels["image_id"].astype(str)
    if n_train_candidates > len(labels):
        n_train_candidates = len(labels)

    train_idx = rng.choice(len(labels), size=n_train_candidates, replace=False)
    train_subset = labels.iloc[train_idx].reset_index(drop=True)

    train_inchi = train_subset["InChI"].astype(str).to_numpy()
    train_ids = train_subset["image_id"].astype(str).to_numpy()

    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"nncache_train_{seed}_{n_train_candidates}_{thumb_size[0]}x{thumb_size[1]}_{'cv2' if _HAS_CV2 else 'pil'}"
    train_npz = os.path.join(cache_dir, cache_key + ".npz")

    train_feats = None
    train_sq = None
    if os.path.exists(train_npz):
        z = np.load(train_npz, allow_pickle=False)
        train_feats = z["train_feats"]
        train_sq = z["train_sq"]
        if (
            train_feats.shape[0] != n_train_candidates
            or train_feats.shape[1] != thumb_size[0] * thumb_size[1]
        ):
            os.remove(train_npz)
            train_feats = None
            train_sq = None

    max_workers = min(8, (os.cpu_count() or 2))

    if train_feats is None:
        train_feats = _feats_from_ids(
            train_root, train_ids, thumb_size=thumb_size, max_workers=max_workers
        )
        train_sq = np.einsum("ij,ij->i", train_feats, train_feats).astype(
            np.float32, copy=False
        )
        np.savez_compressed(train_npz, train_feats=train_feats, train_sq=train_sq)

    preds = np.empty(len(test_ids), dtype=object)

    for start in range(0, len(test_ids), chunk_test):
        end = min(start + chunk_test, len(test_ids))
        batch_ids = test_ids[start:end]
        test_feats = _feats_from_ids(
            test_root, batch_ids, thumb_size=thumb_size, max_workers=max_workers
        )

        test_sq = np.einsum("ij,ij->i", test_feats, test_feats).astype(
            np.float32, copy=False
        )
        dists = (
            test_sq[:, None] + train_sq[None, :] - 2.0 * (test_feats @ train_feats.T)
        )
        nn_idx = np.argmin(dists, axis=1)
        preds[start:end] = train_inchi[nn_idx]

    out = pd.DataFrame({"image_id": test_ids.astype(str), "InChI": preds.astype(str)})

    if only_test_ids is None and n_test_to_process is None:
        out = sample_sub.merge(out, on="image_id", how="left")
        out["InChI"] = out["InChI"].fillna("InChI=1S/H2O/h1H2")

    return out


def _levenshtein_distance(a: str, b: str) -> int:
    """
    Exact Levenshtein distance (used only on a tiny candidate pool).
    """
    if a == b:
        return 0
    if len(a) > len(b):
        a, b = b, a
    la, lb = len(a), len(b)
    prev = np.arange(la + 1, dtype=np.int32)
    curr = np.empty(la + 1, dtype=np.int32)
    for j in range(1, lb + 1):
        bj = b[j - 1]
        curr[0] = j
        for i in range(1, la + 1):
            cost = 0 if a[i - 1] == bj else 1
            curr[i] = min(prev[i] + 1, curr[i - 1] + 1, prev[i - 1] + cost)
        prev, curr = curr, prev
    return int(prev[la])


def most_frequent_inchi_fallback_submission(
    sample_submission_path: str,
    train_labels_csv: str,
    extra_inchis_csv: str | None = None,
    top_k: int = 160,
) -> pd.DataFrame:
    base = pd.read_csv(
        sample_submission_path, usecols=["image_id"], dtype={"image_id": "string"}
    )

    labels = pd.read_csv(
        train_labels_csv,
        usecols=["InChI"],
        dtype={"InChI": "string"},
    )
    s = labels["InChI"].dropna().astype("string").str.strip()
    if len(s) == 0:
        chosen = "InChI=1S/H2O/h1H2"
    else:
        vc = s.value_counts()
        items = list(vc.items())[: min(top_k, len(vc))]

        cand = []
        for st, c in items:
            ss = str(st).strip()
            if not ss.startswith("InChI="):
                ss = "InChI=" + ss
            cand.append((ss, int(c)))

        strings = [x for x, _ in cand]
        weights = np.array([c for _, c in cand], dtype=np.float64)

        lens = np.fromiter((len(x) for x in strings), dtype=np.int32)
        med_len = int(np.median(lens)) if len(lens) else len("InChI=1S/H2O/h1H2")

        best_s = None
        best_val = None
        wsum = float(weights.sum())
        wmean = float(weights.mean()) if len(weights) else 1.0

        for i, si in enumerate(strings):
            di = 0.0
            for j, sj in enumerate(strings):
                if i == j:
                    continue
                di += weights[j] * _levenshtein_distance(si, sj)
            di += 0.25 * abs(len(si) - med_len) * (wsum / max(1.0, wmean))
            if best_val is None or di < best_val:
                best_val = di
                best_s = si

        chosen = best_s if best_s is not None else "InChI=1S/H2O/h1H2"

    out = base.copy()
    out["InChI"] = chosen
    return out




## === cell 2
sub_dfs = []
if len(existing_subs) > 0:
    for p in existing_subs:
        try:
            df = pd.read_csv(
                p,
                usecols=["image_id", "InChI"],
                dtype={"image_id": "string", "InChI": "string"},
            )
        except Exception:
            continue
        if "image_id" not in df.columns or "InChI" not in df.columns:
            continue
        sub_dfs.append(df)

    if len(sub_dfs) == 0:
        existing_subs = []
else:
    existing_subs = []

if sample_path is None:
    raise FileNotFoundError("sample_submission.csv not found in expected locations.")

if len(existing_subs) == 0:
    df = most_frequent_inchi_fallback_submission(
        sample_submission_path=sample_path,
        train_labels_csv=train_labels_path,
        extra_inchis_csv=extra_inchi_path,
    )
    sub_dfs.append(df)

base = pd.read_csv(sample_path, usecols=["image_id"], dtype={"image_id": "string"})
sub_aligned = base.copy()
for i, df in enumerate(sub_dfs):
    df = df.copy()
    df["image_id"] = df["image_id"].astype("string")
    df["InChI"] = df["InChI"].astype("string")
    df = df.drop_duplicates(subset=["image_id"], keep="first")
    sub_aligned = sub_aligned.merge(
        df.rename(columns={"InChI": f"InChI{i}"}), on="image_id", how="left"
    )

for i in range(len(sub_dfs)):
    col = f"InChI{i}"
    sub_aligned[col] = sub_aligned[col].fillna("InChI=1S/H2O/h1H2")

subfinal = sub_aligned
for ii in range(len(sub_dfs)):
    vars()[f"sub{ii}"] = subfinal[["image_id", f"InChI{ii}"]].rename(
        columns={f"InChI{ii}": "InChI"}
    )

print(f"Ensembling {len(sub_dfs)} submissions.")



## === cell 3
n_models = len(sub_dfs)
inchi_cols = [f"InChI{i}" for i in range(n_models)]
arr = subfinal[inchi_cols].to_numpy(dtype=object)
n_rows = arr.shape[0]

flat = arr.reshape(-1)
codes, uniques = pd.factorize(pd.Series(flat, dtype="object"), sort=False)
codes2d = codes.reshape(n_rows, n_models)

order = np.argsort(codes2d, axis=1, kind="quicksort")
sorted_codes = np.take_along_axis(codes2d, order, axis=1)

change = np.empty((n_rows, n_models), dtype=bool)
change[:, 0] = True
change[:, 1:] = sorted_codes[:, 1:] != sorted_codes[:, :-1]

group_id = np.cumsum(change, axis=1) - 1  # int64

row_idx = np.repeat(np.arange(n_rows, dtype=np.int64), n_models)
gid_flat = group_id.reshape(-1).astype(np.int64, copy=False)
keys = row_idx * n_models + gid_flat
group_sizes = np.bincount(keys, minlength=n_rows * n_models).reshape(n_rows, n_models)

sizes_sorted = np.take_along_axis(group_sizes, group_id, axis=1).astype(
    np.int16, copy=False
)
counts_minus1_sorted = (sizes_sorted - 1).astype(np.int16, copy=False)

inv_order = np.empty_like(order)
inv_order[np.arange(n_rows)[:, None], order] = np.arange(n_models)[None, :]
counts_minus1 = np.take_along_axis(counts_minus1_sorted, inv_order, axis=1)

for i in range(n_models):
    subfinal[f"InChI{i}sum"] = counts_minus1[:, i]



## === cell 4
score_cols = [f"InChI{i}sum" for i in range(len(sub_dfs))]
inchi_cols = [f"InChI{i}" for i in range(len(sub_dfs))]

best_idx = subfinal[score_cols].to_numpy().argmax(axis=1)  # ties -> first
best_inchi = subfinal[inchi_cols].to_numpy(dtype=object)[
    np.arange(len(subfinal)), best_idx
]
subfinal["InChI"] = best_inchi

subfinal["image_id"] = subfinal["image_id"].astype(str)
subfinal["InChI"] = subfinal["InChI"].astype(str).str.strip()
subfinal.loc[~subfinal["InChI"].str.startswith("InChI="), "InChI"] = (
    "InChI=" + subfinal.loc[~subfinal["InChI"].str.startswith("InChI="), "InChI"]
)



## === cell 5
subfinal[["image_id", "InChI"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subfinal[["image_id", "InChI"]].shape)
print(subfinal.head())
