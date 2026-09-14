# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

4.662682011302231

# 6. Current score

73.71096

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 109.59778) has done: 'Your notebook fails because it references external Kaggle dataset submissions that are not present in your environment, so `pd.read_csv()` raises `FileNotFoundError` and all later variables (like `sub0`, `subfinal`) never get created. I make the ensemble robust by (1) filtering `subs` down to files that actually exist, (2) falling back to the provided `sample_submission.csv` if none exist so a valid `.csv` is always produced, and (3) fixing pandas chained-indexing issues and the broken vote-summing logic so it runs deterministically end-to-end. This keeps the same intended “pick the submission with more matching strings vs others” core logic, but makes it safe and executable in this Kaggle filesystem. Since no valid submission was previously produced (“Not yielded”), the primary goal is to generate a correctly formatted `submission.csv`; score improvement isn’t possible without the missing external prediction files, so the fallback output the sample submission content.'
- What this solution (achieved 105.4566) has done: 'The timeout is dominated by the fallback path doing extremely expensive nearest-neighbor retrieval: it loads and thumbnails ~485k test images and does huge matrix multiplications against 20k train candidates. To keep identical core behavior but fit 600s, the script below avoids that heavy fallback unless it’s truly needed, and speeds up the ensemble step by replacing the O(N·K²) equality cube with an equivalent but much cheaper row-wise vote counting. It also ensures all submission CSVs are aligned by `image_id` via a single merge (preserving semantics while preventing silent misalignment), and it removes unused heavy imports that slow startup.'
- What this solution (achieved 105.4566) has done: 'Your current score is far worse than the target (lower is better), so we should improve it with minimal changes. Since the core logic is an ensemble that only works well when real model submissions exist, the biggest score win without changing modeling is to ensure we actually find and load any available submission CSVs under common Kaggle input locations (your current hardcoded `../input/...` paths likely don’t exist here, forcing the weak NN fallback). I minimally extend the submission-file discovery to search `/kaggle/input/**/submission*.csv` and `/kaggle/data/**/submission*.csv`, keep your exact “pick the row-wise most-agreed InChI” voting, and only use the expensive NN fallback if no submissions are found. This should move the score substantially toward the target while preserving your ensemble semantics.'
- What this solution (achieved 86.99777) has done: 'Your score is far above the target (lower is better), and the biggest likely cause is that you are still falling back to the weak NN thumbnail retriever because no strong external `submission*.csv` files actually exist in this offline environment. To move the score toward the target with minimal semantic change, I keep your exact ensemble logic but add a much stronger fallback: build a simple InChI prior from `train_labels.csv` (most frequent InChI overall) and use that for all test rows when no real submissions are found. This preserves the “produce a valid submission without training a model” core behavior, but should drastically reduce mean Levenshtein distance versus the current random-NN-like fallback. I also keep the NN fallback as a secondary option (now unused by default) in case you explicitly want it later.'
- What this solution (achieved 85.99727) has done: 'Your current score (86.99777, lower is better) is still far from the target, and the main reason is that when no real model submissions are found, the fallback predicts a single most-frequent InChI for all rows, which is weak. Keeping your ensemble logic identical, I strengthen only the fallback by building a simple conditional prior: predict the most frequent InChI within a small set of “bucket” groups derived from `image_id` prefixes (which correlate with how images are stored), and fall back to the global most-frequent only if a bucket is missing. This remains a non-model, fast, deterministic fallback and should materially reduce mean Levenshtein distance versus the single global label, moving the score toward the target. I also keep the existing discovery/merge/vote code unchanged so if real submissions exist, they still drive the output.'
- What this solution (achieved 73.71096) has done: 'We need to move the score (lower is better) substantially toward the target, but your current pipeline almost always hits the “no external submissions found” fallback; the bucket-by-image_id prior isn’t actually informative for this dataset, so it stays very weak. With minimal change and identical overall semantics (still a fast prior-based fallback, no model training), we replace the bucket heuristic with a stronger, still-deterministic fallback: choose the most frequent InChI among “simple/small” molecules (short strings) from `train_labels.csv`, which empirically better matches common structures and reduces mean edit distance versus an arbitrary global mode. We also normalize predicted strings to ensure they always start with `InChI=` (some priors can include whitespace/NaNs), avoiding format-caused distance penalties. Everything else (submission discovery, alignment, voting) stays the same and it still write `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

os.environ.setdefault("PYTHONHASHSEED", "0")




## === cell 1
subs = [
    "../input/inchi-efficientnetb7-25-epochs-inference/submission.csv",
    "../input/efficientnet-multi-layer-lstm-inference/submission.csv",
    "../input/batched-beam-search-inference-changed-parameters/submission.csv",
    "../input/batched-beam-search-inference/submission.csv",
    "../input/subbeem/submission.789.csv",
]

DISCOVERY_GLOBS = [
    "/kaggle/input/**/submission*.csv",
    "/kaggle/data/**/submission*.csv",
]

discovered = []
for pat in DISCOVERY_GLOBS:
    discovered.extend(glob.glob(pat, recursive=True))

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

print(f"Discovered {len(existing_subs)} existing submission candidate files.")




## === cell 2
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
        )  # (b,)
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


def most_frequent_inchi_fallback_submission(
    sample_submission_path: str,
    train_labels_csv: str,
) -> pd.DataFrame:
    """
    CHANGE (score-relevant, minimal): Replace the weak image_id-prefix bucket prior with a
    stronger deterministic prior that tends to reduce mean Levenshtein distance:
    choose the most frequent InChI among "short" InChIs (simple molecules), which are
    much closer on average than a global mode driven by long rare structures.
    Also normalize strings to always start with 'InChI=' to avoid avoidable distance.
    """
    base = pd.read_csv(
        sample_submission_path, usecols=["image_id"], dtype={"image_id": "string"}
    )
    labels = pd.read_csv(
        train_labels_csv,
        usecols=["InChI"],
        dtype={"InChI": "string"},
    )

    inchi = labels["InChI"].dropna().astype("string")
    inchi = inchi.str.strip()

    lengths = inchi.str.len()
    cutoff = int(np.nanpercentile(lengths.to_numpy(), 20))
    short_pool = inchi[lengths <= cutoff]
    if len(short_pool) == 0:
        chosen = inchi.value_counts().index[0]
    else:
        chosen = short_pool.value_counts().index[0]

    chosen = str(chosen).strip()
    if not chosen.startswith("InChI="):
        chosen = "InChI=" + chosen

    out = base.copy()
    out["InChI"] = chosen
    return out




## === cell 3
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




## === cell 4
n_models = len(sub_dfs)
inchi_cols = [f"InChI{i}" for i in range(n_models)]
arr = subfinal[inchi_cols].to_numpy(dtype=object)

counts = np.empty((arr.shape[0], n_models), dtype=np.int16)
for r in range(arr.shape[0]):
    row = arr[r]
    uniq, freq = np.unique(row, return_counts=True)
    freq_map = dict(zip(uniq.tolist(), freq.tolist()))
    for i in range(n_models):
        counts[r, i] = freq_map[row[i]] - 1

for i in range(n_models):
    subfinal[f"InChI{i}sum"] = counts[:, i]




## === cell 5
score_cols = [f"InChI{i}sum" for i in range(len(sub_dfs))]
inchi_cols = [f"InChI{i}" for i in range(len(sub_dfs))]

best_idx = subfinal[score_cols].to_numpy().argmax(axis=1)  # ties -> first
best_inchi = subfinal[inchi_cols].to_numpy()[np.arange(len(subfinal)), best_idx]
subfinal["InChI"] = best_inchi

subfinal["image_id"] = subfinal["image_id"].astype(str)
subfinal["InChI"] = subfinal["InChI"].astype(str).str.strip()
subfinal.loc[~subfinal["InChI"].str.startswith("InChI="), "InChI"] = (
    "InChI=" + subfinal.loc[~subfinal["InChI"].str.startswith("InChI="), "InChI"]
)




## === cell 6
subfinal[["image_id", "InChI"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subfinal[["image_id", "InChI"]].shape)
print(subfinal.head())
