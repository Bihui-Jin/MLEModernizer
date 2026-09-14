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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image

os.environ.setdefault("PYTHONHASHSEED", "0")



## === cell 1
subs = [
    "../input/inchi-efficientnetb7-25-epochs-inference/submission.csv",
    "../input/efficientnet-multi-layer-lstm-inference/submission.csv",
    "../input/batched-beam-search-inference-changed-parameters/submission.csv",
    "../input/batched-beam-search-inference/submission.csv",
    "../input/subbeem/submission.789.csv",
]

existing_subs = [p for p in subs if os.path.exists(p)]

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



## === cell 2
from concurrent.futures import ThreadPoolExecutor

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
    (nearest neighbor in thumbnail pixel space). Core math/semantics preserved; runtime improved
    by caching features and using parallel I/O for feature extraction.
    """
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




## === cell 3
sub_dfs = []
if len(existing_subs) > 0:
    for p in existing_subs:
        df = pd.read_csv(p, usecols=["image_id", "InChI"])
        if "image_id" not in df.columns or "InChI" not in df.columns:
            raise ValueError(
                f"Submission at {p} must contain columns ['image_id','InChI'], got {list(df.columns)}"
            )
        df = df.sort_values(by="image_id", ascending=True).reset_index(drop=True)
        sub_dfs.append(df)
else:
    if sample_path is None:
        raise FileNotFoundError(
            "No existing submission files found in ../input/* and no sample_submission.csv found in expected locations."
        )

    df = nn_retrieval_fallback_submission(
        sample_submission_path=sample_path,
        train_labels_csv=train_labels_path,
        train_root=train_dir,
        test_root=test_dir,
        n_train_candidates=20000,
        n_test_to_process=None,
        thumb_size=(32, 32),
        chunk_test=256,
        seed=123,
    )
    df = df.sort_values(by="image_id", ascending=True).reset_index(drop=True)
    sub_dfs.append(df)

for ii, df in enumerate(sub_dfs):
    vars()[f"sub{ii}"] = df



## === cell 4
sums = []
subfinal = sub_dfs[0].copy()

for ii in range(len(sub_dfs)):
    col = f"InChI{ii}"
    subfinal[col] = sub_dfs[ii]["InChI"].values
    subfinal[f"{col}sum"] = 0
    sums.append(col)



## === cell 5
n_models = len(sub_dfs)
inchi_cols = [f"InChI{i}" for i in range(n_models)]
arr = subfinal[inchi_cols].to_numpy(dtype=object)

eq = arr[:, :, None] == arr[:, None, :]  # (rows, K, K) boolean
counts = eq.sum(axis=2) - 1  # (rows, K)
for i in range(n_models):
    subfinal[f"InChI{i}sum"] = counts[:, i].astype(np.int16, copy=False)



## === cell 6
score_cols = [f"InChI{i}sum" for i in range(len(sub_dfs))]
inchi_cols = [f"InChI{i}" for i in range(len(sub_dfs))]

best_idx = subfinal[score_cols].to_numpy().argmax(axis=1)  # ties -> first
best_inchi = subfinal[inchi_cols].to_numpy()[np.arange(len(subfinal)), best_idx]
subfinal["InChI"] = best_inchi

subfinal["image_id"] = subfinal["image_id"].astype(str)
subfinal["InChI"] = subfinal["InChI"].astype(str)



## === cell 7
subfinal[["image_id", "InChI"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subfinal[["image_id", "InChI"]].shape)
print(subfinal.head())
