# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Identify hotels from images.

## Metric
Mean Average Precision @ 5 (MAP@5)

## Submission Format
For each image in the test set, you must predict a space-delimited list of hotel IDs that could match that image. The first ID should be the most relevant one and the last the least relevant one. The file should contain a header and have the following format:

```
image,hotel_id
99e91ad5f2870678.jpg,36363 53586 18807 64314 60181
b5cc62ab665591a9.jpg,36363 53586 18807 64314 60181
d5664a972d5a644b.jpg,36363 53586 18807 64314 60181
```

## Dataset
**train.csv** - The training set metadata.

- `image` - The image ID.

- `chain` - An ID code for the hotel chain. A `chain` of zero (0) indicates that the hotel is either not part of a chain or the chain is not known. This field is not available for the test set. The number of hotels per chain varies widely.

- `hotel_id` - The hotel ID. The target class.

- `timestamp` - When the image was taken. Provided for the training set only.

**sample_submission.csv** - A sample submission file in the correct format.

- `image` The image ID

- `hotel_id` The hotel ID. The target class.

**train_images** - The training set contains 97000+ images from around 7700 hotels from across the globe. All of the images for each hotel chain are in a dedicated subfolder for that chain.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 13,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

dill==0.4.0
fastai==2.8.5
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
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
            train/
                train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        input/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
                    test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
            train/
                train/
                    train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        working/
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
```

-> data/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> data/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> input/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> input/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> (stopped after 10 files for performance)

# 5. Target score

0.6125957733434361

# 6. Current score

0.00209

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'The pipeline fails because it expects exported fastai model `.pkl` files that are not present in your `/kaggle/input` datasets, so inference never runs and no submission is written. I fix this by making the model discovery robust (searching all input roots you actually have) and adding a safe fallback baseline that still produces a valid MAP@5 submission when no `.pkl` models exist. The fallback uses the training metadata to predict the 5 most frequent `hotel_id`s (a common baseline) so you get a non-zero score and a valid `submission.csv`. I also fix the cell numbering to start at 1 and keep paths compatible with the Kaggle file structure you listed.'
- What this solution (achieved 0.00209) has done: 'Your score is far below target (0.00209 vs 0.6126), so we should improve (not degrade). The biggest issue is that your current fallback predicts the same global top-5 hotels for every image, which yields an extremely low MAP@5; we can keep the same overall inference structure but make the fallback “metadata-driven” using the provided `chain` folders in `test_images/` and the `chain` column in `train.csv`. Concretely: for each test image, infer its `chain` from the parent folder name, then predict the top-5 most frequent `hotel_id`s within that chain (and backfill with global top hotels if a chain is missing/unknown). This preserves evaluation semantics, avoids any label leakage, runs fast, and should move the score substantially toward the target even with no `.pkl` models.'
- What this solution (achieved 0.00209) has done: 'Your current score is far below the target, so we should improve (not degrade) with minimal, low-risk changes. The biggest lift while preserving your overall structure is to make the chain-based fallback actually work with this dataset: most `test_images` are in a flat folder (no chain subfolders), so your `_parse_chain_from_path` returns `None` and you effectively predict global top-5 for every image. I keep your fastai inference path unchanged, but improve the fallback by extracting `chain` from the **test image filename** using the provided `test.csv` mapping (this is legitimate metadata, not labels) and then using per-chain top-5 hotel IDs, with a global backfill. This should substantially increase MAP@5 while still running fast and producing a valid `submission.csv`.'
- What this solution (achieved 0.00209) has done: 'Your current score is far below target, so the smallest safe way to move toward it is to make the fallback much more informative without changing the fastai inference core. The key bug is that `test.csv` with chain metadata typically isn’t present, so `test_chains` becomes mostly `None` and you effectively submit the same global top-5 for every image (very low MAP@5). I keep your model-loading/tta path unchanged, but improve the fallback by (1) correctly extracting `chain` from the **folder structure** (handling both flat and `test_images/<chain>/...` layouts) and (2) using per-chain top-50 hotel frequency lists with a robust global backfill. This should materially improve MAP@5 while still running fast and producing a valid `submission.csv`.'
- What this solution (achieved 0.00209) has done: 'We keep your fastai inference path untouched, but make the fallback baseline actually use chain information reliably so it stops predicting the same global top-5 for almost every test image (the main reason MAP@5 is stuck near zero). Concretely, we (1) detect whether the provided `test_images` folder is flat and, if so, also look for a `test_images/<chain>/...` variant and build an `image -> chain` map by scanning directories, and (2) use per-chain top hotel lists as you intended, with a global backfill. This is a minimal change that preserves evaluation semantics, avoids leakage (only uses folder structure + train metadata), runs fast, and should move the score substantially toward the 0.61 target. We also ensure the submission ordering exactly matches `sample_submission.csv`.'
- What this solution (achieved 0.00209) has done: 'Your current score (0.00209) is far below the target (0.6126), so we should improve the fallback path (used when no `.pkl` models are found) with the smallest, safest change. The main reason the fallback is near-random is that most `test_images` are stored flat (no chain subfolders), so your `test_chains` are mostly `None` and you end up predicting the same global top-5 for nearly every image. I keep your fastai inference logic unchanged, but I make chain extraction robust by first scanning for a `test_images/<chain>/*.jpg` layout anywhere under the competition root (and common parents), then using that mapping; this directly increases MAP@5 without changing evaluation semantics. Finally, I ensure the submission stays aligned to `sample_submission.csv` ordering and always writes a valid `submission.csv`.'
- What this solution (achieved 0.00209) has done: 'We keep your fastai inference path unchanged, but significantly strengthen the fallback (used when no `.pkl` is found or inference can’t run), because your current score (0.00209) is far below the target (0.6126) and most of that gap is explained by weak, almost-constant predictions. The key fix is to robustly recover `chain` for each test image by scanning the actual `test_images` directory tree (including nested `test_images/test_images/<chain>/...` layouts) and, if needed, doing a broader search under the competition root—so we stop defaulting to global top-5 for most rows. Then we generate chain-specific top-5 hotel predictions with a global backfill, which is a legitimate metadata-driven baseline for MAP@5 and should move the score substantially upward while staying fast and minimal. Finally, we ensure the submission order matches `sample_submission.csv` exactly and always writes `submission.csv`.'
- What this solution (achieved 0.00209) has done: 'We keep your fastai inference path and model ensembling logic unchanged, but fix the fallback (which you are currently using because no `.pkl` models exist) so it becomes meaningfully informative for MAP@5. The key minimal improvement is to use the `chain` folder structure correctly by building an `image -> chain` mapping from the **training image paths** (where chain subfolders definitely exist), then using that mapping to infer each test image’s chain when test images are stored flat. With reliable chains, we can safely predict the per-chain most frequent `hotel_id`s (backfilled by global top hotels), which should move your score substantially upward toward 0.61 while preserving evaluation semantics and staying fast. We also keep submission ordering exactly aligned to `sample_submission.csv` and always write `submission.csv`.'
- What this solution (achieved 0.0011) has done: 'The main timeout is coming from running `learn.tta(..., n=5)` for each of several large fastai models, which multiplies inference cost by ~5× and repeatedly rebuilds the test dataloader each time. I keep the exact same inference logic and TTA semantics, but (1) build the test dataloader once and reuse it across models, (2) avoid unnecessary CPU/GPU sync and autograd overhead by enforcing inference-only context, and (3) reduce Python-loop overhead in the fallback chain-building/hash code by replacing the worst hotspots (bit counting and dhash accumulation) with equivalent vectorized/bit operations. These changes are provably equivalent (same inputs → same outputs up to negligible FP noise) and target only runtime.'
- What this solution (achieved 0.0011) has done: 'Your current score (0.0011) is far below the target (0.6126), so we should improve (not degrade) with the smallest changes that meaningfully increase MAP@5 while preserving your overall pipeline and fallback logic. The main issue is that your fallback chain inference is effectively failing because you build test image paths as `image_dir/<image>` even when the real layout is nested (e.g., `.../test_images/test_images/<image>`), so the hash-based chain inference and folder scans often miss files and you revert to near-constant global top-5. I minimally (1) build a robust `image -> actual test path` resolver by indexing the test_images tree once, (2) use it in both `_build_img_to_chain_from_folders` and the hash-based chain inference, and (3) keep submission ordering exactly aligned to `sample_submission.csv`. This keeps your model/TTA path unchanged and only strengthens the metadata-driven fallback that is currently dominating your score.'
- What this solution (achieved 0.00209) has done: 'The timeout is almost certainly caused by two hot spots: repeated costly `rglob()` scans over the full dataset directory tree (especially `train_images`/`test_images`) and the extremely slow fallback hash-NN path that builds perceptual hashes for many training images and then does an O(N_test × N_proto) Python loop. The optimized script below keeps the exact same model TTA ensemble and the same fallback logic/semantics, but makes it fast by (1) eliminating expensive directory rescans, (2) replacing row-wise pandas `apply` filtering with vectorized merges, (3) caching computed dhashes to disk and using multiprocessing for hashing, and (4) vectorizing Hamming distance computation with `numpy.unpackbits` so nearest-neighbor ranking is done in fast NumPy instead of Python loops. These changes are performance-only and do not alter what is computed (same hashes, same candidate pools, same top-5 selection rules), so accuracy/evaluation behavior is preserved up to negligible float noise.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

import pandas as pd
import numpy as np
import fastai
from fastai.vision.all import *
import dill

import random
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ.setdefault("OMP_NUM_THREADS", str(max(1, os.cpu_count() // 2)))
os.environ.setdefault("MKL_NUM_THREADS", str(max(1, os.cpu_count() // 2)))
torch.set_num_threads(max(1, os.cpu_count() // 2))




## === cell 1
fastai.__version__, torch.__version__




## === cell 2
models = [
    "../input/fgvc8hotel/export_dn161_Fa_CE_bs32.pkl",  # v7
    "../input/fgvc8hotel/export_dn161_Fa_FL_bs32.pkl",  # v8
    "../input/fgvc8hotel/export_res101_Fall_HQAdam.pkl",  # v5
    "../input/fgvc8hotel/export_res101_Fall_5it_4.pkl",  # v11
    "../input/hotel-train-fastai-densnet161/export_dn161_kaggle_notebook.pkl",  # kaggle v2
]




## === cell 3
CANDIDATE_COMP_ROOTS = [
    Path("../input/hotel-id-2021-fgvc8"),
    Path("/kaggle/input/hotel-id-2021-fgvc8"),
    Path("../input/hotel-id-2021-fgvc8/hotel-id-2021-fgvc8"),
    Path("/kaggle/input/hotel-id-2021-fgvc8/hotel-id-2021-fgvc8"),
]

comp_root = None
for p in CANDIDATE_COMP_ROOTS:
    if (p / "sample_submission.csv").exists():
        comp_root = p
        break

if comp_root is None:
    for base in [Path("../input"), Path("/kaggle/input")]:
        if base.exists():
            hits = list(base.rglob("sample_submission.csv"))
            for h in hits:
                if "hotel-id-2021-fgvc8" in str(h):
                    comp_root = h.parent
                    break
            if comp_root is not None:
                break

if comp_root is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv under ../input or /kaggle/input"
    )

image_dir = comp_root / "test_images"
if image_dir.is_dir():
    pass
elif (comp_root / "test_images" / "test_images").is_dir():
    image_dir = comp_root / "test_images" / "test_images"
else:
    candidates = []
    for base in [comp_root, Path("../input"), Path("/kaggle/input")]:
        if base.exists():
            candidates.extend([p for p in base.rglob("test_images") if p.is_dir()])
    picked = None
    for p in candidates:
        if len(list(p.glob("*.jpg"))) > 0:
            picked = p
            break
        if (p / "test_images").is_dir() and len(
            list((p / "test_images").glob("*.jpg"))
        ) > 0:
            picked = p / "test_images"
            break
    if picked is None:
        raise FileNotFoundError(
            "Could not locate test_images directory with .jpg files"
        )
    image_dir = picked

comp_root, image_dir




## === cell 4
submission = pd.read_csv(comp_root / "sample_submission.csv")

test = submission.copy()
test["image"] = (image_dir.as_posix().rstrip("/") + "/") + test["image"].astype(str)
test.head()




## === cell 5
existing_models = [str(Path(m)) for m in models if Path(m).exists()]

if len(existing_models) == 0:
    candidates = []
    for base in [Path("../input"), Path("/kaggle/input")]:
        if base.exists():
            candidates.extend(list(base.rglob("export*.pkl")))
    existing_models = [str(p) for p in candidates]

existing_models[:10], len(existing_models)




## === cell 6
train_csv = None
for p in [
    comp_root / "train.csv",
    Path("../input/hotel-id-2021-fgvc8/train.csv"),
    Path("/kaggle/input/hotel-id-2021-fgvc8/train.csv"),
    Path("../input/train.csv"),
    Path("/kaggle/input/train.csv"),
]:
    if p.exists():
        train_csv = p
        break

if train_csv is None:
    for base in [Path("../input"), Path("/kaggle/input")]:
        if base.exists():
            hits = list(base.rglob("train.csv"))
            for h in hits:
                if "hotel-id-2021-fgvc8" in str(h):
                    train_csv = h
                    break
            if train_csv is not None:
                break

if train_csv is None:
    raise FileNotFoundError("Could not locate train.csv needed for fallback baseline")

train_df = pd.read_csv(train_csv, usecols=["image", "chain", "hotel_id"])
train_df["chain"] = train_df["chain"].astype(np.int32, copy=False)
train_df["hotel_id"] = train_df["hotel_id"].astype(np.int32, copy=False)

global_top = train_df["hotel_id"].value_counts().head(200).index.astype(int).tolist()

chain_to_top = (
    train_df.groupby("chain")["hotel_id"]
    .apply(lambda s: s.value_counts().head(200).index.astype(int).tolist())
    .to_dict()
)
chain_to_top = {int(k): v for k, v in chain_to_top.items()}


def _build_test_image_index(test_images_root: Path) -> dict:
    idx = {}
    root = Path(test_images_root)
    if not root.exists():
        return idx
    for p in root.rglob("*.jpg"):
        idx[p.name] = p
    return idx


test_img_index = _build_test_image_index(image_dir)


def _resolve_test_image_path(image_name: str) -> Path:
    p = test_img_index.get(image_name)
    if p is not None:
        return p
    q = Path(image_dir) / image_name
    return q


def _build_img_to_chain_from_folders(
    test_images_root: Path, comp_root: Path, max_chains_to_scan: int = None
):
    img2ch = {}

    candidates = []
    for r in [
        test_images_root,
        comp_root / "test_images",
        comp_root / "test_images" / "test_images",
    ]:
        r = Path(r)
        if r.exists() and r.is_dir():
            candidates.append(r)

    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)

    def scan_root(root: Path):
        try:
            chain_dirs = [d for d in root.iterdir() if d.is_dir() and d.name.isdigit()]
        except Exception:
            chain_dirs = []
        if max_chains_to_scan is not None:
            chain_dirs = chain_dirs[:max_chains_to_scan]
        for d in chain_dirs:
            ch = int(d.name)
            for imgp in d.glob("*.jpg"):
                img2ch[imgp.name] = ch

        nested = root / "test_images"
        if nested.is_dir():
            try:
                chain_dirs2 = [
                    d for d in nested.iterdir() if d.is_dir() and d.name.isdigit()
                ]
            except Exception:
                chain_dirs2 = []
            if max_chains_to_scan is not None:
                chain_dirs2 = chain_dirs2[:max_chains_to_scan]
            for d in chain_dirs2:
                ch = int(d.name)
                for imgp in d.glob("*.jpg"):
                    img2ch[imgp.name] = ch

    for root in uniq:
        scan_root(root)

    return img2ch


def _parse_chain_from_path(p: str):
    try:
        parts = Path(p).parts
        for i in range(len(parts) - 2, -1, -1):
            if parts[i] == "test_images":
                cand = parts[i + 1] if i + 1 < len(parts) else None
                if cand is not None and str(cand).isdigit():
                    return int(cand)
                break
        parent = Path(p).parent.name
        if parent.isdigit():
            return int(parent)
    except Exception:
        pass
    return None


def _find_train_images_root(comp_root: Path) -> Path:
    candidates = [
        comp_root / "train_images",
        comp_root / "train_images" / "train_images",
        Path("../input/hotel-id-2021-fgvc8/train_images"),
        Path("/kaggle/input/hotel-id-2021-fgvc8/train_images"),
        Path("../input/hotel-id-2021-fgvc8/train_images/train_images"),
        Path("/kaggle/input/hotel-id-2021-fgvc8/train_images/train_images"),
    ]
    for c in candidates:
        if c.is_dir():
            return c
    for hit in comp_root.rglob("train_images"):
        if hit.is_dir():
            return hit
    return None


def _build_img_to_chain_from_train_images(train_images_root: Path) -> dict:
    img2ch = {}
    if train_images_root is None or not Path(train_images_root).is_dir():
        return img2ch
    try:
        for d in Path(train_images_root).iterdir():
            if d.is_dir() and d.name.isdigit():
                ch = int(d.name)
                for imgp in d.glob("*.jpg"):
                    img2ch[imgp.name] = ch
    except Exception:
        pass
    return img2ch


test_csv = None
for p in [
    comp_root / "test.csv",
    Path("../input/hotel-id-2021-fgvc8/test.csv"),
    Path("/kaggle/input/hotel-id-2021-fgvc8/test.csv"),
    Path("../input/test.csv"),
    Path("/kaggle/input/test.csv"),
]:
    if p.exists():
        test_csv = p
        break

if test_csv is None:
    for base in [Path("../input"), Path("/kaggle/input")]:
        if base.exists():
            hits = list(base.rglob("test.csv"))
            for h in hits:
                if "hotel-id-2021-fgvc8" in str(h):
                    test_csv = h
                    break
            if test_csv is not None:
                break

img_to_chain = None
if test_csv is not None:
    test_meta = pd.read_csv(test_csv)
    if "image" in test_meta.columns and "chain" in test_meta.columns:
        tmp = test_meta[["image", "chain"]].copy()
        tmp["chain"] = tmp["chain"].astype(np.int32, copy=False)
        img_to_chain = dict(zip(tmp["image"].astype(str), tmp["chain"].astype(int)))

folder_img_to_chain = {}
train_img_to_chain = {}
if img_to_chain is None:
    folder_img_to_chain = _build_img_to_chain_from_folders(image_dir, comp_root)
    train_images_root = _find_train_images_root(comp_root)
    train_img_to_chain = _build_img_to_chain_from_train_images(train_images_root)

test_images_base = submission["image"].astype(str).tolist()
test_chains = []
for img in test_images_base:
    ch = None
    if img_to_chain is not None:
        ch = img_to_chain.get(img, None)
    if ch is None and folder_img_to_chain:
        ch = folder_img_to_chain.get(img, None)
    if ch is None and train_img_to_chain:
        ch = train_img_to_chain.get(img, None)
    if ch is None:
        rp = _resolve_test_image_path(img)
        ch = _parse_chain_from_path(str(rp))
    test_chains.append(ch)

len(chain_to_top), len(global_top), test_chains[:10], (
    sum([c is None for c in test_chains]),
    len(test_chains),
)




## === cell 7
probs = None
learn = None  # keep last learner to access vocab later
test_dl = None

if len(existing_models) > 0:
    with torch.inference_mode():
        for mi, model in enumerate(existing_models):
            learn = load_learner(fname=Path(model), cpu=False, pickle_module=dill)

            if test_dl is None:
                test_dl = learn.dls.test_dl(test)

            probs_temp, _ = learn.tta(dl=test_dl, n=5)

            if probs is None:
                probs = probs_temp
            else:
                probs += probs_temp

probs is None, learn is None, len(existing_models)




## === cell 8
from PIL import Image

HASH_CACHE_DIR = Path("/kaggle/working/hash_cache")
HASH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_HASH_CACHE = HASH_CACHE_DIR / "train_dhash64.parquet"
TEST_HASH_CACHE = HASH_CACHE_DIR / "test_dhash64.parquet"


def _dhash64_from_path(img_path: Path, hash_size: int = 8):
    try:
        with Image.open(img_path) as im:
            im = im.convert("L").resize(
                (hash_size + 1, hash_size), Image.Resampling.BILINEAR
            )
            arr = np.asarray(im, dtype=np.uint8)
        diff = (arr[:, 1:] > arr[:, :-1]).astype(np.uint8).reshape(-1)  # 64 bits
        packed = np.packbits(diff, bitorder="big")  # 8 bytes, big-endian bits
        h = np.frombuffer(packed.tobytes(), dtype=">u8")[0].astype(np.uint64)
        return h
    except Exception:
        return None


def _hamming_u64(a: np.uint64, b: np.uint64) -> int:
    return int((int(a) ^ int(b)).bit_count())


def _hash_paths_parallel(paths, workers: int):
    from concurrent.futures import ProcessPoolExecutor

    def _job(p):
        h = _dhash64_from_path(Path(p))
        return (str(p), None if h is None else int(h))

    if workers <= 1:
        return [_job(p) for p in paths]

    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_job, paths, chunksize=256))


def _predict_hotels_by_hash_nn(
    comp_root: Path,
    submission_images: list,
    train_df: pd.DataFrame,
    global_top: list,
    chain_to_top: dict,
    max_train_per_chain: int = 30,
    max_hotels_per_chain: int = 200,
):
    train_images_root = _find_train_images_root(comp_root)
    if train_images_root is None or not Path(train_images_root).is_dir():
        return {}

    tir = Path(train_images_root)

    allowed_pairs = []
    for ch, hids in chain_to_top.items():
        for hid in hids[:max_hotels_per_chain]:
            allowed_pairs.append((int(ch), int(hid)))
    allowed_df = pd.DataFrame(allowed_pairs, columns=["chain", "hotel_id"]).astype(
        {"chain": np.int32, "hotel_id": np.int32}
    )

    tmp = train_df[["image", "chain", "hotel_id"]].copy()
    tmp["chain"] = tmp["chain"].astype(np.int32, copy=False)
    tmp["hotel_id"] = tmp["hotel_id"].astype(np.int32, copy=False)

    tmp = tmp.merge(allowed_df, on=["chain", "hotel_id"], how="inner", copy=False)

    train_df_small = (
        tmp.sort_values(["chain", "hotel_id", "image"], kind="mergesort")
        .groupby(["chain", "hotel_id"], sort=True)
        .head(max_train_per_chain)
        .reset_index(drop=True)
    )

    proto_paths = []
    proto_chain = []
    proto_hotel = []
    for row in train_df_small.itertuples(index=False):
        p = tir / str(int(row.chain)) / str(row.image)
        if p.exists():
            proto_paths.append(p)
            proto_chain.append(int(row.chain))
            proto_hotel.append(int(row.hotel_id))

    if len(proto_paths) == 0:
        return {}

    if TRAIN_HASH_CACHE.exists():
        train_hash_df = pd.read_parquet(TRAIN_HASH_CACHE)
        cached = dict(
            zip(
                train_hash_df["path"].astype(str), train_hash_df["hash"].astype("Int64")
            )
        )
    else:
        cached = {}

    need = [p for p in proto_paths if str(p) not in cached]
    if len(need) > 0:
        workers = min(8, max(1, os.cpu_count() // 2))
        new_items = _hash_paths_parallel([str(p) for p in need], workers=workers)
        for sp, hv in new_items:
            cached[sp] = hv
        pd.DataFrame(
            {"path": list(cached.keys()), "hash": list(cached.values())}
        ).to_parquet(TRAIN_HASH_CACHE, index=False)

    proto_hash = []
    keep_chain = []
    keep_hotel = []
    keep_paths = []
    for p, ch, hid in zip(proto_paths, proto_chain, proto_hotel):
        hv = cached.get(str(p))
        if hv is not None and not (isinstance(hv, float) and np.isnan(hv)):
            proto_hash.append(np.uint64(int(hv)))
            keep_chain.append(ch)
            keep_hotel.append(hid)
            keep_paths.append(p)

    if len(proto_hash) == 0:
        return {}

    proto_hash_arr = np.array(proto_hash, dtype=np.uint64)
    proto_bytes = proto_hash_arr.view(np.uint8).reshape(-1, 8)
    proto_bits = np.unpackbits(proto_bytes, axis=1, bitorder="big")  # (P,64) uint8
    proto_bits = proto_bits.astype(np.uint8, copy=False)

    test_paths = [str(_resolve_test_image_path(img)) for img in submission_images]
    if TEST_HASH_CACHE.exists():
        test_hash_df = pd.read_parquet(TEST_HASH_CACHE)
        cached_t = dict(
            zip(test_hash_df["path"].astype(str), test_hash_df["hash"].astype("Int64"))
        )
    else:
        cached_t = {}

    need_t = [p for p in test_paths if p not in cached_t and Path(p).exists()]
    if len(need_t) > 0:
        workers = min(8, max(1, os.cpu_count() // 2))
        new_items = _hash_paths_parallel(need_t, workers=workers)
        for sp, hv in new_items:
            cached_t[sp] = hv
        pd.DataFrame(
            {"path": list(cached_t.keys()), "hash": list(cached_t.values())}
        ).to_parquet(TEST_HASH_CACHE, index=False)

    out = {}
    for img, tp in zip(submission_images, test_paths):
        if not Path(tp).exists():
            continue
        hv = cached_t.get(tp)
        if hv is None or (isinstance(hv, float) and np.isnan(hv)):
            continue

        th = np.uint64(int(hv))
        th_bits = np.unpackbits(
            np.array([th], dtype=np.uint64).view(np.uint8), bitorder="big"
        ).astype(np.uint8, copy=False)

        dists = (
            np.bitwise_xor(proto_bits, th_bits).sum(axis=1).astype(np.int16, copy=False)
        )

        order = np.argsort(dists, kind="stable")
        picks = []
        for idx in order:
            hid = keep_hotel[int(idx)]
            if hid not in picks:
                picks.append(hid)
            if len(picks) >= 5:
                break

        if len(picks) < 5:
            ch0 = keep_chain[int(order[0])] if len(order) else None
            if ch0 is not None:
                for hid in chain_to_top.get(int(ch0), []):
                    if hid not in picks:
                        picks.append(int(hid))
                    if len(picks) == 5:
                        break
        if len(picks) < 5:
            for hid in global_top:
                if hid not in picks:
                    picks.append(int(hid))
                if len(picks) == 5:
                    break

        out[img] = " ".join(map(str, picks))
    return out


hash_nn_preds = {}
if probs is None:
    hash_nn_preds = _predict_hotels_by_hash_nn(
        comp_root=comp_root,
        submission_images=test_images_base,
        train_df=train_df,
        global_top=global_top,
        chain_to_top=chain_to_top,
        max_train_per_chain=30,
        max_hotels_per_chain=200,
    )

len(hash_nn_preds)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
AttributeError: Can't pickle local object '_hash_paths_parallel.<locals>._job'
"""

The above exception was the direct cause of the following exception:

AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3194391898.py in <cell line: 0>()
    205 hash_nn_preds = {}
    206 if probs is None:
--> 207     hash_nn_preds = _predict_hotels_by_hash_nn(
    208         comp_root=comp_root,
    209         submission_images=test_images_base,

/tmp/ipykernel_55/3194391898.py in _predict_hotels_by_hash_nn(comp_root, submission_images, train_df, global_top, chain_to_top, max_train_per_chain, max_hotels_per_chain)
    106     if len(need) > 0:
    107         workers = min(8, max(1, os.cpu_count() // 2))
--> 108         new_items = _hash_paths_parallel([str(p) for p in need], workers=workers)
    109         for sp, hv in new_items:
    110             cached[sp] = hv

/tmp/ipykernel_55/3194391898.py in _hash_paths_parallel(paths, workers)
     38 
     39     with ProcessPoolExecutor(max_workers=workers) as ex:
---> 40         return list(ex.map(_job, paths, chunksize=256))
     41 
     42 

/usr/lib/python3.11/concurrent/futures/process.py in _chain_from_iterable_of_lists(iterable)
    618     careful not to keep references to yielded objects.
    619     """
--> 620     for element in iterable:
    621         element.reverse()
    622         while element:

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    447                     raise CancelledError()
    448                 elif self._state == FINISHED:
--> 449                     return self.__get_result()
    450 
    451                 self._condition.wait(timeout)

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/multiprocessing/queues.py in _feed(buffer, notempty, send_bytes, writelock, reader_close, writer_close, ignore_epipe, onerror, queue_sem)
    242 
    243                         # serialize the data before acquiring the lock
--> 244                         obj = _ForkingPickler.dumps(obj)
    245                         if wacquire is None:
    246                             send_bytes(obj)

/usr/lib/python3.11/multiprocessing/reduction.py in dumps(cls, obj, protocol)
     49     def dumps(cls, obj, protocol=None):
     50         buf = io.BytesIO()
---> 51         cls(buf, protocol).dump(obj)
     52         return buf.getbuffer()
     53 

AttributeError: Can't pickle local object '_hash_paths_parallel.<locals>._job'

## === cell 9
if probs is not None and learn is not None:
    preds_idx = probs.topk(5)[1]
    vocab = learn.dls.vocab
    preds = [" ".join(map(str, vocab[pred])) for pred in preds_idx]
else:
    preds = []
    for img, ch in zip(test_images_base, test_chains):
        hn = hash_nn_preds.get(img, None)
        if hn is not None:
            preds.append(hn)
            continue

        base = chain_to_top.get(int(ch), []) if ch is not None else []
        picks = []
        for hid in base:
            if hid not in picks:
                picks.append(hid)
            if len(picks) == 5:
                break
        if len(picks) < 5:
            for hid in global_top:
                if hid not in picks:
                    picks.append(hid)
                if len(picks) == 5:
                    break
        preds.append(" ".join(map(str, picks)))

preds[:3], len(preds), len(submission)




## === cell 10
submission_out = submission.copy()
submission_out["hotel_id"] = preds

assert len(submission_out) == len(submission)
assert (submission_out["image"].values == submission["image"].values).all()

submission_out.to_csv("submission.csv", index=False)
submission_out.head()
