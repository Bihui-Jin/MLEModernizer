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

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'The pipeline fails because it expects exported fastai model `.pkl` files that are not present in your `/kaggle/input` datasets, so inference never runs and no submission is written. I fix this by making the model discovery robust (searching all input roots you actually have) and adding a safe fallback baseline that still produces a valid MAP@5 submission when no `.pkl` models exist. The fallback uses the training metadata to predict the 5 most frequent `hotel_id`s (a common baseline) so you get a non-zero score and a valid `submission.csv`. I also fix the cell numbering to start at 1 and keep paths compatible with the Kaggle file structure you listed.'
- What this solution (achieved 0.00209) has done: 'Your score is far below target (0.00209 vs 0.6126), so we should improve (not degrade). The biggest issue is that your current fallback predicts the same global top-5 hotels for every image, which yields an extremely low MAP@5; we can keep the same overall inference structure but make the fallback “metadata-driven” using the provided `chain` folders in `test_images/` and the `chain` column in `train.csv`. Concretely: for each test image, infer its `chain` from the parent folder name, then predict the top-5 most frequent `hotel_id`s within that chain (and backfill with global top hotels if a chain is missing/unknown). This preserves evaluation semantics, avoids any label leakage, runs fast, and should move the score substantially toward the target even with no `.pkl` models.'
- What this solution (achieved 0.00209) has done: 'Your current score is far below the target, so we should improve (not degrade) with minimal, low-risk changes. The biggest lift while preserving your overall structure is to make the chain-based fallback actually work with this dataset: most `test_images` are in a flat folder (no chain subfolders), so your `_parse_chain_from_path` returns `None` and you effectively predict global top-5 for every image. I keep your fastai inference path unchanged, but improve the fallback by extracting `chain` from the **test image filename** using the provided `test.csv` mapping (this is legitimate metadata, not labels) and then using per-chain top-5 hotel IDs, with a global backfill. This should substantially increase MAP@5 while still running fast and producing a valid `submission.csv`.'
- What this solution (achieved 0.00209) has done: 'Your current score is far below target, so the smallest safe way to move toward it is to make the fallback much more informative without changing the fastai inference core. The key bug is that `test.csv` with chain metadata typically isn’t present, so `test_chains` becomes mostly `None` and you effectively submit the same global top-5 for every image (very low MAP@5). I keep your model-loading/tta path unchanged, but improve the fallback by (1) correctly extracting `chain` from the **folder structure** (handling both flat and `test_images/<chain>/...` layouts) and (2) using per-chain top-50 hotel frequency lists with a robust global backfill. This should materially improve MAP@5 while still running fast and producing a valid `submission.csv`.'
- What this solution (achieved 0.00209) has done: 'We keep your fastai inference path untouched, but make the fallback baseline actually use chain information reliably so it stops predicting the same global top-5 for almost every test image (the main reason MAP@5 is stuck near zero). Concretely, we (1) detect whether the provided `test_images` folder is flat and, if so, also look for a `test_images/<chain>/...` variant and build an `image -> chain` map by scanning directories, and (2) use per-chain top hotel lists as you intended, with a global backfill. This is a minimal change that preserves evaluation semantics, avoids leakage (only uses folder structure + train metadata), runs fast, and should move the score substantially toward the 0.61 target. We also ensure the submission ordering exactly matches `sample_submission.csv`.'
- What this solution (achieved 0.00209) has done: 'Your current score (0.00209) is far below the target (0.6126), so we should improve the fallback path (used when no `.pkl` models are found) with the smallest, safest change. The main reason the fallback is near-random is that most `test_images` are stored flat (no chain subfolders), so your `test_chains` are mostly `None` and you end up predicting the same global top-5 for nearly every image. I keep your fastai inference logic unchanged, but I make chain extraction robust by first scanning for a `test_images/<chain>/*.jpg` layout anywhere under the competition root (and common parents), then using that mapping; this directly increases MAP@5 without changing evaluation semantics. Finally, I ensure the submission stays aligned to `sample_submission.csv` ordering and always writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

import pandas as pd
import numpy as np
import fastai
from fastai.vision.all import *
import dill



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

train_df = pd.read_csv(train_csv, usecols=["chain", "hotel_id"])
train_df["chain"] = train_df["chain"].astype(int)

global_top = train_df["hotel_id"].value_counts().head(50).index.astype(int).tolist()

chain_to_top = {}
for ch, g in train_df.groupby("chain"):
    top = g["hotel_id"].value_counts().head(50).index.astype(int).tolist()
    chain_to_top[int(ch)] = top


def _build_img_to_chain_from_folders(test_images_root: Path, comp_root: Path):
    img2ch = {}

    candidates = []
    for r in [
        test_images_root,
        comp_root / "test_images",
        comp_root / "test_images" / "test_images",
        test_images_root.parent,
        test_images_root.parent / "test_images",
        comp_root,
    ]:
        if r is not None and Path(r).exists():
            candidates.append(Path(r))

    for base in [comp_root, comp_root.parent]:
        if base.exists():
            for p in base.glob("**/test_images"):
                if p.is_dir():
                    candidates.append(p)

    seen = set()
    for root in candidates:
        if root in seen:
            continue
        seen.add(root)
        if not root.is_dir():
            continue

        if root.name.isdigit():
            ch = int(root.name)
            for imgp in root.glob("*.jpg"):
                img2ch[imgp.name] = ch

        try:
            for d in root.iterdir():
                if d.is_dir() and d.name.isdigit():
                    ch = int(d.name)
                    for imgp in d.glob("*.jpg"):
                        img2ch[imgp.name] = ch
        except Exception:
            pass

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
        tmp["chain"] = tmp["chain"].astype(int)
        img_to_chain = dict(zip(tmp["image"].astype(str), tmp["chain"].astype(int)))

folder_img_to_chain = _build_img_to_chain_from_folders(image_dir, comp_root)

test_images_base = submission["image"].astype(str).tolist()
test_chains = []
for img in test_images_base:
    ch = None
    if img_to_chain is not None:
        ch = img_to_chain.get(img, None)
    if ch is None:
        ch = folder_img_to_chain.get(img, None)
    if ch is None:
        full_path = (image_dir.as_posix().rstrip("/") + "/") + img
        ch = _parse_chain_from_path(full_path)
    test_chains.append(ch)

len(chain_to_top), len(global_top), test_chains[:10], (
    sum([c is None for c in test_chains]),
    len(test_chains),
)



## === cell 7
probs = None
learn = None  # keep last learner to access vocab later

if len(existing_models) > 0:
    for model in existing_models:
        learn = load_learner(fname=Path(model), cpu=False, pickle_module=dill)
        test_dl = learn.dls.test_dl(test)
        probs_temp, _ = learn.tta(dl=test_dl, n=5)

        if probs is None:
            probs = probs_temp
        else:
            probs += probs_temp

probs is None, learn is None, len(existing_models)



## === cell 8
if probs is not None and learn is not None:
    preds_idx = probs.topk(5)[1]
    vocab = learn.dls.vocab
    preds = [" ".join(map(str, vocab[pred])) for pred in preds_idx]
else:
    preds = []
    for ch in test_chains:
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



## === cell 9
submission_out = submission.copy()
submission_out["hotel_id"] = preds

assert len(submission_out) == len(submission)
assert (submission_out["image"].values == submission["image"].values).all()

submission_out.to_csv("submission.csv", index=False)
submission_out.head()
