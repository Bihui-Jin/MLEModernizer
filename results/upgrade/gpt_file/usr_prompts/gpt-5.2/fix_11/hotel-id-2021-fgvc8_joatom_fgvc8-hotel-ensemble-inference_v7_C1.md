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

0.612006398922285

# 6. Current score

0.00209

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'The failure comes from referencing external model `.pkl` paths that don’t exist in this Kaggle environment, which prevents `probs` from being created and causes the downstream `topk`/submission steps to crash. I keep the same fastai inference core logic (load learner → test_dl → TTA → ensemble → top-5) but make it robust by discovering available `.pkl` models under `/kaggle/input` and using only those that actually exist. If no models are found, the notebook still produce a valid submission by falling back to a deterministic “most frequent hotels” baseline from `train.csv` (score be low but it run end-to-end). I also fix the test dataframe to pass file paths via the proper `image` column and ensure submission length/order matches `sample_submission.csv`.'
- What this solution (achieved 0.00209) has done: 'I keep your inference/ensemble logic intact, but make two minimal fixes that materially improve MAP@5: (1) don’t overwrite `test["image"]` with full paths (fastai `test_dl` expects filenames/IDs consistent with the exported `DataLoaders`), and (2) make the test DataFrame column match the learner’s expected input column (`learn.dls.xcol`) when possible. This prevents the common failure mode where the model loads but predicts garbage because it can’t actually find/open the images, which is consistent with your very low score. The fallback “top frequent hotels” stays as-is for robustness, and the output submission remains aligned to `sample_submission.csv`.'
- What this solution (achieved 0.00209) has done: 'The very low MAP@5 is consistent with your inference running on the wrong images (your `test_dl` is built from bare filenames while the exported `Learner` likely expects a specific image root), so the model may be predicting on missing/blank inputs or failing silently. I keep your exact fastai flow (load_learner → test_dl → TTA → sum ensemble → topk5) but make one minimal, targeted fix: dynamically detect the correct test image folder and set `learn.dls.test_dl(..., rm_type_tfms=None)` with the test dataframe column the learner expects, while ensuring the `image` column stays the submission ID (basename) for correct CSV alignment. I also make the vocab-to-string conversion robust (fastai vocab items are often already strings) so the top-5 list is always valid. The fallback “most frequent hotels” baseline remains unchanged for robustness if no models load.'
- What this solution (achieved 0.00209) has done: 'Your current score is extremely low because inference is likely using a mismatched input setup: the exported `Learner`’s `DataLoaders` already encode how to open images, and by feeding full paths into the x-column you can easily break `PILImage.create` / `get_image_files` expectations, producing near-random predictions. I keep your exact fastai flow (load_learner → test_dl → TTA → sum-ensemble → top-5) but make one targeted fix: set the learner’s test dataloader root (`learn.dls.path`) to the resolved `test_images` folder and feed only basenames/IDs in the x-column. I also make the vocab mapping robust for common fastai classifier exports (`dls.vocab` vs `dls.vocab[-1]`) so the top-5 IDs correspond to the correct class list. These are minimal changes that should move MAP@5 substantially upward toward your target without changing the model or training logic, and the fallback submission remains for robustness.'
- What this solution (achieved 0.00209) has done: 'The main reason your MAP@5 is still extremely low is that inference is likely being run on the wrong image root and/or the learner’s test dataloader is not pointing at the actual `test_images` folder (fastai exports often encode a `dls.path` from a different environment). I keep your exact fastai inference core (load_learner → test_dl → TTA → sum ensemble → top-5) but make one targeted fix: set `learn.dls.path` to the resolved dataset root (so relative image opens work) and set `learn.dls.after_item`’s `PILImage` opener base path implicitly via `dls.path`, while feeding only basenames in the x-column. I also ensure we use the same ordering as `sample_submission.csv` and add a very small robustness fix for vocab extraction so the top-5 IDs always map correctly. These changes are minimal, preserve your logic, and should materially increase score toward your target by making the model actually see the correct test images.'
- What this solution (achieved 0.00209) has done: 'The current score is far below target, which is consistent with your inference running against the wrong image root and/or the wrong x-column for the exported fastai `Learner`, so the model effectively predicts on missing/incorrect inputs. I keep your exact fastai inference flow (load_learner → test_dl → TTA → sum-ensemble → top-5) but make a minimal, targeted change: ensure the test dataloader uses the learner’s expected x-column and that this column contains paths that actually exist under the resolved `test_images` directory, while keeping the submission `image` column as basenames for correct CSV alignment. I also robustly extract the label vocab from either `learn.dls.vocab` or `learn.dls.y`/`dls.c` patterns so indices map to correct hotel IDs. These changes should materially increase MAP@5 toward your target without changing model architecture, training, or loss, and still always produce a valid `submission.csv`.'
- What this solution (achieved 0.00209) has done: 'The current score is far below target, so we should make the smallest fix that makes the existing fastai models actually read the correct test images and map indices to the correct hotel_id vocab. I keep your exact inference flow (load_learner → test_dl → TTA → sum ensemble → top-5), but change the test dataloader inputs to use basenames with `dls.path` set to the true dataset root (instead of forcing full paths into the x-column), which commonly causes near-random predictions. I also align the submission row order to `sample_submission.csv` explicitly and make vocab extraction slightly stricter (prefer `dls.vocab[-1]` when it looks like a class list) so indices map correctly. The fallback “most frequent hotels” remains unchanged for robustness if no `.pkl` loads.'
- What this solution (achieved 0.00209) has done: 'I make the smallest changes that most plausibly move MAP@5 up by ensuring inference actually uses the correct image paths and the correct class-vocab mapping from the exported fastai learners. Specifically, I (1) build the test dataloader from a dataframe whose x-column points to *existing files* under the resolved `test_images_dir` (instead of relying on `dls.path` semantics that often mismatch after export), (2) avoid `learn.tta(...)` (which commonly applies training-time augmentation/cropping at inference and can hurt retrieval MAP@5) and use standard `learn.get_preds(...)` on the test_dl, and (3) average probabilities across models (currently you sum but never divide), which stabilizes calibration and can improve top-5 ordering. The architecture/training/loss are unchanged; this is purely an inference data/aggregation fix and still falls back to a valid frequency baseline if no models load.'
- What this solution (achieved 0.00209) has done: 'I make the smallest inference-only changes that are most likely to move MAP@5 up from ~0.002 by ensuring your exported fastai learners actually open the correct test images and that class-index→hotel_id mapping is correct. Concretely: (1) stop forcing full absolute paths into the x-column and instead set `learn.dls.path` to the real dataset root while feeding basenames (fastai exports commonly break when given unexpected path formats), (2) build the test dataloader in the exact order of `sample_submission.csv` and verify files exist to avoid silent misreads, and (3) make vocab extraction stricter (prefer `dls.vocab[-1]` when it matches `learn.dls.c`) so top-5 indices map to the right hotel IDs. The ensemble averaging and fallback baseline remain, and the notebook still always writes a valid `submission.csv`.'
- What this solution (achieved 0.00209) has done: 'The current score is far below the target, so the smallest likely win is to make inference actually read the correct test images and keep the test row order identical to `sample_submission.csv`. I keep your same fastai flow (load_learner → test_dl → get_preds → sum/avg ensemble → topk5), but change the test dataframe fed into `test_dl` so its x-column contains full existing file paths (robust across exported learners that were trained with absolute/relative paths) while preserving the submission `image` column as basenames. I also (minimally) ensure the `test_dl` uses `num_workers=0` for stability in Kaggle and tighten vocab extraction to prefer `learn.dls.vocab[-1]` and validate indices. The fallback most-frequent-hotels submission remains unchanged so you always get a valid `submission.csv`.'

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
    "../input/fgvc8hotel/export_dn161_Fa_CE_bs32.pkl",  # v7 (may not exist here)
    "../input/fgvc8hotel/export_dn161_Fa_FL_bs32.pkl",  # v8 (may not exist here)
    "../input/fgvc8hotel/export_res101_Fall_5it_4.pkl",  # v11 (may not exist here)
    "../input/hotel-train-fastai-densnet161/export_dn161_kaggle_notebook.pkl",  # kaggle v2 (may not exist here)
]



## === cell 3
image_path = "../input/hotel-id-2021-fgvc8/test_images/"



## === cell 4
submission = pd.read_csv("../input/hotel-id-2021-fgvc8/sample_submission.csv")

test = submission.copy()
test["image"] = test["image"].astype(str).map(lambda x: Path(x).name)

test.head()




## === cell 5
def _existing_model_paths(requested_models):
    existing = []
    for m in requested_models:
        p = Path(m)
        if p.exists():
            existing.append(p)
    return existing


def _discover_pkl_models(root="/kaggle/input", limit=20):
    root = Path(root)
    if not root.exists():
        return []
    pkls = sorted(root.rglob("*.pkl"))
    export_pkls = [p for p in pkls if p.name.lower().startswith("export")]
    ordered = export_pkls if export_pkls else pkls
    return ordered[:limit]


def _resolve_test_images_dir():
    candidates = [
        Path("../input/hotel-id-2021-fgvc8/test_images"),
        Path("/kaggle/input/hotel-id-2021-fgvc8/test_images"),
        Path("../input/test_images"),
        Path("/kaggle/input/test_images"),
    ]
    for c in candidates:
        if c.exists():
            return c
    root = Path("/kaggle/input")
    if root.exists():
        for p in root.rglob("test_images"):
            if p.is_dir():
                return p
    return None


def _resolve_dataset_root_from_test_images(test_images_dir: Path):
    if test_images_dir is None:
        return None
    if test_images_dir.name == "test_images" and test_images_dir.parent.exists():
        return test_images_dir.parent
    return test_images_dir


existing_models = _existing_model_paths(models)
if len(existing_models) == 0:
    existing_models = _discover_pkl_models(limit=20)

existing_models = [Path(p) for p in existing_models]
print(f"Using {len(existing_models)} model(s).")
for p in existing_models[:10]:
    print(" -", p)

test_images_dir = _resolve_test_images_dir()
dataset_root = _resolve_dataset_root_from_test_images(test_images_dir)
print("Resolved test_images_dir:", test_images_dir)
print("Resolved dataset_root:", dataset_root)




## === cell 6
def _infer_xcol(learn):
    dls = getattr(learn, "dls", None)
    xcol = getattr(dls, "xcol", None)
    return xcol if xcol is not None else "image"


def _build_test_items_df(test_df, xcol, test_images_dir: Path):
    """
    Score-critical fix (minimal change to improve correctness):
    Many exported fastai learners expect the x-column to be a path (relative or absolute) that PILImage can open.
    Feeding only basenames can silently fail depending on how the export encoded its getters/path.
    So we pass full *existing* file paths in the learner's xcol, while keeping submission 'image' as basenames.
    """
    t = test_df.copy()
    t["image"] = t["image"].map(lambda fn: Path(fn).name)

    if test_images_dir is None:
        t[xcol] = t["image"]
        return t

    t[xcol] = t["image"].map(lambda fn: str((test_images_dir / Path(fn).name)))

    if len(t) > 0:
        probe = Path(t[xcol].iloc[0])
        if not probe.exists():
            print(f"Warning: first test image not found at expected path: {probe}")
    return t


probs = None
learn = None  # keep last loaded learner for vocab mapping
n_models_used = 0

if len(existing_models) > 0:
    for model_path in existing_models:
        try:
            learn = load_learner(fname=model_path, cpu=False, pickle_module=dill)

            if (
                dataset_root is not None
                and hasattr(learn, "dls")
                and hasattr(learn.dls, "path")
            ):
                learn.dls.path = Path(dataset_root)

            xcol = _infer_xcol(learn)
            test_for_dl = _build_test_items_df(
                test, xcol=xcol, test_images_dir=test_images_dir
            )

            test_dl = learn.dls.test_dl(
                test_for_dl, with_labels=False, rm_type_tfms=None, num_workers=0
            )

            probs_temp, _ = learn.get_preds(dl=test_dl, act=nn.Softmax(dim=1))
            probs = probs_temp if probs is None else (probs + probs_temp)
            n_models_used += 1
        except Exception as e:
            print(
                f"Skipping model (failed to load/infer): {model_path}\n  Reason: {type(e).__name__}: {e}"
            )

if probs is not None and n_models_used > 1:
    probs = probs / float(n_models_used)

print("n_models_used:", n_models_used)
print("probs:", None if probs is None else tuple(probs.shape))




## === cell 7
def _vocab_item_to_str(v):
    try:
        return str(v)
    except Exception:
        return f"{v}"


def _get_class_vocab(learn):
    """
    Score-critical: robust class vocab extraction so prob indices map to correct hotel_id strings.
    Prefer dls.vocab[-1] when it looks like the class list (length == dls.c).
    """
    dls = getattr(learn, "dls", None)
    if dls is None:
        return []

    c = getattr(dls, "c", None)
    vocab = getattr(dls, "vocab", None)

    if vocab is not None:
        if isinstance(vocab, (list, tuple)) and len(vocab) >= 1:
            try:
                y_list = list(vocab[-1])
                if len(y_list) > 0 and (c is None or len(y_list) == int(c)):
                    return y_list
            except Exception:
                pass

        try:
            v_list = list(vocab)
            if len(v_list) > 0 and (c is None or len(v_list) == int(c)):
                return v_list
        except Exception:
            pass

    try:
        v_list = list(dls.train_ds.vocab)
        return v_list
    except Exception:
        return []


if probs is not None and learn is not None:
    preds_idx = probs.topk(5)[1].cpu().numpy()
    vocab = _get_class_vocab(learn)

    if len(vocab) == 0:
        preds = ["0 0 0 0 0"] * len(submission)
    else:
        vlen = len(vocab)
        preds = []
        for row in preds_idx:
            row_ids = []
            for i in row:
                ii = int(i)
                if 0 <= ii < vlen:
                    row_ids.append(_vocab_item_to_str(vocab[ii]))
                else:
                    row_ids.append("0")
            preds.append(" ".join(row_ids))
else:
    train_csv_path = Path("../input/hotel-id-2021-fgvc8/train.csv")
    if train_csv_path.exists():
        train_df = pd.read_csv(train_csv_path)
        top_hotels = (
            train_df["hotel_id"].value_counts().head(5).index.astype(str).tolist()
        )
    else:
        top_hotels = ["0", "0", "0", "0", "0"]
    fallback = " ".join(top_hotels)
    preds = [fallback] * len(submission)

len(preds), preds[0]



## === cell 8
submission_out = submission.copy()
submission_out["hotel_id"] = preds

assert len(submission_out) == len(submission)
assert list(submission_out.columns) == ["image", "hotel_id"]

submission_out.to_csv("submission.csv", index=False)
submission_out.head()
