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
- What this solution (achieved 0.00209) has done: 'Your score is far below the target, so the smallest likely win is to make sure inference is actually using the correct test images and (critically for MAP@5) the correct class index → hotel_id mapping. I keep your exact fastai inference flow (load_learner → test_dl → get_preds → avg ensemble → top-5), but make two minimal fixes: (1) build the test dataloader from a dedicated x-column containing valid full file paths while keeping the submission `image` column unchanged, and (2) extract the class vocab more robustly (including from `learn.dls.classes` / `dls.vocab` variants) so top-5 indices convert to the right hotel IDs. I also add a strict check that the constructed test paths exist (otherwise we fall back), because silently missing images can produce near-random predictions like your current 0.00209. The fallback “most frequent hotels” remains unchanged to guarantee a valid `submission.csv`.'
- What this solution (achieved 0.00209) has done: 'We keep your fastai inference/ensemble logic intact, but make two minimal score-critical fixes that commonly cause near-random MAP@5: ensure the learner’s test dataloader is fed the same kind of “x” that the exported `DataLoaders` expects (Path vs string) and avoid accidentally passing a dataframe with a mismatched x-column name/type. We also make the class-vocab extraction stricter by validating that the extracted vocab length matches the probability dimension (so indices always map to the correct `hotel_id`). Finally, we explicitly align the test dataframe row order to `sample_submission.csv` (already mostly done) and keep the fallback baseline unchanged for robustness so a valid `submission.csv` is always produced.'
- What this solution (achieved 0.00209) has done: 'Your current MAP@5 is far below target, so the smallest likely win is to ensure the exported fastai classifier is decoding indices into the *true hotel_id strings* (not an internal categorical code) and that the test dataloader uses the same label mapping the model was trained with. I keep your exact inference flow (load_learner → test_dl → get_preds → average ensemble → top-5), but add a score-critical vocab fix: prefer `learn.dls.vocab[-1]` and, if it looks non-numeric, fall back to `learn.dls.train_ds.vocab`/`Categorize.vocab`, and finally (only if needed) map predicted class names through the training dataframe’s `Categorical` ordering to recover actual `hotel_id`. I also make the test item construction respect the learner’s expected `xcol`, while ensuring that `image` in the submission stays as the basename and row order matches `sample_submission.csv`. These changes preserve your core logic and should move the score substantially upward by fixing a common “near-random predictions due to wrong class mapping” failure mode.'
- What this solution (achieved 0.00209) has done: 'Your current MAP@5 is so far below target that the most likely cause is still “valid-looking but wrong” inference: the learner is not decoding hotel IDs correctly and/or the test dataloader isn’t using the same item pipeline the exported learner expects. I keep your exact fastai inference flow (load learner → build test_dl → get_preds → average ensemble → top-5) but make two minimal score-critical fixes: (1) build the test dataloader using the exported learner’s own `test_dl` expectations (prefer `dls.test_dl(test_files)` when the x-type is Path-like), and (2) fix class decoding by using `learn.dls.vocab` (the most reliable mapping) and only falling back to train-derived categories if it matches the probability dimension. These changes don’t alter the model, loss, or training—only ensure the model sees the correct images and the indices map to the correct `hotel_id` strings—then still write a valid `submission.csv` with the required schema.'

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


def _infer_x_is_pathlike(learn):
    """
    Score-critical:
    Use the same type as the exported training pipeline (Path vs string).
    """
    try:
        dls = learn.dls
        if len(dls.train_ds) == 0:
            return False
        x = dls.train_ds[0][0]
        return isinstance(x, (Path, os.PathLike))
    except Exception:
        return False


def _all_or_most_exist(paths, min_frac=0.98, max_check=200):
    if len(paths) == 0:
        return False
    step = max(1, len(paths) // max_check)
    checked = 0
    ok = 0
    for i in range(0, len(paths), step):
        checked += 1
        p = Path(paths[i])
        if p.exists():
            ok += 1
        if checked >= max_check:
            break
    return (ok / checked) >= min_frac


def _build_test_files(test_df, test_images_dir: Path, as_path: bool):
    """
    Score-critical:
    Provide explicit existing files to the learner when it was trained with Path-based items.
    This avoids subtle dls.path / xcol mismatches after export.
    """
    fns = [Path(fn).name for fn in test_df["image"].tolist()]
    if test_images_dir is None:
        return [Path(fn) if as_path else str(fn) for fn in fns]
    files = [test_images_dir / fn for fn in fns]
    return files if as_path else [str(p) for p in files]


def _build_test_items_df(test_df, xcol, test_images_dir: Path, x_as_path: bool):
    """
    Keep submission 'image' as basename for correct CSV alignment, but feed xcol with valid files.
    """
    t = test_df.copy()
    t["image"] = t["image"].map(lambda fn: Path(fn).name)

    if test_images_dir is None:
        t[xcol] = t["image"]
        return t

    full_paths = [(test_images_dir / Path(fn).name) for fn in t["image"].tolist()]
    t[xcol] = full_paths if x_as_path else [str(p) for p in full_paths]
    return t


probs = None
learn = None
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

            x_as_path = _infer_x_is_pathlike(learn)
            xcol = _infer_xcol(learn)

            if x_as_path and test_images_dir is not None:
                test_files = _build_test_files(
                    test, test_images_dir=test_images_dir, as_path=True
                )
                if not _all_or_most_exist(test_files):
                    raise FileNotFoundError(
                        f"Constructed test file paths do not exist under: {test_images_dir}"
                    )
                test_dl = learn.dls.test_dl(
                    test_files, with_labels=False, rm_type_tfms=None, num_workers=0
                )
            else:
                test_for_dl = _build_test_items_df(
                    test,
                    xcol=xcol,
                    test_images_dir=test_images_dir,
                    x_as_path=x_as_path,
                )
                if test_images_dir is not None and not _all_or_most_exist(
                    test_for_dl[xcol].tolist()
                ):
                    raise FileNotFoundError(
                        f"Constructed test image paths in column '{xcol}' do not exist under: {test_images_dir}"
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


def _looks_like_int_strings(vocab, max_check=50):
    if not vocab:
        return False
    n = min(len(vocab), max_check)
    ok = 0
    for i in range(n):
        s = _vocab_item_to_str(vocab[i]).strip()
        if s.lstrip("-").isdigit():
            ok += 1
    return (ok / n) >= 0.95


def _get_class_vocab(learn, probs_n_classes=None):
    """
    Score-critical:
    Use the exported learner's vocab in the correct order. Prefer dls.vocab[-1] for classification.
    Only accept a vocab whose length matches probs_n_classes to avoid silent mis-decoding.
    """
    dls = getattr(learn, "dls", None)
    if dls is None:
        return []

    candidates = []

    vocab = getattr(dls, "vocab", None)
    if vocab is not None:
        if isinstance(vocab, (list, tuple)) and len(vocab) >= 1:
            try:
                candidates.append(list(vocab[-1]))
            except Exception:
                pass
        try:
            candidates.append(list(vocab))
        except Exception:
            pass

    classes = getattr(dls, "classes", None)
    if classes is not None:
        try:
            candidates.append(list(classes))
        except Exception:
            pass

    try:
        candidates.append(list(dls.train_ds.vocab))
    except Exception:
        pass

    if probs_n_classes is not None:
        exact = [
            c
            for c in candidates
            if isinstance(c, list) and len(c) == int(probs_n_classes)
        ]
        if len(exact) > 0:
            for cand in exact:
                if _looks_like_int_strings(cand):
                    return cand
            return exact[0]
        return []
    else:
        for cand in candidates:
            if isinstance(cand, list) and len(cand) > 0:
                return cand
        return []


def _maybe_map_vocab_to_true_hotel_ids(vocab, probs_n_classes):
    """
    If vocab isn't integer-like but matches probs dimension, try mapping via train.csv categorical order.
    This is only applied when lengths match to keep decoding consistent.
    """
    if vocab is None or len(vocab) != int(probs_n_classes):
        return vocab
    if _looks_like_int_strings(vocab):
        return vocab

    train_csv_path = Path("../input/hotel-id-2021-fgvc8/train.csv")
    if not train_csv_path.exists():
        return vocab

    train_df = pd.read_csv(train_csv_path, usecols=["hotel_id"])
    cats = pd.Categorical(train_df["hotel_id"].astype(str))
    true_classes = list(cats.categories.astype(str))

    if len(true_classes) == len(vocab):
        return true_classes

    return vocab


if probs is not None and learn is not None:
    preds_idx = probs.topk(5)[1].cpu().numpy()

    vocab = _get_class_vocab(learn, probs_n_classes=int(probs.shape[1]))
    vocab = _maybe_map_vocab_to_true_hotel_ids(
        vocab, probs_n_classes=int(probs.shape[1])
    )

    if not vocab or len(vocab) != int(probs.shape[1]):
        train_csv_path = Path("../input/hotel-id-2021-fgvc8/train.csv")
        if train_csv_path.exists():
            train_df = pd.read_csv(train_csv_path, usecols=["hotel_id"])
            top_hotels = (
                train_df["hotel_id"].value_counts().head(5).index.astype(str).tolist()
            )
        else:
            top_hotels = ["0", "0", "0", "0", "0"]
        fallback = " ".join(top_hotels)
        preds = [fallback] * len(submission)
    else:
        vlen = len(vocab)
        preds = []
        for row in preds_idx:
            row_ids = []
            for i in row:
                ii = int(i)
                row_ids.append(_vocab_item_to_str(vocab[ii]) if 0 <= ii < vlen else "0")
            preds.append(" ".join(row_ids))
else:
    train_csv_path = Path("../input/hotel-id-2021-fgvc8/train.csv")
    if train_csv_path.exists():
        train_df = pd.read_csv(train_csv_path, usecols=["hotel_id"])
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
