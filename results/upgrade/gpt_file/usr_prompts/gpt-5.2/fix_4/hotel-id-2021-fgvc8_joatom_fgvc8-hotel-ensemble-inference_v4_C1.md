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

0.6023259240548947

# 6. Current score

0.00209

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'The notebook fails because it references pretrained `.pkl` models in `../input/fgvc8hotel/` that do not exist in your dataset paths; that leaves `probs=None` and cascades into later errors. I fix this by (1) automatically locating any available exported fastai learners (`.pkl`) under the provided `../input/` tree and using them if present, and (2) adding a safe fallback that still produces a valid MAP@5 submission (using the top-5 most frequent `hotel_id` from `train.csv`) when no models are found. I also fix path handling for test images so the test dataloader receives proper file paths, and ensure the submission is written as `submission.csv` with the required columns.'
- What this solution (achieved 0.00209) has done: 'Your current score is low because the notebook almost certainly falls back to the “top-5 most frequent hotels” baseline (meaning no usable exported learner is found/loaded), and even when a learner loads, `test_dl` is built from string paths in a way that often won’t be recognized as valid images by the learner’s pipeline. I make two minimal, score-relevant fixes: (1) reliably locate and load the specific exported fastai learner(s) if present (prefer `export.pkl`), and (2) construct the test dataloader from actual image `Path` objects (not a renamed string column) so transforms work and predictions aren’t garbage. I also correctly ensemble multiple learners by averaging probabilities (instead of summing without normalization) and ensure vocab-to-hotel_id mapping is robust (handles list/CategoryMap). These changes keep the same core approach (load exported learners → TTA → top-5) but should move MAP@5 substantially upward toward your target if any valid exports exist; if not, it still writes a valid submission.'
- What this solution (achieved 0.00209) has done: 'I keep your fastai inference approach (load exported learners → build test_dl → TTA → top-5) but make two minimal, score-relevant fixes that commonly cause near-random MAP@5: (1) only search for exported learners that are likely compatible (and avoid accidentally loading unrelated `.pkl` files from other Kaggle datasets), and (2) construct the test dataloader from a list of `PILImage` objects (not bare Paths) to reliably trigger the same item transforms as training. I also normalize the ensemble probabilities safely and ensure the predicted hotel IDs are always space-delimited strings matching submission order. These changes should move you off the “fallback/top5” behavior and toward your target if an actual export exists, while still producing a valid `submission.csv` in all cases.'

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
DATA_ROOT = Path("../input/hotel-id-2021-fgvc8")
TEST_IMG_DIR = DATA_ROOT / "test_images"
TRAIN_CSV = DATA_ROOT / "train.csv"
SAMPLE_SUB = DATA_ROOT / "sample_submission.csv"

assert SAMPLE_SUB.exists(), f"Missing sample submission at: {SAMPLE_SUB}"
assert TRAIN_CSV.exists(), f"Missing train.csv at: {TRAIN_CSV}"
assert TEST_IMG_DIR.exists(), f"Missing test_images dir at: {TEST_IMG_DIR}"

submission = pd.read_csv(SAMPLE_SUB)

test_paths = [TEST_IMG_DIR / img for img in submission["image"].tolist()]
missing_ct = sum(0 if p.exists() else 1 for p in test_paths)
missing_ct



## === cell 3
INPUT_ROOT = Path("../input")


def find_exported_learners(root: Path) -> list:
    """
    Change (score-relevant): restrict search to fastai exports under the current competition
    folder first, to avoid loading unrelated .pkl files that silently produce nonsense preds.
    Prefer export.pkl patterns; only then consider other .pkl inside the same competition tree.
    """
    if not root.exists():
        return []

    preferred_roots = []
    comp_root = root / "hotel-id-2021-fgvc8"
    if comp_root.exists():
        preferred_roots.append(comp_root)
    preferred_roots.append(root)

    candidates = []
    for rr in preferred_roots:
        candidates += list(rr.rglob("export.pkl"))
        candidates += list(rr.rglob("export_*.pkl"))
        candidates += list(rr.rglob("export*.pkl"))

    if len(candidates) == 0:
        rr = comp_root if comp_root.exists() else root
        candidates = list(rr.rglob("*.pkl"))

    pkl_files = [p for p in candidates if p.is_file() and p.stat().st_size > 1024]

    seen = set()
    out = []
    for p in pkl_files:
        sp = str(p.resolve())
        if sp not in seen:
            out.append(p)
            seen.add(sp)
    return out


models = find_exported_learners(INPUT_ROOT)
models[:10], len(models)




## === cell 4
def _vocab_to_list(v):
    """
    Exported learners may store vocab as CategoryMap/L/list-like.
    Normalize to a plain Python list for correct index->hotel_id mapping.
    """
    if v is None:
        return None
    if hasattr(v, "items"):
        try:
            return list(v.items)
        except Exception:
            pass
    try:
        return list(v)
    except Exception:
        return None


probs_sum = None
n_models_used = 0
learn = None  # keep last successful learner for vocab mapping

test_items = []
for p in test_paths:
    try:
        test_items.append(PILImage.create(p))
    except Exception:
        test_items.append(None)

if len(models) > 0 and all(x is not None for x in test_items):
    for model_path in models:
        try:
            learn_tmp = load_learner(
                fname=Path(model_path), cpu=False, pickle_module=dill
            )

            test_dl = learn_tmp.dls.test_dl(test_items)

            probs_temp, _ = learn_tmp.tta(dl=test_dl, n=5)

            if probs_sum is None:
                probs_sum = probs_temp
            else:
                probs_sum += probs_temp

            n_models_used += 1
            learn = learn_tmp
        except Exception as e:
            print(f"Skipping model {model_path} due to error: {type(e).__name__}: {e}")
            continue

(n_models_used, probs_sum is not None)



## === cell 5
if probs_sum is not None and learn is not None and n_models_used > 0:
    probs = probs_sum / float(n_models_used)

    preds_idx = probs.topk(5)[1].detach().cpu().numpy()

    vocab_raw = getattr(learn.dls, "vocab", None)
    vocab_list = _vocab_to_list(vocab_raw)

    if vocab_list is None or len(vocab_list) == 0:
        preds = [" ".join(map(str, row)) for row in preds_idx]
    else:
        preds = [" ".join([str(vocab_list[i]) for i in row]) for row in preds_idx]
else:
    train_df = pd.read_csv(TRAIN_CSV)
    top5 = train_df["hotel_id"].value_counts().head(5).index.tolist()
    top5_str = " ".join(map(str, top5))
    preds = [top5_str] * len(submission)

preds[:3], len(preds)



## === cell 6
submission["hotel_id"] = preds
submission_path = Path("submission.csv")
submission.to_csv(submission_path, index=False)

submission.head(), str(submission_path.resolve())
