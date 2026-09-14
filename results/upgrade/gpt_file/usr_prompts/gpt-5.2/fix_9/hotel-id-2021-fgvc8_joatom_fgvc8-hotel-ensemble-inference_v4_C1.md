# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
from pathlib import Path

import pandas as pd
import numpy as np
import fastai
from fastai.vision.all import *
import dill

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
fastai.__version__, torch.__version__



## === cell 2
DATA_ROOT = Path("../input/hotel-id-2021-fgvc8")
TEST_IMG_DIR = DATA_ROOT / "test_images"
TRAIN_IMG_DIR = DATA_ROOT / "train_images"
TRAIN_CSV = DATA_ROOT / "train.csv"
SAMPLE_SUB = DATA_ROOT / "sample_submission.csv"

assert SAMPLE_SUB.exists(), f"Missing sample submission at: {SAMPLE_SUB}"
assert TRAIN_CSV.exists(), f"Missing train.csv at: {TRAIN_CSV}"
assert TEST_IMG_DIR.exists(), f"Missing test_images dir at: {TEST_IMG_DIR}"
assert TRAIN_IMG_DIR.exists(), f"Missing train_images dir at: {TRAIN_IMG_DIR}"

submission = pd.read_csv(SAMPLE_SUB)

test_paths = [TEST_IMG_DIR / img for img in submission["image"].tolist()]
missing_ct = int(np.sum([not os.path.exists(str(p)) for p in test_paths]))
missing_ct



## === cell 3
INPUT_ROOT = Path("../input")


def find_exported_learners(root: Path, max_candidates: int = 25) -> list:
    if not root.exists():
        return []

    comp_root = root / "hotel-id-2021-fgvc8"

    likely = []
    if comp_root.exists():
        likely += [
            comp_root / "export.pkl",
            comp_root / "models" / "export.pkl",
            comp_root / "export_0.pkl",
            comp_root / "export_1.pkl",
            comp_root / "export_resnet34.pkl",
            comp_root / "export_resnet50.pkl",
        ]

        for sub in ("", "hotel-id-2021-fgvc8"):
            base = comp_root / sub if sub else comp_root
            models_dir = base / "models"
            if models_dir.exists():
                likely += [
                    models_dir / "export.pkl",
                    models_dir / "export_0.pkl",
                    models_dir / "export_1.pkl",
                ]

    existing = [p for p in likely if p.is_file() and p.stat().st_size > 1024]
    existing = sorted(set(existing), key=lambda p: str(p))

    if len(existing) > 0:
        return existing[:max_candidates] if max_candidates is not None else existing

    candidates = []
    if comp_root.exists():
        candidates += list(comp_root.rglob("export.pkl"))
        candidates += list(comp_root.rglob("export_*.pkl"))
        candidates += list(comp_root.rglob("export*.pkl"))

    pkl_files = [p for p in candidates if p.is_file() and p.stat().st_size > 1024]
    pkl_files = sorted(set(pkl_files), key=lambda p: str(p))
    if max_candidates is not None and len(pkl_files) > max_candidates:
        pkl_files = pkl_files[:max_candidates]
    return pkl_files


models = find_exported_learners(INPUT_ROOT)
models[:10], len(models)




## === cell 4
def _vocab_to_list(v):
    """Normalize vocab to a plain list for correct index->hotel_id mapping."""
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


def _try_load_and_predict_with_exports(models, test_paths):
    probs_sum = None
    n_models_used = 0
    learn_last = None

    if len(models) == 0:
        return None, None, 0

    if not all(os.path.exists(str(p)) for p in test_paths):
        return None, None, 0

    test_items = list(test_paths)

    for model_path in models:
        try:
            learn_tmp = load_learner(
                fname=Path(model_path), cpu=False, pickle_module=dill
            )

            test_dl = learn_tmp.dls.test_dl(test_items, num_workers=0)

            with torch.inference_mode():
                probs_temp, _ = learn_tmp.tta(dl=test_dl, n=5)

            probs_sum = probs_temp if probs_sum is None else (probs_sum + probs_temp)
            n_models_used += 1
            learn_last = learn_tmp
        except Exception as e:
            print(f"Skipping model {model_path} due to error: {type(e).__name__}: {e}")
            continue

    if probs_sum is None or learn_last is None or n_models_used == 0:
        return None, None, 0
    return probs_sum / float(n_models_used), learn_last, n_models_used


probs, learn, n_models_used = _try_load_and_predict_with_exports(models, test_paths)
(n_models_used, probs is not None)




## === cell 5
def _train_fallback_learner_and_predict(train_csv_path, train_img_dir, test_paths):
    """
    Bugfix: build a correct filename column for ImageDataLoaders.from_df.
    The previous code passed fn_col=None which breaks fastai's pipeline and triggers
    an AttributeError deep in augmentation (list has no .device).
    """
    df = pd.read_csv(train_csv_path)

    df["hotel_id"] = df["hotel_id"].astype(str)
    df["chain"] = df["chain"].astype(str)

    df["image_path"] = df["chain"].str.cat(df["image"].astype(str), sep=os.sep)
    df["image_path"] = df["image_path"].map(lambda x: str(train_img_dir / x))

    exists = df["image_path"].map(os.path.exists).to_numpy(dtype=np.bool_)
    df = df.loc[exists].reset_index(drop=True)

    if len(df) < 1000:
        return None, None

    max_train = 30000
    if len(df) > max_train:
        df = df.sample(n=max_train, random_state=42).reset_index(drop=True)

    dls = ImageDataLoaders.from_df(
        df,
        fn_col="image_path",
        label_col="hotel_id",
        valid_pct=0.1,
        seed=42,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(size=224, min_scale=0.75),
        bs=64,
        num_workers=0,
    )

    learn_fb = vision_learner(dls, resnet34, metrics=[])
    learn_fb.fine_tune(1)

    test_dl = learn_fb.dls.test_dl(list(test_paths), num_workers=0)
    with torch.inference_mode():
        probs_fb, _ = learn_fb.tta(dl=test_dl, n=5)

    return probs_fb, learn_fb


if probs is None or learn is None or n_models_used == 0:
    print(
        "No usable exported learner found; training a minimal fallback model for better MAP@5."
    )
    probs_fb, learn_fb = _train_fallback_learner_and_predict(
        TRAIN_CSV, TRAIN_IMG_DIR, test_paths
    )
    if probs_fb is not None and learn_fb is not None:
        probs, learn = probs_fb, learn_fb

(probs is not None, learn is not None)



## === cell 6
if probs is not None and learn is not None:
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



## === cell 7
submission["hotel_id"] = preds
submission_path = Path("submission.csv")
submission.to_csv(submission_path, index=False)

submission.head(), str(submission_path.resolve())
