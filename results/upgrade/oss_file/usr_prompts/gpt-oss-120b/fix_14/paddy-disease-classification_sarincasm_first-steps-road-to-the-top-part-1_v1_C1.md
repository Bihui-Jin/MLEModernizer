# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.12

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
try:
    import fastai
except ImportError as e:
    raise ImportError("fastai is required but not installed in the environment.") from e




## === cell 1
from pathlib import Path
import os

candidates = [
    Path("data/paddy-disease-classification"),
    Path("input/paddy-disease-classification"),
    Path("paddy-disease-classification"),
    Path("."),
]
base_path = None
for cand in candidates:
    if (cand / "train_images").exists():
        base_path = cand
        break

if base_path is None:
    raise FileNotFoundError(
        "Could not locate the dataset folder. Checked: "
        + ", ".join(str(c) for c in candidates)
    )
print(f"Using base path: {base_path.resolve()}")
BASE_PATH = base_path




## === cell 2
import pandas as pd
from fastai.vision.all import *
import torch

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # faster non‑deterministic ops
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    worker_count = min(8, os.cpu_count() or 1)  # more workers for GPU pipelines
    batch_size = 1024  # keep large batch for GPU
    pin_mem = True
else:
    torch.set_num_threads(min(8, os.cpu_count() or 1))  # keep CPU threads reasonable
    worker_count = min(4, os.cpu_count() or 1)
    batch_size = 512  # larger CPU batch reduces iteration overhead
    pin_mem = False

set_seed(42)

trn_path = BASE_PATH / "train_images"
assert trn_path.exists(), f"Training images folder not found at {trn_path}"

dls = ImageDataLoaders.from_folder(
    trn_path,
    valid_pct=0.2,
    seed=42,
    item_tfms=Resize(128, method="squish"),
    batch_tfms=Normalize.from_stats(*imagenet_stats),  # lightweight preprocessing
    bs=batch_size,
    num_workers=worker_count,
    pin_memory=pin_mem,
    loader_kwargs=dict(persistent_workers=True, prefetch_factor=2),
)




## === cell 3
learn = vision_learner(dls, "resnet26d", metrics=error_rate, path=".")

if torch.cuda.is_available():
    learn = learn.to_fp16()

learn.fine_tune(3, 0.01)




## === cell 4
sample_sub_path = BASE_PATH / "sample_submission.csv"
assert sample_sub_path.exists(), f"sample_submission.csv not found at {sample_sub_path}"
ss = pd.read_csv(sample_sub_path)




## === cell 5
test_path = BASE_PATH / "test_images"
assert test_path.exists(), f"Test images folder not found at {test_path}"
test_files = get_image_files(test_path).sorted()
tst_dl = dls.test_dl(test_files)

probs, _, idxs = learn.get_preds(dl=tst_dl, with_decoded=True)

pred_labels = pd.Series(idxs.numpy()).map({i: c for i, c in enumerate(dls.vocab)})

pred_df = pd.DataFrame({"image_id": [f.name for f in test_files], "label": pred_labels})




## === cell 6
submission = ss.drop(columns=["label"]).merge(pred_df, on="image_id", how="left")
submission_path = Path("submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(submission.head())
