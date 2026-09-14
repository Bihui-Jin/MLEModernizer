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

3.11

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
from pathlib import Path
import pandas as pd
import warnings, sys, os

warnings.filterwarnings("ignore")

candidates = [
    Path("/kaggle/input/paddy-disease-classification"),
    Path("data/paddy-disease-classification"),
    Path("input/paddy-disease-classification"),
    Path("working/paddy-disease-classification"),
]

path = None
for cand in candidates:
    if (cand / "train.csv").exists() and (cand / "sample_submission.csv").exists():
        path = cand
        break

if path is None:
    raise FileNotFoundError(
        "Could not locate the competition data folder. Checked: "
        + ", ".join(str(c) for c in candidates)
    )

print(f"Using data path: {path}")



## === cell 1
from fastai.vision.all import *
import torch

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # faster cuDNN kernels on GPU
    torch.backends.cudnn.deterministic = False
else:
    torch.backends.cudnn.benchmark = False  # avoid unnecessary CUDA checks

torch.set_num_threads(os.cpu_count() or 1)

set_seed(42, reproducible=True)

print("FastAI imported successfully.")



## === cell 2
trn_path = path / "train_images"
if not trn_path.exists():
    raise FileNotFoundError(f"Training images folder not found at {trn_path}")

files = get_image_files(trn_path)
print(f"Found {len(files)} training images.")



## === cell 3
img = PILImage.create(files[0])
print("First image size:", img.size)



## === cell 4
sizes = None
print("Skipping size analysis to save time.")



## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_mem = device.type == "cuda"

num_workers = min(4, os.cpu_count() or 1)  # moderate parallelism

dls = ImageDataLoaders.from_folder(
    trn_path,
    valid_pct=0.2,
    seed=42,
    item_tfms=Resize(128, method="squish"),
    batch_tfms=Normalize.from_stats(*imagenet_stats),  # lightweight transform
    bs=256,  # smaller batch for faster, stable training
    num_workers=num_workers,
    pin_memory=pin_mem,
    persistent_workers=True,
)
print("DataLoaders created.")



## === cell 6
learn = vision_learner(dls, "resnet26d", metrics=error_rate, path=".")
if torch.cuda.is_available():
    learn = learn.to_fp16()
learn.to(device)  # ensure learner uses the detected device
print("Learner instantiated.")



## === cell 7
fixed_lr = 1e-3
print(f"Using fixed learning rate: {fixed_lr:.0e}")



## === cell 8
learn.fine_tune(3, base_lr=fixed_lr)
print("Training completed.")



## === cell 9
ss = pd.read_csv(path / "sample_submission.csv")
print(f"Sample submission shape: {ss.shape}")



## === cell 10
tst_files = get_image_files(path / "test_images").sorted()
tst_dl = dls.test_dl(
    tst_files,
    bs=256,  # match training batch size for efficiency
    num_workers=num_workers,
    pin_memory=pin_mem,
    persistent_workers=True,
)
probs, _, idxs = learn.get_preds(dl=tst_dl, with_decoded=True)
print("Predictions generated for test set.")



## === cell 11
mapping = dict(enumerate(dls.vocab))
results = pd.Series(idxs.numpy(), name="idx").map(mapping)
ss["label"] = results.values
ss.to_csv("subm.csv", index=False)
print("Submission file 'subm.csv' saved.")
print(ss.head())



## === cell 12
if os.getenv("KAGGLE_KERNEL_RUN_TYPE") is None:
    print("Running locally; automatic Kaggle submission skipped.")
else:
    from kaggle import api

    api.competition_submit(
        "subm.csv", "resnet26d fine‑tuned", "paddy-disease-classification"
    )
