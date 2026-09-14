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
import os
from pathlib import Path

comp = "paddy-disease-classification"


def resolve_comp_path(comp_name: str) -> Path:
    candidates = [
        Path("/kaggle/input") / comp_name,
        Path("/kaggle/data") / comp_name,
        Path("/kaggle/input") / "paddy-disease-classification",
        Path("/kaggle/data") / "paddy-disease-classification",
        Path(".") / comp_name,
        Path("."),
    ]
    for p in candidates:
        if (p / "train.csv").exists() and (p / "sample_submission.csv").exists():
            return p
    for root in [Path("/kaggle/input"), Path("/kaggle/data")]:
        if root.exists():
            for p in root.rglob("train.csv"):
                pp = p.parent
                if (pp / "sample_submission.csv").exists():
                    return pp
    raise FileNotFoundError(
        "Could not locate competition data folder containing train.csv and sample_submission.csv"
    )


path = resolve_comp_path(comp)
path



## === cell 1
(path / "train.csv").exists(), (path / "sample_submission.csv").exists()



## === cell 2
from fastai.vision.all import *
import pandas as pd
import numpy as np
import torch

set_seed(42, reproducible=True)

cpu_cnt = os.cpu_count() or 4
try:
    torch.set_num_threads(max(1, cpu_cnt // 2))
    torch.set_num_interop_threads(1)
except Exception:
    pass

os.environ.pop("OMP_NUM_THREADS", None)
os.environ.pop("MKL_NUM_THREADS", None)

torch.backends.cudnn.benchmark = torch.cuda.is_available()

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

path



## === cell 3
trn_path = path / "train_images"

trn_path.exists()



## === cell 4
pass



## === cell 5
pass



## === cell 6
pass



## === cell 7
pass



## === cell 8
cpu_cnt = os.cpu_count() or 2
nw = min(2, max(1, cpu_cnt // 4))  # fewer workers often faster/more stable on Kaggle
persist = nw > 0
prefetch = 2

dls = ImageDataLoaders.from_folder(
    trn_path,
    valid_pct=0.2,
    seed=42,
    item_tfms=Resize(128),
    batch_tfms=aug_transforms(size=128, min_scale=0.75),
    bs=64,
    num_workers=nw,
    persistent_workers=persist,
    pin_memory=torch.cuda.is_available(),
    prefetch_factor=prefetch,
    cache="ram",  # key speedup vs repeated decode/resize each epoch
)



## === cell 9
learn = vision_learner(dls, "resnet26d", metrics=error_rate, path=".")
if torch.cuda.is_available():
    learn.model = learn.model.to(memory_format=torch.channels_last)
learn = learn.to_fp16()
learn



## === cell 10
pass



## === cell 11
learn.fine_tune(3, 0.01)



## === cell 12
ss = pd.read_csv(path / "sample_submission.csv")
ss.head(), ss.shape



## === cell 13
tst_files = sorted(get_image_files(path / "test_images"))

tst_dl = dls.test_dl(
    tst_files,
    num_workers=nw,
    persistent_workers=persist,
    pin_memory=torch.cuda.is_available(),
    prefetch_factor=prefetch,
)
len(tst_files), tst_files[0]



## === cell 14
pass



## === cell 15
learn.model.eval()
with torch.inference_mode():
    probs, _ = learn.get_preds(dl=tst_dl)
    idxs = probs.argmax(dim=1)
idxs[:10], probs.shape



## === cell 16
vocab = dls.vocab
len(vocab), vocab[:3]



## === cell 17
pass



## === cell 18
pass



## === cell 19
int(idxs.min()), int(idxs.max()), len(vocab)



## === cell 20
idxs_np = idxs.detach().cpu().numpy().astype(np.int64, copy=False)
vocab_arr = np.asarray(vocab, dtype=object)
pred_labels = vocab_arr[idxs_np].astype(str, copy=False)

pred_df = pd.DataFrame({"image_id": [p.name for p in tst_files], "label": pred_labels})

ss = ss.drop(columns=["label"], errors="ignore").merge(
    pred_df, on="image_id", how="left"
)
if ss["label"].isna().any():
    ss["label"] = ss["label"] = ss["label"].fillna(vocab[0])

ss.head(), ss.shape



## === cell 21
ss.to_csv("submission.csv", index=False)

with open("submission.csv", "r", encoding="utf-8") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))
print(
    "Saved submission.csv with", len(ss), "rows at", os.path.abspath("submission.csv")
)
