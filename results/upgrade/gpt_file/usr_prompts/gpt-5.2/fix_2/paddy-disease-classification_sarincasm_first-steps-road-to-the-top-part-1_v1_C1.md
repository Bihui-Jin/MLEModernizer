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
I’ll remove the dependency on `fastkaggle.setup_comp()` (it fails because Kaggle API credentials aren’t available) and instead point `path` directly to the already-present dataset directory in this environment. I’ll keep the fastai/timm training and inference logic the same, only adding a small, safe path-resolution fallback so it works whether the data is under `/kaggle/input/...` or `/kaggle/data/...`. I’ll also make sure the test predictions are aligned to `sample_submission.csv` by mapping predictions using `image_id`, which avoids accidental row-order mismatches and guarantees a valid `submission.csv` is created. The result will run end-to-end and write `submission.csv` with the required columns.

```python


## === cell 1
import os
from pathlib import Path

comp = 'paddy-disease-classification'

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
    raise FileNotFoundError("Could not locate competition data folder containing train.csv and sample_submission.csv")

path = resolve_comp_path(comp)
path



## === cell 2
path



## === cell 3
from fastai.vision.all import *
import pandas as pd
import numpy as np

set_seed(42)
path.ls()



## === cell 4
trn_path = path/'train_images'
files = get_image_files(trn_path)
len(files), files[0]



## === cell 5
img = PILImage.create(files[0])
print(img.size)
img.to_thumb(128)



## === cell 6
from fastcore.parallel import *

def f(o): return PILImage.create(o).size
sizes = parallel(f, files, n_workers=8)
sizes[:5], len(sizes)



## === cell 7
pd.Series(sizes)



## === cell 8
pd.Series(sizes).value_counts().head(10)



## === cell 9
dls = ImageDataLoaders.from_folder(
    trn_path, valid_pct=0.2, seed=42,
    item_tfms=Resize(480, method='squish'),
    batch_tfms=aug_transforms(size=128, min_scale=0.75)
)

dls.show_batch(max_n=6)



## === cell 10
learn = vision_learner(dls, 'resnet26d', metrics=error_rate, path='.').to_fp16()
learn



## === cell 11
learn.lr_find(suggest_funcs=(valley, slide))



## === cell 12
learn.fine_tune(3, 0.01)



## === cell 13
ss = pd.read_csv(path/'sample_submission.csv')
ss.head(), ss.shape



## === cell 14
tst_files = get_image_files(path/'test_images').sorted()
tst_dl = dls.test_dl(tst_files)
len(tst_files), tst_files[0]



## === cell 15
tst_dl



## === cell 16
probs, _, idxs = learn.get_preds(dl=tst_dl, with_decoded=True)
idxs[:10], probs.shape



## === cell 17
dls.vocab



## === cell 18
list(enumerate(dls.vocab))



## === cell 19
mapping = dict(enumerate(dls.vocab))
mapping



## === cell 20
pd.Series(idxs.cpu().numpy(), name="idxs").head()



## === cell 21
pred_labels = pd.Series(idxs.cpu().numpy(), name="idxs").map(mapping).astype(str)
pred_df = pd.DataFrame({
    "image_id": [p.name for p in tst_files],
    "label": pred_labels.values
})

ss = ss.drop(columns=["label"], errors="ignore").merge(pred_df, on="image_id", how="left")

if ss["label"].isna().any():
    ss["label"] = ss["label"].fillna(dls.vocab[0])

ss.head(), ss.shape



## === cell 22
ss.to_csv('submission.csv', index=False)
!head submission.csv
print("Saved submission.csv with", len(ss), "rows at", os.path.abspath("submission.csv"))
```
