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

3.13

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
import numpy as np
import pandas as pd
import os
from pathlib import Path

for dirname, _, filenames in os.walk("/kaggle/input"):
    for i, filename in enumerate(filenames[:3]):
        print(os.path.join(dirname, filename))



## === cell 1
comp = "paddy-disease-classification"

candidates = [
    Path("/kaggle/input") / comp,
    Path("/kaggle/input") / "paddy-disease-classification",
    Path("/kaggle/data") / comp,
    Path("/kaggle/data") / "paddy-disease-classification",
]
path = next((p for p in candidates if p.exists()), None)
if path is None:
    raise FileNotFoundError(f"Could not locate dataset folder. Tried: {candidates}")

print("Using dataset path:", path)



## === cell 2
path



## === cell 3
from fastai.vision.all import *
from fastcore.parallel import parallel



## === cell 4
path.ls()



## === cell 5
trn_path = path / "train_images"
files = get_image_files(trn_path)
len(files), files[0]



## === cell 6
img = PILImage.create(files[0])
print(img)
img.to_thumb(128)




## === cell 7
def f(o):
    return PILImage.create(o).size


sizes = parallel(f, files, n_workers=8)
pd.Series(sizes).value_counts().head(10)



## === cell 8
pass



## === cell 9
dls = ImageDataLoaders.from_folder(
    trn_path,
    valid_pct=0.2,
    seed=42,
    item_tfms=Resize(480, method="squish"),
    batch_tfms=aug_transforms(size=128, min_scale=0.75),
)



## === cell 10
dls.show_batch(max_n=6)



## === cell 11
learn = vision_learner(dls, "resnet26d", metrics=error_rate, path=Path(".")).to_fp16()



## === cell 12
learn.lr_find(suggest_funcs=(valley, slide))



## === cell 13
learn.fine_tune(3, 0.01)



## === cell 14
dls



## === cell 15
ss = pd.read_csv(path / "sample_submission.csv")
ss.head()



## === cell 16
test_dir = path / "test_images"
tst_map = {p.name: p for p in get_image_files(test_dir)}
missing = set(ss["image_id"]) - set(tst_map.keys())
if missing:
    raise FileNotFoundError(
        f"Some test images listed in sample_submission are missing on disk, e.g. {list(missing)[:5]}"
    )

tst_files = [tst_map[iid] for iid in ss["image_id"].tolist()]
tst_dl = dls.test_dl(tst_files)

len(tst_files), tst_files[0]



## === cell 17
pass



## === cell 18
probs, _, idxs = learn.get_preds(dl=tst_dl, with_decoded=True)



## === cell 19
probs.shape



## === cell 20
idxs[:10]



## === cell 21
pass



## === cell 22
dls.vocab



## === cell 23
mapping = dict(enumerate(dls.vocab))
results = pd.Series(idxs.cpu().numpy(), name="idxs").map(mapping)
results.head()



## === cell 24
ss["label"] = results.values
ss.to_csv("submission.csv", index=False)
print(ss.head())
print("\nWrote submission.csv with shape:", ss.shape)

with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().strip())



## === cell 25
print("Done. submission.csv is ready.")
