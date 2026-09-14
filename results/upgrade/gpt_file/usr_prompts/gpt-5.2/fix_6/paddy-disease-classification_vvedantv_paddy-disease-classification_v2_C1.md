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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

if False:
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            print(os.path.join(dirname, filename))



## === cell 1
from pathlib import Path

comp = "paddy-disease-classification"
base = Path("/kaggle/input")

candidates = [
    base / comp,
    base / comp / comp,
    base / "paddy-disease-classification",
    base / "paddy-disease-classification" / "paddy-disease-classification",
]
path = next((p for p in candidates if p.exists()), None)
assert path is not None, f"Dataset path not found. Tried: {candidates}"

path



## === cell 2
from fastai.vision.all import *

set_seed(42, reproducible=True)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if False:
    path.ls()



## === cell 3
trn_path = path / "train_images"
files = get_image_files(trn_path)
len(files), files[0]



## === cell 4
if False:
    img = PILImage.create(files[0])
    print(img.size)
    img.to_thumb(128)



## === cell 5
if False:
    from fastcore.parallel import *

    def f(o):
        return PILImage.create(o).size

    sizes = parallel(f, files, n_workers=min(8, os.cpu_count() or 1))
    pd.Series(sizes).value_counts()



## === cell 6
n_cpu = os.cpu_count() or 1

item_tfms = Resize(128, method="squish")
batch_tfms = aug_transforms(size=128, min_scale=0.75)

dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=item_tfms,
    batch_tfms=batch_tfms,
)

dls = dblock.dataloaders(
    trn_path,
    bs=64,
    num_workers=min(8, n_cpu),
    prefetch_factor=2,
    pin_memory=True,
    persistent_workers=True,
)

if False:
    dls.show_batch(max_n=6)



## === cell 7
learn = vision_learner(dls, "resnet26d", metrics=error_rate, path=".").to_fp16()



## === cell 8
if False:
    learn.lr_find(suggest_funcs=(valley, slide))



## === cell 9
learn.fine_tune(3, 0.01)



## === cell 10
ss = pd.read_csv(path / "sample_submission.csv")
ss.head(), ss.shape



## === cell 11
tst_files = sorted(get_image_files(path / "test_images"), key=lambda p: p.name)

tst_dl = dls.test_dl(
    tst_files,
    bs=128,
    num_workers=min(8, n_cpu),
    pin_memory=True,
    persistent_workers=True,
)

len(tst_files), tst_files[0].name, tst_files[-1].name



## === cell 12
probs, _, idxs = learn.get_preds(dl=tst_dl, with_decoded=True)
idxs[:10], probs.shape



## === cell 13
dls.vocab



## === cell 14
vocab = np.array(dls.vocab, dtype=object)
preds_labels = vocab[idxs.cpu().numpy()].tolist()
len(preds_labels), preds_labels[:10]



## === cell 15
pred_df = pd.DataFrame({"image_id": [p.name for p in tst_files], "label": preds_labels})

sub = ss.drop(columns=["label"]).merge(pred_df, on="image_id", how="left")

missing = sub["label"].isna().sum()
assert (
    missing == 0
), f"Missing predictions for {missing} rows; check test file discovery / merge."

sub.to_csv("submission.csv", index=False)

with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().rstrip())
