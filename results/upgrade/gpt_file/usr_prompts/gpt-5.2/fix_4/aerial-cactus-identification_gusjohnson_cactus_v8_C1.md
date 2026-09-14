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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.0006

# 6. Current score

0.9996

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.9996) has done: 'I fix the unzip/path logic so the script can reliably locate the extracted `train/` and `test/` folders (your current extraction path doesn’t match where `_find_dir` searches, causing the first failure and cascading `NameError`s). I make extraction idempotent and handle the common case where the dataset is already unzipped under `/kaggle/input/aerial-cactus-identification/train` and `/test`, avoiding unnecessary work. Then I keep your fastai training and prediction pipeline the same, only ensuring predictions are aligned to `sample_submission.csv` ids and a valid `submission.csv` is always written.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd



## === cell 1
import zipfile
from pathlib import Path
from fastai.vision.all import *
import torch

DATA = Path("/kaggle/input/aerial-cactus-identification")
WORK = Path("/kaggle/working")
TMP = WORK / "temp"
TMP.mkdir(parents=True, exist_ok=True)

test_df = pd.read_csv(DATA / "sample_submission.csv")
train_df = pd.read_csv(DATA / "train.csv")

train_zip = DATA / "train.zip"
test_zip = DATA / "test.zip"


def _safe_extract(zip_path: Path, dest: Path):
    """
    Bugfix: some Kaggle variants already include extracted folders; in others, zip exists.
    Extract only if needed, and extract into TMP (as intended).
    """
    if not zip_path.exists():
        return
    if any(dest.rglob("*.jpg")):
        return
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)


def _find_dir(root: Path, target_name: str) -> Path:
    direct = root / target_name
    if direct.exists() and direct.is_dir():
        return direct
    candidates = [p for p in root.rglob(target_name) if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find '{target_name}' directory under {root}"
        )
    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    return candidates[0]


input_train = DATA / "train"
input_test = DATA / "test"

if input_train.exists() and input_test.exists():
    TRAIN_DIR, TEST_DIR = input_train, input_test
else:
    _safe_extract(train_zip, TMP)
    _safe_extract(test_zip, TMP)
    TRAIN_DIR = _find_dir(TMP, "train")
    TEST_DIR = _find_dir(TMP, "test")

assert len(get_image_files(TRAIN_DIR)) > 0, f"No images found under {TRAIN_DIR}"
assert len(get_image_files(TEST_DIR)) > 0, f"No images found under {TEST_DIR}"

TRAIN_DIR, TEST_DIR



## === cell 2
dls = ImageDataLoaders.from_df(
    df=train_df,
    path=TRAIN_DIR.parent,  # parent of the actual train folder
    folder=TRAIN_DIR.name,  # "train"
    fn_col="id",
    label_col="has_cactus",
    valid_pct=0.2,
    seed=42,
    item_tfms=Resize(32),
    batch_tfms=aug_transforms(size=32, min_scale=0.9),
)



## === cell 3
learn = cnn_learner(
    dls, resnet18, metrics=[error_rate, accuracy], loss_func=CrossEntropyLossFlat()
)



## === cell 4
learn.fine_tune(1)
learn.fit_one_cycle(5, slice(0.003))



## === cell 5
test_files = get_image_files(TEST_DIR)
test_dl = learn.dls.test_dl(test_files, shuffle=False, drop_last=False)

preds, _ = learn.get_preds(dl=test_dl)
has_cactus_prob = preds[:, 1].cpu().numpy()



## === cell 6
pred_id_order = [p.name for p in test_files]
pred_map = dict(zip(pred_id_order, has_cactus_prob))

submission_df = test_df.copy()
submission_df["has_cactus"] = submission_df["id"].map(pred_map).astype(float)

submission_df["has_cactus"] = submission_df["has_cactus"].fillna(0.5)

submission_df = submission_df[["id", "has_cactus"]]
submission_df.head()



## === cell 7
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Wrote submission to: {submission_path}")
print(submission_df.shape)
submission_df.head()
