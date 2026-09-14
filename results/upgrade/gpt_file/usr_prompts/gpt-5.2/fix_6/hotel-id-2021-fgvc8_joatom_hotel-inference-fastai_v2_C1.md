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



## === cell 1
fastai.__version__, torch.__version__



## === cell 2
CANDIDATE_ROOTS = [
    Path("../input/hotel-id-2021-fgvc8"),
    Path("/kaggle/input/hotel-id-2021-fgvc8"),
    Path("../kaggle/input/hotel-id-2021-fgvc8"),
    Path("../input"),
    Path("/kaggle/input"),
]

DATA_ROOT = None
for p in CANDIDATE_ROOTS:
    if (p / "train.csv").exists() and (p / "sample_submission.csv").exists():
        DATA_ROOT = p
        break
    if (p / "hotel-id-2021-fgvc8" / "train.csv").exists():
        DATA_ROOT = p / "hotel-id-2021-fgvc8"
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find dataset root containing train.csv and sample_submission.csv under known input paths."
    )

TRAIN_CSV = DATA_ROOT / "train.csv"
SAMPLE_SUB = DATA_ROOT / "sample_submission.csv"
TRAIN_IMG_DIR = DATA_ROOT / "train_images"
TEST_IMG_DIR = DATA_ROOT / "test_images"

TRAIN_CSV, SAMPLE_SUB, TRAIN_IMG_DIR.exists(), TEST_IMG_DIR.exists()



## === cell 3
submission = pd.read_csv(SAMPLE_SUB)
test = submission.copy()

test_items = (TEST_IMG_DIR.as_posix() + "/" + test["image"].astype(str)).tolist()
test_items[:3], len(test_items)



## === cell 4
EXPORT_CANDIDATES = [
    Path("../input/hotel-train-fastai-densnet161/export_dn161_kaggle_notebook.pkl"),
    Path(
        "/kaggle/input/hotel-train-fastai-densnet161/export_dn161_kaggle_notebook.pkl"
    ),
]

export_path = None
for p in EXPORT_CANDIDATES:
    if p.exists():
        export_path = p
        break

if export_path is not None:
    learn = load_learner(fname=export_path, cpu=False, pickle_module=dill)
else:
    train_df = pd.read_csv(TRAIN_CSV)

    train_df["image_path"] = (
        TRAIN_IMG_DIR / train_df["chain"].astype(str) / train_df["image"]
    ).astype(str)

    exists_mask = (
        pd.Series(train_df["image_path"].to_numpy()).apply(os.path.exists).to_numpy()
    )
    train_df = train_df.loc[exists_mask].reset_index(drop=True)

    set_seed(42, reproducible=True)

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=ColReader("image_path"),
        get_y=ColReader("hotel_id"),
        splitter=RandomSplitter(valid_pct=0.1, seed=42),
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(size=224, min_scale=0.75),
    )
    dls = dblock.dataloaders(train_df, bs=32, num_workers=2)

    learn = vision_learner(dls, densenet161, metrics=accuracy)
    learn.fine_tune(1)



## === cell 5
torch.backends.cudnn.benchmark = True

cpu_cnt = os.cpu_count() or 2
num_workers = min(8, max(2, cpu_cnt))

test_dl = learn.dls.test_dl(
    test_items,
    bs=256,  # fewer iterations; does not change predictions, only batching
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)



## === cell 6
set_seed(42, reproducible=True)


@torch.inference_mode()
def tta_probs_fastai(learn, dl, n=4):
    learn.model.eval()
    preds, _ = learn.tta(dl=dl, n=n, beta=0.0)
    return preds


probs = tta_probs_fastai(learn, test_dl, n=4)
probs.shape



## === cell 7
preds_idx = probs.topk(5, dim=1).indices
preds_idx.shape



## === cell 8
vocab = learn.dls.vocab
vocab_str = np.asarray([str(v) for v in vocab], dtype=object)

idx_np = preds_idx.cpu().numpy()
preds = [" ".join(vocab_str[row]) for row in idx_np]

preds[:3], len(preds)



## === cell 9
submission["hotel_id"] = preds
submission.to_csv("submission.csv", index=False)

submission.head(), Path("submission.csv").exists()
