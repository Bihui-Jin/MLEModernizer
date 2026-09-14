# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.5676538688220922

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

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
test = submission  # same object is fine; we only read "image" and later overwrite submission["hotel_id"]

test_items = (TEST_IMG_DIR.as_posix() + "/" + test["image"].astype(str)).to_list()
test_items[:3], len(test_items)



## === cell 4
EXPORT_CANDIDATES = [
    Path("../input/hotel-train-fastai-densnet161/export_dn161_kaggle_notebook.pkl"),
    Path(
        "/kaggle/input/hotel-train-fastai-densnet161/export_dn161_kaggle_notebook.pkl"
    ),
    Path(
        "../kaggle/input/hotel-train-fastai-densnet161/export_dn161_kaggle_notebook.pkl"
    ),
]

export_path = None
for p in EXPORT_CANDIDATES:
    if p.exists():
        export_path = p
        break

if export_path is None:
    raise FileNotFoundError(
        "Missing exported fastai learner (export_dn161_kaggle_notebook.pkl). "
        "Training fallback removed to ensure the notebook finishes within the 600s timeout."
    )

learn = load_learner(fname=export_path, cpu=False, pickle_module=dill)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2341722423.py in <cell line: 0>()
     19 # Failing fast here prevents an inevitable timeout while keeping the same core model/prediction logic.
     20 if export_path is None:
---> 21     raise FileNotFoundError(
     22         "Missing exported fastai learner (export_dn161_kaggle_notebook.pkl). "
     23         "Training fallback removed to ensure the notebook finishes within the 600s timeout."

FileNotFoundError: Missing exported fastai learner (export_dn161_kaggle_notebook.pkl). Training fallback removed to ensure the notebook finishes within the 600s timeout.

## === cell 5
cpu_cnt = os.cpu_count() or 2
torch.set_num_threads(max(1, min(2, cpu_cnt)))
torch.backends.cudnn.benchmark = True

num_workers = 2 if cpu_cnt >= 4 else 0

test_dl = learn.dls.test_dl(
    test_items,
    bs=256,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1301047018.py in <cell line: 0>()
      8 num_workers = 2 if cpu_cnt >= 4 else 0
      9 
---> 10 test_dl = learn.dls.test_dl(
     11     test_items,
     12     bs=256,

NameError: name 'learn' is not defined

## === cell 6
set_seed(42, reproducible=True)

learn.model.eval()
with torch.inference_mode():
    raw_logits, _ = learn.get_preds(
        dl=test_dl, with_decoded=False, with_input=False, act=None
    )
    probs = torch.softmax(raw_logits, dim=1).to(dtype=torch.float32, device="cpu")

probs.shape



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3305395165.py in <cell line: 0>()
      1 set_seed(42, reproducible=True)
      2 
----> 3 learn.model.eval()
      4 # Speed: inference_mode is already optimal; keep act=None to get raw logits exactly as before.
      5 with torch.inference_mode():

NameError: name 'learn' is not defined

## === cell 7
preds_idx = probs.topk(5, dim=1).indices
preds_idx.shape



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4093513863.py in <cell line: 0>()
      1 # Speed: compute topk directly from probs on CPU; identical semantics.
----> 2 preds_idx = probs.topk(5, dim=1).indices
      3 preds_idx.shape
      4 

NameError: name 'probs' is not defined

## === cell 8
vocab = learn.dls.vocab
vocab_str = np.asarray([str(v) for v in vocab], dtype=object)

idx_np = preds_idx.numpy()
preds = [" ".join(vocab_str[row]) for row in idx_np]

preds[:3], len(preds)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/543810692.py in <cell line: 0>()
      1 # Speed: pre-materialize vocab strings once; join per-row (still required to form submission strings).
----> 2 vocab = learn.dls.vocab
      3 vocab_str = np.asarray([str(v) for v in vocab], dtype=object)
      4 
      5 idx_np = preds_idx.numpy()

NameError: name 'learn' is not defined

## === cell 9
submission["hotel_id"] = preds
submission.to_csv("submission.csv", index=False)

submission.head(), Path("submission.csv").exists()

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3287091362.py in <cell line: 0>()
----> 1 submission["hotel_id"] = preds
      2 submission.to_csv("submission.csv", index=False)
      3 
      4 submission.head(), Path("submission.csv").exists()

NameError: name 'preds' is not defined
