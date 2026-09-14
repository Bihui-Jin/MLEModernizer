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

0.5955733771154317

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import gc
from pathlib import Path

import numpy as np
import pandas as pd

import fastai
from fastai.vision.all import *

import dill



## === cell 1
fastai.__version__, torch.__version__



## === cell 2
BASE = Path("/kaggle")
INPUT = BASE / "input" / "hotel-id-2021-fgvc8"

models = [
    INPUT / "export_dn161_Fa_CE_bs32.pkl",
    INPUT / "export_dn161_Fa_FL_bs32.pkl",
    INPUT / "export_res101_Fall_HQAdam.pkl",
]



## === cell 3
BASE = Path("/kaggle")
INPUT = BASE / "input" / "hotel-id-2021-fgvc8"
WORKING = BASE / "working"

assert INPUT.exists(), f"Expected competition data at {INPUT} but it does not exist."

train_csv_path = INPUT / "train.csv"
sample_sub_path = INPUT / "sample_submission.csv"
train_img_root = INPUT / "train_images"
test_img_root = INPUT / "test_images"

assert train_csv_path.exists()
assert sample_sub_path.exists()
assert train_img_root.exists()
assert test_img_root.exists()

train_df = pd.read_csv(
    train_csv_path,
    dtype={
        "image": "string",
        "chain": "int32",
        "hotel_id": "string",
        "timestamp": "string",
    },
)
submission = pd.read_csv(
    sample_sub_path, dtype={"image": "string", "hotel_id": "string"}
)

train_df.head(), submission.head()



## === cell 4
test = submission.copy()
test["image_path"] = (
    test_img_root.as_posix() + "/" + test["image"].astype(str)
).astype(str)
test.head()



## === cell 5
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
set_seed(seed, reproducible=True)

torch.backends.cudnn.benchmark = True

try:
    ncpu = os.cpu_count() or 2
    torch.set_num_threads(max(1, min(8, ncpu)))
    torch.set_num_interop_threads(1)
except Exception:
    pass



## === cell 6
existing_model_files = [Path(m) for m in models if Path(m).exists()]
if len(existing_model_files) == 0:
    raise FileNotFoundError(
        "No exported model .pkl files were found at the expected paths:\n"
        + "\n".join(str(m) for m in models)
        + "\nThis notebook is designed to run inference from provided exported learners; "
        "training from scratch will exceed the 600s limit."
    )
existing_model_files



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_54/2766203405.py in <cell line: 0>()
      5 if len(existing_model_files) == 0:
      6     # Keep paths unchanged; just fail fast with a clear message rather than timing out training.
----> 7     raise FileNotFoundError(
      8         "No exported model .pkl files were found at the expected paths:\n"
      9         + "\n".join(str(m) for m in models)

FileNotFoundError: No exported model .pkl files were found at the expected paths:
/kaggle/input/hotel-id-2021-fgvc8/export_dn161_Fa_CE_bs32.pkl
/kaggle/input/hotel-id-2021-fgvc8/export_dn161_Fa_FL_bs32.pkl
/kaggle/input/hotel-id-2021-fgvc8/export_res101_Fall_HQAdam.pkl
This notebook is designed to run inference from provided exported learners; training from scratch will exceed the 600s limit.

## === cell 7
probs = None
learn = None
vocab = None
test_dl = None


def _predict_probs_fastai(learn, dl):
    learn.model.eval()
    with torch.inference_mode():
        preds, _ = learn.get_preds(
            dl=dl, with_decoded=False, with_input=False, reorder=False
        )
    return preds  # on device (usually GPU)


test_items = test["image_path"].tolist()

ncpu = os.cpu_count() or 2
num_workers = min(8, max(2, ncpu // 2))
pf = 2 if num_workers > 0 else None

first_model_path = existing_model_files[0]
learn = load_learner(fname=Path(first_model_path), cpu=False, pickle_module=dill)
vocab = np.asarray(learn.dls.vocab, dtype=object)

try:
    base_bs = int(getattr(learn.dls, "bs", 32))
except Exception:
    base_bs = 32

test_bs = int(min(max(base_bs, 256), 512))

test_dl = learn.dls.test_dl(
    test_items,
    with_labels=False,
    bs=test_bs,
    num_workers=num_workers,
    prefetch_factor=pf,
    persistent_workers=(num_workers > 0),
    pin_memory=True,
)
base_dls = learn.dls

probs = _predict_probs_fastai(learn, test_dl)
del learn
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

for model_path in existing_model_files[1:]:
    learn = load_learner(fname=Path(model_path), cpu=False, pickle_module=dill)
    learn.dls = base_dls
    probs_temp = _predict_probs_fastai(learn, test_dl)
    probs = probs + probs_temp  # accumulate on-device
    del probs_temp
    del learn
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

probs = probs / float(len(existing_model_files))
probs = probs.detach().to("cpu", dtype=torch.float32)

probs.shape



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_54/3073056204.py in <cell line: 0>()
     23 pf = 2 if num_workers > 0 else None
     24 
---> 25 first_model_path = existing_model_files[0]
     26 learn = load_learner(fname=Path(first_model_path), cpu=False, pickle_module=dill)
     27 vocab = np.asarray(learn.dls.vocab, dtype=object)

IndexError: list index out of range

## === cell 8
k = int(min(5, probs.shape[1]))
topk_idx = probs.topk(k, dim=1).indices
topk_idx_np = topk_idx.numpy()
topk_labels = np.take(vocab, topk_idx_np)

if k < 5:
    last_col = topk_labels[:, [-1]]
    pad = np.repeat(last_col, 5 - k, axis=1)
    topk_labels = np.concatenate([topk_labels, pad], axis=1)
else:
    topk_labels = topk_labels[:, :5]

preds = [" ".join(row) for row in topk_labels]
preds[:5], len(preds)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_54/671076690.py in <cell line: 0>()
----> 1 k = int(min(5, probs.shape[1]))
      2 topk_idx = probs.topk(k, dim=1).indices
      3 topk_idx_np = topk_idx.numpy()
      4 topk_labels = np.take(vocab, topk_idx_np)
      5 

AttributeError: 'NoneType' object has no attribute 'shape'

## === cell 9
assert len(preds) == len(
    submission
), f"Pred length {len(preds)} != submission length {len(submission)}"

submission_out = submission.copy()
submission_out["hotel_id"] = preds

out_path = WORKING / "submission.csv"
submission_out.to_csv(out_path, index=False)

out_path, submission_out.head()

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/4231598080.py in <cell line: 0>()
----> 1 assert len(preds) == len(
      2     submission
      3 ), f"Pred length {len(preds)} != submission length {len(submission)}"
      4 
      5 submission_out = submission.copy()

NameError: name 'preds' is not defined
