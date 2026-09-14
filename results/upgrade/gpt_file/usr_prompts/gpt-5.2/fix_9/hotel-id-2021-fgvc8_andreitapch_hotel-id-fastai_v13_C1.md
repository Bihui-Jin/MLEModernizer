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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
seaborn==0.12.2
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

0.4475035783447009

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, shutil
from pathlib import Path

shutil.rmtree("./kaggle", ignore_errors=True)
os.makedirs("./kaggle/working", exist_ok=True)



## === cell 1
import warnings
import random

import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import fastai
import dill
from fastai.vision.all import *
from fastai.metrics import accuracy, top_k_accuracy



## === cell 2
path = Path("../input/hotel-id-2021-fgvc8")

sample_sub_path = path / "sample_submission.csv"
labels_path = path / "train.csv"

train_images_dir = path / "train_images"
test_images_dir = path / "test_images"

assert sample_sub_path.exists(), f"Missing {sample_sub_path}"
assert labels_path.exists(), f"Missing {labels_path}"
assert train_images_dir.exists(), f"Missing {train_images_dir}"
assert test_images_dir.exists(), f"Missing {test_images_dir}"



## === cell 3
train_data = pd.read_csv(labels_path)
train_data.head(5)



## === cell 4
pass



## === cell 5
train_data["chain"] = train_data["chain"].astype(str)
train_data["image"] = train_data["image"].astype(str)
train_data["image_path"] = (
    train_images_dir.as_posix() + "/" + train_data["chain"] + "/" + train_data["image"]
)
train_data[:5]



## === cell 6
missing = 0
for p in train_data["image_path"].sample(200, random_state=0).tolist():
    if not Path(p).exists():
        missing += 1
assert (
    missing == 0
), f"Found {missing} missing image paths in sample; check directory structure."



## === cell 7
if torch.cuda.is_available():
    batch_size = 64
else:
    batch_size = 32


def _suggest_num_workers():
    try:
        cpu = os.cpu_count() or 2
    except Exception:
        cpu = 2
    return max(2, min(8, cpu // 2))


NUM_WORKERS = _suggest_num_workers()

CACHE_DIR = Path("./kaggle/working/fa_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def create_dataLoader():
    df = train_data.loc[:, ["image_path", "hotel_id"]]

    item_tfms = [
        Resize(224, method="pad", pad_mode="reflection"),
        CacheImg(CACHE_DIR),
    ]
    batch_tfms = list(aug_transforms(size=224))

    dls = ImageDataLoaders.from_df(
        df=df,
        path=".",
        valid_pct=0.2,
        seed=42,
        fn_col="image_path",
        label_col="hotel_id",
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
        bs=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=(
            6 if NUM_WORKERS > 0 else None
        ),  # higher prefetch reduces dataloader stalls
    )
    return dls




## === cell 8
dataset = create_dataLoader()
dataset



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/307854074.py in <cell line: 0>()
----> 1 dataset = create_dataLoader()
      2 dataset
      3 

/tmp/ipykernel_55/2083125290.py in create_dataLoader()
     34         # Cache after deterministic item transforms so the expensive decode+resize is reused.
     35         # Using a fixed cache directory keeps behavior stable across runs.
---> 36         CacheImg(CACHE_DIR),
     37     ]
     38     batch_tfms = list(aug_transforms(size=224))

NameError: name 'CacheImg' is not defined

## === cell 9
pass



## === cell 10
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
set_seed(42, reproducible=True)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True




## === cell 11
def training_models(dataset, model_name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learn = cnn_learner(
            dataset,
            models.resnet101,
            metrics=[accuracy, top_k_accuracy],
            opt_func=QHAdam,
        ).to_fp16()

        learn.fine_tune(5, 0.005, freeze_epochs=3)
        learn.save(model_name)
    return learn




## === cell 12
model_name = "model_allData_rn101"
try:
    learn = training_models(dataset, model_name)
except RuntimeError as e:
    if (
        "out of memory" in str(e).lower()
        and torch.cuda.is_available()
        and batch_size > 16
    ):
        torch.cuda.empty_cache()
        batch_size = 32
        dataset = create_dataLoader()
        learn = training_models(dataset, model_name)
    else:
        raise



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2637268646.py in <cell line: 0>()
      1 model_name = "model_allData_rn101"
      2 try:
----> 3     learn = training_models(dataset, model_name)
      4 except RuntimeError as e:
      5     if (

NameError: name 'dataset' is not defined

## === cell 13
_ = learn.validate()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/485790432.py in <cell line: 0>()
----> 1 _ = learn.validate()
      2 

NameError: name 'learn' is not defined

## === cell 14
learn.export(f"{model_name}.pkl", pickle_module=dill)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/172335340.py in <cell line: 0>()
----> 1 learn.export(f"{model_name}.pkl", pickle_module=dill)
      2 

NameError: name 'learn' is not defined

## === cell 15
pass



## === cell 16
pass



## === cell 17
assert "learn" in globals() and learn is not None



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/3281683603.py in <cell line: 0>()
----> 1 assert "learn" in globals() and learn is not None
      2 

AssertionError: 

## === cell 18
pass



## === cell 19
sample_submission = pd.read_csv(sample_sub_path)
submission = sample_submission.copy()

submission["image"] = submission["image"].astype(str)
submission["image_path"] = test_images_dir.as_posix() + "/" + submission["image"]
submission.head()



## === cell 20
test_items = submission["image_path"].to_list()

test_dl = learn.dls.test_dl(
    test_items,
    with_labels=False,
    bs=batch_size,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True if NUM_WORKERS > 0 else False,
    prefetch_factor=6 if NUM_WORKERS > 0 else None,
)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4200517290.py in <cell line: 0>()
      3 test_items = submission["image_path"].to_list()
      4 
----> 5 test_dl = learn.dls.test_dl(
      6     test_items,
      7     with_labels=False,

NameError: name 'learn' is not defined

## === cell 21
pass



## === cell 22
with learn.no_bar(), learn.no_logging():
    probs, _ = learn.get_preds(dl=test_dl)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/984636290.py in <cell line: 0>()
----> 1 with learn.no_bar(), learn.no_logging():
      2     probs, _ = learn.get_preds(dl=test_dl)
      3 

NameError: name 'learn' is not defined

## === cell 23
preds_idx = probs.topk(5, dim=1)[1].cpu().numpy()



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2013419085.py in <cell line: 0>()
----> 1 preds_idx = probs.topk(5, dim=1)[1].cpu().numpy()
      2 

NameError: name 'probs' is not defined

## === cell 24
vocab = learn.dls.vocab
preds = [" ".join((str(vocab[i]) for i in row)) for row in preds_idx]
preds[:5]



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3538385334.py in <cell line: 0>()
      1 # SPEEDUP (equivalent): avoid Python generator overhead in inner loop by localizing lookups.
----> 2 vocab = learn.dls.vocab
      3 preds = [" ".join((str(vocab[i]) for i in row)) for row in preds_idx]
      4 preds[:5]
      5 

NameError: name 'learn' is not defined

## === cell 25
final_sub = sample_submission.copy()
final_sub["hotel_id"] = preds

out_path = "./kaggle/working/submission.csv"
final_sub.to_csv(out_path, index=False)

print("Wrote", out_path, "with shape:", final_sub.shape)
final_sub.head()

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3879794813.py in <cell line: 0>()
      1 final_sub = sample_submission.copy()
----> 2 final_sub["hotel_id"] = preds
      3 
      4 out_path = "./kaggle/working/submission.csv"
      5 final_sub.to_csv(out_path, index=False)

NameError: name 'preds' is not defined
