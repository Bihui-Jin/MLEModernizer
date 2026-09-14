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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.1403747355696585

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import shutil
import numpy as np
import pandas as pd

from fastai.vision.all import *
from fastai.vision.core import *

train = False

set_seed(42, reproducible=True)



## === cell 1
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

model_loc = Path("~/.torch").expanduser()
model_loc.mkdir(exist_ok=True, parents=True)



## === cell 2
Path("/root/.cache/torch/hub/checkpoints/").mkdir(exist_ok=True, parents=True)

src_dir = Path("../input/cassava-modelresnet34fine-tune10pkl")
pairs = [
    (
        src_dir / "resnet34-333f7ec4.pth",
        Path("/root/.cache/torch/hub/checkpoints/resnet34-333f7ec4.pth"),
    ),
    (
        src_dir / "resnet50-19c8e357.pth",
        Path("/root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth"),
    ),
]
for src, dst in pairs:
    if src.exists():
        shutil.copy(str(src), str(dst))



## === cell 3
import fastai

fastai.__version__



## === cell 4
path = Path("../input/cassava-leaf-disease-classification")



## === cell 5
labels = pd.read_csv(path / "train.csv")
labels["image_id"] = "train_images/" + labels["image_id"].astype(str)



## === cell 6
_ncpu = os.cpu_count() or 4

_tr_workers = min(8, max(2, _ncpu // 2))
_tr_prefetch = 2
_tr_dl_kwargs = dict(
    num_workers=_tr_workers,
    persistent_workers=(_tr_workers > 0),
    pin_memory=True,
    prefetch_factor=_tr_prefetch,
)

_inf_workers = min(16, max(4, _ncpu))  # allow more parallel decode for TTA
_inf_prefetch = 4
_inf_dl_kwargs = dict(
    num_workers=_inf_workers,
    persistent_workers=(_inf_workers > 0),
    pin_memory=True,
    prefetch_factor=_inf_prefetch,
)

if train:
    dls = ImageDataLoaders.from_df(
        labels,
        path=path,
        bs=64,
        fn_col=0,
        label_col=1,
        seed=42,
        valid_pct=0.2,
        item_tfms=RandomResizedCrop(460, min_scale=0.75, ratio=(1.0, 1.0)),
        batch_tfms=aug_transforms(),
        **_tr_dl_kwargs,
    )
else:
    dls = ImageDataLoaders.from_df(
        labels.iloc[:32].copy(),
        path=path,
        bs=2,
        fn_col=0,
        label_col=1,
        seed=42,
        valid_pct=0.2,
        item_tfms=Resize(460, method=ResizeMethod.Pad, pad_mode=PadMode.Reflection),
        batch_tfms=None,
        **_tr_dl_kwargs,
    )



## === cell 7
learn = cnn_learner(dls, resnet50, metrics=[error_rate, accuracy]).to_fp16()



## === cell 8
if train:
    learn.fine_tune(10, cbs=[MixUp(0.5)])



## === cell 9
model_name = "resnet50-fine_tune-10"
pkl_in = Path(
    f"{Path(os.getcwd()).parent}/input/cassava-modelresnet34fine-tune10pkl/{model_name}.pkl"
)
pkl_out = Path(
    f"{Path(os.getcwd()).parent}/output/cassava-leaf-disease-classification/{model_name}.pkl"
)

if train:
    pkl_out.parent.mkdir(exist_ok=True, parents=True)
    learn.export(pkl_out)
else:
    if pkl_in.exists():
        learn = load_learner(pkl_in, cpu=False)
    else:
        learn = cnn_learner(dls, resnet50, metrics=[error_rate, accuracy]).to_fp16()
        learn.fine_tune(10, cbs=[MixUp(0.5)])
        pkl_out.parent.mkdir(exist_ok=True, parents=True)
        learn.export(pkl_out)

learn = learn.to_fp16()



## === cell 10
if train:
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()



## === cell 11
if train:
    from fastai.vision.widgets import *

    cleaner = ImageClassifierCleaner(learn)
    cleaner



## === cell 12
test_files = get_image_files(path / "test_images")
test_files = sorted(test_files, key=lambda p: p.name)
sorted_test_names = [p.name for p in test_files]



## === cell 13
set_seed(42, reproducible=True)

test_bs = 512

test_item_tfms = Resize(460, method=ResizeMethod.Pad, pad_mode=PadMode.Reflection)
test_batch_tfms = None

test_dl = learn.dls.test_dl(
    test_files,
    with_labels=False,
    bs=test_bs,
    item_tfms=test_item_tfms,
    batch_tfms=test_batch_tfms,
    **_inf_dl_kwargs,
)

if torch.cuda.is_available():
    learn.model = learn.model.to(memory_format=torch.channels_last)

learn.model.eval()

with torch.inference_mode():
    preds_tta, _ = learn.tta(dl=test_dl, n=8, beta=0.0, use_max=False, decoded=False)

test_predictions = preds_tta.argmax(dim=1).cpu().numpy().astype(int)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4028985438.py in <cell line: 0>()
     27 
     28 with torch.inference_mode():
---> 29     preds_tta, _ = learn.tta(dl=test_dl, n=8, beta=0.0, use_max=False, decoded=False)
     30 
     31 test_predictions = preds_tta.argmax(dim=1).cpu().numpy().astype(int)

TypeError: Learner.tta() got an unexpected keyword argument 'decoded'

## === cell 14
submission = pd.read_csv(path / "sample_submission.csv")
submission = submission.sort_values("image_id", kind="mergesort").reset_index(drop=True)

assert (
    list(submission["image_id"]) == sorted_test_names
), "Mismatch between test file order and submission order."

submission["label"] = test_predictions.astype(int)
submission = submission[["image_id", "label"]]

submission_path = Path("submission.csv")
submission.to_csv(submission_path, index=False)
submission_path

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1471258146.py in <cell line: 0>()
      6 ), "Mismatch between test file order and submission order."
      7 
----> 8 submission["label"] = test_predictions.astype(int)
      9 submission = submission[["image_id", "label"]]
     10 

NameError: name 'test_predictions' is not defined
