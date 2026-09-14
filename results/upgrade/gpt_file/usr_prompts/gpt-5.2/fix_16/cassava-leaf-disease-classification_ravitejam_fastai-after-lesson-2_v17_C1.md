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

0.8824418253248716

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
import math
import numpy as np
import pandas as pd

import torch
import fastai
from fastai.vision.all import *
from fastai.vision.core import *

try:
    from IPython.core.interactiveshell import InteractiveShell

    InteractiveShell.ast_node_interactivity = "last_expr"
except Exception:
    pass

train = True

set_seed(42, reproducible=True)
random.seed(42)
np.random.seed(42)

torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

path = Path("../input/cassava-leaf-disease-classification")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 1
_ = fastai.__version__
_ = (path.exists(), len(path.ls()))




## === cell 2
labels = pd.read_csv(path / "train.csv")
labels["image_id"] = "train_images/" + labels["image_id"].astype("string")
_ = (labels.head(1), labels.shape)




## === cell 3
n_cpus = os.cpu_count() or 2
if device.type == "cuda":
    num_workers = min(8, max(2, n_cpus // 2))
else:
    num_workers = min(4, max(2, n_cpus // 2))

train_tfrec_paths = sorted((path / "train_tfrecords").ls())
assert len(train_tfrec_paths) > 0, "No train TFRecords found"


def _cassava_tfrecord_labeler(fn: Path):
    return None


imgid_to_label = dict(
    zip(
        labels["image_id"].str.replace("train_images/", "", regex=False),
        labels["label"].astype(int),
    )
)

import tensorflow as tf

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _parse_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, FEATURE_DESC)
    img = tf.io.decode_jpeg(ex["image"], channels=3)
    name = ex["image_name"]
    return img, name


tfrecord_index = []
for tfp in train_tfrec_paths:
    ds = tf.data.TFRecordDataset([str(tfp)], num_parallel_reads=1)
    i = 0
    for raw in ds:
        ex = tf.io.parse_single_example(raw, FEATURE_DESC)
        name = ex["image_name"].numpy().decode("utf-8")
        tfrecord_index.append((str(tfp), i, name))
        i += 1

tfrecord_index = [t for t in tfrecord_index if t[2] in imgid_to_label]
assert len(tfrecord_index) == len(labels), (len(tfrecord_index), len(labels))

tidx = pd.DataFrame(
    tfrecord_index, columns=["tfrecord_path", "record_index", "image_name"]
)
tidx["label"] = tidx["image_name"].map(imgid_to_label).astype(int)


class TFRecordImage(Transform):
    def __init__(self):
        self._cache = {}  # cache TFRecordDataset objects per path (safe, deterministic)

    def encodes(self, o):
        tfp = o["tfrecord_path"] if isinstance(o, dict) else o.tfrecord_path
        ridx = int(o["record_index"] if isinstance(o, dict) else o.record_index)
        key = (tfp, os.getpid())
        st = self._cache.get(key)
        if st is None or st["cur"] > ridx:
            ds = tf.data.TFRecordDataset([tfp], num_parallel_reads=1)
            ds = ds.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
            it = iter(ds)
            st = {"it": it, "cur": 0, "path": tfp}
            self._cache[key] = st
        it = st["it"]
        cur = st["cur"]
        while cur < ridx:
            next(it)
            cur += 1
        img, _name = next(it)
        st["cur"] = cur + 1
        arr = img.numpy()
        return PILImage.create(arr)


def _get_x(r):
    return {"tfrecord_path": r["tfrecord_path"], "record_index": r["record_index"]}


def _get_y(r):
    return r["label"]


splitter = RandomSplitter(valid_pct=0.2, seed=42)

item_tfms = [
    TFRecordImage(),
    Resize(512, method=ResizeMethod.Crop, pad_mode=PadMode.Reflection),
]
batch_tfms = aug_transforms(size=512)
batch_tfms = batch_tfms + [Normalize.from_stats(*imagenet_stats)]

dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_x=_get_x,
    get_y=_get_y,
    splitter=splitter,
    item_tfms=item_tfms,
    batch_tfms=batch_tfms,
)

dls = dblock.dataloaders(
    tidx,
    path=path,
    bs=64,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(num_workers > 0),
)

_ = dls.valid_ds.items[:1]




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
learn = cnn_learner(dls, resnet50, metrics=[error_rate, accuracy])

if device.type == "cuda":
    learn.model = learn.model.to(memory_format=torch.channels_last)

learn = learn.to_fp16()
learn.cbs.append(ProgressCallback(show=False))

use_compile = os.environ.get("TORCH_COMPILE", "0") == "1"
if use_compile and hasattr(torch, "compile"):
    try:
        learn.model = torch.compile(learn.model)
    except Exception:
        pass

learn




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2309280964.py in <cell line: 0>()
      1 # Speed: keep architecture/training loop identical; enable channels_last + fp16 as before.
----> 2 learn = cnn_learner(dls, resnet50, metrics=[error_rate, accuracy])
      3 
      4 if device.type == "cuda":
      5     learn.model = learn.model.to(memory_format=torch.channels_last)

NameError: name 'dls' is not defined

## === cell 5
if train:
    torch.backends.cudnn.benchmark = True
    learn.fine_tune(10, cbs=[MixUp(0.5)])
    torch.backends.cudnn.benchmark = False




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4053961065.py in <cell line: 0>()
      1 if train:
      2     torch.backends.cudnn.benchmark = True
----> 3     learn.fine_tune(10, cbs=[MixUp(0.5)])
      4     torch.backends.cudnn.benchmark = False
      5 

NameError: name 'learn' is not defined

## === cell 6
learn = learn.to_fp32()
learn




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/284423823.py in <cell line: 0>()
----> 1 learn = learn.to_fp32()
      2 learn
      3 
      4 

NameError: name 'learn' is not defined

## === cell 7
test_df = pd.read_csv(path / "sample_submission.csv")
test_df["image_id"] = "test_images/" + test_df["image_id"].astype("string")
_ = (test_df.head(1), test_df.shape)




## === cell 8
test_files = [path / p for p in test_df["image_id"].tolist()]

test_dl = dls.test_dl(
    test_files,
    with_labels=False,
    bs=256,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(num_workers > 0),
)

learn.model.eval()
with torch.inference_mode():
    preds, _ = learn.get_preds(dl=test_dl, reorder=False)

test_predictions = preds.argmax(dim=1).cpu().numpy().astype(np.int64).tolist()
_ = (len(test_predictions), test_predictions[:5])




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2647392064.py in <cell line: 0>()
      3 test_files = [path / p for p in test_df["image_id"].tolist()]
      4 
----> 5 test_dl = dls.test_dl(
      6     test_files,
      7     with_labels=False,

NameError: name 'dls' is not defined

## === cell 9
submission = test_df.copy()
assert len(test_predictions) == len(submission), (
    len(test_predictions),
    len(submission),
)

submission["label"] = test_predictions
submission = submission[["image_id", "label"]]
submission.to_csv("submission.csv", index=False)

_ = (Path("submission.csv").exists(), pd.read_csv("submission.csv").head(1))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1201219231.py in <cell line: 0>()
      1 submission = test_df.copy()
----> 2 assert len(test_predictions) == len(submission), (
      3     len(test_predictions),
      4     len(submission),
      5 )

NameError: name 'test_predictions' is not defined
