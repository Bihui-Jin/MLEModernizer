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

0.26345

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'The timeout is dominated by accidentally training the full ResNet50 for 10 epochs when the exported `.pkl` isn’t present (common on first run), plus expensive 512px RandomResizedCrop/aug transforms. To preserve core logic and accuracy while fitting in 600s, the main speed win is to reliably load pretrained weights (so training is avoided unless explicitly requested) and to reduce overhead in inference by disabling unnecessary augmentations for the test dataloader and using the learner’s native `predict` pipeline efficiently. I also make the dataset path resolution deterministic and add a fast “fail early” message if the model export is missing but `train=False`, rather than silently triggering a long training run. All changes are correctness-preserving (no architecture/loss/loop changes; no approximation/sampling/early stopping).'
- What this solution (achieved 0.26345) has done: 'I make the pipeline run end-to-end by (1) removing the hard failure when the exported `.pkl` is missing and instead loading ImageNet pretrained weights without retraining (preserves the core model/logic and avoids timeouts), and (2) fixing the test dataloader so it uses the same normalization/type pipeline as training to prevent the `ByteTensor` vs `FloatTensor` runtime error. I keep the architecture/training code intact and only adjust the inference dataloader construction to be compatible with fastai v2.8.5. These fixes should also improve accuracy versus the current broken run because they enable a valid submission and ensure proper input preprocessing during inference.'

# 9. Code solution

## === cell 0
import os
import shutil
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from IPython.core.interactiveshell import InteractiveShell

import fastai
from fastai.vision.all import *
from fastai.vision.core import *

InteractiveShell.ast_node_interactivity = "all"

train = False

set_seed(42, reproducible=True)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    n_cpu = os.cpu_count() or 2
    n_threads = min(4, max(1, n_cpu // 2))
    torch.set_num_threads(n_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(n_threads))
except Exception:
    pass



## === cell 1
DATA_CANDIDATES = [
    Path("/kaggle/input/cassava-leaf-disease-classification"),
    Path("/kaggle/data/cassava-leaf-disease-classification"),
    Path("/kaggle/data/input/cassava-leaf-disease-classification"),
]
path = next((p for p in DATA_CANDIDATES if p.exists()), None)
if path is None:
    raise FileNotFoundError(
        f"Could not find cassava dataset in any of: {DATA_CANDIDATES}"
    )
path



## === cell 2
model_loc = Path("/root/.cache/torch/hub/checkpoints/")
model_loc.mkdir(exist_ok=True, parents=True)
model_loc.ls()



## === cell 3
missing_weight_sources = [
    f"{Path(os.getcwd()).parent}/input/cassava-modelresnet34fine-tune10pkl/resnet34-333f7ec4.pth",
    f"{Path(os.getcwd()).parent}/input/cassava-modelresnet34fine-tune10pkl/resnet50-19c8e357.pth",
]
for src in missing_weight_sources:
    if Path(src).exists():
        dst = model_loc / Path(src).name
        if not dst.exists():
            shutil.copy(src, dst)



## === cell 4
fastai.__version__



## === cell 5
labels = pd.read_csv(path / "train.csv")
labels["image_id"] = "train_images/" + labels["image_id"].astype(str)
labels.head()



## === cell 6
n_cpu = os.cpu_count() or 2
n_workers = min(4, max(2, n_cpu // 2))
if n_workers < 0:
    n_workers = 0

dls = ImageDataLoaders.from_df(
    labels,
    path=path,
    bs=64,
    fn_col=0,
    label_col=1,
    seed=42,
    valid_pct=0.2,
    item_tfms=RandomResizedCrop(512, min_scale=0.75, ratio=(1.0, 1.0)),
    batch_tfms=aug_transforms(),
    num_workers=n_workers,
    persistent_workers=(n_workers > 0),
    prefetch_factor=2 if n_workers > 0 else None,
    pin_memory=torch.cuda.is_available(),
)

dls.valid_ds.items[:3]



## === cell 7
learn = cnn_learner(dls, resnet50, metrics=[error_rate, accuracy])
learn



## === cell 8
if train:
    with learn.no_bar(), learn.no_logging():
        learn.fine_tune(10, cbs=[MixUp(0.5)])



## === cell 9
model_name = "resnet50-fine_tune-10_resize-512"
export_path = Path("/kaggle/working") / f"{model_name}.pkl"

if train:
    learn.export(export_path)
else:
    if export_path.exists():
        learn = load_learner(export_path, cpu=not torch.cuda.is_available())
    else:
        print(
            f"Warning: exported model not found at {export_path}. "
            f"Proceeding with ImageNet-pretrained resnet50 head (no fine-tune)."
        )

learn



## === cell 10
if train:
    pass



## === cell 11
sample_sub = pd.read_csv(path / "sample_submission.csv")

test_df = sample_sub.copy()
test_df["image_id"] = "test_images/" + test_df["image_id"].astype(str)
test_df.head()



## === cell 12
test_item_tfms = Resize(512, method=ResizeMethod.Squish)

test_dl = dls.test_dl(
    test_df,
    with_labels=False,
    item_tfms=test_item_tfms,
    num_workers=n_workers,
    persistent_workers=(n_workers > 0),
    prefetch_factor=2 if n_workers > 0 else None,
    pin_memory=torch.cuda.is_available(),
)
test_dl.items[:5]



## === cell 13
with learn.no_bar(), learn.no_logging(), torch.inference_mode():
    probs, _ = learn.get_preds(dl=test_dl)
pred_labels = probs.argmax(dim=1).cpu().numpy().astype(np.int64)
pred_labels[:10], probs.shape



## === cell 14
submission = sample_sub
if len(submission) != len(pred_labels):
    raise ValueError(
        f"Prediction length mismatch: submission rows={len(submission)} preds={len(pred_labels)}"
    )

submission = submission.copy()
submission["label"] = pred_labels
submission.head()



## === cell 15
out_path = Path("/kaggle/working/submission.csv")
submission.to_csv(out_path, index=False)
out_path, out_path.exists(), out_path.stat().st_size
