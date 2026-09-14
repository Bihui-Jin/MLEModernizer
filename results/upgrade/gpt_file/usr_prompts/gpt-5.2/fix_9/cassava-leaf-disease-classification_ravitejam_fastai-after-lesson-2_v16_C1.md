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

0.11584

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.11584) has done: 'The timeout is dominated by accidentally training the full ResNet50 for 10 epochs when the exported `.pkl` isn’t present (common on first run), plus expensive 512px RandomResizedCrop/aug transforms. To preserve core logic and accuracy while fitting in 600s, the main speed win is to reliably load pretrained weights (so training is avoided unless explicitly requested) and to reduce overhead in inference by disabling unnecessary augmentations for the test dataloader and using the learner’s native `predict` pipeline efficiently. I also make the dataset path resolution deterministic and add a fast “fail early” message if the model export is missing but `train=False`, rather than silently triggering a long training run. All changes are correctness-preserving (no architecture/loss/loop changes; no approximation/sampling/early stopping).'

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
        raise FileNotFoundError(
            f"Expected exported model at {export_path} but it does not exist. "
            f"To create it, set train=True (will fine_tune for 10 epochs as coded) and rerun."
        )

learn




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3848813538.py in <cell line: 0>()
     10         learn = load_learner(export_path, cpu=not torch.cuda.is_available())
     11     else:
---> 12         raise FileNotFoundError(
     13             f"Expected exported model at {export_path} but it does not exist. "
     14             f"To create it, set train=True (will fine_tune for 10 epochs as coded) and rerun."

FileNotFoundError: Expected exported model at /kaggle/working/resnet50-fine_tune-10_resize-512.pkl but it does not exist. To create it, set train=True (will fine_tune for 10 epochs as coded) and rerun.

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
    after_batch=[],
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




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1029131550.py in <cell line: 0>()
      1 # Optimization: inference_mode + no progress/log overhead. Uses identical prediction logic.
      2 with learn.no_bar(), learn.no_logging(), torch.inference_mode():
----> 3     probs, _ = learn.get_preds(dl=test_dl)
      4 pred_labels = probs.argmax(dim=1).cpu().numpy().astype(np.int64)
      5 pred_labels[:10], probs.shape

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in get_preds(self, ds_idx, dl, with_input, with_decoded, with_loss, act, inner, reorder, cbs, **kwargs)
    314         if with_loss: ctx_mgrs.append(self.loss_not_reduced())
    315         with ContextManagers(ctx_mgrs):
--> 316             self._do_epoch_validate(dl=dl)
    317             if act is None: act = getcallable(self.loss_func, 'activation')
    318             res = cb.all_tensors()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_epoch_validate(self, ds_idx, dl)
    250         if dl is None: dl = self.dls[ds_idx]
    251         self.dl = dl
--> 252         with torch.no_grad(): self._with_events(self.all_batches, 'validate', CancelValidException)
    253 
    254     def _do_epoch(self):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in all_batches(self)
    211     def all_batches(self):
    212         self.n_iter = len(self.dl)
--> 213         for o in enumerate(self.dl): self.one_batch(*o)
    214 
    215     def _backward(self): self.loss_grad.backward()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in one_batch(self, i, b)
    241         b = self._set_device(b)
    242         self._split(b)
--> 243         self._with_events(self._do_one_batch, 'batch', CancelBatchException)
    244 
    245     def _do_epoch_train(self):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_one_batch(self)
    222 
    223     def _do_one_batch(self):
--> 224         self.pred = self.model(*self.xb)
    225         self('after_pred')
    226         if len(self.yb):

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in forward(self, input)
    552 
    553     def forward(self, input: Tensor) -> Tensor:
--> 554         return self._conv_forward(input, self.weight, self.bias)
    555 
    556 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in _conv_forward(self, input, weight, bias)
    547                 self.groups,
    548             )
--> 549         return F.conv2d(
    550             input, weight, bias, self.stride, self.padding, self.dilation, self.groups
    551         )

RuntimeError: Input type (torch.cuda.ByteTensor) and weight type (torch.cuda.FloatTensor) should be the same

## === cell 14
submission = sample_sub
if len(submission) != len(pred_labels):
    raise ValueError(
        f"Prediction length mismatch: submission rows={len(submission)} preds={len(pred_labels)}"
    )

submission = submission.copy()
submission["label"] = pred_labels
submission.head()




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3035776985.py in <cell line: 0>()
      1 submission = sample_sub
----> 2 if len(submission) != len(pred_labels):
      3     raise ValueError(
      4         f"Prediction length mismatch: submission rows={len(submission)} preds={len(pred_labels)}"
      5     )

NameError: name 'pred_labels' is not defined

## === cell 15
out_path = Path("/kaggle/working/submission.csv")
submission.to_csv(out_path, index=False)
out_path, out_path.exists(), out_path.stat().st_size
