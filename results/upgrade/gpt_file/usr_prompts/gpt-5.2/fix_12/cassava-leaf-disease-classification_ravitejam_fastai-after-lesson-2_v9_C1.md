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

0.880930794802055

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

from matplotlib import pyplot as plt
from IPython.core.interactiveshell import InteractiveShell

from fastai.vision.all import *
from fastai.vision.core import *

InteractiveShell.ast_node_interactivity = "none"

train = True

set_seed(42, reproducible=True)

import torch

try:
    import multiprocessing as mp

    if mp.get_start_method(allow_none=True) != "fork":
        mp.set_start_method("fork", force=True)
except Exception:
    pass

torch.backends.cudnn.benchmark = True
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

try:
    torch.set_num_threads(min(8, os.cpu_count() or 2))
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    "torch:",
    torch.__version__,
    "| cuda available:",
    torch.cuda.is_available(),
    "| device:",
    device,
)



## === cell 1
model_loc = Path("~/.torch").expanduser()
model_loc.mkdir(exist_ok=True, parents=True)



## === cell 2
Path("/root/.cache/torch/hub/checkpoints/").mkdir(exist_ok=True, parents=True)



## === cell 3
import fastai

fastai.__version__



## === cell 4
path = Path("../input/cassava-leaf-disease-classification")
assert path.exists(), f"Dataset path not found: {path}"
os.listdir(path)[:10]



## === cell 5
labels = pd.read_csv(path / "train.csv")
labels["image_id"] = "train_images/" + labels["image_id"].astype(str)
labels.head()



## === cell 6
cpu = os.cpu_count() or 2
num_workers = min(8, max(2, cpu - 1))
use_persistent = bool(num_workers and num_workers > 0)

dl_kwargs = dict(
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),  # only beneficial on GPU
)
if use_persistent:
    dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))


class CachedPILImage(PILImage):
    _cache = {}

    @classmethod
    def create(cls, fn, **kwargs):
        fn = str(fn)
        img = cls._cache.get(fn, None)
        if img is None:
            img = super().create(fn, **kwargs)
            cls._cache[fn] = img
        return img


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
    dl_type=CachedPILImage,
    **dl_kwargs,
)
dls.valid_ds.items[:3]



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1921974470.py in <cell line: 0>()
     27 # Passing `cls=CachedPILImage` collides and raises "multiple values for argument 'cls'".
     28 # Correct way to override the image opener class is `dl_type=...`.
---> 29 dls = ImageDataLoaders.from_df(
     30     labels,
     31     path=path,

/usr/local/lib/python3.11/dist-packages/fastai/vision/data.py in from_df(cls, df, path, valid_pct, seed, fn_col, folder, suff, label_col, label_delim, y_block, valid_col, item_tfms, batch_tfms, img_cls, **kwargs)
    177                            item_tfms=item_tfms,
    178                            batch_tfms=batch_tfms)
--> 179         return cls.from_dblock(dblock, df, path=path, **kwargs)
    180 
    181     @classmethod

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in from_dblock(cls, dblock, source, path, bs, val_bs, shuffle, device, **kwargs)
    278         **kwargs
    279     ):
--> 280         return dblock.dataloaders(source, path=path, bs=bs, val_bs=val_bs, shuffle=shuffle, device=device, **kwargs)
    281 
    282     _docs=dict(__getitem__="Retrieve `DataLoader` at `i` (`0` is training, `1` is validation)",

/usr/local/lib/python3.11/dist-packages/fastai/data/block.py in dataloaders(self, source, path, verbose, **kwargs)
    157         dsets = self.datasets(source, verbose=verbose)
    158         kwargs = {**self.dls_kwargs, **kwargs, 'verbose': verbose}
--> 159         return dsets.dataloaders(path=path, after_item=self.item_tfms, after_batch=self.batch_tfms, **kwargs)
    160 
    161     _docs = dict(new="Create a new `DataBlock` with other `item_tfms` and `batch_tfms`",

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in dataloaders(self, bs, shuffle_train, shuffle, val_shuffle, n, path, dl_type, dl_kwargs, device, drop_last, val_bs, **kwargs)
    329         val_kwargs={k[4:]:v for k,v in kwargs.items() if k.startswith('val_')}
    330         def_kwargs = {'bs':bs,'shuffle':shuffle,'drop_last':drop_last,'n':n,'device':device}
--> 331         dl = dl_type(self.subset(0), **merge(kwargs,def_kwargs, dl_kwargs[0]))
    332         def_kwargs = {'bs':bs if val_bs is None else val_bs,'shuffle':val_shuffle,'n':None,'drop_last':False}
    333         dls = [dl] + [dl.new(self.subset(i), **merge(kwargs,def_kwargs,val_kwargs,dl_kwargs[i]))

/usr/local/lib/python3.11/dist-packages/fastcore/meta.py in __call__(cls, x, *args, **kwargs)
     63         if hasattr(cls, '_new_meta'): x = cls._new_meta(x, *args, **kwargs)
     64         elif not isinstance(x,getattr(cls,'_bypass_type',object)) or len(args) or len(kwargs):
---> 65             x = super().__call__(*((x,)+args), **kwargs)
     66         if cls!=x.__class__: x.__class__ = cls
     67         return x

TypeError: Image.__init__() got an unexpected keyword argument 'after_item'

## === cell 7
learn = cnn_learner(dls, resnet50, metrics=[error_rate, accuracy]).to_fp16()
learn



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2076868562.py in <cell line: 0>()
----> 1 learn = cnn_learner(dls, resnet50, metrics=[error_rate, accuracy]).to_fp16()
      2 learn
      3 

NameError: name 'dls' is not defined

## === cell 8
model_name = "resnet50-fine_tune-10"
export_path = Path(
    f"{Path(os.getcwd()).parent}/output/cassava-leaf-disease-classification/{model_name}.pkl"
)
export_path.parent.mkdir(exist_ok=True, parents=True)

if export_path.exists():
    train = False
    learn = load_learner(export_path)
    learn.model.eval()
else:
    if train:
        learn.fine_tune(10, cbs=[MixUp(0.5)])
        learn.export(export_path)

learn



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3723332421.py in <cell line: 0>()
     11 else:
     12     if train:
---> 13         learn.fine_tune(10, cbs=[MixUp(0.5)])
     14         learn.export(export_path)
     15 

NameError: name 'learn' is not defined

## === cell 9
if False and train:
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(6, 6))



## === cell 10
if False and train:
    from fastai.vision.widgets import *

    cleaner = ImageClassifierCleaner(learn)
    cleaner



## === cell 11
test_df = pd.read_csv(path / "sample_submission.csv")
test_df["image_id"] = "test_images/" + test_df["image_id"].astype(str)
test_df.head()



## === cell 12
test_files = (path / test_df["image_id"]).tolist()

test_dl = learn.dls.test_dl(
    test_files,
    with_labels=False,
    bs=256,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=use_persistent if num_workers > 0 else False,
    prefetch_factor=4 if use_persistent else None,
)

learn.model.eval()
preds, _ = learn.get_preds(dl=test_dl, with_decoded=False)
test_predictions = preds.argmax(dim=1).cpu().numpy().astype(np.int64)

len(test_predictions), test_predictions[:5]



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1012552666.py in <cell line: 0>()
      1 test_files = (path / test_df["image_id"]).tolist()
      2 
----> 3 test_dl = learn.dls.test_dl(
      4     test_files,
      5     with_labels=False,

NameError: name 'learn' is not defined

## === cell 13
submission = test_df[["image_id"]].copy()
submission["image_id"] = submission["image_id"].str.removeprefix("test_images/")

assert len(submission) == len(
    test_predictions
), f"Prediction length {len(test_predictions)} != submission length {len(submission)}"

submission["label"] = test_predictions
submission.head()

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Path:", Path("submission.csv").resolve())

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3668777471.py in <cell line: 0>()
      3 
      4 assert len(submission) == len(
----> 5     test_predictions
      6 ), f"Prediction length {len(test_predictions)} != submission length {len(submission)}"
      7 

NameError: name 'test_predictions' is not defined
