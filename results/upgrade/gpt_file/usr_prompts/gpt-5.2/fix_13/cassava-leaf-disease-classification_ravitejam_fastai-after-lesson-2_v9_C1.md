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


dblock = DataBlock(
    blocks=(TransformBlock(type_tfms=CachedPILImage.create), CategoryBlock),
    get_x=ColReader("image_id"),
    get_y=ColReader("label"),
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=RandomResizedCrop(460, min_scale=0.75, ratio=(1.0, 1.0)),
    batch_tfms=aug_transforms(),
)

dls = dblock.dataloaders(labels, path=path, bs=64, **dl_kwargs)
dls.valid_ds.items[:3]



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_56/1457273305.py in <cell line: 0>()
     36 )
     37 
---> 38 dls = dblock.dataloaders(labels, path=path, bs=64, **dl_kwargs)
     39 dls.valid_ds.items[:3]
     40 

/usr/local/lib/python3.11/dist-packages/fastai/data/block.py in dataloaders(self, source, path, verbose, **kwargs)
    155         **kwargs
    156     ) -> DataLoaders:
--> 157         dsets = self.datasets(source, verbose=verbose)
    158         kwargs = {**self.dls_kwargs, **kwargs, 'verbose': verbose}
    159         return dsets.dataloaders(path=path, after_item=self.item_tfms, after_batch=self.batch_tfms, **kwargs)

/usr/local/lib/python3.11/dist-packages/fastai/data/block.py in datasets(self, source, verbose)
    147         splits = (self.splitter or RandomSplitter())(items)
    148         pv(f"{len(splits)} datasets of sizes {','.join([str(len(s)) for s in splits])}", verbose)
--> 149         return Datasets(items, tfms=self._combine_type_tfms(), splits=splits, dl_type=self.dl_type, n_inp=self.n_inp, verbose=verbose)
    150 
    151     def dataloaders(self, 

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in __init__(self, items, tfms, tls, n_inp, dl_type, **kwargs)
    448     ):
    449         super().__init__(dl_type=dl_type)
--> 450         self.tls = L(tls if tls else [TfmdLists(items, t, **kwargs) for t in L(ifnone(tfms,[None]))])
    451         self.n_inp = ifnone(n_inp, max(1, len(self.tls)-1))
    452 

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in <listcomp>(.0)
    448     ):
    449         super().__init__(dl_type=dl_type)
--> 450         self.tls = L(tls if tls else [TfmdLists(items, t, **kwargs) for t in L(ifnone(tfms,[None]))])
    451         self.n_inp = ifnone(n_inp, max(1, len(self.tls)-1))
    452 

/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py in __call__(cls, x, *args, **kwargs)
    103     def __call__(cls, x=None, *args, **kwargs):
    104         if not args and not kwargs and x is not None and isinstance(x,cls): return x
--> 105         return super().__call__(x, *args, **kwargs)
    106 
    107 # %% ../nbs/02_foundation.ipynb

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in __init__(self, items, tfms, use_list, do_setup, split_idx, train_setup, splits, types, verbose, dl_type)
    362         if do_setup:
    363             pv(f"Setting up {self.tfms}", verbose)
--> 364             self.setup(train_setup=train_setup)
    365 
    366     def _new(self, items, split_idx=None, **kwargs):

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in setup(self, train_setup)
    389             for f in self.tfms.fs:
    390                 self.types.append(getattr(f, 'input_types', type(x)))
--> 391                 x = f(x)
    392             self.types.append(type(x))
    393         types = L(t if is_listy(t) else [t] for t in self.types).concat().unique()

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in __call__(self, split_idx, *args, **kwargs)
    112         dec = len(self.decodes.methods) if hasattr(self, 'decodes') else 0
    113         return f'{self.name}(enc:{enc},dec:{dec})'
--> 114     def __call__(self,*args,split_idx=None, **kwargs): return self._call('encodes', *args, split_idx=split_idx, **kwargs)
    115     def decode(self, *args,split_idx=None, **kwargs): return self._call('decodes', *args, split_idx=split_idx, **kwargs)
    116     def setup(self, items=None, train_setup=False):

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in _call(self, nm, split_idx, *args, **kwargs)
    123         if split_idx!=self.split_idx and self.split_idx is not None: return args[0]
    124         if not hasattr(self, nm): return args[0]
--> 125         return self._do_call(nm, *args, **kwargs)
    126 
    127     def _do_call(self, nm, *args, **kwargs):

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in _do_call(self, nm, *args, **kwargs)
    134         try: method, ret_type = f._resolve_method_with_cache(f_args)
    135         except NotFoundLookupError: return x
--> 136         return retain_type(method(*f_args,**kwargs), x, ret_type)
    137 
    138 add_docs(Transform, decode="Delegate to decodes to undo transform", setup="Delegate to setups to set up transform")

/tmp/ipykernel_56/1457273305.py in create(cls, fn, **kwargs)
     22         img = cls._cache.get(fn, None)
     23         if img is None:
---> 24             img = super().create(fn, **kwargs)
     25             cls._cache[fn] = img
     26         return img

/usr/local/lib/python3.11/dist-packages/fastai/vision/core.py in create(cls, fn, **kwargs)
    125         if isinstance(fn,bytes): fn = io.BytesIO(fn)
    126         if isinstance(fn,Image.Image): return cls(fn)
--> 127         return cls(load_image(fn, **merge(cls._open_args, kwargs)))
    128 
    129     def show(self, ctx=None, **kwargs):

/usr/local/lib/python3.11/dist-packages/fastai/vision/core.py in load_image(fn, mode)
     98 def load_image(fn, mode=None):
     99     "Open and load a `PIL.Image` and convert to `mode`"
--> 100     im = Image.open(fn)
    101     im.load()
    102     im = im._new(im.im)

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

FileNotFoundError: [Errno 2] No such file or directory: 'train_images/488165307.jpg'

## === cell 7
learn = cnn_learner(dls, resnet50, metrics=[error_rate, accuracy]).to_fp16()
learn



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2076868562.py in <cell line: 0>()
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
/tmp/ipykernel_56/3723332421.py in <cell line: 0>()
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
/tmp/ipykernel_56/1012552666.py in <cell line: 0>()
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
/tmp/ipykernel_56/3668777471.py in <cell line: 0>()
      3 
      4 assert len(submission) == len(
----> 5     test_predictions
      6 ), f"Prediction length {len(test_predictions)} != submission length {len(submission)}"
      7 

NameError: name 'test_predictions' is not defined
