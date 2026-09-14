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
labels["image_path"] = (
    labels["image_id"].astype(str).map(lambda x: path / "train_images" / x)
)

_missing = [p for p in labels["image_path"].head(50).tolist() if not Path(p).exists()]
assert (
    len(_missing) == 0
), f"Some training images not found under {path}: e.g. {_missing[:3]}"
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
    get_x=ColReader("image_path"),
    get_y=ColReader("label"),
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=RandomResizedCrop(460, min_scale=0.75, ratio=(1.0, 1.0)),
    batch_tfms=aug_transforms(),
)

dls = dblock.dataloaders(labels, bs=64, **dl_kwargs)
dls.valid_ds.items[:3]



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_56/214778091.py in <cell line: 0>()
     33 )
     34 
---> 35 dls = dblock.dataloaders(labels, bs=64, **dl_kwargs)
     36 dls.valid_ds.items[:3]
     37 

/usr/local/lib/python3.11/dist-packages/fastai/data/block.py in dataloaders(self, source, path, verbose, **kwargs)
    157         dsets = self.datasets(source, verbose=verbose)
    158         kwargs = {**self.dls_kwargs, **kwargs, 'verbose': verbose}
--> 159         return dsets.dataloaders(path=path, after_item=self.item_tfms, after_batch=self.batch_tfms, **kwargs)
    160 
    161     _docs = dict(new="Create a new `DataBlock` with other `item_tfms` and `batch_tfms`",

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in dataloaders(self, bs, shuffle_train, shuffle, val_shuffle, n, path, dl_type, dl_kwargs, device, drop_last, val_bs, **kwargs)
    331         dl = dl_type(self.subset(0), **merge(kwargs,def_kwargs, dl_kwargs[0]))
    332         def_kwargs = {'bs':bs if val_bs is None else val_bs,'shuffle':val_shuffle,'n':None,'drop_last':False}
--> 333         dls = [dl] + [dl.new(self.subset(i), **merge(kwargs,def_kwargs,val_kwargs,dl_kwargs[i]))
    334                       for i in range(1, self.n_subsets)]
    335         return self._dbunch_type(*dls, path=path, device=device)

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in <listcomp>(.0)
    331         dl = dl_type(self.subset(0), **merge(kwargs,def_kwargs, dl_kwargs[0]))
    332         def_kwargs = {'bs':bs if val_bs is None else val_bs,'shuffle':val_shuffle,'n':None,'drop_last':False}
--> 333         dls = [dl] + [dl.new(self.subset(i), **merge(kwargs,def_kwargs,val_kwargs,dl_kwargs[i]))
    334                       for i in range(1, self.n_subsets)]
    335         return self._dbunch_type(*dls, path=path, device=device)

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in new(self, dataset, cls, **kwargs)
    102         if not hasattr(self, '_n_inp') or not hasattr(self, '_types'):
    103             try:
--> 104                 self._one_pass()
    105                 res._n_inp,res._types = self._n_inp,self._types
    106             except Exception as e:

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in _one_pass(self)
     85         b = self.do_batch([self.do_item(None)])
     86         if self.device is not None: b = to_device(b, self.device)
---> 87         its = self.after_batch(b)
     88         self._n_inp = 1 if not isinstance(its, (list,tuple)) or len(its)==1 else len(its)-1
     89         self._types = explode_types(its)

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in __call__(self, o)
    246         self.fs = self.fs.sorted(key='order')
    247 
--> 248     def __call__(self, o): return compose_tfms(o, tfms=self.fs, split_idx=self.split_idx)
    249     def __repr__(self): return f"Pipeline: {' -> '.join([f.name for f in self.fs if f.name != 'noop'])}"
    250     def __getitem__(self,i): return self.fs[i]

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in compose_tfms(x, tfms, is_enc, reverse, **kwargs)
    195     for f in tfms:
    196         if not is_enc: f = f.decode
--> 197         x = f(x, **kwargs)
    198     return x
    199 

/usr/local/lib/python3.11/dist-packages/fastai/vision/augment.py in __call__(self, b, split_idx, **kwargs)
     48         **kwargs
     49     ):
---> 50         self.before_call(b, split_idx=split_idx)
     51         return super().__call__(b, split_idx=split_idx, **kwargs) if self.do else b
     52 

/usr/local/lib/python3.11/dist-packages/fastai/vision/augment.py in before_call(self, b, split_idx)
    479         while isinstance(b, tuple): b = b[0]
    480         self.split_idx = split_idx
--> 481         self.do,self.mat = True,self._get_affine_mat(b)
    482         for t in self.coord_fs: t.before_call(b)
    483 

/usr/local/lib/python3.11/dist-packages/fastai/vision/augment.py in _get_affine_mat(self, x)
    492         aff_m = _init_mat(x)
    493         if self.split_idx: return _prepare_mat(x, aff_m)
--> 494         ms = [f(x) for f in self.aff_fs]
    495         ms = [m for m in ms if m is not None]
    496         for m in ms: aff_m = aff_m @ m

/usr/local/lib/python3.11/dist-packages/fastai/vision/augment.py in <listcomp>(.0)
    492         aff_m = _init_mat(x)
    493         if self.split_idx: return _prepare_mat(x, aff_m)
--> 494         ms = [f(x) for f in self.aff_fs]
    495         ms = [m for m in ms if m is not None]
    496         for m in ms: aff_m = aff_m @ m

/usr/local/lib/python3.11/dist-packages/fastai/vision/augment.py in rotate_mat(x, max_deg, p, draw, batch)
    723     def _def_draw(x):   return x.new_empty(x.size(0)).uniform_(-max_deg, max_deg)
    724     def _def_draw_b(x): return x.new_zeros(x.size(0)) + random.uniform(-max_deg, max_deg)
--> 725     thetas = _draw_mask(x, _def_draw_b if batch else _def_draw, draw=draw, p=p, batch=batch) * math.pi/180
    726     return affine_mat(thetas.cos(), thetas.sin(), t0(thetas),
    727                      -thetas.sin(), thetas.cos(), t0(thetas))

/usr/local/lib/python3.11/dist-packages/fastai/vision/augment.py in _draw_mask(x, def_draw, draw, p, neutral, batch)
    569     "Creates mask_tensor based on `x` with `neutral` with probability `1-p`. "
    570     if draw is None: draw=def_draw
--> 571     if callable(draw): res=draw(x)
    572     elif is_listy(draw):
    573         assert len(draw)>=x.size(0)

/usr/local/lib/python3.11/dist-packages/fastai/vision/augment.py in _def_draw(x)
    721 ):
    722     "Return a random rotation matrix with `max_deg` and `p`"
--> 723     def _def_draw(x):   return x.new_empty(x.size(0)).uniform_(-max_deg, max_deg)
    724     def _def_draw_b(x): return x.new_zeros(x.size(0)) + random.uniform(-max_deg, max_deg)
    725     thetas = _draw_mask(x, _def_draw_b if batch else _def_draw, draw=draw, p=p, batch=batch) * math.pi/180

/usr/local/lib/python3.11/dist-packages/fastai/torch_core.py in __torch_function__(cls, func, types, args, kwargs)
    382         if cls.debug and func.__name__ not in ('__str__','__repr__'): print(func, types, args, kwargs)
    383         if _torch_handled(args, cls._opt, func): types = (torch.Tensor,)
--> 384         res = super().__torch_function__(func, types, args, ifnone(kwargs, {}))
    385         dict_objs = _find_args(args) if args else _find_args(list(kwargs.values()))
    386         if issubclass(type(res),TensorBase) and dict_objs: res.set_meta(dict_objs[0],as_copy=True)

/usr/local/lib/python3.11/dist-packages/torch/_tensor.py in __torch_function__(cls, func, types, args, kwargs)
   1646 
   1647         with _C.DisableTorchFunctionSubclass():
-> 1648             ret = func(*args, **kwargs)
   1649             if func in get_default_nowrap_functions():
   1650                 return ret

RuntimeError: "check_uniform_bounds" not implemented for 'Byte'

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
test_df["image_path"] = (
    test_df["image_id"].astype(str).map(lambda x: path / "test_images" / x)
)
test_df.head()



## === cell 12
test_files = test_df["image_path"].tolist()
_missing_test = [str(p) for p in test_files[:50] if not Path(p).exists()]
assert len(_missing_test) == 0, f"Some test images not found: e.g. {_missing_test[:3]}"

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
/tmp/ipykernel_56/1457043652.py in <cell line: 0>()
      3 assert len(_missing_test) == 0, f"Some test images not found: e.g. {_missing_test[:3]}"
      4 
----> 5 test_dl = learn.dls.test_dl(
      6     test_files,
      7     with_labels=False,

NameError: name 'learn' is not defined

## === cell 13
submission = test_df[["image_id"]].copy()

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
/tmp/ipykernel_56/967488711.py in <cell line: 0>()
      2 
      3 assert len(submission) == len(
----> 4     test_predictions
      5 ), f"Prediction length {len(test_predictions)} != submission length {len(submission)}"
      6 

NameError: name 'test_predictions' is not defined
