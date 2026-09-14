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


def _get_x(r):
    return path / r["image_id"]


def _get_y(r):
    return r["label"]


splitter = RandomSplitter(valid_pct=0.2, seed=42)

if device.type == "cuda" and "RandomResizedCropGPU" in globals():
    item_tfms = [RandomResizedCropGPU(512, min_scale=0.75, ratio=(1.0, 1.0))]
else:
    item_tfms = [RandomResizedCrop(512, min_scale=0.75, ratio=(1.0, 1.0))]

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

cache_dir = Path("/kaggle/working/fa_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

dls = dblock.dataloaders(
    labels,
    path=path,
    bs=64,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(num_workers > 0),
    prefetch_factor=(4 if num_workers > 0 else None),
    cache_dir=cache_dir,
)

_ = dls.valid_ds.items[:1]



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2741022782.py in <cell line: 0>()
     44 
     45 # CHANGE (timeout): bump prefetch_factor to keep GPU fed; keep persistent workers + pin_memory.
---> 46 dls = dblock.dataloaders(
     47     labels,
     48     path=path,

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
     83 
     84     def _one_pass(self):
---> 85         b = self.do_batch([self.do_item(None)])
     86         if self.device is not None: b = to_device(b, self.device)
     87         its = self.after_batch(b)

/usr/local/lib/python3.11/dist-packages/fastai/data/load.py in do_item(self, s)
    168     def prebatched(self): return self.bs is None
    169     def do_item(self, s):
--> 170         try: return self.after_item(self.create_item(s))
    171         except SkipItemException: return None
    172     def chunkify(self, b): return b if self.prebatched else chunked(b, self.bs, self.drop_last)

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
     49     ):
     50         self.before_call(b, split_idx=split_idx)
---> 51         return super().__call__(b, split_idx=split_idx, **kwargs) if self.do else b
     52 
     53 # %% ../../nbs/09_vision.augment.ipynb 14

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
    127     def _do_call(self, nm, *args, **kwargs):
    128         if _is_tuple(x:=args[0]):
--> 129             res = tuple(self._do_call(nm, x_, *args[1:], **kwargs) for x_ in x)
    130             return retain_type(res, x, Any)
    131         f = getattr(self,nm)

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in <genexpr>(.0)
    127     def _do_call(self, nm, *args, **kwargs):
    128         if _is_tuple(x:=args[0]):
--> 129             res = tuple(self._do_call(nm, x_, *args[1:], **kwargs) for x_ in x)
    130             return retain_type(res, x, Any)
    131         f = getattr(self,nm)

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in _do_call(self, nm, *args, **kwargs)
    134         try: method, ret_type = f._resolve_method_with_cache(f_args)
    135         except NotFoundLookupError: return x
--> 136         return retain_type(method(*f_args,**kwargs), x, ret_type)
    137 
    138 add_docs(Transform, decode="Delegate to decodes to undo transform", setup="Delegate to setups to set up transform")

/usr/local/lib/python3.11/dist-packages/fastai/vision/augment.py in encodes(self, x)
    546         return x.affine_coord(sz=self.size, mode=mode)
    547 
--> 548     def encodes(self, x:TensorImage|TensorPoint|TensorBBox): return self._encode(x, self.mode)
    549     def encodes(self, x:TensorMask):                         return self._encode(x, self.mode_mask)
    550 

/usr/local/lib/python3.11/dist-packages/fastai/vision/augment.py in _encode(self, x, mode)
    544     def _encode(self, x, mode):
    545         x = x[...,self.tl[0]:self.tl[0]+self.cp_size[0], self.tl[1]:self.tl[1]+self.cp_size[1]]
--> 546         return x.affine_coord(sz=self.size, mode=mode)
    547 
    548     def encodes(self, x:TensorImage|TensorPoint|TensorBBox): return self._encode(x, self.mode)

/usr/local/lib/python3.11/dist-packages/fastai/vision/augment.py in affine_coord(x, mat, coord_tfm, sz, mode, pad_mode, align_corners)
    391     coords = affine_grid(mat, x.shape[:2] + size, align_corners=align_corners)
    392     if coord_tfm is not None: coords = coord_tfm(coords)
--> 393     return TensorImage(_grid_sample(x, coords, mode=mode, padding_mode=pad_mode, align_corners=align_corners))
    394 
    395 @patch

/usr/local/lib/python3.11/dist-packages/fastai/vision/augment.py in _grid_sample(x, coords, mode, padding_mode, align_corners)
    364         if d>1 and d>z:
    365             x = F.interpolate(x, scale_factor=1/d, mode='area', recompute_scale_factor=True)
--> 366     return F.grid_sample(x, coords, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    367 
    368 # %% ../../nbs/09_vision.augment.ipynb 90

/usr/local/lib/python3.11/dist-packages/torch/nn/functional.py in grid_sample(input, grid, mode, padding_mode, align_corners)
   4974     """
   4975     if has_torch_function_variadic(input, grid):
-> 4976         return handle_torch_function(
   4977             grid_sample,
   4978             (input, grid),

/usr/local/lib/python3.11/dist-packages/torch/overrides.py in handle_torch_function(public_api, relevant_args, *args, **kwargs)
   1740         # Use `public_api` instead of `implementation` so __torch_function__
   1741         # implementations can do equality/identity comparisons.
-> 1742         result = torch_func_method(public_api, types, args, kwargs)
   1743 
   1744         if result is not NotImplemented:

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

/usr/local/lib/python3.11/dist-packages/torch/nn/functional.py in grid_sample(input, grid, mode, padding_mode, align_corners)
   5021         align_corners = False
   5022 
-> 5023     return torch.grid_sampler(input, grid, mode_enum, padding_mode_enum, align_corners)
   5024 
   5025 

RuntimeError: grid_sampler(): expected grid to have size 1 in last dimension, but got grid with sizes [3, 512, 512, 2]

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
/tmp/ipykernel_55/3366323160.py in <cell line: 0>()
----> 1 learn = cnn_learner(dls, resnet50, metrics=[error_rate, accuracy])
      2 
      3 # CHANGE (timeout): keep channels_last + fp16 as originally intended for GPU throughput; no logic change.
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
/tmp/ipykernel_55/3730637765.py in <cell line: 0>()
      2     # CHANGE (timeout): enable benchmark only during training (same as before) for best kernel selection.
      3     torch.backends.cudnn.benchmark = True
----> 4     learn.fine_tune(10, cbs=[MixUp(0.5)])
      5     torch.backends.cudnn.benchmark = False
      6 

NameError: name 'learn' is not defined

## === cell 6
learn = learn.to_fp32()
learn



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/198077790.py in <cell line: 0>()
----> 1 learn = learn.to_fp32()
      2 learn
      3 

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
    prefetch_factor=(4 if num_workers > 0 else None),
)

learn.model.eval()
with torch.inference_mode():
    preds, _ = learn.get_preds(dl=test_dl, reorder=False)

test_predictions = preds.argmax(dim=1).cpu().numpy().astype(np.int64).tolist()
_ = (len(test_predictions), test_predictions[:5])



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1150105202.py in <cell line: 0>()
      2 
      3 # CHANGE (timeout): keep a high inference batch size and strong prefetch to minimize input overhead.
----> 4 test_dl = dls.test_dl(
      5     test_files,
      6     with_labels=False,

NameError: name 'dls' is not defined

## === cell 9
submission = test_df.copy()
assert len(test_predictions) == len(submission), (
    len(test_predictions),
    len(submission),
)

submission["label"] = test_predictions
submission.to_csv("submission.csv", index=False)

_ = (Path("submission.csv").exists(), pd.read_csv("submission.csv").head(1))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/336727909.py in <cell line: 0>()
      1 submission = test_df.copy()
----> 2 assert len(test_predictions) == len(submission), (
      3     len(test_predictions),
      4     len(submission),
      5 )

NameError: name 'test_predictions' is not defined
