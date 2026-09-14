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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
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
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
        input/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
            test_images/
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
            train_images/
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> input/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> working/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.8922866227853757

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import sys
import importlib
from pathlib import Path
from functools import partial

try:
    from efficientnet_pytorch import EfficientNet as EfficientNetPT
except ModuleNotFoundError:
    EfficientNetPT = None

try:
    from rectified_adam.radam import RAdam
except ModuleNotFoundError:
    from torch.optim import Adam as RAdam

from fastai.vision.all import *
from fastai.torch_core import defaults
import torch.nn.functional as F
from fastai.vision.augment import RandomErasing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
defaults.device = device

try:
    from fastai.metrics import KappaScore
except Exception:
    from sklearn.metrics import cohen_kappa_score

    class KappaScore(Metric):
        """Quadratic weighted Kappa metric compatible with fastai Learner."""

        def __init__(self, weights: str = "quadratic"):
            self.weights = weights
            self._preds, self._targs = [], []

        def reset(self):
            self._preds, self._targs = [], []

        def accumulate(self, learn):
            preds = learn.pred.argmax(dim=1).detach().cpu().numpy()
            targs = learn.y.detach().cpu().numpy()
            self._preds.append(preds)
            self._targs.append(targs)

        @property
        def value(self):
            if not self._preds:
                return None
            pred = np.concatenate(self._preds)
            targ = np.concatenate(self._targs)
            return cohen_kappa_score(targ, pred, weights=self.weights)


import torch
import numpy as np
import pandas as pd



## === cell 1
data_dir = Path("/kaggle/input/aptos2019-blindness-detection")
model_dir = Path("./model")
model_dir.mkdir(parents=True, exist_ok=True)  # ensure export folder exists
batch_size = 8
im_size = (528, 528)

model_name = "efficientnet-b6"
export_path = model_dir / f"{model_name}.pkl"
train_flag = not export_path.is_file()  # train if no exported model


def preprocess(im):
    """Crop the image to its non‑zero bounding box."""
    bbox = pil2tensor(im, dtype=np.uint8).nonzero().transpose(1, 0)
    im = im.crop(
        (
            bbox[2].min().item(),
            bbox[1].min().item(),
            bbox[2].max().item(),
            bbox[1].max().item(),
        )
    )
    return im


if train_flag:
    df = pd.read_csv(data_dir / "train.csv")
    counts = df["diagnosis"].value_counts().reindex(range(5), fill_value=0).sort_index()
    w = 1 - torch.tensor(
        [counts[i] for i in range(5)], dtype=torch.float32, device=device
    )

    train_images_path = data_dir / "train_images"

    dls = ImageDataLoaders.from_df(
        df,
        path=train_images_path,
        folder="",  # no extra sub‑folder
        suffix=".png",
        fn_col="id_code",
        label_col="diagnosis",
        valid_pct=0.2,
        seed=42,
        after_open=preprocess,
        item_tfms=Resize(im_size, method=ResizeMethod.Pad),
        batch_tfms=[
            *aug_transforms(
                max_rotate=0.0,
                max_zoom=1.0,
                max_lighting=0.0,
                max_warp=0.0,
                xtra_tfms=[RandomErasing(p=0.5, max_count=1, sl=0.02, sh=0.33)],
            ),
            Normalize.from_stats(*imagenet_stats),
        ],
        bs=batch_size,
    )
    data = dls
else:
    data = None  # placeholder; will load learner later




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_10/1644039030.py in <cell line: 0>()
     34     train_images_path = data_dir / "train_images"
     35 
---> 36     dls = ImageDataLoaders.from_df(
     37         df,
     38         path=train_images_path,

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/aptos2019-blindness-detection/train_images/7ea756985353'

## === cell 2
if train_flag:
    if EfficientNetPT is not None:
        model = EfficientNetPT.from_pretrained(model_name, num_classes=5)
    else:
        from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights

        model = efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 5)

    kappa = KappaScore(weights="quadratic")

    learn = Learner(
        data,
        model,
        loss_func=partial(F.cross_entropy, weight=w),
        opt_func=RAdam,
        metrics=[kappa],
    )
    if torch.cuda.is_available():
        learn = learn.to_fp16()
    learn.layer_groups = split_model_idx(learn.model, idxs=[-1])
    learn.summary()
else:
    if export_path.is_file():
        learn = load_learner(model_dir, f"{model_name}.pkl")
    else:
        raise FileNotFoundError(f"Exported model not found at {export_path}")

learn.path = Path(".")  # ensure predictions are saved relative to current dir




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/2489965773.py in <cell line: 0>()
     11 
     12     learn = Learner(
---> 13         data,
     14         model,
     15         loss_func=partial(F.cross_entropy, weight=w),

NameError: name 'data' is not defined

## === cell 3
if train_flag:
    learn.freeze()
    learn.fit_one_cycle(2, lr_max=1e-3)  # frozen backbone
    learn.unfreeze()
    learn.fit_one_cycle(5, lr_max=1e-4)  # fine‑tuning
    learn.export(export_path)
    learn.save(model_dir / model_name)

learn.model.eval()
mean = torch.tensor(imagenet_stats[0])
std = torch.tensor(imagenet_stats[1])


def diagnose(row):
    """Predict the diagnosis for a single test image."""
    fn = row["id_code"]
    im_path = data_dir / "test_images" / f"{fn}.png"
    im = PILImage.create(im_path)
    im = preprocess(im)
    im = im.apply_tfms(
        None,
        size=im_size,
        resize_method=ResizeMethod.Pad,
        padding_mode="zeros",
    )
    x = im.data.unsqueeze(0)  # (1, C, H, W)
    x = (x - mean[None, :, None, None]) / std[None, :, None, None]
    x = x.to(device).half()
    with torch.no_grad():
        logits = learn.model(x)
    return logits.argmax(dim=1).item()


test_df = pd.read_csv(data_dir / "test.csv")
test_df["diagnosis"] = test_df.apply(diagnose, axis=1)
test_df.to_csv("submission.csv", index=False, line_terminator="\n")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/1393804053.py in <cell line: 0>()
      1 if train_flag:
----> 2     learn.freeze()
      3     learn.fit_one_cycle(2, lr_max=1e-3)  # frozen backbone
      4     learn.unfreeze()
      5     learn.fit_one_cycle(5, lr_max=1e-4)  # fine‑tuning

NameError: name 'learn' is not defined
