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
data_dir = Path("../input/aptos2019-blindness-detection")
model_dir = Path("./model")
model_dir.mkdir(parents=True, exist_ok=True)  # ensure export folder exists
batch_size = 8
im_size = (528, 528)

model_name = "efficientnet-b6"
export_path = model_dir / f"{model_name}.pkl"
train_flag = not export_path.is_file()  # train if no exported model
c = 1 / 3  # retained for compatibility (unused)


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

    dls = ImageDataLoaders.from_df(
        df,
        path=data_dir,
        folder="train_images",
        suffix=".png",
        fn_col="id_code",
        label_col="diagnosis",
        valid_pct=0.2,
        seed=42,
        after_open=preprocess,
        item_tfms=Resize(im_size, method=ResizeMethod.Pad),
        batch_tfms=[
            *aug_transforms(
                max_warp=None,
                xtra_tfms=[
                    RandomErasing(p=0.5, max_count=1, min_area=0.02, max_area=0.33)
                ],
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
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4052874394.py in <cell line: 0>()
     47                 max_warp=None,
     48                 xtra_tfms=[
---> 49                     RandomErasing(p=0.5, max_count=1, min_area=0.02, max_area=0.33)
     50                 ],
     51             ),

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in __call__(cls, *args, **kwargs)
     65             else: getattr(cls,nm).dispatch(f)
     66             return cls
---> 67         obj = super().__call__(*args, **kwargs)
     68         # _TfmMeta.__new__ replaces cls.__signature__ which breaks the signature of a callable
     69         # instances of cls, fix it

TypeError: RandomErasing.__init__() got an unexpected keyword argument 'min_area'

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
/tmp/ipykernel_11/2489965773.py in <cell line: 0>()
     11 
     12     learn = Learner(
---> 13         data,
     14         model,
     15         loss_func=partial(F.cross_entropy, weight=w),

NameError: name 'data' is not defined

## === cell 3
if train_flag:
    learn.freeze()
    learn.fit_one_cycle(1, lr_max=1e-3)

    learn.unfreeze()
    learn.fit_one_cycle(2, lr_max=1e-4)  # short training for quick turnaround

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
/tmp/ipykernel_11/3116324748.py in <cell line: 0>()
      1 if train_flag:
----> 2     learn.freeze()
      3     learn.fit_one_cycle(1, lr_max=1e-3)
      4 
      5     learn.unfreeze()

NameError: name 'learn' is not defined
