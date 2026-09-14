# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
!pip install albumentations > /dev/null 2>&1
!pip install pretrainedmodels > /dev/null 2>&1
!pip install catalyst > /dev/null 2>&1


## === cell 1
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")


def train_test_split(
    *arrays,
    test_size=0.25,
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None
):
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")

    n = len(arrays[0])
    for a in arrays[1:]:
        if len(a) != n:
            raise ValueError("All input arrays must have the same length")

    if train_size is None and test_size is None:
        test_size = 0.25
    if train_size is None:
        n_test = int(np.ceil(n * float(test_size)))
        n_train = n - n_test
    elif test_size is None:
        n_train = int(np.floor(n * float(train_size)))
        n_test = n - n_train
    else:
        n_train = int(np.floor(n * float(train_size)))
        n_test = int(np.ceil(n * float(test_size)))
        if n_train + n_test > n:
            n_test = n - n_train

    rng = np.random.RandomState(random_state)

    if not shuffle:
        idx = np.arange(n)
    else:
        if stratify is None:
            idx = rng.permutation(n)
        else:
            y = np.asarray(stratify)
            if len(y) != n:
                raise ValueError(
                    "stratify array must be the same length as input arrays"
                )
            idx_parts = []
            for cls in np.unique(y):
                cls_idx = np.where(y == cls)[0]
                cls_idx = rng.permutation(cls_idx)
                idx_parts.append(cls_idx)
            idx = np.concatenate(idx_parts)
            idx = rng.permutation(idx)

    test_idx = idx[:n_test]
    train_idx = idx[n_test : n_test + n_train]

    out = []
    for a in arrays:
        if hasattr(a, "iloc"):  # pandas objects
            out.append(a.iloc[train_idx])
            out.append(a.iloc[test_idx])
        else:
            a_np = np.asarray(a)
            out.append(a_np[train_idx])
            out.append(a_np[test_idx])
    return tuple(out)


def _rankdata_average(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    classes = np.unique(y_true)
    if classes.size != 2:
        raise ValueError(
            "roc_auc_score is defined for binary classification only in this notebook"
        )

    ranks = _rankdata_average(y_score)
    pos = y_true == 1
    n_pos = np.sum(pos)
    n_neg = y_true.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "Only one class present in y_true. ROC AUC score is not defined in that case."
        )
    sum_ranks_pos = np.sum(ranks[pos])
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
from PIL import Image

train_on_gpu = True
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import cv2

try:
    from sklearn.preprocessing import OneHotEncoder  # noqa: F401
except Exception as e:
    OneHotEncoder = None
    _SKLEARN_IMPORT_ERROR = e

try:
    import albumentations  # noqa: F401
    from albumentations.pytorch import ToTensor  # type: ignore
except Exception:

    class _AlbumentationsCompose:
        def __init__(self, transforms_list):
            self.transforms_list = transforms_list or []

        def __call__(self, **kwargs):
            data = dict(kwargs)
            for t in self.transforms_list:
                if t is None:
                    continue
                try:
                    data = t(**data)
                except TypeError:
                    if "image" in data:
                        data["image"] = t(data["image"])
            return data

    class _ToTensor:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, **kwargs):
            if "image" not in kwargs:
                return kwargs
            img = kwargs["image"]
            if isinstance(img, torch.Tensor):
                out = img
            else:
                img_np = np.asarray(img)
                if img_np.ndim == 2:
                    img_np = img_np[:, :, None]
                img_np = np.ascontiguousarray(img_np)
                out = torch.from_numpy(img_np).permute(2, 0, 1).float()
            kwargs["image"] = out
            return kwargs

    class _AlbumentationsModule:
        Compose = _AlbumentationsCompose

        class pytorch:  # mimic albumentations.pytorch namespace
            ToTensor = _ToTensor

    albumentations = _AlbumentationsModule()
    ToTensor = _ToTensor

import pretrainedmodels

import collections


## === cell 2
import collections
import collections.abc

for _name in (
    "MutableMapping",
    "Mapping",
    "MutableSequence",
    "Sequence",
    "MutableSet",
    "Set",
):
    if not hasattr(collections, _name) and hasattr(collections.abc, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

try:
    import skimage.color as _skc  # noqa: F401

    if not hasattr(_skc, "label2rgb"):
        _label2rgb = None
        for _path in (
            "skimage.color",
            "skimage.color.colorlabel",
        ):
            try:
                _mod = __import__(_path, fromlist=["label2rgb"])
                _label2rgb = getattr(_mod, "label2rgb", None)
                if _label2rgb is not None:
                    break
            except Exception:
                continue
        if _label2rgb is None:

            def _label2rgb(*args, **kwargs):  # type: ignore
                raise ImportError(
                    "skimage.color.label2rgb is unavailable in this environment"
                )

        _skc.label2rgb = _label2rgb  # type: ignore[attr-defined]

    if not hasattr(_skc, "rgb2gray"):
        _rgb2gray = None
        for _path in (
            "skimage.color",
            "skimage.color.colorconv",
        ):
            try:
                _mod = __import__(_path, fromlist=["rgb2gray"])
                _rgb2gray = getattr(_mod, "rgb2gray", None)
                if _rgb2gray is not None:
                    break
            except Exception:
                continue
        if _rgb2gray is None:

            def _rgb2gray(*args, **kwargs):  # type: ignore
                raise ImportError(
                    "skimage.color.rgb2gray is unavailable in this environment"
                )

        _skc.rgb2gray = _rgb2gray  # type: ignore[attr-defined]
except Exception:
    pass

import os
import sys
import types

os.environ["CATALYST_ML_REQUIRED"] = "0"
os.environ["CATALYST_ML"] = "0"
os.environ.setdefault("CATALYST_SETTINGS", '{"ml_required": false, "ml": false}')
os.environ.setdefault("CATALYST_DISABLE_ML", "1")

try:
    import sklearn  # noqa: F401
except Exception:
    sys.modules["sklearn"] = types.ModuleType("sklearn")

from catalyst.dl.utils import UtilsFactory
from catalyst.dl.experiments import SupervisedRunner
from catalyst.dl.callbacks import EarlyStoppingCallback, OneCycleLR, InferCallback


## --- ERROR in cell 2, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3814046582.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     79[0m     [0msys[0m[0;34m.[0m[0mmodules[0m[0;34m[[0m[0;34m"sklearn"[0m[0;34m][0m [0;34m=[0m [0mtypes[0m[0;34m.[0m[0mModuleType[0m[0;34m([0m[0;34m"sklearn"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     80[0m [0;34m[0m[0m
[0;32m---> 81[0;31m [0;32mfrom[0m [0mcatalyst[0m[0;34m.[0m[0mdl[0m[0;34m.[0m[0mutils[0m [0;32mimport[0m [0mUtilsFactory[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     82[0m [0;32mfrom[0m [0mcatalyst[0m[0;34m.[0m[0mdl[0m[0;34m.[0m[0mexperiments[0m [0;32mimport[0m [0mSupervisedRunner[0m[0;34m[0m[0;34m[0m[0m
[1;32m     83[0m [0;32mfrom[0m [0mcatalyst[0m[0;34m.[0m[0mdl[0m[0;34m.[0m[0mcallbacks[0m [0;32mimport[0m [0mEarlyStoppingCallback[0m[0;34m,[0m [0mOneCycleLR[0m[0;34m,[0m [0mInferCallback[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/catalyst/dl/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0;31m# flake8: noqa[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;34m[0m[0m
[0;32m----> 3[0;31m [0;32mfrom[0m [0mcatalyst[0m[0;34m.[0m[0mcore[0m [0;32mimport[0m [0;34m*[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0;32mfrom[0m [0mcatalyst[0m[0;34m.[0m[0mengines[0m [0;32mimport[0m [0;34m*[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/catalyst/core/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      4[0m [0;31m# callback[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;34m[0m[0m
[0;32m----> 6[0;31m [0;32mfrom[0m [0mcatalyst[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mrunner[0m [0;32mimport[0m [0mIRunner[0m[0;34m,[0m [0mRunnerException[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      7[0m from catalyst.core.callback import (
[1;32m      8[0m     [0mICallback[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/catalyst/core/runner.py[0m in [0;36m<module>[0;34m[0m
[1;32m     13[0m [0;32mfrom[0m [0mcatalyst[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mengine[0m [0;32mimport[0m [0mIEngine[0m[0;34m[0m[0;34m[0m[0m
[1;32m     14[0m [0;32mfrom[0m [0mcatalyst[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mlogger[0m [0;32mimport[0m [0mILogger[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 15[0;31m [0;32mfrom[0m [0mcatalyst[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mmisc[0m [0;32mimport[0m [0mfilter_callbacks_by_node[0m[0;34m,[0m [0msort_callbacks_by_order[0m[0;34m,[0m [0mvalidate_loaders[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     16[0m [0;32mfrom[0m [0mcatalyst[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mtrial[0m [0;32mimport[0m [0mITrial[0m[0;34m[0m[0;34m[0m[0m
[1;32m     17[0m [0;32mfrom[0m [0mcatalyst[0m[0;34m.[0m[0mengines[0m[0;34m.[0m[0mtorch[0m [0;32mimport[0m [0mDistributedDataParallelEngine[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/catalyst/core/misc.py[0m in [0;36m<module>[0;34m[0m
[1;32m      7[0m [0;34m[0m[0m
[1;32m      8[0m [0;32mfrom[0m [0mcatalyst[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mcallback[0m [0;32mimport[0m [0mCallback[0m[0;34m,[0m [0mCallbackNode[0m[0;34m,[0m [0mCallbackWrapper[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 9[0;31m [0;32mfrom[0m [0mcatalyst[0m[0;34m.[0m[0mdata[0m[0;34m.[0m[0msampler[0m [0;32mimport[0m [0mDistributedSamplerWrapper[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     10[0m [0;32mfrom[0m [0mcatalyst[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mdistributed[0m [0;32mimport[0m [0mget_rank[0m[0;34m[0m[0;34m[0m[0m
[1;32m     11[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/catalyst/data/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0;31m# flake8: noqa[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;32mfrom[0m [0mcatalyst[0m[0;34m.[0m[0mdata[0m[0;34m.[0m[0mcollate_fn[0m [0;32mimport[0m [0mFilteringCollateFn[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 3[0;31m from catalyst.data.dataset import (
[0m[1;32m      4[0m     [0mDatasetFromSampler[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m     [0mListDataset[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/catalyst/data/dataset/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0;31m# flake8: noqa[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m from catalyst.data.dataset.torch import (
[0m[1;32m      3[0m     [0mDatasetFromSampler[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m     [0mListDataset[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m     [0mMergeDataset[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/catalyst/data/dataset/torch.py[0m in [0;36m<module>[0;34m[0m
[1;32m      5[0m [0;32mfrom[0m [0mtorch[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mdata[0m [0;32mimport[0m [0mDataset[0m[0;34m,[0m [0mSampler[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;34m[0m[0m
[0;32m----> 7[0;31m [0;32mfrom[0m [0mcatalyst[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mmisc[0m [0;32mimport[0m [0mmerge_dicts[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      8[0m [0;34m[0m[0m
[1;32m      9[0m [0m_Path[0m [0;34m=[0m [0mUnion[0m[0;34m[[0m[0mstr[0m[0;34m,[0m [0mPath[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/catalyst/utils/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     89[0m [0;34m[0m[0m
[1;32m     90[0m [0;34m[0m[0m
[0;32m---> 91[0;31m [0;32mfrom[0m [0mcatalyst[0m[0;34m.[0m[0mcontrib[0m[0;34m.[0m[0mutils[0m [0;32mimport[0m [0;34m*[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/catalyst/contrib/utils/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     20[0m [0;34m[0m[0m
[1;32m     21[0m [0;32mif[0m [0mSETTINGS[0m[0;34m.[0m[0mml_required[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 22[0;31m     from catalyst.contrib.utils.pandas import (
[0m[1;32m     23[0m         [0mdataframe_to_list[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     24[0m         [0mfolds_to_list[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/catalyst/contrib/utils/pandas.py[0m in [0;36m<module>[0;34m[0m
[1;32m      8[0m [0;32mimport[0m [0mnumpy[0m [0;32mas[0m [0mnp[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m [0;32mimport[0m [0mpandas[0m [0;32mas[0m [0mpd[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 10[0;31m [0;32mfrom[0m [0msklearn[0m[0;34m.[0m[0mmodel_selection[0m [0;32mimport[0m [0mStratifiedKFold[0m[0;34m,[0m [0mtrain_test_split[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     11[0m [0;32mfrom[0m [0msklearn[0m[0;34m.[0m[0mutils[0m [0;32mimport[0m [0mshuffle[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m [0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'sklearn.model_selection'; 'sklearn' is not a package

## === cell 3
data_transforms = albumentations.Compose([
    albumentations.HorizontalFlip(),
    albumentations.VerticalFlip(),
    albumentations.RandomBrightness(),
    albumentations.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensor()
    ])
data_transforms_test = albumentations.Compose([
    albumentations.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensor()
    ])
