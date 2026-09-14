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
    X, y=None, test_size=0.25, random_state=None, shuffle=True, stratify=None
):
    rng = np.random.RandomState(random_state) if random_state is not None else np.random
    X_arr = np.asarray(X)
    n = len(X_arr)
    if isinstance(test_size, float):
        n_test = int(np.ceil(n * test_size))
    else:
        n_test = int(test_size)
    n_test = max(0, min(n, n_test))
    indices = np.arange(n)

    if shuffle:
        if stratify is not None:
            s = np.asarray(stratify)
            unique = np.unique(s)
            test_idx = []
            train_idx = []
            for cls in unique:
                cls_idx = indices[s == cls]
                rng.shuffle(cls_idx)
                cls_n_test = int(np.round(len(cls_idx) * (n_test / n))) if n > 0 else 0
                test_idx.append(cls_idx[:cls_n_test])
                train_idx.append(cls_idx[cls_n_test:])
            test_idx = (
                np.concatenate(test_idx) if len(test_idx) else np.array([], dtype=int)
            )
            train_idx = (
                np.concatenate(train_idx) if len(train_idx) else np.array([], dtype=int)
            )

            if len(test_idx) > n_test:
                rng.shuffle(test_idx)
                extra = test_idx[n_test:]
                test_idx = test_idx[:n_test]
                train_idx = (
                    np.concatenate([train_idx, extra]) if len(extra) else train_idx
                )
            elif len(test_idx) < n_test:
                need = n_test - len(test_idx)
                rng.shuffle(train_idx)
                moved = train_idx[:need]
                train_idx = train_idx[need:]
                test_idx = np.concatenate([test_idx, moved]) if len(moved) else test_idx

            rng.shuffle(train_idx)
            rng.shuffle(test_idx)
        else:
            rng.shuffle(indices)
            test_idx = indices[:n_test]
            train_idx = indices[n_test:]
    else:
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    def _split(arr):
        arr = np.asarray(arr)
        return arr[train_idx], arr[test_idx]

    if y is None:
        return _split(X_arr)

    y_arr = np.asarray(y)
    X_train, X_test = _split(X_arr)
    y_train, y_test = _split(y_arr)
    return X_train, X_test, y_train, y_test


def roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).astype(float).ravel()
    if y_true.size == 0:
        raise ValueError("y_true is empty")
    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "Only one class present in y_true. ROC AUC score is not defined in that case."
        )

    order = np.argsort(y_score, kind="mergesort")
    scores_sorted = y_score[order]
    y_sorted = y_true[order]

    ranks = np.empty_like(scores_sorted, dtype=float)
    i = 0
    n = len(scores_sorted)
    while i < n:
        j = i
        while j + 1 < n and scores_sorted[j + 1] == scores_sorted[i]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks[i : j + 1] = avg_rank
        i = j + 1

    sum_ranks_pos = ranks[y_sorted == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float((y_true == y_pred).mean())


class OneHotEncoder:
    def __init__(self, sparse=False, handle_unknown="error"):
        self.sparse = sparse
        self.handle_unknown = handle_unknown
        self.categories_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self

    def transform(self, X):
        if self.categories_ is None:
            raise ValueError("OneHotEncoder instance is not fitted yet.")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples, n_features = X.shape
        if n_features != len(self.categories_):
            raise ValueError("X has different number of features than during fit.")

        total_dims = sum(len(c) for c in self.categories_)
        out = np.zeros((n_samples, total_dims), dtype=float)
        col_offset = 0
        for i, cats in enumerate(self.categories_):
            cat_to_idx = {c: idx for idx, c in enumerate(cats)}
            for r in range(n_samples):
                val = X[r, i]
                if val in cat_to_idx:
                    out[r, col_offset + cat_to_idx[val]] = 1.0
                else:
                    if self.handle_unknown == "error":
                        raise ValueError(
                            f"Found unknown category {val} in column {i} during transform"
                        )
            col_offset += len(cats)
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


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
    import albumentations  # noqa: F401

    try:
        from albumentations.pytorch import ToTensorV2 as ToTensor
    except Exception:
        from albumentations.pytorch import ToTensor  # type: ignore
except Exception:
    albumentations = None
    ToTensor = None

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
    import skimage.color as _skc  # type: ignore

    if not hasattr(_skc, "label2rgb"):

        def _label2rgb_fallback(
            label,
            image=None,
            colors=None,
            alpha=0.3,
            bg_label=0,
            bg_color=(0, 0, 0),
            image_alpha=1.0,
            kind="overlay",
        ):  # type: ignore
            import numpy as _np

            if image is not None:
                return _np.asarray(image)
            return _np.asarray(label, dtype=float)

        setattr(_skc, "label2rgb", _label2rgb_fallback)

    if not hasattr(_skc, "rgb2gray"):

        def _rgb2gray_fallback(rgb):  # type: ignore
            import numpy as _np

            arr = _np.asarray(rgb)
            if arr.ndim >= 3 and arr.shape[-1] >= 3:
                return (
                    0.2125 * arr[..., 0] + 0.7154 * arr[..., 1] + 0.0721 * arr[..., 2]
                )
            return arr.astype(float)

        setattr(_skc, "rgb2gray", _rgb2gray_fallback)
except Exception:
    pass

from catalyst.dl.utils import UtilsFactory
from catalyst.dl.experiments import SupervisedRunner
from catalyst.dl.callbacks import EarlyStoppingCallback, OneCycleLR, InferCallback


## --- ERROR in cell 2, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2223776983.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     56[0m     [0;32mpass[0m[0;34m[0m[0;34m[0m[0m
[1;32m     57[0m [0;34m[0m[0m
[0;32m---> 58[0;31m [0;32mfrom[0m [0mcatalyst[0m[0;34m.[0m[0mdl[0m[0;34m.[0m[0mutils[0m [0;32mimport[0m [0mUtilsFactory[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     59[0m [0;32mfrom[0m [0mcatalyst[0m[0;34m.[0m[0mdl[0m[0;34m.[0m[0mexperiments[0m [0;32mimport[0m [0mSupervisedRunner[0m[0;34m[0m[0;34m[0m[0m
[1;32m     60[0m [0;32mfrom[0m [0mcatalyst[0m[0;34m.[0m[0mdl[0m[0;34m.[0m[0mcallbacks[0m [0;32mimport[0m [0mEarlyStoppingCallback[0m[0;34m,[0m [0mOneCycleLR[0m[0;34m,[0m [0mInferCallback[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/catalyst/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      7[0m [0;34m[0m[0m
[1;32m      8[0m [0;32mfrom[0m [0mcatalyst[0m[0;34m.[0m[0m__version__[0m [0;32mimport[0m [0m__version__[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 9[0;31m [0;32mfrom[0m [0mcatalyst[0m[0;34m.[0m[0msettings[0m [0;32mimport[0m [0mSETTINGS[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/catalyst/settings.py[0m in [0;36m<module>[0;34m[0m
[1;32m    333[0m [0;34m[0m[0m
[1;32m    334[0m [0;34m[0m[0m
[0;32m--> 335[0;31m [0mDEFAULT_SETTINGS[0m [0;34m=[0m [0mSettings[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    336[0m [0;34m[0m[0m
[1;32m    337[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/catalyst/settings.py[0m in [0;36m__init__[0;34m(self, cv_required, nifti_required, ml_required, hydra_required, optuna_required, amp_required, apex_required, xla_required, onnx_required, pruning_required, quantization_required, neptune_required, mlflow_required, wandb_required, use_lz4, use_pyarrow, use_libjpeg_turbo)[0m
[1;32m    208[0m         )
[1;32m    209[0m [0;34m[0m[0m
[0;32m--> 210[0;31m         self.ml_required: bool = _get_optional_value(
[0m[1;32m    211[0m             [0mml_required[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    212[0m             [0m_is_ml_available[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/catalyst/settings.py[0m in [0;36m_get_optional_value[0;34m(is_required, is_available_fn, assert_msg)[0m
[1;32m    154[0m ) -> bool:
[1;32m    155[0m     [0;32mif[0m [0mis_required[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 156[0;31m         [0;32mreturn[0m [0mis_available_fn[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    157[0m     [0;32melif[0m [0mis_required[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    158[0m         [0;32massert[0m [0mis_available_fn[0m[0;34m([0m[0;34m)[0m[0;34m,[0m [0massert_msg[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/catalyst/settings.py[0m in [0;36m_is_ml_available[0;34m()[0m
[1;32m    116[0m         [0;32mimport[0m [0mpandas[0m  [0;31m# noqa: F401[0m[0;34m[0m[0;34m[0m[0m
[1;32m    117[0m         [0;32mimport[0m [0mscipy[0m  [0;31m# noqa: F401[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 118[0;31m         [0;32mimport[0m [0msklearn[0m  [0;31m# noqa: F401[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    119[0m [0;34m[0m[0m
[1;32m    120[0m         [0;32mreturn[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     80[0m     [0;32mfrom[0m [0;34m.[0m [0;32mimport[0m [0m_distributor_init[0m  [0;31m# noqa: F401[0m[0;34m[0m[0;34m[0m[0m
[1;32m     81[0m     [0;32mfrom[0m [0;34m.[0m [0;32mimport[0m [0m__check_build[0m  [0;31m# noqa: F401[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 82[0;31m     [0;32mfrom[0m [0;34m.[0m[0mbase[0m [0;32mimport[0m [0mclone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     83[0m     [0;32mfrom[0m [0;34m.[0m[0mutils[0m[0;34m.[0m[0m_show_versions[0m [0;32mimport[0m [0mshow_versions[0m[0;34m[0m[0;34m[0m[0m
[1;32m     84[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/base.py[0m in [0;36m<module>[0;34m[0m
[1;32m     15[0m [0;32mfrom[0m [0;34m.[0m [0;32mimport[0m [0m__version__[0m[0;34m[0m[0;34m[0m[0m
[1;32m     16[0m [0;32mfrom[0m [0;34m.[0m[0m_config[0m [0;32mimport[0m [0mget_config[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 17[0;31m [0;32mfrom[0m [0;34m.[0m[0mutils[0m [0;32mimport[0m [0m_IS_32BIT[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     18[0m [0;32mfrom[0m [0;34m.[0m[0mutils[0m[0;34m.[0m[0m_set_output[0m [0;32mimport[0m [0m_SetOutputMixin[0m[0;34m[0m[0;34m[0m[0m
[1;32m     19[0m from .utils._tags import (

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     23[0m [0;32mfrom[0m [0;34m.[0m[0mdeprecation[0m [0;32mimport[0m [0mdeprecated[0m[0;34m[0m[0;34m[0m[0m
[1;32m     24[0m [0;32mfrom[0m [0;34m.[0m[0mdiscovery[0m [0;32mimport[0m [0mall_estimators[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 25[0;31m [0;32mfrom[0m [0;34m.[0m[0mfixes[0m [0;32mimport[0m [0mparse_version[0m[0;34m,[0m [0mthreadpool_info[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     26[0m [0;32mfrom[0m [0;34m.[0m[0m_estimator_html_repr[0m [0;32mimport[0m [0mestimator_html_repr[0m[0;34m[0m[0;34m[0m[0m
[1;32m     27[0m from .validation import (

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/fixes.py[0m in [0;36m<module>[0;34m[0m
[1;32m     17[0m [0;32mimport[0m [0mnumpy[0m [0;32mas[0m [0mnp[0m[0;34m[0m[0;34m[0m[0m
[1;32m     18[0m [0;32mimport[0m [0mscipy[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 19[0;31m [0;32mimport[0m [0mscipy[0m[0;34m.[0m[0mstats[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     20[0m [0;32mimport[0m [0mthreadpoolctl[0m[0;34m[0m[0;34m[0m[0m
[1;32m     21[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/scipy/stats/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m    622[0m from ._warnings_errors import (ConstantInputWarning, NearConstantInputWarning,
[1;32m    623[0m                                DegenerateDataWarning, FitError)
[0;32m--> 624[0;31m [0;32mfrom[0m [0;34m.[0m[0m_stats_py[0m [0;32mimport[0m [0;34m*[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    625[0m [0;32mfrom[0m [0;34m.[0m[0m_variation[0m [0;32mimport[0m [0mvariation[0m[0;34m[0m[0;34m[0m[0m
[1;32m    626[0m [0;32mfrom[0m [0;34m.[0m[0mdistributions[0m [0;32mimport[0m [0;34m*[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/scipy/stats/_stats_py.py[0m in [0;36m<module>[0;34m[0m
[1;32m     37[0m [0;34m[0m[0m
[1;32m     38[0m [0;32mfrom[0m [0mscipy[0m [0;32mimport[0m [0msparse[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 39[0;31m [0;32mfrom[0m [0mscipy[0m[0;34m.[0m[0mspatial[0m [0;32mimport[0m [0mdistance_matrix[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     40[0m [0;34m[0m[0m
[1;32m     41[0m [0;32mfrom[0m [0mscipy[0m[0;34m.[0m[0moptimize[0m [0;32mimport[0m [0mmilp[0m[0;34m,[0m [0mLinearConstraint[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/scipy/spatial/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m    114[0m [0;32mfrom[0m [0;34m.[0m[0m_plotutils[0m [0;32mimport[0m [0;34m*[0m[0;34m[0m[0;34m[0m[0m
[1;32m    115[0m [0;32mfrom[0m [0;34m.[0m[0m_procrustes[0m [0;32mimport[0m [0mprocrustes[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 116[0;31m [0;32mfrom[0m [0;34m.[0m[0m_geometric_slerp[0m [0;32mimport[0m [0mgeometric_slerp[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    117[0m [0;34m[0m[0m
[1;32m    118[0m [0;31m# Deprecated namespaces, to be removed in v2.0.0[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/scipy/spatial/_geometric_slerp.py[0m in [0;36m<module>[0;34m[0m
[1;32m      5[0m [0;34m[0m[0m
[1;32m      6[0m [0;32mimport[0m [0mnumpy[0m [0;32mas[0m [0mnp[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 7[0;31m [0;32mfrom[0m [0mscipy[0m[0;34m.[0m[0mspatial[0m[0;34m.[0m[0mdistance[0m [0;32mimport[0m [0meuclidean[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      8[0m [0;34m[0m[0m
[1;32m      9[0m [0;32mif[0m [0mTYPE_CHECKING[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/scipy/spatial/distance.py[0m in [0;36m<module>[0;34m[0m
[1;32m    119[0m [0;32mfrom[0m [0;34m.[0m [0;32mimport[0m [0m_hausdorff[0m[0;34m[0m[0;34m[0m[0m
[1;32m    120[0m [0;32mfrom[0m [0;34m.[0m[0;34m.[0m[0mlinalg[0m [0;32mimport[0m [0mnorm[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 121[0;31m [0;32mfrom[0m [0;34m.[0m[0;34m.[0m[0mspecial[0m [0;32mimport[0m [0mrel_entr[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    122[0m [0;34m[0m[0m
[1;32m    123[0m [0;32mfrom[0m [0;34m.[0m [0;32mimport[0m [0m_distance_pybind[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/scipy/special/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m    824[0m     chdtr, chdtrc, betainc, betaincc, stdtr)
[1;32m    825[0m [0;34m[0m[0m
[0;32m--> 826[0;31m [0;32mfrom[0m [0;34m.[0m [0;32mimport[0m [0m_basic[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    827[0m [0;32mfrom[0m [0;34m.[0m[0m_basic[0m [0;32mimport[0m [0;34m*[0m[0;34m[0m[0;34m[0m[0m
[1;32m    828[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/scipy/special/_basic.py[0m in [0;36m<module>[0;34m[0m
[1;32m     20[0m [0;32mfrom[0m [0;34m.[0m [0;32mimport[0m [0m_specfun[0m[0;34m[0m[0;34m[0m[0m
[1;32m     21[0m [0;32mfrom[0m [0;34m.[0m[0m_comb[0m [0;32mimport[0m [0m_comb_int[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 22[0;31m from ._multiufuncs import (assoc_legendre_p_all,
[0m[1;32m     23[0m                            legendre_p_all)
[1;32m     24[0m [0;32mfrom[0m [0mscipy[0m[0;34m.[0m[0m_lib[0m[0;34m.[0m[0mdeprecation[0m [0;32mimport[0m [0m_deprecated[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/scipy/special/_multiufuncs.py[0m in [0;36m<module>[0;34m[0m
[1;32m    140[0m [0;34m[0m[0m
[1;32m    141[0m [0;34m[0m[0m
[0;32m--> 142[0;31m sph_legendre_p = MultiUFunc(
[0m[1;32m    143[0m     [0msph_legendre_p[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    144[0m     r"""sph_legendre_p(n, m, theta, *, diff_n=0)

[0;32m/usr/local/lib/python3.11/dist-packages/scipy/special/_multiufuncs.py[0m in [0;36m__init__[0;34m(self, ufunc_or_ufuncs, doc, force_complex_output, **default_kwargs)[0m
[1;32m     39[0m             [0;32mfor[0m [0mufunc[0m [0;32min[0m [0mufuncs_iter[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     40[0m                 [0;32mif[0m [0;32mnot[0m [0misinstance[0m[0;34m([0m[0mufunc[0m[0;34m,[0m [0mnp[0m[0;34m.[0m[0mufunc[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 41[0;31m                     raise ValueError("All ufuncs must have type `numpy.ufunc`."
[0m[1;32m     42[0m                                      f" Received {ufunc_or_ufuncs}")
[1;32m     43[0m                 [0mseen_input_types[0m[0;34m.[0m[0madd[0m[0;34m([0m[0mfrozenset[0m[0;34m([0m[0mx[0m[0;34m.[0m[0msplit[0m[0;34m([0m[0;34m"->"[0m[0;34m)[0m[0;34m[[0m[0;36m0[0m[0;34m][0m [0;32mfor[0m [0mx[0m [0;32min[0m [0mufunc[0m[0;34m.[0m[0mtypes[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: All ufuncs must have type `numpy.ufunc`. Received (<ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>)

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
