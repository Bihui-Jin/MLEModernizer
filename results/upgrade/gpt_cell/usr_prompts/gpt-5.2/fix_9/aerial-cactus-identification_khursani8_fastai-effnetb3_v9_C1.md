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

fastai==2.8.5
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))



## === cell 1
from pathlib import Path
from fastai.vision import *

path = Path("../input/")


## === cell 2
sorted(path.iterdir())


## === cell 3
from sklearn.metrics import roc_auc_score

import torch
import torch.nn.functional as F
from fastai.callback.core import Callback


def auroc_score(input, target):
    input, target = input.cpu().numpy()[:, 1], target.cpu().numpy()
    return roc_auc_score(target, input)


class AUROC(Callback):
    _order = -20  # Needs to run before the recorder

    def __init__(self, learn, **kwargs):
        self.learn = learn

    def on_train_begin(self, **kwargs):
        self.learn.recorder.add_metric_names(["AUROC"])

    def on_epoch_begin(self, **kwargs):
        self.output, self.target = [], []

    def on_batch_end(self, last_target, last_output, train, **kwargs):
        if not train:
            self.output.append(last_output)
            self.target.append(last_target)

    def on_epoch_end(self, last_metrics, **kwargs):
        if len(self.output) > 0:
            output = torch.cat(self.output)
            target = torch.cat(self.target)
            preds = F.softmax(output, dim=1)
            metric = auroc_score(preds, target)
            return add_metrics(last_metrics, [metric])

    def before_fit(self):
        if hasattr(self, "learn") and hasattr(self.learn, "recorder"):
            try:
                self.learn.recorder.add_metric_names(["AUROC"])
            except Exception:
                pass

    def before_epoch(self):
        self.output, self.target = [], []

    def after_batch(self):
        if getattr(self, "training", False):
            return
        if hasattr(self, "pred") and hasattr(self, "y"):
            self.output.append(self.pred.detach())
            self.target.append(self.y.detach())

    def after_epoch(self):
        if len(getattr(self, "output", [])) == 0:
            return
        output = torch.cat(self.output)
        target = torch.cat(self.target)
        preds = F.softmax(output, dim=1)
        metric = auroc_score(preds, target)
        return metric


## === cell 4
train_df = pd.read_csv(path/'train.csv')
test_df = pd.read_csv(path/'sample_submission.csv')


## === cell 5
from fastai.vision.all import *
from torchvision import models as _tv_models

models = _tv_models

tfms = aug_transforms(
    do_flip=True,
    flip_vert=True,
    max_rotate=10.0,
    max_zoom=1.1,
    max_lighting=0.2,
    max_warp=0.2,
    p_affine=0.75,
    p_lighting=0.75,
)

train_img = ImageDataLoaders.from_df(
    train_df,
    path=path,
    folder="train",
    valid_pct=0.2,
    seed=42,
    fn_col="id",
    label_col="has_cactus",
    item_tfms=Resize(128, method=ResizeMethod.Squish),
    batch_tfms=[*tfms, Normalize.from_stats(*imagenet_stats)],
    bs=64,
)

test_files = [path / "test" / f for f in test_df["id"].tolist()]
test_img = train_img.test_dl(test_files)


## === cell 6

from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from fastai.callback.core import Callback


def auroc_score(input, target):
    input, target = input.cpu().numpy()[:, 1], target.cpu().numpy()
    return roc_auc_score(target, input)


class AUROC(Callback):
    _order = -20  # Needs to run before the recorder

    def __init__(self, **kwargs):
        pass

    def before_epoch(self):
        self.output, self.target = [], []

    def after_batch(self):
        if getattr(self, "training", False):
            return
        if hasattr(self, "pred") and hasattr(self, "y"):
            self.output.append(self.pred.detach())
            self.target.append(self.y.detach())

    def after_epoch(self):
        if len(getattr(self, "output", [])) == 0:
            return
        output = torch.cat(self.output)
        target = torch.cat(self.target)
        preds = F.softmax(output, dim=1)
        metric = auroc_score(preds, target)
        return metric


learn = vision_learner(train_img, models.resnet50, metrics=[accuracy], cbs=[AUROC()])


## === cell 8
lr=3e-2


## === cell 9
learn.fit_one_cycle(1,lr)


## === cell 10
learn.unfreeze()
learn.fit_one_cycle(1,slice(lr/10))


## === cell 11
train_img = train_img


## === cell 12
learn.data = train_img


## === cell 13
learn.unfreeze()
learn.fit_one_cycle(1,slice(lr/100))


## === cell 14
tta_out = learn.tta(dl=test_img)
preds = tta_out[0] if isinstance(tta_out, (tuple, list)) else tta_out

test_df.has_cactus = preds.numpy()[:, 0]
test_df.to_csv("submission.csv", index=False)


## --- ERROR in cell 14, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1305680153.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;31m# fastai v2 uses `tta` (lowercase) instead of fastai v1 `TTA` (uppercase).[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;31m# Also, pass the already-created test dataloader to ensure we predict on the test set.[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 3[0;31m [0mtta_out[0m [0;34m=[0m [0mlearn[0m[0;34m.[0m[0mtta[0m[0;34m([0m[0mdl[0m[0;34m=[0m[0mtest_img[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0mpreds[0m [0;34m=[0m [0mtta_out[0m[0;34m[[0m[0;36m0[0m[0;34m][0m [0;32mif[0m [0misinstance[0m[0;34m([0m[0mtta_out[0m[0;34m,[0m [0;34m([0m[0mtuple[0m[0;34m,[0m [0mlist[0m[0;34m)[0m[0;34m)[0m [0;32melse[0m [0mtta_out[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36mtta[0;34m(self, ds_idx, dl, n, item_tfms, batch_tfms, beta, use_max)[0m
[1;32m    688[0m             [0;32mfor[0m [0mi[0m [0;32min[0m [0mself[0m[0;34m.[0m[0mprogress[0m[0;34m.[0m[0mmbar[0m [0;32mif[0m [0mhasattr[0m[0;34m([0m[0mself[0m[0;34m,[0m[0;34m'progress'[0m[0;34m)[0m [0;32melse[0m [0mrange[0m[0;34m([0m[0mn[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    689[0m                 [0mself[0m[0;34m.[0m[0mepoch[0m [0;34m=[0m [0mi[0m [0;31m#To keep track of progress on mbar since the progress callback will use self.epoch[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 690[0;31m                 [0maug_preds[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mget_preds[0m[0;34m([0m[0mdl[0m[0;34m=[0m[0mdl[0m[0;34m,[0m [0minner[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m[[0m[0;32mNone[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    691[0m         [0maug_preds[0m [0;34m=[0m [0mtorch[0m[0;34m.[0m[0mcat[0m[0;34m([0m[0maug_preds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    692[0m         [0maug_preds[0m [0;34m=[0m [0maug_preds[0m[0;34m.[0m[0mmax[0m[0;34m([0m[0;36m0[0m[0;34m)[0m[0;34m[[0m[0;36m0[0m[0;34m][0m [0;32mif[0m [0muse_max[0m [0;32melse[0m [0maug_preds[0m[0;34m.[0m[0mmean[0m[0;34m([0m[0;36m0[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36mget_preds[0;34m(self, ds_idx, dl, with_input, with_decoded, with_loss, act, inner, reorder, cbs, **kwargs)[0m
[1;32m    314[0m         [0;32mif[0m [0mwith_loss[0m[0;34m:[0m [0mctx_mgrs[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mloss_not_reduced[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    315[0m         [0;32mwith[0m [0mContextManagers[0m[0;34m([0m[0mctx_mgrs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 316[0;31m             [0mself[0m[0;34m.[0m[0m_do_epoch_validate[0m[0;34m([0m[0mdl[0m[0;34m=[0m[0mdl[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    317[0m             [0;32mif[0m [0mact[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m [0mact[0m [0;34m=[0m [0mgetcallable[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mloss_func[0m[0;34m,[0m [0;34m'activation'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    318[0m             [0mres[0m [0;34m=[0m [0mcb[0m[0;34m.[0m[0mall_tensors[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_do_epoch_validate[0;34m(self, ds_idx, dl)[0m
[1;32m    250[0m         [0;32mif[0m [0mdl[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m [0mdl[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mdls[0m[0;34m[[0m[0mds_idx[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m    251[0m         [0mself[0m[0;34m.[0m[0mdl[0m [0;34m=[0m [0mdl[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 252[0;31m         [0;32mwith[0m [0mtorch[0m[0;34m.[0m[0mno_grad[0m[0;34m([0m[0;34m)[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0m_with_events[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mall_batches[0m[0;34m,[0m [0;34m'validate'[0m[0;34m,[0m [0mCancelValidException[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    253[0m [0;34m[0m[0m
[1;32m    254[0m     [0;32mdef[0m [0m_do_epoch[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_with_events[0;34m(self, f, event_type, ex, final)[0m
[1;32m    205[0m [0;34m[0m[0m
[1;32m    206[0m     [0;32mdef[0m [0m_with_events[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mf[0m[0;34m,[0m [0mevent_type[0m[0;34m,[0m [0mex[0m[0;34m,[0m [0mfinal[0m[0;34m=[0m[0mnoop[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 207[0;31m         [0;32mtry[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'before_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mf[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    208[0m         [0;32mexcept[0m [0mex[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'after_cancel_{event_type}'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    209[0m         [0mself[0m[0;34m([0m[0;34mf'after_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mfinal[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36mall_batches[0;34m(self)[0m
[1;32m    211[0m     [0;32mdef[0m [0mall_batches[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    212[0m         [0mself[0m[0;34m.[0m[0mn_iter[0m [0;34m=[0m [0mlen[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mdl[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 213[0;31m         [0;32mfor[0m [0mo[0m [0;32min[0m [0menumerate[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mdl[0m[0;34m)[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0mone_batch[0m[0;34m([0m[0;34m*[0m[0mo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    214[0m [0;34m[0m[0m
[1;32m    215[0m     [0;32mdef[0m [0m_backward[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0mloss_grad[0m[0;34m.[0m[0mbackward[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36mone_batch[0;34m(self, i, b)[0m
[1;32m    241[0m         [0mb[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_set_device[0m[0;34m([0m[0mb[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    242[0m         [0mself[0m[0;34m.[0m[0m_split[0m[0;34m([0m[0mb[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 243[0;31m         [0mself[0m[0;34m.[0m[0m_with_events[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_do_one_batch[0m[0;34m,[0m [0;34m'batch'[0m[0;34m,[0m [0mCancelBatchException[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    244[0m [0;34m[0m[0m
[1;32m    245[0m     [0;32mdef[0m [0m_do_epoch_train[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_with_events[0;34m(self, f, event_type, ex, final)[0m
[1;32m    207[0m         [0;32mtry[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'before_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mf[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    208[0m         [0;32mexcept[0m [0mex[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'after_cancel_{event_type}'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 209[0;31m         [0mself[0m[0;34m([0m[0;34mf'after_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mfinal[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    210[0m [0;34m[0m[0m
[1;32m    211[0m     [0;32mdef[0m [0mall_batches[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m__call__[0;34m(self, event_name)[0m
[1;32m    178[0m [0;34m[0m[0m
[1;32m    179[0m     [0;32mdef[0m [0mordered_cbs[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mevent[0m[0;34m)[0m[0;34m:[0m [0;32mreturn[0m [0;34m[[0m[0mcb[0m [0;32mfor[0m [0mcb[0m [0;32min[0m [0mself[0m[0;34m.[0m[0mcbs[0m[0;34m.[0m[0msorted[0m[0;34m([0m[0;34m'order'[0m[0;34m)[0m [0;32mif[0m [0mhasattr[0m[0;34m([0m[0mcb[0m[0;34m,[0m [0mevent[0m[0;34m)[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 180[0;31m     [0;32mdef[0m [0m__call__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mevent_name[0m[0;34m)[0m[0;34m:[0m [0mL[0m[0;34m([0m[0mevent_name[0m[0;34m)[0m[0;34m.[0m[0mmap[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_call_one[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    181[0m [0;34m[0m[0m
[1;32m    182[0m     [0;32mdef[0m [0m_call_one[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mevent_name[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py[0m in [0;36mmap[0;34m(self, f, *args, **kwargs)[0m
[1;32m    166[0m     [0;32mdef[0m [0mrange[0m[0;34m([0m[0mcls[0m[0;34m,[0m [0ma[0m[0;34m,[0m [0mb[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0mstep[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m:[0m [0;32mreturn[0m [0mcls[0m[0;34m([0m[0mrange_of[0m[0;34m([0m[0ma[0m[0;34m,[0m [0mb[0m[0;34m=[0m[0mb[0m[0;34m,[0m [0mstep[0m[0;34m=[0m[0mstep[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    167[0m [0;34m[0m[0m
[0;32m--> 168[0;31m     [0;32mdef[0m [0mmap[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mf[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_new[0m[0;34m([0m[0mmap_ex[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mf[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0mgen[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    169[0m     [0;32mdef[0m [0margwhere[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mf[0m[0;34m,[0m [0mnegate[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_new[0m[0;34m([0m[0margwhere[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mf[0m[0;34m,[0m [0mnegate[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    170[0m     [0;32mdef[0m [0margfirst[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mf[0m[0;34m,[0m [0mnegate[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastcore/basics.py[0m in [0;36mmap_ex[0;34m(iterable, f, gen, *args, **kwargs)[0m
[1;32m    949[0m     [0mres[0m [0;34m=[0m [0mmap[0m[0;34m([0m[0mg[0m[0;34m,[0m [0miterable[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    950[0m     [0;32mif[0m [0mgen[0m[0;34m:[0m [0;32mreturn[0m [0mres[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 951[0;31m     [0;32mreturn[0m [0mlist[0m[0;34m([0m[0mres[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    952[0m [0;34m[0m[0m
[1;32m    953[0m [0;31m# %% ../nbs/01_basics.ipynb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastcore/basics.py[0m in [0;36m__call__[0;34m(self, *args, **kwargs)[0m
[1;32m    934[0m             [0;32mif[0m [0misinstance[0m[0;34m([0m[0mv[0m[0;34m,[0m[0m_Arg[0m[0;34m)[0m[0;34m:[0m [0mkwargs[0m[0;34m[[0m[0mk[0m[0;34m][0m [0;34m=[0m [0margs[0m[0;34m.[0m[0mpop[0m[0;34m([0m[0mv[0m[0;34m.[0m[0mi[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    935[0m         [0mfargs[0m [0;34m=[0m [0;34m[[0m[0margs[0m[0;34m[[0m[0mx[0m[0;34m.[0m[0mi[0m[0;34m][0m [0;32mif[0m [0misinstance[0m[0;34m([0m[0mx[0m[0;34m,[0m [0m_Arg[0m[0;34m)[0m [0;32melse[0m [0mx[0m [0;32mfor[0m [0mx[0m [0;32min[0m [0mself[0m[0;34m.[0m[0mpargs[0m[0;34m][0m [0;34m+[0m [0margs[0m[0;34m[[0m[0mself[0m[0;34m.[0m[0mmaxi[0m[0;34m+[0m[0;36m1[0m[0;34m:[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 936[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mfunc[0m[0;34m([0m[0;34m*[0m[0mfargs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    937[0m [0;34m[0m[0m
[1;32m    938[0m [0;31m# %% ../nbs/01_basics.ipynb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_call_one[0;34m(self, event_name)[0m
[1;32m    182[0m     [0;32mdef[0m [0m_call_one[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mevent_name[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    183[0m         [0;32mif[0m [0;32mnot[0m [0mhasattr[0m[0;34m([0m[0mevent[0m[0;34m,[0m [0mevent_name[0m[0;34m)[0m[0;34m:[0m [0;32mraise[0m [0mException[0m[0;34m([0m[0;34mf'missing {event_name}'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 184[0;31m         [0;32mfor[0m [0mcb[0m [0;32min[0m [0mself[0m[0;34m.[0m[0mcbs[0m[0;34m.[0m[0msorted[0m[0;34m([0m[0;34m'order'[0m[0;34m)[0m[0;34m:[0m [0mcb[0m[0;34m([0m[0mevent_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    185[0m [0;34m[0m[0m
[1;32m    186[0m     [0;32mdef[0m [0m_bn_bias_state[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mwith_bias[0m[0;34m)[0m[0;34m:[0m [0;32mreturn[0m [0mnorm_bias_params[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mmodel[0m[0;34m,[0m [0mwith_bias[0m[0;34m)[0m[0;34m.[0m[0mmap[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mopt[0m[0;34m.[0m[0mstate[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/callback/core.py[0m in [0;36m__call__[0;34m(self, event_name)[0m
[1;32m     62[0m             [0;32mtry[0m[0;34m:[0m [0mres[0m [0;34m=[0m [0mgetcallable[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mevent_name[0m[0;34m)[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     63[0m             [0;32mexcept[0m [0;34m([0m[0mCancelBatchException[0m[0;34m,[0m [0mCancelBackwardException[0m[0;34m,[0m [0mCancelEpochException[0m[0;34m,[0m [0mCancelFitException[0m[0;34m,[0m [0mCancelStepException[0m[0;34m,[0m [0mCancelTrainException[0m[0;34m,[0m [0mCancelValidException[0m[0;34m)[0m[0;34m:[0m [0;32mraise[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 64[0;31m             [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me[0m[0;34m:[0m [0;32mraise[0m [0mmodify_exception[0m[0;34m([0m[0me[0m[0;34m,[0m [0;34mf'Exception occured in `{self.__class__.__name__}` when calling event `{event_name}`:\n\t{e.args[0]}'[0m[0;34m,[0m [0mreplace[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     65[0m         [0;32mif[0m [0mevent_name[0m[0;34m==[0m[0;34m'after_fit'[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0mrun[0m[0;34m=[0m[0;32mTrue[0m [0;31m#Reset self.run to True at each end of fit[0m[0;34m[0m[0;34m[0m[0m
[1;32m     66[0m         [0;32mreturn[0m [0mres[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/callback/core.py[0m in [0;36m__call__[0;34m(self, event_name)[0m
[1;32m     60[0m         [0mres[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[1;32m     61[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mrun[0m [0;32mand[0m [0m_run[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 62[0;31m             [0;32mtry[0m[0;34m:[0m [0mres[0m [0;34m=[0m [0mgetcallable[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mevent_name[0m[0;34m)[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     63[0m             [0;32mexcept[0m [0;34m([0m[0mCancelBatchException[0m[0;34m,[0m [0mCancelBackwardException[0m[0;34m,[0m [0mCancelEpochException[0m[0;34m,[0m [0mCancelFitException[0m[0;34m,[0m [0mCancelStepException[0m[0;34m,[0m [0mCancelTrainException[0m[0;34m,[0m [0mCancelValidException[0m[0;34m)[0m[0;34m:[0m [0;32mraise[0m[0;34m[0m[0;34m[0m[0m
[1;32m     64[0m             [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me[0m[0;34m:[0m [0;32mraise[0m [0mmodify_exception[0m[0;34m([0m[0me[0m[0;34m,[0m [0;34mf'Exception occured in `{self.__class__.__name__}` when calling event `{event_name}`:\n\t{e.args[0]}'[0m[0;34m,[0m [0mreplace[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3997081177.py[0m in [0;36mafter_batch[0;34m(self)[0m
[1;32m     28[0m         [0;32mif[0m [0mhasattr[0m[0;34m([0m[0mself[0m[0;34m,[0m [0;34m"pred"[0m[0;34m)[0m [0;32mand[0m [0mhasattr[0m[0;34m([0m[0mself[0m[0;34m,[0m [0;34m"y"[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     29[0m             [0mself[0m[0;34m.[0m[0moutput[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mpred[0m[0;34m.[0m[0mdetach[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 30[0;31m             [0mself[0m[0;34m.[0m[0mtarget[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mself[0m[0;34m.[0m[0my[0m[0;34m.[0m[0mdetach[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     31[0m [0;34m[0m[0m
[1;32m     32[0m     [0;32mdef[0m [0mafter_epoch[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: Exception occured in `AUROC` when calling event `after_batch`:
	'NoneType' object has no attribute 'detach'
