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
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))



## === cell 1
%reload_ext autoreload
%autoreload 2
%matplotlib inline


## === cell 2
import torch
from fastai.vision.all import *


## === cell 3
torch.cuda.set_device(0)


## === cell 4
torch.backends.cudnn.enabled


## === cell 5
PATH = '../input/'
sz = 224
arch = resnet101
bs = 128


## === cell 6
!ln -s {PATH}train
!ln -s {PATH}test
!ls


## === cell 7
label_csv = f"{PATH}labels.csv"
n = len(list(open(label_csv))) - 1  # header is not counted (-1)


def get_cv_idxs(n, cv_idx=0, val_pct=0.2, seed=42):
    rng = np.random.RandomState(seed + int(cv_idx))
    idxs = np.arange(n)
    rng.shuffle(idxs)
    n_val = int(round(n * val_pct))
    return idxs[:n_val]


val_idxs = get_cv_idxs(n)  # random 20% data for validation set


## === cell 8
def get_data(sz,bs):
    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
    data = ImageClassifierData.from_csv('.', 'train', label_csv, test_name='test',
                                       val_idxs=val_idxs, suffix='.jpg', tfms=tfms, bs=bs)
    return data if sz > 300 else data.resize(340, 'tmp')


## === cell 9
from types import SimpleNamespace


def get_data(sz, bs):
    df = pd.read_csv(label_csv)

    df["fname"] = df["id"].astype(str) + ".jpg"

    dls = ImageDataLoaders.from_df(
        df,
        path=".",
        fn_col="fname",
        folder="train",
        label_col="breed",
        valid_idx=list(val_idxs),
        item_tfms=Resize(sz),
        batch_tfms=aug_transforms(max_zoom=1.1),
        bs=bs,
    )

    trn_items = dls.train.items
    dls.trn_ds = SimpleNamespace(
        fnames=[p.name if hasattr(p, "name") else str(p) for p in trn_items]
    )

    return dls


data = get_data(sz, bs)


## === cell 10
from PIL import Image
from pathlib import Path
import pandas as pd

items = data.train.items
if hasattr(items, "iloc"):
    first_item = items.iloc[0]
else:
    try:
        first_item = items[0]
    except Exception:
        first_item = next(iter(items))

if isinstance(first_item, pd.Series):
    if "fname" in first_item:
        fname = str(first_item["fname"])
    elif "id" in first_item:
        fname = str(first_item["id"]) + ".jpg"
    else:
        fname = str(first_item)
else:
    fname = (
        Path(first_item).name
        if hasattr(first_item, "__fspath__") or isinstance(first_item, (str, Path))
        else str(first_item)
    )
    if not fname.lower().endswith(".jpg"):
        fname = f"{fname}.jpg"

fn = str(Path(PATH) / "train" / fname)
fn
img = Image.open(fn)
img


## === cell 11
learn = cnn_learner(data, arch, pretrained=True)


## === cell 12
learning_rate = learn.lr_find()
learn.recorder.plot_lr_find()


## === cell 13
learn.fit(1e-2, 5)


## --- ERROR in cell 13, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2845758315.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mlearn[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0;36m1e-2[0m[0;34m,[0m [0;36m5[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36mfit[0;34m(self, n_epoch, lr, wd, cbs, reset_opt, start_epoch)[0m
[1;32m    270[0m             [0mself[0m[0;34m.[0m[0mopt[0m[0;34m.[0m[0mset_hypers[0m[0;34m([0m[0mlr[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mlr[0m [0;32mif[0m [0mlr[0m [0;32mis[0m [0;32mNone[0m [0;32melse[0m [0mlr[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    271[0m             [0mself[0m[0;34m.[0m[0mn_epoch[0m [0;34m=[0m [0mn_epoch[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 272[0;31m             [0mself[0m[0;34m.[0m[0m_with_events[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_do_fit[0m[0;34m,[0m [0;34m'fit'[0m[0;34m,[0m [0mCancelFitException[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m_end_cleanup[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    273[0m [0;34m[0m[0m
[1;32m    274[0m     [0;32mdef[0m [0m_end_cleanup[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0mdl[0m[0;34m,[0m[0mself[0m[0;34m.[0m[0mxb[0m[0;34m,[0m[0mself[0m[0;34m.[0m[0myb[0m[0;34m,[0m[0mself[0m[0;34m.[0m[0mpred[0m[0;34m,[0m[0mself[0m[0;34m.[0m[0mloss[0m [0;34m=[0m [0;32mNone[0m[0;34m,[0m[0;34m([0m[0;32mNone[0m[0;34m,[0m[0;34m)[0m[0;34m,[0m[0;34m([0m[0;32mNone[0m[0;34m,[0m[0;34m)[0m[0;34m,[0m[0;32mNone[0m[0;34m,[0m[0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_with_events[0;34m(self, f, event_type, ex, final)[0m
[1;32m    205[0m [0;34m[0m[0m
[1;32m    206[0m     [0;32mdef[0m [0m_with_events[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mf[0m[0;34m,[0m [0mevent_type[0m[0;34m,[0m [0mex[0m[0;34m,[0m [0mfinal[0m[0;34m=[0m[0mnoop[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 207[0;31m         [0;32mtry[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'before_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mf[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    208[0m         [0;32mexcept[0m [0mex[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'after_cancel_{event_type}'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    209[0m         [0mself[0m[0;34m([0m[0;34mf'after_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mfinal[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

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

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/callback/progress.py[0m in [0;36mbefore_fit[0;34m(self)[0m
[1;32m     18[0m     [0;32mdef[0m [0mbefore_fit[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     19[0m         [0;32massert[0m [0mhasattr[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mlearn[0m[0;34m,[0m [0;34m'recorder'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 20[0;31m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mcreate_mbar[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0mmbar[0m [0;34m=[0m [0mmaster_bar[0m[0;34m([0m[0mlist[0m[0;34m([0m[0mrange[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mn_epoch[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     21[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mlearn[0m[0;34m.[0m[0mlogger[0m [0;34m!=[0m [0mnoop[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     22[0m             [0mself[0m[0;34m.[0m[0mold_logger[0m[0;34m,[0m[0mself[0m[0;34m.[0m[0mlearn[0m[0;34m.[0m[0mlogger[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mlogger[0m[0;34m,[0m[0mself[0m[0;34m.[0m[0m_write_stats[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: Exception occured in `ProgressCallback` when calling event `before_fit`:
	'float' object cannot be interpreted as an integer

## === cell 14
from sklearn import metrics
