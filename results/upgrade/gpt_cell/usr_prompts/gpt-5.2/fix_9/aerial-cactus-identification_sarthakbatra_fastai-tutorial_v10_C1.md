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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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
import numpy as np
import pandas as pd

import torch

import os
print(os.listdir("../input"))



## === cell 1
%reload_ext autoreload
%autoreload 2
%matplotlib inline


## === cell 2
from fastai import *
from fastai.vision import *


## === cell 3
bs = 64


## === cell 4
from pathlib import Path

path = Path("../input")
path_train = path / "train/train"
path_test = path / "test/test/"
path, path_train, path_test


## === cell 5
labels_df = pd.read_csv(path/'train.csv')
test_df = pd.read_csv(path/'sample_submission.csv')
labels_df.head()


## === cell 6
from fastai.vision.all import *
from fastai.vision.data import *

np.random.seed(42)

item_tfms = Resize(128)
batch_tfms = [
    *aug_transforms(flip_vert=True, max_warp=0.0, max_lighting=0.0),
    Normalize.from_stats(*imagenet_stats),
]

dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_x=ColReader("id", pref=str(path_train) + os.sep),
    get_y=ColReader("has_cactus"),
    splitter=RandomSplitter(valid_pct=0.1, seed=42),
    item_tfms=item_tfms,
    batch_tfms=batch_tfms,
)

data = dblock.dataloaders(labels_df, bs=bs)

test_dl = data.test_dl(test_df["id"].tolist(), with_labels=False, num_workers=0)
data.test_dl = test_dl


## === cell 7
data


## === cell 8
data.show_batch(nrows=3, figsize=(10, 8))


## === cell 9
data.vocab


## === cell 10
learn = cnn_learner(data, models.resnet101, metrics = accuracy, model_dir='/tmp/model/')


## === cell 11
learn.lr_find()


## === cell 12
learn.recorder.plot_loss()


## === cell 13
lr = 2e-02


## === cell 14
learn.fit_one_cycle(3, slice(lr))


## === cell 15
learn.save('resnet-101-1')


## === cell 16
preds, _ = learn.get_preds(dl=data.test_dl)


## --- ERROR in cell 16, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2324375412.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;31m# fastai v2 removed DatasetType; use the test dataloader created in cell 6[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0mpreds[0m[0;34m,[0m [0m_[0m [0;34m=[0m [0mlearn[0m[0;34m.[0m[0mget_preds[0m[0;34m([0m[0mdl[0m[0;34m=[0m[0mdata[0m[0;34m.[0m[0mtest_dl[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
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

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/data/load.py[0m in [0;36m__iter__[0;34m(self)[0m
[1;32m    127[0m         [0mself[0m[0;34m.[0m[0mbefore_iter[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    128[0m         [0mself[0m[0;34m.[0m[0m__idxs[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mget_idxs[0m[0;34m([0m[0;34m)[0m [0;31m# called in context of main process (not workers/subprocesses)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 129[0;31m         [0;32mfor[0m [0mb[0m [0;32min[0m [0m_loaders[0m[0;34m[[0m[0mself[0m[0;34m.[0m[0mfake_l[0m[0;34m.[0m[0mnum_workers[0m[0;34m==[0m[0;36m0[0m[0;34m][0m[0;34m([0m[0mself[0m[0;34m.[0m[0mfake_l[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    130[0m             [0;31m# pin_memory causes tuples to be converted to lists, so convert them back to tuples[0m[0;34m[0m[0;34m[0m[0m
[1;32m    131[0m             [0;32mif[0m [0mself[0m[0;34m.[0m[0mpin_memory[0m [0;32mand[0m [0mtype[0m[0;34m([0m[0mb[0m[0;34m)[0m [0;34m==[0m [0mlist[0m[0;34m:[0m [0mb[0m [0;34m=[0m [0mtuple[0m[0;34m([0m[0mb[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m__next__[0;34m(self)[0m
[1;32m    706[0m                 [0;31m# TODO(https://github.com/pytorch/pytorch/issues/76750)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    707[0m                 [0mself[0m[0;34m.[0m[0m_reset[0m[0;34m([0m[0;34m)[0m  [0;31m# type: ignore[call-arg][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 708[0;31m             [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_next_data[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    709[0m             [0mself[0m[0;34m.[0m[0m_num_yielded[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m    710[0m             if (

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m_next_data[0;34m(self)[0m
[1;32m    762[0m     [0;32mdef[0m [0m_next_data[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    763[0m         [0mindex[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_next_index[0m[0;34m([0m[0;34m)[0m  [0;31m# may raise StopIteration[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 764[0;31m         [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_dataset_fetcher[0m[0;34m.[0m[0mfetch[0m[0;34m([0m[0mindex[0m[0;34m)[0m  [0;31m# may raise StopIteration[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    765[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0m_pin_memory[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    766[0m             [0mdata[0m [0;34m=[0m [0m_utils[0m[0;34m.[0m[0mpin_memory[0m[0;34m.[0m[0mpin_memory[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m_pin_memory_device[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py[0m in [0;36mfetch[0;34m(self, possibly_batched_index)[0m
[1;32m     40[0m                 [0;32mraise[0m [0mStopIteration[0m[0;34m[0m[0;34m[0m[0m
[1;32m     41[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 42[0;31m             [0mdata[0m [0;34m=[0m [0mnext[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mdataset_iter[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     43[0m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mcollate_fn[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     44[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/data/load.py[0m in [0;36mcreate_batches[0;34m(self, samps)[0m
[1;32m    138[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mdataset[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0mit[0m [0;34m=[0m [0miter[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mdataset[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    139[0m         [0mres[0m [0;34m=[0m [0mfilter[0m[0;34m([0m[0;32mlambda[0m [0mo[0m[0;34m:[0m[0mo[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m,[0m [0mmap[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mdo_item[0m[0;34m,[0m [0msamps[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 140[0;31m         [0;32myield[0m [0;32mfrom[0m [0mmap[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mdo_batch[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mchunkify[0m[0;34m([0m[0mres[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    141[0m [0;34m[0m[0m
[1;32m    142[0m     [0;32mdef[0m [0mnew[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mdataset[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0mcls[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastcore/basics.py[0m in [0;36mchunked[0;34m(it, chunk_sz, drop_last, n_chunks, pad, pad_val)[0m
[1;32m    263[0m     [0;32mif[0m [0;32mnot[0m [0misinstance[0m[0;34m([0m[0mit[0m[0;34m,[0m [0mIterator[0m[0;34m)[0m[0;34m:[0m [0mit[0m [0;34m=[0m [0miter[0m[0;34m([0m[0mit[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    264[0m     [0;32mwhile[0m [0;32mTrue[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 265[0;31m         [0mres[0m [0;34m=[0m [0mlist[0m[0;34m([0m[0mitertools[0m[0;34m.[0m[0mislice[0m[0;34m([0m[0mit[0m[0;34m,[0m [0mchunk_sz[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    266[0m         [0;32mif[0m [0mres[0m [0;32mand[0m [0;34m([0m[0mlen[0m[0;34m([0m[0mres[0m[0;34m)[0m[0;34m==[0m[0mchunk_sz[0m [0;32mor[0m [0;32mnot[0m [0mdrop_last[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    267[0m             [0;32mif[0m [0mpad[0m[0;34m:[0m [0;32myield[0m [0mres[0m [0;34m+[0m [0;34m[[0m[0mpad_val[0m[0;34m][0m[0;34m*[0m[0;34m([0m[0mchunk_sz[0m[0;34m-[0m[0mlen[0m[0;34m([0m[0mres[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/data/load.py[0m in [0;36mdo_item[0;34m(self, s)[0m
[1;32m    168[0m     [0;32mdef[0m [0mprebatched[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mbs[0m [0;32mis[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[1;32m    169[0m     [0;32mdef[0m [0mdo_item[0m[0;34m([0m[0mself[0m[0;34m,[0m [0ms[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 170[0;31m         [0;32mtry[0m[0;34m:[0m [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mafter_item[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mcreate_item[0m[0;34m([0m[0ms[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    171[0m         [0;32mexcept[0m [0mSkipItemException[0m[0;34m:[0m [0;32mreturn[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[1;32m    172[0m     [0;32mdef[0m [0mchunkify[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mb[0m[0;34m)[0m[0;34m:[0m [0;32mreturn[0m [0mb[0m [0;32mif[0m [0mself[0m[0;34m.[0m[0mprebatched[0m [0;32melse[0m [0mchunked[0m[0;34m([0m[0mb[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mbs[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mdrop_last[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/data/load.py[0m in [0;36mcreate_item[0;34m(self, s)[0m
[1;32m    175[0m     [0;32mdef[0m [0mretain[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mres[0m[0;34m,[0m [0mb[0m[0;34m)[0m[0;34m:[0m  [0;32mreturn[0m [0mretain_types[0m[0;34m([0m[0mres[0m[0;34m,[0m [0mb[0m[0;34m[[0m[0;36m0[0m[0;34m][0m [0;32mif[0m [0mis_listy[0m[0;34m([0m[0mb[0m[0;34m)[0m [0;32melse[0m [0mb[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    176[0m     [0;32mdef[0m [0mcreate_item[0m[0;34m([0m[0mself[0m[0;34m,[0m [0ms[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 177[0;31m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mindexed[0m[0;34m:[0m [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mdataset[0m[0;34m[[0m[0ms[0m [0;32mor[0m [0;36m0[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    178[0m         [0;32melif[0m [0ms[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m  [0;32mreturn[0m [0mnext[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mit[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    179[0m         [0;32melse[0m[0;34m:[0m [0;32mraise[0m [0mIndexError[0m[0;34m([0m[0;34m"Cannot index an iterable dataset numerically - must use `None`."[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/data/core.py[0m in [0;36m__getitem__[0;34m(self, it)[0m
[1;32m    452[0m [0;34m[0m[0m
[1;32m    453[0m     [0;32mdef[0m [0m__getitem__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mit[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 454[0;31m         [0mres[0m [0;34m=[0m [0mtuple[0m[0;34m([0m[0;34m[[0m[0mtl[0m[0;34m[[0m[0mit[0m[0;34m][0m [0;32mfor[0m [0mtl[0m [0;32min[0m [0mself[0m[0;34m.[0m[0mtls[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    455[0m         [0;32mreturn[0m [0mres[0m [0;32mif[0m [0mis_indexer[0m[0;34m([0m[0mit[0m[0;34m)[0m [0;32melse[0m [0mlist[0m[0;34m([0m[0mzip[0m[0;34m([0m[0;34m*[0m[0mres[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    456[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/data/core.py[0m in [0;36m<listcomp>[0;34m(.0)[0m
[1;32m    452[0m [0;34m[0m[0m
[1;32m    453[0m     [0;32mdef[0m [0m__getitem__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mit[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 454[0;31m         [0mres[0m [0;34m=[0m [0mtuple[0m[0;34m([0m[0;34m[[0m[0mtl[0m[0;34m[[0m[0mit[0m[0;34m][0m [0;32mfor[0m [0mtl[0m [0;32min[0m [0mself[0m[0;34m.[0m[0mtls[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    455[0m         [0;32mreturn[0m [0mres[0m [0;32mif[0m [0mis_indexer[0m[0;34m([0m[0mit[0m[0;34m)[0m [0;32melse[0m [0mlist[0m[0;34m([0m[0mzip[0m[0;34m([0m[0;34m*[0m[0mres[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    456[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/data/core.py[0m in [0;36m__getitem__[0;34m(self, idx)[0m
[1;32m    411[0m         [0mres[0m [0;34m=[0m [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__getitem__[0m[0;34m([0m[0midx[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    412[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0m_after_item[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m [0;32mreturn[0m [0mres[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 413[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_after_item[0m[0;34m([0m[0mres[0m[0;34m)[0m [0;32mif[0m [0mis_indexer[0m[0;34m([0m[0midx[0m[0;34m)[0m [0;32melse[0m [0mres[0m[0;34m.[0m[0mmap[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_after_item[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    414[0m [0;34m[0m[0m
[1;32m    415[0m [0;31m# %% ../../nbs/03_data.core.ipynb 54[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/data/core.py[0m in [0;36m_after_item[0;34m(self, o)[0m
[1;32m    371[0m             [0;32mraise[0m[0;34m[0m[0;34m[0m[0m
[1;32m    372[0m     [0;32mdef[0m [0msubset[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mi[0m[0;34m)[0m[0;34m:[0m [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_new[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_get[0m[0;34m([0m[0mself[0m[0;34m.[0m[0msplits[0m[0;34m[[0m[0mi[0m[0;34m][0m[0;34m)[0m[0;34m,[0m [0msplit_idx[0m[0;34m=[0m[0mi[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 373[0;31m     [0;32mdef[0m [0m_after_item[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mo[0m[0;34m)[0m[0;34m:[0m [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mtfms[0m[0;34m([0m[0mo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    374[0m     [0;32mdef[0m [0m__repr__[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m [0;32mreturn[0m [0;34mf"{self.__class__.__name__}: {self.items}\ntfms - {self.tfms.fs}"[0m[0;34m[0m[0;34m[0m[0m
[1;32m    375[0m     [0;32mdef[0m [0m__iter__[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m [0;32mreturn[0m [0;34m([0m[0mself[0m[0;34m[[0m[0mi[0m[0;34m][0m [0;32mfor[0m [0mi[0m [0;32min[0m [0mrange[0m[0;34m([0m[0mlen[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py[0m in [0;36m__call__[0;34m(self, o)[0m
[1;32m    246[0m         [0mself[0m[0;34m.[0m[0mfs[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mfs[0m[0;34m.[0m[0msorted[0m[0;34m([0m[0mkey[0m[0;34m=[0m[0;34m'order'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    247[0m [0;34m[0m[0m
[0;32m--> 248[0;31m     [0;32mdef[0m [0m__call__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mo[0m[0;34m)[0m[0;34m:[0m [0;32mreturn[0m [0mcompose_tfms[0m[0;34m([0m[0mo[0m[0;34m,[0m [0mtfms[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mfs[0m[0;34m,[0m [0msplit_idx[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0msplit_idx[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    249[0m     [0;32mdef[0m [0m__repr__[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m [0;32mreturn[0m [0;34mf"Pipeline: {' -> '.join([f.name for f in self.fs if f.name != 'noop'])}"[0m[0;34m[0m[0;34m[0m[0m
[1;32m    250[0m     [0;32mdef[0m [0m__getitem__[0m[0;34m([0m[0mself[0m[0;34m,[0m[0mi[0m[0;34m)[0m[0;34m:[0m [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mfs[0m[0;34m[[0m[0mi[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py[0m in [0;36mcompose_tfms[0;34m(x, tfms, is_enc, reverse, **kwargs)[0m
[1;32m    195[0m     [0;32mfor[0m [0mf[0m [0;32min[0m [0mtfms[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    196[0m         [0;32mif[0m [0;32mnot[0m [0mis_enc[0m[0;34m:[0m [0mf[0m [0;34m=[0m [0mf[0m[0;34m.[0m[0mdecode[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 197[0;31m         [0mx[0m [0;34m=[0m [0mf[0m[0;34m([0m[0mx[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    198[0m     [0;32mreturn[0m [0mx[0m[0;34m[0m[0;34m[0m[0m
[1;32m    199[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py[0m in [0;36m__call__[0;34m(self, split_idx, *args, **kwargs)[0m
[1;32m    112[0m         [0mdec[0m [0;34m=[0m [0mlen[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mdecodes[0m[0;34m.[0m[0mmethods[0m[0;34m)[0m [0;32mif[0m [0mhasattr[0m[0;34m([0m[0mself[0m[0;34m,[0m [0;34m'decodes'[0m[0;34m)[0m [0;32melse[0m [0;36m0[0m[0;34m[0m[0;34m[0m[0m
[1;32m    113[0m         [0;32mreturn[0m [0;34mf'{self.name}(enc:{enc},dec:{dec})'[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 114[0;31m     [0;32mdef[0m [0m__call__[0m[0;34m([0m[0mself[0m[0;34m,[0m[0;34m*[0m[0margs[0m[0;34m,[0m[0msplit_idx[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call[0m[0;34m([0m[0;34m'encodes'[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0msplit_idx[0m[0;34m=[0m[0msplit_idx[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    115[0m     [0;32mdef[0m [0mdecode[0m[0;34m([0m[0mself[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m[0msplit_idx[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call[0m[0;34m([0m[0;34m'decodes'[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0msplit_idx[0m[0;34m=[0m[0msplit_idx[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    116[0m     [0;32mdef[0m [0msetup[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mitems[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0mtrain_setup[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py[0m in [0;36m_call[0;34m(self, nm, split_idx, *args, **kwargs)[0m
[1;32m    123[0m         [0;32mif[0m [0msplit_idx[0m[0;34m!=[0m[0mself[0m[0;34m.[0m[0msplit_idx[0m [0;32mand[0m [0mself[0m[0;34m.[0m[0msplit_idx[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m [0;32mreturn[0m [0margs[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m    124[0m         [0;32mif[0m [0;32mnot[0m [0mhasattr[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mnm[0m[0;34m)[0m[0;34m:[0m [0;32mreturn[0m [0margs[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 125[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_do_call[0m[0;34m([0m[0mnm[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    126[0m [0;34m[0m[0m
[1;32m    127[0m     [0;32mdef[0m [0m_do_call[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mnm[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py[0m in [0;36m_do_call[0;34m(self, nm, *args, **kwargs)[0m
[1;32m    134[0m         [0;32mtry[0m[0;34m:[0m [0mmethod[0m[0;34m,[0m [0mret_type[0m [0;34m=[0m [0mf[0m[0;34m.[0m[0m_resolve_method_with_cache[0m[0;34m([0m[0mf_args[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    135[0m         [0;32mexcept[0m [0mNotFoundLookupError[0m[0;34m:[0m [0;32mreturn[0m [0mx[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 136[0;31m         [0;32mreturn[0m [0mretain_type[0m[0;34m([0m[0mmethod[0m[0;34m([0m[0;34m*[0m[0mf_args[0m[0;34m,[0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m,[0m [0mx[0m[0;34m,[0m [0mret_type[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    137[0m [0;34m[0m[0m
[1;32m    138[0m [0madd_docs[0m[0;34m([0m[0mTransform[0m[0;34m,[0m [0mdecode[0m[0;34m=[0m[0;34m"Delegate to decodes to undo transform"[0m[0;34m,[0m [0msetup[0m[0;34m=[0m[0;34m"Delegate to setups to set up transform"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/vision/core.py[0m in [0;36mcreate[0;34m(cls, fn, **kwargs)[0m
[1;32m    125[0m         [0;32mif[0m [0misinstance[0m[0;34m([0m[0mfn[0m[0;34m,[0m[0mbytes[0m[0;34m)[0m[0;34m:[0m [0mfn[0m [0;34m=[0m [0mio[0m[0;34m.[0m[0mBytesIO[0m[0;34m([0m[0mfn[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    126[0m         [0;32mif[0m [0misinstance[0m[0;34m([0m[0mfn[0m[0;34m,[0m[0mImage[0m[0;34m.[0m[0mImage[0m[0;34m)[0m[0;34m:[0m [0;32mreturn[0m [0mcls[0m[0;34m([0m[0mfn[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 127[0;31m         [0;32mreturn[0m [0mcls[0m[0;34m([0m[0mload_image[0m[0;34m([0m[0mfn[0m[0;34m,[0m [0;34m**[0m[0mmerge[0m[0;34m([0m[0mcls[0m[0;34m.[0m[0m_open_args[0m[0;34m,[0m [0mkwargs[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    128[0m [0;34m[0m[0m
[1;32m    129[0m     [0;32mdef[0m [0mshow[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mctx[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/vision/core.py[0m in [0;36mload_image[0;34m(fn, mode)[0m
[1;32m     98[0m [0;32mdef[0m [0mload_image[0m[0;34m([0m[0mfn[0m[0;34m,[0m [0mmode[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     99[0m     [0;34m"Open and load a `PIL.Image` and convert to `mode`"[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 100[0;31m     [0mim[0m [0;34m=[0m [0mImage[0m[0;34m.[0m[0mopen[0m[0;34m([0m[0mfn[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    101[0m     [0mim[0m[0;34m.[0m[0mload[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    102[0m     [0mim[0m [0;34m=[0m [0mim[0m[0;34m.[0m[0m_new[0m[0;34m([0m[0mim[0m[0;34m.[0m[0mim[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/PIL/Image.py[0m in [0;36mopen[0;34m(fp, mode, formats)[0m
[1;32m   3511[0m     [0;32mif[0m [0mis_path[0m[0;34m([0m[0mfp[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3512[0m         [0mfilename[0m [0;34m=[0m [0mos[0m[0;34m.[0m[0mfspath[0m[0;34m([0m[0mfp[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3513[0;31m         [0mfp[0m [0;34m=[0m [0mbuiltins[0m[0;34m.[0m[0mopen[0m[0;34m([0m[0mfilename[0m[0;34m,[0m [0;34m"rb"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3514[0m         [0mexclusive_fp[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3515[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '09034a34de0e2015a8a28dfe18f423f6.jpg'

## === cell 17
preds[:, 0]
