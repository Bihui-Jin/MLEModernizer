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
import numpy as np
import pandas as pd

from fastai.vision.all import *
from fastai.callback.all import *

from pathlib import Path

import os
import shutil

np.random.seed(10)

try:
    get_ipython().run_line_magic("matplotlib", "inline")
    get_ipython().run_line_magic("reload_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except Exception:
    pass


## === cell 1
data_folder = Path("../input")
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/sample_submission.csv")


## === cell 2

item_tfms = Resize(128)
batch_tfms = aug_transforms(
    do_flip=True,
    flip_vert=True,
    max_rotate=12.0,
    max_zoom=1.1,
    max_lighting=0.2,
    max_warp=0.2,
    p_affine=0.75,
    p_lighting=0.75,
) + [Normalize.from_stats(*imagenet_stats)]

train_img = ImageDataLoaders.from_df(
    train_df,
    path=data_folder,  # base path
    folder="train",  # images live in ../input/train/
    valid_pct=0.01,
    seed=10,
    fn_col="id",
    label_col="has_cactus",
    y_block=CategoryBlock,
    item_tfms=item_tfms,
    batch_tfms=batch_tfms,
    bs=64,
)


## === cell 3
learn_densnet = cnn_learner(train_img, models.densenet201, metrics=[error_rate, accuracy], model_dir="/tmp/model/")


## === cell 4
lr_find_res = learn_densnet.lr_find()


## === cell 5
lr = 1e-02
learn_densnet.fit_one_cycle(5 , slice(lr))


## === cell 6
test_dl = learn_densnet.dls.test_dl(
    test_df, with_labels=False, path=data_folder, folder="test"
)
preds, _ = learn_densnet.get_preds(dl=test_dl)


## --- ERROR in cell 6, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_12/4028465675.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      4[0m     [0mtest_df[0m[0;34m,[0m [0mwith_labels[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m [0mpath[0m[0;34m=[0m[0mdata_folder[0m[0;34m,[0m [0mfolder[0m[0;34m=[0m[0;34m"test"[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m )
[0;32m----> 6[0;31m [0mpreds[0m[0;34m,[0m [0m_[0m [0;34m=[0m [0mlearn_densnet[0m[0;34m.[0m[0mget_preds[0m[0;34m([0m[0mdl[0m[0;34m=[0m[0mtest_dl[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
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
[1;32m   1478[0m                 [0;32mdel[0m [0mself[0m[0;34m.[0m[0m_task_info[0m[0;34m[[0m[0midx[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1479[0m                 [0mself[0m[0;34m.[0m[0m_rcvd_idx[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1480[0;31m                 [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_process_data[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1481[0m [0;34m[0m[0m
[1;32m   1482[0m     [0;32mdef[0m [0m_try_put_index[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m_process_data[0;34m(self, data)[0m
[1;32m   1503[0m         [0mself[0m[0;34m.[0m[0m_try_put_index[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1504[0m         [0;32mif[0m [0misinstance[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mExceptionWrapper[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1505[0;31m             [0mdata[0m[0;34m.[0m[0mreraise[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1506[0m         [0;32mreturn[0m [0mdata[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1507[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_utils.py[0m in [0;36mreraise[0;34m(self)[0m
[1;32m    731[0m             [0;31m# instantiate since we don't know how to[0m[0;34m[0m[0;34m[0m[0m
[1;32m    732[0m             [0;32mraise[0m [0mRuntimeError[0m[0;34m([0m[0mmsg[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 733[0;31m         [0;32mraise[0m [0mexception[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    734[0m [0;34m[0m[0m
[1;32m    735[0m [0;34m[0m[0m

[0;31mFileNotFoundError[0m: Caught FileNotFoundError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 42, in fetch
    data = next(self.dataset_iter)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/load.py", line 140, in create_batches
    yield from map(self.do_batch, self.chunkify(res))
  File "/usr/local/lib/python3.11/dist-packages/fastcore/basics.py", line 265, in chunked
    res = list(itertools.islice(it, chunk_sz))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/load.py", line 170, in do_item
    try: return self.after_item(self.create_item(s))
                                ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/load.py", line 177, in create_item
    if self.indexed: return self.dataset[s or 0]
                            ~~~~~~~~~~~~^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/core.py", line 454, in __getitem__
    res = tuple([tl[it] for tl in self.tls])
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/core.py", line 454, in <listcomp>
    res = tuple([tl[it] for tl in self.tls])
                 ~~^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/core.py", line 413, in __getitem__
    return self._after_item(res) if is_indexer(idx) else res.map(self._after_item)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/core.py", line 373, in _after_item
    def _after_item(self, o): return self.tfms(o)
                                     ^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py", line 248, in __call__
    def __call__(self, o): return compose_tfms(o, tfms=self.fs, split_idx=self.split_idx)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py", line 197, in compose_tfms
    x = f(x, **kwargs)
        ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py", line 114, in __call__
    def __call__(self,*args,split_idx=None, **kwargs): return self._call('encodes', *args, split_idx=split_idx, **kwargs)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py", line 125, in _call
    return self._do_call(nm, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py", line 136, in _do_call
    return retain_type(method(*f_args,**kwargs), x, ret_type)
                       ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/vision/core.py", line 127, in create
    return cls(load_image(fn, **merge(cls._open_args, kwargs)))
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/vision/core.py", line 100, in load_image
    im = Image.open(fn)
         ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/PIL/Image.py", line 3513, in open
    fp = builtins.open(filename, "rb")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '../input/train/09034a34de0e2015a8a28dfe18f423f6.jpg'


## === cell 7
test_df.has_cactus = preds.numpy()[:, 0]
test_df.head()
