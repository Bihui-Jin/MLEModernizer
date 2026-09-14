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
from fastai.vision import *

try:
    from fastai.vision.all import Resize
except Exception:
    pass

try:
    from fastai.vision.transform import get_transforms  # fastai v1
except ModuleNotFoundError:
    from fastai.vision.all import aug_transforms as _aug_transforms

    def get_transforms(flip_vert=False, max_warp=0.0, **kwargs):
        return _aug_transforms(flip_vert=flip_vert, max_warp=max_warp, **kwargs)


try:
    ImageList
except NameError:
    from fastai.vision.all import (
        ImageDataLoaders,
        imagenet_stats,
    )  # imagenet_stats for normalize()

    class ImageList:
        @classmethod
        def from_df(cls, df, path):
            obj = cls()
            obj._df = df.copy()
            obj._path = Path(path)
            obj._test = None
            obj._valid_pct = None
            obj._label_col = None
            obj._tfms = None
            obj._size = None
            obj._bs = None
            obj._dls_path = None
            obj._norm_stats = None
            return obj

        def split_by_rand_pct(self, valid_pct):
            self._valid_pct = valid_pct
            return self

        def label_from_df(self, label_cls=None, cols=None):
            if cols is None:
                cols = self._df.columns[1]
            self._label_col = cols
            return self

        def add_test(self, test):
            self._test = test
            return self

        def transform(self, tfms, size=None):
            self._tfms = tfms
            self._size = size
            return self

        def databunch(self, path=None, bs=64, **kwargs):
            self._bs = bs
            self._dls_path = Path(path) if path is not None else None

            train_df = self._df
            files = train_df.iloc[:, 0].astype(str).tolist()

            item_tfms = [Resize(self._size)] if self._size is not None else None
            batch_tfms = self._tfms if self._tfms is not None else None

            dls = ImageDataLoaders.from_name_func(
                self._path,
                files,
                label_func=lambda fn: int(
                    train_df.loc[
                        train_df.iloc[:, 0].astype(str) == Path(fn).name,
                        self._label_col,
                    ].values[0]
                ),
                valid_pct=self._valid_pct if self._valid_pct is not None else 0.2,
                seed=42,
                item_tfms=item_tfms,
                batch_tfms=batch_tfms,
                bs=self._bs,
            )

            if self._test is not None:
                test_files = self._test._df.iloc[:, 0].astype(str).tolist()
                dls.test = dls.test_dl([self._test._path / f for f in test_files])
            return dls


try:
    from fastai.vision.transform import get_transforms as _unused  # fastai v1 only
except ModuleNotFoundError:
    _unused = None

np.random.seed(42)
test = ImageList.from_df(test_df, path=path_test)
data = (
    ImageList.from_df(labels_df, path=path_train)
    .split_by_rand_pct(0.05)
    .label_from_df()
    .add_test(test)
    .transform(get_transforms(flip_vert=True, max_warp=0.0), size=128)
    .databunch(path=path, bs=bs)
    .normalize(imagenet_stats)
)
data


## --- ERROR in cell 6, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2422682075.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    106[0m     [0;34m.[0m[0madd_test[0m[0;34m([0m[0mtest[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    107[0m     [0;34m.[0m[0mtransform[0m[0;34m([0m[0mget_transforms[0m[0;34m([0m[0mflip_vert[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m [0mmax_warp[0m[0;34m=[0m[0;36m0.0[0m[0;34m)[0m[0;34m,[0m [0msize[0m[0;34m=[0m[0;36m128[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 108[0;31m     [0;34m.[0m[0mdatabunch[0m[0;34m([0m[0mpath[0m[0;34m=[0m[0mpath[0m[0;34m,[0m [0mbs[0m[0;34m=[0m[0mbs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    109[0m     [0;34m.[0m[0mnormalize[0m[0;34m([0m[0mimagenet_stats[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    110[0m )

[0;32m/tmp/ipykernel_11/2422682075.py[0m in [0;36mdatabunch[0;34m(self, path, bs, **kwargs)[0m
[1;32m     71[0m             [0mbatch_tfms[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_tfms[0m [0;32mif[0m [0mself[0m[0;34m.[0m[0m_tfms[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m [0;32melse[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[1;32m     72[0m [0;34m[0m[0m
[0;32m---> 73[0;31m             dls = ImageDataLoaders.from_name_func(
[0m[1;32m     74[0m                 [0mself[0m[0;34m.[0m[0m_path[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     75[0m                 [0mfiles[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/vision/data.py[0m in [0;36mfrom_name_func[0;34m(cls, path, fnames, label_func, **kwargs)[0m
[1;32m    148[0m             [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"label_func couldn't be lambda function on Windows"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    149[0m         [0mf[0m [0;34m=[0m [0musing_attr[0m[0;34m([0m[0mlabel_func[0m[0;34m,[0m [0;34m'name'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 150[0;31m         [0;32mreturn[0m [0mcls[0m[0;34m.[0m[0mfrom_path_func[0m[0;34m([0m[0mpath[0m[0;34m,[0m [0mfnames[0m[0;34m,[0m [0mf[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    151[0m [0;34m[0m[0m
[1;32m    152[0m     [0;34m@[0m[0mclassmethod[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/vision/data.py[0m in [0;36mfrom_path_func[0;34m(cls, path, fnames, label_func, valid_pct, seed, item_tfms, batch_tfms, img_cls, **kwargs)[0m
[1;32m    134[0m                            [0mitem_tfms[0m[0;34m=[0m[0mitem_tfms[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    135[0m                            batch_tfms=batch_tfms)
[0;32m--> 136[0;31m         [0;32mreturn[0m [0mcls[0m[0;34m.[0m[0mfrom_dblock[0m[0;34m([0m[0mdblock[0m[0;34m,[0m [0mfnames[0m[0;34m,[0m [0mpath[0m[0;34m=[0m[0mpath[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    137[0m [0;34m[0m[0m
[1;32m    138[0m     [0;34m@[0m[0mclassmethod[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/data/core.py[0m in [0;36mfrom_dblock[0;34m(cls, dblock, source, path, bs, val_bs, shuffle, device, **kwargs)[0m
[1;32m    278[0m         [0;34m**[0m[0mkwargs[0m[0;34m[0m[0;34m[0m[0m
[1;32m    279[0m     ):
[0;32m--> 280[0;31m         [0;32mreturn[0m [0mdblock[0m[0;34m.[0m[0mdataloaders[0m[0;34m([0m[0msource[0m[0;34m,[0m [0mpath[0m[0;34m=[0m[0mpath[0m[0;34m,[0m [0mbs[0m[0;34m=[0m[0mbs[0m[0;34m,[0m [0mval_bs[0m[0;34m=[0m[0mval_bs[0m[0;34m,[0m [0mshuffle[0m[0;34m=[0m[0mshuffle[0m[0;34m,[0m [0mdevice[0m[0;34m=[0m[0mdevice[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    281[0m [0;34m[0m[0m
[1;32m    282[0m     _docs=dict(__getitem__="Retrieve `DataLoader` at `i` (`0` is training, `1` is validation)",

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/data/block.py[0m in [0;36mdataloaders[0;34m(self, source, path, verbose, **kwargs)[0m
[1;32m    155[0m         [0;34m**[0m[0mkwargs[0m[0;34m[0m[0;34m[0m[0m
[1;32m    156[0m     ) -> DataLoaders:
[0;32m--> 157[0;31m         [0mdsets[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mdatasets[0m[0;34m([0m[0msource[0m[0;34m,[0m [0mverbose[0m[0;34m=[0m[0mverbose[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    158[0m         [0mkwargs[0m [0;34m=[0m [0;34m{[0m[0;34m**[0m[0mself[0m[0;34m.[0m[0mdls_kwargs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m,[0m [0;34m'verbose'[0m[0;34m:[0m [0mverbose[0m[0;34m}[0m[0;34m[0m[0;34m[0m[0m
[1;32m    159[0m         [0;32mreturn[0m [0mdsets[0m[0;34m.[0m[0mdataloaders[0m[0;34m([0m[0mpath[0m[0;34m=[0m[0mpath[0m[0;34m,[0m [0mafter_item[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mitem_tfms[0m[0;34m,[0m [0mafter_batch[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mbatch_tfms[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/data/block.py[0m in [0;36mdatasets[0;34m(self, source, verbose)[0m
[1;32m    147[0m         [0msplits[0m [0;34m=[0m [0;34m([0m[0mself[0m[0;34m.[0m[0msplitter[0m [0;32mor[0m [0mRandomSplitter[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m([0m[0mitems[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    148[0m         [0mpv[0m[0;34m([0m[0;34mf"{len(splits)} datasets of sizes {','.join([str(len(s)) for s in splits])}"[0m[0;34m,[0m [0mverbose[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 149[0;31m         [0;32mreturn[0m [0mDatasets[0m[0;34m([0m[0mitems[0m[0;34m,[0m [0mtfms[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0m_combine_type_tfms[0m[0;34m([0m[0;34m)[0m[0;34m,[0m [0msplits[0m[0;34m=[0m[0msplits[0m[0;34m,[0m [0mdl_type[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mdl_type[0m[0;34m,[0m [0mn_inp[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mn_inp[0m[0;34m,[0m [0mverbose[0m[0;34m=[0m[0mverbose[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    150[0m [0;34m[0m[0m
[1;32m    151[0m     def dataloaders(self, 

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/data/core.py[0m in [0;36m__init__[0;34m(self, items, tfms, tls, n_inp, dl_type, **kwargs)[0m
[1;32m    448[0m     ):
[1;32m    449[0m         [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__init__[0m[0;34m([0m[0mdl_type[0m[0;34m=[0m[0mdl_type[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 450[0;31m         [0mself[0m[0;34m.[0m[0mtls[0m [0;34m=[0m [0mL[0m[0;34m([0m[0mtls[0m [0;32mif[0m [0mtls[0m [0;32melse[0m [0;34m[[0m[0mTfmdLists[0m[0;34m([0m[0mitems[0m[0;34m,[0m [0mt[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m [0;32mfor[0m [0mt[0m [0;32min[0m [0mL[0m[0;34m([0m[0mifnone[0m[0;34m([0m[0mtfms[0m[0;34m,[0m[0;34m[[0m[0;32mNone[0m[0;34m][0m[0;34m)[0m[0;34m)[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    451[0m         [0mself[0m[0;34m.[0m[0mn_inp[0m [0;34m=[0m [0mifnone[0m[0;34m([0m[0mn_inp[0m[0;34m,[0m [0mmax[0m[0;34m([0m[0;36m1[0m[0;34m,[0m [0mlen[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mtls[0m[0;34m)[0m[0;34m-[0m[0;36m1[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    452[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/data/core.py[0m in [0;36m<listcomp>[0;34m(.0)[0m
[1;32m    448[0m     ):
[1;32m    449[0m         [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__init__[0m[0;34m([0m[0mdl_type[0m[0;34m=[0m[0mdl_type[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 450[0;31m         [0mself[0m[0;34m.[0m[0mtls[0m [0;34m=[0m [0mL[0m[0;34m([0m[0mtls[0m [0;32mif[0m [0mtls[0m [0;32melse[0m [0;34m[[0m[0mTfmdLists[0m[0;34m([0m[0mitems[0m[0;34m,[0m [0mt[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m [0;32mfor[0m [0mt[0m [0;32min[0m [0mL[0m[0;34m([0m[0mifnone[0m[0;34m([0m[0mtfms[0m[0;34m,[0m[0;34m[[0m[0;32mNone[0m[0;34m][0m[0;34m)[0m[0;34m)[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    451[0m         [0mself[0m[0;34m.[0m[0mn_inp[0m [0;34m=[0m [0mifnone[0m[0;34m([0m[0mn_inp[0m[0;34m,[0m [0mmax[0m[0;34m([0m[0;36m1[0m[0;34m,[0m [0mlen[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mtls[0m[0;34m)[0m[0;34m-[0m[0;36m1[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    452[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py[0m in [0;36m__call__[0;34m(cls, x, *args, **kwargs)[0m
[1;32m    103[0m     [0;32mdef[0m [0m__call__[0m[0;34m([0m[0mcls[0m[0;34m,[0m [0mx[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    104[0m         [0;32mif[0m [0;32mnot[0m [0margs[0m [0;32mand[0m [0;32mnot[0m [0mkwargs[0m [0;32mand[0m [0mx[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m [0;32mand[0m [0misinstance[0m[0;34m([0m[0mx[0m[0;34m,[0m[0mcls[0m[0;34m)[0m[0;34m:[0m [0;32mreturn[0m [0mx[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 105[0;31m         [0;32mreturn[0m [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__call__[0m[0;34m([0m[0mx[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    106[0m [0;34m[0m[0m
[1;32m    107[0m [0;31m# %% ../nbs/02_foundation.ipynb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/data/core.py[0m in [0;36m__init__[0;34m(self, items, tfms, use_list, do_setup, split_idx, train_setup, splits, types, verbose, dl_type)[0m
[1;32m    362[0m         [0;32mif[0m [0mdo_setup[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    363[0m             [0mpv[0m[0;34m([0m[0;34mf"Setting up {self.tfms}"[0m[0;34m,[0m [0mverbose[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 364[0;31m             [0mself[0m[0;34m.[0m[0msetup[0m[0;34m([0m[0mtrain_setup[0m[0;34m=[0m[0mtrain_setup[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    365[0m [0;34m[0m[0m
[1;32m    366[0m     [0;32mdef[0m [0m_new[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mitems[0m[0;34m,[0m [0msplit_idx[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/data/core.py[0m in [0;36msetup[0;34m(self, train_setup)[0m
[1;32m    389[0m             [0;32mfor[0m [0mf[0m [0;32min[0m [0mself[0m[0;34m.[0m[0mtfms[0m[0;34m.[0m[0mfs[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    390[0m                 [0mself[0m[0;34m.[0m[0mtypes[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mgetattr[0m[0;34m([0m[0mf[0m[0;34m,[0m [0;34m'input_types'[0m[0;34m,[0m [0mtype[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 391[0;31m                 [0mx[0m [0;34m=[0m [0mf[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    392[0m             [0mself[0m[0;34m.[0m[0mtypes[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mtype[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    393[0m         [0mtypes[0m [0;34m=[0m [0mL[0m[0;34m([0m[0mt[0m [0;32mif[0m [0mis_listy[0m[0;34m([0m[0mt[0m[0;34m)[0m [0;32melse[0m [0;34m[[0m[0mt[0m[0;34m][0m [0;32mfor[0m [0mt[0m [0;32min[0m [0mself[0m[0;34m.[0m[0mtypes[0m[0;34m)[0m[0;34m.[0m[0mconcat[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0munique[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

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

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '17fb09f3617d81bc17d6286da8c3b877.jpg'

## === cell 7
data.show_batch(rows = 3, figsize = (10,8))
