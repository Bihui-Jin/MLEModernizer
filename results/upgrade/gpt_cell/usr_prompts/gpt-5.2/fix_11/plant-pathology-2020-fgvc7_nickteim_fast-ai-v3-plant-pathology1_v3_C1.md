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

3.8

# 2. Installed packages

fastai==2.8.5
geopandas==0.14.4
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
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
from fastai import *
from fastai.vision import *


## === cell 1
from pathlib import Path as _Path

path1 = _Path("/kaggle/input/")
list(path1.iterdir())


## === cell 2
from pathlib import Path

path = Path("/kaggle/input/plant-pathology-2020-fgvc7")

list(path.iterdir())


## === cell 3
path2 = Path('/kaggle/input/plant-pathology-2020-fgvc7/images')


## === cell 4
import pandas as pd

df = pd.read_csv(path / "train.csv")
df.head()


## === cell 5
test_df = pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')


## === cell 6
from fastai.vision.augment import aug_transforms

tfms = aug_transforms(flip_vert=True, max_lighting=0.2, max_zoom=1.05, max_warp=0.0)


## === cell 7
LABEL_COLS = ['healthy', 'multiple_diseases', 'rust', 'scab']


## === cell 8
from fastai.vision.all import *

test_ids = set(test_df["image_id"].astype(str).values)
all_test_files = get_image_files(path / "images")
test = L([p for p in all_test_files if p.stem in test_ids])


## === cell 9
np.random.seed(42)

dblock = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock(encoded=True, vocab=LABEL_COLS)),
    get_x=ColReader("image_id", pref=str(path / "images") + "/", suff=".jpg"),
    get_y=ColReader(LABEL_COLS),
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
)

src = dblock.datasets(df)


## === cell 10
data = dblock.dataloaders(
    df,
    item_tfms=tfms + [Resize(128)],
    batch_tfms=Normalize.from_stats(*imagenet_stats),
    bs=64,
    num_workers=0,
)
data.test_dl(test, with_labels=False)


## === cell 13
data.show_batch(rows=3, figsize=(12,9))


## --- ERROR in cell 13, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3499888354.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mdata[0m[0;34m.[0m[0mshow_batch[0m[0;34m([0m[0mrows[0m[0;34m=[0m[0;36m3[0m[0;34m,[0m [0mfigsize[0m[0;34m=[0m[0;34m([0m[0;36m12[0m[0;34m,[0m[0;36m9[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/fastai/data/core.py[0m in [0;36mshow_batch[0;34m(self, b, max_n, ctxs, show, unique, **kwargs)[0m
[1;32m    156[0m         [0;32mif[0m [0mb[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m [0mb[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mone_batch[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    157[0m         [0;32mif[0m [0;32mnot[0m [0mshow[0m[0;34m:[0m [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_pre_show_batch[0m[0;34m([0m[0mb[0m[0;34m,[0m [0mmax_n[0m[0;34m=[0m[0mmax_n[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 158[0;31m         [0mshow_batch[0m[0;34m([0m[0;34m*[0m[0mself[0m[0;34m.[0m[0m_pre_show_batch[0m[0;34m([0m[0mb[0m[0;34m,[0m [0mmax_n[0m[0;34m=[0m[0mmax_n[0m[0;34m)[0m[0;34m,[0m [0mctxs[0m[0;34m=[0m[0mctxs[0m[0;34m,[0m [0mmax_n[0m[0;34m=[0m[0mmax_n[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    159[0m         [0;32mif[0m [0munique[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0mget_idxs[0m [0;34m=[0m [0mold_get_idxs[0m[0;34m[0m[0;34m[0m[0m
[1;32m    160[0m [0;34m[0m[0m

    [0;31m[... skipping hidden 1 frame][0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/vision/data.py[0m in [0;36mshow_batch[0;34m(x, y, samples, ctxs, max_n, nrows, ncols, figsize, **kwargs)[0m
[1;32m     69[0m [0;32mdef[0m [0mshow_batch[0m[0;34m([0m[0mx[0m[0;34m:[0m[0mTensorImage[0m[0;34m,[0m [0my[0m[0;34m,[0m [0msamples[0m[0;34m,[0m [0mctxs[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0mmax_n[0m[0;34m=[0m[0;36m10[0m[0;34m,[0m [0mnrows[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0mncols[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0mfigsize[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     70[0m     [0;32mif[0m [0mctxs[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m [0mctxs[0m [0;34m=[0m [0mget_grid[0m[0;34m([0m[0mmin[0m[0;34m([0m[0mlen[0m[0;34m([0m[0msamples[0m[0;34m)[0m[0;34m,[0m [0mmax_n[0m[0;34m)[0m[0;34m,[0m [0mnrows[0m[0;34m=[0m[0mnrows[0m[0;34m,[0m [0mncols[0m[0;34m=[0m[0mncols[0m[0;34m,[0m [0mfigsize[0m[0;34m=[0m[0mfigsize[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 71[0;31m     [0;32mreturn[0m [0mget_show_batch_func[0m[0;34m([0m[0mobject[0m[0;34m)[0m[0;34m([0m[0mx[0m[0;34m,[0m [0my[0m[0;34m,[0m [0msamples[0m[0;34m,[0m [0mctxs[0m[0;34m=[0m[0mctxs[0m[0;34m,[0m [0mmax_n[0m[0;34m=[0m[0mmax_n[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     72[0m [0;34m[0m[0m
[1;32m     73[0m [0;31m# %% ../../nbs/08_vision.data.ipynb 17[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/data/core.py[0m in [0;36mshow_batch[0;34m(x, y, samples, ctxs, max_n, **kwargs)[0m
[1;32m     28[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     29[0m         [0;32mfor[0m [0mi[0m [0;32min[0m [0mrange_of[0m[0;34m([0m[0msamples[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 30[0;31m             [0mctxs[0m [0;34m=[0m [0;34m[[0m[0mb[0m[0;34m.[0m[0mshow[0m[0;34m([0m[0mctx[0m[0;34m=[0m[0mc[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m [0;32mfor[0m [0mb[0m[0;34m,[0m[0mc[0m[0;34m,[0m[0m_[0m [0;32min[0m [0mzip[0m[0;34m([0m[0msamples[0m[0;34m.[0m[0mitemgot[0m[0;34m([0m[0mi[0m[0;34m)[0m[0;34m,[0m[0mctxs[0m[0;34m,[0m[0mrange[0m[0;34m([0m[0mmax_n[0m[0;34m)[0m[0;34m)[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     31[0m     [0;32mreturn[0m [0mctxs[0m[0;34m[0m[0;34m[0m[0m
[1;32m     32[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/data/core.py[0m in [0;36m<listcomp>[0;34m(.0)[0m
[1;32m     28[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     29[0m         [0;32mfor[0m [0mi[0m [0;32min[0m [0mrange_of[0m[0;34m([0m[0msamples[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 30[0;31m             [0mctxs[0m [0;34m=[0m [0;34m[[0m[0mb[0m[0;34m.[0m[0mshow[0m[0;34m([0m[0mctx[0m[0;34m=[0m[0mc[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m [0;32mfor[0m [0mb[0m[0;34m,[0m[0mc[0m[0;34m,[0m[0m_[0m [0;32min[0m [0mzip[0m[0;34m([0m[0msamples[0m[0;34m.[0m[0mitemgot[0m[0;34m([0m[0mi[0m[0;34m)[0m[0;34m,[0m[0mctxs[0m[0;34m,[0m[0mrange[0m[0;34m([0m[0mmax_n[0m[0;34m)[0m[0;34m)[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     31[0m     [0;32mreturn[0m [0mctxs[0m[0;34m[0m[0;34m[0m[0m
[1;32m     32[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/torch_core.py[0m in [0;36mshow[0;34m(self, ctx, **kwargs)[0m
[1;32m    430[0m     [0m_show_args[0m [0;34m=[0m [0mArrayImageBase[0m[0;34m.[0m[0m_show_args[0m[0;34m[0m[0;34m[0m[0m
[1;32m    431[0m     [0;32mdef[0m [0mshow[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mctx[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 432[0;31m         [0;32mreturn[0m [0mshow_image[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mctx[0m[0;34m=[0m[0mctx[0m[0;34m,[0m [0;34m**[0m[0;34m{[0m[0;34m**[0m[0mself[0m[0;34m.[0m[0m_show_args[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m}[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    433[0m [0;34m[0m[0m
[1;32m    434[0m [0;31m# %% ../nbs/00_torch_core.ipynb 107[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/torch_core.py[0m in [0;36mshow_image[0;34m(im, ax, figsize, title, ctx, **kwargs)[0m
[1;32m     78[0m     [0;32mif[0m [0mfigsize[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m [0mfigsize[0m [0;34m=[0m [0;34m([0m[0m_fig_bounds[0m[0;34m([0m[0mim[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m)[0m[0;34m,[0m [0m_fig_bounds[0m[0;34m([0m[0mim[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     79[0m     [0;32mif[0m [0max[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m [0m_[0m[0;34m,[0m[0max[0m [0;34m=[0m [0mplt[0m[0;34m.[0m[0msubplots[0m[0;34m([0m[0mfigsize[0m[0;34m=[0m[0mfigsize[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 80[0;31m     [0max[0m[0;34m.[0m[0mimshow[0m[0;34m([0m[0mim[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     81[0m     [0;32mif[0m [0mtitle[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m [0max[0m[0;34m.[0m[0mset_title[0m[0;34m([0m[0mtitle[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     82[0m     [0max[0m[0;34m.[0m[0maxis[0m[0;34m([0m[0;34m'off'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/matplotlib/__init__.py[0m in [0;36minner[0;34m(ax, data, *args, **kwargs)[0m
[1;32m   1444[0m     [0;32mdef[0m [0minner[0m[0;34m([0m[0max[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0mdata[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1445[0m         [0;32mif[0m [0mdata[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1446[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0max[0m[0;34m,[0m [0;34m*[0m[0mmap[0m[0;34m([0m[0msanitize_sequence[0m[0;34m,[0m [0margs[0m[0;34m)[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1447[0m [0;34m[0m[0m
[1;32m   1448[0m         [0mbound[0m [0;34m=[0m [0mnew_sig[0m[0;34m.[0m[0mbind[0m[0;34m([0m[0max[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/matplotlib/axes/_axes.py[0m in [0;36mimshow[0;34m(self, X, cmap, norm, aspect, interpolation, alpha, vmin, vmax, origin, extent, interpolation_stage, filternorm, filterrad, resample, url, **kwargs)[0m
[1;32m   5654[0m             [0maspect[0m [0;34m=[0m [0mmpl[0m[0;34m.[0m[0mrcParams[0m[0;34m[[0m[0;34m'image.aspect'[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m   5655[0m         [0mself[0m[0;34m.[0m[0mset_aspect[0m[0;34m([0m[0maspect[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 5656[0;31m         im = mimage.AxesImage(self, cmap=cmap, norm=norm,
[0m[1;32m   5657[0m                               [0minterpolation[0m[0;34m=[0m[0minterpolation[0m[0;34m,[0m [0morigin[0m[0;34m=[0m[0morigin[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   5658[0m                               [0mextent[0m[0;34m=[0m[0mextent[0m[0;34m,[0m [0mfilternorm[0m[0;34m=[0m[0mfilternorm[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/matplotlib/_api/deprecation.py[0m in [0;36mwrapper[0;34m(*args, **kwargs)[0m
[1;32m    452[0m                 [0;34m"parameter will become keyword-only %(removal)s."[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    453[0m                 name=name, obj_type=f"parameter of {func.__name__}()")
[0;32m--> 454[0;31m         [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    455[0m [0;34m[0m[0m
[1;32m    456[0m     [0;31m# Don't modify *func*'s signature, as boilerplate.py needs it.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/matplotlib/image.py[0m in [0;36m__init__[0;34m(self, ax, cmap, norm, interpolation, origin, extent, filternorm, filterrad, resample, interpolation_stage, **kwargs)[0m
[1;32m    920[0m         [0mself[0m[0;34m.[0m[0m_extent[0m [0;34m=[0m [0mextent[0m[0;34m[0m[0;34m[0m[0m
[1;32m    921[0m [0;34m[0m[0m
[0;32m--> 922[0;31m         super().__init__(
[0m[1;32m    923[0m             [0max[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    924[0m             [0mcmap[0m[0;34m=[0m[0mcmap[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/matplotlib/image.py[0m in [0;36m__init__[0;34m(self, ax, cmap, norm, interpolation, origin, filternorm, filterrad, resample, interpolation_stage, **kwargs)[0m
[1;32m    272[0m         [0mself[0m[0;34m.[0m[0m_imcache[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[1;32m    273[0m [0;34m[0m[0m
[0;32m--> 274[0;31m         [0mself[0m[0;34m.[0m[0m_internal_update[0m[0;34m([0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    275[0m [0;34m[0m[0m
[1;32m    276[0m     [0;32mdef[0m [0m__str__[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/matplotlib/artist.py[0m in [0;36m_internal_update[0;34m(self, kwargs)[0m
[1;32m   1221[0m         [0mThe[0m [0mlack[0m [0mof[0m [0mprenormalization[0m [0;32mis[0m [0mto[0m [0mmaintain[0m [0mbackcompatibility[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1222[0m         """
[0;32m-> 1223[0;31m         return self._update_props(
[0m[1;32m   1224[0m             [0mkwargs[0m[0;34m,[0m [0;34m"{cls.__name__}.set() got an unexpected keyword argument "[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1225[0m             "{prop_name!r}")

[0;32m/usr/local/lib/python3.11/dist-packages/matplotlib/artist.py[0m in [0;36m_update_props[0;34m(self, props, errfmt)[0m
[1;32m   1195[0m                     [0mfunc[0m [0;34m=[0m [0mgetattr[0m[0;34m([0m[0mself[0m[0;34m,[0m [0;34mf"set_{k}"[0m[0;34m,[0m [0;32mNone[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1196[0m                     [0;32mif[0m [0;32mnot[0m [0mcallable[0m[0;34m([0m[0mfunc[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1197[0;31m                         raise AttributeError(
[0m[1;32m   1198[0m                             errfmt.format(cls=type(self), prop_name=k))
[1;32m   1199[0m                     [0mret[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mfunc[0m[0;34m([0m[0mv[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: AxesImage.set() got an unexpected keyword argument 'rows'

## === cell 15
arch = models.resnet50
