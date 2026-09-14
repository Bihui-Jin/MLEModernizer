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
imageio==2.37.0
imageio-ffmpeg==0.6.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-image==0.25.2
sklearn-pandas==2.2.0
torchvision==0.21.0+cu124

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        input/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> data/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

# 4. Code solution

## === cell 0
%reload_ext autoreload
%autoreload 2
%matplotlib inline


## === cell 1
import pathlib
from pathlib import Path

import fastai

from fastai.vision.all import *
from fastai.callback.all import *

try:
    from fastai.utils.mem import *  # type: ignore
except ModuleNotFoundError:
    pass

from torchvision.models import vgg16_bn
from subprocess import check_output


## === cell 2
input_path = Path('/kaggle/input/denoising-dirty-documents')
items = list(input_path.glob("*.zip"))
print([x for x in items])


## === cell 3
import zipfile

for item in items:
    print(item)
    with zipfile.ZipFile(str(item), "r") as z:
        z.extractall(".")


## === cell 4
bs, size = 4, 128
arch = models.resnet34
path_train = Path("train")
path_train_cleaned = Path("train_cleaned")
path_test = Path("test")
path_submission = Path("submission")


## === cell 5
src = DataBlock(
    blocks=(ImageBlock, ImageBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=lambda x: path_train_cleaned / x.name,
)

dls = src.dataloaders(path_train, bs=bs, item_tfms=Resize(size))


## === cell 6
def get_data(src, bs, size):
    data = (
        src.label_from_func(lambda x: path_train_cleaned / x.name)
           .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
           .databunch(bs=bs)       
           .normalize(imagenet_stats, do_y=True)
    )
    data.c = 3
    return data


## === cell 7
def get_data(src, bs, size):
    block = DataBlock(
        blocks=(ImageBlock, ImageBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=lambda x: path_train_cleaned / x.name,
        item_tfms=Resize(size),
        batch_tfms=[
            *aug_transforms(max_zoom=2.0),
            Normalize.from_stats(*imagenet_stats, do_y=True),
        ],
    )
    data = block.dataloaders(path_train, bs=bs)
    data.c = 3
    return data


## === cell 8
_orig_from_stats = Normalize.from_stats


def _from_stats_compat(*args, **kwargs):
    kwargs.pop("do_y", None)
    return _orig_from_stats(*args, **kwargs)


Normalize.from_stats = _from_stats_compat

try:
    data
except NameError:
    data = get_data(src, bs, size)

try:
    data.show_batch(ds_idx=1, rows=2, figsize=(5, 5), title="Some image")
except (TypeError, AttributeError):
    data.show_batch(dl=data.valid, rows=2, figsize=(5, 5), title="Some image")


## --- ERROR in cell 8, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2283865276.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     19[0m [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 20[0;31m     [0mdata[0m[0;34m.[0m[0mshow_batch[0m[0;34m([0m[0mds_idx[0m[0;34m=[0m[0;36m1[0m[0;34m,[0m [0mrows[0m[0;34m=[0m[0;36m2[0m[0;34m,[0m [0mfigsize[0m[0;34m=[0m[0;34m([0m[0;36m5[0m[0;34m,[0m [0;36m5[0m[0;34m)[0m[0;34m,[0m [0mtitle[0m[0;34m=[0m[0;34m"Some image"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     21[0m [0;32mexcept[0m [0;34m([0m[0mTypeError[0m[0;34m,[0m [0mAttributeError[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/data/core.py[0m in [0;36mshow_batch[0;34m(self, b, max_n, ctxs, show, unique, **kwargs)[0m
[1;32m    157[0m         [0;32mif[0m [0;32mnot[0m [0mshow[0m[0;34m:[0m [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_pre_show_batch[0m[0;34m([0m[0mb[0m[0;34m,[0m [0mmax_n[0m[0;34m=[0m[0mmax_n[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 158[0;31m         [0mshow_batch[0m[0;34m([0m[0;34m*[0m[0mself[0m[0;34m.[0m[0m_pre_show_batch[0m[0;34m([0m[0mb[0m[0;34m,[0m [0mmax_n[0m[0;34m=[0m[0mmax_n[0m[0;34m)[0m[0;34m,[0m [0mctxs[0m[0;34m=[0m[0mctxs[0m[0;34m,[0m [0mmax_n[0m[0;34m=[0m[0mmax_n[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    159[0m         [0;32mif[0m [0munique[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0mget_idxs[0m [0;34m=[0m [0mold_get_idxs[0m[0;34m[0m[0;34m[0m[0m

    [0;31m[... skipping hidden 1 frame][0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/vision/data.py[0m in [0;36mshow_batch[0;34m(x, y, samples, ctxs, max_n, nrows, ncols, figsize, **kwargs)[0m
[1;32m     77[0m     [0;32mfor[0m [0mi[0m [0;32min[0m [0mrange[0m[0;34m([0m[0;36m2[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 78[0;31m         [0mctxs[0m[0;34m[[0m[0mi[0m[0;34m:[0m[0;34m:[0m[0;36m2[0m[0;34m][0m [0;34m=[0m [0;34m[[0m[0mb[0m[0;34m.[0m[0mshow[0m[0;34m([0m[0mctx[0m[0;34m=[0m[0mc[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m [0;32mfor[0m [0mb[0m[0;34m,[0m[0mc[0m[0;34m,[0m[0m_[0m [0;32min[0m [0mzip[0m[0;34m([0m[0msamples[0m[0;34m.[0m[0mitemgot[0m[0;34m([0m[0mi[0m[0;34m)[0m[0;34m,[0m[0mctxs[0m[0;34m[[0m[0mi[0m[0;34m:[0m[0;34m:[0m[0;36m2[0m[0;34m][0m[0;34m,[0m[0mrange[0m[0;34m([0m[0mmax_n[0m[0;34m)[0m[0;34m)[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     79[0m     [0;32mreturn[0m [0mctxs[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/vision/data.py[0m in [0;36m<listcomp>[0;34m(.0)[0m
[1;32m     77[0m     [0;32mfor[0m [0mi[0m [0;32min[0m [0mrange[0m[0;34m([0m[0;36m2[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 78[0;31m         [0mctxs[0m[0;34m[[0m[0mi[0m[0;34m:[0m[0;34m:[0m[0;36m2[0m[0;34m][0m [0;34m=[0m [0;34m[[0m[0mb[0m[0;34m.[0m[0mshow[0m[0;34m([0m[0mctx[0m[0;34m=[0m[0mc[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m [0;32mfor[0m [0mb[0m[0;34m,[0m[0mc[0m[0;34m,[0m[0m_[0m [0;32min[0m [0mzip[0m[0;34m([0m[0msamples[0m[0;34m.[0m[0mitemgot[0m[0;34m([0m[0mi[0m[0;34m)[0m[0;34m,[0m[0mctxs[0m[0;34m[[0m[0mi[0m[0;34m:[0m[0;34m:[0m[0;36m2[0m[0;34m][0m[0;34m,[0m[0mrange[0m[0;34m([0m[0mmax_n[0m[0;34m)[0m[0;34m)[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     79[0m     [0;32mreturn[0m [0mctxs[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/torch_core.py[0m in [0;36mshow[0;34m(self, ctx, **kwargs)[0m
[1;32m    431[0m     [0;32mdef[0m [0mshow[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mctx[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 432[0;31m         [0;32mreturn[0m [0mshow_image[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mctx[0m[0;34m=[0m[0mctx[0m[0;34m,[0m [0;34m**[0m[0;34m{[0m[0;34m**[0m[0mself[0m[0;34m.[0m[0m_show_args[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m}[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    433[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/torch_core.py[0m in [0;36mshow_image[0;34m(im, ax, figsize, title, ctx, **kwargs)[0m
[1;32m     79[0m     [0;32mif[0m [0max[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m [0m_[0m[0;34m,[0m[0max[0m [0;34m=[0m [0mplt[0m[0;34m.[0m[0msubplots[0m[0;34m([0m[0mfigsize[0m[0;34m=[0m[0mfigsize[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 80[0;31m     [0max[0m[0;34m.[0m[0mimshow[0m[0;34m([0m[0mim[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     81[0m     [0;32mif[0m [0mtitle[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m [0max[0m[0;34m.[0m[0mset_title[0m[0;34m([0m[0mtitle[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/matplotlib/__init__.py[0m in [0;36minner[0;34m(ax, data, *args, **kwargs)[0m
[1;32m   1445[0m         [0;32mif[0m [0mdata[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1446[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0max[0m[0;34m,[0m [0;34m*[0m[0mmap[0m[0;34m([0m[0msanitize_sequence[0m[0;34m,[0m [0margs[0m[0;34m)[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1447[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/matplotlib/axes/_axes.py[0m in [0;36mimshow[0;34m(self, X, cmap, norm, aspect, interpolation, alpha, vmin, vmax, origin, extent, interpolation_stage, filternorm, filterrad, resample, url, **kwargs)[0m
[1;32m   5655[0m         [0mself[0m[0;34m.[0m[0mset_aspect[0m[0;34m([0m[0maspect[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 5656[0;31m         im = mimage.AxesImage(self, cmap=cmap, norm=norm,
[0m[1;32m   5657[0m                               [0minterpolation[0m[0;34m=[0m[0minterpolation[0m[0;34m,[0m [0morigin[0m[0;34m=[0m[0morigin[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/matplotlib/_api/deprecation.py[0m in [0;36mwrapper[0;34m(*args, **kwargs)[0m
[1;32m    453[0m                 name=name, obj_type=f"parameter of {func.__name__}()")
[0;32m--> 454[0;31m         [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    455[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/matplotlib/image.py[0m in [0;36m__init__[0;34m(self, ax, cmap, norm, interpolation, origin, extent, filternorm, filterrad, resample, interpolation_stage, **kwargs)[0m
[1;32m    921[0m [0;34m[0m[0m
[0;32m--> 922[0;31m         super().__init__(
[0m[1;32m    923[0m             [0max[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/matplotlib/image.py[0m in [0;36m__init__[0;34m(self, ax, cmap, norm, interpolation, origin, filternorm, filterrad, resample, interpolation_stage, **kwargs)[0m
[1;32m    273[0m [0;34m[0m[0m
[0;32m--> 274[0;31m         [0mself[0m[0;34m.[0m[0m_internal_update[0m[0;34m([0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    275[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/matplotlib/artist.py[0m in [0;36m_internal_update[0;34m(self, kwargs)[0m
[1;32m   1222[0m         """
[0;32m-> 1223[0;31m         return self._update_props(
[0m[1;32m   1224[0m             [0mkwargs[0m[0;34m,[0m [0;34m"{cls.__name__}.set() got an unexpected keyword argument "[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/matplotlib/artist.py[0m in [0;36m_update_props[0;34m(self, props, errfmt)[0m
[1;32m   1196[0m                     [0;32mif[0m [0;32mnot[0m [0mcallable[0m[0;34m([0m[0mfunc[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1197[0;31m                         raise AttributeError(
[0m[1;32m   1198[0m                             errfmt.format(cls=type(self), prop_name=k))

[0;31mAttributeError[0m: AxesImage.set() got an unexpected keyword argument 'ds_idx'

During handling of the above exception, another exception occurred:

[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2283865276.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     20[0m     [0mdata[0m[0;34m.[0m[0mshow_batch[0m[0;34m([0m[0mds_idx[0m[0;34m=[0m[0;36m1[0m[0;34m,[0m [0mrows[0m[0;34m=[0m[0;36m2[0m[0;34m,[0m [0mfigsize[0m[0;34m=[0m[0;34m([0m[0;36m5[0m[0;34m,[0m [0;36m5[0m[0;34m)[0m[0;34m,[0m [0mtitle[0m[0;34m=[0m[0;34m"Some image"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     21[0m [0;32mexcept[0m [0;34m([0m[0mTypeError[0m[0;34m,[0m [0mAttributeError[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 22[0;31m     [0mdata[0m[0;34m.[0m[0mshow_batch[0m[0;34m([0m[0mdl[0m[0;34m=[0m[0mdata[0m[0;34m.[0m[0mvalid[0m[0;34m,[0m [0mrows[0m[0;34m=[0m[0;36m2[0m[0;34m,[0m [0mfigsize[0m[0;34m=[0m[0;34m([0m[0;36m5[0m[0;34m,[0m [0;36m5[0m[0;34m)[0m[0;34m,[0m [0mtitle[0m[0;34m=[0m[0;34m"Some image"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/fastai/data/core.py[0m in [0;36mshow_batch[0;34m(self, b, max_n, ctxs, show, unique, **kwargs)[0m
[1;32m    156[0m         [0;32mif[0m [0mb[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m [0mb[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mone_batch[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    157[0m         [0;32mif[0m [0;32mnot[0m [0mshow[0m[0;34m:[0m [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_pre_show_batch[0m[0;34m([0m[0mb[0m[0;34m,[0m [0mmax_n[0m[0;34m=[0m[0mmax_n[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 158[0;31m         [0mshow_batch[0m[0;34m([0m[0;34m*[0m[0mself[0m[0;34m.[0m[0m_pre_show_batch[0m[0;34m([0m[0mb[0m[0;34m,[0m [0mmax_n[0m[0;34m=[0m[0mmax_n[0m[0;34m)[0m[0;34m,[0m [0mctxs[0m[0;34m=[0m[0mctxs[0m[0;34m,[0m [0mmax_n[0m[0;34m=[0m[0mmax_n[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    159[0m         [0;32mif[0m [0munique[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0mget_idxs[0m [0;34m=[0m [0mold_get_idxs[0m[0;34m[0m[0;34m[0m[0m
[1;32m    160[0m [0;34m[0m[0m

    [0;31m[... skipping hidden 1 frame][0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/vision/data.py[0m in [0;36mshow_batch[0;34m(x, y, samples, ctxs, max_n, nrows, ncols, figsize, **kwargs)[0m
[1;32m     76[0m     [0;32mif[0m [0mctxs[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m [0mctxs[0m [0;34m=[0m [0mget_grid[0m[0;34m([0m[0mmin[0m[0;34m([0m[0mlen[0m[0;34m([0m[0msamples[0m[0;34m)[0m[0;34m,[0m [0mmax_n[0m[0;34m)[0m[0;34m,[0m [0mnrows[0m[0;34m=[0m[0mnrows[0m[0;34m,[0m [0mncols[0m[0;34m=[0m[0mncols[0m[0;34m,[0m [0mfigsize[0m[0;34m=[0m[0mfigsize[0m[0;34m,[0m [0mdouble[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     77[0m     [0;32mfor[0m [0mi[0m [0;32min[0m [0mrange[0m[0;34m([0m[0;36m2[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 78[0;31m         [0mctxs[0m[0;34m[[0m[0mi[0m[0;34m:[0m[0;34m:[0m[0;36m2[0m[0;34m][0m [0;34m=[0m [0;34m[[0m[0mb[0m[0;34m.[0m[0mshow[0m[0;34m([0m[0mctx[0m[0;34m=[0m[0mc[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m [0;32mfor[0m [0mb[0m[0;34m,[0m[0mc[0m[0;34m,[0m[0m_[0m [0;32min[0m [0mzip[0m[0;34m([0m[0msamples[0m[0;34m.[0m[0mitemgot[0m[0;34m([0m[0mi[0m[0;34m)[0m[0;34m,[0m[0mctxs[0m[0;34m[[0m[0mi[0m[0;34m:[0m[0;34m:[0m[0;36m2[0m[0;34m][0m[0;34m,[0m[0mrange[0m[0;34m([0m[0mmax_n[0m[0;34m)[0m[0;34m)[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     79[0m     [0;32mreturn[0m [0mctxs[0m[0;34m[0m[0;34m[0m[0m
[1;32m     80[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/vision/data.py[0m in [0;36m<listcomp>[0;34m(.0)[0m
[1;32m     76[0m     [0;32mif[0m [0mctxs[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m [0mctxs[0m [0;34m=[0m [0mget_grid[0m[0;34m([0m[0mmin[0m[0;34m([0m[0mlen[0m[0;34m([0m[0msamples[0m[0;34m)[0m[0;34m,[0m [0mmax_n[0m[0;34m)[0m[0;34m,[0m [0mnrows[0m[0;34m=[0m[0mnrows[0m[0;34m,[0m [0mncols[0m[0;34m=[0m[0mncols[0m[0;34m,[0m [0mfigsize[0m[0;34m=[0m[0mfigsize[0m[0;34m,[0m [0mdouble[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     77[0m     [0;32mfor[0m [0mi[0m [0;32min[0m [0mrange[0m[0;34m([0m[0;36m2[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 78[0;31m         [0mctxs[0m[0;34m[[0m[0mi[0m[0;34m:[0m[0;34m:[0m[0;36m2[0m[0;34m][0m [0;34m=[0m [0;34m[[0m[0mb[0m[0;34m.[0m[0mshow[0m[0;34m([0m[0mctx[0m[0;34m=[0m[0mc[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m [0;32mfor[0m [0mb[0m[0;34m,[0m[0mc[0m[0;34m,[0m[0m_[0m [0;32min[0m [0mzip[0m[0;34m([0m[0msamples[0m[0;34m.[0m[0mitemgot[0m[0;34m([0m[0mi[0m[0;34m)[0m[0;34m,[0m[0mctxs[0m[0;34m[[0m[0mi[0m[0;34m:[0m[0;34m:[0m[0;36m2[0m[0;34m][0m[0;34m,[0m[0mrange[0m[0;34m([0m[0mmax_n[0m[0;34m)[0m[0;34m)[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     79[0m     [0;32mreturn[0m [0mctxs[0m[0;34m[0m[0;34m[0m[0m
[1;32m     80[0m [0;34m[0m[0m

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

[0;31mAttributeError[0m: AxesImage.set() got an unexpected keyword argument 'dl'

## === cell 9
t = data.valid_ds[0][1].data
t = torch.stack([t,t])
