# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

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

# 4. Data file paths

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

# 5. Target score

0.9999

# 6. Current score

0.00047

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 4 crashes because `Path` is not defined; in this environment `Path` is not imported by the earlier cells (and `fastai.vision` here doesn’t guarantee `Path` in globals). This is an import-scope issue, not a data/path issue.  
Patch summary: Import `Path` from `pathlib` inside cell 4 (minimal, localized), then keep the existing path construction unchanged so downstream cells keep using `path`, `path_train`, and `path_test` as before.  
Updated cells: Only cell 4 is changed.  
Compatibility notes for cell k+1: `path` remains a `Path` object, so `pd.read_csv(path/'train.csv')` and `pd.read_csv(path/'sample_submission.csv')` in cell 5 work unchanged.  
Assumptions: The dataset root is still `../input` as referenced, and the expected files exist there (as shown in the provided directory listing).'
- What this solution (achieved 0.5) has done: 'Diagnosis: The crash happens because `ImageList` is not defined in fastai v2 (`fastai.vision` no longer exports the v1 `ImageList`/`databunch` API). The notebook is written for fastai v1, but the environment has fastai 2.8.5 installed. The minimal fix is to import the fastai v1 compatibility layer so `ImageList`, `get_transforms`, and `.databunch()` exist as expected.

Patch summary: Update only cell 6 to import `fastai.vision.all` (v2) and then explicitly import `ImageList` and `get_transforms` from `fastai.vision.data`/`fastai.vision.transform` to restore the expected symbols. No changes to the pipeline logic, transforms, split, normalization, or variable names.

Updated cells: (cell 6 only)

Compatibility notes for cell k+1: `data` remains a fastai DataBunch-like object with `.show_batch()` so cell 7 work unchanged.

Assumptions: fastai v2 is installed (as provided) and still includes the legacy `ImageList` API under `fastai.vision.data` and transforms under `fastai.vision.transform`.'
- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 6 is mixing fastai v2 (`fastai.vision.all`) with fastai v1 API objects (`ImageList`, `get_transforms`, `.databunch()`), but fastai==2.8.5 no longer provides `ImageList` in `fastai.vision.data`, causing the ImportError. The simplest fix is to import the legacy fastai v1 vision module (`fastai.vision`) which still exposes `ImageList` and `get_transforms` for this pipeline. This keeps the existing v1-style datablock code unchanged and ensures `data` is created as expected for cell 7.

Patch summary: In cell 6, replace the v2-only imports with v1-compatible imports by importing `fastai.vision` and pulling `ImageList` and `get_transforms` from there. No changes are made to the core data pipeline logic; only the import sources are corrected so the notebook runs under fastai v2.

Updated cells: Only cell 6 is updated.

Compatibility notes for cell k+1: `data` remains a fastai v1 `DataBunch` with `show_batch`, so `data.show_batch(...)` in cell 7 work unchanged.

Assumptions: fastai v2 still includes the `fastai.vision` legacy shim providing `ImageList`/`get_transforms` and the v1 databunch API (as commonly packaged for backward compatibility).'
- What this solution (achieved 0.5) has done: 'Diagnosis: The crash happens because the notebook uses the legacy fastai v1 API (`ImageList`, `get_transforms`, and `fastai.vision.transform`), but the environment has fastai v2 (2.8.5) where `fastai.vision.transform` no longer exists and those classes/functions were removed/relocated. Since the error is raised at import time, execution cannot proceed to build the `data` object. To keep the rest of the notebook running without redesigning the modeling/training logic, we need to provide a small compatibility shim inside the failing cell so the existing `ImageList...databunch().normalize(...)` pipeline can still be constructed.

Patch summary: In cell 6, replace the failing `from fastai.vision.transform import get_transforms` import with a version-tolerant definition of `get_transforms` (only if it cannot be imported), and provide minimal fastai v1-style aliases for `ImageList` and `imagenet_stats` when running under fastai v2 by importing them from the `fastai.vision.all` API. This keeps the downstream calls (`ImageList.from_df`, `.transform(get_transforms(...), size=128)`, `.databunch(...)`, `.normalize(imagenet_stats)`) intact and prevents the `ModuleNotFoundError`.

Updated cells: Only cell 6 is changed.

Compatibility notes for cell k+1: `data` is still created and remains an object that supports `data.show_batch(...)` as used in cell 7.

Assumptions: fastai v2 (`fastai.vision.all`) is available (it is, per installed packages), and the dataset paths/CSVs are correct as previously defined in cells 4–5.'
- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 6 crashes because it unconditionally imports `fastai.vision.transform`, which exists in fastai v1 but not in fastai v2 (your environment has fastai==2.8.5). Earlier in the same cell you already provide a compatibility `get_transforms` fallback for v2, so this later “namespace parity” import is both unnecessary and breaks execution. The root cause is the unconditional import statement near the end of the cell that bypasses the earlier try/except.

Patch summary: Remove/guard the failing `from fastai.vision.transform import get_transforms as _unused` import so it does not run under fastai v2. Keep the rest of the data pipeline unchanged so downstream cells (including `data.show_batch`) continue to work.

Updated cells: Only cell 6 is modified.

Compatibility notes for cell k+1: `data` is still created and remains a fastai DataLoaders-like object supporting `show_batch`, so cell 7 run unchanged.

Assumptions: fastai v2 APIs (`fastai.vision.all`, `ImageDataLoaders`, `Resize`, `imagenet_stats`) are available and the dataset paths used earlier are correct.'
- What this solution (achieved 0.5) has done: 'Diagnosis: The crash happens in cell 6 inside the `ImageList.databunch()` compatibility shim, where it calls `Resize(self._size)` but `Resize` is never imported/defined in this cell. In fastai v2, `Resize` lives in `fastai.vision.all`, so the symbol is missing when only `from fastai.vision import *` is used.  
Patch summary: Import `Resize` from `fastai.vision.all` (v2) with a safe fallback, so the existing shim can build `item_tfms` without changing any training/data logic.  
Updated cells: Only cell 6 is modified, adding the minimal import needed.  
Compatibility notes for cell k+1: `data` remains the same type/behavior as before (a normalized `DataLoaders`-like object with `show_batch`), so `data.show_batch(...)` in cell 7 continue to work.  
Assumptions: Environment uses fastai v2 (as installed), and `fastai.vision.all.Resize` is available.'
- What this solution (achieved 0.5) has done: 'The failure comes from a path mismatch: the notebook sets `path_train = ../input/train/train` and `path_test = ../input/test/test/`, but in this environment the images live under `../input/aerial-cactus-identification/train/` and `../input/aerial-cactus-identification/test/`. Because `ImageDataLoaders.from_name_func` is given only filenames, it looks for them relative to `self._path`; with the wrong folder it can’t find the JPGs and crashes. The minimal fix is to make cell 6 auto-detect the correct train/test image directories under `../input`, without changing the rest of the data pipeline, transforms, or labels logic. This keeps the `data` object created exactly as expected for cell 7.'
- What this solution (achieved 0.5) has done: 'The crash happens because the custom fastai-v2 compatibility shim for `ImageList.from_df(...).databunch()` passes only filenames (e.g., `"xxx.jpg"`) to `ImageDataLoaders.from_name_func`, but in fastai v2 those must be valid paths relative to the provided `path`. Since `self._path` is already the image directory, the simplest fix is to pass full `Path` objects (`self._path / filename`) as items so that image opening succeeds. This preserves the same data split, transforms, and labeling logic, while making file resolution deterministic and correct. The rest of the pipeline (including `.add_test(...)`) remains compatible because the test dataloader already uses full paths.'
- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 6 builds `data` via a compatibility `ImageList` shim that returns a fastai v2 `ImageDataLoaders` object from `.databunch(...)`. In fastai v2, `ImageDataLoaders` does not have the v1 `.normalize(...)` method, so calling `.normalize(imagenet_stats)` raises `AttributeError: normalize`. The fix is to apply normalization through fastai v2’s `Normalize.from_stats(...)` as a `batch_tfms` at DataLoaders creation time, and then skip the `.normalize(...)` call when running on v2. This preserves the same normalization semantics and keeps `data` as the same object used by cell 7 (`data.show_batch`).

Patch summary: In cell 6 only, update the `ImageList.databunch` shim to include `Normalize.from_stats(*imagenet_stats)` in `batch_tfms` when available (fastai v2), and wrap the final `.normalize(imagenet_stats)` call so it runs only when the returned object actually supports `.normalize` (fastai v1 behavior). No other logic (splits, transforms, labels, test DL) is changed.

Updated cells: cell 6.

Compatibility notes for cell k+1: `data` remains a fastai v2 `ImageDataLoaders` object with `.show_batch(...)` available, so cell 7 continues to work unchanged.

Assumptions: fastai v2 is installed (as listed) and provides `Normalize.from_stats`; `imagenet_stats` is available from `fastai.vision.all` (as already imported in the shim path).'
- What this solution (achieved 0.5) has done: 'Diagnosis: The crash happens because in fastai v2 `DataLoaders.show_batch` does not accept the fastai v1 argument name `rows`; it expects `nrows` instead. The call in cell 7 passes `rows=3`, which is forwarded down to matplotlib’s `imshow`, where it becomes an unexpected keyword and raises `AttributeError: AxesImage.set() got an unexpected keyword argument 'rows'`. This is purely an API-compatibility issue and does not affect data construction or training logic.

Patch summary: Update the `show_batch` call in cell 7 to use `nrows=3` (fastai v2 parameter) instead of `rows=3`, keeping the same intent and output. No other logic is changed.

Updated cells: Only cell 7 is modified.

Compatibility notes for cell k+1: Cell 8 only accesses `data.classes`; `data` remains unchanged, so cell 8 behave identically.

Assumptions: The notebook is running with fastai v2 (as per installed packages), so `nrows` is the correct argument name for `show_batch`.'
- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 8 fails because `data` is a fastai v2 `DataLoaders` object (created via `ImageDataLoaders.from_name_func` in cell 6), which does not expose a `.classes` attribute like fastai v1 `DataBunch` did. The class vocabulary is stored on the training `Datasets` via `data.vocab` (or `data.train_ds.vocab`). Accessing `.classes` therefore raises `AttributeError`.  
Patch summary: In cell 8 only, replace `data.classes` with a version-compatible lookup that returns the same concept (the label vocabulary) for both fastai v1 and v2, preferring `.classes` when present and otherwise falling back to `.vocab`.  
Updated cells: Cell 8 is updated to compute and display the classes robustly without changing any training/inference logic.  
Compatibility notes for cell k+1: Cell 9 uses `cnn_learner(data, ...)` and is unaffected; `data` remains the same object and only the way we print classes changes.  
Assumptions: The classification labels are binary (0/1) and fastai v2 provides `data.vocab` (or `data.train_ds.vocab`) as the label names.'
- What this solution (achieved 0.5) has done: 'The crash happens because `fastai.metrics` in fastai v2 no longer exposes `roc_curve`, so `from fastai.metrics import roc_curve` raises an `ImportError`. This notebook doesn’t use `roc_curve` in the shown cells, so the smallest safe fix is to remove that import while keeping the learner construction unchanged. To preserve compatibility in case later cells reference `roc_curve`, we provide a fallback import from `sklearn.metrics`. No changes are made to the model, data, or training logic.'
- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 9 crashes because `cnn_learner` is a fastai v1 API symbol and isn’t imported/available under the current environment (fastai==2.8.5). In fastai v2 the equivalent is `vision_learner`, and metrics like `accuracy` should be imported from `fastai.metrics` (or `fastai.vision.all`). The current notebook earlier tries to maintain v1-style objects, so the minimal fix is to make cell 9 version-adaptive: use `cnn_learner` if it exists, otherwise fall back to fastai v2’s `vision_learner` while keeping the same backbone (resnet50), metric (accuracy), and model_dir behavior.  

Patch summary: Update cell 9 to import the needed fastai v2 symbols when `cnn_learner` is unavailable, then create `learn` using `cnn_learner` (v1) or `vision_learner` (v2) accordingly. This resolves the NameError without changing the model/backbone or training semantics.  

Updated cells: Only cell 9 is modified.  

Compatibility notes for cell k+1: Cell 10 expects `learn` to exist and support `lr_find()`. The patched `learn` is a valid fastai v2 `Learner` (or v1 `Learner`), so `learn.lr_find()` remains callable.  

Assumptions: The `data` object returned in earlier cells is compatible with fastai v2 learner creation (it is an `ImageDataLoaders` instance in this environment).'
- What this solution (achieved 0.5) has done: 'Diagnosis: The crash happens in cell 11 because, under fastai v2 (`vision_learner`), `learn.recorder` is not a v1-style `Recorder` with a `.plot()` method; attribute forwarding ends up resolving `recorder` to a model component (a `torch.nn.Sequential`), which indeed has no `.plot`. In fastai v2 the equivalent plot lives on `learn.recorder` but is named `plot_loss()`. So we need to call the correct plotting method depending on which recorder API is available, without changing training/evaluation logic.

Patch summary: Update cell 11 to call `learn.recorder.plot_loss()` when `.plot()` is not present, falling back to `.plot()` for older fastai versions. This keeps all objects (`learn`) unchanged and only fixes the plotting call that crashes.

Updated cells: cell 11 only.

Compatibility notes for cell k+1: Cell 12 only defines `lr`; this patch does not change any variables used later, and `learn` remains the same object.

Assumptions: Matplotlib is available/enabled in the notebook environment (as implied by earlier `%matplotlib inline`), and the intent of cell 11 is to visualize the LR finder/training loss recorder without affecting training.'
- What this solution (achieved 0.00047) has done: 'Diagnosis: The crash happens because `DatasetType` is a fastai v1 enum and is not available/imported in fastai v2, which your environment uses (fastai==2.8.5). In fastai v2, `Learner.get_preds` selects the test dataloader via `dl=` (e.g., `learn.dls.test_dl(...)`), not `ds_type=DatasetType.Test`. The minimal fix is to make cell 14 version-compatible by trying the v1 call when `DatasetType` exists, otherwise using the v2 test dataloader already created earlier (`dls.test` or `dls.test_dl`). This preserves the same semantics: predictions on the test set, returning `preds` for downstream use in cell 15.

Patch summary: Update only cell 14 to (1) use `DatasetType.Test` when available, and (2) otherwise call `learn.get_preds(dl=...)` with the test dataloader from `data`/`learn.dls`. Keep variable names (`preds, _`) unchanged so cell 15 continues to work.

Updated cells: cell 14 only.

Compatibility notes for cell k+1: `preds` remains a tensor-like object supporting `preds[:, 0]` exactly as cell 15 expects.

Assumptions: `data` is a fastai v2 `DataLoaders` (or v1 `DataBunch`) created in earlier cells, and for fastai v2 a test dataloader exists either as `data.test` (set in cell 6) or can be built via `learn.dls.test_dl(...)`.'

# 9. Code solution

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

            items = [self._path / f for f in files]

            item_tfms = [Resize(self._size)] if self._size is not None else None

            batch_tfms = self._tfms if self._tfms is not None else None
            try:
                from fastai.vision.all import Normalize  # fastai v2

                norm_tfm = Normalize.from_stats(*imagenet_stats)
                if batch_tfms is None:
                    batch_tfms = [norm_tfm]
                else:
                    batch_tfms = list(batch_tfms) + [norm_tfm]
            except Exception:
                pass

            dls = ImageDataLoaders.from_name_func(
                self._path,
                items,
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


def _resolve_img_dir(p: Path, candidates):
    for c in candidates:
        d = p / c
        if d.exists():
            return d
    return None


path = Path("../input")
path_train = _resolve_img_dir(
    path,
    ["train/train", "aerial-cactus-identification/train", "train"],
)
path_test = _resolve_img_dir(
    path,
    ["test/test", "aerial-cactus-identification/test", "test"],
)

if path_train is None or path_test is None:
    raise FileNotFoundError(
        f"Could not locate train/test image folders under {path}. "
        f"train={path_train}, test={path_test}"
    )

np.random.seed(42)
test = ImageList.from_df(test_df, path=path_test)
data = (
    ImageList.from_df(labels_df, path=path_train)
    .split_by_rand_pct(0.05)
    .label_from_df()
    .add_test(test)
    .transform(get_transforms(flip_vert=True, max_warp=0.0), size=128)
    .databunch(path=path, bs=bs)
)

if hasattr(data, "normalize"):
    data = data.normalize(imagenet_stats)

data


## === cell 7
data.show_batch(nrows=3, figsize=(10, 8))


## === cell 8
if hasattr(data, "classes"):
    data.classes
elif hasattr(data, "vocab"):
    data.vocab
elif hasattr(data, "train_ds") and hasattr(data.train_ds, "vocab"):
    data.train_ds.vocab
else:
    raise AttributeError(
        "Could not find classes/vocab on `data` for this fastai version."
    )


## === cell 9
try:
    from fastai.metrics import roc_curve  # fastai v1 (if available)
except Exception:
    from sklearn.metrics import roc_curve  # compatible replacement if referenced later

try:
    learn = cnn_learner(
        data, models.resnet50, metrics=accuracy, model_dir="/tmp/model/"
    )
except NameError:
    from fastai.vision.all import vision_learner, resnet50
    from fastai.metrics import accuracy as _accuracy

    learn = vision_learner(data, resnet50, metrics=_accuracy, model_dir="/tmp/model/")


## === cell 10
learn.lr_find()


## === cell 11
if hasattr(getattr(learn, "recorder", None), "plot"):
    learn.recorder.plot()
else:
    learn.recorder.plot_loss()


## === cell 12
lr = 1e-2


## === cell 13
learn.fit_one_cycle(5, slice(lr))


## === cell 14
try:
    preds, _ = learn.get_preds(ds_type=DatasetType.Test)
except NameError:
    if hasattr(data, "test") and data.test is not None:
        test_dl = data.test
    elif (
        hasattr(learn, "dls")
        and hasattr(learn.dls, "test")
        and learn.dls.test is not None
    ):
        test_dl = learn.dls.test
    else:
        test_files = test_df.iloc[:, 0].astype(str).tolist()
        test_items = [path_test / f for f in test_files]
        test_dl = learn.dls.test_dl(test_items)
    preds, _ = learn.get_preds(dl=test_dl)


## === cell 15
preds[:, 0]


## === cell 16
test_df['has_cactus'] = np.array(preds[:, 0])
test_df.head()


## === cell 17
test_df.to_csv('submission.csv', index = False)
