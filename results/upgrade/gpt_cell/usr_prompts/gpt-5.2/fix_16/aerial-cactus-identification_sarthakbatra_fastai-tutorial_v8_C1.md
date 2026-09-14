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

0.9998

# 6. Current score

0.00102

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 4 crashes because `Path` is not defined in this environment. In fastai v2, `Path` is not automatically injected into the global namespace by `from fastai import *` / `from fastai.vision import *`, so relying on it causes a `NameError`.  
Patch summary: Import `Path` from `pathlib` inside cell 4 (the failing cell) to make `Path('../input')` valid without changing any dataset paths or downstream variables.  
Updated cells: Only cell 4 is modified.  
Compatibility notes for cell k+1: `path`, `path_train`, and `path_test` remain `pathlib.Path` objects with the same values, so cell 5 continues to work unchanged.  
Assumptions: `../input` is the correct base directory for the Kaggle-style dataset layout in this runtime.'
- What this solution (achieved 0.5) has done: 'Diagnosis: The crash happens because `ImageList` (a fastai v1 API class) is not available in fastai==2.8.5, so the import `from fastai.vision import *` does not define `ImageList`. The rest of the cell also uses other fastai v1 DataBunch APIs (`split_by_rand_pct`, `label_from_df`, `add_test`, `transform`, `databunch`, `imagenet_stats`). The minimal fix is to import the legacy fastai v1 compatibility module that still provides these symbols.

Patch summary: In cell 6, add an explicit import from `fastai.vision.all` and (critically) `fastai.vision.data` to bring back the v1-style `ImageList`/`imagenet_stats` names into scope, without changing the data pipeline logic.

Updated cells: Only cell 6 is modified.

Compatibility notes for cell k+1: `data` remains a fastai v1-style `DataBunch` with `show_batch`, so `data.show_batch(...)` in cell 7 continues to work unchanged.

Assumptions: The environment’s fastai installation includes the v1 compatibility layer (`fastai.vision.data.ImageList`) as shipped with fastai==2.8.5.'
- What this solution (achieved 0.5) has done: 'Diagnosis: The crash happens because `ImageList` is a fastai v1 API symbol, but this environment has fastai 2 installed; in fastai v2 `ImageList` is not defined, causing the `NameError`. Cell 6 is mixing v1-style data block code with v2 imports (`fastai.vision.all`). To keep the same pipeline semantics (load images from a dataframe, random split, label from dataframe, add test set, apply transforms, normalize), we should rewrite cell 6 using fastai v2’s `DataBlock`/`ImageDataLoaders` equivalents while keeping the same parameters (seed, split pct, image size, augmentation options, batch size, and imagenet normalization). This also preserve `data` so cell 7 (`data.show_batch`) continues to work.

Patch summary: Replace the v1 `ImageList`/`.databunch()` pipeline in cell 6 with a v2 `DataBlock` that reads filenames from `id`, labels from `has_cactus`, performs the same random split (1%), uses the same item size (128), uses similar augmentation (`flip_vert=True` and no warping), and normalizes with `imagenet_stats`. Ensure `data` is an `ImageDataLoaders` instance and includes a test dataloader built from `test_df`.

Updated cells: Only cell 6 is modified.

Compatibility notes for cell k+1: `data` remains the main dataloaders object and supports `data.show_batch(...)` as used in cell 7. The test set is attached via `data.test_dl(...)`, matching the prior intent of `.add_test(test)`.

Assumptions: The CSV column containing filenames is `id` (as shown) and label column is `has_cactus`; images exist under `path_train` and `path_test` with those filenames; fastai v2 is the intended library version in this environment.'
- What this solution (achieved 0.5) has done: 'Diagnosis: The crash happens while FastAI tries to open images using only the bare filename returned by `get_x`. Because `dblock.dataloaders(..., path=path_train)` is used with `path_train` pointing to `../input/train/train` (which does not exist in this dataset layout), FastAI cannot resolve filenames to real files and ends up trying to open `'ee0b1f1841d6bd4a786c430753917bc2.jpg'` relative to the current working directory. The dataset in this environment is stored under `../input/aerial-cactus-identification/train/` and `../input/aerial-cactus-identification/test/`. Fixing the data root paths inside cell 6 (where the loaders are created) unblocks image loading without changing the model/training semantics.

Patch summary: In cell 6, derive `train_path` and `test_path` from the existing `path` variable by selecting the actual available directory (`../input/aerial-cactus-identification/...`) with a safe fallback to the previously defined paths if they exist. Then pass these corrected paths into `dblock.dataloaders` and `test_dl` so `_get_x` can remain unchanged and filenames resolve correctly.

Updated cells: Only cell 6 is modified.

Compatibility notes for cell k+1: `data` remains a FastAI `DataLoaders` object, so `data.show_batch(...)` in cell 7 work unchanged.

Assumptions: The extracted images are present in `../input/aerial-cactus-identification/train/` and `../input/aerial-cactus-identification/test/` as shown in the provided filesystem listing; if the older `path_train/path_test` exist they be used.'
- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 6 builds a `DataBlock` where `get_x` returns only the filename (e.g., `"abc.jpg"`), but `DataLoaders` is created with `path=train_path` that is currently pointing to a non-existent directory (`../input/train/train`). Since that path doesn’t exist, fastai falls back to looking for the image in the current working directory, causing `FileNotFoundError` when trying to open `"ee0b1f...jpg"` without its parent folder. The intended images are under `../input/aerial-cactus-identification/train` and `.../test` in this environment.

Patch summary: In cell 6, override `train_path` and `test_path` with a small deterministic path resolver that selects the first existing candidate among the known dataset layouts, including the nested `aerial-cactus-identification/aerial-cactus-identification/{train,test}` structure shown in the file tree. This keeps the core DataBlock logic unchanged and only fixes the directory selection so file lookups succeed. `data` remains a `DataLoaders` object compatible with cell 7.

Updated cells: (cell 6 only)

Compatibility notes for cell k+1: `data` is still defined as the test dataloader (`data = data.test_dl(...)`), so `data.show_batch(...)` in cell 7 continue to work as before.

Assumptions: The dataset exists in one of the listed candidate folders under `../input`, and the CSV `id` values match the filenames in those folders.'
- What this solution (achieved 0.5) has done: 'Diagnosis: The crash happens when FastAI tries to open an image using only the filename (e.g., `ee0b1f...jpg`) without joining it to the `train_path`. In a `DataBlock`, `get_x` should return a full path (or a path relative to the provided `path=`), but here `_get_x` returns just the `id` string, so `PIL.Image.open` looks in the current working directory and fails. The same issue would affect the test dataloader as well.  

Patch summary: Modify `_get_x` to return a `Path` by joining the filename to the correct root directory (`train_path` for training/valid and `test_path` for test). Keep the rest of the `DataBlock` and augmentation/training semantics unchanged; only fix how file paths are constructed so the images can be found.  

Updated cells:  

Compatibility notes for cell k+1: `data` remains a valid FastAI `DataLoaders` object, so `data.show_batch(...)` in cell 7 work unchanged.  

Assumptions: The `id` column values in both `train.csv` and `sample_submission.csv` are image filenames located directly under the resolved `train_path` and `test_path` directories, respectively.'
- What this solution (achieved 0.5) has done: 'Diagnosis: The crash happens because `data` in cell 6 is overwritten from a `DataLoaders` (train/valid) into a single test `DataLoader` via `data = data.test_dl(...)`. Then `data.show_batch()` in cell 7 tries to open items using `_get_x`, but `_get_x` uses `train_path/test_path` roots that don’t match the `path=` passed into `test_dl`, causing it to look for files under `../input/train/train/...` that don’t exist in this environment. The simplest fix is to stop overwriting `data` and instead keep the original `DataLoaders` in `data` and store the test dataloader in a separate variable.

Patch summary: In cell 6, keep `data` as the training/validation `DataLoaders` and assign the test dataloader to `test_dl` (new variable) rather than overwriting `data`. This preserves the intended training semantics and prevents `show_batch` from trying to index through the test set with incorrect paths.

Updated cells: only cell 6 is changed.

Compatibility notes for cell k+1: Cell 8 uses `data.classes`; with this patch `data` remains a `DataLoaders` and still has `.classes`, so cell 8 remains compatible. (If later cells expect the test dataloader, it is now available as `test_dl`.)

Assumptions: Later cells can reference a separately named test dataloader (or can be trivially updated later if needed), but the immediate goal is to unblock execution at cell 7 without changing model/training logic.'
- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 8 fails because `data` is no longer a `DataLoaders` object with a vocab/classes attached; in cell 6 `data` is overwritten with the result of `data.test_dl(...)`, which is a `DataLoader`. A `DataLoader` does not expose `.classes`, so the attribute lookup raises `AttributeError: classes`. The core issue is the accidental overwrite of the training `DataLoaders` object.

Patch summary: In cell 8, avoid relying on `data.classes` and instead retrieve the class names from the training `DataLoaders` created in cell 7 (which remains intact) via `data.vocab` (fastai v2). This is a minimal change localized to the failing cell and preserves downstream behavior.

Updated cells: cell 8 only.

Compatibility notes for cell k+1: Cell 9 expects `data` to be a `DataLoaders` instance for `cnn_learner(...)`. With this patch, `data` remains unchanged (still the `DataLoaders` from cell 7), so cell 9 continues to work, and you still get the class list displayed.

Assumptions: `data` in cell 7 is a `fastai.data.core.DataLoaders` with a populated `vocab` attribute (standard for `CategoryBlock`).'
- What this solution (achieved 0.5) has done: 'Diagnosis: The crash happens in cell 11 because `learn.recorder` is not the fastai `Recorder` callback in this fastai v2 setup; instead it resolves to the model (a `torch.nn.Sequential`) via fastcore attribute delegation, so calling `.plot()` raises `AttributeError`. In fastai v2, the recorder plots are available as methods on the `Learner` itself (e.g., `learn.plot_loss()`), which preserves the same intent (plotting training loss) without changing training logic.

Patch summary: Replace the invalid `learn.recorder.plot()` call with the fastai v2-compatible `learn.plot_loss()`. This is a minimal change localized to cell 11 and keeps all variables and training semantics unchanged.

Updated cells: (cell 11 only)

Compatibility notes for cell k+1: Cell 12 only sets `lr` and does not depend on the plotting call; no interface/variable changes occur.

Assumptions: fastai v2 plotting utilities are available in the current environment and `learn` has been created successfully in cell 9 and `lr_find()` has been run in cell 10.'
- What this solution (achieved 0.5) has done: 'Diagnosis: The crash in cell 11 happens because `learn.plot_loss()` is being resolved via fastcore’s attribute delegation onto `learn.model` (a `torch.nn.Sequential`), not onto the fastai `Learner`/`Recorder` helper. As a result, Python tries to call `plot_loss` on the underlying model, which doesn’t exist, raising `AttributeError: 'Sequential' object has no attribute 'plot_loss'`. The minimal fix is to call the recorder’s plot method directly (which is where loss plotting lives in fastai v2).

Patch summary: In cell 11, replace `learn.plot_loss()` with `learn.recorder.plot_loss()` so plotting uses the `Recorder` instead of the model.

Updated cells: Only cell 11 is changed.

Compatibility notes for cell k+1: Cell 12 (`lr = 3e-03`) is unaffected; no variables or interfaces are changed besides fixing the plotting call.

Assumptions: Training (`learn.lr_find()`) ran and populated `learn.recorder` with loss history; if it didn’t, the plot be empty but not crash.'
- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 14 crashes because `DatasetType` is a fastai v1 symbol; in fastai v2 (installed here), `DatasetType` is not defined, so `learn.get_preds(ds_type=DatasetType.Test)` raises a `NameError`. In fastai v2, `get_preds` selects the test dataloader via `dl=` (or you can use `ds_idx=1`), and you already created `test_dl` in cell 7.  
Patch summary: Update cell 14 to call `learn.get_preds(dl=test_dl)` so predictions are computed on the intended test dataloader without relying on the removed `DatasetType` enum.  
Updated cells: Only cell 14 is changed.  
Compatibility notes for cell k+1: The variables `preds` and `_` are still created with the same meaning, so cell 15 (`preds[:, 0]`) continues to work.  
Assumptions: `test_dl` exists (created in cell 7) and corresponds to the Kaggle test set.'
- What this solution (achieved 0.5) has done: 'Diagnosis: The crash in cell 14 happens because `test_dl` is built from `test_df` but the `DataBlock`’s `get_x` chooses the root folder using the expression `"has_cactus" in r`, which is not a reliable way to detect whether a row has labels. For `test_df` rows this condition can evaluate truthy, so it incorrectly points to the training image directory, and then tries to open a test image filename under `../input/train/train/...`, causing the FileNotFoundError.  
Patch summary: Update cell 14 to rebuild a safe `test_dl` (only for inference) whose `get_x` always reads from `test_path`, and then call `learn.get_preds` on that dataloader. This keeps the model, training, and prediction semantics unchanged—only fixes the path resolution for test images.  
Updated cells: Only cell 14 is changed.  
Compatibility notes for cell k+1: `preds` remains a tensor of shape `(len(test_df), 2)` as before, so `preds[:, 0]` in cell 15 still works identically.  
Assumptions: `train_path`, `test_path`, `test_df`, `bs`, and `learn` already exist from earlier cells (as shown), and the test images live under `test_path` as resolved earlier.'
- What this solution (achieved 0.5) has done: 'Diagnosis: The crash in cell 14 is a `FileNotFoundError` while building/iterating the test DataLoader: it tries to open `../input/train/train/<id>.jpg`, which indicates the test dataloader is resolving image paths against the training folder (or a mismatched folder layout). This happens because `test_path` (and/or the folder layout under `../input`) is not guaranteed to match `../input/test/test`, and the custom `get_x` can still yield non-existent paths if the chosen `test_path` is wrong or if `id` values need to be resolved against the actually existing directory. The minimal fix is to make `_get_x_test` deterministically locate each test image by trying the known candidate directories (including the nested Kaggle dataset structure) and returning the first existing file; this prevents fastai from attempting to open missing files.

Patch summary: Modify only cell 14 by replacing `_get_x_test` with a robust resolver that checks multiple candidate directories for each `id` and returns the first existing path, raising a clear error if none exist. Keep the same `learn.dls.test_dl(...)` + `learn.get_preds(...)` flow and output tensors unchanged.

Updated cells: Only cell 14 is updated below.

Compatibility notes for cell k+1: `preds` remains a tensor of model predictions with the same shape and meaning, so `preds[:, 0]` in cell 15 continues to work unchanged.

Assumptions: The test images exist in one of the provided dataset folder variants (e.g., `../input/test/test`, `../input/test`, `../input/aerial-cactus-identification/test`, or the nested `../input/aerial-cactus-identification/aerial-cactus-identification/test`). The `id` column values include the `.jpg` suffix as in the original CSVs.'
- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 14 recreates a test dataloader using `learn.dls.test_dl(test_df, ...)`, but the existing `learn.dls` still uses the training `DataBlock`’s `get_x` logic (from earlier cells) which can resolve paths incorrectly and, during iteration, tries to open an image from `../input/train/train/...` that doesn’t exist in this environment layout. This happens because the dataset item transform pipeline is still tied to the original `DataLoaders` and not guaranteed to honor the intended `test_path`/custom lookup for every item as expected. The earlier notebook already built a correct `test_dl` in cell 7 (`test_dl = data.test_dl(...)`) using the resolved `test_path`.  

Patch summary: In cell 14, stop rebuilding the test dataloader from `learn.dls` and instead reuse the already-correct `test_dl` created in cell 7. This is the smallest change that avoids the incorrect train/test path resolution and keeps the prediction logic identical (`learn.get_preds(dl=test_dl)`).  

Updated cells:  

Compatibility notes for cell k+1: Cell 15 expects `preds` to exist; this patch still defines `preds` with the same type/shape returned by `learn.get_preds(...)`, so `preds[:, 0]` remains valid.  

Assumptions: Cell 7 executed successfully and produced a valid `test_dl` pointing at the real test image directory.'
- What this solution (achieved 0.00102) has done: 'Diagnosis: The crash happens during `learn.get_preds(dl=test_dl)` because `test_dl` is built from `data.test_dl(test_df, ...)` in cell 7, but the `get_x` function decides whether to use `train_path` or `test_path` by checking `"has_cactus" in r`. For a pandas Series row, that expression checks membership in the *values*, not column names, so it can incorrectly route some test rows to `train_path`, leading to missing-file lookups like `../input/train/train/<id>.jpg`.  
Patch summary: In cell 14 only, rebuild a safe test DataLoader that always resolves images under `test_path` (and uses the same item/batch transforms as training), then call `learn.get_preds` on that loader. This keeps the model and prediction semantics unchanged while preventing accidental path switching.  
Updated cells: Only cell 14 is changed.  
Compatibility notes for cell k+1: The variables `preds` (a tensor of predictions) and `_` are still produced with the same shapes/types expected by cell 15 (`preds[:, 0]`).  
Assumptions: `test_path` and `test_df` were successfully created in earlier cells (as shown), and `learn.dls` exists and contains the training transforms needed to create a compatible test DataLoader.'

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
from fastai.vision.all import *
from fastai.vision.data import *

np.random.seed(42)


def _get_x(r):
    root = train_path if "has_cactus" in r else test_path
    return root / r["id"]


def _get_y(r):
    return r["has_cactus"]


dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_x=_get_x,
    get_y=_get_y,
    splitter=RandomSplitter(valid_pct=0.01, seed=42),
    item_tfms=Resize(128),
    batch_tfms=[
        *aug_transforms(do_flip=True, flip_vert=True, max_warp=0.0, size=128),
        Normalize.from_stats(*imagenet_stats),
    ],
)


def _first_existing_path(*candidates):
    for p in candidates:
        p = Path(p)
        if p.exists():
            return p
    raise FileNotFoundError(
        f"None of the candidate paths exist: {[str(Path(c)) for c in candidates]}"
    )


train_path = _first_existing_path(
    path / "train" / "train",
    path / "train",
    path / "aerial-cactus-identification" / "train",
    path / "aerial-cactus-identification" / "aerial-cactus-identification" / "train",
)

test_path = _first_existing_path(
    path / "test" / "test",
    path / "test",
    path / "aerial-cactus-identification" / "test",
    path / "aerial-cactus-identification" / "aerial-cactus-identification" / "test",
)

data = dblock.dataloaders(labels_df, path=train_path, bs=bs)
data = data.test_dl(test_df, with_labels=False, path=test_path)


## === cell 7
from fastai.vision.all import *
from fastai.vision.data import *

np.random.seed(42)


def _get_x(r):
    root = train_path if "has_cactus" in r else test_path
    return root / r["id"]


def _get_y(r):
    return r["has_cactus"]


dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_x=_get_x,
    get_y=_get_y,
    splitter=RandomSplitter(valid_pct=0.01, seed=42),
    item_tfms=Resize(128),
    batch_tfms=[
        *aug_transforms(do_flip=True, flip_vert=True, max_warp=0.0, size=128),
        Normalize.from_stats(*imagenet_stats),
    ],
)


def _first_existing_path(*candidates):
    for p in candidates:
        p = Path(p)
        if p.exists():
            return p
    raise FileNotFoundError(
        f"None of the candidate paths exist: {[str(Path(c)) for c in candidates]}"
    )


train_path = _first_existing_path(
    path / "train" / "train",
    path / "train",
    path / "aerial-cactus-identification" / "train",
    path / "aerial-cactus-identification" / "aerial-cactus-identification" / "train",
)

test_path = _first_existing_path(
    path / "test" / "test",
    path / "test",
    path / "aerial-cactus-identification" / "test",
    path / "aerial-cactus-identification" / "aerial-cactus-identification" / "test",
)

data = dblock.dataloaders(labels_df, path=train_path, bs=bs)
test_dl = data.test_dl(test_df, with_labels=False, path=test_path)


## === cell 8
data.vocab


## === cell 9
learn = cnn_learner(data, models.resnet101, metrics = accuracy, model_dir='/tmp/model/')


## === cell 10
learn.lr_find()


## === cell 11
learn.recorder.plot_loss()


## === cell 12
lr = 3e-03


## === cell 13
learn.fit_one_cycle(5, slice(lr))


## === cell 14
from fastai.vision.all import *

test_files = (test_path / test_df["id"]).tolist()
test_dl_safe = learn.dls.test_dl(test_files)

preds, _ = learn.get_preds(dl=test_dl_safe)


## === cell 15
preds[:, 0]


## === cell 16
test_df['has_cactus'] = np.array(preds[:, 0])
test_df.head()


## === cell 17
test_df.to_csv('submission.csv', index = False)
