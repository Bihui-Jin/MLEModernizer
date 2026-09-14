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
Given a dataset of images of scanned text that is noisy, remove the noise.

## Metric
Root mean squared error between the cleaned pixel intensities and the actual grayscale pixel intensities.

## Submission Format
Form the submission file by melting each images into a set of pixels, assigning each pixel an id of image_row_col (e.g. 1_2_1 is image 1, row 2, column 1). Intensity values range from 0 (black) to 1 (white). The file should contain a header and have the following format:

```
id,value
1_1_1,1
1_2_1,1
1_3_1,1
etc.
```

## Dataset
You are provided two sets of images, train and test. These images contain various styles of text, to which synthetic noise has been added to simulate real-world, messy artifacts. The training set includes the test without the noise (train_cleaned).

# 2. Python version

3.9

# 3. Installed packages

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
protobuf==6.33.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tqdm==4.67.1

# 4. Data file paths

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

# 5. Target score

0.02895

# 6. Current score

0.28616

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28616) has done: 'The crash happens immediately in cell 0 while importing `imgaug`, due to an incompatibility between `imgaug` and the installed `protobuf==6.33.0` (the `MessageFactory.GetPrototype` API was removed/changed). Since later cells only need `imgaug` for augmentation, the minimal unblock is to avoid importing `imgaug` at import-time so the notebook can run; this keeps the rest of the pipeline intact and doesn’t change any model/training logic. The patch wraps the `imgaug` import in a safe `try/except` and provides a clear error if/when augmentation is actually invoked later, rather than crashing at startup. No other imports or logic are changed.'
- What this solution (achieved 0.28616) has done: 'The crash happens in cell 0 while importing `imgaug`, which internally relies on older protobuf APIs; with protobuf 6.x this commonly fails with `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. Since `imgaug` is optional in this notebook (you already guard it with `try/except`), the minimal deterministic fix is to avoid importing `imgaug` when protobuf is too new, and fall back to `ia=None, iaa=None` without triggering the protobuf internals. This preserves all existing logic and keeps downstream cells working exactly the same way they would have after a failed optional import. No other cells are modified.'
- What this solution (achieved 0.47918) has done: 'Diagnosis: The crash happens during `imgaug` import because this environment uses `protobuf==6.33.0`, and `imgaug` relies on older protobuf APIs (it triggers `'MessageFactory' object has no attribute 'GetPrototype'`). The current guard in cell 0 only skips `imgaug` for protobuf >= 4.0, but the failure occurs during the import attempt itself, so we must reliably prevent importing `imgaug` under incompatible protobuf versions.  
Patch summary: In cell 0, tighten the protobuf compatibility check to skip `imgaug` for protobuf >= 4 (including 6.x) and avoid executing any `imgaug` imports in that case; keep `ia/iaa` as `None` exactly as the notebook expects. This preserves downstream behavior while eliminating the crash.  
Updated cells: Only cell 0 is modified.  
Compatibility notes for cell k+1: Cell 1 does not depend on `imgaug`; it only needs the imports in cell 0 to succeed. Variables `ia` and `iaa` remain defined (as `None`) as before, so any later optional augmentation code remains guarded/compatible.  
Assumptions: The notebook can run without `imgaug` (augmentations are optional or guarded elsewhere), and no later cell unconditionally calls `iaa.*` without checking for `None`.'
- What this solution (achieved 0.28616) has done: 'Diagnosis: The notebook crashes in cell 0 during import-time initialization because `imgaug` is being imported in an environment with `protobuf==6.33.0`. `imgaug` relies on older protobuf APIs and triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` when protobuf is too new. The code is already trying to skip `imgaug` for protobuf>=4, but the failure occurs before this guard can safely prevent the problematic import path. The fix is to remove any attempt to import `imgaug` entirely in cell 0 and set `ia/iaa` to `None`, keeping the rest of the notebook intact.

Patch summary: In cell 0, replace the conditional `imgaug` import block with a deterministic “disabled” stub (sets `ia=None`, `iaa=None`, and records the reason). This prevents the protobuf-related crash while preserving downstream variable names (`ia`, `iaa`, `_imgaug_import_error`) so later cells (if any) remain compatible.

Updated cells: Only cell 0 is modified.

Compatibility notes for cell k+1: Cell 1 and later cells still see `ia`, `iaa`, and `_imgaug_import_error` defined; no other interfaces are changed.

Assumptions: `imgaug` is not strictly required for core execution in this environment, and any later augmentation code either checks for `iaa is None` or can proceed without augmentations.'
- What this solution (achieved 0.46935) has done: 'Diagnosis: The crash happens immediately in cell 0 due to an `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`, which is a known incompatibility between `imgaug` (and its protobuf usage) and the installed `protobuf==6.33.0`. Even though the code already tries to disable `imgaug`, the import attempt is still occurring (implicitly or via earlier code), so execution stops before any variables are set. The fix is to make cell 0 robust by explicitly guarding the optional `imgaug` import and always defining `ia`/`iaa` placeholders when the import fails.

Patch summary: In cell 0 only, add a safe `try/except` around importing `imgaug` so the protobuf-related `AttributeError` is caught, and ensure `ia`, `iaa`, and `_imgaug_import_error` are defined deterministically. This preserves the original intent (disable imgaug in this environment) while preventing the crash.

Updated cells: Cell 0 only (buggy cell).

Compatibility notes for cell k+1: No interface/variable changes; `ia` and `iaa` still exist (set to `None` when unavailable), so any later code referencing them continue to run.

Assumptions: `imgaug` is either installed but incompatible or not installed; in both cases it should be treated as optional and disabled without stopping execution.'
- What this solution (achieved 0.46859) has done: 'Diagnosis: The crash happens during `import imgaug`, which indirectly imports protobuf-generated code. With `protobuf==6.33.0`, older `imgaug`/dependencies can fail at import time with `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This occurs before any model logic runs and is unrelated to your core training/inference code. Since your notebook already treats `imgaug` as optional, the safest fix is to prevent the incompatible protobuf API from being used by switching protobuf to its pure-Python implementation before the import attempt.

Patch summary: In cell 0 only, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and a version env var for compatibility) before attempting to import `imgaug`. This avoids the `MessageFactory.GetPrototype` crash while preserving the existing fallback behavior (`ia/iaa=None` if import still fails).

Updated cells: cell 0 only.

Compatibility notes for cell k+1: No variables used by cell 1 are changed; imports and names remain identical (`np`, `pd`, `plt`, `sns`, `zipfile`, `os`, `cv2`, `tqdm`, `tf`, `Model`, `layers`, `callbacks`, `utils`, `ia`, `iaa`, `_imgaug_import_error`).

Assumptions: `imgaug` is optional for the rest of the notebook (as implied by the existing try/except), so forcing the pure-Python protobuf implementation is acceptable and does not change model/training semantics.'
- What this solution (achieved 0.28616) has done: 'Diagnosis: The crash happens during the TensorFlow import in cell 0, before any notebook logic runs. With `protobuf==6.33.0` installed, TensorFlow/TFDS can hit an incompatibility where some code still expects `google.protobuf.message_factory.MessageFactory.GetPrototype`, which was removed in protobuf 5+ (leading to `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`). Setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is not sufficient to restore that removed method. The minimal fix is to monkey‑patch `MessageFactory.GetPrototype` to call the modern equivalent `GetMessageClass` before importing TensorFlow.

Patch summary: In cell 0 only, add a small compatibility shim that defines `MessageFactory.GetPrototype` when missing, using `GetMessageClass` if available, then proceed with the existing imports unchanged. This keeps the notebook’s core logic intact and simply unblocks TensorFlow import under protobuf 6.x.

Updated cells: (cell 0 only)

Compatibility notes for cell k+1: No variables, paths, or interfaces are changed; all imports remain available exactly as before. Cell 1 run identically once TensorFlow imports successfully.

Assumptions: `google.protobuf` is installed (it is, via `protobuf==6.33.0`) and exposes `message_factory.MessageFactory` and `GetMessageClass` in this environment.'
- What this solution (achieved 0.28616) has done: 'The crash happens in the protobuf compatibility shim: it tries to access `MessageFactory.GetPrototype` on a `MessageFactory` implementation where that attribute lookup itself raises `AttributeError`. The simplest deterministic fix is to avoid touching `MessageFactory.GetPrototype` directly and instead use `getattr(..., None)` so the check is safe across protobuf versions. Then we only monkey-patch `GetPrototype` when it’s missing and `GetMessageClass` exists. This keeps the rest of the notebook unchanged and unblocks TensorFlow imports.'
- What this solution (achieved 0.28616) has done: 'Diagnosis: The crash happens in cell 0 while trying to monkey‑patch `google.protobuf.message_factory.MessageFactory.GetPrototype`. In protobuf v6, `MessageFactory` can be implemented differently (and may not allow adding that attribute or may not expose the class/instance members as expected), so the attempted access/assignment triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This patching is also not required for this notebook to run in this environment; it was likely a compatibility workaround for older TF/protobuf combinations. The minimal safe fix is to guard the patch more defensively and silently skip it when `MessageFactory` is not patchable.

Patch summary: In cell 0, wrap the protobuf patch so it only executes when `MessageFactory` is a normal patchable class and `GetPrototype` is truly missing but `GetMessageClass` exists, and ensure we never raise if protobuf internals differ. No other logic (imports, model code, training, etc.) is changed.

Updated cells: Only cell 0 is modified.

Compatibility notes for cell k+1: All imports and variables defined in cell 0 (`np`, `pd`, `plt`, `sns`, `zipfile`, `cv2`, `tqdm`, `tf`, Keras imports, and optional `imgaug` variables) remain unchanged and available for cell 1; only the protobuf workaround behavior changes from “crash” to “skip”.

Assumptions: TensorFlow 2.18.0 with protobuf 6.33.0 does not require this `GetPrototype` monkey‑patch for the rest of the notebook to execute; skipping it is safe and preserves intended execution.'
- What this solution (achieved 0.28616) has done: 'Diagnosis: The crash happens during TensorFlow import because the environment’s `google.protobuf` (v6.x) no longer provides `MessageFactory.GetPrototype`, and TensorFlow still expects it to exist while importing. The current monkey-patch only adds `GetPrototype` if `GetMessageClass` exists, but in protobuf 6 the `MessageFactory` instance may have neither method, so the patch doesn’t apply and the later call to `GetPrototype` fails. We need to ensure `MessageFactory.GetPrototype` always exists (at least as a safe fallback) before importing TensorFlow. This is a localized compatibility fix and does not change any modeling/training logic.

Patch summary: In cell 0, strengthen the protobuf compatibility patch by unconditionally attaching a `GetPrototype` method to `google.protobuf.message_factory.MessageFactory` when missing. If `GetMessageClass` exists, delegate to it; otherwise provide a minimal stub that raises a clear error if called (but avoids crashing during import). Keep the environment variables and all other imports/logic unchanged.

Updated cells: only cell 0.

Compatibility notes for cell k+1: Cell 1 depends on `os`, `zipfile`, `cv2`, `tqdm`, `tf`, etc., which remain imported under the same names; no interfaces/variables used by later cells are changed.

Assumptions: The failure occurs during TensorFlow/protobuf import-time code paths that require the attribute `MessageFactory.GetPrototype` to exist; providing the method (even if rarely invoked) resolves the immediate crash.'
- What this solution (achieved 0.28616) has done: 'Diagnosis: Cell 8 crashes because `iaa` is `None`, which happens when `imgaug` failed to import in cell 0 (your code already captures that failure into `_imgaug_import_error`). Cell 8 unconditionally calls `iaa.Rot90(...)`, producing `AttributeError: 'NoneType' object has no attribute 'Rot90'`. The minimal safe fix is to guard cell 8 so it fails fast with a clear, actionable error when `imgaug` is unavailable, instead of an opaque `NoneType` attribute error.

Patch summary: Add a small check at the top of cell 8 that raises an `ImportError` with the original import exception message when `iaa` is `None`. This preserves the existing augmentation definitions and variable names unchanged when `imgaug` is installed, and avoids undefined variables later.

Updated cells: Only cell 8 is modified.

Compatibility notes for cell k+1: When `imgaug` is available, all variables used in cell 9 (`rotate90`, `rotate180`, `rotate270`, `hflip`, `vflip`) are created exactly as before. When `imgaug` is unavailable, execution stops with a clear error before reaching cell 9, preventing downstream `NameError`s.

Assumptions: `imgaug` is not installed in this environment (consistent with the observed `iaa is None`), and installing new packages is not part of the allowed patch; therefore the best we can do is provide a deterministic, informative failure mode.'
- What this solution (achieved 0.47461) has done: 'Cell 8 fails because `imgaug` is not installed in this environment, so the current “fail fast” `ImportError` stops the notebook. To keep the notebook runnable without changing the augmentation/training semantics, I add a minimal, deterministic fallback that defines the same augmenter objects used later (`rotate90`, `rotate180`, `rotate270`, etc.) using simple numpy/OpenCV operations with an `augment_images()` method. This preserves the interface expected by `augment_pipeline()` in cell 7 and by `pipeline` in cell 9. No other cells are changed, and when `imgaug` is available the original behavior remains.'
- What this solution (achieved 0.28616) has done: 'The crash happens because the augmentation stub’s `Rot90((1, 3))` path can return rotated images with swapped height/width (540×420 instead of 420×540), so `np.append(..., axis=0)` fails when concatenating. We can fix this locally in the failing cell by forcing every augmented batch to match the original `(H,W,1)` shape before concatenation. This keeps the same augmentation pipeline and semantics, but makes the output deterministic and stackable. No other cells need changes, and the produced `processed_train` / `processed_train_cleaned` arrays keep the expected shape for the model in cell 11.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    from google.protobuf import message_factory as _message_factory  # type: ignore

    _mf_cls = getattr(_message_factory, "MessageFactory", None)
    if _mf_cls is not None and getattr(_mf_cls, "GetPrototype", None) is None:

        def _GetPrototype(self, descriptor):  # type: ignore
            get_message_class = getattr(self, "GetMessageClass", None)
            if callable(get_message_class):
                return get_message_class(descriptor)
            raise AttributeError(
                "MessageFactory.GetPrototype is not available and cannot be emulated "
                "because MessageFactory.GetMessageClass is also missing."
            )

        try:
            setattr(_mf_cls, "GetPrototype", _GetPrototype)  # type: ignore[attr-defined]
        except Exception:
            pass
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import zipfile, cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks, utils

try:
    import imgaug as ia  # type: ignore
    import imgaug.augmenters as iaa  # type: ignore

    _imgaug_import_error = None
except Exception as e:
    ia = None
    iaa = None
    _imgaug_import_error = e

sns.set_style("darkgrid")


## === cell 1
path_zip = '../input/denoising-dirty-documents/'
path = '/kaggle/working/'

with zipfile.ZipFile(path_zip + 'train.zip', 'r') as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(path_zip + 'test.zip', 'r') as zip_ref:
    zip_ref.extractall(path)  
    
with zipfile.ZipFile(path_zip + 'train_cleaned.zip', 'r') as zip_ref:
    zip_ref.extractall(path)  
    
with zipfile.ZipFile(path_zip + 'sampleSubmission.csv.zip', 'r') as zip_ref:
    zip_ref.extractall(path)
    
train_img = sorted(os.listdir(path + '/train'))
train_cleaned_img = sorted(os.listdir(path + '/train_cleaned'))
test_img = sorted(os.listdir(path + '/test'))


## === cell 2
class config():
    IMG_SIZE = (420, 540)

imgs = [cv2.imread(path + 'train/' + f) for f in sorted(os.listdir(path + 'train/'))]
print('Median Dimensions:', np.median([len(img) for img in imgs]), np.median([len(img[0]) for img in imgs]))
del imgs


## === cell 3
def process_image(path):
    img = cv2.imread(path)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, config.IMG_SIZE[::-1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img/255.0
    img = np.reshape(img, (*config.IMG_SIZE, 1))
    
    return img


## === cell 4
train = []
train_cleaned = []
test = []

for f in sorted(os.listdir(path + 'train/')):
    train.append(process_image(path + 'train/' + f))

for f in sorted(os.listdir(path + 'train_cleaned/')):
    train_cleaned.append(process_image(path + 'train_cleaned/' + f))
    
for f in sorted(os.listdir(path + 'test/')):
    test.append(process_image(path + 'test/' + f))
    
train = np.asarray(train)
train_cleaned = np.asarray(train_cleaned)
test = np.asarray(test)


## === cell 5
train.shape, train_cleaned.shape, test.shape


## === cell 6
fig, ax = plt.subplots(4, 2, figsize=(15,25))
for i in range(4):
    ax[i][0].imshow(tf.squeeze(train[i]), cmap='gray')
    ax[i][0].set_title('Noise image: {}'.format(train_img[i]))
    
    ax[i][1].imshow(tf.squeeze(train_cleaned[i]), cmap='gray')
    ax[i][1].set_title('Denoised image: {}'.format(train_img[i]))
    
    ax[i][0].get_xaxis().set_visible(False)
    ax[i][0].get_yaxis().set_visible(False)
    ax[i][1].get_xaxis().set_visible(False)
    ax[i][1].get_yaxis().set_visible(False)


## === cell 7
def augment_pipeline(pipeline, images, seed=19):
    ia.seed(seed)
    processed_images = images.copy()
    for step in pipeline:
        temp = np.array(step.augment_images(images))
        processed_images = np.append(processed_images, temp, axis=0)
    return(processed_images)


## === cell 8

if iaa is None:
    class _IAStub:
        @staticmethod
        def seed(seed):
            np.random.seed(seed)

    ia = _IAStub()

    class _AugmenterBase:
        def augment_images(self, images):
            raise NotImplementedError

    class _Rot90(_AugmenterBase):
        def __init__(self, k):
            self.k = k

        def augment_images(self, images):
            imgs = np.asarray(images)
            out = []
            for img in imgs:
                out.append(np.rot90(img, k=int(self.k), axes=(0, 1)))
            return np.asarray(out)

    class _PerspectiveTransform(_AugmenterBase):
        def __init__(self, scale=(0.02, 0.1)):
            self.scale = scale

        def augment_images(self, images):
            return np.asarray(images)

    class _Affine(_AugmenterBase):
        def __init__(self, rotate=0):
            self.rotate = float(rotate)

        def augment_images(self, images):
            imgs = np.asarray(images)
            out = []
            for img in imgs:
                h, w = img.shape[:2]
                center = (w / 2.0, h / 2.0)
                M = cv2.getRotationMatrix2D(center, self.rotate, 1.0)
                rotated = cv2.warpAffine(
                    img,
                    M,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                if rotated.ndim == 2:
                    rotated = rotated[:, :, None]
                out.append(rotated.astype(img.dtype, copy=False))
            return np.asarray(out)

    class _Crop(_AugmenterBase):
        def __init__(self, px=(5, 32)):
            self.px = px

        def augment_images(self, images):
            imgs = np.asarray(images)
            out = []
            for img in imgs:
                p = int(self.px[0] if isinstance(self.px, (tuple, list)) else self.px)
                h, w = img.shape[:2]
                if 2 * p >= h or 2 * p >= w:
                    cropped = img
                else:
                    cropped = img[p : h - p, p : w - p, ...]
                resized = cv2.resize(cropped, (w, h))
                if resized.ndim == 2:
                    resized = resized[:, :, None]
                out.append(resized.astype(img.dtype, copy=False))
            return np.asarray(out)

    class _Fliplr(_AugmenterBase):
        def __init__(self, p=1):
            self.p = float(p)

        def augment_images(self, images):
            imgs = np.asarray(images)
            out = []
            for img in imgs:
                out.append(np.flip(img, axis=1) if self.p >= 1.0 else img)
            return np.asarray(out)

    class _Flipud(_AugmenterBase):
        def __init__(self, p=1):
            self.p = float(p)

        def augment_images(self, images):
            imgs = np.asarray(images)
            out = []
            for img in imgs:
                out.append(np.flip(img, axis=0) if self.p >= 1.0 else img)
            return np.asarray(out)

    class _GaussianBlur(_AugmenterBase):
        def __init__(self, sigma=(1, 1.5)):
            self.sigma = sigma

        def augment_images(self, images):
            imgs = np.asarray(images)
            out = []
            for img in imgs:
                blurred = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=0)
                if blurred.ndim == 2:
                    blurred = blurred[:, :, None]
                out.append(blurred.astype(img.dtype, copy=False))
            return np.asarray(out)

    class _MotionBlur(_AugmenterBase):
        def __init__(self, k=6):
            self.k = int(k)

        def augment_images(self, images):
            imgs = np.asarray(images)
            out = []
            k = max(1, self.k)
            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[k // 2, :] = 1.0 / k
            for img in imgs:
                mb = cv2.filter2D(img, -1, kernel)
                if mb.ndim == 2:
                    mb = mb[:, :, None]
                out.append(mb.astype(img.dtype, copy=False))
            return np.asarray(out)

    class _Sequential(_AugmenterBase):
        def __init__(self, augmenters):
            self.augmenters = list(augmenters)

        def augment_images(self, images):
            out = np.asarray(images)
            for aug in self.augmenters:
                out = np.asarray(aug.augment_images(out))
            return out

    class _IAAStub:
        Rot90 = _Rot90
        PerspectiveTransform = _PerspectiveTransform
        Affine = _Affine
        Crop = _Crop
        Fliplr = _Fliplr
        Flipud = _Flipud
        GaussianBlur = _GaussianBlur
        MotionBlur = _MotionBlur
        Sequential = _Sequential

    iaa = _IAAStub()

rotate90 = iaa.Rot90(1)  # rotate image 90 degrees
rotate180 = iaa.Rot90(2)  # rotate image 180 degrees
rotate270 = iaa.Rot90(3)  # rotate image 270 degrees
random_rotate = iaa.Rot90((1, 3))  # randomly rotate image from 90,180,270 degrees
perc_transform = iaa.PerspectiveTransform(
    scale=(0.02, 0.1)
)  # Skews and transform images without black bg
rotate10 = iaa.Affine(rotate=(10))  # rotate image 10 degrees
rotate10r = iaa.Affine(rotate=(-10))  # rotate image 30 degrees in reverse
crop = iaa.Crop(px=(5, 32))  # Crop between 5 to 32 pixels
hflip = iaa.Fliplr(1)  # horizontal flips for 100% of images
vflip = iaa.Flipud(1)  # vertical flips for 100% of images
gblur = iaa.GaussianBlur(
    sigma=(1, 1.5)
)  # gaussian blur images with a sigma of 1.0 to 1.5
motionblur = iaa.MotionBlur(8)  # motion blur images with a kernel size 8

seq_rp = iaa.Sequential(
    [
        iaa.Rot90((1, 3)),  # randomly rotate image from 90,180,270 degrees
        iaa.PerspectiveTransform(
            scale=(0.02, 0.1)
        ),  # Skews and transform images without black bg
    ]
)

seq_cfg = iaa.Sequential(
    [
        iaa.Crop(
            px=(5, 32)
        ),  # crop images from each side by 5 to 32px (randomly chosen)
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 1.5)),  # blur images with a sigma of 0 to 1.5
    ]
)

seq_fm = iaa.Sequential(
    [
        iaa.Flipud(1),  # vertical flips all the images
        iaa.MotionBlur(k=6),  # motion blur images with a kernel size 6
    ]
)


## === cell 9
pipeline = [
    rotate90, rotate180, rotate270, hflip, vflip
]


## === cell 10
def _ensure_batch_hwcn(batch, ref_hwcn):
    batch = np.asarray(batch)
    ref_h, ref_w, ref_c = ref_hwcn
    out = []
    for img in batch:
        img = np.asarray(img)
        if img.ndim == 2:
            img = img[:, :, None]
        if img.shape[0] != ref_h or img.shape[1] != ref_w:
            img = cv2.resize(img, (ref_w, ref_h))
            if img.ndim == 2:
                img = img[:, :, None]
        if img.shape[2] != ref_c:
            img = (
                img[:, :, :ref_c]
                if img.shape[2] > ref_c
                else np.repeat(img, ref_c, axis=2)
            )
        out.append(img.astype(batch.dtype, copy=False))
    return np.asarray(out)


processed_train = train.copy()
for step in pipeline:
    temp = np.array(step.augment_images(train))
    temp = _ensure_batch_hwcn(temp, train.shape[1:4])
    processed_train = np.append(processed_train, temp, axis=0)

processed_train_cleaned = train_cleaned.copy()
for step in pipeline:
    temp = np.array(step.augment_images(train_cleaned))
    temp = _ensure_batch_hwcn(temp, train_cleaned.shape[1:4])
    processed_train_cleaned = np.append(processed_train_cleaned, temp, axis=0)

processed_train.shape, processed_train_cleaned.shape


## === cell 11
class DenoisingAutoencoder(Model):
  def __init__(self):
    super(DenoisingAutoencoder, self).__init__()
    self.encoder = tf.keras.Sequential([
        layers.Input(shape=(*config.IMG_SIZE, 1)), 
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Dropout(0.5),
    ])

    self.decoder = tf.keras.Sequential([
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = DenoisingAutoencoder()
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])


## === cell 12
es = callbacks.EarlyStopping(
    monitor='loss', patience=30, verbose=1, restore_best_weights=True
)


history =  autoencoder.fit(
    processed_train, processed_train_cleaned, 
    shuffle=True,
    callbacks=[es], epochs=500, batch_size=24
)


## === cell 13
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).plot(ax=ax)
del history


## === cell 14
autoencoder.encoder.summary()
autoencoder.decoder.summary()


## === cell 15
decoded_imgs = autoencoder(train[:4]).numpy()
    
fig, ax = plt.subplots(4, 2, figsize=(15,25))
for i in range(4):
    ax[i][0].imshow(tf.squeeze(train_cleaned[i]), cmap='gray')
    ax[i][0].set_title('Denoised image: {}'.format(train_img[i]))
    
    ax[i][1].imshow(tf.squeeze(decoded_imgs[i]), cmap='gray')
    ax[i][1].set_title('Predicted image: {}'.format(train_img[i]))
    
    ax[i][0].get_xaxis().set_visible(False)
    ax[i][0].get_yaxis().set_visible(False)
    ax[i][1].get_xaxis().set_visible(False)
    ax[i][1].get_yaxis().set_visible(False)    

del decoded_imgs    


## === cell 16
ids = []
vals = []
for i, f in tqdm(enumerate(test_img)):
    file = path + 'test/' + f
    imgid = int(f[:-4])
    img = cv2.imread(file, 0)
    img_shape = img.shape
    decoded_img = np.squeeze(autoencoder.decoder(autoencoder.encoder(test[i:i+1]).numpy()).numpy())
    preds_reshaped = cv2.resize(decoded_img, (img_shape[1], img_shape[0]))

    for r in range(img_shape[0]):
        for c in range(img_shape[1]):
            ids.append(str(imgid)+'_'+str(r + 1)+'_'+str(c + 1))
            vals.append(preds_reshaped[r, c])

print('Length of IDs: {}'.format(len(ids)))            
pd.DataFrame({'id': ids, 'value': vals}).to_csv('submission.csv',index = False)
print('Results saved to submission.csv!')
