# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Create a model to automatically segment the stomach and intestines on MRI scans.

## Metric
Mean Dice coefficient and 3D Hausdorff distance. 

The Dice coefficient can be used to compare the pixel-wise agreement between a predicted segmentation and its corresponding ground truth. The formula is given by:

$$
\frac{2 \cdot |X \cap Y|}{|X| + |Y|}
$$

where $X$ is the predicted set of pixels and $Y$ is the ground truth. The Dice coefficient is defined to be 0 when both $X$ and $Y$ are empty. 

Hausdorff distance is a method for calculating the distance between segmentation objects A and B, by calculating the furthest point on object A from the nearest point on object B. For 3D Hausdorff, we construct 3D volumes by combining each 2D segmentation with slice depth as the Z coordinate and then find the Hausdorff distance between them. (Here the slice depth for all scans is set to 1). The expected / predicted pixel locations are normalized by image size to create a bounded 0-1 score.

The two metrics are combined, with a weight of 0.4 for the Dice metric and 0.6 for the Hausdorff distance.

## Submission Format
Use run-length encoding on the pixel values.  Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the mask should be binary, meaning the masks for all objects in an image are joined into a single large mask. A value of 0 should indicate pixels that are not masked, and a value of 1 will indicate pixels that are masked.

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The file should contain a header and have the following format:

```
id,class,predicted
1,large_bowel,1 1 5 1
1,small_bowel,1 1
1,stomach,1 1
2,large_bowel,1 5 2 17
etc.
```

## Dataset
Each case is represented by multiple sets of scan slices (each set is identified by the day the scan took place). Some cases are split by time (early days are in train, later days are in test) while some cases are split by case - the entirety of the case is in train or test. The goal is to be able to generalize to both partially and wholly unseen cases.

### Files
- train.csv - IDs and masks for all training objects.
- sample_submission.csv - a sample submission file in the correct format
- train - a folder of case/day folders, each containing slice images for a particular case on a given day.

Note that the image filenames include 4 numbers (ex. 276_276_1.63_1.63.png). These four numbers are slice width / height (integers in pixels) and width/height pixel spacing (floating points in mm). The first two defines the resolution of the slide. The last two record the physical size of each pixel.

Physical pixel thickness in superior-inferior direction is 3mm.

### Columns
- `id` - unique identifier for object
- `class` - the predicted class for the object
- `segmentation` - RLE-encoded pixels for the identified object

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (126 lines)
            sample_submission.csv (20401 lines)
            sample_submission.csv.zip (57.4 kB)
            test.csv (20401 lines)
            test.csv.zip (55.2 kB)
            test.zip (432.9 MB)
            train.csv (95089 lines)
            train.csv.zip (6.7 MB)
            train.zip (2.0 GB)
            test/
                case110/
                    case110_day12/
                        scans/
                            ... (max depth reached)
                    case110_day16/
                        scans/
                            ... (max depth reached)
                case113/
                    case113_day22/
                        scans/
                            ... (max depth reached)
                ... and 27 other folders
            train/
                case101/
                    case101_day20/
                        scans/
                            ... (max depth reached)
                    case101_day22/
                        scans/
                            ... (max depth reached)
                    case101_day26/
                        scans/
                            ... (max depth reached)
                    case101_day32/
                        scans/
                            ... (max depth reached)
                case102/
                    case102_day0/
                        scans/
                            ... (max depth reached)
                ... and 75 other folders
            uw-madison-gi-tract-image-segmentation/
                description.md (126 lines)
                sample_submission.csv (20401 lines)
                ... and 7 other files
                test/
                    case110/
                        case110_day12/
                            ... (max depth reached)
                        case110_day16/
                            ... (max depth reached)
                    case113/
                        case113_day22/
                            ... (max depth reached)
                    ... and 27 other folders
                train/
                    case101/
                        case101_day20/
                            ... (max depth reached)
                        case101_day22/
                            ... (max depth reached)
                        case101_day26/
                            ... (max depth reached)
                        case101_day32/
                            ... (max depth reached)
                    case102/
                        case102_day0/
                            ... (max depth reached)
                    ... and 75 other folders
                uw-madison-gi-tract-image-segmentation/
        input/
            description.md (126 lines)
            sample_submission.csv (20401 lines)
            sample_submission.csv.zip (57.4 kB)
            test.csv (20401 lines)
            test.csv.zip (55.2 kB)
            test.zip (432.9 MB)
            train.csv (95089 lines)
            train.csv.zip (6.7 MB)
            train.zip (2.0 GB)
            test/
                case110/
                    case110_day12/
                        scans/
                            ... (max depth reached)
                    case110_day16/
                        scans/
                            ... (max depth reached)
                case113/
                    case113_day22/
                        scans/
                            ... (max depth reached)
                ... and 27 other folders
            train/
                case101/
                    case101_day20/
                        scans/
                            ... (max depth reached)
                    case101_day22/
                        scans/
                            ... (max depth reached)
                    case101_day26/
                        scans/
                            ... (max depth reached)
                    case101_day32/
                        scans/
                            ... (max depth reached)
                case102/
                    case102_day0/
                        scans/
                            ... (max depth reached)
                ... and 75 other folders
            uw-madison-gi-tract-image-segmentation/
                description.md (126 lines)
                sample_submission.csv (20401 lines)
                ... and 7 other files
                test/
                    case110/
                        case110_day12/
                            ... (max depth reached)
                        case110_day16/
                            ... (max depth reached)
                    case113/
                        case113_day22/
                            ... (max depth reached)
                    ... and 27 other folders
                train/
                    case101/
                        case101_day20/
                            ... (max depth reached)
                        case101_day22/
                            ... (max depth reached)
                        case101_day26/
                            ... (max depth reached)
                        case101_day32/
                            ... (max depth reached)
                    case102/
                        case102_day0/
                            ... (max depth reached)
                    ... and 75 other folders
                uw-madison-gi-tract-image-segmentation/
        working/
            uw-madison-gi-tract-image-segmentation/
                description.md (126 lines)
                sample_submission.csv (20401 lines)
                ... and 7 other files
                test/
                    case110/
                        case110_day12/
                            ... (max depth reached)
                        case110_day16/
                            ... (max depth reached)
                    case113/
                        case113_day22/
                            ... (max depth reached)
                    ... and 27 other folders
                train/
                    case101/
                        case101_day20/
                            ... (max depth reached)
                        case101_day22/
                            ... (max depth reached)
                        case101_day26/
                            ... (max depth reached)
                        case101_day32/
                            ... (max depth reached)
                    case102/
                        case102_day0/
                            ... (max depth reached)
                    ... and 75 other folders
                uw-madison-gi-tract-image-segmentation/
```

-> data/sample_submission.csv has 20400 rows and 3 columns.
The columns are: id, class, predicted

-> data/test.csv has 20400 rows and 2 columns.
The columns are: id, class

-> data/train.csv has 95088 rows and 3 columns.
The columns are: id, class, segmentation

-> data/uw-madison-gi-tract-image-segmentation/sample_submission.csv has 20400 rows and 3 columns.
The columns are: id, class, predicted

-> data/uw-madison-gi-tract-image-segmentation/test.csv has 20400 rows and 2 columns.
The columns are: id, class

-> data/uw-madison-gi-tract-image-segmentation/train.csv has 95088 rows and 3 columns.
The columns are: id, class, segmentation

-> input/sample_submission.csv has 20400 rows and 3 columns.
The columns are: id, class, predicted

-> (stopped after 10 files for performance)

# 5. Target score

0.8099985307671078

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the environment-breaking Keras import/protobuf issue by removing legacy standalone `keras` imports and using `tensorflow.keras` consistently, which also resolves the downstream `pd`/`tf` NameErrors caused by the first-cell crash. I also fix model loading under Keras 3 by replacing `load_model()` on a SavedModel directory with `keras.layers.TFSMLayer(...)` and wrapping it into a Keras `Model` so `predict()` works as before (inference-only, same semantics). Next, I correct the training-data “mislabel removal” boolean logic (it currently removes nothing due to `|` instead of `&`), and fix RLE encoding to correctly handle empty masks and edge transitions so the submission is valid. Finally, I ensure the generated submission strictly matches `sample_submission.csv` ordering and writes a proper `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the environment-breaking TensorFlow/protobuf import crash by setting the safe protobuf implementation env vars before importing TensorFlow, which restores `model` creation and prevents the downstream `NameError`. Then I fix the missing SavedModel path by automatically locating the correct model directory under `../input/unet-model/` (or falling back to a no-op dummy model that outputs empty masks so the notebook always produces a valid CSV). Finally, I keep your inference loop and RLE logic intact but ensure the produced `submission.csv` exactly matches `sample_submission.csv` ordering and required columns so it is accepted by Kaggle.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf crash by switching the protobuf implementation env vars to the safer `cpp` setting (and falling back to `python` only if needed) before importing TensorFlow, which is the root cause of the current `MessageFactory.GetPrototype` error. I also ensure image normalization is correct for 16-bit PNGs (your generator currently divides by 255, which badly scales intensities and can tank predictions), while keeping the same model/inference logic. Finally, I make the SavedModel wrapping more robust by selecting the correct output tensor from `serving_default` (when multiple keys exist), and keep the submission merge/order aligned to `sample_submission.csv` so a valid `submission.csv` is always produced.'
- What this solution (achieved 0.0) has done: 'I fix the root-cause import crash by setting protobuf/TensorFlow environment variables to force the pure-Python protobuf implementation (the current `cpp` setting is what triggers the missing `_message` ImportError). Once TensorFlow imports, the downstream `tf`/`keras` NameErrors disappear and the model wrapping/inference code can run as intended. I also make the SavedModel wrapper slightly more robust by accepting both common endpoints (`serving_default` and `infer`) without changing model semantics. Finally, I ensure the submission is always written as `submission.csv` with the exact same row ordering as `sample_submission.csv`, so Kaggle accepts it and the score is no longer 0.0.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf import crash that currently prevents the notebook from running at all (the root cause of the 0.0 score). To do that with minimal behavioral impact, I force a protobuf implementation setting that avoids the `MessageFactory.GetPrototype` error before importing TensorFlow, and I add a safe fallback attempt so TensorFlow can import reliably in Kaggle’s environment. I also make the SavedModel wrapping a bit more defensive about output keys (without changing inference semantics), and ensure the submission strictly matches `sample_submission.csv` ordering and required columns and is always written as `submission.csv`. No changes are made to the model architecture/training approach (this script is inference-only), only runtime/stability fixes so you get a valid non-empty submission.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf import crash that currently stops the notebook in the first cell (root cause of the 0.0 score) by forcing the pure-Python protobuf implementation *before* TensorFlow is ever imported, and by avoiding the unsafe fallback to `cpp` that triggers the `MessageFactory.GetPrototype` error in this environment. I also add a safe inference wrapper so the model output is always converted to a NumPy array with shape `(B, 128, 128, 3)` (handling dict outputs from SavedModels) without changing your model/inference logic. Finally, I make the submission merge strictly align to `sample_submission.csv` order and ensure `submission.csv` is always written with the required columns and valid RLE strings.'
- What this solution (achieved 0.0) has done: 'I fix the root-cause TensorFlow/protobuf crash by forcing a compatible protobuf version/implementation **before** importing TensorFlow, and by adding a safe fallback that downgrades `protobuf` in-notebook if the `MessageFactory.GetPrototype` error still occurs (this is what currently prevents any submission and yields 0.0). I keep your model/inference logic the same, but make the DataFrame preprocessing more robust to path parsing (px spacing slice) and ensure the final submission is **exactly** in `sample_submission.csv` row order to avoid any hidden misalignment issues. I also add a small guard so the RLE encoding always receives a 2D mask (avoids edge-case shape bugs), without changing thresholding or model outputs. These changes are execution/stability focused and should move the score from 0.0 to a valid non-zero score by enabling the model to run and produce a correct CSV.'
- What this solution (achieved 0.0) has done: 'I fix the runtime crash that prevents TensorFlow from importing (root cause of the 0.0 score) by applying the protobuf-compat workaround earlier and more reliably, and by adding a safe fallback that uses the pure-Python protobuf implementation without attempting a pip downgrade (pip installs are slow/unreliable in offline Kaggle). I also make the SavedModel wrapping tolerant to different output key names so inference always returns a `(B,128,128,3)`-like array for your existing post-processing. Finally, I ensure the submission is aligned exactly to `sample_submission.csv` row order and that RLE encoding is strictly competition-compliant (empty masks become empty strings), without changing your model, threshold, or inference loop semantics.'
- What this solution (achieved 0.0) has done: 'We fix the root-cause TensorFlow/protobuf crash by setting a broader set of protobuf-related environment variables *before* TensorFlow import and by making the TF import retry once with a safer configuration if the first attempt fails. We also make the submission building strictly align to `sample_submission.csv` row order (no accidental reordering from merges) and ensure RLE encoding is competition-compliant for empty masks. These changes are execution/stability focused (so you get a non-zero valid score instead of 0.0) and do not alter the core model/inference logic beyond making it actually run in the Kaggle runtime.'
- What this solution (achieved 0.0) has done: 'I fix the root-cause TensorFlow/protobuf import crash that currently stops execution (and yields a 0.0 score) by forcing a protobuf implementation that’s compatible with Kaggle’s TF build *before* TensorFlow is imported, and by retrying the import once with a different safe setting if the first attempt fails. I also correct RLE encoding to use the competition’s required column-major (“Fortran”) flattening order and 1-indexed runs, which is necessary for the submission to be evaluated correctly (otherwise you can get effectively zero score even with reasonable masks). Finally, I keep your model/inference loop intact but make the SavedModel wrapping output selection and prediction tensor shaping more defensive so it always produces a `(B,128,128,3)`-like array, and I ensure the saved `submission.csv` matches `sample_submission.csv` row ordering.'
- What this solution (achieved 0.0) has done: 'I fix the root-cause TensorFlow import crash (which currently stops the notebook before any predictions are made, yielding a 0.0 score) by forcing the pure-Python protobuf implementation **before** TensorFlow is imported and by adding a safe single retry with a different implementation if needed. I keep your model/inference logic intact, but make the TF SavedModel wrapping robust to different output signatures so `predict()` always returns a usable `(B,H,W,C)` tensor. Finally, I keep your existing RLE/merge logic but ensure the script always completes and writes a valid `submission.csv` aligned exactly to `sample_submission.csv` order.'
- What this solution (achieved 0.0) has done: 'I fix the root-cause TensorFlow/protobuf import crash that currently prevents the notebook from running (and thus yields a 0.0 score) by retrying the TF import with the legacy pure-Python protobuf path and, if needed, falling back to a safe inference-only path that uses a SavedModel loaded via `tf.saved_model.load` (no Keras/TFSM layer dependency). I also make the SavedModel wrapping more defensive about output keys and always normalize predictions into a `(B,128,128,3)` NumPy array so downstream resizing/RLE works reliably. Finally, I keep your existing preprocessing/inference/RLE and submission ordering semantics intact, ensuring `submission.csv` is always written in exact `sample_submission.csv` order with valid (possibly empty) RLE strings.'
- What this solution (achieved 0.0) has done: 'I fix the root-cause TensorFlow/protobuf import crash that currently stops the notebook (and causes a 0.0 score) by applying a safer TF import routine that forces the pure-Python protobuf backend and clears conflicting TF/protobuf modules before retrying. I also make the SavedModel inference wrapper robust by converting dict/list outputs into a `(B,H,W,C)` NumPy array without changing your thresholding or inference loop. Finally, I keep your existing preprocessing/inference/RLE logic but ensure the submission is always written as `submission.csv` in the exact `sample_submission.csv` row order with valid empty-string RLEs for empty masks.'
- What this solution (achieved 0.0) has done: 'We fix the root-cause TensorFlow/protobuf import crash (the reason you currently get 0.0) by setting additional safe environment flags *before* any TensorFlow import and by using a robust import routine that retries with the pure-Python protobuf backend (without changing your model/inference semantics). Next, we fix a real RLE decoding shape bug (`rle_decode` currently mis-handles `(H,W,1)` by treating it as `(H,W,D)`), which can silently corrupt masks if used and can also break training-style code paths. Finally, we keep your inference loop and thresholding intact, but ensure the SavedModel callable fallback returns a consistent tensor (extracting the first output tensor when a signature returns a dict) so downstream resizing/RLE always runs and the submission is written correctly as `submission.csv` in `sample_submission.csv` order.'
- What this solution (achieved 0.0) has done: 'I fix the root-cause TensorFlow/protobuf import crash (the `MessageFactory.GetPrototype` error) by forcing the pure-Python protobuf backend early and importing TensorFlow in a clean subprocess-like manner (clearing conflicting protobuf/TF modules before retry). This is execution-critical: once TensorFlow loads, the existing inference pipeline can run and produce a non-empty, valid submission instead of scoring 0.0. I also make the SavedModel inference wrapper more defensive about output signatures/keys without changing thresholding or post-processing semantics, and ensure the RLE encoding receives a 2D mask and the final `submission.csv` matches `sample_submission.csv` row order exactly.'

# 9. Code solution

## === cell 0
import os
import random
import warnings
from glob import glob
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # keep CPU-only unless user overrides
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices=false")

import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def _import_tensorflow_with_retries():
    """
    Import tensorflow while avoiding protobuf descriptor crashes in Kaggle.

    Root issue: some Kaggle TF builds + protobuf C++ backend can crash with:
      AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
    Fix: force pure-Python protobuf backend and clear preloaded TF/protobuf modules.
    """
    import importlib
    import sys

    last = None
    for attempt in (1, 2):
        try:
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"

            for m in list(sys.modules.keys()):
                if (
                    m == "tensorflow"
                    or m.startswith("tensorflow.")
                    or m.startswith("google.protobuf")
                ):
                    del sys.modules[m]

            tf_mod = importlib.import_module("tensorflow")
            return tf_mod
        except Exception as e:
            last = e
            continue

    raise RuntimeError(
        "TensorFlow failed to import with the pure-Python protobuf backend."
    ) from last


tf = _import_tensorflow_with_retries()

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy

tf.random.set_seed(SEED)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def rle_decode(mask_rle, shape, color=1):
    """Decode RLE string to mask array with given shape."""
    if (
        mask_rle is None
        or mask_rle == ""
        or (isinstance(mask_rle, float) and np.isnan(mask_rle))
    ):
        return np.zeros(shape, dtype=np.float32)

    s = np.array(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths

    if len(shape) == 3 and shape[-1] == 1:
        h, w, _ = shape
        img = np.zeros((h * w,), dtype=np.float32)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = color
        return img.reshape((h, w, 1))

    if len(shape) == 3:
        h, w, d = shape
        img = np.zeros((h * w, d), dtype=np.float32)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = color
        return img.reshape(shape)

    h, w = shape
    img = np.zeros((h * w,), dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = color
    return img.reshape(shape)


def rle_encode(arr):
    """
    Competition RLE expects pixels numbered top-to-bottom then left-to-right,
    which corresponds to flattening in Fortran order for a (H,W) array.
    Also requires 1-indexed starts and empty mask => ''.
    """
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    arr = (arr > 0).astype(np.uint8)

    if arr.sum() == 0:
        return ""

    pixels = arr.flatten(order="F")
    pads = np.concatenate([[0], pixels, [0]])
    changes = np.where(pads[1:] != pads[:-1])[0] + 1
    runs = changes[::2]
    ends = changes[1::2]
    lengths = ends - runs
    out = np.column_stack([runs, lengths]).reshape(-1)
    return " ".join(map(str, out.tolist()))


def open_gray16(_path, normalize=True, to_rgb=False):
    """Helper to open 16-bit png scans."""
    img = cv2.imread(_path, cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {_path}")
    if normalize:
        img = img.astype(np.float32) / 65535.0
    if to_rgb:
        img = np.tile(np.expand_dims(img, axis=-1), 3)
    return img




## === cell 2
pass




## === cell 3
def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def mean_iou(y_true, y_pred):
    yt0 = y_true[:, :, :, 0]
    yp0 = K.cast(y_pred[:, :, :, 0] > 0.5, "float32")
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1.0, tf.cast(inter / union, "float32"))
    return iou


def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2.0 * K.sum(intersection) + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )
    return 1.0 - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(tf.cast(y_true, tf.float32), y_pred) + 0.5 * dice_loss(
        tf.cast(y_true, tf.float32), y_pred
    )




## === cell 4
pass



## === cell 5
custom_objects = {
    "dice_coef": dice_coef,
    "iou_coef": iou_coef,
    "bce_dice_loss": bce_dice_loss,
}


def find_savedmodel_dir(root="../input/unet-model"):
    root = Path(root)
    if not root.exists():
        return None
    candidates = []
    for p in root.rglob("*"):
        if p.is_dir() and (
            (p / "saved_model.pb").exists() or (p / "saved_model.pbtxt").exists()
        ):
            candidates.append(p)
    if not candidates:
        return None
    candidates = sorted(
        candidates, key=lambda x: (0 if "model_32" in str(x) else 1, len(str(x)))
    )
    return str(candidates[0])


def build_tfsm_infer_model(savedmodel_dir, input_shape=(128, 128, 3)):
    """
    Wrap a TF SavedModel for inference under Keras using TFSMLayer (if available).
    """
    inp = keras.Input(shape=input_shape, name="image")
    last_err = None
    for endpoint in ("serving_default", "infer"):
        try:
            tfsm = keras.layers.TFSMLayer(savedmodel_dir, call_endpoint=endpoint)
            out = tfsm(inp)

            if isinstance(out, dict):
                preferred = [
                    k for k in ("output_0", "outputs", "predictions") if k in out
                ]
                key = preferred[0] if preferred else sorted(out.keys())[0]
                out = out[key]

            if isinstance(out, (list, tuple)):
                out = out[0]

            return keras.Model(inputs=inp, outputs=out, name=f"unet_infer_{endpoint}")
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Could not wrap SavedModel with TFSMLayer. Last error: {last_err}"
    )


class SavedModelCallable:
    """
    Fallback inference wrapper that does NOT rely on Keras TFSMLayer.
    Uses tf.saved_model.load and a concrete signature to run predictions.
    """

    def __init__(self, savedmodel_dir):
        self.savedmodel_dir = savedmodel_dir
        self.sm = tf.saved_model.load(savedmodel_dir)

        sigs = getattr(self.sm, "signatures", {})
        if not sigs:
            raise RuntimeError("SavedModel has no signatures; cannot run inference.")
        self.signature_name = (
            "serving_default" if "serving_default" in sigs else list(sigs.keys())[0]
        )
        self.fn = sigs[self.signature_name]

    def __call__(self, x):
        out = self.fn(x)
        if isinstance(out, dict):
            preferred = [k for k in ("output_0", "outputs", "predictions") if k in out]
            key = preferred[0] if preferred else sorted(out.keys())[0]
            out = out[key]
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out

    def predict(self, X, verbose=0):
        x = tf.convert_to_tensor(X, dtype=tf.float32)
        return self.__call__(x)


MODEL_DIR = find_savedmodel_dir("../input/unet-model")

model = None
if MODEL_DIR is not None:
    try:
        model = build_tfsm_infer_model(MODEL_DIR, input_shape=(128, 128, 3))
        print("Loaded SavedModel via TFSMLayer from:", MODEL_DIR)
    except Exception as e:
        print(
            "TFSMLayer wrap failed, falling back to tf.saved_model.load. Error:",
            repr(e),
        )
        try:
            model = SavedModelCallable(MODEL_DIR)
            print("Loaded SavedModel via tf.saved_model.load from:", MODEL_DIR)
        except Exception as e2:
            print("tf.saved_model.load fallback failed. Error:", repr(e2))
            model = None

if model is None:
    inp = keras.Input(shape=(128, 128, 3), name="image")
    out = keras.layers.Lambda(
        lambda x: tf.zeros((tf.shape(x)[0], 128, 128, 3), tf.float32)
    )(inp)
    model = keras.Model(inp, out, name="dummy_empty_mask_model")
    print("WARNING: Could not load model. Using dummy model (empty masks).")


def _normalize_model_output_to_numpy(y):
    """
    Normalize possible SavedModel/Keras outputs to numpy array (B,H,W,C).
    Handles dict/list outputs and ensures float array.
    """
    if isinstance(y, dict):
        preferred = [k for k in ("output_0", "outputs", "predictions") if k in y]
        key = preferred[0] if preferred else sorted(y.keys())[0]
        y = y[key]
    if isinstance(y, (list, tuple)):
        y = y[0]
    if tf.is_tensor(y):
        y = y.numpy()
    y = np.asarray(y)
    return y


def predict_batch(m, X):
    """Normalize possible model outputs to a numpy array with shape (B,H,W,C)."""
    y = m.predict(X, verbose=0) if hasattr(m, "predict") else m(X)
    y = _normalize_model_output_to_numpy(y)
    return y




## === cell 6
pass



## === cell 7
df1 = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/train.csv")
print(df1.head())



## === cell 8
test_df = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)

if len(test_df) == 0:
    DEBUG = True
    test_df = df1.iloc[:300, :].copy()
    test_df = test_df[["id", "class"]]
    test_df["predicted"] = ""
else:
    DEBUG = False

submission1 = test_df.copy()
print(test_df.head())



## === cell 9
submission1.head()




## === cell 10
def preprocessing(df, subset="train"):
    df = df.copy()
    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
    df["slice"] = df["id"].apply(lambda x: x.split("_")[3])

    if (subset == "train") or (DEBUG):
        DIR = "../input/uw-madison-gi-tract-image-segmentation/train"
    else:
        DIR = "../input/uw-madison-gi-tract-image-segmentation/test"

    all_images = glob(os.path.join(DIR, "**", "*.png"), recursive=True)
    if len(all_images) == 0:
        raise FileNotFoundError(f"No png images found under: {DIR}")

    x = all_images[0].rsplit("/", 4)[0]

    path_partial_list = []
    for i in range(0, df.shape[0]):
        path_partial_list.append(
            os.path.join(
                x,
                "case" + str(df["case"].values[i]),
                "case"
                + str(df["case"].values[i])
                + "_"
                + "day"
                + str(df["day"].values[i]),
                "scans",
                "slice_" + str(df["slice"].values[i]),
            )
        )
    df["path_partial"] = path_partial_list

    path_partial_list = []
    for i in range(0, len(all_images)):
        path_partial_list.append(str(all_images[i].rsplit("_", 4)[0]))

    tmp_df = pd.DataFrame()
    tmp_df["path_partial"] = path_partial_list
    tmp_df["path"] = all_images

    df = df.merge(tmp_df, on="path_partial").drop(columns=["path_partial"])
    df["filename"] = df["path"].apply(lambda x: x.split("/")[-1])
    df["unique_filename"] = df.apply(
        lambda row: str(row.case) + "_" + str(row.day) + "_" + str(row.filename), axis=1
    )

    parts = df["path"].apply(lambda x: x[:-4].rsplit("_", 4))
    df["width"] = parts.apply(lambda p: int(p[1]))
    df["height"] = parts.apply(lambda p: int(p[2]))
    df["px_spacing_h"] = parts.apply(lambda p: float(p[3]))
    df["px_spacing_w"] = parts.apply(lambda p: float(p[4]))

    return df




## === cell 11
test_df = preprocessing(test_df, subset="test")
print(test_df.shape)
test_df.head()



## === cell 12
train_df = preprocessing(df1, subset="train")
train_df.head()



## === cell 13
train_df = train_df[(train_df["case"] != 7) & (train_df["case"] != 0)].reset_index(
    drop=True
)
train_df = train_df[(train_df["case"] != 81) & (train_df["case"] != 30)].reset_index(
    drop=True
)




## === cell 14
def segment(df, subset="train"):
    df_out = pd.DataFrame({"id": df["id"][::3].values})

    if subset == "train":
        df_out["large_bowel"] = df["segmentation"][::3].values
        df_out["small_bowel"] = df["segmentation"][1::3].values
        df_out["stomach"] = df["segmentation"][2::3].values

    df_out["path"] = df["path"][::3].values
    df_out["case"] = df["case"][::3].values
    df_out["day"] = df["day"][::3].values
    df_out["slice"] = df["slice"][::3].values
    df_out["width"] = df["width"][::3].values
    df_out["height"] = df["height"][::3].values
    df_out["px_spacing_h"] = df["px_spacing_h"][::3].values
    df_out["px_spacing_w"] = df["px_spacing_w"][::3].values
    df_out["filename"] = df["filename"][::3].values
    df_out["unique_filename"] = df["unique_filename"][::3].values

    df_out.reset_index(inplace=True, drop=True)
    df_out.fillna("", inplace=True)
    if subset == "train":
        df_out["count"] = np.sum(df_out.iloc[:, 1:4] != "", axis=1).values

    return df_out


train_df = segment(train_df, subset="train")
test_df_seg = segment(test_df, subset="test")



## === cell 15
train_df.sample(5)



## === cell 16
pass



## === cell 17
BATCH_SIZE = 32
im_height = 128
im_width = 128




## === cell 18
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=BATCH_SIZE, subset="train", shuffle=False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        cur_bs = len(batch_indexes)

        X = np.empty((cur_bs, im_height, im_width, 3), dtype=np.float32)
        if self.subset == "train":
            y = np.empty((cur_bs, im_height, im_width, 3), dtype=np.float32)

        ids = list(self.df["id"].iloc[batch_indexes].values)

        for i, idx in enumerate(batch_indexes):
            img_path = self.df["path"].iloc[idx]
            w = int(self.df["width"].iloc[idx])
            h = int(self.df["height"].iloc[idx])

            img = self.__load_grayscale(img_path)  # (128,128,3)
            X[i] = img

            if self.subset == "train":
                for k, j in enumerate(["large_bowel", "small_bowel", "stomach"]):
                    rles = self.df[j].iloc[idx]
                    mask = rle_decode(rles, shape=(h, w, 1))
                    mask = cv2.resize(
                        mask, (im_width, im_height), interpolation=cv2.INTER_NEAREST
                    )
                    y[i, :, :, k] = mask[:, :, 0] if mask.ndim == 3 else mask

        if self.subset == "train":
            return X, y
        else:
            return X, ids

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)
        if img.max() > 255.0:
            img = img / 65535.0
        else:
            img = img / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.tile(img, (1, 1, 3))
        return img




## === cell 19
pass



## === cell 20
val_generator = DataGenerator(
    test_df_seg, batch_size=BATCH_SIZE, subset="test", shuffle=False
)



## === cell 21
print(len(val_generator))
print(BATCH_SIZE)



## === cell 22
_, idsss = val_generator[1]
print(idsss[:5])



## === cell 23
pass



## === cell 24
ids = []
classs = []
predics = []
seen = set()

id_to_wh = dict(
    zip(
        test_df_seg["id"].values,
        zip(test_df_seg["width"].values, test_df_seg["height"].values),
    )
)

num_batches = len(val_generator)
for i in range(num_batches):
    X, batch_ids = val_generator[i]
    preds = predict_batch(model, X)

    if preds.ndim == 3:
        preds = preds[..., None]
    if preds.shape[-1] == 1:
        preds = np.repeat(preds, 3, axis=-1)

    for j in range(len(batch_ids)):
        img_id = batch_ids[j]
        if img_id in seen:
            continue
        seen.add(img_id)

        w, h = id_to_wh[img_id]
        w = int(w)
        h = int(h)

        ids.extend((img_id, img_id, img_id))
        classs.extend(("large_bowel", "small_bowel", "stomach"))

        for k in range(3):
            pred_img = cv2.resize(
                preds[j, :, :, k], (w, h), interpolation=cv2.INTER_NEAREST
            )
            pred_img = (pred_img > 0.5).astype(np.uint8)
            predics.append(rle_encode(pred_img))



## === cell 25
sub_pred = pd.DataFrame({"id": ids, "class": classs, "predicted": predics})
print(sub_pred.shape)

key_to_pred = dict(
    zip(
        zip(sub_pred["id"].values, sub_pred["class"].values),
        sub_pred["predicted"].values,
    )
)

submission_df = submission1[["id", "class"]].copy()
submission_df["predicted"] = [
    key_to_pred.get((i, c), "")
    for i, c in zip(submission_df["id"].values, submission_df["class"].values)
]
submission_df["predicted"] = submission_df["predicted"].fillna("").astype(str)

submission_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_df.shape)



## === cell 26
submission_df.head(10)



## === cell 27
submission_df.sample(10)
