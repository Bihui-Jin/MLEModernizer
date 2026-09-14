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

0.02953

# 6. Current score

0.28613

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28616) has done: 'Diagnosis: The crash happens immediately in cell 0 during imports, before any data is loaded. The traceback (`'MessageFactory' object has no attribute 'GetPrototype'`) is a known incompatibility between `imgaug` (via its protobuf usage) and the installed `protobuf==6.33.0`. Since `imgaug` is only imported in cell 0, the minimal fix is to avoid importing `imgaug` so the notebook can run with the current environment. This preserves all other logic and prevents the import-time protobuf crash.

Patch summary: Remove the `imgaug` imports in cell 0 and keep the rest of the imports unchanged so downstream cells can execute. This is the smallest localized change that eliminates the protobuf-related AttributeError.

Updated cells: Only cell 0 is modified.

Compatibility notes for cell k+1: Cell 1 does not reference `ia`, `iaa`, or `imgaug`, so removing these imports does not affect cell 1’s variables or behavior.

Assumptions: This notebook can run without `imgaug` at least up to and including cell 1, and any later augmentation code either is not required for execution in this environment or can be handled elsewhere (not modified here per constraints).'
- What this solution (achieved 0.28616) has done: 'The crash happens during the TensorFlow import in cell 0, before any notebook logic runs. With TensorFlow 2.18.0 and protobuf 6.33.0, this `MessageFactory.GetPrototype` AttributeError is a known incompatibility caused by TensorFlow/TFDS expecting an older protobuf runtime API. The minimal deterministic fix is to pin protobuf to a compatible 4.x version at runtime (before importing TensorFlow), then restart-import TensorFlow cleanly within the cell. This keeps the model/training logic unchanged and only adjusts the environment dependency that triggers the import-time crash.'
- What this solution (achieved 0.28607) has done: 'The timeout is dominated by (1) reading/resizing images one-by-one in Python loops, (2) training a fairly large ConvNet on full-resolution images without using a performant `tf.data` pipeline, and (3) the submission creation step doing ~5.8M pixel ids via nested Python loops and repeated model calls. I keep the exact same model/loss/training semantics, but speed up the data input with parallelized + preallocated loading, switch training to a cached/prefetched `tf.data.Dataset` (same epochs/batch size/shuffle behavior), and vectorize submission generation by batching predictions and building the `id`/`value` columns without Python per-pixel loops. I also remove redundant `.numpy()` roundtrips and call `autoencoder` directly for inference to reduce overhead while producing identical outputs up to negligible float diffs. All paths and core logic (architecture, loss, training loop/epochs) remain unchanged.'
- What this solution (achieved 0.28607) has done: 'Diagnosis: The crash happens during the TensorFlow import in cell 0, and the traceback indicates an incompatibility between the installed `protobuf==6.33.0` and TensorFlow’s expected protobuf runtime, leading to `MessageFactory.GetPrototype` missing. The current code tries to fix protobuf *after* importing TensorFlow, which is too late because the failing import already occurred. The minimal fix is to ensure a compatible protobuf version is installed and loaded *before* importing TensorFlow.

Patch summary: Move the protobuf compatibility check/installation to occur before `import tensorflow as tf`, and force a runtime restart of the protobuf module import by importing it only after the install step. Keep all other logic (seeds, determinism, threading, imports) unchanged.

Updated cells: cell 0 only.

Compatibility notes for cell k+1: All variables and imports used by cell 1 (`os`, `sys`, `zipfile`, `subprocess`, etc.) remain available exactly as before; only the ordering of the protobuf/TensorFlow import is adjusted to prevent the crash.

Assumptions: The environment allows `pip install` during execution (as the original code already attempted), and TensorFlow 2.18 is compatible with `protobuf<5` in this environment.'
- What this solution (achieved 0.28613) has done: 'Your score is far from the target (0.286 vs 0.0295; lower is better), so we should make a small but meaningful improvement without changing the model/training core. The biggest accuracy issue here is that you resize all training images to a fixed 420×540, which distorts many documents; learning on distorted targets hurts RMSE heavily. I keep the same autoencoder architecture and training loop, but change preprocessing to preserve aspect ratio via padding (“letterbox”) so inputs/targets remain geometrically consistent, and use the same padding-aware resize when preparing test images and unpadding predictions back to original size before submission. This is a minimal semantic change (still grayscale normalization + same model/loss) but typically yields a large RMSE improvement toward the target.'
- What this solution (achieved 0.28612) has done: 'Diagnosis: The crash happens when indexing `flat[pi]` because `pi` comes from the submission template and refers to a pixel index in the original image’s flattened size, while `pred_orig.reshape(-1)` is currently sized to the *actual* (h,w) of each test image after unletterboxing. For at least one image, the template expects 227,100 pixels but the current flattened prediction only has 226,800 pixels, so `pi` can exceed bounds. This mismatch is due to differing original image dimensions vs. the fixed dimensions implied by the competition’s id/pixel indexing scheme. The minimal fix is to build `preds_flat_by_id` as a flat array of a consistent, template-compatible length by resizing each prediction to the target height/width inferred from the maximum `pix_idx` per image id in `sampleSubmission.csv`.

Patch summary: In cell 14 only, infer the expected flat length per image from the submission ids (`max(pix_idx)+1` per image), derive expected (h,w) assuming row-major flattening with a fixed width (use the modal width across images, falling back safely), and resize each predicted image to that (h,w) before flattening. Add a defensive bounds check to raise a clear error if an unexpected index still occurs. No changes to model, training, or earlier preprocessing.

Updated cells / Compatibility notes for cell k+1 / Assumptions:
- Updated cell 14 below; there is no cell 15 shown, but outputs (`submission.csv` written, `submission` DataFrame) remain identical in interface.
- Assumes the sample submission encodes pixels in row-major order and that a single common width is used across test images (typical for this competition); we infer it from the data rather than hardcoding.'
- What this solution (achieved 0.28613) has done: 'Your score is far from the target (0.28612 vs 0.02953; lower is better), so we should make a small change that improves correctness rather than tuning the model. The biggest likely issue is submission pixel indexing: the competition’s `id` uses 1-based `(row, col)` coordinates, but the current code treats the suffix as a 0-based flat index, which badly scramble pixels and inflate RMSE. I minimally fix cell 14 to parse `id` into `(img, row, col)`, infer each image’s expected `(H, W)` from the sample submission, resize each predicted image to that exact shape, and then pick the correct pixel via `(row-1, col-1)`. This preserves your model, preprocessing, and training, but should move the score strongly toward the target by aligning predictions with the evaluation format.'

# 9. Code solution

## === cell 0
import os, sys, zipfile, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm.auto import tqdm

try:
    import google.protobuf  # noqa: F401
    from packaging.version import parse as _vparse
    import google.protobuf as _gp

    if _vparse(_gp.__version__) >= _vparse("5.0.0"):
        raise RuntimeError("Incompatible protobuf version for this TF build")
except Exception:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf<5,>=4.21.12"]
    )
    if "google.protobuf" in sys.modules:
        del sys.modules["google.protobuf"]

import tensorflow as tf

tf.keras.utils.set_random_seed(19)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks

sns.set_style("darkgrid")

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass



## === cell 1
path_zip = "../input/denoising-dirty-documents/"
path = "/kaggle/working/"


def _extract_if_missing(zip_path, out_dir, marker_dir):
    if not os.path.exists(os.path.join(out_dir, marker_dir)):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)


_extract_if_missing(path_zip + "train.zip", path, "train")
_extract_if_missing(path_zip + "test.zip", path, "test")
_extract_if_missing(path_zip + "train_cleaned.zip", path, "train_cleaned")
if not os.path.exists(os.path.join(path, "sampleSubmission.csv")):
    _extract_if_missing(path_zip + "sampleSubmission.csv.zip", path, "")

train_img = sorted(os.listdir(path + "/train"))
train_cleaned_img = sorted(os.listdir(path + "/train_cleaned"))
test_img = sorted(os.listdir(path + "/test"))




## === cell 2
class config:
    IMG_SIZE = (420, 540)


_sample_files = train_img[:10]
imgs = [cv2.imread(path + "train/" + f) for f in _sample_files]
print(
    "Median Dimensions:",
    np.median([len(img) for img in imgs]),
    np.median([len(img[0]) for img in imgs]),
)
del imgs, _sample_files




## === cell 3
def _to_gray_float(img_bgr):
    img = np.asarray(img_bgr, dtype="float32")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    return img


def _letterbox_gray(gray, target_hw):
    th, tw = target_hw
    h, w = gray.shape[:2]
    scale = min(tw / w, th / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_left = (tw - new_w) // 2
    pad_right = tw - new_w - pad_left
    pad_top = (th - new_h) // 2
    pad_bottom = th - new_h - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=1.0,  # white background
    ).astype(np.float32, copy=False)

    meta = {
        "orig_hw": (h, w),
        "new_hw": (new_h, new_w),
        "pad": (pad_top, pad_bottom, pad_left, pad_right),
        "scale": scale,
    }
    return padded, meta


def _unletterbox_to_orig(pred_gray, meta):
    pad_top, pad_bottom, pad_left, pad_right = meta["pad"]
    new_h, new_w = meta["new_hw"]
    h, w = meta["orig_hw"]

    cropped = pred_gray[
        pad_top : pad_top + new_h,
        pad_left : pad_left + new_w,
    ]
    out = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR).astype(
        np.float32, copy=False
    )
    return out


def process_image(path_):
    img = cv2.imread(path_)
    gray = _to_gray_float(img)
    padded, _ = _letterbox_gray(gray, config.IMG_SIZE)
    img = np.reshape(padded, (*config.IMG_SIZE, 1))
    return img




## === cell 4
def _load_folder_to_array(folder, files):
    n = len(files)
    arr = np.empty((n, config.IMG_SIZE[0], config.IMG_SIZE[1], 1), dtype=np.float32)
    for i, f in enumerate(files):
        arr[i] = process_image(folder + f)
    return arr


train = _load_folder_to_array(path + "train/", sorted(os.listdir(path + "train/")))
train_cleaned = _load_folder_to_array(
    path + "train_cleaned/", sorted(os.listdir(path + "train_cleaned/"))
)
test = _load_folder_to_array(path + "test/", sorted(os.listdir(path + "test/")))

train.shape, train_cleaned.shape, test.shape



## === cell 5
SHOW_PLOTS = False

if SHOW_PLOTS:
    fig, ax = plt.subplots(4, 2, figsize=(15, 25))
    for i in range(4):
        ax[i][0].imshow(tf.squeeze(train[i]), cmap="gray")
        ax[i][0].set_title("Noise image: {}".format(train_img[i]))

        ax[i][1].imshow(tf.squeeze(train_cleaned[i]), cmap="gray")
        ax[i][1].set_title("Denoised image: {}".format(train_img[i]))

        ax[i][0].get_xaxis().set_visible(False)
        ax[i][0].get_yaxis().set_visible(False)
        ax[i][1].get_xaxis().set_visible(False)
        ax[i][1].get_yaxis().set_visible(False)




## === cell 6
def augment_pipeline(pipeline, images, seed=19):
    return images




## === cell 7
rotate90 = rotate180 = rotate270 = random_rotate = perc_transform = None
rotate10 = rotate10r = crop = hflip = vflip = gblur = motionblur = None
seq_rp = seq_cfg = seq_fm = None



## === cell 8
pipeline = []




## === cell 9
class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(*config.IMG_SIZE, 1)),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Dropout(0.5),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.UpSampling2D((2, 2)),
                layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same"),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = DenoisingAutoencoder()
autoencoder.compile(
    optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"]
)



## === cell 10
es = callbacks.EarlyStopping(
    monitor="loss", patience=30, verbose=1, restore_best_weights=True
)

BATCH_SIZE = 24
train_ds = tf.data.Dataset.from_tensor_slices((train, train_cleaned))
train_ds = train_ds.shuffle(
    buffer_size=train.shape[0], seed=19, reshuffle_each_iteration=True
)
train_ds = (
    train_ds.batch(BATCH_SIZE, drop_remainder=False).cache().prefetch(tf.data.AUTOTUNE)
)

history = autoencoder.fit(train_ds, callbacks=[es], epochs=500)



## === cell 11
if SHOW_PLOTS:
    fig, ax = plt.subplots(figsize=(20, 6))
    pd.DataFrame(history.history).iloc[:, :-1].plot(ax=ax)
del history



## === cell 12
autoencoder.encoder.summary()
autoencoder.decoder.summary()



## === cell 13
if SHOW_PLOTS:
    decoded_imgs = autoencoder(train[:4], training=False).numpy()

    fig, ax = plt.subplots(4, 2, figsize=(15, 25))
    for i in range(4):
        ax[i][0].imshow(tf.squeeze(train_cleaned[i]), cmap="gray")
        ax[i][0].set_title("Denoised image: {}".format(train_img[i]))

        ax[i][1].imshow(tf.squeeze(decoded_imgs[i]), cmap="gray")
        ax[i][1].set_title("Predicted image: {}".format(train_img[i]))

        ax[i][0].get_xaxis().set_visible(False)
        ax[i][0].get_yaxis().set_visible(False)
        ax[i][1].get_xaxis().set_visible(False)
        ax[i][1].get_yaxis().set_visible(False)

    del decoded_imgs



## === cell 14
test_batch = 4  # keep same batching intent

preds = autoencoder.predict(test, batch_size=test_batch, verbose=0)  # (N, 420, 540, 1)
preds = np.clip(preds, 0.0, 1.0).astype(np.float32, copy=False)

sub_path = os.path.join(path, "sampleSubmission.csv")
if not os.path.exists(sub_path):
    for _cand in (
        "../input/denoising-dirty-documents/sampleSubmission.csv",
        "../input/sampleSubmission.csv",
        "/kaggle/input/denoising-dirty-documents/sampleSubmission.csv",
        "/kaggle/input/sampleSubmission.csv",
        "/kaggle/data/denoising-dirty-documents/sampleSubmission.csv",
        "/kaggle/data/sampleSubmission.csv",
    ):
        if os.path.exists(_cand):
            sub_path = _cand
            break

sample = pd.read_csv(sub_path)

parts = sample["id"].astype(str).str.split("_", expand=True)
if parts.shape[1] != 3:
    raise ValueError(
        "Unexpected id format in sampleSubmission.csv; expected image_row_col"
    )

img_ids = parts[0].astype(np.int32).to_numpy()
rows_1b = parts[1].astype(np.int32).to_numpy()
cols_1b = parts[2].astype(np.int32).to_numpy()

test_ids = np.array([int(os.path.splitext(f)[0]) for f in test_img], dtype=np.int32)
id_to_idx = {imgid: i for i, imgid in enumerate(test_ids)}

max_row_by_id = {}
max_col_by_id = {}
for iid, r, c in zip(img_ids, rows_1b, cols_1b):
    pr = max_row_by_id.get(iid)
    pc = max_col_by_id.get(iid)
    if pr is None or r > pr:
        max_row_by_id[iid] = int(r)
    if pc is None or c > pc:
        max_col_by_id[iid] = int(c)

preds_resized_by_id = {}
for imgid in test_ids:
    i = id_to_idx[imgid]

    orig_path = os.path.join(path, "test", f"{imgid}.png")
    orig_bgr = cv2.imread(orig_path)
    if orig_bgr is None:
        raise FileNotFoundError(f"Could not read test image: {orig_path}")
    orig_gray = _to_gray_float(orig_bgr)
    _, meta = _letterbox_gray(orig_gray, config.IMG_SIZE)

    pred_gray = preds[i, ..., 0]
    pred_orig = _unletterbox_to_orig(pred_gray, meta)

    exp_h = max_row_by_id.get(imgid)
    exp_w = max_col_by_id.get(imgid)
    if exp_h is None or exp_w is None:
        exp_h, exp_w = pred_orig.shape[:2]

    pred_grid = cv2.resize(
        pred_orig, (int(exp_w), int(exp_h)), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32, copy=False)
    preds_resized_by_id[imgid] = pred_grid

values = np.empty(len(sample), dtype=np.float32)
for r_i, (imgid, r1, c1) in enumerate(zip(img_ids, rows_1b, cols_1b)):
    grid = preds_resized_by_id[imgid]
    rr = int(r1) - 1
    cc = int(c1) - 1
    if rr < 0 or cc < 0 or rr >= grid.shape[0] or cc >= grid.shape[1]:
        raise IndexError(
            f"(row,col)=({r1},{c1}) out of bounds for image id {imgid} grid {grid.shape}"
        )
    values[r_i] = grid[rr, cc]

submission = pd.DataFrame({"id": sample["id"].values, "value": values})
submission.to_csv("submission.csv", index=False)
print("submission.csv written:", submission.shape)
print(submission.head())
