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

3.8

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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
tf_keras==2.18.0
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

0.03254

# 6. Current score

0.17376

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.28616) has done: 'I first fix the crash caused by an incompatibility between `keras` (v3) and the protobuf runtime by removing the standalone `keras` import and consistently using `tf.keras` throughout. Then I fix two data/logic issues that are hurting score: ensure images and labels are paired by filename (not separately listed/shuffled), and do a proper train/validation split before fitting (so validation is meaningful and the LR scheduler can work). Finally, I make submission generation deterministic and correctly aligned to the sample submission order by parsing each `id` and indexing into the correct predicted image/row/col, and I write the output to a `.csv` file as required.'
- What this solution (achieved 0.04966) has done: 'I fix the protobuf/Keras crash by avoiding the standalone `keras` stack entirely and using `tf.keras.utils.load_img` (which works with the Kaggle TF runtime) plus a small environment setting that prevents the protobuf C++ implementation mismatch. I also fix submission generation by replacing `np.char.split` (which fails on object arrays in some NumPy builds) with a fast, robust pandas-based parse of the `id` column, keeping the sampleSubmission order intact. Finally, I make the train/validation split deterministic but shuffled (same data, no leakage) so training isn’t biased by filename ordering; this should improve RMSE toward your target without changing the model or training loop semantics.'
- What this solution (achieved 0.0531) has done: 'We fix the crash happening at import time by pinning protobuf to the pure-Python backend before TensorFlow loads and (critically) doing it early enough and with a safe fallback if the env var isn’t honored. Then we keep your model/training logic unchanged, but make image reading consistent (use `tf.keras.utils.load_img` everywhere instead of mixing `matplotlib.image.imread` and PIL) to avoid dtype/range inconsistencies that can hurt RMSE. Finally, we keep the same submission-building approach but add clipping to [0,1] and ensure the output filename ends with `.csv` while preserving exact row order from `sampleSubmission.csv`.'
- What this solution (achieved 0.1752) has done: 'We fix the import-time protobuf crash by setting the protobuf implementation environment variables before TensorFlow loads and adding a safe fallback that forces the pure-Python protobuf backend when the C++ backend triggers the `MessageFactory.GetPrototype` error. Then we keep your model and training loop the same, but make training/inference preprocessing consistent by explicitly loading images as `uint8`, scaling to `[0,1]`, and ensuring test images are resized to the same `(h, w)` used for training so the network output matches the submission pixel grid. Finally, we keep your submission-building logic but make it robust to shape mismatches by indexing into a correctly sized prediction array per image and writing a valid `.csv` file.'
- What this solution (achieved 0.28616) has done: 'We fix the import-time protobuf crash by forcing the pure-Python protobuf backend *before* TensorFlow is imported and by avoiding any direct use of `google.protobuf.internal.api_implementation` (which can trigger the `MessageFactory.GetPrototype` failure in this Kaggle runtime). Then we keep your exact model/training loop intact, but we make the unzip step idempotent (skip extracting if already present) so reruns don’t waste time or fail on partial extractions. Finally, we keep the submission logic and order exactly aligned to `sampleSubmission.csv`, ensuring a valid `.csv` is always written.'
- What this solution (achieved 0.28616) has done: 'We fix the import-time crash (`MessageFactory` missing `GetPrototype`) by forcing the pure-Python protobuf backend early *and* pre-importing protobuf before TensorFlow, with a safe fallback that disables the C++ protobuf implementation if it’s already partially loaded. Then we keep your model and training loop unchanged, but add a small, score-improving post-processing step at inference: blend the network output with the original noisy input (a standard denoising calibration trick) so predictions don’t oversmooth; this is a minimal semantic change that typically reduces RMSE substantially for this competition. Finally, we keep submission generation order identical to `sampleSubmission.csv` while ensuring dtype/range safety and writing a valid `.csv` file.'
- What this solution (achieved 0.28616) has done: 'We fix the current hard crash (`MessageFactory` missing `GetPrototype`) by forcing protobuf to use the pure-Python backend *before* any TensorFlow import and by ensuring TensorFlow is imported in a clean state. After that, we keep your model/training core logic intact, but add a small, metric-aligned, score-improving inference calibration: blend predictions with the original noisy input and tune the blend weight toward the target RMSE band (your current 0.286 is far from 0.03254, so this is a minimal, legitimate improvement lever). We also make image file listing consistently numeric-sorted everywhere to avoid accidental misalignment, and keep submission generation strictly in `sampleSubmission.csv` order with a guaranteed `.csv` output.'
- What this solution (achieved 0.17378) has done: 'We fix the hard crash at TensorFlow import caused by the protobuf C++ backend mismatch by forcing the pure-Python protobuf implementation *before* any protobuf/TensorFlow import, and by ensuring no standalone `keras` is imported. Then we make extraction robust to Kaggle’s nested `denoising-dirty-documents/` directory so the code always finds `train/`, `test/`, `train_cleaned/`, and `sampleSubmission.csv` regardless of where unzip places them. Finally, because your current score (0.28616) is far from the target (0.03254) for a lower-is-better metric, we keep the model/training loop unchanged but apply a minimal, metric-aligned inference calibration: optimize the existing blend between model output and noisy input using the validation set (grid over a few alphas) and then use that alpha for test predictions, which should move RMSE substantially toward the target without changing architecture/training semantics.'
- What this solution (achieved 0.17349) has done: 'I fix the hard crash at import time by forcing protobuf to use the pure-Python implementation *and* ensuring TensorFlow is imported in a clean order (this `MessageFactory.GetPrototype` issue is a known TF/protobuf mismatch in some Kaggle images). I keep your model, training loop, and blending calibration intact, but make the preprocessing more consistent by explicitly using float32 arrays end-to-end (avoids silent dtype upcasts/downs that can hurt RMSE). Finally, I make cleanup safe (no failing `rmtree` on busy mounts) so the notebook always completes and still writes a valid `.csv` submission in the exact `sampleSubmission.csv` row order.'
- What this solution (achieved 0.17377) has done: 'The current blocker is the TensorFlow import crash caused by a protobuf C++/Python implementation mismatch; to fix it reliably, I force the pure-Python protobuf backend *before anything protobuf-related loads*, and I also prevent an accidental standalone `keras` import path from being used. After that, I keep your model/training loop and the validation-tuned blending logic intact (so score behavior stays consistent), only making the import order and environment setup more robust so the notebook runs end-to-end. Finally, I ensure the submission CSV is always written with the correct name and format in the exact `sampleSubmission.csv` order.'
- What this solution (achieved 0.17376) has done: 'We fix the import-time crash by forcing protobuf to use the pure-Python backend *before anything that can transitively import protobuf/TensorFlow* and by removing the standalone `keras` stack from `sys.path` so TensorFlow doesn’t accidentally route into Keras 3. Then we keep your model, training loop, and validation-tuned blending exactly the same, only making the TensorFlow import robust and deterministic in this Kaggle environment. Finally, we ensure the submission is always written as a valid `.csv` (keeping the same filename you used) and that all paths resolve correctly after extraction.'

# 9. Code solution

## === cell 0
import os
import sys
import zipfile
import shutil
import gc

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def _prune_keras3_from_syspath():
    bad = []
    for p in list(sys.path):
        lp = p.lower()
        if ("site-packages" in lp or "dist-packages" in lp) and (
            "/keras" in lp or "\\keras" in lp
        ):
            bad.append(p)
        if "keras-" in lp and ("dist-info" in lp or "egg-info" in lp):
            bad.append(p)
    for p in dict.fromkeys(bad):
        while p in sys.path:
            sys.path.remove(p)


_prune_keras3_from_syspath()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import google.protobuf  # noqa: F401

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.utils import load_img

np.random.seed(42)
tf.random.set_seed(42)

print("TF:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_zip_path = "/kaggle/input/denoising-dirty-documents/train.zip"
test_zip_path = "/kaggle/input/denoising-dirty-documents/test.zip"
sample_zip_path = "/kaggle/input/denoising-dirty-documents/sampleSubmission.csv.zip"
trainclean_zip_path = "/kaggle/input/denoising-dirty-documents/train_cleaned.zip"
extracting_path = "/kaggle/working"




## === cell 2
def _find_existing_path(root, rel_name):
    """
    Kaggle dataset sometimes extracts under a nested folder
    like /kaggle/working/denoising-dirty-documents/train.
    Return the first existing path matching either root/rel_name or root/**/rel_name.
    """
    direct = os.path.join(root, rel_name)
    if os.path.exists(direct):
        return direct

    for base, dirs, files in os.walk(root):
        cand = os.path.join(base, rel_name)
        if os.path.exists(cand):
            return cand
    return direct  # fallback (may not exist yet)


def maybe_extract(zip_path, out_dir, must_contain=None):
    if must_contain is not None:
        already = True
        for p in must_contain:
            if not os.path.exists(_find_existing_path(out_dir, p)):
                already = False
                break
        if already:
            print(f"Skip extract (already present): {os.path.basename(zip_path)}")
            return

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    print(f"Extracted: {os.path.basename(zip_path)}")


maybe_extract(train_zip_path, extracting_path, must_contain=["train"])
maybe_extract(test_zip_path, extracting_path, must_contain=["test"])
maybe_extract(trainclean_zip_path, extracting_path, must_contain=["train_cleaned"])
maybe_extract(sample_zip_path, extracting_path, must_contain=["sampleSubmission.csv"])

train_dir = _find_existing_path(extracting_path, "train")
test_dir = _find_existing_path(extracting_path, "test")
train_cleaned_dir = _find_existing_path(extracting_path, "train_cleaned")
sample_csv_path = _find_existing_path(extracting_path, "sampleSubmission.csv")

print("Resolved paths:")
print(" train_dir:", train_dir, "exists:", os.path.exists(train_dir))
print(" test_dir:", test_dir, "exists:", os.path.exists(test_dir))
print(
    " train_cleaned_dir:",
    train_cleaned_dir,
    "exists:",
    os.path.exists(train_cleaned_dir),
)
print(" sample_csv_path:", sample_csv_path, "exists:", os.path.exists(sample_csv_path))



## === cell 3
img_pil = load_img(
    os.path.join(train_dir, "107.png"),
    color_mode="grayscale",
)
img_arr = np.array(img_pil)
h, w = img_arr.shape
print("Height: ", h, "- Width: ", w)
print("dtype:", img_arr.dtype)



## === cell 4
image_names = [f for f in os.listdir(train_dir) if f.lower().endswith(".png")]
image_names = sorted(image_names, key=lambda x: int(os.path.splitext(x)[0]))

data_size = len(image_names)
X_shapes = np.zeros([data_size, 2], dtype=np.uint16)
for i in tqdm(range(data_size)):
    image_name = image_names[i]
    img_dir = os.path.join(train_dir, image_name)
    img_pixels = np.array(load_img(img_dir, color_mode="grayscale"))
    X_shapes[i] = img_pixels.shape

print("Number of training images:", data_size)
print("Differnet image hights: {}".format(set(X_shapes[:, 0])))
print("Differnet image widths: {}".format(set(X_shapes[:, 1])))




## === cell 5
def images_to_array(data_dir, label_dir=None, img_size=(h, w)):
    """
    Ensure input images are paired with their correct cleaned labels
    by sorting filenames and using the same names in both directories.
    """
    image_names = sorted(
        [f for f in os.listdir(data_dir) if f.lower().endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    data_size_local = len(image_names)

    X = np.zeros([data_size_local, img_size[0], img_size[1], 1], dtype=np.float32)
    for i, image_name in enumerate(tqdm(image_names, desc="Loading X")):
        img_dir = os.path.join(data_dir, image_name)
        img_pixels = load_img(img_dir, color_mode="grayscale", target_size=img_size)
        X[i, :, :, 0] = np.array(img_pixels, dtype=np.float32) / 255.0

    if label_dir is not None:
        y = np.zeros([data_size_local, img_size[0], img_size[1], 1], dtype=np.float32)
        missing = []
        for i, image_name in enumerate(tqdm(image_names, desc="Loading y")):
            lbl_path = os.path.join(label_dir, image_name)
            if not os.path.exists(lbl_path):
                missing.append(image_name)
                continue
            img_pixels = load_img(
                lbl_path, color_mode="grayscale", target_size=img_size
            )
            y[i, :, :, 0] = np.array(img_pixels, dtype=np.float32) / 255.0

        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} label files, e.g. {missing[:5]}"
            )

        print("Output Data Size:", X.shape)
        print("Output Label Size:", y.shape)
        return X, y, image_names

    print("Output Data Size:", X.shape)
    return X, image_names




## === cell 6
X, y, train_names = images_to_array(
    train_dir,
    train_cleaned_dir,
    img_size=(h, w),
)



## === cell 7
n = X.shape[0]
rng = np.random.RandomState(42)
idx = rng.permutation(n)

val_split = int(0.3 * n)
val_idx = idx[:val_split]
train_idx = idx[val_split:]

X_val, y_val = X[val_idx], y[val_idx]
X_train, y_train = X[train_idx], y[train_idx]

print("Train data shape:", X_train.shape)
print("Val data shape:", X_val.shape)



## === cell 8
samples = np.concatenate((X_train[:3], y_train[:3]), axis=0)
f, ax = plt.subplots(2, 3, figsize=(20, 10))
for i, img in enumerate(samples):
    ax[i // 3, i % 3].imshow(img[:, :, 0], cmap="gray")
    ax[i // 3, i % 3].axis("off")
plt.show()



## === cell 9
input_layer = Input(shape=(None, None, 1))
x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), padding="same")(x)

x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = UpSampling2D((2, 2))(x)
output_layer = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])
model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()



## === cell 10
LR_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", patience=4, verbose=1, factor=0.4, min_lr=1e-5
)



## === cell 11
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=16,
    callbacks=[LR_callback],
    verbose=2,
)



## === cell 12
val_loss = model.evaluate(X_val, y_val, verbose=0)
print("Validation loss (MSE):", val_loss)



## === cell 13
test_samples, test_labels = X_val[:3], y_val[:3]
test_pred = model.predict(X_val[:3], verbose=0)

samples = np.concatenate((test_samples, test_labels, test_pred), axis=0)
f, ax = plt.subplots(3, 3, figsize=(25, 15))
for i, img in enumerate(samples):
    ax[i // 3, i % 3].imshow(img[:, :, 0], cmap="gray")
    ax[i // 3, i % 3].axis("off")
plt.show()



## === cell 14
test_image_names = sorted(
    [f for f in os.listdir(test_dir) if f.lower().endswith(".png")],
    key=lambda x: int(os.path.splitext(x)[0]),
)
test_ids = [int(os.path.splitext(f)[0]) for f in test_image_names]

X_test = np.zeros((len(test_image_names), h, w, 1), dtype=np.float32)
for i, image_name in enumerate(tqdm(test_image_names, desc="Loading test")):
    img_dir = os.path.join(test_dir, image_name)
    img_pixels = load_img(img_dir, color_mode="grayscale", target_size=(h, w))
    X_test[i, :, :, 0] = np.array(img_pixels, dtype=np.float32) / 255.0

print("X_test shape:", X_test.shape, "dtype:", X_test.dtype)
print("Num test images:", len(X_test))



## === cell 15
val_pred = model.predict(X_val, verbose=0)[:, :, :, 0]
val_noisy = X_val[:, :, :, 0]
val_true = y_val[:, :, :, 0]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


alphas = np.array([0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0], dtype=np.float32)
best_alpha = None
best_rmse = None
for a in alphas:
    blended = np.clip(a * val_pred + (1.0 - a) * val_noisy, 0.0, 1.0)
    r = rmse(blended, val_true)
    if (best_rmse is None) or (r < best_rmse):
        best_rmse = r
        best_alpha = float(a)

print("Best blend_alpha on val:", best_alpha, "val_RMSE:", best_rmse)



## === cell 16
yh_test = model.predict(X_test, verbose=0)[:, :, :, 0]  # (N, h, w)

blend_alpha = best_alpha  # tuned on validation to reduce RMSE
yh_test = blend_alpha * yh_test + (1.0 - blend_alpha) * X_test[:, :, :, 0]
yh_test = np.clip(yh_test, 0.0, 1.0)

pred_by_imgid = {img_id: pred for img_id, pred in zip(test_ids, yh_test)}



## === cell 17
f, ax = plt.subplots(3, 2, figsize=(20, 10))
for i in range(min(3, len(test_image_names))):
    ax[i, 0].imshow(X_test[i, :, :, 0], cmap="gray")
    ax[i, 0].axis("off")

    ax[i, 1].imshow(yh_test[i], cmap="gray")
    ax[i, 1].axis("off")
plt.show()



## === cell 18
sample_csv = pd.read_csv(sample_csv_path)
id_series = sample_csv["id"].astype(str)

split_df = id_series.str.split("_", expand=True)
img_ids = split_df[0].astype(np.int32).to_numpy()
rows = (split_df[1].astype(np.int32) - 1).to_numpy()
cols = (split_df[2].astype(np.int32) - 1).to_numpy()

values = np.empty(len(sample_csv), dtype=np.float32)
unique_img_ids = np.unique(img_ids)
for img_id in tqdm(unique_img_ids, desc="Building submission"):
    mask = img_ids == img_id
    pred_img = pred_by_imgid[int(img_id)]
    r = np.clip(rows[mask], 0, pred_img.shape[0] - 1)
    c = np.clip(cols[mask], 0, pred_img.shape[1] - 1)
    values[mask] = pred_img[r, c]

values = np.clip(values, 0.0, 1.0)

submission = pd.DataFrame({"id": sample_csv["id"].values, "value": values})

submission_path = "Cleared.csv"
submission.to_csv(submission_path, index=False)
print(submission.head())
print("Wrote submission:", os.path.abspath(submission_path), "rows:", len(submission))




## === cell 19
def safe_rmtree(path):
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
            return True
    except OSError as e:
        print(f"Skipping cleanup for {path} due to OSError: {e}")
    return False


for folder in ["train", "test", "train_cleaned"]:
    direct = os.path.join(extracting_path, folder)
    safe_rmtree(direct)

gc.collect()
print("Cleanup done (best-effort).")
