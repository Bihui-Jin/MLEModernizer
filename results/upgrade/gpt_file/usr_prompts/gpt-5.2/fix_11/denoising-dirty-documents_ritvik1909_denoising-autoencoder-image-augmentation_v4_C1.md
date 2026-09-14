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

0.28615

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.28616) has done: 'I remove the dependency on `imgaug` because it’s failing to import in this Kaggle environment (protobuf compatibility), and replace it with equivalent augmentation using TensorFlow ops so the pipeline can run. I also fix the earlier protobuf-related import crash by dropping unused plotting/augmentation imports that trigger it. The model architecture, loss, and training loop remain the same; only the augmentation implementation changes so `processed_train`/`processed_train_cleaned` are created correctly. Finally, I keep the submission-writing logic but make sure predictions are clipped to [0, 1] for valid intensity outputs and consistent RMSE.'
- What this solution (achieved 0.28616) has done: 'I fix the protobuf-related import crash by removing the unused `matplotlib` import (it triggers the `MessageFactory.GetPrototype` issue in this environment) while keeping all training/inference logic unchanged. Then I fix the augmentation bug: `rot90/180/270` swaps height/width for non-square images, so concatenation fails; I replace rotations with shape-preserving deterministic augmentations (flips + small brightness/contrast jitter) to keep the “append augmented copies” core idea and improve denoising performance toward the target. Finally, I ensure the training cells run by making `processed_train`/`history` always defined, and I keep the submission format identical while clipping predictions to `[0,1]` (already correct) and writing `submission.csv`.'
- What this solution (achieved 0.28616) has done: 'The timeout is dominated by (1) slow Python loops when building the 5.8M-row submission and (2) per-image TensorFlow overhead from calling `encoder` and `decoder` separately for each test image. I keep the exact same model and training loop, but speed up data I/O by loading images in parallel and avoid redundant work. For inference, I run the full autoencoder in one batched forward pass (same computation as encoder+decoder) and only resize once per image. Finally, I generate the submission using vectorized NumPy/Pandas (no per-pixel Python loops) and stream-write to CSV to avoid huge Python-list overhead.'
- What this solution (achieved 0.28616) has done: 'The timeout is dominated by two hotspots: (1) training on large 420×540 images with an expensive Keras input pipeline (NumPy feeding, no prefetch), and (2) extremely slow submission generation due to per-image pandas DataFrame creation plus repeated disk reads inside the prediction loop. I keep the same model, loss, and training semantics, but switch training to a cached/prefetched `tf.data.Dataset` (same shuffle behavior) and move submission writing to a fully vectorized, chunked writer that avoids pandas and avoids re-reading test images from disk. I also precompute original test image sizes once and reuse them, and compile the model call into a `tf.function` for faster inference while preserving identical outputs (negligible FP differences only). These changes remove the major Python overhead and redundant I/O, which is what typically pushes this notebook past 600 seconds.'
- What this solution (achieved 0.28616) has done: 'The timeout is most likely dominated by (1) expensive Python-side augmentation that copies the full training set 5× into RAM, (2) slow training input pipeline due to huge cached tensors, and (3) submission writing that builds millions of Python strings/joins in loops. The optimizations below keep the exact same model, loss, optimizer, and training semantics, but move augmentation into a `tf.data` pipeline (no materialized 5× arrays), remove unnecessary `.cache()` on a very large dataset, and vectorize/stream submission generation using NumPy to avoid per-pixel Python overhead. These changes are correctness-preserving (same deterministic stateless augmentations, same training pairs, same resizing/normalization), but dramatically reduce memory pressure and Python time. I also keep determinism by using stateless RNG and explicit seeds.'
- What this solution (achieved 0.28616) has done: 'The main timeout culprit is the submission writer: it generates ~5.8M string IDs via costly NumPy string ops and per-image joins, which is far slower than the model itself. I keep the model/training exactly the same, but replace submission creation with a provably equivalent, much faster approach that reuses the official `sampleSubmission.csv` IDs and only fills the `value` column in the same row order. I also eliminate repeated Tensor conversions during prediction and build a single `tf.data` pipeline for test inference to reduce Python overhead while keeping identical prediction semantics. These changes preserve correctness because Kaggle evaluates only the mapping from each `id` to its pixel value, and we keep the exact same ordering/format as the sample submission.'
- What this solution (achieved 0.28616) has done: 'The timeout is dominated by (1) building a 5x augmented tensor in memory and then training up to 500 epochs with a large model on large images, and (2) submission creation by reading a 5.8M-row CSV and assigning values. To keep core logic identical while reducing wall time, I switch augmentation and training input to an on-the-fly `tf.data` pipeline (same transforms, deterministic seeds) so we avoid huge concat tensors and speed up input throughput. I also enable TensorFlow XLA JIT compilation for the training step (same math, typically faster on Kaggle CPUs) and streamline prediction/submission writing by avoiding pandas CSV read and writing the required `id,value` directly using the known test filenames and original image shapes. All paths and the model/training semantics are preserved (same epochs/patience/loss/model), with only equivalent runtime-focused changes.'
- What this solution (achieved 0.28616) has done: 'The timeout is dominated by two hotspots: (1) `tf.data` augmentation uses `flat_map` + per-sample dataset construction, which is slow and prevents efficient pipelining; and (2) submission writing uses deeply nested Python loops to emit ~5.8M lines. I keep the exact same 5-variant augmentation logic and the same model/training semantics, but implement augmentation as a pure tensor mapping (no per-element dataset creation) and then `unbatch`, enabling much faster graph execution and prefetching. I also preserve identical prediction/resizing logic but vectorize CSV creation by using `np.repeat/np.tile` for ids and `DataFrame.to_csv`, eliminating the Python per-pixel write loop. All changes are deterministic and do not alter the model, loss, number of epochs, early stopping behavior, or output values (aside from negligible float formatting differences).'
- What this solution (achieved 0.28616) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow, which is a known compatibility workaround in Kaggle-style environments. Then I fix the submission ID generation error by avoiding NumPy string ufunc concatenation (which can fail with mixed unicode dtypes) and instead fill the submission by reusing the official `sampleSubmission.csv` row order and only replacing the `value` column—this is both faster and guarantees the exact expected ID format. Finally, I keep the model/training core logic unchanged, but I align the training metric to the competition’s RMSE by using `RootMeanSquaredError` as an additional metric (loss unchanged), which is score-neutral in training behavior and helps sanity-check. These changes should make the notebook run end-to-end and produce a valid `submission.csv`, and the submission fix can improve score if the previous ID mapping/order was wrong.'
- What this solution (achieved 0.28615) has done: 'Your current score is far worse than the target (RMSE 0.286 vs 0.02895), so we should make a small, score-relevant correction rather than performance tweaks. The biggest issue is that your augmentation applies brightness/contrast jitter only to the noisy input `x` but not to the clean target `y`, which teaches the network to “undo” random brightness changes that are not noise and hurts RMSE badly. I keep the same augmentation idea (5 variants), same model, same loss, and same training loop, but make the jitter transforms consistent by applying the exact same stateless jitter to both `x` and `y`. Everything else (paths, resizing, clipping, submission creation using sampleSubmission IDs) stays the same to preserve semantics and ensure a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os, zipfile

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks

np.random.seed(19)
tf.random.set_seed(19)

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path_zip = "../input/denoising-dirty-documents/"
path = "/kaggle/working/"


def _extract_if_missing(zip_path, target_dir, expected_subdir):
    if (
        not os.path.exists(os.path.join(target_dir, expected_subdir))
        or len(os.listdir(os.path.join(target_dir, expected_subdir))) == 0
    ):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)


_extract_if_missing(path_zip + "train.zip", path, "train")
_extract_if_missing(path_zip + "test.zip", path, "test")
_extract_if_missing(path_zip + "train_cleaned.zip", path, "train_cleaned")

if not os.path.exists(os.path.join(path, "sampleSubmission.csv")):
    with zipfile.ZipFile(path_zip + "sampleSubmission.csv.zip", "r") as zip_ref:
        zip_ref.extractall(path)

train_img = sorted(os.listdir(path + "/train"))
train_cleaned_img = sorted(os.listdir(path + "/train_cleaned"))
test_img = sorted(os.listdir(path + "/test"))




## === cell 2
class config:
    IMG_SIZE = (420, 540)


train_files = sorted(os.listdir(path + "train/"))
heights = []
widths = []
for f in train_files:
    img = cv2.imread(path + "train/" + f, cv2.IMREAD_GRAYSCALE)
    heights.append(img.shape[0])
    widths.append(img.shape[1])
del img
print("Median Dimensions:", np.median(heights), np.median(widths))
del heights, widths, train_files




## === cell 3
def process_image(path_):
    img = cv2.imread(path_, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, config.IMG_SIZE[::-1])
    img = img.astype("float32") / 255.0
    img = np.reshape(img, (*config.IMG_SIZE, 1))
    return img




## === cell 4
from concurrent.futures import ThreadPoolExecutor


def _load_folder(folder_path):
    files = sorted(os.listdir(folder_path))
    full_paths = [os.path.join(folder_path, f) for f in files]
    max_workers = min(8, (os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        imgs = list(ex.map(process_image, full_paths))
    return np.asarray(imgs, dtype=np.float32), files


train, _train_files = _load_folder(path + "train/")
train_cleaned, _trainc_files = _load_folder(path + "train_cleaned/")
test, _test_files = _load_folder(path + "test/")

train_img = sorted(_train_files)
train_cleaned_img = sorted(_trainc_files)
test_img = sorted(_test_files)

del _train_files, _trainc_files, _test_files



## === cell 5
train.shape, train_cleaned.shape, test.shape




## === cell 6
def _augment_5x_tf(images_tf, seed=19):
    hflip = tf.image.flip_left_right(images_tf)
    vflip = tf.image.flip_up_down(images_tf)

    jitter1 = tf.image.stateless_random_brightness(
        images_tf, max_delta=0.08, seed=[seed, 1]
    )
    jitter1 = tf.image.stateless_random_contrast(
        jitter1, lower=0.85, upper=1.15, seed=[seed, 2]
    )

    jitter2 = tf.image.stateless_random_brightness(
        images_tf, max_delta=0.05, seed=[seed, 3]
    )
    jitter2 = tf.image.stateless_random_contrast(
        jitter2, lower=0.90, upper=1.10, seed=[seed, 4]
    )

    augmented = tf.concat([images_tf, hflip, vflip, jitter1, jitter2], axis=0)
    augmented = tf.clip_by_value(augmented, 0.0, 1.0)
    return augmented


processed_train = None
processed_train_cleaned = None




## === cell 7
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
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mean_absolute_error", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
)



## === cell 8
BATCH_SIZE = 24

train_tf = tf.convert_to_tensor(train, dtype=tf.float32)
trainc_tf = tf.convert_to_tensor(train_cleaned, dtype=tf.float32)

base_ds = tf.data.Dataset.from_tensor_slices((train_tf, trainc_tf))


@tf.function(reduce_retracing=True)
def _make_aug_variants_tensor(x, y, seed=19):
    x_h = tf.image.flip_left_right(x)
    y_h = tf.image.flip_left_right(y)
    x_v = tf.image.flip_up_down(x)
    y_v = tf.image.flip_up_down(y)

    x_j1 = tf.image.stateless_random_brightness(x, max_delta=0.08, seed=[seed, 1])
    y_j1 = tf.image.stateless_random_brightness(y, max_delta=0.08, seed=[seed, 1])
    x_j1 = tf.image.stateless_random_contrast(
        x_j1, lower=0.85, upper=1.15, seed=[seed, 2]
    )
    y_j1 = tf.image.stateless_random_contrast(
        y_j1, lower=0.85, upper=1.15, seed=[seed, 2]
    )

    x_j2 = tf.image.stateless_random_brightness(x, max_delta=0.05, seed=[seed, 3])
    y_j2 = tf.image.stateless_random_brightness(y, max_delta=0.05, seed=[seed, 3])
    x_j2 = tf.image.stateless_random_contrast(
        x_j2, lower=0.90, upper=1.10, seed=[seed, 4]
    )
    y_j2 = tf.image.stateless_random_contrast(
        y_j2, lower=0.90, upper=1.10, seed=[seed, 4]
    )

    x0 = tf.clip_by_value(x, 0.0, 1.0)
    x_h = tf.clip_by_value(x_h, 0.0, 1.0)
    x_v = tf.clip_by_value(x_v, 0.0, 1.0)
    x_j1 = tf.clip_by_value(x_j1, 0.0, 1.0)
    x_j2 = tf.clip_by_value(x_j2, 0.0, 1.0)

    y0 = tf.clip_by_value(y, 0.0, 1.0)
    y_h = tf.clip_by_value(y_h, 0.0, 1.0)
    y_v = tf.clip_by_value(y_v, 0.0, 1.0)
    y_j1 = tf.clip_by_value(y_j1, 0.0, 1.0)
    y_j2 = tf.clip_by_value(y_j2, 0.0, 1.0)

    x_stack = tf.stack([x0, x_h, x_v, x_j1, x_j2], axis=0)
    y_stack = tf.stack([y0, y_h, y_v, y_j1, y_j2], axis=0)
    return x_stack, y_stack


train_ds = base_ds.map(
    lambda x, y: _make_aug_variants_tensor(x, y, seed=19),
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=True,
).unbatch()

train_ds = train_ds.shuffle(
    buffer_size=int(train.shape[0] * 5), seed=19, reshuffle_each_iteration=True
)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

es = callbacks.EarlyStopping(
    monitor="loss", patience=30, verbose=1, restore_best_weights=True
)

history = autoencoder.fit(
    train_ds,
    callbacks=[es],
    epochs=500,
)



## === cell 9
_ = history.history  # no-op to ensure history exists
del history



## === cell 10
autoencoder.encoder.summary()
autoencoder.decoder.summary()



## === cell 11
decoded_imgs = autoencoder(train[:4]).numpy()
print("Sanity decoded batch shape:", decoded_imgs.shape)
del decoded_imgs



## === cell 12
sample_path = os.path.join(path, "sampleSubmission.csv")
sample = pd.read_csv(sample_path)
if list(sample.columns) != ["id", "value"]:
    raise ValueError(f"Unexpected sampleSubmission columns: {sample.columns.tolist()}")

test_hw = []
for fname in test_img:
    fp = os.path.join(path, "test", fname)
    im0 = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    if im0 is None:
        raise FileNotFoundError(f"Could not read test image: {fp}")
    test_hw.append(im0.shape)  # (h,w)
del im0


@tf.function(reduce_retracing=True)
def _predict(x):
    return autoencoder(x, training=False)


batch_size = 8  # unchanged
test_ds = (
    tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(test, dtype=tf.float32))
    .batch(batch_size, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

total_pixels = int(sum(h * w for (h, w) in test_hw))
values = np.empty((total_pixels,), dtype=np.float32)

offset = 0
num_batches = int(np.ceil(test.shape[0] / batch_size))
for b, x in enumerate(tqdm(test_ds, total=num_batches)):
    preds_batch = _predict(x).numpy()  # (B,420,540,1)
    preds_batch = np.squeeze(preds_batch, axis=-1)  # (B,420,540)

    start = b * batch_size
    end = min(test.shape[0], start + preds_batch.shape[0])

    for i in range(end - start):
        h, w = test_hw[start + i]
        decoded_img = preds_batch[i]
        preds_reshaped = cv2.resize(decoded_img, (w, h), interpolation=cv2.INTER_LINEAR)
        preds_reshaped = (
            np.clip(preds_reshaped, 0.0, 1.0).reshape(-1).astype(np.float32, copy=False)
        )

        n = preds_reshaped.size
        values[offset : offset + n] = preds_reshaped
        offset += n

if offset != total_pixels:
    raise ValueError(f"Filled pixels ({offset}) != expected ({total_pixels})")

if len(sample) != total_pixels:
    raise ValueError(
        f"sampleSubmission rows ({len(sample)}) != predicted pixels ({total_pixels})"
    )

sample["value"] = values.astype(np.float32, copy=False)
out_path = "submission.csv"
sample.to_csv(out_path, index=False)
print(f"Results saved to {out_path} with shape={sample.shape}!")
