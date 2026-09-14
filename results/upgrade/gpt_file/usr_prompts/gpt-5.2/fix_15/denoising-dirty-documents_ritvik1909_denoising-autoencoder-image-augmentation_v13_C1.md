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

0.03077

# 6. Current score

0.28616

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.47318) has done: 'I fix the immediate import/runtime failure by removing the unused `imgaug` dependency that is incompatible in this environment (it triggers the protobuf `MessageFactory.GetPrototype` error), and replace the augmentation step with a minimal, equivalent geometric augmentation implemented via NumPy (rotations/flips) so the pipeline still expands the dataset without changing the model/training loop. I also ensure image/pair alignment by building train/train_cleaned using the intersection of filenames (sorted) to avoid any accidental mismatch that can severely hurt RMSE. Finally, I keep the autoencoder architecture and training semantics intact, but fix the inference call to avoid mixing `.numpy()` inside Keras calls and guarantee values are clipped to `[0,1]` before writing a valid `submission.csv`.'
- What this solution (achieved 0.28616) has done: 'I fix the two runtime blockers: (1) the protobuf-related import crash by removing the unused plotting/seaborn imports that pull in incompatible extras in this environment, and (2) the augmentation shape-mismatch by ensuring rotations keep the (420,540) shape via resize-back after `np.rot90`. These are execution/correctness fixes and are score-positive because the model actually train on correctly paired and consistently-shaped augmented data instead of failing or learning from mis-shaped arrays. I also make the submission writing robust and fast by vectorizing the pixel “melt” using the provided `sampleSubmission.csv` ids (guaranteeing exact row count/order and correct formatting) while keeping the same prediction values. Core model architecture, loss, and training loop are preserved.'
- What this solution (achieved 0.28616) has done: 'I fix the immediate runtime crash (`MessageFactory` / protobuf incompatibility) by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow, which is a minimal and score-neutral stability fix. I also fix a logic bug in your augmentation pipeline that was accidentally applying every augmentation step to the original images rather than sequentially to the progressively augmented set, which can materially hurt learning and RMSE while keeping the same augmentation “idea.” Finally, I make submission filling robust to any ordering differences by assigning values using the `ids` order directly (no dependence on sorting keys), while preserving your use of `sampleSubmission.csv` to guarantee exact format/row count.'
- What this solution (achieved 0.28616) has done: 'The timeout is dominated by (1) the Python-loop augmentation that repeatedly concatenates huge 420×540 tensors and resizes per-image in Python, and (2) submission assembly that parses 5.7M string ids and then does per-image boolean masking over the full array. I keep the exact same augmentations and model/training logic, but implement augmentation as a single preallocated array with vectorized flips/rotations and only the necessary OpenCV resizes, eliminating repeated `np.concatenate`. For prediction/submission, I predict the whole test set in one batch, resize outputs once per test image, and assemble the 5.7M values in a single pass using an O(N) stable sort + contiguous slicing per image id (no repeated global boolean masks). These are equivalent transformations and preserve evaluation semantics while reducing overhead drastically.'
- What this solution (achieved 0.28616) has done: 'The timeout is dominated by (1) building a 32× augmented dataset in RAM (major CPU + memory traffic) and then training on it, and (2) constructing the 5.8M-row submission via heavy pandas string splitting and per-image Python loops. To keep the exact same model/loss/training semantics while cutting runtime, I (a) stream the exact same augmentation set through a `tf.data` pipeline (no materialized 32× arrays, same transforms) and (b) generate the submission values in a fully vectorized way by parsing IDs with NumPy and gathering pixel values from a pre-resized prediction tensor. These changes preserve core logic and results (same data, same network, same optimization), but eliminate the major Python/pandas overhead and huge intermediate allocations that cause the 10-minute timeout.'
- What this solution (achieved 0.28616) has done: 'The main timeout driver is the `tf.data` pipeline doing a heavy per-batch augmentation (`map` + `unbatch`) and then training for up to 500 epochs with a very large `steps_per_epoch`; this causes enormous Python/graph overhead and excessive per-step input work. I keep the exact same augmentation logic and training semantics, but move augmentation to a one-time, fully-vectorized TensorFlow precompute (single pass) and then train from in-memory tensors using a cached, prefetched dataset—this removes the repeated per-step augmentation cost while preserving identical augmented samples. I also compile the model with XLA (`jit_compile=True`) and enable dataset options that reduce overhead, without changing the model, loss, optimizer, or epoch/step schedule. Finally, submission generation is kept identical, but avoids extra image reads and redundant conversions.'
- What this solution (achieved 0.28616) has done: 'The timeout is dominated by constructing a full 32× augmented dataset in memory (and caching it), which explodes both CPU time (many resizes/rotations) and RAM, and by pushing the whole augmented tensor into a `Dataset` before training. I keep the exact same augmentation semantics and training loop, but generate those 32 variants lazily via `tf.data` (repeat + map) so we never materialize/cache the full augmented array. I also speed up image loading by reading directly as grayscale and resizing once, and I make submission creation avoid expensive string parsing by reading the sample submission once and using vectorized indexing as before. These changes preserve core logic/model/loss and produce the same effective training set and steps-per-epoch, but avoid the major bottlenecks.'
- What this solution (achieved 0.28616) has done: 'The timeout is dominated by the tf.data pipeline creating a 32× augmented dataset on the fly each epoch (flat_map over 32 and per-sample map), plus very slow Python-side submission building due to parsing 5.7M ids and dict-based indexing. I keep the exact same 32-way augmentation logic and model/training semantics, but materialize the 32× augmented arrays once (using your existing equivalent NumPy augmentation), then train on a simple cached/batched tf.data dataset to remove repeated augmentation overhead. For submission, I replace the expensive string parsing and Python dict/index generator with a fully vectorized, deterministic mapping from `sampleSubmission` ids to test array indices using sorting/searchsorted, and avoid creating an unnecessary large DataFrame in memory before writing.'
- What this solution (achieved 0.28616) has done: 'The main timeout driver is the 32× offline augmentation that explodes the in-memory training set and makes each epoch ~32× more expensive, plus a very high default epoch cap. I keep the exact same 32 augmentation variants and the same model/training semantics, but generate those variants on-the-fly in a deterministic `tf.data` pipeline so you train on the same effective data without materializing a huge array. I also remove unnecessary double-augmentation code paths and cut Python overhead by using `tf.data` caching/prefetching properly while preserving determinism and the same `steps_per_epoch`. Finally, I speed up submission creation by avoiding loading/parsing the 5.7M-row sample submission more than necessary and by using faster vectorized string parsing.'
- What this solution (achieved 0.28616) has done: 'The timeout is dominated by the tf.data pipeline generating 32x augmented samples on the fly and caching them, which forces expensive per-image `tf.image` ops (rotations + resizes) and huge memory traffic during training. To preserve identical core logic (same 32 deterministic augmentations, same model, same optimizer/loss/fit semantics), I precompute the 32 augmentations once in NumPy/OpenCV (equivalent transforms) and build a simple cached/batched tf.data dataset from the already-augmented arrays, eliminating the repeated augmentation cost and the expensive tf.data `flat_map(map())` graph. I also keep determinism/seeds, keep all file paths, and speed up inference/submission creation by batching prediction and avoiding extra intermediate allocations. These changes are equivalence-preserving: the model sees the same augmented images, just produced once instead of every epoch.'
- What this solution (achieved 0.28616) has done: 'The main timeout is coming from materializing a 32× augmented training set in NumPy (large CPU work + huge RAM traffic) before training even begins, plus extra copying from caching that giant tensor-slice dataset. I keep the *exact same augmentation logic and training loop semantics* but move augmentation into a deterministic `tf.data` pipeline so batches are generated on-the-fly (no change to model, loss, or optimizer), and I compile the augmentation mapping with `tf.numpy_function` so it reuses your same `_apply_aug_np` function. I also remove the expensive dataset `.cache()` of the full augmented set (it defeats the purpose and is often a big slowdown here) and add prefetching plus deterministic options to preserve repeatability. Submission generation be kept identical logically, with minor I/O parsing tweaks left as-is since training is the real bottleneck.'
- What this solution (achieved 0.47572) has done: 'The timeout is dominated by the on-the-fly augmentation pipeline: `tf.numpy_function` runs Python/OpenCV for every augmented sample (32×114 images) across many steps/epochs, which is far slower than the model itself. To preserve the exact augmentation semantics while removing Python from the hot path, I replace the augmentation with an equivalent pure-TensorFlow implementation (same 32-case transform mapping) and apply it to both input/target tensors inside the `tf.data` pipeline. I also switch the dataset construction to cache the base tensors and vectorize the augmentation with `Dataset.from_tensors(...).repeat(N_AUG)` + enumeration, avoiding the expensive `flat_map(range).map(...)` nesting. Submission creation remains the same logic but uses slightly more memory-efficient pandas parsing and avoids unnecessary temporary arrays.'
- What this solution (achieved 0.28616) has done: 'I fix the two runtime blockers: (1) the protobuf `MessageFactory.GetPrototype` crash by ensuring we don’t import TensorFlow under an incompatible protobuf runtime (pure-Python protobuf + safe TF import), and (2) the augmentation batching failure by guaranteeing every augmentation variant keeps the fixed `(420, 540, 1)` shape (no 90° rotations that swap H/W without resizing back). These are minimal changes that preserve your model, loss, and training loop semantics, but allow training to actually run and should materially improve RMSE versus the current broken/misaligned training. I also keep the submission generation logic, but make it robust to any missing ids by guarding the searchsorted indexing and ensure the output is always a valid `submission.csv`.'
- What this solution (achieved 0.28616) has done: 'The main timeout driver is the training data pipeline expanding the dataset by 32× and then forcing `steps_per_epoch` to iterate through all augmented samples every epoch; this multiplies training time by ~32 without changing the model itself. I keep the exact same augmentation function and training loop semantics, but move augmentation to happen inside each epoch without physically repeating the dataset 32×, and I reduce per-step overhead by avoiding `enumerate()` and heavy `tf.cond` branching in the hot path. I also speed up image loading by parallelizing disk reads with a thread pool (identical pixels, just faster I/O) and ensure the `tf.data` pipeline is fully optimized (cache/prefetch/options) while staying deterministic. Submission generation be kept identical, but I avoid unnecessary intermediate objects to reduce memory/time.'

# 9. Code solution

## === cell 0
import os, zipfile

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks

np.random.seed(19)
tf.random.set_seed(19)

AUTOTUNE = tf.data.AUTOTUNE

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide / use env
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path_zip = "../input/denoising-dirty-documents/"
path = "/kaggle/working/"


def _maybe_extract(zip_path, out_path, must_exist_relpath):
    target = os.path.join(out_path, must_exist_relpath)
    if os.path.exists(target) and (os.path.isdir(target) or os.path.isfile(target)):
        return
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_path)


_maybe_extract(path_zip + "train.zip", path, "train")
_maybe_extract(path_zip + "test.zip", path, "test")
_maybe_extract(path_zip + "train_cleaned.zip", path, "train_cleaned")
_maybe_extract(path_zip + "sampleSubmission.csv.zip", path, "sampleSubmission.csv")

train_dir = os.path.join(path, "train")
train_cleaned_dir = os.path.join(path, "train_cleaned")
test_dir = os.path.join(path, "test")

train_img = sorted([f for f in os.listdir(train_dir) if f.lower().endswith(".png")])
train_cleaned_img = sorted(
    [f for f in os.listdir(train_cleaned_dir) if f.lower().endswith(".png")]
)
test_img = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".png")])

common = sorted(set(train_img).intersection(set(train_cleaned_img)))
if len(common) == 0:
    raise RuntimeError(
        "No overlapping filenames between train and train_cleaned. Check extracted paths."
    )
train_img = common
train_cleaned_img = common

print(
    "Train images:",
    len(train_img),
    "Train cleaned images:",
    len(train_cleaned_img),
    "Test images:",
    len(test_img),
)




## === cell 2
class config:
    IMG_SIZE = (420, 540)


imgs = [cv2.imread(os.path.join(train_dir, f)) for f in train_img[:10]]
print("Sample Dimensions (H,W):", [img.shape[:2] for img in imgs if img is not None])
del imgs




## === cell 3
def process_image(path_):
    img = cv2.imread(path_, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path_}")
    img = cv2.resize(img, config.IMG_SIZE[::-1], interpolation=cv2.INTER_LINEAR)
    img = img.astype("float32") / 255.0
    img = img.reshape((*config.IMG_SIZE, 1))
    return img




## === cell 4
from concurrent.futures import ThreadPoolExecutor

n_train = len(train_img)
n_test = len(test_img)
H, W = config.IMG_SIZE

train = np.empty((n_train, H, W, 1), dtype=np.float32)
train_cleaned = np.empty((n_train, H, W, 1), dtype=np.float32)
test = np.empty((n_test, H, W, 1), dtype=np.float32)


def _load_into_array(file_list, root_dir, out_arr, desc):
    paths = [os.path.join(root_dir, f) for f in file_list]
    with ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4))) as ex:
        for i, img in enumerate(
            tqdm(ex.map(process_image, paths), total=len(paths), desc=desc)
        ):
            out_arr[i] = img


_load_into_array(train_img, train_dir, train, "Loading train")
_load_into_array(
    train_cleaned_img, train_cleaned_dir, train_cleaned, "Loading train_cleaned"
)
_load_into_array(test_img, test_dir, test, "Loading test")



## === cell 5
train.shape, train_cleaned.shape, test.shape




## === cell 6
@tf.function(jit_compile=True)
def _tf_apply_aug_fast(x, k):
    k = tf.cast(k, tf.int32)

    flip_lr = k >= 16
    k2 = tf.where(flip_lr, k - 16, k)

    flip_ud = k2 >= 8
    k3 = tf.where(flip_ud, k2 - 8, k2)

    base = k3  # 0..7
    lr_inner = base >= 4
    base4 = tf.where(lr_inner, base - 4, base)  # 0..3

    rot_k = tf.where(tf.equal(base4, 2), 2, 0)  # {0,2} only
    z = tf.image.rot90(x, k=rot_k)

    z = tf.cond(lr_inner, lambda: tf.image.flip_left_right(z), lambda: z)
    z = tf.cond(flip_ud, lambda: tf.image.flip_up_down(z), lambda: z)
    z = tf.cond(flip_lr, lambda: tf.image.flip_left_right(z), lambda: z)

    z.set_shape((H, W, 1))
    return z


BATCH_SIZE = 12
N_AUG = 32

base = tf.data.Dataset.from_tensor_slices((train, train_cleaned)).cache()

counter = tf.data.experimental.Counter()  # infinite 0,1,2,...
indexed = tf.data.Dataset.zip((counter, base.repeat()))


def _map_with_aug(i, xy):
    k = tf.math.floormod(i, N_AUG)
    return _tf_apply_aug_fast(xy[0], k), _tf_apply_aug_fast(xy[1], k)


opt = tf.data.Options()
opt.experimental_deterministic = True

train_ds = (
    indexed.map(_map_with_aug, num_parallel_calls=AUTOTUNE, deterministic=True)
    .with_options(opt)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

steps_per_epoch = int(np.ceil(train.shape[0] / BATCH_SIZE))
print(
    "Train samples:",
    train.shape[0],
    "N_AUG (streamed):",
    N_AUG,
    "steps_per_epoch:",
    steps_per_epoch,
)




## === cell 7
class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(*config.IMG_SIZE, 1)),
                layers.Conv2D(48, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(72, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(144, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Dropout(0.5),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Conv2D(144, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(72, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(48, (3, 3), activation="relu", padding="same"),
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
    metrics=["mean_absolute_error"],
    jit_compile=True,
)



## === cell 8
es = callbacks.EarlyStopping(
    monitor="loss", patience=30, verbose=1, restore_best_weights=True
)
rlp = callbacks.ReduceLROnPlateau(
    monitor="loss", factor=0.8, patience=5, min_lr=1e-6, mode="min", verbose=1
)

history = autoencoder.fit(
    train_ds,
    callbacks=[es, rlp],
    epochs=500,
    steps_per_epoch=steps_per_epoch,
    batch_size=None,
)



## === cell 9
print("Final training loss:", float(history.history["loss"][-1]))



## === cell 10
autoencoder.encoder.summary()
autoencoder.decoder.summary()



## === cell 11
decoded_imgs = autoencoder(train[:4], training=False).numpy()
decoded_imgs = np.clip(decoded_imgs, 0.0, 1.0)
print(
    "Sanity decoded batch:", decoded_imgs.shape, decoded_imgs.min(), decoded_imgs.max()
)
del decoded_imgs



## === cell 12
sample_path = os.path.join(path, "sampleSubmission.csv")
if not os.path.exists(sample_path):
    sample_path = "../input/denoising-dirty-documents/sampleSubmission.csv"

sample_sub = pd.read_csv(sample_path, usecols=["id"], dtype={"id": "string"})
ids = sample_sub["id"].to_numpy()

parts = sample_sub["id"].str.split("_", n=2, expand=True)
img_ids = parts[0].astype("int32").to_numpy()
rows = parts[1].astype("int32").to_numpy() - 1
cols = parts[2].astype("int32").to_numpy() - 1
del sample_sub, parts

decoded_test = autoencoder.predict(test, batch_size=BATCH_SIZE, verbose=0)
decoded_test = np.clip(decoded_test, 0.0, 1.0)

test_imgids = np.fromiter(
    (int(f[:-4]) for f in test_img), dtype=np.int32, count=len(test_img)
)

h0, w0 = cv2.imread(os.path.join(test_dir, test_img[0]), cv2.IMREAD_GRAYSCALE).shape
if (h0, w0) != config.IMG_SIZE:
    decoded_test_rs = (
        tf.image.resize(decoded_test, size=(h0, w0), method="bilinear", antialias=False)
        .numpy()
        .astype(np.float32)
    )
else:
    decoded_test_rs = decoded_test.astype(np.float32, copy=False)
del decoded_test

order = np.argsort(test_imgids)
test_sorted = test_imgids[order]
pos = np.searchsorted(test_sorted, img_ids)
pos = np.clip(pos, 0, len(test_sorted) - 1)
idx = order[pos].astype(np.int32, copy=False)

values = decoded_test_rs[idx, rows, cols, 0].astype(np.float32, copy=False)
del decoded_test_rs, idx, order, test_sorted, pos, img_ids, rows, cols, test_imgids

sub = pd.DataFrame({"id": ids, "value": values})
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv with shape:", sub.shape)
print(sub.head())
