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

0.0429

# 6. Current score

0.28616

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.28616) has done: 'I fix the import/runtime issues by removing the unsupported `imgaug` dependency (it’s not installed here and also triggers protobuf-related import errors) and replacing it with a minimal, equivalent augmentation routine using `numpy` flips/rotations so the training pipeline stays the same. I also shift the zip extraction to `/kaggle/working/` using robust `os.path.join` so the image folders are found reliably. To improve the score toward your target (lower RMSE), I ensure the model trains on the augmented data (same intent as before) and I fix inference to run the full autoencoder (not decoder(encoder(...)) with extra numpy conversions) plus clip predictions to [0,1] before writing. Finally, I keep the submission format identical (`id,value`) and write `submission.csv` in the working directory.'
- What this solution (achieved 0.28616) has done: 'I fix the protobuf-related TensorFlow import crash by forcing the pure-Python protobuf implementation before importing TensorFlow (a common Kaggle env incompatibility causing `MessageFactory.GetPrototype` errors). Then I fix the augmentation bug that swaps height/width during `rot90` so augmented arrays can concatenate cleanly (this also restores the intended training data expansion, improving score toward your target). Finally, I ensure the training/inference cells don’t cascade-fail (so `processed_train`/`history` always exist) and keep the submission writer producing a correctly formatted `submission.csv` with values clipped to `[0,1]`.'
- What this solution (achieved 0.28616) has done: 'The timeout is dominated by Python-level loops and repeated work in image I/O + submission generation: nested per-pixel loops over ~5.8M pixels and per-image single-item model calls. I keep the same model and training semantics, but speed up data loading with preallocated arrays and deterministic parallel OpenCV decoding, speed up augmentation rotations by using OpenCV for vectorized resize without per-image Python loops, and make inference run in batches. Finally, I generate the submission in a fully vectorized way (no per-pixel Python loops) by reading IDs from `sampleSubmission.csv` and filling values using NumPy indexing, which is exactly equivalent to the required id/value mapping.'
- What this solution (achieved 0.28616) has done: 'The timeout is dominated by (1) doing 5 full-image augmentations via expensive per-image OpenCV resizing loops and (2) building the 5.8M-row submission by parsing `id` strings and doing Python-level grouping. I keep the same model, loss, and training loop, but make augmentation provably equivalent and much faster by using pure NumPy operations for rotations/flips (no resize needed because 180°/90°/270° preserve shape via axis transposes). I also keep the exact same submission semantics but make it fast by precomputing a dense `(img,row,col)->flat_index` map once, resizing predictions once per image, then filling the output values via vectorized indexing (no string split, no dict lookup, no Python while-grouping). Finally, I remove expensive plotting/summaries that are not required for training or submission.'
- What this solution (achieved 0.28616) has done: 'The timeout is dominated by (1) creating very large augmented arrays via repeated `np.concatenate`, which is quadratic and memory-heavy, and (2) building the 5.7M-row submission by reading the huge sample CSV and looping in Python while doing redundant image-size reads/resizes. I preserve the exact model and training semantics, but make augmentation O(N) by preallocating and filling, and make the tf.data pipeline more efficient by avoiding `buffer_size=len(...)` and using `cache` after batching to reduce peak memory. For submission, I avoid loading `sampleSubmission.csv` entirely (ids are deterministic) and generate the `id` column in a streaming writer while filling `value` per image in the same order, eliminating the 5.7M-row pandas read and large intermediate DataFrame. All changes are correctness-preserving (same images, same augmentations, same training objective/epochs/callbacks, same prediction/resizing semantics and output format/order).'
- What this solution (achieved 0.28616) has done: 'The timeout is dominated by (1) training overhead from feeding large NumPy arrays through a cached `tf.data` pipeline and (2) extremely slow Python-level loops when writing the 5.7M-row submission. I keep the same model, augmentations, loss, and training semantics, but speed up the data pipeline by caching/shuffling/batching in a more efficient order and by using `tf.data` options to reduce overhead without changing results. The biggest win is replacing the nested per-pixel Python write loop with vectorized ID generation (cached once) and fast chunked CSV writing, producing identical `id,value` rows. I also avoid unnecessary extra disk reads during submission generation.'
- What this solution (achieved 0.28616) has done: 'The timeout is dominated by (1) huge in-memory augmentation (6× the training set) and (2) slow submission writing that repeatedly builds 30 pandas DataFrames and concatenates strings per-image. I keep the same model, loss, optimizer, and effective training data (original + 5 deterministic augmentations), but move augmentation into the `tf.data` pipeline so it’s generated on-the-fly without materializing a 6× array. I also make image loading faster by decoding directly to grayscale at the target size (eliminating extra color conversion + resize work), and I rewrite submission generation to stream rows using a single precomputed id-prefix column and chunked writes (same exact ids/values). These changes are equivalence-preserving (same pixels, same augmentations, same training semantics) and cut both peak memory and Python overhead significantly.'
- What this solution (achieved 0.28616) has done: 'I fix the TensorFlow/protobuf crash by removing the forced pure-Python protobuf override (it causes the `MessageFactory.GetPrototype` error in this environment) and importing TensorFlow normally. Then I fix the training-time shape mismatch by correcting the `rot90/rot270` augmentations in `tf.data` so they return the same `(420,540,1)` shape as the base images (no swapped height/width). Finally, I fix submission generation by building the `id` strings without NumPy string-ufunc pitfalls, while keeping the same required `id,value` format and clipping predictions to `[0,1]` before writing `submission.csv`.'
- What this solution (achieved 0.28616) has done: 'The timeout is most likely dominated by (1) an extremely slow training loop because each step runs eager Python `Model.call` without `tf.function` compilation, and (2) heavy input overhead from concatenating many datasets and using costly `tf.image.resize` in some augmentation branches. I keep the exact same model architecture, optimizer/loss/metrics, epochs/callbacks, and the same augmentation semantics (same 6 variants), but compile the model with XLA/JIT and wrap `call` with `@tf.function` to make training run as a compiled graph. I also remove the unnecessary `tf.image.resize` calls after 90/270-degree rotations (the shapes are unchanged for 420x540 after rot90/rot270), and simplify the augmentation dataset construction by using a single `map` that branches on an integer “mode” (still deterministic and exactly equivalent), which reduces dataset graph overhead. Finally, submission generation is made vectorized (no Python loop creating ~5.8M strings), which can save minutes and avoids pandas chunk overhead while preserving identical output format.'
- What this solution (achieved 0.28616) has done: 'I fix the TensorFlow import crash by setting a safe protobuf environment flag before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` failure in this Kaggle image. Then I fix the training shape error by resizing the 90°/270° rotated tensors back to `(420, 540)` inside the augmentation switch (this keeps the same augmentation intent while making shapes compatible). Finally, I fix submission generation by avoiding NumPy string-ufunc addition (which is erroring) and instead build IDs with vectorized Pandas string operations, while keeping the exact required `id,value` format and clipping predictions to `[0,1]`.'

# 9. Code solution

## === cell 0
import os, zipfile

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks

sns.set_style("darkgrid")

np.random.seed(19)
tf.random.set_seed(19)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    cv2.setNumThreads(0)
except Exception:
    pass

try:
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

try:
    cpu_cnt = os.cpu_count() or 2
    tf.config.threading.set_intra_op_parallelism_threads(min(8, cpu_cnt))
    tf.config.threading.set_inter_op_parallelism_threads(min(2, cpu_cnt))
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path_zip = "../input/denoising-dirty-documents/"
path = "/kaggle/working/"

for zname in ["train.zip", "test.zip", "train_cleaned.zip", "sampleSubmission.csv.zip"]:
    zpath = os.path.join(path_zip, zname)
    expected = os.path.join(path, zname.replace(".zip", ""))
    expected_csv = os.path.join(path, "sampleSubmission.csv")
    if zname == "sampleSubmission.csv.zip":
        if os.path.exists(expected_csv):
            continue
    else:
        if os.path.isdir(expected):
            continue
    with zipfile.ZipFile(zpath, "r") as zip_ref:
        zip_ref.extractall(path)

train_dir = os.path.join(path, "train")
train_cleaned_dir = os.path.join(path, "train_cleaned")
test_dir = os.path.join(path, "test")

train_img = sorted(os.listdir(train_dir))
train_cleaned_img = sorted(os.listdir(train_cleaned_dir))
test_img = sorted(os.listdir(test_dir))

print("Counts:", len(train_img), len(train_cleaned_img), len(test_img))




## === cell 2
class config:
    IMG_SIZE = (420, 540)


imgs = [
    cv2.imread(os.path.join(train_dir, f)) for f in sorted(os.listdir(train_dir))[:10]
]
print(
    "Median Dimensions (sample):",
    np.median([img.shape[0] for img in imgs]),
    np.median([img.shape[1] for img in imgs]),
)
del imgs




## === cell 3
def process_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # (H,W) uint8
    img = cv2.resize(img, config.IMG_SIZE[::-1], interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    img = img.reshape((*config.IMG_SIZE, 1))
    return img




## === cell 4
from concurrent.futures import ThreadPoolExecutor


def _load_images_to_array(file_list, folder, n_workers=8):
    n = len(file_list)
    arr = np.empty((n, config.IMG_SIZE[0], config.IMG_SIZE[1], 1), dtype=np.float32)

    def _one(i_f):
        i, f = i_f
        arr[i] = process_image(os.path.join(folder, f))

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(
            tqdm(
                ex.map(_one, enumerate(file_list)),
                total=n,
                desc=f"Loading {os.path.basename(folder)}",
            )
        )
    return arr


train = _load_images_to_array(
    train_img, train_dir, n_workers=min(8, (os.cpu_count() or 2))
)
train_cleaned = _load_images_to_array(
    train_cleaned_img, train_cleaned_dir, n_workers=min(8, (os.cpu_count() or 2))
)
test = _load_images_to_array(
    test_img, test_dir, n_workers=min(8, (os.cpu_count() or 2))
)

train.shape, train_cleaned.shape, test.shape




## === cell 5
def augment_pipeline(pipeline, images, seed=19):
    n = images.shape[0]
    k = len(pipeline) + 1
    out = np.empty((n * k, *images.shape[1:]), dtype=images.dtype)
    out[:n] = images
    offset = n
    for aug_fn in pipeline:
        out[offset : offset + n] = aug_fn(images)
        offset += n
    return out


def aug_rot90(images):
    return np.rot90(images, k=1, axes=(1, 2)).copy()


def aug_rot180(images):
    return np.rot90(images, k=2, axes=(1, 2)).copy()


def aug_rot270(images):
    return np.rot90(images, k=3, axes=(1, 2)).copy()


def aug_hflip(images):
    return images[:, :, ::-1, :].copy()


def aug_vflip(images):
    return images[:, ::-1, :, :].copy()




## === cell 6
pipeline = [aug_rot90, aug_rot180, aug_rot270, aug_hflip, aug_vflip]

print(
    "Will generate augmented variants on-the-fly via tf.data (no 6x materialization)."
)




## === cell 7
class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(*config.IMG_SIZE, 1)),
                layers.DepthwiseConv2D(
                    (3, 3), depth_multiplier=32, activation="relu", padding="same"
                ),
                layers.DepthwiseConv2D(
                    (3, 3), depth_multiplier=2, activation="relu", padding="same"
                ),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Dropout(0.5),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.DepthwiseConv2D(
                    (3, 3), depth_multiplier=1, activation="relu", padding="same"
                ),
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.UpSampling2D((2, 2)),
                layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same"),
            ]
        )

    @tf.function(jit_compile=True)
    def call(self, x, training=False):
        encoded = self.encoder(x, training=training)
        decoded = self.decoder(encoded, training=training)
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
    monitor="loss", factor=0.8, patience=5, min_lr=1e-15, mode="min", verbose=1
)

base_ds = tf.data.Dataset.from_tensor_slices((train, train_cleaned))

H0, W0 = config.IMG_SIZE
target_hw = tf.constant([H0, W0], dtype=tf.int32)


def _apply_aug(mode, x, y):
    def _id():
        return x, y

    def _r90():
        x2 = tf.image.rot90(x, k=1)
        y2 = tf.image.rot90(y, k=1)
        x2 = tf.image.resize(x2, target_hw, method="area", antialias=False)
        y2 = tf.image.resize(y2, target_hw, method="area", antialias=False)
        return x2, y2

    def _r180():
        return tf.image.rot90(x, k=2), tf.image.rot90(y, k=2)

    def _r270():
        x2 = tf.image.rot90(x, k=3)
        y2 = tf.image.rot90(y, k=3)
        x2 = tf.image.resize(x2, target_hw, method="area", antialias=False)
        y2 = tf.image.resize(y2, target_hw, method="area", antialias=False)
        return x2, y2

    def _hf():
        return tf.image.flip_left_right(x), tf.image.flip_left_right(y)

    def _vf():
        return tf.image.flip_up_down(x), tf.image.flip_up_down(y)

    x2, y2 = tf.switch_case(
        branch_index=mode,
        branch_fns=[_id, _r90, _r180, _r270, _hf, _vf],
    )
    x2 = tf.ensure_shape(x2, [H0, W0, 1])
    y2 = tf.ensure_shape(y2, [H0, W0, 1])
    return x2, y2


num_parallel = tf.data.AUTOTUNE

modes = tf.data.Dataset.from_tensor_slices(
    tf.constant([0, 1, 2, 3, 4, 5], dtype=tf.int32)
)
ds = modes.flat_map(
    lambda m: base_ds.map(
        lambda x, y: _apply_aug(m, x, y),
        num_parallel_calls=num_parallel,
        deterministic=True,
    )
)

processed_len = train.shape[0] * (len(pipeline) + 1)
shuffle_buf = min(int(processed_len), 2048)

ds = ds.cache()
ds = ds.shuffle(buffer_size=shuffle_buf, seed=19, reshuffle_each_iteration=True)
ds = ds.batch(12, drop_remainder=False)
ds = ds.prefetch(tf.data.AUTOTUNE)

options = tf.data.Options()
options.experimental_deterministic = True
ds = ds.with_options(options)

history = autoencoder.fit(
    ds,
    callbacks=[es, rlp],
    epochs=500,
    verbose=2,
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
UnimplementedError                        Traceback (most recent call last)
/tmp/ipykernel_11/2789019027.py in <cell line: 0>()
     76 ds = ds.with_options(options)
     77 
---> 78 history = autoencoder.fit(
     79     ds,
     80     callbacks=[es, rlp],

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

UnimplementedError: Graph execution error:

Detected at node gradients/sequential_1/max_pooling2d_1/MaxPool2d_grad/MaxPoolGrad defined at (most recent call last):
<stack traces unavailable>
GPU MaxPool gradient ops do not yet have a deterministic XLA implementation.
	 [[{{node gradients/sequential_1/max_pooling2d_1/MaxPool2d_grad/MaxPoolGrad}}]]
	tf2xla conversion failed while converting __inference___backward_call_1153_1430[]. Run with TF_DUMP_GRAPH_PREFIX=/path/to/dump/dir and --vmodule=xla_compiler=2 to obtain a dump of the compiled functions.
	 [[StatefulPartitionedCall/PartitionedCall]] [Op:__inference_multi_step_on_iterator_2955]

## === cell 9
try:
    del history
except Exception:
    pass




## === cell 10
pass




## === cell 11
pass




## === cell 12
batch_size = 4
decoded_test = autoencoder.predict(test, batch_size=batch_size, verbose=0).astype(
    np.float32
)  # (N,420,540,1)

test_ids = np.array([int(f[:-4]) for f in test_img], dtype=np.int32)
H0, W0 = config.IMG_SIZE

sub_path = os.path.join(path, "submission.csv")

rows = np.arange(1, H0 + 1, dtype=np.int32)
cols = np.arange(1, W0 + 1, dtype=np.int32)
rr, cc = np.meshgrid(rows, cols, indexing="ij")

rc = (
    pd.Series(rr.ravel(), dtype="int32").astype(str)
    + "_"
    + pd.Series(cc.ravel(), dtype="int32").astype(str)
).to_numpy(
    dtype=object
)  # (pix,)

prefix = (pd.Series(test_ids, dtype="int32").astype(str) + "_").to_numpy(dtype=object)[
    :, None
]  # (n_test,1)
ids_all = (prefix + rc[None, :]).reshape(-1)

vals_all = np.clip(decoded_test[..., 0], 0.0, 1.0).reshape(-1).astype(np.float32)

with open(sub_path, "w", encoding="utf-8") as f:
    f.write("id,value\n")
    chunk = 750_000
    for start in range(0, ids_all.shape[0], chunk):
        end = min(start + chunk, ids_all.shape[0])
        block_ids = ids_all[start:end]
        block_vals = vals_all[start:end]
        lines = block_ids + "," + np.char.mod("%.8f", block_vals)
        f.write("\n".join(lines.tolist()))
        f.write("\n")

print(f"Results saved to {sub_path}!")
print("Submission preview:")
with open(sub_path, "r", encoding="utf-8") as f:
    for _ in range(5):
        print(f.readline().strip())
