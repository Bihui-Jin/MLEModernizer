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

0.02838

# 6. Current score

0.28616

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28616) has done: 'I fix the import-time crash coming from `imgaug` (it’s incompatible with the current `protobuf`/TF stack in this environment), which is why `iaa` never gets defined and all later augmentation/training cells fail. To keep the core training and model logic intact, I replace the `imgaug` augmentation pipeline with an equivalent, minimal NumPy-based augmentation that performs the same flips/90-degree rotations. I also add a small safety fix to ensure predictions are clipped to `[0, 1]` before writing the submission (score-neutral but prevents invalid values). Finally, I keep the same file paths and ensure `submission.csv` is produced with the required `id,value` columns.'
- What this solution (achieved 0.28616) has done: 'I fix the TensorFlow/protobuf import-time crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error in this Kaggle environment. Then I fix the augmentation bug that swaps height/width during `np.rot90`, which currently causes shape mismatches and prevents training from running. After that, the training and inference run unchanged, and I keep the submission writing logic but ensure values remain clipped to `[0,1]` and the CSV is written as `submission.csv` with the required `id,value` columns. These changes are execution-unblocking and should materially improve the score versus the current broken/incorrect augmentation pipeline.'
- What this solution (achieved 0.28616) has done: 'I fix the TensorFlow/protobuf crash by switching to the C++ protobuf implementation (the `GetPrototype` AttributeError happens when TF ends up using the pure-Python protobuf runtime). Next, I fix the augmentation rotation bug: `np.rot90` swaps height/width for 90/270-degree rotations, so we must resize back to the expected `(420,540)` shape to keep the same training pipeline semantics and unblock training. Finally, I keep the model/training loop intact but ensure inference writes a correctly formatted `submission.csv` with values clipped to `[0,1]` (already present) and with robust id construction aligned to the original image shapes.'
- What this solution (achieved 0.28616) has done: 'I fix the TensorFlow/protobuf import-time crash by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow (this resolves the `MessageFactory.GetPrototype` error in this environment). Then I keep your existing data loading, augmentation, model, and training logic intact, only adding small robustness tweaks (safe directory creation and consistent path joining) that don’t change the learning semantics. Finally, I ensure the submission is always written as `submission.csv` with the required `id,value` columns and values clipped to `[0,1]` (already present) so Kaggle accepts it. These changes are execution-unblocking and should let the model actually train/infer properly, moving RMSE down toward your target.'
- What this solution (achieved 0.28616) has done: 'You’re currently crashing at import time because TensorFlow 2.18 expects a newer protobuf API (`MessageFactory.GetPrototype`), and forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` triggers the incompatible pure-Python protobuf runtime. I remove that override and instead force the C++ protobuf runtime before importing TensorFlow, which fixes the import error while keeping the rest of your pipeline identical. Then I make sure the extracted data path is consistent (`/kaggle/working/` as you already use) and keep your augmentation/model/training logic unchanged. Finally, I keep your submission generation logic but add a small safety sort by numeric image id to ensure stable ordering and always write `submission.csv` with the required `id,value` columns.'
- What this solution (achieved 0.28616) has done: 'I fix the TensorFlow import crash by removing the invalid `"cpp"` protobuf override and instead forcing the compatible pure-Python protobuf runtime before importing TensorFlow. This unblocks `tf`, `Model`, `layers`, and `callbacks` so the rest of your pipeline (augmentation, autoencoder architecture, training loop, and inference) can run unchanged end-to-end. I also remove the earlier “NameError cascade” by ensuring those imports succeed and by keeping cell order consistent. Finally, I keep your submission logic intact but make it robust and faster by vectorizing the pixel-id/value melting per image (same semantics/values), while still writing an accepted `submission.csv` with `id,value` and values clipped to `[0,1]`.'
- What this solution (achieved 0.28616) has done: 'I fix the immediate runtime crash by removing the protobuf “python” override that’s incompatible with TensorFlow 2.18 in this environment, letting TF import normally so training/inference can run. Then I fix a logic bug where test filenames are sorted but the `test` numpy array is not reordered accordingly, which misaligns inputs to ids and severely hurts RMSE; reordering `test` to match `test_img` is a minimal, semantics-preserving correction. I also make submission generation robust by using vectorized id construction (same ids/values) and by ensuring the output is exactly `submission.csv` with `id,value` and values clipped to `[0,1]`. These changes should move the score substantially toward the target while keeping your model/augmentation/training approach intact.'
- What this solution (achieved 0.28616) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the Python protobuf implementation *before* importing TensorFlow, which is a minimal environment-compatibility change and keeps your model/training logic intact. Then I fix the submission-id construction error by building the `id` strings via NumPy `char` operations (so string concatenation works elementwise and stays fast), preserving the exact required `image_row_col` format. I also keep your existing “reorder `test` to match sorted filenames” logic (important for score) and ensure the submission is written as `submission.csv` with `id,value` and predictions clipped to `[0,1]`. No changes are made to the model architecture, loss, training loop, or augmentation semantics.'
- What this solution (achieved 0.28616) has done: 'The timeout is dominated by (1) very slow CSV generation that builds ~5.8M string IDs in Python loops and stores them in huge Python lists, and (2) avoidable per-image overhead during prediction and resizing. I keep the same model, training loop, augmentations, and prediction semantics, but vectorize and stream the submission creation: precompute the row/col id suffixes once per unique image shape, write the output CSV incrementally, and batch model inference so TensorFlow runs fewer calls. I also avoid unnecessary full-image reads during submission (use cv2 to get shape only), and enable `tf.data` prefetching for training without changing epochs/batch size/early stopping behavior.'
- What this solution (achieved 0.28616) has done: 'The timeout is dominated by (1) repeated full-array concatenations during augmentation, (2) slow single-threaded image decoding/resizing loops, and (3) extremely expensive per-image Pandas `to_csv` appends for ~5.8M rows. I keep the exact same model, loss, training loop, and augmentation semantics, but make augmentation preallocated (no repeated `np.concatenate`), load/resize images in parallel with `cv2.IMREAD_GRAYSCALE`, and write the submission via fast buffered string/NumPy output without per-image DataFrames. These changes are provably equivalent (same pixels produced and same ids/values), but remove large constant-factor overhead and avoid repeated work.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import zipfile, cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks, utils

sns.set_style("darkgrid")

np.random.seed(19)
tf.random.set_seed(19)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass




## === cell 1
path_zip = "../input/denoising-dirty-documents/"
path = "/kaggle/working/"

os.makedirs(path, exist_ok=True)

with zipfile.ZipFile(os.path.join(path_zip, "train.zip"), "r") as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(os.path.join(path_zip, "test.zip"), "r") as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(os.path.join(path_zip, "train_cleaned.zip"), "r") as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(
    os.path.join(path_zip, "sampleSubmission.csv.zip"), "r"
) as zip_ref:
    zip_ref.extractall(path)

train_img = sorted(os.listdir(os.path.join(path, "train")))
train_cleaned_img = sorted(os.listdir(os.path.join(path, "train_cleaned")))
test_img = sorted(os.listdir(os.path.join(path, "test")))




## === cell 2
class config:
    IMG_SIZE = (420, 540)


_subset_files = sorted(os.listdir(os.path.join(path, "train")))[:10]
imgs = [cv2.imread(os.path.join(path, "train", f)) for f in _subset_files]
print(
    "Median Dimensions (subset of 10):",
    np.median([len(img) for img in imgs]),
    np.median([len(img[0]) for img in imgs]),
)
del imgs, _subset_files




## === cell 3
def process_image(path_):
    img = cv2.imread(path_, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, config.IMG_SIZE[::-1], interpolation=cv2.INTER_LINEAR)
    img = img.astype("float32") / 255.0
    img = np.reshape(img, (*config.IMG_SIZE, 1))
    return img




## === cell 4
train_files = sorted(os.listdir(os.path.join(path, "train")))
train_cleaned_files = sorted(os.listdir(os.path.join(path, "train_cleaned")))
test_files = sorted(os.listdir(os.path.join(path, "test")))

train = np.empty((len(train_files), *config.IMG_SIZE, 1), dtype=np.float32)
train_cleaned = np.empty(
    (len(train_cleaned_files), *config.IMG_SIZE, 1), dtype=np.float32
)
test = np.empty((len(test_files), *config.IMG_SIZE, 1), dtype=np.float32)

from concurrent.futures import ThreadPoolExecutor


def _load_into_array(files, folder, out_array, max_workers=8):
    full_paths = [os.path.join(folder, f) for f in files]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, img in enumerate(ex.map(process_image, full_paths)):
            out_array[i] = img


_load_into_array(
    train_files,
    os.path.join(path, "train"),
    train,
    max_workers=min(8, (os.cpu_count() or 2)),
)
_load_into_array(
    train_cleaned_files,
    os.path.join(path, "train_cleaned"),
    train_cleaned,
    max_workers=min(8, (os.cpu_count() or 2)),
)
_load_into_array(
    test_files,
    os.path.join(path, "test"),
    test,
    max_workers=min(8, (os.cpu_count() or 2)),
)




## === cell 5
train.shape, train_cleaned.shape, test.shape




## === cell 6
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




## === cell 7
def _resize_back(images, target_hw):
    out = np.empty(
        (images.shape[0], target_hw[0], target_hw[1], images.shape[-1]),
        dtype=images.dtype,
    )
    for i in range(images.shape[0]):
        out[i, ..., 0] = cv2.resize(
            images[i, ..., 0],
            (target_hw[1], target_hw[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return out


def _rot90(images, k):
    rotated = np.rot90(images, k=k, axes=(1, 2)).copy()
    if rotated.shape[1] != config.IMG_SIZE[0] or rotated.shape[2] != config.IMG_SIZE[1]:
        rotated = _resize_back(rotated, config.IMG_SIZE)
    return rotated


def _hflip(images):
    return np.flip(images, axis=2).copy()  # flip width


def _vflip(images):
    return np.flip(images, axis=1).copy()  # flip height


def augment_pipeline(pipeline, images, seed=19):
    n0 = images.shape[0]
    nsteps = len(pipeline)
    out = np.empty((n0 * (nsteps + 1),) + images.shape[1:], dtype=images.dtype)
    out[:n0] = images
    write_pos = n0
    for step in pipeline:
        temp = np.asarray(step(images), dtype=images.dtype)
        if temp.shape[1:] != images.shape[1:]:
            raise ValueError(
                f"Augmentation produced shape {temp.shape} but expected {images.shape}"
            )
        out[write_pos : write_pos + n0] = temp
        write_pos += n0
    return out




## === cell 8
rotate90 = lambda x: _rot90(x, 1)
rotate180 = lambda x: _rot90(x, 2)
rotate270 = lambda x: _rot90(x, 3)
hflip = lambda x: _hflip(x)
vflip = lambda x: _vflip(x)




## === cell 9
pipeline = [rotate90, rotate180, rotate270, hflip, vflip]




## === cell 10
processed_train = augment_pipeline(pipeline, train)
processed_train_cleaned = augment_pipeline(pipeline, train_cleaned)

processed_train.shape, processed_train_cleaned.shape




## === cell 11
class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(*config.IMG_SIZE, 1)),
                layers.Conv2D(96, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(192, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Dropout(0.5),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Conv2D(192, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(96, (3, 3), activation="relu", padding="same"),
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




## === cell 12
es = callbacks.EarlyStopping(
    monitor="loss", patience=30, verbose=1, restore_best_weights=True
)

batch_size = 12
train_ds = (
    tf.data.Dataset.from_tensor_slices((processed_train, processed_train_cleaned))
    .shuffle(buffer_size=len(processed_train), seed=19, reshuffle_each_iteration=True)
    .batch(batch_size, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

history = autoencoder.fit(
    train_ds,
    callbacks=[es],
    epochs=500,
)




## === cell 13
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).plot(ax=ax)
del history




## === cell 14
autoencoder.encoder.summary()
autoencoder.decoder.summary()




## === cell 15
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




## === cell 16
test_img = sorted(test_img, key=lambda x: int(os.path.splitext(x)[0]))

test_files_loaded = sorted(os.listdir(os.path.join(path, "test")))
idx_map = {fn: i for i, fn in enumerate(test_files_loaded)}
test = test[[idx_map[fn] for fn in test_img]]


def _get_hw_fast(filepath):
    im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return im.shape  # (H, W)


_suffix_cache = {}  # (H,W) -> np.ndarray of strings like "_r_c"


def _get_suffixes(H, W):
    key = (H, W)
    if key in _suffix_cache:
        return _suffix_cache[key]
    r = np.repeat(np.arange(1, H + 1, dtype=np.int32), W).astype(str)
    c = np.tile(np.arange(1, W + 1, dtype=np.int32), H).astype(str)
    suffix = np.char.add(np.char.add("_", r), np.char.add("_", c))  # "_r_c"
    _suffix_cache[key] = suffix
    return suffix


out_path = "submission.csv"
with open(out_path, "w", buffering=16 * 1024 * 1024) as f:
    f.write("id,value\n")

    batch = 8  # does not change predictions, only groups inference calls
    n = len(test_img)

    hw_list = []
    suffix_list = []
    imgid_list = []
    for fname in test_img:
        file = os.path.join(path, "test", fname)
        imgid = int(os.path.splitext(fname)[0])
        H, W = _get_hw_fast(file)
        hw_list.append((H, W))
        suffix_list.append(_get_suffixes(H, W))
        imgid_list.append(str(imgid))

    for start in tqdm(range(0, n, batch)):
        end = min(start + batch, n)
        batch_imgs = test[start:end]
        decoded_batch = autoencoder(batch_imgs, training=False).numpy()  # (B,420,540,1)

        for j in range(end - start):
            H, W = hw_list[start + j]
            suffix = suffix_list[start + j]
            imgid_str = imgid_list[start + j]

            decoded_img = decoded_batch[j, ..., 0]  # (420,540)
            preds_reshaped = cv2.resize(
                decoded_img, (W, H), interpolation=cv2.INTER_LINEAR
            )
            preds_reshaped = (
                np.clip(preds_reshaped, 0.0, 1.0).astype(np.float32).reshape(-1)
            )

            ids = np.char.add(imgid_str, suffix)

            vals = np.char.mod("%.7f", preds_reshaped)
            lines = np.char.add(np.char.add(ids, ","), vals)
            f.write("\n".join(lines.tolist()))
            f.write("\n")

print(f"Results saved to {out_path}!")
