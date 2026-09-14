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

0.02953

# 6. Current score

0.28616

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.28616) has done: 'I fix the runtime crash caused by `imgaug`/protobuf incompatibility by removing that dependency (the augmentation pipeline isn’t used for training anyway), which unblocks execution. I also make the dataset path handling robust to Kaggle’s directory layout so the zips are found and extracted correctly. Finally, I correct the submission generation to match the required `id,value` order and pixel intensity scaling by writing predictions in exactly the same order as `sampleSubmission.csv` (avoids subtle ordering mismatches that can severely hurt RMSE). These changes keep the core model/training logic intact while addressing the main source of the poor score and the existing runtime errors.'
- What this solution (achieved 0.28616) has done: 'I fix the protobuf/TensorFlow crash by forcing TensorFlow to use the pure-Python protobuf implementation before importing it (this resolves the `MessageFactory.GetPrototype` error in Kaggle’s environment without changing model logic). I also make image reading safer by explicitly using grayscale reads and ensuring `cv2.imread` failures are caught early, preventing silent shape/type issues that can corrupt training/predictions. Finally, I keep your submission-generation logic but ensure it always writes a valid `submission.csv` with the correct `id,value` columns and clipped `[0,1]` predictions in the exact `sampleSubmission.csv` order (score-critical alignment). No changes to the model architecture, loss, or training loop semantics are introduced.'
- What this solution (achieved 0.28616) has done: 'I fix the current runtime crash by ensuring TensorFlow imports use a protobuf version that’s compatible with TF 2.18 in Kaggle, without changing your model/training logic. Specifically, I remove the unsupported `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` workaround (it can trigger the `MessageFactory.GetPrototype` failure with newer protobuf) and instead set safe TF runtime env flags before importing TensorFlow. I also make the submission-generation step much faster and less error-prone by vectorizing the pixel lookup per image (same exact semantics/ordering as `sampleSubmission.csv`, just avoiding a 5.7M-iteration Python loop that risks timeouts). These changes are correctness/stability-focused and should also materially improve score by ensuring the pipeline runs fully and predictions align perfectly with the required `id` ordering.'
- What this solution (achieved 0.28616) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow (this is the direct cause of the `MessageFactory.GetPrototype` error in this environment). Then I fix submission id parsing by avoiding `np.char.split` (which is fragile with mixed/object arrays) and instead using Pandas vectorized `str.split`, ensuring correct alignment with `sampleSubmission.csv`. Finally, I keep your model/training exactly the same but make inference/submission generation robust and deterministic, and always write a valid `submission.csv` with `id,value` and values clipped to `[0,1]`.'
- What this solution (achieved 0.28616) has done: 'The timeout is dominated by (1) slow CPU input pipeline (loading all images into Python lists) and especially (2) extremely expensive submission construction that loops over image ids and repeatedly does boolean masking over ~5.8M rows. I keep the same model, loss, training loop semantics, and exact prediction math, but switch data loading to a cached `tf.data` pipeline (still reading the same files and producing identical resized float32 tensors), and rewrite submission generation to do one forward pass over the whole test set and then fill the 5.8M values in a single vectorized gather (no per-image mask scans). I also avoid unnecessary plotting in the timed environment and ensure deterministic behavior is preserved. These changes are provably equivalent in outputs (up to negligible FP differences) but cut runtime by eliminating O(num_images * num_pixels) pandas/numpy masking and reducing Python overhead.'
- What this solution (achieved 0.28598) has done: 'I fix the immediate runtime crash by removing the protobuf “python” implementation override that is incompatible with TensorFlow 2.18/protobuf 6 in this Kaggle environment (it triggers `MessageFactory.GetPrototype` missing). Then I keep the exact same model/training/inference logic, but ensure TensorFlow imports cleanly and deterministically by setting only safe TF env flags before importing TF. Finally, I keep your submission-building semantics unchanged (same `sampleSubmission.csv` order, same resizing/back-mapping), ensuring the pipeline always completes and writes a valid `submission.csv` with `id,value` and clipped `[0,1]` predictions.'
- What this solution (achieved 0.26779) has done: 'I fix the TensorFlow import crash caused by the TensorFlow 2.18 + protobuf 6 incompatibility (`MessageFactory.GetPrototype`), by forcing protobuf to use the pure-Python implementation *before* importing TensorFlow (this is a runtime fix and does not change model logic). I also keep your existing training/inference/submission logic intact, but make sure the environment variables are set early enough and that the submission is always written to `/kaggle/working/submission.csv`. These changes are primarily to unblock execution end-to-end; the score improvement is expected because the model actually train/infer and the submission alignment remains correct. No architecture, loss, or training-loop semantic changes are introduced.'
- What this solution (achieved 0.28616) has done: 'The crash is coming from forcing the pure-Python protobuf implementation (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`) with TensorFlow 2.18 + protobuf 6, which triggers the `MessageFactory.GetPrototype` attribute error before any training happens. I remove that override (and keep only safe TF env flags) so TensorFlow imports cleanly and the notebook runs end-to-end. Then, to move the RMSE score materially toward your target (your current 0.26779 is far outside the ±10% band around 0.02953), I fix a key logic issue: the submission expects original-size pixels, but the current code trains on resized images (420×540) and then upscales predictions back—this mismatch heavily hurts RMSE. The minimal, core-logic-preserving change is to train/infer at the competition’s native image size (using the sampleSubmission max row/col), while keeping the same autoencoder architecture/training loop/loss; submission generation remains in the exact `sampleSubmission.csv` order.'
- What this solution (achieved 0.28616) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by setting a protobuf environment override *before* importing TensorFlow, which is the direct runtime blocker. Then I keep your model, loss, and training loop intact, but fix a key shape/semantic issue: with `MaxPooling2D` + `UpSampling2D`, odd image sizes (from sampleSubmission) can cause output to be 2×floor(H/2) and 2×floor(W/2), silently misaligning labels/pixels and badly hurting RMSE; I pad/crop predictions to exactly `config.IMG_SIZE` (score-improving, architecture unchanged). Finally, I make submission pixel gathering fully vectorized (same exact ordering as `sampleSubmission.csv`) to avoid timeouts and ensure correct `id,value` alignment.'
- What this solution (achieved 0.28616) has done: 'I fix the TensorFlow/protobuf runtime crash by removing the incompatible `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override that triggers `MessageFactory.GetPrototype` in this TF 2.18 + protobuf 6 environment, while keeping the rest of the pipeline unchanged. Then I fix a core logic issue that is severely hurting RMSE: the decoder output is larger than the target due to `UpSampling2D` without a corresponding downsampling in the decoder path, so your current `_pad_or_crop_to` always crops away content (especially for odd H/W), misaligning predictions vs labels; I minimally correct this by cropping/padding the model output *inside* `call()` to exactly `config.IMG_SIZE` (architecture/loss/training loop remain the same, just enforcing correct output shape). Finally, I keep the same submission generation semantics (sampleSubmission order) but ensure inference uses batched `predict` to avoid memory spikes/timeouts and always writes `/kaggle/working/submission.csv` with correct `id,value` and `[0,1]` clipping.'
- What this solution (achieved 0.28616) has done: 'The crash happens before any training due to a TensorFlow 2.18 + protobuf 6 incompatibility that raises `MessageFactory.GetPrototype`; the minimal reliable fix in Kaggle is to force protobuf to use the pure-Python implementation *before* importing TensorFlow. After unblocking execution, the main score issue (0.286 vs target 0.0295, lower-is-better) is almost certainly incorrect pixel scaling: `tf.io.decode_png` yields values in `[0,65535]` for 16-bit PNGs, but the code divides by 255, which destroys calibration and inflates RMSE. I keep the exact same model architecture, loss, and training loop, but fix preprocessing to correctly normalize based on the decoded dtype/max value and ensure the same scaling is used for train/test. Submission generation stays aligned to `sampleSubmission.csv` order and remains fully vectorized.'
- What this solution (achieved 0.28616) has done: 'The main timeout drivers are (1) materializing all images into NumPy arrays via `as_numpy_iterator()` (forces a full graph-to-host transfer and duplicates data already cached by `tf.data`) and (2) creating the 5.8M-row submission by reading/parsing `sampleSubmission.csv` and doing Python-level mapping. The changes below keep the exact same model, loss, training loop, and evaluation semantics, but avoid redundant dataset-to-NumPy conversion, cache decoded/resized images to disk once (so repeated runs don’t re-decode), and generate the submission IDs directly in a fully vectorized way from image dimensions and test image ids (no CSV parse, no Python dict/`fromiter`). These are correctness-preserving transformations: they don’t change inputs, targets, architecture, or optimization settings, and submission IDs match Kaggle’s required format.'
- What this solution (achieved 0.28616) has done: 'The main bottlenecks are (1) extremely slow submission writing due to per-image pandas object/string construction inside a Python loop, and (2) suboptimal tf.data input pipeline settings that leave I/O/CPU underutilized during training/prediction. I keep the exact same model, loss, and training semantics, but make the dataset pipeline faster by adding `num_parallel_calls`, `prefetch`, and `cache` in the right places, and avoid expensive repeated string work. For submission generation, I replace the per-image pandas Series/DataFrame construction with a deterministic, vectorized NumPy-based writer that produces identical `id,value` rows in the same order, but runs orders of magnitude faster. All paths, shapes, and numeric outputs are preserved (only negligible float formatting differences may occur).'

# 9. Code solution

## === cell 0
import os, zipfile

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

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
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

print("TF version:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
CANDIDATE_INPUT_DIRS = [
    "/kaggle/input/denoising-dirty-documents/",
    "/kaggle/input/",
    "../input/denoising-dirty-documents/",
    "../input/",
]
path_zip = None
for d in CANDIDATE_INPUT_DIRS:
    if os.path.isdir(d) and (
        os.path.exists(os.path.join(d, "train.zip"))
        or os.path.exists(os.path.join(d, "denoising-dirty-documents", "train.zip"))
    ):
        if os.path.exists(os.path.join(d, "denoising-dirty-documents", "train.zip")):
            path_zip = os.path.join(d, "denoising-dirty-documents") + "/"
        else:
            path_zip = d if d.endswith("/") else d + "/"
        break

if path_zip is None:
    raise FileNotFoundError(
        "Could not find denoising-dirty-documents zips under expected /kaggle/input paths."
    )

path = "/kaggle/working/"


def maybe_extract(zip_name, target_dir):
    zip_path = os.path.join(path_zip, zip_name)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Missing {zip_path}")
    if zip_name.endswith(".csv.zip"):
        out_csv = os.path.join(path, "sampleSubmission.csv")
        if os.path.exists(out_csv):
            return
    else:
        folder = zip_name.replace(".zip", "")
        out_dir = os.path.join(path, folder)
        if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
            return
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(path)


maybe_extract("train.zip", path)
maybe_extract("test.zip", path)
maybe_extract("train_cleaned.zip", path)
maybe_extract("sampleSubmission.csv.zip", path)

train_dir = os.path.join(path, "train")
train_cleaned_dir = os.path.join(path, "train_cleaned")
test_dir = os.path.join(path, "test")
sample_sub_path = os.path.join(path, "sampleSubmission.csv")

train_img = sorted(os.listdir(train_dir))
train_cleaned_img = sorted(os.listdir(train_cleaned_dir))
test_img = sorted(os.listdir(test_dir))

print(
    "Num train:",
    len(train_img),
    "Num train_cleaned:",
    len(train_cleaned_img),
    "Num test:",
    len(test_img),
)
print("Sample submission exists:", os.path.exists(sample_sub_path))




## === cell 2
_probe = cv2.imread(os.path.join(test_dir, test_img[0]), cv2.IMREAD_UNCHANGED)
if _probe is None:
    raise FileNotFoundError(f"Could not read test image: {test_img[0]}")
if _probe.ndim == 3:
    _probe = cv2.cvtColor(_probe, cv2.COLOR_BGR2GRAY)
max_row, max_col = int(_probe.shape[0]), int(_probe.shape[1])
del _probe


class config:
    IMG_SIZE = (max_row, max_col)  # (H, W)


print("Using IMG_SIZE from image shape:", config.IMG_SIZE)

imgs = []
for f in sorted(os.listdir(train_dir))[:10]:
    im = cv2.imread(os.path.join(train_dir, f), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"Could not read train image: {f}")
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imgs.append(im)
print(
    "Median raw Dimensions:",
    int(np.median([img.shape[0] for img in imgs])),
    int(np.median([img.shape[1] for img in imgs])),
    "Median dtype:",
    str(np.median([img.dtype == np.uint16 for img in imgs])),
)
del imgs




## === cell 3
def process_image(p):
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {p}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, config.IMG_SIZE[::-1], interpolation=cv2.INTER_AREA)
    img = img.astype("float32")
    denom = 65535.0 if img.max() > 255.0 else 255.0
    img = img / denom
    img = np.reshape(img, (*config.IMG_SIZE, 1))
    return img


AUTOTUNE = tf.data.AUTOTUNE


def _tf_read_and_preprocess(path_tensor):
    img_bytes = tf.io.read_file(path_tensor)
    img = tf.io.decode_png(img_bytes, channels=1)  # uint8 or uint16 [H,W,1]
    img = tf.image.convert_image_dtype(img, tf.float32)  # now in [0,1]
    img = tf.image.resize(img, config.IMG_SIZE, method=tf.image.ResizeMethod.AREA)
    img = tf.ensure_shape(img, (*config.IMG_SIZE, 1))
    return img


def _make_path_list(dir_path, file_list):
    return [os.path.join(dir_path, fn) for fn in file_list]


train_paths = _make_path_list(train_dir, train_img)
train_cleaned_paths = _make_path_list(train_cleaned_dir, train_cleaned_img)
test_paths = _make_path_list(test_dir, test_img)

data_opts = tf.data.Options()
data_opts.experimental_deterministic = True

train_ds = tf.data.Dataset.from_tensor_slices(train_paths).with_options(data_opts)
train_cleaned_ds = tf.data.Dataset.from_tensor_slices(train_cleaned_paths).with_options(
    data_opts
)
test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(data_opts)

cache_dir = os.path.join(path, "tf_cache")
os.makedirs(cache_dir, exist_ok=True)

train_ds = (
    train_ds.map(_tf_read_and_preprocess, num_parallel_calls=AUTOTUNE)
    .cache(os.path.join(cache_dir, "train.cache"))
    .prefetch(AUTOTUNE)
)
train_cleaned_ds = (
    train_cleaned_ds.map(_tf_read_and_preprocess, num_parallel_calls=AUTOTUNE)
    .cache(os.path.join(cache_dir, "train_cleaned.cache"))
    .prefetch(AUTOTUNE)
)
test_ds = (
    test_ds.map(_tf_read_and_preprocess, num_parallel_calls=AUTOTUNE)
    .cache(os.path.join(cache_dir, "test.cache"))
    .prefetch(AUTOTUNE)
)

train_pair_ds = tf.data.Dataset.zip((train_ds, train_cleaned_ds)).with_options(
    data_opts
)

train_pair_ds = train_pair_ds.prefetch(AUTOTUNE)




## === cell 4
num_train = len(train_paths)
num_test = len(test_paths)

print("Prepared tf.data datasets. num_train:", num_train, "num_test:", num_test)




## === cell 5
print(
    (num_train, config.IMG_SIZE[0], config.IMG_SIZE[1], 1),
    (num_train, config.IMG_SIZE[0], config.IMG_SIZE[1], 1),
    (num_test, config.IMG_SIZE[0], config.IMG_SIZE[1], 1),
)




## === cell 6
if False:
    fig, ax = plt.subplots(4, 2, figsize=(15, 25))
    pass




## === cell 7
def augment_pipeline(pipeline, images, seed=19):
    return images




## === cell 8
pipeline = []  # not used




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

        th, tw = int(config.IMG_SIZE[0]), int(config.IMG_SIZE[1])
        decoded = tf.image.resize_with_crop_or_pad(decoded, th, tw)
        decoded = tf.ensure_shape(decoded, (None, th, tw, 1))
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
train_fit_ds = (
    train_pair_ds.shuffle(buffer_size=num_train, seed=19, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

history = autoencoder.fit(
    train_fit_ds,
    shuffle=False,  # shuffle handled in dataset (equivalent to shuffle=True on arrays)
    callbacks=[es],
    epochs=500,
    verbose=1,
)




## === cell 11
if False:
    fig, ax = plt.subplots(figsize=(20, 6))
    pd.DataFrame(history.history).iloc[:, :-1].plot(ax=ax)
del history




## === cell 12
autoencoder.encoder.summary()
autoencoder.decoder.summary()




## === cell 13
if False:
    pass




## === cell 14
def _pad_or_crop_to(imgs_nhwc, target_hw):
    th, tw = int(target_hw[0]), int(target_hw[1])
    h = int(imgs_nhwc.shape[1])
    w = int(imgs_nhwc.shape[2])

    imgs = imgs_nhwc[:, : min(h, th), : min(w, tw), :]

    pad_h = th - imgs.shape[1]
    pad_w = tw - imgs.shape[2]
    if pad_h > 0 or pad_w > 0:
        imgs = np.pad(
            imgs,
            pad_width=((0, 0), (0, max(0, pad_h)), (0, max(0, pad_w)), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
    return imgs[:, :th, :tw, :]


test_pred_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
pred_test = autoencoder.predict(test_pred_ds, verbose=1)
pred_test = _pad_or_crop_to(pred_test, config.IMG_SIZE)  # safety no-op
pred_test = np.clip(pred_test, 0.0, 1.0).astype(np.float32)
pred_test = np.squeeze(pred_test, axis=-1)  # (N,H,W)

H, W = config.IMG_SIZE
test_ids = np.array([int(fn[:-4]) for fn in test_img], dtype=np.int32)

rows = np.arange(1, H + 1, dtype=np.int32)
cols = np.arange(1, W + 1, dtype=np.int32)

grid_r, grid_c = np.meshgrid(rows, cols, indexing="ij")  # (H,W)
grid_r = grid_r.reshape(-1)
grid_c = grid_c.reshape(-1)

vals = pred_test.reshape(pred_test.shape[0], -1)  # (N, H*W)

out_path = "/kaggle/working/submission.csv"

with open(out_path, "w", buffering=8 * 1024 * 1024, newline="") as f:
    f.write("id,value\n")

    suffix = np.char.add(
        np.char.add(np.char.add("_", grid_r.astype(str)), "_"), grid_c.astype(str)
    )  # shape (H*W,)

    for i in range(vals.shape[0]):
        img_id = str(int(test_ids[i]))
        ids_img = np.char.add(img_id, suffix)  # shape (H*W,)

        v = vals[i].astype(np.float32, copy=False)
        lines = np.char.add(np.char.add(ids_img, ","), v.astype(str))
        f.write("\n".join(lines.tolist()))
        f.write("\n")

print("Saved submission.csv")
print("Path:", out_path)
print(pd.read_csv(out_path, nrows=5))
