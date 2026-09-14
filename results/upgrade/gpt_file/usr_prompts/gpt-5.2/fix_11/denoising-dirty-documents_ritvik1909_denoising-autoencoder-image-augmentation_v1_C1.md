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

# 9. Code solution

## === cell 0
import os, zipfile

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
sample_ids = pd.read_csv(sample_sub_path, usecols=["id"])["id"].astype(str)
parts = sample_ids.str.split("_", expand=True)
if parts.shape[1] != 3:
    raise ValueError(
        "Unexpected id format in sampleSubmission; expected image_row_col."
    )
max_row = int(parts[1].astype(np.int32).max())
max_col = int(parts[2].astype(np.int32).max())


class config:
    IMG_SIZE = (max_row, max_col)  # (H, W)


print("Using IMG_SIZE from sampleSubmission:", config.IMG_SIZE)

imgs = []
for f in sorted(os.listdir(train_dir))[:10]:
    im = cv2.imread(os.path.join(train_dir, f), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(f"Could not read train image: {f}")
    imgs.append(im)
print(
    "Median raw Dimensions:",
    int(np.median([img.shape[0] for img in imgs])),
    int(np.median([img.shape[1] for img in imgs])),
)
del imgs




## === cell 3
def process_image(p):
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {p}")
    img = cv2.resize(img, config.IMG_SIZE[::-1], interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    img = np.reshape(img, (*config.IMG_SIZE, 1))
    return img


AUTOTUNE = tf.data.AUTOTUNE


def _tf_read_and_preprocess(path_tensor):
    img_bytes = tf.io.read_file(path_tensor)
    img = tf.io.decode_png(img_bytes, channels=1)  # uint8 [H,W,1]
    img = tf.image.resize(
        img, config.IMG_SIZE, method=tf.image.ResizeMethod.AREA
    )  # float32
    img = tf.cast(img, tf.float32) / 255.0
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

train_ds = train_ds.map(_tf_read_and_preprocess, num_parallel_calls=AUTOTUNE).cache()
train_cleaned_ds = train_cleaned_ds.map(
    _tf_read_and_preprocess, num_parallel_calls=AUTOTUNE
).cache()
test_ds = test_ds.map(_tf_read_and_preprocess, num_parallel_calls=AUTOTUNE).cache()

train_pair_ds = tf.data.Dataset.zip((train_ds, train_cleaned_ds)).with_options(
    data_opts
)



## === cell 4
train = np.stack(list(train_ds.as_numpy_iterator()), axis=0)
train_cleaned = np.stack(list(train_cleaned_ds.as_numpy_iterator()), axis=0)
test = np.stack(list(test_ds.as_numpy_iterator()), axis=0)



## === cell 5
train.shape, train_cleaned.shape, test.shape



## === cell 6
if False:
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
    train_pair_ds.shuffle(
        buffer_size=len(train), seed=19, reshuffle_each_iteration=True
    )
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
    decoded_imgs = autoencoder(train[:4]).numpy()

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


sample = pd.read_csv(sample_sub_path, usecols=["id"])
ids = sample["id"].astype(str)

id_parts = ids.str.split("_", expand=True)
if id_parts.shape[1] != 3:
    raise ValueError("Unexpected id format; expected 3 underscore-separated parts.")

img_ids = id_parts[0].astype(np.int32).to_numpy()
rows_1b = id_parts[1].astype(np.int32).to_numpy()
cols_1b = id_parts[2].astype(np.int32).to_numpy()

pred_test = autoencoder.predict(test, batch_size=BATCH_SIZE, verbose=1)
pred_test = _pad_or_crop_to(
    pred_test, config.IMG_SIZE
)  # should be no-op now, kept for safety
pred_test = np.clip(pred_test, 0.0, 1.0).astype(np.float32)
pred_test = np.squeeze(pred_test, axis=-1)  # (N,H,W)

test_ids = np.array([int(fn[:-4]) for fn in test_img], dtype=np.int32)
test_id_to_index = {int(k): int(i) for i, k in enumerate(test_ids.tolist())}

test_index_for_row = np.fromiter(
    (test_id_to_index.get(int(x), -1) for x in img_ids.tolist()),
    dtype=np.int32,
    count=len(img_ids),
)
if (test_index_for_row < 0).any():
    missing = np.unique(img_ids[test_index_for_row < 0])[:10]
    raise KeyError(
        f"Some image ids from sampleSubmission not found in test files. Examples: {missing}"
    )

rr = rows_1b.astype(np.int64) - 1
cc = cols_1b.astype(np.int64) - 1

vals = pred_test[test_index_for_row, rr, cc].astype(np.float32)
vals = np.clip(vals, 0.0, 1.0)

sub = pd.DataFrame({"id": ids.to_numpy(), "value": vals})

out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)

print("Saved submission.csv with rows:", len(sub))
print("Path:", out_path)
print(sub.head())
