# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.9

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

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

test_metas = []
for f in test_img:
    img_bgr = cv2.imread(path + "test/" + f)
    gray = _to_gray_float(img_bgr)
    _, meta = _letterbox_gray(gray, config.IMG_SIZE)
    test_metas.append(meta)

preds = autoencoder.predict(test, batch_size=test_batch, verbose=0)  # (N, 420, 540, 1)
preds = np.clip(preds, 0.0, 1.0).astype(np.float32, copy=False)

preds_orig_flat = []
for i in range(preds.shape[0]):
    pred_gray = preds[i, ..., 0]  # (420, 540)
    pred_orig = _unletterbox_to_orig(pred_gray, test_metas[i])  # (orig_h, orig_w)
    preds_orig_flat.append(pred_orig.reshape(-1).astype(np.float32, copy=False))
preds_orig_flat = np.asarray(
    preds_orig_flat, dtype=np.float32
)  # (N_test, orig_h*orig_w)

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

test_ids = np.array([int(os.path.splitext(f)[0]) for f in test_img], dtype=np.int32)
id_to_idx = {imgid: i for i, imgid in enumerate(test_ids)}

parts = sample["id"].astype(str).str.split("_", n=1, expand=True)
img_ids = parts[0].astype(np.int32).to_numpy()
pix_idx = parts[1].astype(np.int32).to_numpy()

row_img_idx = np.fromiter(
    (id_to_idx[i] for i in img_ids), count=len(img_ids), dtype=np.int32
)

values = preds_orig_flat[row_img_idx, pix_idx].astype(np.float32, copy=False)

submission = pd.DataFrame({"id": sample["id"].values, "value": values})
submission.to_csv("submission.csv", index=False)
print("submission.csv written:", submission.shape)
print(submission.head())


## --- ERROR in cell 14, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/395313598.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     22[0m     [0mpred_orig[0m [0;34m=[0m [0m_unletterbox_to_orig[0m[0;34m([0m[0mpred_gray[0m[0;34m,[0m [0mtest_metas[0m[0;34m[[0m[0mi[0m[0;34m][0m[0;34m)[0m  [0;31m# (orig_h, orig_w)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     23[0m     [0mpreds_orig_flat[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mpred_orig[0m[0;34m.[0m[0mreshape[0m[0;34m([0m[0;34m-[0m[0;36m1[0m[0;34m)[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mfloat32[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 24[0;31m preds_orig_flat = np.asarray(
[0m[1;32m     25[0m     [0mpreds_orig_flat[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mnp[0m[0;34m.[0m[0mfloat32[0m[0;34m[0m[0;34m[0m[0m
[1;32m     26[0m )  # (N_test, orig_h*orig_w)

[0;31mValueError[0m: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (29,) + inhomogeneous part.
