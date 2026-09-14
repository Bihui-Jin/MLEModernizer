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
preds = autoencoder.predict(test, batch_size=test_batch, verbose=0)  # (N, 420, 540, 1)
preds = np.clip(preds, 0.0, 1.0).astype(np.float32, copy=False)

sub_path = os.path.join(path, "sampleSubmission.csv")
sample = pd.read_csv(sub_path)

pred_map = {}
for i, f in enumerate(test_img):
    imgid = int(os.path.splitext(f)[0])
    pred_map[imgid] = preds[i].reshape(
        -1
    )  # row-major: (row1 col1..col540, row2..., row420...)

img_ids = sample["id"].str.split("_", expand=True)[0].astype(int).to_numpy()
unique_imgs, inv = np.unique(img_ids, return_inverse=True)

values = np.empty(len(sample), dtype=np.float32)
for u_i, imgid in enumerate(unique_imgs):
    mask = inv == u_i
    values[mask] = pred_map[imgid]

submission = pd.DataFrame({"id": sample["id"].values, "value": values})
submission.to_csv("submission.csv", index=False)
print("submission.csv written:", submission.shape)
print(submission.head())

## --- ERROR in cell 14, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/682534140.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     10[0m [0;31m# Load submission template to guarantee correct IDs/order/row count[0m[0;34m[0m[0;34m[0m[0m
[1;32m     11[0m [0msub_path[0m [0;34m=[0m [0mos[0m[0;34m.[0m[0mpath[0m[0;34m.[0m[0mjoin[0m[0;34m([0m[0mpath[0m[0;34m,[0m [0;34m"sampleSubmission.csv"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 12[0;31m [0msample[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0msub_path[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     13[0m [0;34m[0m[0m
[1;32m     14[0m [0;31m# Build a fast lookup from image id -> flattened prediction in the required row-major order[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py[0m in [0;36mread_csv[0;34m(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)[0m
[1;32m   1024[0m     [0mkwds[0m[0;34m.[0m[0mupdate[0m[0;34m([0m[0mkwds_defaults[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1025[0m [0;34m[0m[0m
[0;32m-> 1026[0;31m     [0;32mreturn[0m [0m_read[0m[0;34m([0m[0mfilepath_or_buffer[0m[0;34m,[0m [0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1027[0m [0;34m[0m[0m
[1;32m   1028[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py[0m in [0;36m_read[0;34m(filepath_or_buffer, kwds)[0m
[1;32m    618[0m [0;34m[0m[0m
[1;32m    619[0m     [0;31m# Create the parser.[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 620[0;31m     [0mparser[0m [0;34m=[0m [0mTextFileReader[0m[0;34m([0m[0mfilepath_or_buffer[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    621[0m [0;34m[0m[0m
[1;32m    622[0m     [0;32mif[0m [0mchunksize[0m [0;32mor[0m [0miterator[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py[0m in [0;36m__init__[0;34m(self, f, engine, **kwds)[0m
[1;32m   1618[0m [0;34m[0m[0m
[1;32m   1619[0m         [0mself[0m[0;34m.[0m[0mhandles[0m[0;34m:[0m [0mIOHandles[0m [0;34m|[0m [0;32mNone[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1620[0;31m         [0mself[0m[0;34m.[0m[0m_engine[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_make_engine[0m[0;34m([0m[0mf[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mengine[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1621[0m [0;34m[0m[0m
[1;32m   1622[0m     [0;32mdef[0m [0mclose[0m[0;34m([0m[0mself[0m[0;34m)[0m [0;34m->[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py[0m in [0;36m_make_engine[0;34m(self, f, engine)[0m
[1;32m   1878[0m                 [0;32mif[0m [0;34m"b"[0m [0;32mnot[0m [0;32min[0m [0mmode[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1879[0m                     [0mmode[0m [0;34m+=[0m [0;34m"b"[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1880[0;31m             self.handles = get_handle(
[0m[1;32m   1881[0m                 [0mf[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1882[0m                 [0mmode[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/common.py[0m in [0;36mget_handle[0;34m(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)[0m
[1;32m    871[0m         [0;32mif[0m [0mioargs[0m[0;34m.[0m[0mencoding[0m [0;32mand[0m [0;34m"b"[0m [0;32mnot[0m [0;32min[0m [0mioargs[0m[0;34m.[0m[0mmode[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    872[0m             [0;31m# Encoding[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 873[0;31m             handle = open(
[0m[1;32m    874[0m                 [0mhandle[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    875[0m                 [0mioargs[0m[0;34m.[0m[0mmode[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '/kaggle/working/sampleSubmission.csv'
