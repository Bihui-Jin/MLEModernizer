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

0.05382

# 6. Current score

0.36805

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.36351) has done: 'I fix the crash caused by the `protobuf`/`keras` import conflict by using `tf.keras` only (same layers/architecture/training loop), which makes the notebook import reliably in the Kaggle environment. I also fix two silent logic issues that hurt score: inconsistent image ordering between noisy and cleaned sets (labels get mismatched), and an incorrect train/validation split size derived from the wrong `data_size`. Finally, I make submission creation deterministic and aligned with the sample submission (and ensure the filename ends with `.csv`).'
- What this solution (achieved 0.35823) has done: 'I fix the crash in the first cell caused by the protobuf/Keras import conflict by forcing TensorFlow’s bundled protobuf implementation before importing TensorFlow, which is the minimal change needed to make the notebook run in this Kaggle image. I also switch image reading to `tf.io.decode_png` (instead of `matplotlib.image.imread`) to ensure consistent grayscale decoding and avoid dtype/normalization inconsistencies that can silently hurt RMSE. Finally, I keep the same model/training loop but make the submission filename lowercase with a `.csv` suffix and ensure predictions are clipped to `[0, 1]` for metric-safe outputs.'
- What this solution (achieved 0.45865) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime and avoiding the standalone `keras` import path, while keeping the same `tf.keras` model/training loop. I also fix a key data/label loading inconsistency: training images were decoded via `tf.io.decode_png` for shape probing but loaded via `load_img` for training, which can silently differ; switching to `tf.io.decode_png` everywhere keeps grayscale decoding consistent and improves RMSE without changing the model. Finally, I keep the submission construction logic but ensure it always aligns with the sample submission ordering/length and writes a valid `.csv` file.'
- What this solution (achieved 0.36805) has done: 'We fix two execution blockers while keeping the same core model/training loop: (1) resolve the TensorFlow/protobuf import crash by using the fast C++ protobuf implementation (removing the forced pure-Python setting that triggers the `MessageFactory` error in this Kaggle image), and (2) fix the data pipeline to handle varying image heights by padding all images to a common (max) height/width so training can proceed without resizing/cropping. This preserves the denoising objective and model architecture while ensuring labels stay correctly paired by sorting filenames. Finally, we keep submission creation aligned to `sampleSubmission.csv` and ensure a valid `.csv` is written.'

# 9. Code solution

## === cell 0
import os
import zipfile

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)



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
os.makedirs(extracting_path, exist_ok=True)


def _extract_if_missing(zip_path, expected_dir_or_file):
    expected_path = os.path.join(extracting_path, expected_dir_or_file)
    if os.path.exists(expected_path):
        return
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracting_path)


_extract_if_missing(train_zip_path, "train")
_extract_if_missing(test_zip_path, "test")
_extract_if_missing(sample_zip_path, "sampleSubmission.csv")
_extract_if_missing(trainclean_zip_path, "train_cleaned")

print(
    "Extracted dirs in working:",
    [
        p
        for p in ["train", "train_cleaned", "test"]
        if os.path.exists(os.path.join(extracting_path, p))
    ],
)



## === cell 3
sample_path = os.path.join(extracting_path, "train", "107.png")
img_bytes = tf.io.read_file(sample_path)
img = tf.io.decode_png(img_bytes, channels=1)
img_arr = img.numpy()[:, :, 0]
h, w = img_arr.shape
print("Example image shape:", (h, w), "dtype:", img_arr.dtype)




## === cell 4
def _list_pngs(dir_path):
    return sorted([f for f in os.listdir(dir_path) if f.lower().endswith(".png")])


def _probe_max_hw(dir_path):
    names = _list_pngs(dir_path)
    max_h, max_w = 0, 0
    shapes = set()
    for f in tqdm(names, desc=f"Probing {os.path.basename(dir_path)}"):
        p = os.path.join(dir_path, f)
        im = tf.io.decode_png(tf.io.read_file(p), channels=1)
        hh, ww = int(im.shape[0]), int(im.shape[1])
        shapes.add((hh, ww))
        max_h = max(max_h, hh)
        max_w = max(max_w, ww)
    return max_h, max_w, shapes


train_dir = os.path.join(extracting_path, "train")
clean_dir = os.path.join(extracting_path, "train_cleaned")
test_dir = os.path.join(extracting_path, "test")

max_h1, max_w1, shapes_train = _probe_max_hw(train_dir)
max_h2, max_w2, shapes_clean = _probe_max_hw(clean_dir)
max_h3, max_w3, shapes_test = _probe_max_hw(test_dir)

MAX_H = max(max_h1, max_h2, max_h3)
MAX_W = max(max_w1, max_w2, max_w3)

print(
    "Train shapes (unique):",
    len(shapes_train),
    "example:",
    list(sorted(shapes_train))[:5],
)
print(
    "Clean shapes (unique):",
    len(shapes_clean),
    "example:",
    list(sorted(shapes_clean))[:5],
)
print(
    "Test shapes  (unique):",
    len(shapes_test),
    "example:",
    list(sorted(shapes_test))[:5],
)
print("Using padded canvas (H, W):", (MAX_H, MAX_W))




## === cell 5
def _decode_png_uint8(path):
    return tf.io.decode_png(tf.io.read_file(path), channels=1).numpy().astype(np.uint8)


def _pad_to_canvas(img_uint8, canvas_hw):
    ch, cw = canvas_hw
    hh, ww = img_uint8.shape[0], img_uint8.shape[1]
    if hh > ch or ww > cw:
        raise ValueError(f"Image larger than canvas: {(hh, ww)} vs {(ch, cw)}")
    out = np.zeros((ch, cw, 1), dtype=np.uint8)
    out[:hh, :ww, :] = img_uint8
    return out


def images_to_array(data_dir, label_dir=None, img_size=(MAX_H, MAX_W)):
    """
    Fix: handle varying image sizes by padding to a common canvas (max H/W).
    Keep sorted filename pairing between noisy and cleaned images.
    """
    image_names = _list_pngs(data_dir)
    data_size_local = len(image_names)

    X = np.zeros([data_size_local, img_size[0], img_size[1], 1], dtype=np.uint8)
    for i, image_name in enumerate(
        tqdm(image_names, desc=f"Loading {os.path.basename(data_dir)}")
    ):
        img_path = os.path.join(data_dir, image_name)
        img = _decode_png_uint8(img_path)
        X[i] = _pad_to_canvas(img, img_size)

    if label_dir:
        label_names = _list_pngs(label_dir)

        if label_names != image_names:
            common = sorted(set(image_names).intersection(set(label_names)))
            if len(common) == 0:
                raise ValueError(
                    "No matching filenames between data_dir and label_dir."
                )
            image_names = common
            label_names = common
            data_size_local = len(common)

            X = np.zeros([data_size_local, img_size[0], img_size[1], 1], dtype=np.uint8)
            for i, image_name in enumerate(
                tqdm(
                    image_names, desc=f"Reloading {os.path.basename(data_dir)} (common)"
                )
            ):
                img_path = os.path.join(data_dir, image_name)
                img = _decode_png_uint8(img_path)
                X[i] = _pad_to_canvas(img, img_size)

        y = np.zeros([data_size_local, img_size[0], img_size[1], 1], dtype=np.uint8)
        for i, image_name in enumerate(
            tqdm(label_names, desc=f"Loading {os.path.basename(label_dir)}")
        ):
            img_path = os.path.join(label_dir, image_name)
            img = _decode_png_uint8(img_path)
            y[i] = _pad_to_canvas(img, img_size)

        ind = np.random.permutation(data_size_local)
        X = X[ind]
        y = y[ind]

        print("Output Data Size:", X.shape, "Label Size:", y.shape)
        return X / 255.0, y / 255.0, image_names

    print("Output Data Size:", X.shape)
    return X / 255.0, image_names


X, y, train_filenames = images_to_array(train_dir, clean_dir, img_size=(MAX_H, MAX_W))



## === cell 6
data_size_loaded = X.shape[0]
val_split = int(0.15 * data_size_loaded)

X_val, y_val = X[:val_split], y[:val_split]
X_train, y_train = X[val_split:], y[val_split:]

print("Train data shape:", X_train.shape, "Val data shape:", X_val.shape)



## === cell 7
samples = np.concatenate((X_train[:3], y_train[:3]), axis=0)

f, ax = plt.subplots(2, 3, figsize=(20, 10))
for i, img in enumerate(samples):
    ax[i // 3, i % 3].imshow(img[:, :, 0], cmap="gray")
    ax[i // 3, i % 3].axis("off")
plt.show()



## === cell 8
input_layer = Input(shape=(None, None, 1))
x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), padding="same")(x)

x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = UpSampling2D((2, 2))(x)
output_layer = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
model = Model(inputs=[input_layer], outputs=[output_layer])

model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()



## === cell 9
LR_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", patience=4, verbose=1, factor=0.4, min_lr=1e-5
)



## === cell 10
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=70,
    batch_size=16,
    callbacks=[LR_callback],
    verbose=2,
)



## === cell 11
val_loss = model.evaluate(X_val, y_val, verbose=0)
print("Validation loss:", val_loss)



## === cell 12
test_samples, test_labels = X_val[:3], y_val[:3]
test_pred = model.predict(X_val[:3], verbose=0)

samples = np.concatenate((test_samples, test_labels, test_pred), axis=0)

f, ax = plt.subplots(3, 3, figsize=(25, 15))
for i, img in enumerate(samples):
    ax[i // 3, i % 3].imshow(img[:, :, 0], cmap="gray")
    ax[i // 3, i % 3].axis("off")
plt.show()



## === cell 13
image_names = _list_pngs(test_dir)
data_size_test = len(image_names)

X_test = []
orig_hw = []
for image_name in tqdm(image_names, desc="Loading test"):
    img_path = os.path.join(test_dir, image_name)
    img = _decode_png_uint8(img_path)  # (h,w,1) uint8
    hh, ww = img.shape[0], img.shape[1]
    orig_hw.append((hh, ww))
    X_test.append(img.reshape(1, hh, ww, 1) / 255.0)

print("Test sample shape:", X_test[0].shape, "dtype:", X_test[0].dtype)



## === cell 14
yh_test = []
for img in tqdm(X_test, desc="Predicting test"):
    pred = model.predict(img, verbose=0)[0, :, :, 0]
    pred = np.clip(pred, 0.0, 1.0)
    yh_test.append(pred)



## === cell 15
f, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(X_test[0][0, :, :, 0], cmap="gray")
ax[0].axis("off")

ax[1].imshow(yh_test[0], cmap="gray")
ax[1].axis("off")
plt.show()



## === cell 16
submit_vector = []
for img in yh_test:
    submit_vector.extend(img.T.reshape(-1).tolist())

print("Submission vector length:", len(submit_vector))



## === cell 17
sample_csv = pd.read_csv(os.path.join(extracting_path, "sampleSubmission.csv"))
print(sample_csv.head(3))
print("Expected submission rows:", sample_csv.shape[0])



## === cell 18
expected_len = sample_csv.shape[0]
if len(submit_vector) != expected_len:
    raise ValueError(
        f"Submission length mismatch: got {len(submit_vector)} values, expected {expected_len}."
    )



## === cell 19
id_col = sample_csv["id"]
value_col = pd.Series(submit_vector, name="value", dtype=np.float32)

submission = pd.concat([id_col, value_col], axis=1)
print(submission.head(3))
print(submission.tail(3))



## === cell 20
out_path = "cleared.csv"
submission.to_csv(out_path, index=False)
print("Wrote submission:", os.path.abspath(out_path), "shape:", submission.shape)
