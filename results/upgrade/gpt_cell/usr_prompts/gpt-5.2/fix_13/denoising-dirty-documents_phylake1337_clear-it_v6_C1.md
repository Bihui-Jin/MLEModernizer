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

0.28616

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28616) has done: 'Diagnosis: The crash happens immediately on `import keras` because the installed stack mixes Keras 3 (`keras==3.8.0`) with TensorFlow 2.18 and `protobuf==6.33.0`, and this combination can trigger a protobuf MessageFactory API mismatch (`GetPrototype` removed) during Keras/TensorFlow initialization. Since the notebook already imports TensorFlow and uses `tensorflow.keras.preprocessing.image`, the safest minimal fix is to avoid importing standalone Keras 3 and instead use `tf.keras` throughout for layers. This keeps the same core model/layer logic while preventing the incompatible import path that triggers the protobuf error.

Patch summary: In cell 0 only, remove the standalone `import keras` and switch the layer imports to come from `tensorflow.keras.layers`. Keep all other imports and variable names intact so later cells continue to work without modification.

Updated cells: cell 0 only.

Compatibility notes for cell k+1: No variables from cell 0 are consumed by cell 1 except standard modules; `os`, `tf`, `np`, `pd`, `plt`, `mpimg`, `tqdm`, and `load_img` remain available. Layer symbols (`Input`, `Dense`, etc.) keep the same names, only their import source changes, so any subsequent model-building code remains compatible.

Assumptions: The notebook uses standard Keras/TensorFlow layers and APIs that are available in `tf.keras` (as implied by the existing `tensorflow.keras.preprocessing.image` import), and it does not require any Keras 3–only features.'
- What this solution (achieved 0.28616) has done: 'Diagnosis: The crash happens immediately when importing TensorFlow in cell 0. With TensorFlow 2.18.0 and protobuf 6.33.0 installed, TensorFlow can fail at import time due to an incompatibility with protobuf 6.x, surfacing as `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This is an environment/runtime issue, not a code-logic issue. The minimal fix is to force TensorFlow to use the pure-Python protobuf implementation (which avoids the missing API) *before* importing TensorFlow.

Patch summary: In cell 0, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version=2 for safety) via `os.environ` before importing `tensorflow as tf`. No other logic is changed.

Updated cells: Only cell 0 is modified.

Compatibility notes for cell k+1: All imports and symbols (`tf`, `np`, `pd`, `plt`, `mpimg`, `tqdm`, `load_img`, and Keras layers) remain available under the same names, so cell 1 and later cells work unchanged.

Assumptions: This environment permits using the Python protobuf implementation and does not require the C++-backed protobuf runtime for performance; the notebook’s core ML logic is unaffected.'
- What this solution (achieved 0.28616) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` because the environment forces the pure-Python protobuf runtime via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`. With `protobuf==6.33.0` and TensorFlow 2.18, this combination can trigger the `'MessageFactory' object has no attribute 'GetPrototype'` error due to incompatibilities in the pure-Python implementation. TensorFlow expects the faster C++ protobuf backend in this setup.

Patch summary: In cell 0, stop forcing the pure-Python protobuf implementation and instead force the C++ implementation (or leave it unset) before importing TensorFlow. This is the minimal change needed to make the TensorFlow import succeed without altering any model/training logic.

Updated cells: Only cell 0 is changed.

Compatibility notes for cell k+1: No variables or interfaces used by cell 1 are modified; imports and downstream behavior remain the same, except TensorFlow can now import successfully.

Assumptions: The runtime has the protobuf C++ backend available (typical in Kaggle/TF installs), and changing these env vars is acceptable as they are purely for runtime compatibility and not model logic.'
- What this solution (achieved 0.28616) has done: 'Diagnosis: The crash happens immediately when importing TensorFlow: `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This is a known incompatibility between TensorFlow 2.18 and `protobuf==6.x`; TensorFlow expects protobuf 4/5 APIs and fails under protobuf 6. Your current environment variables force the pure-Python protobuf implementation, but that doesn’t fix the missing API—TensorFlow still imports against protobuf 6 and crashes. The minimal fix is to ensure TensorFlow uses a compatible protobuf version by downgrading protobuf at runtime *before* importing TensorFlow.

Patch summary: In cell 0, install/downgrade protobuf to a TensorFlow-compatible version (5.28.3) at runtime before importing TensorFlow, then proceed with the same imports and code. This keeps the rest of the notebook unchanged and preserves all model/training logic.

Updated cells: Only cell 0 is modified.

Compatibility notes for cell k+1: No variables or interfaces used by cell 1 are changed; imports remain available as before after TensorFlow successfully loads.

Assumptions: The environment allows `pip` installs during runtime (standard in Kaggle/Notebook environments) and restarting the kernel is not required because protobuf is imported only after the install in this cell.'
- What this solution (achieved 0.28616) has done: 'Diagnosis: Cell 9 crashes because it references the top-level `keras` module (`keras.models.Model`, `keras.optimizers.*`) but only `tensorflow as tf` and some `tensorflow.keras.*` symbols were imported earlier; no `import keras` exists, so `keras` is undefined. This is an import/namespace issue, not a model-definition problem.  
Patch summary: Add a minimal `import keras` inside cell 9 so the existing `keras.models.Model`, `keras.optimizers.*`, and subsequent `keras.callbacks.*` (cell 10) resolve correctly without changing the model architecture or training semantics.  
Updated cells: Only cell 9 is changed.  
Compatibility notes for cell k+1: Cell 10 uses `keras.callbacks...`; after importing `keras` in cell 9, `keras` be defined for cell 10 as well (notebooks share the same global namespace).  
Assumptions: The environment has the standalone `keras==3.8.0` package available (it is listed in installed packages), and importing it alongside TensorFlow is supported.'
- What this solution (achieved 0.28616) has done: 'Your validation split is currently ignored during training (you train on `X, y` without `validation_data`), so the model has no feedback for generalization and tends to overfit, which likely contributes to the high RMSE. I make the smallest change to train on `X_train, y_train` and pass `validation_data=(X_val, y_val)` while keeping the same model, loss, optimizer, epochs, and batch size. I also make the train/label pairing deterministic by sorting filenames in `images_to_array`, preventing accidental mismatches that can severely hurt RMSE without changing the core approach. Finally, I write the submission as `submission.csv` (standard Kaggle expectation) while keeping the required `id,value` columns intact.'
- What this solution (achieved 0.28616) has done: 'Your current score is far from the target (0.28616 vs 0.03254; lower is better), and the biggest likely cause is a train/test preprocessing mismatch: you train on fixed-sized resized images (target_size=(h,w)) but you predict on original test sizes without resizing, which breaks the learned mapping and inflates RMSE. I make the smallest change to ensure test images are resized exactly the same way as training (and keep output intensities clipped to [0,1] for safety), without changing the model, loss, optimizer, epochs, or training loop. I also ensure the submission vector length matches sampleSubmission exactly, so ids align deterministically.'
- What this solution (achieved 0.36642) has done: 'Diagnosis: Cell 19 raises a `ValueError` because `submit_vector` is built by flattening each prediction using the (resized) dimensions derived from `X_test` (`hi, wi = x_img.shape[1], x_img.shape[2]`), which are the fixed training resize `(h, w)` for every test image. However, the competition’s `sampleSubmission.csv` length corresponds to the sum of pixels across the original (varying) test image sizes on disk, so the produced pixel count (6577200) doesn’t match the required 5789880. The fix is to use each test image’s true on-disk shape when flattening, and resize predictions to that shape before appending pixels, keeping the same column-major order `(x then y)` used elsewhere in the notebook.

Patch summary: In cell 19 only, read each test file’s original height/width from disk (via `mpimg.imread`), resize the corresponding prediction to that size, and flatten into `submit_vector` in the same nested-loop order as before. Keep the existing length check against `sample_csv` to fail fast if anything still mismatches.

Updated cells: Cell 19 updated as below.

Compatibility notes for cell k+1: The variables `submit_vector` and `sample_csv` are still produced with the same types and intended semantics; cell 20 (which computes pixel totals from `yh_test`) is unaffected.

Assumptions: Test images are grayscale (2D) or readable by `matplotlib.image.imread`; if an alpha channel exists, only the first channel is used for shape and resizing.'
- What this solution (achieved 0.28616) has done: 'Your score is far above the target (0.36642 vs 0.03254, lower is better), so we should improve RMSE with the smallest changes that keep your model/training intact. The biggest issue still hurting performance is a mismatch between how training/validation targets are paired and how data is shuffled: you currently shuffle `X` and `y` with a random permutation that can silently misalign inputs/labels if anything in ordering ever differs, and it also makes runs non-reproducible. I make pairing deterministic by deriving `y` from the *same filenames* as `X` (no independent label list) and only shuffling via a seeded RNG, keeping the exact same resizing, architecture, optimizer/loss, and training loop. This should reduce RMSE meaningfully without changing core logic, and the submission formatting remains identical.'
- What this solution (achieved 0.28616) has done: 'Your current RMSE (0.28616) is far worse than the target (0.03254, lower is better), and the biggest remaining issue is that the network is trained to output 2×2-upsampled feature maps without an explicit constraint to match the label size—so it can learn a slightly mis-scaled mapping that still “looks OK” but yields poor pixel RMSE. To keep the core model/training approach intact while improving RMSE, I make the input size divisible by 2 (since you pool by 2 then upsample by 2) by cropping all images to an even height/width consistently for train/val/test, so the output aligns exactly with targets. I also ensure the submission flattening order remains identical, and keep all optimizers/loss/epochs/loop unchanged. These are minimal preprocessing/alignment fixes aimed at reducing RMSE toward your target without changing the architecture or training procedure.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==5.28.3"]
)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.layers import (
    Input,
    Dense,
    Activation,
    BatchNormalization,
    Flatten,
    Conv2D,
)
from tensorflow.keras.layers import MaxPooling2D, Dropout, UpSampling2D

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)



## === cell 1
train_zip_path = "/kaggle/input/denoising-dirty-documents/train.zip"
test_zip_path = "/kaggle/input/denoising-dirty-documents/test.zip"
sample_zip_path = "/kaggle/input/denoising-dirty-documents/sampleSubmission.csv.zip"
trainclean_zip_path = "/kaggle/input/denoising-dirty-documents/train_cleaned.zip"
extracting_path = "/kaggle/working"



## === cell 2
import zipfile

with zipfile.ZipFile(train_zip_path, "r") as zip_ref:
    zip_ref.extractall(extracting_path)

with zipfile.ZipFile(test_zip_path, "r") as zip_ref:
    zip_ref.extractall(extracting_path)

with zipfile.ZipFile(sample_zip_path, "r") as zip_ref:
    zip_ref.extractall(extracting_path)

with zipfile.ZipFile(trainclean_zip_path, "r") as zip_ref:
    zip_ref.extractall(extracting_path)



## === cell 3
img_arr = mpimg.imread(extracting_path + "/train/107.png")
h, w = img_arr.shape
print("Height: ", h, "- Width: ", w)
print(img_arr.dtype)

h = (h // 2) * 2
w = (w // 2) * 2
print("Using even crop size -> Height:", h, "Width:", w)



## === cell 4
image_names = os.listdir(extracting_path + "/train")
data_size = len(image_names)
X = np.zeros([data_size, 2], dtype=np.uint16)
for i in tqdm(range(data_size)):
    image_name = image_names[i]
    img_dir = os.path.join(extracting_path + "/train", image_name)
    img_pixels = mpimg.imread(img_dir)
    X[i] = img_pixels.shape

print("Number of training images:", data_size)
print("Differnet image hights: {}".format(set(X[:, 0])))
print("Differnet image widths: {}".format(set(X[:, 1])))




## === cell 5
def images_to_array(data_dir, label_dir=None, img_size=(h, w)):
    """
    1- Read image samples from certain directory.
    2- Stack them into one big numpy array.
    -- And if there are labels images ..
    3- Read sample's label from the labels directory (paired by filename).
    4- Stack them into one big numpy array.
    5- Shuffle Data and label arrays (seeded for determinism).
    """
    image_names = sorted(os.listdir(data_dir))
    data_size = len(image_names)

    X = np.zeros([data_size, img_size[0], img_size[1]], dtype=np.uint8)
    for i in tqdm(range(data_size)):
        image_name = image_names[i]
        img_dir = os.path.join(data_dir, image_name)
        img_pixels = load_img(img_dir, color_mode="grayscale", target_size=(h, w))
        arr = np.array(img_pixels, dtype=np.uint8)[:h, :w]
        X[i] = arr
    X = X.reshape(data_size, h, w, 1)

    if label_dir:
        y = np.zeros([data_size, img_size[0], img_size[1]], dtype=np.uint8)
        for i in tqdm(range(data_size)):
            image_name = image_names[i]  # pair labels to inputs by the same filename
            lbl_path = os.path.join(label_dir, image_name)
            img_pixels = load_img(lbl_path, color_mode="grayscale", target_size=(h, w))
            arr = np.array(img_pixels, dtype=np.uint8)[:h, :w]
            y[i] = arr
        y = y.reshape(data_size, h, w, 1)

        rng = np.random.RandomState(SEED)
        ind = rng.permutation(data_size)
        X = X[ind]
        y = y[ind]

        print("Ouptut Data Size: ", X.shape)
        print("Ouptut Label Size: ", y.shape)
        return X / 255.0, y / 255.0

    print("Ouptut Data Size: ", X.shape)
    return X / 255.0




## === cell 6
X, y = images_to_array(extracting_path + "/train", extracting_path + "/train_cleaned")



## === cell 7
val_split = int(0.3 * data_size)
X_val, y_val = X[:val_split], y[:val_split]
X_train, y_train = X[val_split:], y[val_split:]
print("Train data shape: ", X_train.shape)
print("Test data shape: ", X_val.shape)



## === cell 8
samples = np.concatenate((X_train[:3], y_train[:3]), axis=0)

f, ax = plt.subplots(2, 3, figsize=(20, 10))
for i, img in enumerate(samples):
    ax[i // 3, i % 3].imshow(img[:, :, 0], cmap="gray")
    ax[i // 3, i % 3].axis("off")
plt.show()



## === cell 9
import keras

input_layer = Input(shape=(None, None, 1))
x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), padding="same")(x)

x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = UpSampling2D((2, 2))(x)
output_layer = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
model = keras.models.Model(inputs=[input_layer], outputs=[output_layer])

sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
rms = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
ada = keras.optimizers.Adagrad(learning_rate=0.01)

model.compile(optimizer="adam", loss="mean_squared_error")



## === cell 10
LR_callback = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", patience=4, verbose=10, factor=0.4, min_lr=0.00001
)



## === cell 11
history = model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[LR_callback],
    verbose=1,
)



## === cell 12
model.evaluate(X_val, y_val)



## === cell 13
test_samples, test_labels = X_val[:3], y_val[:3]
test_pred = model.predict(X_val[:3])

samples = np.concatenate((test_samples, test_labels, test_pred), axis=0)

f, ax = plt.subplots(3, 3, figsize=(25, 15))
for i, img in enumerate(samples):
    ax[i // 3, i % 3].imshow(img[:, :, 0], cmap="gray")
    ax[i // 3, i % 3].axis("off")
plt.show()



## === cell 14
image_names = sorted(os.listdir(extracting_path + "/test"))
data_size = len(image_names)
X_test = []
for i in tqdm(range(data_size)):
    image_name = image_names[i]
    img_dir = os.path.join(extracting_path + "/test", image_name)
    img_pixels = load_img(img_dir, color_mode="grayscale", target_size=(h, w))
    arr = np.array(img_pixels, dtype=np.uint8)[:h, :w]
    X_test.append(arr.reshape(1, h, w, 1) / 255.0)

print("Test sample shape: ", X_test[0].shape)
print("Test sample dtype: ", X_test[0].dtype)



## === cell 15
yh_test = []
for img in X_test:
    pred = model.predict(img, verbose=0)[0, :, :, 0]
    yh_test.append(np.clip(pred, 0.0, 1.0))



## === cell 16
f, ax = plt.subplots(3, 2, figsize=(20, 10))
for i, (img, lbl) in enumerate(zip(X_test[:3], yh_test[:3])):
    ax[i, 0].imshow(img[0, :, :, 0], cmap="gray")
    ax[i, 0].axis("off")

    ax[i, 1].imshow(lbl, cmap="gray")
    ax[i, 1].axis("off")
plt.show()



## === cell 17
submit_vector = []
for img in yh_test:
    hi, wi = img.shape
    for i in range(wi):
        for j in range(hi):
            submit_vector.append(float(img[j, i]))
print(len(submit_vector))



## === cell 18
sample_csv = pd.read_csv(extracting_path + "/sampleSubmission.csv")
sample_csv.head(10)



## === cell 19
submit_vector = []
test_image_names = sorted(os.listdir(extracting_path + "/test"))

for pred_img, image_name in zip(yh_test, test_image_names):
    img_path = os.path.join(extracting_path + "/test", image_name)
    orig = mpimg.imread(img_path)
    if orig.ndim == 3:
        orig = orig[:, :, 0]
    hi, wi = orig.shape[0], orig.shape[1]  # original test image size from disk

    if pred_img.shape != (hi, wi):
        pred_img_resized = tf.image.resize(
            pred_img[..., None], size=(hi, wi), method="bilinear"
        ).numpy()[:, :, 0]
        pred_img_resized = np.clip(pred_img_resized, 0.0, 1.0)
    else:
        pred_img_resized = pred_img

    for i in range(wi):
        for j in range(hi):
            submit_vector.append(float(pred_img_resized[j, i]))

if len(submit_vector) != len(sample_csv):
    raise ValueError(
        f"Submission length mismatch: predicted {len(submit_vector)} pixels, "
        f"but sampleSubmission has {len(sample_csv)} rows."
    )



## === cell 20
c = 0
for img in yh_test:
    hi, wi = img.shape
    c += hi * wi



## === cell 21
id_col = sample_csv["id"]
value_col = pd.Series(submit_vector, name="value")
submission = pd.concat([id_col, value_col], axis=1)
submission.head(10)



## === cell 22
submission.to_csv("submission.csv", index=False)



## === cell 23
import shutil

shutil.rmtree(extracting_path + "/train")
shutil.rmtree(extracting_path + "/test")
shutil.rmtree(extracting_path + "/train_cleaned")
