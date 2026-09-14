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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0
tf_keras==2.18.0

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

0.08208

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.26464) has done: 'Implemented fixes to resolve import conflicts, optimizer initialization, and dataset ordering. Replaced legacy Keras imports with TensorFlow‑Keras, corrected the Adam learning‑rate argument, ensured deterministic file ordering, added a modest model capacity boost, and aligned the prediction‑to‑submission loop. These changes allow the notebook to run end‑to‑end, produce a valid `submission.csv`, and improve the denoising performance toward the target RMSE.'
- What this solution (achieved 0.26441) has done: 'The fix adds a protobuf compatibility flag before any TensorFlow import to stop the `"MessageFactory" object has no attribute 'GetPrototype"` error, and modestly enlarges the auto‑encoder (doubling filter counts) and gives early stopping a longer patience so the model can train a bit longer—both expected to lower the RMSE toward the target. No core logic is changed, and the script still writes a correctly formatted `submission.csv`.'
- What this solution (achieved 0.22584) has done: 'The changes set the protobuf implementation flag **before any imports**, lower the optimizer learning rate, expand the auto‑encoder capacity, and give the training loop more epochs and patience so the model can learn a stronger denoising mapping, which should lower the RMSE toward the target while keeping the original workflow unchanged.'
- What this solution (achieved 0.224) has done: 'Implemented two minor adjustments to boost denoising performance while keeping the core architecture unchanged:  
- Increased early‑stopping patience from 30 to 60 so training can continue longer before stopping.  
- Raised the maximum epoch count from 400 to 800 to give the model more opportunity to converge.  

These changes are minimal, preserve the original workflow, and are expected to lower the RMSE toward the target score.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import glob
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    BatchNormalization,
)

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("GPU memory growth setup failed:", e)

plt.rcParams["figure.figsize"] = (10.0, 5.0)  # default plot size


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import zipfile

zipfiles = ["train", "test", "train_cleaned"]
if not (
    os.path.isdir("./train")
    and os.path.isdir("./test")
    and os.path.isdir("./train_cleaned")
):
    for each_zip in zipfiles:
        zip_path = f"/kaggle/input/denoising-dirty-documents/{each_zip}.zip"
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("./")
else:
    print("Data already extracted; skipping zip extraction.")




## === cell 2
def load_image_from_dir(img_path):
    """Load all PNG images from a directory, resize to (258,540) and normalize."""
    file_list = sorted(glob.glob(os.path.join(img_path, "*.png")))
    img_list = np.empty((len(file_list), 258, 540, 1), dtype=np.float32)
    for i, fig in enumerate(file_list):
        raw = cv2.imread(fig, cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(raw, (540, 258))  # width, height
        normalized = resized.astype(np.float32) / 255.0
        img_list[i, :, :, 0] = normalized
    return img_list


def train_val_split(data, random_seed=55, split=0.75):
    """Simple shuffled split."""
    rng = np.random.RandomState(seed=random_seed)
    dsize = len(data)
    indices = rng.permutation(dsize)
    cut = int(split * dsize)
    return data[indices[:cut]], data[indices[cut:]]




## === cell 3
full_train = load_image_from_dir("./train")
full_target = load_image_from_dir("./train_cleaned")
test = load_image_from_dir("./test")

train, val = train_val_split(full_train, random_seed=9, split=0.8)
target_train, target_val = train_val_split(full_target, random_seed=9, split=0.8)


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2435342076.py in <cell line: 0>()
----> 1 full_train = load_image_from_dir("./train")
      2 full_target = load_image_from_dir("./train_cleaned")
      3 test = load_image_from_dir("./test")
      4 
      5 train, val = train_val_split(full_train, random_seed=9, split=0.8)

/tmp/ipykernel_55/4293880555.py in load_image_from_dir(img_path)
      2     """Load all PNG images from a directory, resize to (258,540) and normalize."""
      3     file_list = sorted(glob.glob(os.path.join(img_path, "*.png")))
----> 4     img_list = np.empty((len(file_list), 258, 540, 1), dtype=np.float32)
      5     for i, fig in enumerate(file_list):
      6         raw = cv2.imread(fig, cv2.IMREAD_GRAYSCALE)

NameError: name 'np' is not defined

## === cell 4
optimizer = Adam(learning_rate=1e-4)


## === cell 5
input_layer = Input(shape=train[0].shape)  # (258,540,1)

h = Conv2D(128, (3, 3), activation="relu", padding="same")(input_layer)
h = BatchNormalization()(h)
h = Conv2D(256, (3, 3), activation="relu", padding="same")(h)
h = BatchNormalization()(h)
h = MaxPooling2D((2, 2), padding="same")(h)

h = Conv2D(256, (3, 3), activation="relu", padding="same")(h)
h = BatchNormalization()(h)

h = Conv2D(64, (3, 3), activation="relu", padding="same")(h)
h = BatchNormalization()(h)

h = Conv2D(128, (3, 3), activation="relu", padding="same")(h)
h = BatchNormalization()(h)
h = UpSampling2D((2, 2))(h)

output_layer = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(h)

ae_model = Model(inputs=input_layer, outputs=output_layer)
ae_model.compile(loss="mse", optimizer=optimizer)
ae_model.summary()


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2760457543.py in <cell line: 0>()
----> 1 input_layer = Input(shape=train[0].shape)  # (258,540,1)
      2 
      3 h = Conv2D(128, (3, 3), activation="relu", padding="same")(input_layer)
      4 h = BatchNormalization()(h)
      5 h = Conv2D(256, (3, 3), activation="relu", padding="same")(h)

NameError: name 'train' is not defined

## === cell 6
early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=80,  # longer patience to allow more training
    verbose=1,
    mode="auto",
    restore_best_weights=True,
)


## === cell 7
train_dataset = (
    tf.data.Dataset.from_tensor_slices((train, target_train))
    .shuffle(buffer_size=len(train), reshuffle_each_iteration=True)
    .batch(20, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((val, target_val))
    .batch(20, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

history = ae_model.fit(
    train_dataset,
    epochs=250,  # lower bound that still allows early stopping to act
    validation_data=val_dataset,
    callbacks=[early_stopping],
    verbose=2,
)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/442461900.py in <cell line: 0>()
      1 train_dataset = (
----> 2     tf.data.Dataset.from_tensor_slices((train, target_train))
      3     .shuffle(buffer_size=len(train), reshuffle_each_iteration=True)
      4     .batch(20, drop_remainder=False)
      5     .prefetch(tf.data.AUTOTUNE)

NameError: name 'train' is not defined

## === cell 8
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(loc="upper left")
plt.show()


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1399623072.py in <cell line: 0>()
----> 1 plt.plot(history.history["loss"], label="train")
      2 plt.plot(history.history["val_loss"], label="val")
      3 plt.title("Model loss")
      4 plt.ylabel("Loss")
      5 plt.xlabel("Epoch")

NameError: name 'history' is not defined

## === cell 9
preds = ae_model.predict(test, batch_size=20)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1014578725.py in <cell line: 0>()
----> 1 preds = ae_model.predict(test, batch_size=20)

NameError: name 'ae_model' is not defined

## === cell 10
if test.shape[0] > 0:
    plt.imshow(test[0].reshape(258, 540), cmap="gray")
    plt.title("Noisy Input (first test image)")
    plt.show()
if preds.shape[0] > 0:
    plt.imshow(preds[0].reshape(258, 540), cmap="gray")
    plt.title("Denoised Output (first test image)")
    plt.show()


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/420432069.py in <cell line: 0>()
----> 1 if test.shape[0] > 0:
      2     plt.imshow(test[0].reshape(258, 540), cmap="gray")
      3     plt.title("Noisy Input (first test image)")
      4     plt.show()
      5 if preds.shape[0] > 0:

NameError: name 'test' is not defined

## === cell 11
file_list = sorted(glob.glob("./test/*.png"))  # same order as loading
ids_parts = []
vals_parts = []

for i, f in enumerate(file_list):
    img_id = int(os.path.basename(f)[:-4])  # strip .png
    raw_img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    orig_h, orig_w = raw_img.shape

    pred_img = preds[i] if i < len(preds) else np.zeros((258, 540, 1), dtype=np.float32)
    pred_img = np.array(pred_img).squeeze()
    pred_resized = cv2.resize(pred_img, (orig_w, orig_h))

    rows = np.arange(1, orig_h + 1, dtype=np.int32)
    cols = np.arange(1, orig_w + 1, dtype=np.int32)
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")

    ids_img = np.char.add(
        np.char.add(str(img_id) + "_", row_grid.astype(str) + "_"),
        col_grid.astype(str),
    ).ravel()

    ids_parts.append(ids_img)
    vals_parts.append(pred_resized.ravel().astype(float))

print("Writing to csv file")
ids = np.concatenate(ids_parts)
vals = np.concatenate(vals_parts)
pd.DataFrame({"id": ids, "value": vals}).to_csv("submission.csv", index=False)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1974749850.py in <cell line: 0>()
      9 
     10     # Safeguard: ensure prediction exists and is a proper numpy array
---> 11     pred_img = preds[i] if i < len(preds) else np.zeros((258, 540, 1), dtype=np.float32)
     12     pred_img = np.array(pred_img).squeeze()
     13     # Resize prediction back to original dimensions

NameError: name 'preds' is not defined
