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
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
                train_cleaned/
                    182.png (55.7 kB)
                    153.png (58.7 kB)
                    ... and 113 other files
        input/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
                train_cleaned/
                    182.png (55.7 kB)
                    153.png (58.7 kB)
                    ... and 113 other files
            test/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
            train/
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
                train_cleaned/
                    182.png (55.7 kB)
                    153.png (58.7 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
Here is some information about the columns:
id (object) has 2000 unique values. Some example values: ['110_1_1', '110_3_250', '110_3_263', '110_3_262']
value (int64) has 1 unique values: [1]

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
Here is some information about the columns:
id (object) has 2000 unique values. Some example values: ['110_1_1', '110_3_250', '110_3_263', '110_3_262']
value (int64) has 1 unique values: [1]

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
Here is some information about the columns:
id (object) has 2000 unique values. Some example values: ['110_1_1', '110_3_250', '110_3_263', '110_3_262']
value (int64) has 1 unique values: [1]

# 5. Target score

0.03254

# 6. Current score

0.47627

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.28616) has done: 'The fix updates the import of Keras to use TensorFlow’s bundled version (avoiding the protobuf error), correctly converts loaded PIL images to NumPy arrays in `images_to_array`, and corrects the pixel‑flattening order when building the submission vector so that the predicted values align with the required `id` ordering. These changes resolve the runtime crash and substantially improve the RMSE by ensuring proper data handling and submission formatting.'
- What this solution (achieved 0.22077) has done: 'I set the protobuf environment variable before importing TensorFlow to avoid the import error, and I make the submission‑building step robust by removing the strict length assertion and safely truncating or padding the prediction vector so it always matches the sample‑submission IDs. This ensures the notebook runs end‑to‑end and writes a valid *.csv* file while keeping the core model unchanged.'
- What this solution (achieved 0.22089) has done: 'The fix adds a model checkpoint that saves the weights with the lowest validation loss and reloads them before making predictions. This prevents later over‑fitting from degrading the denoising performance, lowering the RMSE toward the target while keeping the original architecture and training loop unchanged.'
- What this solution (achieved 0.28616) has done: 'The fix adds all missing imports, defines the dataset paths, and ensures the required Keras/TensorFlow objects are available before they are used. This resolves the `NameError` issues, lets the model train and predict, and guarantees that a correctly‑formatted `submission.csv` is written at the end.'
- What this solution (achieved 0.28616) has done: 'To fix the protobuf import error we set the required environment variable **before** importing TensorFlow.  
We also give the model more training data (20 % validation instead of 30 %) and allow a longer early‑stopping patience (20 epochs) so it can converge better, which should lower the RMSE toward the target while keeping the original architecture intact.'
- What this solution (achieved 0.28616) has done: 'We move the protobuf‑environment setting to the very top (before any imports) so TensorFlow loads correctly, tighten the validation split to 10 % (giving the model more data to learn), and give the training loop more room to converge (longer patience and more epochs). These fixes resolve the runtime error and let the auto‑encoder train better, moving the RMSE toward the target while keeping the original architecture unchanged.'
- What this solution (achieved 0.47627) has done: 'Implemented fixes to eliminate the protobuf import error, shuffle the dataset before creating train/validation splits for better learning, and extended training capacity with a higher epoch limit and adjusted early‑stopping patience. These changes ensure the notebook runs fully, produces a correctly‑formatted CSV submission, and nudges the RMSE toward the target score.'

# 9. Code solution

## === cell 0
perm = np.random.permutation(X.shape[0])
X = X[perm]
y = y[perm]

val_split = int(0.1 * X.shape[0])
X_val, y_val = X[:val_split], y[:val_split]
X_train, y_train = X[val_split:], y[val_split:]

print("Train data shape :", X_train.shape)
print("Validation data shape :", X_val.shape)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2192797745.py in <cell line: 0>()
      1 # Shuffle data before splitting to ensure a representative validation set
----> 2 perm = np.random.permutation(X.shape[0])
      3 X = X[perm]
      4 y = y[perm]
      5 

NameError: name 'np' is not defined

## === cell 1
lr_callback = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.4, patience=4, verbose=1, min_lr=1e-5
)

early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=100, restore_best_weights=True, verbose=1
)

checkpoint_path = "best_model.weights.h5"
ckpt_callback = callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    mode="min",
    verbose=1,
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=1000,
    batch_size=16,
    callbacks=[lr_callback, early_stop, ckpt_callback],
    verbose=2,
)

model.load_weights(checkpoint_path)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/34752116.py in <cell line: 0>()
----> 1 lr_callback = callbacks.ReduceLROnPlateau(
      2     monitor="val_loss", factor=0.4, patience=4, verbose=1, min_lr=1e-5
      3 )
      4 
      5 # Allow more epochs to improve convergence while still restoring the best weights

NameError: name 'callbacks' is not defined

## === cell 2
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import Input, Model, callbacks
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

_possible_paths = [
    "/kaggle/input/denoising-dirty-documents",
    "/kaggle/working/denoising-dirty-documents",
    "/kaggle/working",
]
BASE_DIR = next(p for p in _possible_paths if os.path.isdir(p))

TRAIN_DIR = os.path.join(BASE_DIR, "train")
TRAIN_CLEANED_DIR = os.path.join(BASE_DIR, "train_cleaned")
TEST_DIR = os.path.join(BASE_DIR, "test")
SAMPLE_SUBMISSION_PATH = os.path.join(BASE_DIR, "sampleSubmission.csv")

first_train_image = os.listdir(TRAIN_DIR)[0]
sample_img = load_img(
    os.path.join(TRAIN_DIR, first_train_image), color_mode="grayscale"
)
img_width, img_height = sample_img.size  # width, height
IMG_SIZE = (img_height, img_width)  # (height, width)


def images_to_array(data_dir, label_dir=None, img_size=IMG_SIZE):
    """
    Load grayscale images from `data_dir` (and optionally matching labels from `label_dir`)
    into a NumPy array of shape (N, H, W, 1) normalized to [0, 1].
    """
    image_names = sorted(os.listdir(data_dir))
    n = len(image_names)
    X = np.zeros((n, img_size[0], img_size[1]), dtype=np.uint8)

    for i, name in enumerate(tqdm(image_names, desc="Loading images")):
        img_path = os.path.join(data_dir, name)
        img = load_img(img_path, color_mode="grayscale", target_size=img_size)
        arr = np.array(img)
        if arr.ndim == 3:  # (H, W, 1) -> squeeze
            arr = arr.squeeze()
        X[i] = arr

    X = X.reshape(n, img_size[0], img_size[1], 1).astype(np.float32) / 255.0

    if label_dir:
        label_names = sorted(os.listdir(label_dir))
        y = np.zeros((len(label_names), img_size[0], img_size[1]), dtype=np.uint8)
        for i, name in enumerate(tqdm(label_names, desc="Loading labels")):
            lbl_path = os.path.join(label_dir, name)
            lbl = load_img(lbl_path, color_mode="grayscale", target_size=img_size)
            arr = np.array(lbl)
            if arr.ndim == 3:
                arr = arr.squeeze()
            y[i] = arr
        y = (
            y.reshape(len(label_names), img_size[0], img_size[1], 1).astype(np.float32)
            / 255.0
        )

        perm = np.random.permutation(n)
        X, y = X[perm], y[perm]
        print("Output Data Shape :", X.shape)
        print("Output Label Shape:", y.shape)
        return X, y

    print("Output Data Shape :", X.shape)
    return X


X, y = images_to_array(TRAIN_DIR, label_dir=TRAIN_CLEANED_DIR)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
samples = np.concatenate((X_train[:3], y_train[:3]), axis=0)
f, ax = plt.subplots(2, 3, figsize=(12, 6))
for i, img in enumerate(samples):
    ax[i // 3, i % 3].imshow(img[:, :, 0], cmap="gray")
    ax[i // 3, i % 3].axis("off")
plt.show()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/507310332.py in <cell line: 0>()
----> 1 samples = np.concatenate((X_train[:3], y_train[:3]), axis=0)
      2 f, ax = plt.subplots(2, 3, figsize=(12, 6))
      3 for i, img in enumerate(samples):
      4     ax[i // 3, i % 3].imshow(img[:, :, 0], cmap="gray")
      5     ax[i // 3, i % 3].axis("off")

NameError: name 'X_train' is not defined

## === cell 4
tf.random.set_seed(42)
np.random.seed(42)

input_layer = Input(shape=(None, None, 1))

x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), padding="same")(x)
x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), padding="same")(x)

x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)

x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)

output_layer = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer="adam", loss="mean_squared_error")



## === cell 5
test_image_names = sorted(os.listdir(TEST_DIR))
X_test = []
for name in tqdm(test_image_names, desc="Preparing test set"):
    img_path = os.path.join(TEST_DIR, name)
    img = load_img(img_path, color_mode="grayscale", target_size=IMG_SIZE)
    arr = (
        np.array(img).reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1).astype(np.float32) / 255.0
    )
    X_test.append(arr)
X_test = np.concatenate(X_test, axis=0)
print("Test set shape:", X_test.shape)



## === cell 6
preds = model.predict(X_test, batch_size=16, verbose=0)  # (N, H, W, 1)

submit_vector = []
for img_pred in preds:  # img_pred shape (H, W, 1)
    flat = img_pred[:, :, 0].ravel(order="C")  # rows first, then columns
    submit_vector.extend(flat.tolist())

print("Total prediction values:", len(submit_vector))

sample_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)

if len(submit_vector) != len(sample_df):
    diff = len(sample_df) - len(submit_vector)
    if diff > 0:
        mean_val = np.mean(submit_vector) if submit_vector else 0.5
        submit_vector.extend([mean_val] * diff)
    else:
        submit_vector = submit_vector[: len(sample_df)]

submission = pd.DataFrame({"id": sample_df["id"], "value": submit_vector})



## === cell 7
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
