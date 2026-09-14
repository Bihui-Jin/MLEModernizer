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
Predict whether an image contains a ship or an iceberg.

## Metric
Log loss.

## Submission Format
For each id in the test set, you must predict the probability that the image contains an iceberg (a number between 0 and 1). The file should contain a header and have the following format:

```
id,is_iceberg
809385f7,0.5
7535f0cd,0.4
3aa99a38,0.9
etc.
```

## Dataset
The labels are provided by human experts and geographic knowledge on the target. All the images are 75x75 images with two bands.

The data (`train.json`, `test.json`) is presented in `json` format.

The files consist of a list of images, and for each image, you can find the following fields:

- **id** - the id of the image
- **band_1, band_2** - the [flattened](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.flatten.html) image data. Each band has 75x75 pixel values in the list, so the list has 5625 elements. Note that these values are not the normal non-negative integers in image files since they have physical meanings - these are **float** numbers with unit being [dB](https://en.wikipedia.org/wiki/Decibel). Band 1 and Band 2 are signals characterized by radar backscatter produced from different polarizations at a particular incidence angle. The polarizations correspond to HH (transmit/receive horizontally) and HV (transmit horizontally and receive vertically).
- **inc_angle** - the incidence angle of which the image was taken. Note that this field has missing data marked as "na", and those images with "na" incidence angles are all in the training data to prevent leakage.
- **is_iceberg** - the target variable, set to 1 if it is an iceberg, and 0 if it is a ship. This field only exists in `train.json`.

Please note that we have included machine-generated images in the test set to prevent hand labeling. They are excluded in scoring.

sample_submission.csv: The submission file in the correct format:

# 2. Python version

3.6

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
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (99 lines)
            sample_submission.csv (322 lines)
            sample_submission.csv.7z (2.0 kB)
            sample_submission.csv.zip (2.0 kB)
            test.json (1 lines)
            test.json.7z (8.9 MB)
            train.json (1 lines)
            train.json.7z (35.8 MB)
            statoil-iceberg-classifier-challenge/
                description.md (99 lines)
                sample_submission.csv (322 lines)
                ... and 6 other files
                statoil-iceberg-classifier-challenge/
        input/
            description.md (99 lines)
            sample_submission.csv (322 lines)
            sample_submission.csv.7z (2.0 kB)
            sample_submission.csv.zip (2.0 kB)
            test.json (1 lines)
            test.json.7z (8.9 MB)
            train.json (1 lines)
            train.json.7z (35.8 MB)
            statoil-iceberg-classifier-challenge/
                description.md (99 lines)
                sample_submission.csv (322 lines)
                ... and 6 other files
                statoil-iceberg-classifier-challenge/
        working/
            statoil-iceberg-classifier-challenge/
                description.md (99 lines)
                sample_submission.csv (322 lines)
                ... and 6 other files
                statoil-iceberg-classifier-challenge/
```

-> data/sample_submission.csv has 321 rows and 2 columns.
The columns are: id, is_iceberg

-> data/statoil-iceberg-classifier-challenge/sample_submission.csv has 321 rows and 2 columns.
The columns are: id, is_iceberg

-> data/statoil-iceberg-classifier-challenge/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle"
    ]
  }
}

-> data/statoil-iceberg-classifier-challenge/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      },
      "is_iceberg": {
        "type": "integer"
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle",
      "is_iceberg"
    ]
  }
}

-> data/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle"
    ]
  }
}

-> data/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      },
      "is_iceberg": {
        "type": "integer"
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle",
      "is_iceberg"
    ]
  }
}

-> input/sample_submission.csv has 321 rows and 2 columns.
The columns are: id, is_iceberg

-> (stopped after 10 files for performance)

# 5. Target score

0.31379

# 6. Current score

0.40606

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.40445) has done: 'I remove the failing `skimage.util.montage` import (it is not available in this environment) and replace montage visualizations with a small local montage helper so the exploratory cells run without extra dependencies. I also fix the input path to point at the provided Kaggle dataset location and make the script resilient by not depending on earlier plotting cells having run. Finally, I fix the Keras utility import (`to_categorical`) to avoid the protobuf-related error and ensure the pipeline trains, predicts, and writes a valid `submission.csv` with the required `id,is_iceberg` columns.'
- What this solution (achieved 0.45563) has done: 'I fix the protobuf-related crash by avoiding `tf_keras.utils.to_categorical` and instead building the 2-column one-hot targets with NumPy (same semantics, no dependency on the failing protobuf path). I also add lightweight determinism (seeds + TF deterministic ops where available) to stabilize training and slightly improve logloss without changing the model or training loop. Finally, I ensure the loss is numerically aligned with the softmax+one-hot setup by using `categorical_crossentropy` (same objective intent, but correct pairing), and keep the submission format/columns exactly as required.'
- What this solution (achieved 0.40606) has done: 'I fix the TensorFlow/Keras protobuf crash by removing the dependency on `tensorflow`/`tf_keras` (which is triggering `MessageFactory.GetPrototype`) and instead using the already-installed standalone `keras==3.8.0` API to build and train the exact same CNN architecture. I keep the model layers, loss, optimizer, epochs, and data split semantics identical so the core logic and evaluation intent remain unchanged, while ensuring the notebook runs end-to-end. I also make the import paths consistent for Keras 3 and keep the submission writing exactly in `id,is_iceberg` format to produce a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 2017
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

CANDIDATE_BASE_PATHS = [
    "/kaggle/input/statoil-iceberg-classifier-challenge",
    "/kaggle/input",
    os.path.join("..", "input"),
    os.path.join("..", "data"),
]
base_path = None
for p in CANDIDATE_BASE_PATHS:
    if os.path.exists(os.path.join(p, "train.json")) and os.path.exists(
        os.path.join(p, "test.json")
    ):
        base_path = p
        break
if base_path is None:
    p = "/kaggle/input/statoil-iceberg-classifier-challenge/statoil-iceberg-classifier-challenge"
    if os.path.exists(os.path.join(p, "train.json")):
        base_path = p
    else:
        raise FileNotFoundError(
            "Could not locate train.json/test.json under expected Kaggle input paths."
        )

print("Using base_path:", base_path)


def montage2d_local(img_stack, grid_shape=None, fill_value=0.0):
    """
    img_stack: (N, H, W) array
    returns: (grid_H, grid_W) montage image
    """
    img_stack = np.asarray(img_stack)
    if img_stack.ndim != 3:
        raise ValueError("montage2d_local expects a 3D array (N, H, W).")
    n, h, w = img_stack.shape
    if grid_shape is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    else:
        rows, cols = grid_shape
    out = np.full((rows * h, cols * w), fill_value, dtype=img_stack.dtype)
    for i in range(n):
        r, c = divmod(i, cols)
        if r >= rows:
            break
        out[r * h : (r + 1) * h, c * w : (c + 1) * w] = img_stack[i]
    return out




## === cell 1
def load_and_format(in_path):
    out_df = pd.read_json(in_path)

    if "inc_angle" in out_df.columns:
        out_df["inc_angle"] = pd.to_numeric(out_df["inc_angle"], errors="coerce")

    out_images = out_df.apply(
        lambda c_row: [
            np.stack([c_row["band_1"], c_row["band_2"]], -1).reshape((75, 75, 2))
        ],
        axis=1,
    )
    out_images = np.stack(out_images).squeeze().astype(np.float32)
    return out_df, out_images


train_df, train_images = load_and_format(os.path.join(base_path, "train.json"))
print("training", train_df.shape, "loaded", train_images.shape)
test_df, test_images = load_and_format(os.path.join(base_path, "test.json"))
print("testing", test_df.shape, "loaded", test_images.shape)
train_df.sample(3, random_state=SEED)



## === cell 2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.matshow(train_images[0, :, :, 0])
ax1.set_title("Band 1")
ax2.matshow(train_images[0, :, :, 1])
ax2.set_title("Band 2")
plt.tight_layout()
plt.show()



## === cell 3
fig, (ax1s, ax2s) = plt.subplots(2, 2, figsize=(8, 8))
obj_list = dict(
    ships=train_df.query("is_iceberg==0").sample(16, random_state=SEED).index,
    icebergs=train_df.query("is_iceberg==1").sample(16, random_state=SEED).index,
)
for ax1, ax2, (obj_type, idx_list) in zip(ax1s, ax2s, obj_list.items()):
    ax1.imshow(montage2d_local(train_images[idx_list, :, :, 0]))
    ax1.set_title("%s Band 1" % obj_type)
    ax1.axis("off")
    ax2.imshow(montage2d_local(train_images[idx_list, :, :, 1]))
    ax2.set_title("%s Band 2" % obj_type)
    ax2.axis("off")
plt.tight_layout()
plt.show()



## === cell 4
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
idx_list = test_df.sample(49, random_state=SEED).index
obj_type = "Test Data"
ax1.imshow(montage2d_local(test_images[idx_list, :, :, 0]))
ax1.set_title("%s Band 1" % obj_type)
ax1.axis("off")
ax2.imshow(montage2d_local(test_images[idx_list, :, :, 1]))
ax2.set_title("%s Band 2" % obj_type)
ax2.axis("off")
plt.tight_layout()
plt.show()



## === cell 5
from sklearn.model_selection import train_test_split

y = train_df["is_iceberg"].values.astype(np.int64)
y_onehot = np.eye(2, dtype=np.float32)[y]  # shape (N,2)

X_train, X_test, y_train, y_test = train_test_split(
    train_images,
    y_onehot,
    random_state=SEED,
    test_size=0.5,
    stratify=y,
)

print("Train", X_train.shape, y_train.shape)
print("Validation", X_test.shape, y_test.shape)



## === cell 6
import keras
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    BatchNormalization,
    Dropout,
    MaxPooling2D,
    GlobalMaxPooling2D,
    Dense,
)

try:
    keras.utils.set_random_seed(SEED)
except Exception:
    pass

simple_cnn = Sequential()
simple_cnn.add(BatchNormalization(input_shape=(75, 75, 2)))
for i in range(4):
    simple_cnn.add(Conv2D(8 * 2**i, kernel_size=(3, 3)))
    simple_cnn.add(MaxPooling2D((2, 2)))
simple_cnn.add(GlobalMaxPooling2D())
simple_cnn.add(Dropout(0.5))
simple_cnn.add(Dense(8))
simple_cnn.add(Dense(2, activation="softmax"))

simple_cnn.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
simple_cnn.summary()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7
history = simple_cnn.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    shuffle=True,
    verbose=2,
)



## === cell 8
test_predictions = simple_cnn.predict(test_images, verbose=0)

iceberg_prob = np.clip(test_predictions[:, 1].astype(np.float64), 0.0, 1.0)

pred_df = test_df[["id"]].copy()
pred_df["is_iceberg"] = iceberg_prob

submission_path = "submission.csv"
pred_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "shape:", pred_df.shape)
pred_df.head(3)
