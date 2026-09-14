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

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from skimage.util import montage2d  # older versions
except ImportError:
    from skimage.util import montage as montage2d  # newer versions
base_path = os.getenv("KAGGLE_INPUT_DIR", os.path.join(os.getcwd(), "input"))




## === cell 1
def load_and_format(in_path):
    """Load the json file and reshape the two bands into (75,75,2) images."""
    df = pd.read_json(in_path)
    images = np.stack(
        df.apply(
            lambda r: np.stack([r["band_1"], r["band_2"]], axis=-1).reshape(75, 75, 2),
            axis=1,
        ).values
    )
    return df, images


train_df, train_images = load_and_format(os.path.join(base_path, "train.json"))
print("training", train_df.shape, "loaded", train_images.shape)
test_df, test_images = load_and_format(os.path.join(base_path, "test.json"))
print("testing", test_df.shape, "loaded", test_images.shape)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3446528820.py in <cell line: 0>()
     11 
     12 
---> 13 train_df, train_images = load_and_format(os.path.join(base_path, "train.json"))
     14 print("training", train_df.shape, "loaded", train_images.shape)
     15 test_df, test_images = load_and_format(os.path.join(base_path, "test.json"))

/tmp/ipykernel_55/3446528820.py in load_and_format(in_path)
      1 def load_and_format(in_path):
      2     """Load the json file and reshape the two bands into (75,75,2) images."""
----> 3     df = pd.read_json(in_path)
      4     images = np.stack(
      5         df.apply(

/usr/local/lib/python3.11/dist-packages/pandas/io/json/_json.py in read_json(path_or_buf, orient, typ, dtype, convert_axes, convert_dates, keep_default_dates, precise_float, date_unit, encoding, encoding_errors, lines, chunksize, compression, nrows, storage_options, dtype_backend, engine)
    789         convert_axes = True
    790 
--> 791     json_reader = JsonReader(
    792         path_or_buf,
    793         orient=orient,

/usr/local/lib/python3.11/dist-packages/pandas/io/json/_json.py in __init__(self, filepath_or_buffer, orient, typ, dtype, convert_axes, convert_dates, keep_default_dates, precise_float, date_unit, encoding, lines, chunksize, compression, nrows, storage_options, encoding_errors, dtype_backend, engine)
    902             self.data = filepath_or_buffer
    903         elif self.engine == "ujson":
--> 904             data = self._get_data_from_filepath(filepath_or_buffer)
    905             self.data = self._preprocess_data(data)
    906 

/usr/local/lib/python3.11/dist-packages/pandas/io/json/_json.py in _get_data_from_filepath(self, filepath_or_buffer)
    958             and not file_exists(filepath_or_buffer)
    959         ):
--> 960             raise FileNotFoundError(f"File {filepath_or_buffer} does not exist")
    961         else:
    962             warnings.warn(

FileNotFoundError: File /kaggle/working/input/train.json does not exist

## === cell 2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.matshow(train_images[0, :, :, 0])
ax1.set_title("Band 1")
ax2.matshow(train_images[0, :, :, 1])
ax2.set_title("Band 2")
plt.close(fig)  # close to avoid display in non‑interactive runs



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2621914267.py in <cell line: 0>()
      1 # Simple visual sanity‑check (optional, does not affect training)
      2 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
----> 3 ax1.matshow(train_images[0, :, :, 0])
      4 ax1.set_title("Band 1")
      5 ax2.matshow(train_images[0, :, :, 1])

NameError: name 'train_images' is not defined

## === cell 3
fig, (ax1s, ax2s) = plt.subplots(2, 2, figsize=(8, 8))
obj_list = dict(
    ships=train_df.query("is_iceberg==0").sample(16, random_state=42).index,
    icebergs=train_df.query("is_iceberg==1").sample(16, random_state=42).index,
)
for ax1, ax2, (obj_type, idx_list) in zip(ax1s, ax2s, obj_list.items()):
    ax1.imshow(montage2d(train_images[idx_list, :, :, 0]))
    ax1.set_title(f"{obj_type} Band 1")
    ax1.axis("off")
    ax2.imshow(montage2d(train_images[idx_list, :, :, 1]))
    ax2.set_title(f"{obj_type} Band 2")
    ax2.axis("off")
plt.close(fig)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1141528599.py in <cell line: 0>()
      2 fig, (ax1s, ax2s) = plt.subplots(2, 2, figsize=(8, 8))
      3 obj_list = dict(
----> 4     ships=train_df.query("is_iceberg==0").sample(16, random_state=42).index,
      5     icebergs=train_df.query("is_iceberg==1").sample(16, random_state=42).index,
      6 )

NameError: name 'train_df' is not defined

## === cell 4
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical  # use keras utils to avoid TF protobuf issue

X_train, X_val, y_train, y_val = train_test_split(
    train_images,
    to_categorical(train_df["is_iceberg"]),
    test_size=0.5,
    random_state=2017,
    shuffle=True,
)
print("Train", X_train.shape, y_train.shape)
print("Validation", X_val.shape, y_val.shape)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    BatchNormalization,
    Dropout,
    MaxPooling2D,
    GlobalMaxPooling2D,
    Dense,
)

simple_cnn = Sequential()
simple_cnn.add(BatchNormalization(input_shape=(75, 75, 2)))
for i in range(4):
    simple_cnn.add(Conv2D(8 * (2**i), kernel_size=(3, 3), activation="relu"))
    simple_cnn.add(MaxPooling2D(pool_size=(2, 2)))
simple_cnn.add(GlobalMaxPooling2D())
simple_cnn.add(Dropout(0.5))
simple_cnn.add(Dense(8, activation="relu"))
simple_cnn.add(Dense(2, activation="softmax"))

simple_cnn.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
simple_cnn.summary()



## === cell 6
simple_cnn.fit(
    X_train, y_train, validation_data=(X_val, y_val), epochs=10, shuffle=True, verbose=2
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2950515967.py in <cell line: 0>()
      1 simple_cnn.fit(
----> 2     X_train, y_train, validation_data=(X_val, y_val), epochs=10, shuffle=True, verbose=2
      3 )
      4 

NameError: name 'X_train' is not defined

## === cell 7
test_predictions = simple_cnn.predict(test_images, verbose=0)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4036987914.py in <cell line: 0>()
----> 1 test_predictions = simple_cnn.predict(test_images, verbose=0)
      2 

NameError: name 'test_images' is not defined

## === cell 8
pred_df = test_df[["id"]].copy()
pred_df["is_iceberg"] = test_predictions[:, 1]  # probability of class 1 (iceberg)
pred_df.to_csv("predictions.csv", index=False)
print("Saved predictions to predictions.csv")
pred_df.sample(3)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/904215376.py in <cell line: 0>()
----> 1 pred_df = test_df[["id"]].copy()
      2 pred_df["is_iceberg"] = test_predictions[:, 1]  # probability of class 1 (iceberg)
      3 pred_df.to_csv("predictions.csv", index=False)
      4 print("Saved predictions to predictions.csv")
      5 pred_df.sample(3)

NameError: name 'test_df' is not defined
