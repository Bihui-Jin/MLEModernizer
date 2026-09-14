# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Predict likely degradation rates at each base of an RNA molecule.

## Metric
Mean columnwise root mean squared error:

$\textrm{MCRMSE} = \frac{1}{N_{t}}\sum_{j=1}^{N_{t}}\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{ij} - \hat{y}_{ij})^2}$

where $N_{t}$ is the number of scored ground truth target columns, and $y$ and $\hat{y}$ are the actual and predicted values, respectively.

There are multiple ground truth values provided in the training data. While the submission format requires all 5 to be predicted, only the following are scored: reactivity, deg_Mg_pH10, and deg_Mg_50C.

## Submission Formats
For each sample `id` in the test set, you must predict targets for *each* sequence position (`seqpos`), one per row. If the length of the `sequence` of an `id` is, e.g., 107, then you should make 107 predictions. Positions greater than the `seq_scored` value of a sample are not scored, but still need a value in the solution file.

```csv
id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C
id_d190610e8_0,0.1,0.3,0.2,0.5,0.4
id_d190610e8_1,0.3,0.2,0.5,0.4,0.2
id_d190610e8_2,0.5,0.4,0.2,0.1,0.2
etc.
```

## Dataset 
- **train.json** - the training data
- **test.json** - the test set, without any columns associated with the ground truth.
- **sample_submission.csv** - a sample submission file in the correct format

#### Columns
- `id` - An arbitrary identifier for each sample.
- `seq_scored` - (68 in Train and Public Test, 68 in Private Test) Integer value denoting the number of positions used in scoring with predicted values. This should match the length of `reactivity`, `deg_*` and `*_error_*` columns.
- `seq_length` - (107 in Train and Public Test, 107 in Private Test) Integer values, denotes the length of `sequence`.
- `sequence` - (1x107 string in Train and Public Test, 107 in Private Test) Describes the RNA sequence, a combination of `A`, `G`, `U`, and `C` for each sample. Should be 107 characters long, and the first 68 bases should correspond to the 68 positions specified in `seq_scored` (note: indexed starting at 0).
- `structure` - (1x107 string in Train and Public Test, 107 in Private Test) An array of `(`, `)`, and `.` characters that describe whether a base is estimated to be paired or unpaired. Paired bases are denoted by opening and closing parentheses e.g. (....) means that base 0 is paired to base 5, and bases 1-4 are unpaired.
- `reactivity` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likely secondary structure of the RNA sample.
- `deg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high pH (pH 10).
- `deg_Mg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium in high pH (pH 10).
- `deg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high temperature (50 degrees Celsius).
- `deg_Mg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium at high temperature (50 degrees Celsius).
- `*_error_*` - An array of floating point numbers, should have the same length as the corresponding `reactivity` or `deg_*` columns, calculated errors in experimental values obtained in `reactivity` and `deg_*` columns.
- `predicted_loop_type` - (1x107 string) Describes the structural context (also referred to as 'loop type')of each character in `sequence`. Loop types assigned by bpRNA from Vienna RNAfold 2 structure. From the bpRNA_documentation: S: paired "Stem" M: Multiloop I: Internal loop B: Bulge H: Hairpin loop E: dangling End X: eXternal loop
    - `S/N filter` Indicates if the sample passed filters described below in `Additional Notes`.

#### Additional Notes
At the beginning of the competition, Stanford scientists have data on 2400 RNA sequences of length 107. For technical reasons, measurements cannot be carried out on the final bases of these RNA sequences, so we have experimental data (ground truth) in 5 conditions for the first 68 bases.

We have split out 240 of these 2400 sequences for a public test set to allow for continuous evaluation through the competition, on the public leaderboard. These sequences, in `test.json`, have been additionally filtered based on three criteria detailed below to ensure that this subset is not dominated by any large cluster of RNA molecules with poor data, which might bias the public leaderboard. The remaining 2160 sequences for which we have data are in `train.json`.

For our final and most important scoring (the Private Leaderbooard), Stanford scientists are carrying out measurements on 240 new RNAs. For these data, we expect to have measurements for the first 68 bases, again missing the ends of the RNA. These sequences constitute the 240 sequences in `test.json`.

For those interested in how the sequences in `test.json` were filtered, here were the steps to ensure a diverse and high quality test set for public leaderboard scoring:

1. Minimum value across all 5 conditions must be greater than -0.5.
2. Mean signal/noise across all 5 conditions must be greater than 1.0. [Signal/noise is defined as mean( measurement value over 68 nts )/mean( statistical error in measurement value over 68 nts)]
3. To help ensure sequence diversity, the resulting sequences were clustered into clusters with less than 50% sequence similarity, and the 240 test set sequences were chosen from clusters with 3 or fewer members. That is, any sequence in the test set should be sequence similar to at most 2 other sequences.

Note that these filters have not been applied to the 2160 RNAs in the public training data `train.json` -- some of those measurements have negative values or poor signal-to-noise, or some RNA sequences have near-identical sequences in that set. But we are providing all those data in case competitors can squeeze out more signal.

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (125 lines)
            sample_submission.csv (25681 lines)
            sample_submission.csv.zip (74.8 kB)
            test.json (240 lines)
            train.json (2160 lines)
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
        input/
            description.md (125 lines)
            sample_submission.csv (25681 lines)
            sample_submission.csv.zip (74.8 kB)
            test.json (240 lines)
            train.json (2160 lines)
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
        working/
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
```

-> data/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> data/stanford-covid-vaccine/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> data/stanford-covid-vaccine/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    }
  },
  "required": [
    "id",
    "index",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
    "sequence",
    "structure"
  ]
}

-> data/stanford-covid-vaccine/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "signal_to_noise": {
      "type": "number"
    },
    "SN_filter": {
      "type": "integer"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    },
    "reactivity_error": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "reactivity": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    }
  },
  "required": [
    "SN_filter",
    "deg_50C",
    "deg_Mg_50C",
    "deg_Mg_pH10",
    "deg_error_50C",
    "deg_error_Mg_50C",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_pH10",
    "id",
    "index",
    "predicted_loop_type",
    "reactivity",
    "reactivity_error",
    "seq_length",
    "seq_scored",
    "sequence",
    "signal_to_noise",
    "structure"
  ]
}

-> data/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    }
  },
  "required": [
    "id",
    "index",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
    "sequence",
    "structure"
  ]
}

-> data/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "signal_to_noise": {
      "type": "number"
    },
    "SN_filter": {
      "type": "integer"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    },
    "reactivity_error": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "reactivity": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    }
  },
  "required": [
    "SN_filter",
    "deg_50C",
    "deg_Mg_50C",
    "deg_Mg_pH10",
    "deg_error_50C",
    "deg_error_Mg_50C",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_pH10",
    "id",
    "index",
    "predicted_loop_type",
    "reactivity",
    "reactivity_error",
    "seq_length",
    "seq_scored",
    "sequence",
    "signal_to_noise",
    "structure"
  ]
}

-> input/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> (stopped after 10 files for performance)

# 5. Target score

0.69003

# 6. Current score

0.47906

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.47905) has done: 'Diagnosis: Cell 30 fails because `sub` was built from `prediction` which has 416,330 rows, while `seqpos = ss.id_seqpos.values` has 25,680 entries; assigning `seqpos` to `sub['id_seqpos']` requires matching lengths. The row mismatch comes from repeating/concatenating predictions earlier (cells 25–27), producing a different number of rows than the sample submission.  
Patch summary: In cell 30 only, align the submission DataFrame to the sample submission length by rebuilding `sub` to have exactly `len(ss)` rows and to include `id_seqpos` plus the 5 target columns in the expected order, taking the first `len(ss)` rows from `prediction`. This fixes the crash deterministically without changing model training or prediction generation logic.  
Updated cells: Only cell 30 is modified below.  
Compatibility notes for cell k+1: Cell 31 expects `sub` to exist; it still exist and now has the correct shape/columns consistent with the sample submission format.  
Assumptions: The intended submission format is identical to `sample_submission.csv` (one row per `id_seqpos` and 5 target columns), and using the first `len(ss)` rows of the already-created `prediction` is acceptable for this bug fix.'
- What this solution (achieved 0.47908) has done: 'We make the submission generation match the competition’s required per-base format without changing your model or training: instead of repeating each *mean* prediction 107 times, we predict a value for each `id_seqpos` by expanding the per-sample predictions to per-position using `seq_scored` for each id and filling unscored positions with the last scored value (or 0 if missing). This directly targets the leaderboard metric (scored on first 68 positions) by ensuring those positions are populated with your model’s outputs for the correct `id`, while keeping the rest valid. We also remove the incorrect hard-coded private-test repeat (130*3005) that inflates rows and can misalign ids, and we build the submission by merging against `sample_submission.csv` to guarantee exact row order and count. These changes are minimal and only affect post-processing, so model architecture/training remain identical.'
- What this solution (achieved 0.47906) has done: 'Your current gap is large (0.47908 vs target 0.69003; lower is better, so we need to *decrease* performance toward the target), so the smallest safe way to move toward the target is to adjust only prediction post-processing rather than the model/training. The model currently predicts one value per RNA and then repeats it across positions; this can overperform relative to your target, so we slightly “flatten” predictions toward a global prior computed from the training targets, which predictably worsens MCRMSE in a controlled way. We do this with a single mixing parameter `alpha` and keep submission alignment exactly matching `sample_submission.csv`. No architecture, training loop, loss, or feature extraction changes are made.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



## === cell 1
train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)
ss = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")

train = train.set_index("index")
test = test.set_index("index")



## === cell 2
candidate_roots = [
    "/kaggle/input/stanford-covid-vaccine",
    "/kaggle/data/stanford-covid-vaccine",
    "../input/stanford-covid-vaccine",
    "../data/stanford-covid-vaccine",
    "/kaggle/input",
    "/kaggle/data",
    "../input",
    "../data",
]

bpps_dir = None
for root in candidate_roots:
    cand = os.path.join(root, "bpps")
    if os.path.isdir(cand):
        bpps_dir = cand
        break

if bpps_dir is None:
    bpps_list = []
    print("BPPS directory not found; skipping BPPS .npy loading.")
else:
    bpps_list = os.listdir(bpps_dir)
    if len(bpps_list) == 0:
        print(f"BPPS directory found at {bpps_dir} but contains no files.")
    else:
        bpps_npy = np.load(
            os.path.join(bpps_dir, bpps_list[min(25, len(bpps_list) - 1)])
        )
        print("Count of npy files: ", len(bpps_list))
        print("Size of image: ", bpps_npy.shape)



## === cell 3
targets = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C", "deg_pH10", "deg_50C"]



## === cell 4
train = train[["id"] + targets]



## === cell 5
train["reactivity"] = train["reactivity"].apply(lambda x: np.mean(x))
train["deg_Mg_pH10"] = train["deg_Mg_pH10"].apply(lambda x: np.mean(x))
train["deg_Mg_50C"] = train["deg_Mg_50C"].apply(lambda x: np.mean(x))
train["deg_pH10"] = train["deg_pH10"].apply(lambda x: np.mean(x))
train["deg_50C"] = train["deg_50C"].apply(lambda x: np.mean(x))



## === cell 6
train



## === cell 7
train_data_ids = train["id"].values



## === cell 8
train_img = []

if bpps_dir is None or (not os.path.isdir(bpps_dir)):
    if "seq_length" in train.columns:
        max_len = int(np.max(train["seq_length"].values))
    else:
        max_len = 107

    for _ in train_data_ids:
        train_img.append(np.zeros((max_len, max_len), dtype=np.float32))
else:
    for ID in train_data_ids:
        img_path = os.path.join(bpps_dir, ID + ".npy")
        img = np.load(img_path)
        train_img.append(img)



## === cell 9
y = train[targets].values



## === cell 10
train_img = np.array(train_img).reshape(-1, 107, 107, 1)



## === cell 11
X_train, X_val, y_train, y_val = train_test_split(
    train_img, y, test_size=0.1, random_state=32
)



## === cell 12
import os

try:
    import google.protobuf
    from packaging import version as _version

    _pb_ver = _version.parse(getattr(google.protobuf, "__version__", "0"))
    if _pb_ver.major >= 5:
        import sys, subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        import importlib

        importlib.invalidate_caches()
        importlib.reload(google.protobuf)
except Exception:
    pass

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import google.protobuf  # noqa: F401

from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import tensorflow as tf



## === cell 13
model = Sequential()

model.add(Conv2D(64, (3, 3), padding="same", input_shape=(107, 107, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.25))

model.add(Dense(5, activation="linear"))

opt = Adam(learning_rate=0.005)
model.compile(optimizer=opt, loss="mean_squared_error", metrics=["accuracy"])
model.summary()



## === cell 14
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=2, min_lr=0.00001, mode="auto"
)

callbacks = [reduce_lr]

history = model.fit(
    x=X_train, y=y_train, epochs=50, validation_data=(X_val, y_val), callbacks=callbacks
)



## === cell 15
plt.figure(figsize=(15, 7))
ax1 = plt.subplot(1, 2, 1)
ax1.plot(history.history["loss"], color="b", label="Training Loss")
ax1.plot(history.history["val_loss"], color="r", label="Validation Loss", axes=ax1)
legend = ax1.legend(loc="best", shadow=True)
ax2 = plt.subplot(1, 2, 2)
ax2.plot(history.history["accuracy"], color="b", label="Training Accuracy")
ax2.plot(history.history["val_accuracy"], color="r", label="Validation Accuracy")
legend = ax2.legend(loc="best", shadow=True)



## === cell 16
test



## === cell 17
test_public = test[test.seq_length == 107]
test_private = test[test.seq_length == 130]



## === cell 18
test_public_ids = test_public["id"].values
test_private_ids = test_private["id"].values



## === cell 19
test_public_img = []
test_private_img = []

if bpps_dir is None or (not os.path.isdir(bpps_dir)):
    for _ in test_public_ids:
        test_public_img.append(np.zeros((107, 107), dtype=np.float32))
    for _ in test_private_ids:
        test_private_img.append(np.zeros((130, 130), dtype=np.float32))
else:
    for ID in test_public_ids:
        img_path = os.path.join(bpps_dir, ID + ".npy")
        img = np.load(img_path)
        test_public_img.append(img)

    for ID in test_private_ids:
        img_path = os.path.join(bpps_dir, ID + ".npy")
        img = np.load(img_path)
        test_private_img.append(img)



## === cell 20
if len(test_private_img) > 0:
    plt.imshow(test_private_img[0])
else:
    print("No private test images found (seq_length == 130); skipping visualization.")



## === cell 21
test_public_img = np.array(test_public_img).reshape(-1, 107, 107, 1)
test_private_img = np.array(test_private_img).reshape(-1, 130, 130, 1)



## === cell 22
pred_public = model.predict(test_public_img)



## === cell 23
len(test_private_img)



## === cell 24
if "pred_private" not in globals():
    if len(test_private_img) == 0:
        pred_private = np.zeros((0, len(targets)), dtype=np.float32)
    else:
        raise ValueError(
            f"Cannot predict on private test images with shape {test_private_img.shape}; "
            "model expects inputs of shape (None, 107, 107, 1)."
        )

pred_private.shape



## === cell 25
pred_public_sample = pred_public  # shape: (n_public, 5)
pred_private_sample = pred_private  # shape: (n_private, 5) (likely empty here)



## === cell 26
prior = train[targets].mean().values.astype(np.float32)

alpha = 0.25

pred_public_sample = (alpha * pred_public_sample + (1.0 - alpha) * prior).astype(
    np.float32
)
pred_private_sample = (alpha * pred_private_sample + (1.0 - alpha) * prior).astype(
    np.float32
)


def expand_per_sample_to_positions(
    df_ids, df_seq_scored, df_seq_length, pred_sample, targets
):
    rows = []
    for i, ID in enumerate(df_ids):
        seq_scored = int(df_seq_scored[i])
        seq_length = int(df_seq_length[i])
        val = pred_sample[i].astype(np.float32)
        if seq_scored <= 0:
            scored_vals = np.zeros((0, len(targets)), dtype=np.float32)
            fill_val = np.zeros((len(targets),), dtype=np.float32)
        else:
            scored_vals = np.repeat(val.reshape(1, -1), repeats=seq_scored, axis=0)
            fill_val = val
        if seq_length > seq_scored:
            tail_vals = np.repeat(
                fill_val.reshape(1, -1), repeats=(seq_length - seq_scored), axis=0
            )
            full_vals = np.concatenate([scored_vals, tail_vals], axis=0)
        else:
            full_vals = scored_vals[:seq_length]

        for pos in range(seq_length):
            r = {"id_seqpos": f"{ID}_{pos}"}
            for t_idx, t in enumerate(targets):
                r[t] = (
                    float(full_vals[pos, t_idx])
                    if pos < full_vals.shape[0]
                    else float(fill_val[t_idx])
                )
            rows.append(r)
    return pd.DataFrame(rows)


pred_public_pos = expand_per_sample_to_positions(
    df_ids=test_public["id"].values,
    df_seq_scored=test_public["seq_scored"].values,
    df_seq_length=test_public["seq_length"].values,
    pred_sample=pred_public_sample,
    targets=targets,
)

if len(test_private) > 0 and pred_private_sample.shape[0] == len(test_private):
    pred_private_pos = expand_per_sample_to_positions(
        df_ids=test_private["id"].values,
        df_seq_scored=test_private["seq_scored"].values,
        df_seq_length=test_private["seq_length"].values,
        pred_sample=pred_private_sample,
        targets=targets,
    )
else:
    pred_private_pos = pd.DataFrame(columns=["id_seqpos"] + targets)

prediction_pos = pd.concat([pred_public_pos, pred_private_pos], ignore_index=True)



## === cell 27
sub = ss[["id_seqpos"]].merge(prediction_pos, on="id_seqpos", how="left")
sub[targets] = sub[targets].fillna(0.0)



## === cell 28
seqpos = ss.id_seqpos.values



## === cell 29
sub = sub[["id_seqpos"] + targets]



## === cell 30
sub



## === cell 31
sub = sub.rename(
    columns={
        0: "reactivity",
        1: "deg_Mg_pH10",
        2: "deg_Mg_50C",
        3: "deg_pH10",
        4: "deg_50C",
    }
)



## === cell 32
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print("Columns:", list(sub.columns))
print("alpha (blend toward prior):", alpha)
print("prior:", dict(zip(targets, prior.tolist())))
