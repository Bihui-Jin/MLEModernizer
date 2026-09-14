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

3.7

# 3. Installed packages

No external packages required in the script and installed.

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

0.3247

# 6. Current score

0.47099

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.4167) has done: 'The changes replace the relatively slow `GradientBoostingClassifier` with the much faster `HistGradientBoostingClassifier`, which implements the same gradient‑boosted‑tree idea but is highly optimized for large, dense feature matrices.  The feature‑building code and all preprocessing steps remain unchanged, and the model’s hyper‑parameters (number of trees, learning rate, depth, random state) are kept equivalent, preserving the original learning behavior while cutting execution time well below the 600 s limit.'
- What this solution (achieved 0.44667) has done: 'The patch tightens the Gradient Boosting model to be more expressive (more trees, slightly deeper) and uses a lower learning rate plus class balancing, which should reduce log‑loss and move the validation score closer to the target 0.3247 while keeping the overall pipeline unchanged. No other parts of the code are altered.'
- What this solution (achieved 0.47099) has done: 'I add a few simple statistical image features (min and max of each band) to the existing feature set, and make the HistGradientBoosting model a bit more expressive by raising the number of trees, decreasing the learning rate, and allowing slightly deeper trees. These changes keep the overall pipeline and model type intact while providing richer information for the learner, which should lower the validation log‑loss and move the score closer to the target 0.3247.'

# 9. Code solution

## === cell 0
import os, glob, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler



## === cell 1
train_matches = glob.glob(os.path.join("**", "train.json"), recursive=True)
test_matches = glob.glob(os.path.join("**", "test.json"), recursive=True)

if not train_matches:
    raise FileNotFoundError("train.json not found in the directory tree")
if not test_matches:
    raise FileNotFoundError("test.json not found in the directory tree")

train_path = train_matches[0]
test_path = test_matches[0]

with open(train_path, "r") as f:
    train_data = json.load(f)
with open(test_path, "r") as f:
    test_data = json.load(f)

train = pd.DataFrame(train_data)
test = pd.DataFrame(test_data)


def build_features(df, is_train=True):
    """
    Build image‑based features.
    Existing flat pixel values plus simple statistics (mean, std, min, max) for each band
    and the incidence angle.
    """
    band1 = np.asarray(df["band_1"].tolist(), dtype=np.float32).reshape(-1, 75, 75)
    band2 = np.asarray(df["band_2"].tolist(), dtype=np.float32).reshape(-1, 75, 75)

    combined = (band1 + band2) / 2.0
    X_img = np.stack([band1, band2, combined], axis=-1)  # (n,75,75,3)

    n_samples = X_img.shape[0]
    X_flat = X_img.reshape(n_samples, -1)  # raw pixels

    b1_mean = band1.mean(axis=(1, 2)).astype(np.float32)
    b2_mean = band2.mean(axis=(1, 2)).astype(np.float32)
    b1_std = band1.std(axis=(1, 2)).astype(np.float32)
    b2_std = band2.std(axis=(1, 2)).astype(np.float32)
    b1_min = band1.min(axis=(1, 2)).astype(np.float32)
    b2_min = band2.min(axis=(1, 2)).astype(np.float32)
    b1_max = band1.max(axis=(1, 2)).astype(np.float32)
    b2_max = band2.max(axis=(1, 2)).astype(np.float32)

    inc_angle = pd.to_numeric(df["inc_angle"], errors="coerce")
    inc_filled = inc_angle.fillna(inc_angle.median()).values.astype(np.float32)

    extra_feats = np.column_stack(
        [b1_mean, b2_mean, b1_std, b2_std, b1_min, b2_min, b1_max, b2_max, inc_filled]
    )

    X_full = np.concatenate([X_flat, extra_feats], axis=1).astype(np.float32)

    if is_train:
        y = df["is_iceberg"].values
        return X_full, y
    else:
        return X_full


X_full, y = build_features(train, is_train=True)



## === cell 2
x_train, x_val, y_train, y_val = train_test_split(
    X_full, y, test_size=0.2, random_state=1, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train).astype(np.float32)
x_val = scaler.transform(x_val).astype(np.float32)



## === cell 3
model = HistGradientBoostingClassifier(
    max_iter=800,  # more trees
    learning_rate=0.02,  # finer updates
    max_depth=6,  # slightly deeper trees
    class_weight="balanced",
    random_state=42,
    verbose=0,
)
model.fit(x_train, y_train)

val_pred = model.predict_proba(x_val)[:, 1]
print("Validation LogLoss:", log_loss(y_val, val_pred))



## === cell 4
X_test_full = build_features(test, is_train=False)
X_test_full = scaler.transform(X_test_full).astype(np.float32)

test_pred = model.predict_proba(X_test_full)[:, 1]

submission = pd.DataFrame({"id": test["id"], "is_iceberg": test_pred})
submission_path = "IceBerg_FCN_sub.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")




## === cell 5
def show_placeholder():
    print(
        "Training completed with HistGradientBoostingClassifier; no Keras history available."
    )


show_placeholder()
