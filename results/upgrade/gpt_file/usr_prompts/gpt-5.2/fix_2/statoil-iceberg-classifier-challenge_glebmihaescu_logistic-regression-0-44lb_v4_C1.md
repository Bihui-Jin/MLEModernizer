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

3.6

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

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

0.69959

# 6. Current score

0.42207

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.42207) has done: 'I fix the runtime error by making `inc_angle` consistently numeric in both train and test (converting `"na"` to NaN) and filling missing values with the training mean so `LogisticRegression` receives only floats. I also fix a logic bug in feature construction: the training loop mistakenly reuses one `mas1` image for every row; it should recompute `mas1` per sample exactly like the test loop. Finally, I ensure the script always writes a valid `sub.csv` with the required `id,is_iceberg` columns after successful prediction.'

# 9. Code solution

## === cell 0
import pandas as pd

pd.options.display.max_columns = 999
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from pylab import rcParams

rcParams["figure.figsize"] = 8, 8

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



## === cell 1
train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")

train["inc_angle"] = pd.to_numeric(train["inc_angle"], errors="coerce")
test["inc_angle"] = pd.to_numeric(test["inc_angle"], errors="coerce")



## === cell 2
i = 28
if train.is_iceberg[i] == 1:
    print("iceberg")
else:
    print("not_iceberg")
rcParams["figure.figsize"] = 8, 8

k = 2.5

mas2 = np.array(train.band_2[i])
mas1 = np.array(train.band_1[i])

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.matshow(mas1.reshape(75, 75))
ax1.grid(True)
ax2.matshow(mas2.reshape(75, 75))
ax2.grid(True)

fig, (ax3, ax4) = plt.subplots(1, 2)

ax3.matshow(
    ((mas1 > ((np.max(mas1) + np.min(mas1)) / k).astype(int)) * (-mas1)).reshape(75, 75)
)
ax3.grid(True)
ax4.matshow(
    ((mas2 > ((np.max(mas2) + np.min(mas2)) / k).astype(int)) * (-mas2)).reshape(75, 75)
)
ax4.grid(True)

plt.show()



## === cell 3
k = 2.5



## === cell 4
supertrain1 = []
for i in range(train.shape[0]):
    mas1 = np.array(train.band_1[i])
    supertrain1.append(
        ((mas1 > (np.max(mas1) + np.min(mas1)) / k).astype(int)) * (-mas1)
    )

train_band_1 = pd.DataFrame(
    supertrain1,
    columns=[("(" + str(i) + "," + str(j) + ")") for i in range(75) for j in range(75)],
)
train_band_1["inc_angle"] = train["inc_angle"]

train_inc_mean = train_band_1["inc_angle"].mean()
train_band_1["inc_angle"].fillna(train_inc_mean, inplace=True)



## === cell 5
Y = train.is_iceberg



## === cell 6
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold



## === cell 7
supertest1 = []
for i in range(test.shape[0]):
    mas1 = np.array(test.band_1[i])
    supertest1.append(
        ((mas1 > (np.max(mas1) + np.min(mas1)) / k).astype(int)) * (-mas1)
    )

test_band_1 = pd.DataFrame(
    supertest1,
    columns=[("(" + str(i) + "," + str(j) + ")") for i in range(75) for j in range(75)],
)
test_band_1["inc_angle"] = test["inc_angle"]

test_band_1["inc_angle"].fillna(train_inc_mean, inplace=True)



## === cell 8
model = LogisticRegression(penalty="l2", C=0.0004, random_state=100)
model.fit(train_band_1, Y)

predict = model.predict_proba(test_band_1)[:, 1]
sub = pd.DataFrame({"id": test.id, "is_iceberg": predict})

sub = sub[["id", "is_iceberg"]]



## === cell 9
sub.to_csv("sub.csv", index=False)
print("Wrote submission to sub.csv with shape:", sub.shape)
print(sub.head())
