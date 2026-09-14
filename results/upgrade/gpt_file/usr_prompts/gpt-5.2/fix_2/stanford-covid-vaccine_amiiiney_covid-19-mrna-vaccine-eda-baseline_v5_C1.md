# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tf_keras==2.18.0

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

0.52825

# 6. Current score

0.45443

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.45443) has done: 'I fix the two plotting cells that crash due to Seaborn/Pandas API expectations (string categories and correlation on non-numeric columns) so the notebook can continue. Then I remove the incompatible mixed `keras` / `tf.keras` imports that trigger the protobuf `MessageFactory` error in this environment, and use only `tf.keras` so `Sequential`, `Dense`, and `Dropout` are defined and training runs. Finally, I fix the test feature building so it matches the submission’s required 107 positions per `id` (not only 68/91) and ensure train/test one-hot columns are aligned before inference, producing a complete `submission.csv` with correct columns.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter(action="ignore")

train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)
sub = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")


def plotd(f1, f2):
    plt.style.use("seaborn-v0_8")
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(15, 5))
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    plt.hist(a[f1], bins=4, color="black", alpha=0.5)
    plt.title(f"{f1}", weight="bold", fontsize=18)
    ax1 = plt.subplot2grid((1, 2), (0, 1))
    plt.hist(a[f2], bins=4, color="crimson", alpha=0.5)
    plt.title(f"{f2}", weight="bold", fontsize=18)
    plt.show()


def plotc(f1, f2):
    plt.style.use("seaborn-v0_8")
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(15, 5))
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    plt.hist(a[f1], bins=7, color="black", alpha=0.7)
    plt.title(f"{f1}", weight="bold", fontsize=18)
    ax1 = plt.subplot2grid((1, 2), (0, 1))
    plt.hist(a[f2], bins=5, color="crimson", alpha=0.7)
    plt.title(f"{f2}", weight="bold", fontsize=18)
    plt.xticks(weight="bold")
    plt.show()


def ploth(data, w=15, h=9):
    plt.figure(figsize=(w, h))
    corr = data.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, cmap="hot", annot=False)
    plt.title("Correlation between the features", fontsize=18, weight="bold")
    plt.xticks(weight="bold")
    plt.yticks(weight="bold")
    return plt.show()




## === cell 1
train.head()




## === cell 2
def length(feature):
    column = train[[feature]].copy()
    column["length"] = column[feature].apply(len)
    return column.head()


length("sequence")



## === cell 3
length("reactivity")



## === cell 4
train_data = []
for mol_id in train["id"].unique():
    sample_data = train.loc[train["id"] == mol_id]
    for i in range(68):
        sample_tuple = (
            sample_data["id"].values[0],
            sample_data["sequence"].values[0][i],
            sample_data["structure"].values[0][i],
            sample_data["predicted_loop_type"].values[0][i],
            sample_data["reactivity"].values[0][i],
            sample_data["reactivity_error"].values[0][i],
            sample_data["deg_Mg_pH10"].values[0][i],
            sample_data["deg_error_Mg_pH10"].values[0][i],
            sample_data["deg_pH10"].values[0][i],
            sample_data["deg_error_pH10"].values[0][i],
            sample_data["deg_Mg_50C"].values[0][i],
            sample_data["deg_error_Mg_50C"].values[0][i],
            sample_data["deg_50C"].values[0][i],
            sample_data["deg_error_50C"].values[0][i],
        )
        train_data.append(sample_tuple)



## === cell 5
a = pd.DataFrame(
    train_data,
    columns=[
        "id",
        "sequence",
        "structure",
        "predicted_loop_type",
        "reactivity",
        "reactivity_error",
        "deg_Mg_pH10",
        "deg_error_Mg_pH10",
        "deg_pH10",
        "deg_error_pH10",
        "deg_Mg_50C",
        "deg_error_Mg_50C",
        "deg_50C",
        "deg_error_50C",
    ],
)
a.head()



## === cell 6
plotd("reactivity", "reactivity_error")



## === cell 7
plotd("deg_50C", "deg_Mg_50C")



## === cell 8
plotd("deg_pH10", "deg_Mg_pH10")



## === cell 9
plotc("predicted_loop_type", "structure")



## === cell 10
sns.countplot(x=a["sequence"], palette="terrain", alpha=0.8)
plt.title("Nucleotides count per sequence", weight="bold", fontsize=12)
plt.show()



## === cell 11
b = a[["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]]
ploth(b, 10, 4)



## === cell 12
c = a[["id", "sequence", "structure", "predicted_loop_type"]].copy()
c = pd.get_dummies(c, columns=["sequence", "structure", "predicted_loop_type"])
ploth(c)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/945321280.py in <cell line: 0>()
      1 c = a[["id", "sequence", "structure", "predicted_loop_type"]].copy()
      2 c = pd.get_dummies(c, columns=["sequence", "structure", "predicted_loop_type"])
----> 3 ploth(c)
      4 

/tmp/ipykernel_11/3838781478.py in ploth(data, w, h)
     43     plt.figure(figsize=(w, h))
     44     corr = data.select_dtypes(include=[np.number]).corr()
---> 45     sns.heatmap(corr, cmap="hot", annot=False)
     46     plt.title("Correlation between the features", fontsize=18, weight="bold")
     47     plt.xticks(weight="bold")

/usr/local/lib/python3.11/dist-packages/seaborn/matrix.py in heatmap(data, vmin, vmax, cmap, center, robust, annot, fmt, annot_kws, linewidths, linecolor, cbar, cbar_kws, cbar_ax, square, xticklabels, yticklabels, mask, ax, **kwargs)
    444     """
    445     # Initialize the plotter object
--> 446     plotter = _HeatMapper(data, vmin, vmax, cmap, center, robust, annot, fmt,
    447                           annot_kws, cbar, cbar_kws, xticklabels,
    448                           yticklabels, mask)

/usr/local/lib/python3.11/dist-packages/seaborn/matrix.py in __init__(self, data, vmin, vmax, cmap, center, robust, annot, fmt, annot_kws, cbar, cbar_kws, xticklabels, yticklabels, mask)
    161 
    162         # Determine good default values for the colormapping
--> 163         self._determine_cmap_params(plot_data, vmin, vmax,
    164                                     cmap, center, robust)
    165 

/usr/local/lib/python3.11/dist-packages/seaborn/matrix.py in _determine_cmap_params(self, plot_data, vmin, vmax, cmap, center, robust)
    200                 vmin = np.nanpercentile(calc_data, 2)
    201             else:
--> 202                 vmin = np.nanmin(calc_data)
    203         if vmax is None:
    204             if robust:

/usr/local/lib/python3.11/dist-packages/numpy/lib/nanfunctions.py in nanmin(a, axis, out, keepdims, initial, where)
    341         # Fast, but not safe for subclasses of ndarray, or object arrays,
    342         # which do not implement isnan (gh-9009), or fmin correctly (gh-8975)
--> 343         res = np.fmin.reduce(a, axis=axis, out=out, **kwargs)
    344         if np.isnan(res).any():
    345             warnings.warn("All-NaN slice encountered", RuntimeWarning,

ValueError: zero-size array to reduction operation fmin which has no identity

## === cell 13
test_data = []
for mol_id in test["id"].unique():
    sample_data = test.loc[test["id"] == mol_id]
    seq = sample_data["sequence"].values[0]
    struct = sample_data["structure"].values[0]
    loop = sample_data["predicted_loop_type"].values[0]
    seq_len = int(sample_data["seq_length"].values[0])
    for i in range(seq_len):
        sample_tuple = (
            sample_data["id"].values[0] + "_" + str(i),
            seq[i],
            struct[i],
            loop[i],
        )
        test_data.append(sample_tuple)

tdf = pd.DataFrame(
    test_data, columns=["id_seqpos", "sequence", "structure", "predicted_loop_type"]
)

X = c.drop("id", axis=1)
X_test = pd.get_dummies(
    tdf, columns=["sequence", "structure", "predicted_loop_type"]
).drop("id_seqpos", axis=1)

X_test = X_test.reindex(columns=X.columns, fill_value=0)



## === cell 14
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout


def build_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(
        Dense(
            1024, input_dim=n_inputs, kernel_initializer="he_uniform", activation="relu"
        )
    )
    model.add(Dropout(0.5))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(n_outputs))
    model.compile(loss="mae", optimizer="adam")
    return model


callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(),
    tf.keras.callbacks.ModelCheckpoint("model.keras", save_best_only=False),
]
n_inputs, n_outputs = X.shape[1], b.shape[1]
model = build_model(n_inputs, n_outputs)
history = model.fit(
    X,
    b,
    batch_size=128,
    epochs=50,
    callbacks=callbacks,
    validation_split=0.3,
    verbose=2,
)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 15
plt.style.use("seaborn-v0_8")
sns.set_style("whitegrid")
fig = plt.figure(figsize=(15, 5))
train_loss = history.history["loss"]
test_loss = history.history["val_loss"]
x = list(range(1, len(test_loss) + 1))
plt.plot(x, test_loss, color="cyan", label="Validation loss")
plt.plot(x, train_loss, label="Training loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("MAE Loss")
plt.title("Loss vs. Epoch", weight="bold", fontsize=18)
plt.show()



## === cell 16
test_preds = model.predict(X_test, verbose=0)



## === cell 17
pred_df = pd.DataFrame(test_preds, columns=b.columns)
pred_df["id_seqpos"] = tdf["id_seqpos"].values



## === cell 18
final = sub[["id_seqpos"]].merge(pred_df, on="id_seqpos", how="left")

final[["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]] = final[
    ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
].fillna(0.0)

submission = final



## === cell 19
submission.head()



## === cell 20
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
