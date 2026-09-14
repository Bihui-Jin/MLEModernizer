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
Given simulated manufacturing control data, predict whether the machine is in state `0` or state `1`.

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
900000,0.65
900001,0.97
900002,0.02
etc.
```

## Dataset
- **train.csv** - the training data, which includes normalized continuous data and categorical data
- **test.csv** - the test set; your task is to predict binary `target` variable which represents the state of a manufacturing process
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        input/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        working/
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
```

-> data/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/tabular-playground-series-may-2022/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> data/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> input/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 5. Target score

0.95965

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import warnings, os

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from fastai.tabular.all import (
    TabularDataLoaders,
    FillMissing,
    Categorify,
    Normalize,
    tabular_learner,
    tabular_config,
    accuracy,
    RocAuc,
    CategoryBlock,
)

pd.options.display.float_format = "{:,.5f}".format
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 25)



## === cell 1
DATA_PATH = "/kaggle/input/tabular-playground-series-may-2022"
train_path = os.path.join(DATA_PATH, "train.csv")
test_path = os.path.join(DATA_PATH, "test.csv")
sample_path = os.path.join(DATA_PATH, "sample_submission.csv")

trn_data = pd.read_csv(train_path)
tst_data = pd.read_csv(test_path)
sub = pd.read_csv(sample_path)




## === cell 2
def reduce_memory_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type).startswith("int"):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            f"Mem. usage decreased to {end_mem:.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )
    return df


trn_data = reduce_memory_usage(trn_data, verbose=False)
tst_data = reduce_memory_usage(tst_data, verbose=False)




## === cell 3
def count_chars(df, field):
    for i in range(10):
        df[f"ch_{i}"] = (
            df[field]
            .astype(str)
            .str.get(i)
            .apply(lambda x: ord(x) - ord("A") if pd.notnull(x) else np.nan)
        )
    df["unique_characters"] = (
        df[field].astype(str).apply(lambda s: len(set(s)) if pd.notnull(s) else np.nan)
    )
    return df


def stat_features(df, cols):
    df["f_sum"] = df[cols].sum(axis=1)
    df["f_min"] = df[cols].min(axis=1)
    df["f_max"] = df[cols].max(axis=1)
    df["f_std"] = df[cols].std(axis=1)
    df["f_mad"] = df[cols].mad(axis=1)
    df["f_mean"] = df[cols].mean(axis=1)
    df["f_kurt"] = df[cols].kurt(axis=1)
    df["f_prod"] = df[cols].prod(axis=1)
    df["f_range"] = df["f_max"] - df["f_min"]
    df["f_count_pos"] = (df[cols] > 0).sum(axis=1)
    df["f_count_neg"] = (df[cols] < 0).sum(axis=1)
    return df




## === cell 4
trn_data = count_chars(trn_data, "f_27")
tst_data = count_chars(tst_data, "f_27")

orig_continuous = [
    "f_00",
    "f_01",
    "f_02",
    "f_03",
    "f_04",
    "f_05",
    "f_06",
    "f_19",
    "f_20",
    "f_21",
    "f_22",
    "f_23",
    "f_24",
    "f_25",
    "f_26",
    "f_28",
]

trn_data = stat_features(trn_data, orig_continuous)
tst_data = stat_features(tst_data, orig_continuous)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/977909562.py in <cell line: 0>()
     23 ]
     24 
---> 25 trn_data = stat_features(trn_data, orig_continuous)
     26 tst_data = stat_features(tst_data, orig_continuous)
     27 

/tmp/ipykernel_55/178685689.py in stat_features(df, cols)
     19     df["f_max"] = df[cols].max(axis=1)
     20     df["f_std"] = df[cols].std(axis=1)
---> 21     df["f_mad"] = df[cols].mad(axis=1)
     22     df["f_mean"] = df[cols].mean(axis=1)
     23     df["f_kurt"] = df[cols].kurt(axis=1)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'mad'

## === cell 5
continuous_feat = [
    "unique_characters",
    "f_06",
    "ch_7",
    "ch_0",
    "ch_8",
    "f_std",
    "f_range",
    "f_24",
    "f_min",
    "f_21",
    "ch_2",
    "f_03",
    "f_sum",
    "f_05",
    "f_count_neg",
    "f_22",
    "f_02",
    "ch_3",
    "f_26",
    "f_00",
    "ch_6",
    "f_23",
    "f_mean",
    "f_count_pos",
    "ch_9",
    "f_prod",
    "f_kurt",
    "ch_4",
    "f_mad",
    "f_max",
    "f_25",
    "f_04",
    "f_20",
    "f_19",
    "f_01",
    "f_28",
    "ch_1",
    "ch_5",
    "f_07",
    "f_08",
    "f_09",
    "f_10",
    "f_11",
    "f_12",
    "f_13",
    "f_14",
    "f_15",
    "f_16",
    "f_17",
    "f_18",
    "f_29",
    "f_30",
]

categorical_feat = []  # keep empty to avoid embeddings



## === cell 6
data_processing = [FillMissing, Categorify, Normalize]



## === cell 7
valid_pct = 0.10
batch_size = 1024

dls = TabularDataLoaders.from_df(
    df=trn_data,
    path=".",
    procs=data_processing,
    cat_names=categorical_feat,
    cont_names=continuous_feat,
    y_names="target",
    y_block=CategoryBlock,
    valid_pct=valid_pct,
    bs=batch_size,
    device="cuda" if torch.cuda.is_available() else "cpu",
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/635306263.py in <cell line: 0>()
     13     valid_pct=valid_pct,
     14     bs=batch_size,
---> 15     device="cuda" if torch.cuda.is_available() else "cpu",
     16 )
     17 

NameError: name 'torch' is not defined

## === cell 8
layers_definition = [256, 128, 64, 64, 16]
my_config = tabular_config(y_range=(0, 1))

learn = tabular_learner(
    dls,
    layers=layers_definition,
    emb_szs=None,  # no embeddings
    metrics=[accuracy, RocAuc()],
    config=my_config,
    loss_func=None,  # default CrossEntropyLoss for classification
).to_fp16()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2041353980.py in <cell line: 0>()
      4 
      5 learn = tabular_learner(
----> 6     dls,
      7     layers=layers_definition,
      8     emb_szs=None,  # no embeddings

NameError: name 'dls' is not defined

## === cell 9
learn.fit_one_cycle(3, 1e-3)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/426336544.py in <cell line: 0>()
      1 # Train the model (few epochs – sufficient for a baseline)
----> 2 learn.fit_one_cycle(3, 1e-3)
      3 

NameError: name 'learn' is not defined

## === cell 10
test_dl = learn.dls.test_dl(tst_data)
preds, _ = learn.get_preds(dl=test_dl)
prob_class1 = preds[:, 1].cpu().numpy()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1047016339.py in <cell line: 0>()
      1 # Predict on test set
----> 2 test_dl = learn.dls.test_dl(tst_data)
      3 preds, _ = learn.get_preds(dl=test_dl)
      4 # preds shape: (n_samples, n_classes); probability of class 1 is column 1
      5 prob_class1 = preds[:, 1].cpu().numpy()

NameError: name 'learn' is not defined

## === cell 11
sub["target"] = prob_class1
submission_path = "submission_fastai.csv"
sub.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4018610799.py in <cell line: 0>()
      1 # Build submission file
----> 2 sub["target"] = prob_class1
      3 submission_path = "submission_fastai.csv"
      4 sub.to_csv(submission_path, index=False)
      5 print(f"Submission written to {submission_path}")

NameError: name 'prob_class1' is not defined

## === cell 12
sub.head()
