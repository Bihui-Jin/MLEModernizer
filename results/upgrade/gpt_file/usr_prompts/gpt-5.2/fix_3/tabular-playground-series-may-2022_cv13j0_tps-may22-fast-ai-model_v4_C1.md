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
import os, warnings

warnings.filterwarnings("ignore")



## === cell 1
import numpy as np
import pandas as pd

from fastai.tabular.all import *

pd.options.display.float_format = "{:,.5f}".format



## === cell 2
for dirname, _, filenames in os.walk(
    "/kaggle/input/tabular-playground-series-may-2022"
):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))
    break



## === cell 3
trn_data = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/train.csv")
tst_data = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/test.csv")
sub = pd.read_csv(
    "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"
)




## === cell 4
def reduce_memory_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
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
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df




## === cell 5
def count_chars(df, field):
    """
    Extract per-position character codes and number of unique characters from a string field.
    """
    for i in range(10):
        df[f"ch_{i}"] = df[field].str.get(i).apply(ord) - ord("A")
    df["unique_characters"] = df[field].apply(lambda s: len(set(s)))
    return df


trn_data = count_chars(trn_data, "f_27")
tst_data = count_chars(tst_data, "f_27")



## === cell 6
continuous_feat_base = [
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


def stat_features(df, cols=continuous_feat_base):
    """
    Calculate aggregated features across selected continuous columns.

    BUGFIX: pandas 2.x removed DataFrame.mad(), so we compute mean absolute deviation
    explicitly to keep the original engineered feature 'f_mad' present.
    """
    df["f_sum"] = df[cols].sum(axis=1)
    df["f_min"] = df[cols].min(axis=1)
    df["f_max"] = df[cols].max(axis=1)
    df["f_std"] = df[cols].std(axis=1)

    row_mean = df[cols].mean(axis=1)
    df["f_mad"] = df[cols].sub(row_mean, axis=0).abs().mean(axis=1)

    df["f_mean"] = row_mean
    df["f_kurt"] = df[cols].kurt(axis=1)

    df["f_prod"] = df[cols].prod(axis=1)
    df["f_range"] = df["f_max"] - df["f_min"]
    df["f_count_pos"] = df[cols].gt(0).sum(axis=1)
    df["f_count_neg"] = df[cols].lt(0).sum(axis=1)
    return df


trn_data = stat_features(trn_data, continuous_feat_base)
tst_data = stat_features(tst_data, continuous_feat_base)



## === cell 7
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
categorical_feat = []  # avoid embeddings as originally intended

continuous_feat = [c for c in continuous_feat if c in trn_data.columns]



## === cell 8
data_processing = [
    FillMissing,
    Categorify,
    Normalize,
]

batch_size = 1024
valid_pct = 0.10

set_seed(42, reproducible=True)

data = TabularDataLoaders.from_df(
    df=trn_data,
    path=".",
    procs=data_processing,
    cat_names=categorical_feat,
    cont_names=continuous_feat,
    valid_pct=valid_pct,
    bs=batch_size,
    y_block=CategoryBlock,
    y_names="target",
)



## === cell 9
layers_definition = [256, 128, 64, 64, 16]
emb_size = None
my_config = tabular_config(y_range=(0, 1))

learn = tabular_learner(
    dls=data,
    layers=layers_definition,
    emb_szs=emb_size,
    metrics=[accuracy],
    config=my_config,
).to_fp16()



## === cell 10
learn.fit_one_cycle(1)



## === cell 11
_ = learn.lr_find()



## === cell 12
lr = 0.00120
learn.fit_one_cycle(3, lr_max=lr)



## === cell 13
learn.fine_tune(5, base_lr=lr, freeze_epochs=3)



## === cell 14
dl = learn.dls.test_dl(tst_data)

probs, _ = learn.get_preds(dl=dl)  # shape: (n,2)

pos_idx = (
    int(np.where(learn.dls.vocab == "1")[0][0]) if hasattr(learn.dls, "vocab") else 1
)
test_pred = probs[:, pos_idx].cpu().numpy()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/2480074904.py in <cell line: 0>()
      4 
      5 pos_idx = (
----> 6     int(np.where(learn.dls.vocab == "1")[0][0]) if hasattr(learn.dls, "vocab") else 1
      7 )
      8 test_pred = probs[:, pos_idx].cpu().numpy()

IndexError: index 0 is out of bounds for axis 0 with size 0

## === cell 15
sub["target"] = test_pred.astype(np.float64)
sub.to_csv("submission_fastai.csv", index=False)

print(sub.head())
print("Wrote:", os.path.abspath("submission_fastai.csv"), "rows:", len(sub))

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4046543138.py in <cell line: 0>()
----> 1 sub["target"] = test_pred.astype(np.float64)
      2 sub.to_csv("submission_fastai.csv", index=False)
      3 
      4 print(sub.head())
      5 print("Wrote:", os.path.abspath("submission_fastai.csv"), "rows:", len(sub))

NameError: name 'test_pred' is not defined
