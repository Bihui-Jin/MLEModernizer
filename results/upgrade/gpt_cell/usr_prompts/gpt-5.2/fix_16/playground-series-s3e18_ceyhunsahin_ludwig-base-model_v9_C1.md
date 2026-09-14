# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.11

# 2. Installed packages

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
seaborn==0.12.2
sklearn-pandas==2.2.0

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (1485 lines)
            sample_submission.csv.zip (4.4 kB)
            test.csv (1485 lines)
            test.csv.zip (147.8 kB)
            train.csv (13355 lines)
            train.csv.zip (1.4 MB)
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
        input/
            description.md (56 lines)
            sample_submission.csv (1485 lines)
            sample_submission.csv.zip (4.4 kB)
            test.csv (1485 lines)
            test.csv.zip (147.8 kB)
            train.csv (13355 lines)
            train.csv.zip (1.4 MB)
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
        working/
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
```

-> data/playground-series-s3e18/sample_submission.csv has 1484 rows and 3 columns.
The columns are: id, EC1, EC2

-> data/playground-series-s3e18/test.csv has 1484 rows and 32 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 17 more columns

-> data/playground-series-s3e18/train.csv has 13354 rows and 38 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 23 more columns

-> data/sample_submission.csv has 1484 rows and 3 columns.
The columns are: id, EC1, EC2

-> data/test.csv has 1484 rows and 32 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 17 more columns

-> data/train.csv has 13354 rows and 38 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 23 more columns

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import pandas as pd
df_train = pd.read_csv("/kaggle/input/playground-series-s3e18/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s3e18/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s3e18/sample_submission.csv")


## === cell 2
df_train.head()


## === cell 3

my_palette = sns.cubehelix_palette(n_colors = 7, start=.46, rot=-.45, dark = .2, hue=0.95)
sns.palplot(my_palette)
plt.gcf().set_size_inches(13,2)

for idx,values in enumerate(my_palette.as_hex()):
    plt.text(idx-0.375,0, my_palette.as_hex()[idx],{'font': "Courier New", 'size':16, 'weight':'bold','color':'black'}, alpha =0.7)
plt.gcf().set_facecolor('white')

plt.show()


## === cell 4
df_train.describe().T


## === cell 5
df_train.info()


## === cell 6
df_train[['MaxAbsEStateIndex','MinEStateIndex']]


## === cell 7
df_train.drop('id', axis=1, inplace = True)
df_test.drop('id', axis=1, inplace = True)


## === cell 8
target_col1 = 'EC1'
target_col2 = 'EC2'
num_cols = [
    'BertzCT',
    'Chi1','Chi1n','Chi1v','Chi2n','Chi2v','Chi3v','Chi4n','EState_VSA1','EState_VSA2',
    'ExactMolWt','FpDensityMorgan1','FpDensityMorgan2','FpDensityMorgan3',
    'HallKierAlpha','HeavyAtomMolWt','Kappa3','MaxAbsEStateIndex','MinEStateIndex','NumHeteroatoms',
    'PEOE_VSA10','PEOE_VSA14','PEOE_VSA6','PEOE_VSA7','PEOE_VSA8','SMR_VSA10','SMR_VSA5',
    'SlogP_VSA3','VSA_EState9',
]
binary_cols = [
    'EC1',
    'EC2',
    'EC3',
    'EC4',
    'EC5',
    'EC6'
]
cat_cols = df_test.select_dtypes(include=['object']).columns.tolist()


print(f'[INFO] Shapes:'
      f'\n train: {df_train.shape}'
      f'\n test: {df_test.shape}\n')

print(f'[INFO] Any missing values:'

      f'\n train: {df_train.isna().any().any()}'
      f'\n test: {df_test.isna().any().any()}')


## === cell 9
plt.figure(figsize = (14, 8))
sns.set_style('white')

colors = my_palette

plt.barh(df_train[target_col1].value_counts().index,
        df_train[target_col1].value_counts(),
        color = colors[1:3])

plt.title('EC1 Distribution in df_train', fontsize = 14, fontweight = 'bold')

sns.despine()

plt.show()


## === cell 10
plt.barh(df_train[target_col2].value_counts().index,
        df_train[target_col2].value_counts(),
        color = colors[1:3])

plt.title('EC1 Distribution in df_train', fontsize = 14, fontweight = 'bold')

sns.despine()

plt.show()


## === cell 11
!pip install ludwig


## === cell 12
from sklearn.preprocessing import power_transform
df_tr_copy = df_train.copy()
df_tst_copy = df_test.copy()

df_tr_copy.iloc[:,:] = power_transform(df_tr_copy.iloc[:,:] , method = "yeo-johnson")
df_tst_copy.iloc[:,:] = power_transform(df_tst_copy.iloc[:,:] , method = "yeo-johnson")


## === cell 13
def out_iqr(df):
    columns = df.columns 
    iqrs,lower_out , upper_out = [],[],[]
    for column in columns:
        q25, q75 = np.quantile(df[column], 0.25), np.quantile(df[column], 0.75)
        iqr = q75 - q25
        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off
        df1 = df[df[column] > upper].shape[0]
        df2 = df[df[column] < lower].shape[0]
        iqrs.append(iqr)
        upper_out.append(df1)
        lower_out.append(df2)
    return pd.DataFrame(
        data = {
            "columns": df.columns,
            "iqr":iqrs,
            "lower_outliers":lower_out,
            "upper_outliers": upper_out,
        }
    ).style.background_gradient(axis=0, cmap="Greens")


## === cell 14
out_iqr(df_tr_copy)


## === cell 15
numerical_columns = num_cols

fig, axes = plt.subplots(len(numerical_columns), 2, figsize=(20, 40))

for i, column in enumerate(numerical_columns):
    sns.histplot(df_tr_copy[column], bins=30, kde=True, ax=axes[i, 0], color = my_palette[2])
    axes[i, 0].set_title(f'Distribution of {column} in df_train')
    axes[i, 0].set_xlabel('Value')
    axes[i, 0].set_ylabel('Frequency')

    sns.boxplot(df_train[column], ax=axes[i, 1], color = my_palette[1])
    axes[i, 1].set_title(f'Box plot of {column} in df_train')
    axes[i, 1].set_xlabel(column)
    axes[i, 1].set_ylabel('Value')

plt.tight_layout()
plt.show()


## === cell 16
import sys
import types
import importlib.machinery


def _ensure_module(name: str) -> types.ModuleType:
    if name in sys.modules:
        mod = sys.modules[name]
        if getattr(mod, "__spec__", None) is None:
            mod.__spec__ = importlib.machinery.ModuleSpec(name=name, loader=None)
        return mod
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name=name, loader=None)
    sys.modules[name] = mod
    return mod


bnb_stub = _ensure_module("bitsandbytes")
bnb_stub.__dict__.setdefault("__version__", "0.0")

bnb_optim_stub = _ensure_module("bitsandbytes.optim")
bnb_optim_stub.__dict__.setdefault("COMPILED_WITH_CUDA", False)


class _Dummy8BitOptimizer:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "bitsandbytes is not available in this environment; 8-bit/paged optimizers cannot be used."
        )


for _name in [
    "Lion",
    "Adam",
    "AdamW",
    "SGD",
    "RMSprop",
    "Adagrad",
    "SGD8bit",
    "Adam8bit",
    "AdamW8bit",
    "Adagrad8bit",
    "RMSprop8bit",
    "Lion8bit",
    "LARS",
    "LARS8bit",
    "LAMB",
    "LAMB8bit",
    "PagedAdam",
    "PagedAdamW",
    "PagedLion",
    "PagedAdam8bit",
    "PagedAdamW8bit",
    "PagedLion8bit",
    "PagedAdam32bit",
    "PagedAdamW32bit",
    "PagedLion32bit",
]:
    if not hasattr(bnb_optim_stub, _name):
        setattr(bnb_optim_stub, _name, type(_name, (_Dummy8BitOptimizer,), {}))

bnb_nn_stub = _ensure_module("bitsandbytes.nn")
bnb_nn_modules_stub = _ensure_module("bitsandbytes.nn.modules")


class _DummyEmbedding:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "bitsandbytes is not available in this environment; Embedding cannot be used."
        )


if not hasattr(bnb_nn_modules_stub, "Embedding"):
    bnb_nn_modules_stub.Embedding = _DummyEmbedding


class _DummyLinear4bit:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "bitsandbytes is not available in this environment; Linear4bit cannot be used."
        )


if not hasattr(bnb_nn_modules_stub, "Linear4bit"):
    bnb_nn_modules_stub.Linear4bit = _DummyLinear4bit

bnb_functional_stub = _ensure_module("bitsandbytes.functional")


def _dequantize_4bit(*args, **kwargs):
    raise RuntimeError(
        "bitsandbytes is not available in this environment; dequantize_4bit cannot be used."
    )


if not hasattr(bnb_functional_stub, "dequantize_4bit"):
    bnb_functional_stub.dequantize_4bit = _dequantize_4bit

bnb_stub.optim = bnb_optim_stub
bnb_stub.nn = bnb_nn_stub
bnb_stub.functional = bnb_functional_stub
bnb_nn_stub.modules = bnb_nn_modules_stub

if "torchtext" not in sys.modules:
    torchtext_stub = types.ModuleType("torchtext")
    torchtext_stub.__dict__["__version__"] = "0.0"
    torchtext_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="torchtext", loader=None
    )
    sys.modules["torchtext"] = torchtext_stub
else:
    mod = sys.modules["torchtext"]
    if getattr(mod, "__spec__", None) is None:
        mod.__spec__ = importlib.machinery.ModuleSpec(name="torchtext", loader=None)

try:
    from google.protobuf import message_factory as _message_factory

    _MF = getattr(_message_factory, "MessageFactory", None)
    if _MF is not None and not hasattr(_MF, "GetPrototype"):
        if hasattr(_MF, "GetMessageClass"):

            def _GetPrototype(self, descriptor):
                return self.GetMessageClass(descriptor)

            _MF.GetPrototype = _GetPrototype
        elif hasattr(_message_factory, "GetMessageClass"):

            def _GetPrototype(self, descriptor):
                return _message_factory.GetMessageClass(descriptor)

            _MF.GetPrototype = _GetPrototype
except Exception:
    pass

import ludwig
from ludwig.api import LudwigModel
import logging


## === cell 17
def configs(par, sample_key, sample_item):
    config = {
        "combiner": {
            "dropout": 0.2,
            "num_fc_layers": 3,
            "output_size": 256,
            "type": "concat",
        },
        "input_features": [
            {"name": "BertzCT", "type": "number"},
            {"name": "Chi1", "type": "number"},
            {"name": "Chi1n", "type": "number"},
            {"name": "Chi1v", "type": "number"},
            {"name": "Chi2n", "type": "number"},
            {"name": "Chi2v", "type": "number"},
            {"name": "Chi3v", "type": "number"},
            {"name": "Chi4n", "type": "number"},
            {"name": "EState_VSA1", "type": "number"},
            {"name": "EState_VSA2", "type": "number"},
            {"name": "ExactMolWt", "type": "number"},
            {"name": "FpDensityMorgan1", "type": "number"},
            {"name": "FpDensityMorgan2", "type": "number"},
            {"name": "FpDensityMorgan3", "type": "number"},
            {"name": "HallKierAlpha", "type": "number"},
            {"name": "HeavyAtomMolWt", "type": "number"},
            {"name": "Kappa3", "type": "number"},
            {"name": "MaxAbsEStateIndex", "type": "number"},
            {"name": "MinEStateIndex", "type": "number"},
            {"name": "NumHeteroatoms", "type": "number"},
            {"name": "PEOE_VSA10", "type": "number"},
            {"name": "PEOE_VSA14", "type": "number"},
            {"name": "PEOE_VSA6", "type": "number"},
            {"name": "PEOE_VSA7", "type": "number"},
            {"name": "PEOE_VSA8", "type": "number"},
            {"name": "SMR_VSA10", "type": "number"},
            {"name": "SMR_VSA5", "type": "number"},
            {"name": "SlogP_VSA3", "type": "number"},
            {"name": "VSA_EState9", "type": "number"},
            {"name": "fr_COO", "type": "category"},
            {"name": "fr_COO2", "type": "category"},
        ],
        "output_features": [
            {
                "name": par,
                "encoder": {
                    "activations": "relu",
                    "weights_initializer": "xavier_uniform",
                },
                "decoder": {
                    "num_fc_layers": 4,
                    "fc_activation": "relu",
                    "output_size": 1,
                },
                "preprocessing": {"fallback_true_label": "1"},
                "loss": {"type": "binary_weighted_cross_entropy"},
                "type": "binary",
            }
        ],
        "defaults": {
            "number": {
                "preprocessing": {
                    "missing_value_strategy": "fill_with_mean",
                    "normalization": "zscore",
                    sample_key: sample_item,
                }
            }
        },
        "trainer": {"epochs": 100, "optimizer": {"type": "adam"}},
        "hyperopt": {
            "executor": {"num_samples": 16},
            "parameters": {
                f"{par}.decoder.num_fc_layers": {
                    "space": "randint",
                    "lower": 2,
                    "upper": 10,
                },
                "trainer.learning_rate": {
                    "space": "loguniform",
                    "lower": 0.0001,
                    "upper": 0.1,
                },
            },
            "search_alg": {
                "type": "hyperopt",
                "random_state": 42,
            },
            "goal": "maximize",
            "metric": "roc_auc",
            "output_feature": par,
        },
    }
    return config


config1 = configs("EC1", "oversample_minority", 0.5)

model = LudwigModel(config=config1, logging_level=logging.INFO)


## === cell 18
train_stats1, preprocessed_data1, output_directory1 = model.train(training_set=df_train,
                                                               test_set=df_test)


## === cell 19
predictions1, prediction_results1 = model.predict(dataset=df_test, skip_save_predictions=False, output_directory="predictions_results")


## === cell 20
predictions1


## === cell 21
config2 = configs('EC2','undersample_majority', 0.7)


## === cell 22
model = LudwigModel(config=config2, logging_level=logging.INFO)


## === cell 23
train_stats2, preprocessed_data2, output_directory2 = model.train(training_set=df_train,
                                                               test_set=df_test)


## === cell 24
predictions2, prediction_results2 = model.predict(dataset=df_test, skip_save_predictions=False, output_directory="predictions_results2")


## === cell 25
predictions2


## === cell 26
sample_submission['EC1'] = predictions1['EC1_probabilities_True'].values
sample_submission['EC2'] = predictions2['EC2_probabilities_True'].values
sample_submission.to_csv(f'submission.csv', index=False)
sample_submission


## --- ERROR in cell 26, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1686627247.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0msample_submission[0m[0;34m[[0m[0;34m'EC1'[0m[0;34m][0m [0;34m=[0m [0mpredictions1[0m[0;34m[[0m[0;34m'EC1_probabilities_True'[0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0msample_submission[0m[0;34m[[0m[0;34m'EC2'[0m[0;34m][0m [0;34m=[0m [0mpredictions2[0m[0;34m[[0m[0;34m'EC2_probabilities_True'[0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 3[0;31m [0msample_submission[0m[0;34m.[0m[0mto_csv[0m[0;34m([0m[0;34mf'submission.csv'[0m[0;34m,[0m [0mindex[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0msample_submission[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/util/_decorators.py[0m in [0;36mwrapper[0;34m(*args, **kwargs)[0m
[1;32m    331[0m                     [0mstacklevel[0m[0;34m=[0m[0mfind_stack_level[0m[0;34m([0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    332[0m                 )
[0;32m--> 333[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    334[0m [0;34m[0m[0m
[1;32m    335[0m         [0;31m# error: "Callable[[VarArg(Any), KwArg(Any)], Any]" has no[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36mto_csv[0;34m(self, path_or_buf, sep, na_rep, float_format, columns, header, index, index_label, mode, encoding, compression, quoting, quotechar, lineterminator, chunksize, date_format, doublequote, escapechar, decimal, errors, storage_options)[0m
[1;32m   3965[0m         [0mReturn[0m [0mthe[0m [0melements[0m [0;32min[0m [0mthe[0m [0mgiven[0m [0;34m*[0m[0mpositional[0m[0;34m*[0m [0mindices[0m [0malong[0m [0man[0m [0maxis[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3966[0m [0;34m[0m[0m
[0;32m-> 3967[0;31m         [0mThis[0m [0mmeans[0m [0mthat[0m [0mwe[0m [0mare[0m [0;32mnot[0m [0mindexing[0m [0maccording[0m [0mto[0m [0mactual[0m [0mvalues[0m [0;32min[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3968[0m         [0mthe[0m [0mindex[0m [0mattribute[0m [0mof[0m [0mthe[0m [0mobject[0m[0;34m.[0m [0mWe[0m [0mare[0m [0mindexing[0m [0maccording[0m [0mto[0m [0mthe[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3969[0m         [0mactual[0m [0mposition[0m [0mof[0m [0mthe[0m [0melement[0m [0;32min[0m [0mthe[0m [0mobject[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/formats/format.py[0m in [0;36mto_csv[0;34m(self, path_or_buf, encoding, sep, columns, index_label, mode, compression, quoting, quotechar, lineterminator, chunksize, date_format, doublequote, escapechar, errors, storage_options)[0m
[1;32m    993[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    994[0m             [0;32mreturn[0m [0madjoined[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 995[0;31m [0;34m[0m[0m
[0m[1;32m    996[0m     [0;32mdef[0m [0m_get_column_name_list[0m[0;34m([0m[0mself[0m[0;34m)[0m [0;34m->[0m [0mlist[0m[0;34m[[0m[0mHashable[0m[0;34m][0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    997[0m         [0mnames[0m[0;34m:[0m [0mlist[0m[0;34m[[0m[0mHashable[0m[0;34m][0m [0;34m=[0m [0;34m[[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/formats/csvs.py[0m in [0;36m__init__[0;34m(self, formatter, path_or_buf, sep, cols, index_label, mode, encoding, errors, compression, quoting, lineterminator, chunksize, quotechar, date_format, doublequote, escapechar, storage_options)[0m
[1;32m     94[0m         [0mself[0m[0;34m.[0m[0mlineterminator[0m [0;34m=[0m [0mlineterminator[0m [0;32mor[0m [0mos[0m[0;34m.[0m[0mlinesep[0m[0;34m[0m[0;34m[0m[0m
[1;32m     95[0m         [0mself[0m[0;34m.[0m[0mdate_format[0m [0;34m=[0m [0mdate_format[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 96[0;31m         [0mself[0m[0;34m.[0m[0mcols[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_initialize_columns[0m[0;34m([0m[0mcols[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     97[0m         [0mself[0m[0;34m.[0m[0mchunksize[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_initialize_chunksize[0m[0;34m([0m[0mchunksize[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     98[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/formats/csvs.py[0m in [0;36m_initialize_columns[0;34m(self, cols)[0m
[1;32m    166[0m         [0;31m# and make sure cols is just a list of labels[0m[0;34m[0m[0;34m[0m[0m
[1;32m    167[0m         [0mnew_cols[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mobj[0m[0;34m.[0m[0mcolumns[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 168[0;31m         [0;32mreturn[0m [0mnew_cols[0m[0;34m.[0m[0m_format_native_types[0m[0;34m([0m[0;34m**[0m[0mself[0m[0;34m.[0m[0m_number_format[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    169[0m [0;34m[0m[0m
[1;32m    170[0m     [0;32mdef[0m [0m_initialize_chunksize[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mchunksize[0m[0;34m:[0m [0mint[0m [0;34m|[0m [0;32mNone[0m[0;34m)[0m [0;34m->[0m [0mint[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'Index' object has no attribute '_format_native_types'
