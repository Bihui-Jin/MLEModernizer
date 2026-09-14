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
numerical_columns = num_cols

fig, axes = plt.subplots(len(numerical_columns), 2, figsize=(20, 40))

for i, column in enumerate(numerical_columns):
    sns.histplot(df_train[column], bins=30, kde=True, ax=axes[i, 0], color = my_palette[2])
    axes[i, 0].set_title(f'Distribution of {column} in df_train')
    axes[i, 0].set_xlabel('Value')
    axes[i, 0].set_ylabel('Frequency')

    sns.boxplot(df_train[column], ax=axes[i, 1], color = my_palette[1])
    axes[i, 1].set_title(f'Box plot of {column} in df_train')
    axes[i, 1].set_xlabel(column)
    axes[i, 1].set_ylabel('Value')

plt.tight_layout()
plt.show()


## === cell 13
import os
import sys
import types
import importlib.machinery

import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

if "torchtext" not in sys.modules:
    torchtext_stub = types.ModuleType("torchtext")
    torchtext_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="torchtext", loader=None
    )
    torchtext_stub.__version__ = "0.0.0"
    sys.modules["torchtext"] = torchtext_stub

if "bitsandbytes" not in sys.modules:
    bnb_stub = types.ModuleType("bitsandbytes")
    bnb_stub.__spec__ = importlib.machinery.ModuleSpec(name="bitsandbytes", loader=None)
    bnb_stub.__version__ = "0.0.0"
    bnb_stub.__path__ = []

    optim_stub = types.ModuleType("bitsandbytes.optim")
    optim_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="bitsandbytes.optim", loader=None
    )

    class _Dummy8BitOptimizer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "bitsandbytes is not installed in this environment; "
                "8-bit / paged optimizers are unavailable."
            )

    optim_stub.SGD8bit = _Dummy8BitOptimizer
    optim_stub.Adam8bit = _Dummy8BitOptimizer
    optim_stub.AdamW8bit = _Dummy8BitOptimizer
    optim_stub.RMSprop8bit = _Dummy8BitOptimizer
    optim_stub.Lion8bit = _Dummy8BitOptimizer

    optim_stub.Lion = _Dummy8BitOptimizer
    optim_stub.LION = _Dummy8BitOptimizer  # defensive alias for potential case variants

    optim_stub.PagedAdam = _Dummy8BitOptimizer
    optim_stub.PagedAdamW = _Dummy8BitOptimizer
    optim_stub.PagedLion = _Dummy8BitOptimizer
    optim_stub.PagedSGD = _Dummy8BitOptimizer
    optim_stub.PagedRMSprop = _Dummy8BitOptimizer

    optim_stub.PagedAdam8bit = _Dummy8BitOptimizer
    optim_stub.PagedAdamW8bit = _Dummy8BitOptimizer
    optim_stub.PagedLion8bit = _Dummy8BitOptimizer
    optim_stub.PagedSGD8bit = _Dummy8BitOptimizer
    optim_stub.PagedRMSprop8bit = _Dummy8BitOptimizer

    optim_stub.Adagrad8bit = _Dummy8BitOptimizer
    optim_stub.Adagrad = _Dummy8BitOptimizer
    optim_stub.Adagrad8bit = _Dummy8BitOptimizer
    optim_stub.AdaGrad8bit = _Dummy8BitOptimizer  # defensive alias
    optim_stub.Adagrad32bit = _Dummy8BitOptimizer  # defensive
    optim_stub.AdaFactor = _Dummy8BitOptimizer  # defensive (some builds expose it)
    optim_stub.AdaFactor8bit = _Dummy8BitOptimizer  # defensive
    optim_stub.LAMB8bit = _Dummy8BitOptimizer  # defensive (some builds expose it)

    optim_stub.LAMB = _Dummy8BitOptimizer
    optim_stub.Lamb = (
        _Dummy8BitOptimizer  # defensive alias (case variants sometimes used)
    )
    optim_stub.LAMB32bit = _Dummy8BitOptimizer  # defensive

    optim_stub.LARS = _Dummy8BitOptimizer
    optim_stub.Lars = _Dummy8BitOptimizer  # defensive alias
    optim_stub.LARS8bit = (
        _Dummy8BitOptimizer  # defensive (variant sometimes referenced)
    )

    nn_stub = types.ModuleType("bitsandbytes.nn")
    nn_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="bitsandbytes.nn", loader=None
    )

    nn_modules_stub = types.ModuleType("bitsandbytes.nn.modules")
    nn_modules_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="bitsandbytes.nn.modules", loader=None
    )

    class _DummyEmbedding:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "bitsandbytes is not installed in this environment; "
                "bitsandbytes.nn.modules.Embedding is unavailable."
            )

    class _DummyLinear4bit:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "bitsandbytes is not installed in this environment; "
                "bitsandbytes.nn.modules.Linear4bit is unavailable."
            )

    class _DummyLinear8bitLt:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "bitsandbytes is not installed in this environment; "
                "bitsandbytes.nn.modules.Linear8bitLt is unavailable."
            )

    nn_modules_stub.Embedding = _DummyEmbedding
    nn_modules_stub.Linear4bit = _DummyLinear4bit
    nn_modules_stub.Linear8bitLt = _DummyLinear8bitLt

    nn_stub.modules = nn_modules_stub
    bnb_stub.optim = optim_stub
    bnb_stub.nn = nn_stub

    functional_stub = types.ModuleType("bitsandbytes.functional")
    functional_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="bitsandbytes.functional", loader=None
    )

    def _missing_bitsandbytes_functional(*args, **kwargs):
        raise RuntimeError(
            "bitsandbytes is not installed in this environment; "
            "bitsandbytes.functional functions are unavailable."
        )

    functional_stub.dequantize_4bit = _missing_bitsandbytes_functional

    sys.modules["bitsandbytes"] = bnb_stub
    sys.modules["bitsandbytes.optim"] = optim_stub
    sys.modules["bitsandbytes.nn"] = nn_stub
    sys.modules["bitsandbytes.nn.modules"] = nn_modules_stub
    sys.modules["bitsandbytes.functional"] = functional_stub

import ludwig
from ludwig.api import LudwigModel
import logging


## === cell 14
def configs(par):
    config = {'combiner': {'dropout': 0.2,
                  'num_fc_layers': 3,
                  'output_size': 128,
                  'type': 'concat'},

     'input_features': [{'name': 'BertzCT', 'type': 'number'},
                        {'name': 'Chi1', 'type': 'number'},
                        {'name': 'Chi1n', 'type': 'number'},
                        {'name': 'Chi1v', 'type': 'number'},
                        {'name': 'Chi2n', 'type': 'number'},
                        {'name': 'Chi2v', 'type': 'number'},
                        {'name': 'Chi3v', 'type': 'number'},
                        {'name': 'Chi4n', 'type': 'number'},
                        {'name': 'EState_VSA1', 'type': 'number'},
                        {'name': 'EState_VSA2', 'type': 'number'},
                        {'name': 'ExactMolWt', 'type': 'number'},
                        {'name': 'FpDensityMorgan1', 'type': 'number'},
                        {'name': 'FpDensityMorgan2', 'type': 'number'},
                        {'name': 'FpDensityMorgan3', 'type': 'number'},
                        {'name': 'HallKierAlpha', 'type': 'number'},
                        {'name': 'HeavyAtomMolWt', 'type': 'number'},
                        {'name': 'Kappa3', 'type': 'number'},
                        {'name': 'MaxAbsEStateIndex', 'type': 'number'},
                        {'name': 'MinEStateIndex', 'type': 'number'},
                        {'name': 'NumHeteroatoms', 'type': 'number'},
                        {'name': 'PEOE_VSA10', 'type': 'number'},
                        {'name': 'PEOE_VSA14', 'type': 'number'},
                        {'name': 'PEOE_VSA6', 'type': 'number'},
                        {'name': 'PEOE_VSA7', 'type': 'number'},
                        {'name': 'PEOE_VSA8', 'type': 'number'},
                        {'name': 'SMR_VSA10', 'type': 'number'},
                        {'name': 'SMR_VSA5', 'type': 'number'},
                        {'name': 'SlogP_VSA3', 'type': 'number'},
                        {'name': 'VSA_EState9', 'type': 'number'},
                        {'name': 'fr_COO', 'type': 'category'},
                        {'name': 'fr_COO2', 'type': 'category'},



                       ],
     'output_features': [{'name': par,
                          'decoder': {
                                'num_fc_layers': 4,
                                'output_size': 1
                          },
                          'preprocessing': {'fallback_true_label': '1'},
                          'loss': {'type': 'binary_weighted_cross_entropy'},
                          'type': 'binary'}],
     'defaults': {
        'number': {
          'preprocessing': {
            'missing_value_strategy': 'fill_with_mean',
            'normalization': 'zscore',
            'oversample_minority': 0.5
          }
        }
     },

     'trainer': {'epochs': 10, 'optimizer': {'type': 'adam'}},

     'hyperopt':  {
            'executor': {'num_samples': 16,},
            'search_alg': {'type': 'hyperopt'},
            'parameters': {
                        'Machine failure.decoder.num_fc_layers': {
                            'space': 'randint',
                            'lower': 2,
                            'upper': 9
                        },
                        'trainer.learning_rate': {
                            'space': 'loguniform',
                            'lower': 0.0001,
                            'upper': 0.1}
                        },
            'search_alg': {'type': 'optuna', 'random_state': 1919, },
            'goal': 'maximize',
            'metric': 'roc_auc',
            'output_feature': par,
            }        
             }
    return config

config1 = configs('EC1')

model = LudwigModel(config=config1, logging_level=logging.INFO)


## --- ERROR in cell 14, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mConfigValidationError[0m                     Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/561955769.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     91[0m [0;34m[0m[0m
[1;32m     92[0m [0;31m# instantiate Ludwig model object[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 93[0;31m [0mmodel[0m [0;34m=[0m [0mLudwigModel[0m[0;34m([0m[0mconfig[0m[0;34m=[0m[0mconfig1[0m[0;34m,[0m [0mlogging_level[0m[0;34m=[0m[0mlogging[0m[0;34m.[0m[0mINFO[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/ludwig/api.py[0m in [0;36m__init__[0;34m(self, config, logging_level, backend, gpus, gpu_memory_limit, allow_parallel_threads, callbacks)[0m
[1;32m    319[0m [0;34m[0m[0m
[1;32m    320[0m         [0;31m# Initialize the config object[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 321[0;31m         [0mself[0m[0;34m.[0m[0mconfig_obj[0m [0;34m=[0m [0mModelConfig[0m[0;34m.[0m[0mfrom_dict[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_user_config[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    322[0m [0;34m[0m[0m
[1;32m    323[0m         [0;31m# setup logging[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/ludwig/schema/model_types/base.py[0m in [0;36mfrom_dict[0;34m(config)[0m
[1;32m    139[0m         [0mschema[0m [0;34m=[0m [0mcls[0m[0;34m.[0m[0mget_class_schema[0m[0;34m([0m[0;34m)[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    140[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 141[0;31m             [0mconfig_obj[0m[0;34m:[0m [0mModelConfig[0m [0;34m=[0m [0mschema[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mconfig[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    142[0m         [0;32mexcept[0m [0mValidationError[0m [0;32mas[0m [0me[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    143[0m             [0;32mraise[0m [0mConfigValidationError[0m[0;34m([0m[0;34mf"Config validation error raised during config deserialization: {e}"[0m[0;34m)[0m [0;32mfrom[0m [0me[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/marshmallow_dataclass/__init__.py[0m in [0;36mload[0;34m(self, data, many, **kwargs)[0m
[1;32m    728[0m                 [0;32mreturn[0m [0;34m[[0m[0mclazz[0m[0;34m([0m[0;34m**[0m[0mloaded[0m[0;34m)[0m [0;32mfor[0m [0mloaded[0m [0;32min[0m [0mall_loaded[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m    729[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 730[0;31m                 [0;32mreturn[0m [0mclazz[0m[0;34m([0m[0;34m**[0m[0mall_loaded[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    731[0m [0;34m[0m[0m
[1;32m    732[0m     [0;32mreturn[0m [0mBaseSchema[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/ludwig/schema/model_types/ecd.py[0m in [0;36m__init__[0;34m(self, input_features, output_features, model_type, trainer, preprocessing, defaults, hyperopt, backend, ludwig_version, combiner)[0m

[0;32m/usr/local/lib/python3.11/dist-packages/ludwig/schema/model_types/base.py[0m in [0;36m__post_init__[0;34m(self)[0m
[1;32m     81[0m [0;34m[0m[0m
[1;32m     82[0m         [0;31m# Auxiliary checks.[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 83[0;31m         [0mget_config_check_registry[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mcheck_config[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     84[0m [0;34m[0m[0m
[1;32m     85[0m     [0;34m@[0m[0mstaticmethod[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/ludwig/config_validation/checks.py[0m in [0;36mcheck_config[0;34m(self, config)[0m
[1;32m     47[0m     [0;32mdef[0m [0mcheck_config[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mconfig[0m[0;34m:[0m [0;34m"ModelConfig"[0m[0;34m)[0m [0;34m->[0m [0;32mNone[0m[0;34m:[0m  [0;31m# noqa: F821[0m[0;34m[0m[0;34m[0m[0m
[1;32m     48[0m         [0;32mfor[0m [0mcheck_fn[0m [0;32min[0m [0mself[0m[0;34m.[0m[0m_registry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 49[0;31m             [0mcheck_fn[0m[0;34m([0m[0mconfig[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     50[0m [0;34m[0m[0m
[1;32m     51[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/ludwig/config_validation/checks.py[0m in [0;36mcheck_hyperopt_parameter_dicts[0;34m(config)[0m
[1;32m    380[0m [0;34m[0m[0m
[1;32m    381[0m             [0;32mif[0m [0;32mnot[0m [0mpassed[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 382[0;31m                 raise ConfigValidationError(
[0m[1;32m    383[0m                     [0;34mf"The supplied hyperopt parameter {parameter} is not a valid config field. Check the Ludwig "[0m[0;34m[0m[0;34m[0m[0m
[1;32m    384[0m                     [0;34m"docs for the list of valid parameters."[0m[0;34m[0m[0;34m[0m[0m

[0;31mConfigValidationError[0m: The supplied hyperopt parameter Machine failure.decoder.num_fc_layers is not a valid config field. Check the Ludwig docs for the list of valid parameters.

## === cell 15
train_stats1, preprocessed_data1, output_directory1 = model.train(training_set=df_train,
                                                               test_set=df_test)
