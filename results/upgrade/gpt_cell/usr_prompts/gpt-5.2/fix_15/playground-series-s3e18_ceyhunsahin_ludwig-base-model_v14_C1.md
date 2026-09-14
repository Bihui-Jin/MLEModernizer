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

if "bitsandbytes" not in sys.modules:
    bnb_stub = types.ModuleType("bitsandbytes")
    bnb_stub.__dict__["__version__"] = "0.0.0-stub"
    bnb_stub.__spec__ = importlib.machinery.ModuleSpec(name="bitsandbytes", loader=None)

    bnb_cext_stub = types.ModuleType("bitsandbytes.cextension")
    bnb_cext_stub.__dict__["COMPILED_WITH_CUDA"] = False
    bnb_cext_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="bitsandbytes.cextension", loader=None
    )

    bnb_optim_stub = types.ModuleType("bitsandbytes.optim")
    bnb_optim_stub.__dict__["COMPILED_WITH_CUDA"] = False
    bnb_optim_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="bitsandbytes.optim", loader=None
    )

    class _Dummy8BitOptimizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "bitsandbytes is not available in this environment; "
                "8-bit optimizers cannot be instantiated."
            )

    bnb_optim_stub.__dict__["SGD8bit"] = _Dummy8BitOptimizer
    bnb_optim_stub.__dict__["Adam8bit"] = _Dummy8BitOptimizer
    bnb_optim_stub.__dict__["AdamW8bit"] = _Dummy8BitOptimizer
    bnb_optim_stub.__dict__["Adagrad8bit"] = _Dummy8BitOptimizer
    bnb_optim_stub.__dict__["RMSprop8bit"] = _Dummy8BitOptimizer
    bnb_optim_stub.__dict__["LAMB"] = _Dummy8BitOptimizer
    bnb_optim_stub.__dict__["LAMB8bit"] = _Dummy8BitOptimizer
    bnb_optim_stub.__dict__["Lion8bit"] = _Dummy8BitOptimizer
    bnb_optim_stub.__dict__["Lion"] = _Dummy8BitOptimizer

    bnb_optim_stub.__dict__["LARS"] = _Dummy8BitOptimizer
    bnb_optim_stub.__dict__["LARS8bit"] = _Dummy8BitOptimizer

    bnb_optim_stub.__dict__["PagedAdam"] = _Dummy8BitOptimizer
    bnb_optim_stub.__dict__["PagedAdam8bit"] = _Dummy8BitOptimizer  # required by Ludwig
    bnb_optim_stub.__dict__["PagedAdamW"] = _Dummy8BitOptimizer  # required by Ludwig
    bnb_optim_stub.__dict__["PagedAdamW8bit"] = (
        _Dummy8BitOptimizer  # required by Ludwig
    )

    bnb_optim_stub.__dict__["PagedLion"] = _Dummy8BitOptimizer  # required by Ludwig
    bnb_optim_stub.__dict__["PagedLion8bit"] = _Dummy8BitOptimizer  # required by Ludwig

    bnb_optim_manager_stub = type("GlobalOptimManager", (), {})
    bnb_optim_stub.__dict__["GlobalOptimManager"] = bnb_optim_manager_stub

    bnb_research_stub = types.ModuleType("bitsandbytes.research")
    bnb_research_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="bitsandbytes.research", loader=None
    )

    bnb_research_nn_stub = types.ModuleType("bitsandbytes.research.nn")
    bnb_research_nn_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="bitsandbytes.research.nn", loader=None
    )

    bnb_research_nn_modules_stub = types.ModuleType("bitsandbytes.research.nn.modules")
    bnb_research_nn_modules_stub.__dict__["GlobalOptimManager"] = bnb_optim_manager_stub
    bnb_research_nn_modules_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="bitsandbytes.research.nn.modules", loader=None
    )

    bnb_stub.__dict__["optim"] = bnb_optim_stub
    bnb_stub.__dict__["cextension"] = bnb_cext_stub
    bnb_stub.__dict__["research"] = bnb_research_stub

    sys.modules["bitsandbytes"] = bnb_stub
    sys.modules["bitsandbytes.cextension"] = bnb_cext_stub
    sys.modules["bitsandbytes.optim"] = bnb_optim_stub
    sys.modules["bitsandbytes.research"] = bnb_research_stub
    sys.modules["bitsandbytes.research.nn"] = bnb_research_nn_stub
    sys.modules["bitsandbytes.research.nn.modules"] = bnb_research_nn_modules_stub

if "torchtext" not in sys.modules:
    torchtext_stub = types.ModuleType("torchtext")
    torchtext_stub.__dict__["__version__"] = "0.0.0-stub"
    torchtext_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="torchtext", loader=None
    )

    torchtext_extension_stub = types.ModuleType("torchtext._extension")
    torchtext_extension_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="torchtext._extension", loader=None
    )

    sys.modules["torchtext"] = torchtext_stub
    sys.modules["torchtext._extension"] = torchtext_extension_stub

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import ludwig
from ludwig.api import LudwigModel
import logging


## --- ERROR in cell 16, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1319455.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    112[0m [0;34m[0m[0m
[1;32m    113[0m [0;32mimport[0m [0mludwig[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 114[0;31m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mLudwigModel[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    115[0m [0;32mimport[0m [0mlogging[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/ludwig/api.py[0m in [0;36m<module>[0;34m[0m
[1;32m     39[0m [0;34m[0m[0m
[1;32m     40[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mapi_annotations[0m [0;32mimport[0m [0mPublicAPI[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 41[0;31m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mbackend[0m [0;32mimport[0m [0mBackend[0m[0;34m,[0m [0minitialize_backend[0m[0;34m,[0m [0mprovision_preprocessing_workers[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     42[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mcallbacks[0m [0;32mimport[0m [0mCallback[0m[0;34m[0m[0;34m[0m[0m
[1;32m     43[0m from ludwig.constants import (

[0;32m/usr/local/lib/python3.11/dist-packages/ludwig/backend/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     20[0m [0;34m[0m[0m
[1;32m     21[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mapi_annotations[0m [0;32mimport[0m [0mDeveloperAPI[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 22[0;31m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mbase[0m [0;32mimport[0m [0mBackend[0m[0;34m,[0m [0mLocalBackend[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     23[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mhorovod_utils[0m [0;32mimport[0m [0mhas_horovodrun[0m[0;34m[0m[0;34m[0m[0m
[1;32m     24[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/ludwig/backend/base.py[0m in [0;36m<module>[0;34m[0m
[1;32m     32[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mstorage[0m [0;32mimport[0m [0mStorageManager[0m[0;34m[0m[0;34m[0m[0m
[1;32m     33[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mconstants[0m [0;32mimport[0m [0mMODEL_LLM[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 34[0;31m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mdata[0m[0;34m.[0m[0mcache[0m[0;34m.[0m[0mmanager[0m [0;32mimport[0m [0mCacheManager[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     35[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mdata[0m[0;34m.[0m[0mdataframe[0m[0;34m.[0m[0mbase[0m [0;32mimport[0m [0mDataFrameEngine[0m[0;34m[0m[0;34m[0m[0m
[1;32m     36[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mdata[0m[0;34m.[0m[0mdataframe[0m[0;34m.[0m[0mpandas[0m [0;32mimport[0m [0mPANDAS[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/ludwig/data/cache/manager.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mdata[0m[0;34m.[0m[0mcache[0m[0;34m.[0m[0mtypes[0m [0;32mimport[0m [0malphanum[0m[0;34m,[0m [0mCacheableDataset[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mdata[0m[0;34m.[0m[0mcache[0m[0;34m.[0m[0mutil[0m [0;32mimport[0m [0mcalculate_checksum[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 8[0;31m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mdata[0m[0;34m.[0m[0mdataset[0m[0;34m.[0m[0mbase[0m [0;32mimport[0m [0mDatasetManager[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mutils[0m [0;32mimport[0m [0mdata_utils[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mfs_utils[0m [0;32mimport[0m [0mdelete[0m[0;34m,[0m [0mpath_exists[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/ludwig/data/dataset/base.py[0m in [0;36m<module>[0;34m[0m
[1;32m     23[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mdata[0m[0;34m.[0m[0mbatcher[0m[0;34m.[0m[0mbase[0m [0;32mimport[0m [0mBatcher[0m[0;34m[0m[0;34m[0m[0m
[1;32m     24[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mdistributed[0m [0;32mimport[0m [0mDistributedStrategy[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 25[0;31m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mfeatures[0m[0;34m.[0m[0mbase_feature[0m [0;32mimport[0m [0mBaseFeature[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     26[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mdefaults[0m [0;32mimport[0m [0mdefault_random_seed[0m[0;34m[0m[0;34m[0m[0m
[1;32m     27[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mtypes[0m [0;32mimport[0m [0mDataFrame[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/ludwig/features/base_feature.py[0m in [0;36m<module>[0;34m[0m
[1;32m     32[0m )
[1;32m     33[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mdecoders[0m[0;34m.[0m[0mregistry[0m [0;32mimport[0m [0mget_decoder_cls[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 34[0;31m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mencoders[0m[0;34m.[0m[0mregistry[0m [0;32mimport[0m [0mget_encoder_cls[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     35[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mfeatures[0m[0;34m.[0m[0mfeature_utils[0m [0;32mimport[0m [0mget_input_size_with_dependencies[0m[0;34m[0m[0;34m[0m[0m
[1;32m     36[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mmodules[0m[0;34m.[0m[0mfully_connected_modules[0m [0;32mimport[0m [0mFCStack[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/ludwig/encoders/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      8[0m [0;32mimport[0m [0mludwig[0m[0;34m.[0m[0mencoders[0m[0;34m.[0m[0msequence_encoders[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m [0;32mimport[0m [0mludwig[0m[0;34m.[0m[0mencoders[0m[0;34m.[0m[0mset_encoders[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 10[0;31m [0;32mimport[0m [0mludwig[0m[0;34m.[0m[0mencoders[0m[0;34m.[0m[0mtext_encoders[0m  [0;31m# noqa[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/ludwig/encoders/text_encoders.py[0m in [0;36m<module>[0;34m[0m
[1;32m     58[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mdata_utils[0m [0;32mimport[0m [0mclear_data_cache[0m[0;34m[0m[0;34m[0m[0m
[1;32m     59[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mhf_utils[0m [0;32mimport[0m [0mload_pretrained_hf_model_with_hub_fallback[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 60[0;31m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mllm_utils[0m [0;32mimport[0m [0mget_context_len[0m[0;34m,[0m [0minitialize_adapter[0m[0;34m,[0m [0mload_pretrained_from_config[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     61[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mtokenizers[0m [0;32mimport[0m [0mHFTokenizer[0m[0;34m[0m[0;34m[0m[0m
[1;32m     62[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mtorch_utils[0m [0;32mimport[0m [0mFreezeModule[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/ludwig/utils/llm_utils.py[0m in [0;36m<module>[0;34m[0m
[1;32m      7[0m [0;32mimport[0m [0mtorch[0m[0;34m.[0m[0mnn[0m[0;34m.[0m[0mfunctional[0m [0;32mas[0m [0mF[0m[0;34m[0m[0;34m[0m[0m
[1;32m      8[0m [0;32mimport[0m [0mtransformers[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 9[0;31m [0;32mfrom[0m [0mbitsandbytes[0m[0;34m.[0m[0mnn[0m[0;34m.[0m[0mmodules[0m [0;32mimport[0m [0mEmbedding[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     10[0m [0;32mfrom[0m [0mpackaging[0m [0;32mimport[0m [0mversion[0m[0;34m[0m[0;34m[0m[0m
[1;32m     11[0m [0;32mfrom[0m [0mtransformers[0m [0;32mimport[0m [0mAutoConfig[0m[0;34m,[0m [0mAutoModelForCausalLM[0m[0;34m,[0m [0mPreTrainedModel[0m[0;34m,[0m [0mPreTrainedTokenizer[0m[0;34m,[0m [0mTextStreamer[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'bitsandbytes.nn'; 'bitsandbytes' is not a package

## === cell 17
def configs(par, sample_key, sample_item):
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
                          'encoder': {
                                    'activations' : 'relu',
                                    'weights_initializer': 'xavier_uniform',},
                          'decoder': {
                                'num_fc_layers': 4,
                                'fc_activation': 'relu',
                                'output_size': 16
                          },
                          'preprocessing': {'fallback_true_label': '1'},
                          'loss': {'type': 'binary_weighted_cross_entropy'},
                          'type': 'binary'}],
     'defaults': {
        'number': {
          'preprocessing': {
            'missing_value_strategy': 'fill_with_mean',
            'normalization': 'zscore',
            sample_key : sample_item
          }
        }
     },

     'trainer': {'epochs': 150, 'optimizer': {'type': 'adam'}},

     'hyperopt':  {
            'executor': {'num_samples': 8},
            'parameters': {
                        'Machine failure.decoder.num_fc_layers': {
                            'space': 'randint',
                            'lower': 2,
                            'upper': 10
                        },
                        'trainer.learning_rate': {
                            'space': 'loguniform',
                            'lower': 0.0001,
                            'upper': 0.1}
                        },
            'search_alg': {'type': 'hyperopt', 'random_state': 42, },
            'goal': 'maximize',
            'metric': 'roc_auc',
            'output_feature': par,
            }        
             }
    return config

config1 = configs('EC1','oversample_minority', 0.5)

model = LudwigModel(config=config1, logging_level=logging.INFO)
