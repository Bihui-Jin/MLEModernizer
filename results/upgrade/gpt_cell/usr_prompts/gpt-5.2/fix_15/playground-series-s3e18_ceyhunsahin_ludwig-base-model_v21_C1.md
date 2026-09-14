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
def create_features(df):
    
    new_features = {
        'BertzCT_MaxAbsEStateIndex_Ratio': df['BertzCT'] / (df['MaxAbsEStateIndex'] + 1e-12),
        'BertzCT_ExactMolWt_Product': df['BertzCT'] * df['ExactMolWt'],
        'NumHeteroatoms_FpDensityMorgan1_Ratio': df['NumHeteroatoms'] / (df['FpDensityMorgan1'] + 1e-12),
        'VSA_EState9_EState_VSA1_Ratio': df['VSA_EState9'] / (df['EState_VSA1'] + 1e-12),
        'PEOE_VSA10_SMR_VSA5_Ratio': df['PEOE_VSA10'] / (df['SMR_VSA5'] + 1e-12),
        'Chi1v_ExactMolWt_Product': df['Chi1v'] * df['ExactMolWt'],
        'Chi2v_ExactMolWt_Product': df['Chi2v'] * df['ExactMolWt'],
        'Chi3v_ExactMolWt_Product': df['Chi3v'] * df['ExactMolWt'],
        'EState_VSA1_NumHeteroatoms_Product': df['EState_VSA1'] * df['NumHeteroatoms'],
        'PEOE_VSA10_Chi1_Ratio': df['PEOE_VSA10'] / (df['Chi1'] + 1e-12),
        'MaxAbsEStateIndex_NumHeteroatoms_Ratio': df['MaxAbsEStateIndex'] / (df['NumHeteroatoms'] + 1e-12),
        'BertzCT_Chi1_Ratio': df['BertzCT'] / (df['Chi1'] + 1e-12),
    }
    
    df = df.assign(**new_features)
    new_cols = list(new_features.keys())
    
    return df


## === cell 17
df_train = create_features(df_train)


## === cell 18
df_test = create_features(df_test)


## === cell 19
import os
import sys
import types

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

try:
    import torchtext  # noqa: F401

    if not hasattr(torchtext, "__version__"):
        torchtext.__version__ = "0.0.0"
except Exception:
    torchtext_stub = types.ModuleType("torchtext")
    torchtext_stub.__version__ = "0.0.0"
    torchtext_extension_stub = types.ModuleType("torchtext._extension")
    torchtext_stub._extension = torchtext_extension_stub
    sys.modules["torchtext"] = torchtext_stub
    sys.modules["torchtext._extension"] = torchtext_extension_stub

if "bitsandbytes" not in sys.modules:
    import importlib.machinery

    bnb_stub = types.ModuleType("bitsandbytes")
    bnb_stub.__version__ = "0.0.0"
    bnb_stub.__spec__ = importlib.machinery.ModuleSpec(name="bitsandbytes", loader=None)

    bnb_optim_stub = types.ModuleType("bitsandbytes.optim")
    bnb_optim_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="bitsandbytes.optim", loader=None
    )

    class GlobalOptimManager:  # minimal placeholder
        pass

    bnb_optim_stub.GlobalOptimManager = GlobalOptimManager

    try:
        import torch

        class _BNBOptimizer(torch.optim.Optimizer):
            def __init__(self, params, defaults=None, *args, **kwargs):
                if defaults is None:
                    defaults = {}
                super().__init__(params, defaults)

            def step(self, closure=None):
                if closure is not None:
                    return closure()
                return None

        for _name in [
            "SGD8bit",
            "Adam8bit",
            "AdamW8bit",
            "RMSprop8bit",
            "Adagrad8bit",
            "Adadelta8bit",
            "Adam",
            "AdamW",
            "SGD",
            "RMSprop",
            "Adagrad",
            "Adadelta",
            "PagedAdam",
            "LAMB",
        ]:
            setattr(bnb_optim_stub, _name, type(_name, (_BNBOptimizer,), {}))
    except Exception:
        for _name in [
            "SGD8bit",
            "Adam8bit",
            "AdamW8bit",
            "RMSprop8bit",
            "Adagrad8bit",
            "Adadelta8bit",
            "Adam",
            "AdamW",
            "SGD",
            "RMSprop",
            "Adagrad",
            "Adadelta",
            "PagedAdam",
            "LAMB",
        ]:
            setattr(bnb_optim_stub, _name, object)

    if not hasattr(bnb_optim_stub, "LARS"):
        bnb_optim_stub.LARS = getattr(bnb_optim_stub, "SGD", object)
    if not hasattr(bnb_optim_stub, "LARS8bit"):
        bnb_optim_stub.LARS8bit = getattr(bnb_optim_stub, "LARS")
    if not hasattr(bnb_optim_stub, "LARS8Bit"):
        bnb_optim_stub.LARS8Bit = getattr(bnb_optim_stub, "LARS8bit")

    if not hasattr(bnb_optim_stub, "LAMB8bit"):
        bnb_optim_stub.LAMB8bit = getattr(bnb_optim_stub, "LAMB")
    if not hasattr(bnb_optim_stub, "LAMB8Bit"):
        bnb_optim_stub.LAMB8Bit = getattr(bnb_optim_stub, "LAMB8bit")

    if not hasattr(bnb_optim_stub, "PagedAdam8bit"):
        bnb_optim_stub.PagedAdam8bit = getattr(bnb_optim_stub, "PagedAdam")
    if not hasattr(bnb_optim_stub, "PagedAdamW"):
        bnb_optim_stub.PagedAdamW = getattr(
            bnb_optim_stub, "AdamW", getattr(bnb_optim_stub, "PagedAdam")
        )

    if not hasattr(bnb_optim_stub, "PagedAdamW8bit"):
        bnb_optim_stub.PagedAdamW8bit = getattr(
            bnb_optim_stub, "PagedAdamW", getattr(bnb_optim_stub, "AdamW", object)
        )
    if not hasattr(bnb_optim_stub, "PagedAdamW8Bit"):
        bnb_optim_stub.PagedAdamW8Bit = bnb_optim_stub.PagedAdamW8bit

    bnb_cextension_stub = types.ModuleType("bitsandbytes.cextension")
    bnb_cextension_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="bitsandbytes.cextension", loader=None
    )
    bnb_cextension_stub.COMPILED_WITH_CUDA = False

    bnb_stub.optim = bnb_optim_stub

    sys.modules["bitsandbytes"] = bnb_stub
    sys.modules["bitsandbytes.optim"] = bnb_optim_stub
    sys.modules["bitsandbytes.cextension"] = bnb_cextension_stub
else:
    import importlib.machinery

    if getattr(sys.modules["bitsandbytes"], "__spec__", None) is None:
        sys.modules["bitsandbytes"].__spec__ = importlib.machinery.ModuleSpec(
            name="bitsandbytes", loader=None
        )

import ludwig
from ludwig.api import LudwigModel
import logging


## --- ERROR in cell 19, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3613483579.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    138[0m [0;34m[0m[0m
[1;32m    139[0m [0;32mimport[0m [0mludwig[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 140[0;31m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mLudwigModel[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    141[0m [0;32mimport[0m [0mlogging[0m[0;34m[0m[0;34m[0m[0m

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
[1;32m     31[0m     [0mPROBABILITIES[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     32[0m )
[0;32m---> 33[0;31m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mdecoders[0m[0;34m.[0m[0mregistry[0m [0;32mimport[0m [0mget_decoder_cls[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     34[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mencoders[0m[0;34m.[0m[0mregistry[0m [0;32mimport[0m [0mget_encoder_cls[0m[0;34m[0m[0;34m[0m[0m
[1;32m     35[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mfeatures[0m[0;34m.[0m[0mfeature_utils[0m [0;32mimport[0m [0mget_input_size_with_dependencies[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/ludwig/decoders/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0;31m# register all decoders[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0;32mimport[0m [0mludwig[0m[0;34m.[0m[0mdecoders[0m[0;34m.[0m[0mgeneric_decoders[0m  [0;31m# noqa[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0;32mimport[0m [0mludwig[0m[0;34m.[0m[0mdecoders[0m[0;34m.[0m[0mimage_decoders[0m  [0;31m# noqa[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mimport[0m [0mludwig[0m[0;34m.[0m[0mdecoders[0m[0;34m.[0m[0mllm_decoders[0m  [0;31m# noqa[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mimport[0m [0mludwig[0m[0;34m.[0m[0mdecoders[0m[0;34m.[0m[0msequence_decoders[0m  [0;31m# noqa[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/ludwig/decoders/generic_decoders.py[0m in [0;36m<module>[0;34m[0m
[1;32m     23[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mdecoders[0m[0;34m.[0m[0mbase[0m [0;32mimport[0m [0mDecoder[0m[0;34m[0m[0;34m[0m[0m
[1;32m     24[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mdecoders[0m[0;34m.[0m[0mregistry[0m [0;32mimport[0m [0mregister_decoder[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 25[0;31m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mschema[0m[0;34m.[0m[0mdecoders[0m[0;34m.[0m[0mbase[0m [0;32mimport[0m [0mClassifierConfig[0m[0;34m,[0m [0mPassthroughDecoderConfig[0m[0;34m,[0m [0mProjectorConfig[0m[0;34m,[0m [0mRegressorConfig[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     26[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mtorch_utils[0m [0;32mimport[0m [0mDense[0m[0;34m,[0m [0mget_activation[0m[0;34m[0m[0;34m[0m[0m
[1;32m     27[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/ludwig/schema/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      3[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mschema[0m[0;34m.[0m[0mfeatures[0m[0;34m.[0m[0mutils[0m [0;32mimport[0m [0mget_input_feature_jsonschema[0m[0;34m,[0m [0mget_output_feature_jsonschema[0m  [0;31m# noqa[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mschema[0m[0;34m.[0m[0mhyperopt[0m [0;32mimport[0m [0mget_hyperopt_jsonschema[0m  [0;31m# noqa[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mschema[0m[0;34m.[0m[0mtrainer[0m [0;32mimport[0m [0mget_model_type_jsonschema[0m[0;34m,[0m [0mget_trainer_jsonschema[0m  [0;31m# noqa[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/ludwig/schema/trainer.py[0m in [0;36m<module>[0;34m[0m
[1;32m     22[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mschema[0m[0;34m.[0m[0mlr_scheduler[0m [0;32mimport[0m [0mLRSchedulerConfig[0m[0;34m,[0m [0mLRSchedulerDataclassField[0m[0;34m[0m[0;34m[0m[0m
[1;32m     23[0m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mschema[0m[0;34m.[0m[0mmetadata[0m [0;32mimport[0m [0mTRAINER_METADATA[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 24[0;31m from ludwig.schema.optimizers import (
[0m[1;32m     25[0m     [0mBaseOptimizerConfig[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     26[0m     [0mGradientClippingConfig[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/ludwig/schema/optimizers.py[0m in [0;36m<module>[0;34m[0m
[1;32m    768[0m [0;34m@[0m[0mregister_optimizer[0m[0;34m([0m[0mname[0m[0;34m=[0m[0;34m"lion"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    769[0m [0;34m@[0m[0mludwig_dataclass[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 770[0;31m [0;32mclass[0m [0mLIONOptimizerConfig[0m[0;34m([0m[0mBaseOptimizerConfig[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    771[0m     """Evolved Sign Momentum.
[1;32m    772[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/ludwig/schema/optimizers.py[0m in [0;36mLIONOptimizerConfig[0;34m()[0m
[1;32m    774[0m     """
[1;32m    775[0m [0;34m[0m[0m
[0;32m--> 776[0;31m     [0moptimizer_class[0m[0;34m:[0m [0mClassVar[0m[0;34m[[0m[0mtorch[0m[0;34m.[0m[0moptim[0m[0;34m.[0m[0mOptimizer[0m[0;34m][0m [0;34m=[0m [0mbnb[0m[0;34m.[0m[0moptim[0m[0;34m.[0m[0mLion[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    777[0m [0;34m[0m[0m
[1;32m    778[0m     [0mtype[0m[0;34m:[0m [0mstr[0m [0;34m=[0m [0mschema_utils[0m[0;34m.[0m[0mProtectedString[0m[0;34m([0m[0;34m"lion"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: module 'bitsandbytes.optim' has no attribute 'Lion'

## === cell 20
SEED=13

def hyperopt_par(par):

    hyperopt_configs = {
        "parameters": {
            "trainer.learning_rate": {
                "type": "float",
                "space": "loguniform",
                "lower": 0.0001,
                "upper": 0.01,
                "q": 3,
            },
            "trainer.batch_size": {
                "type": "int",
                "space": "qlograndint",
                "base" : 2,
                "lower": 32,
                "upper": 256,
                "q": 5,
            },
            f"{par}.fc_size": {
                "type": "int",
                'space': 'qrandint',
                "lower": 32,
                "upper": 256,
                "q": 5,
            },
            f"{par}.num_fc_layers": {
                'type': 'int',
                'space': 'qrandint',
                'lower': 1,
                'upper': 5,
                'q': 4,
            }
        },
        'executor': {'num_samples': 16},
        'goal': 'maximize',
        'metric': 'roc_auc',
        'output_feature': par,
        'validation_metrics': 'loss'
    }
    return hyperopt_configs
