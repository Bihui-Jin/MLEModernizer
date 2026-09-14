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
import os
import sys
import types
import subprocess
import importlib.machinery

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<4"])

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

if "torchtext" not in sys.modules:
    torchtext_stub = types.ModuleType("torchtext")
    torchtext_stub.__dict__["__version__"] = "0.0.0-stub"
    torchtext_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="torchtext", loader=None
    )
    sys.modules["torchtext"] = torchtext_stub

if "bitsandbytes" not in sys.modules:
    bnb_stub = types.ModuleType("bitsandbytes")
    bnb_stub.__dict__["__version__"] = "0.0.0-stub"
    bnb_stub.__spec__ = importlib.machinery.ModuleSpec(name="bitsandbytes", loader=None)

    optim_stub = types.ModuleType("bitsandbytes.optim")
    optim_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="bitsandbytes.optim", loader=None
    )

    class GlobalOptimManager:  # minimal placeholder for import-time references
        pass

    class Adam8bit:
        pass

    class AdamW8bit:
        pass

    class SGD8bit:
        pass

    class RMSprop8bit:
        pass

    class Adagrad8bit:
        pass

    class Adam:
        pass

    class AdamW:
        pass

    class SGD:
        pass

    class RMSprop:
        pass

    class Adagrad:
        pass

    class PagedAdam:
        pass

    class PagedAdamW:
        pass

    optim_stub.GlobalOptimManager = GlobalOptimManager
    optim_stub.Adam8bit = Adam8bit
    optim_stub.AdamW8bit = AdamW8bit
    optim_stub.SGD8bit = SGD8bit
    optim_stub.RMSprop8bit = RMSprop8bit
    optim_stub.Adagrad8bit = Adagrad8bit

    optim_stub.Adam = Adam
    optim_stub.AdamW = AdamW
    optim_stub.SGD = SGD
    optim_stub.RMSprop = RMSprop
    optim_stub.Adagrad = Adagrad

    optim_stub.PagedAdam = PagedAdam
    optim_stub.PagedAdamW = PagedAdamW
    optim_stub.PagedAdam8bit = PagedAdam
    optim_stub.PagedAdamW8bit = PagedAdamW
    optim_stub.PagedAdam8Bit = PagedAdam
    optim_stub.PagedAdamW8Bit = PagedAdamW

    cext_stub = types.ModuleType("bitsandbytes.cextension")
    cext_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="bitsandbytes.cextension", loader=None
    )
    cext_stub.COMPILED_WITH_CUDA = False

    bnb_stub.optim = optim_stub
    bnb_stub.cextension = cext_stub

    sys.modules["bitsandbytes"] = bnb_stub
    sys.modules["bitsandbytes.optim"] = optim_stub
    sys.modules["bitsandbytes.cextension"] = cext_stub

import ludwig
from ludwig.api import LudwigModel
import logging


## --- ERROR in cell 16, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1873415261.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    103[0m [0;34m[0m[0m
[1;32m    104[0m [0;32mimport[0m [0mludwig[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 105[0;31m [0;32mfrom[0m [0mludwig[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mLudwigModel[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    106[0m [0;32mimport[0m [0mlogging[0m[0;34m[0m[0;34m[0m[0m

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
[1;32m    617[0m [0;34m@[0m[0mregister_optimizer[0m[0;34m([0m[0mname[0m[0;34m=[0m[0;34m"lamb"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    618[0m [0;34m@[0m[0mludwig_dataclass[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 619[0;31m [0;32mclass[0m [0mLAMBOptimizerConfig[0m[0;34m([0m[0mBaseOptimizerConfig[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    620[0m     """Layer-wise Adaptive Moments optimizer for Batch training.
[1;32m    621[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/ludwig/schema/optimizers.py[0m in [0;36mLAMBOptimizerConfig[0;34m()[0m
[1;32m    623[0m     """
[1;32m    624[0m [0;34m[0m[0m
[0;32m--> 625[0;31m     [0moptimizer_class[0m[0;34m:[0m [0mClassVar[0m[0;34m[[0m[0mtorch[0m[0;34m.[0m[0moptim[0m[0;34m.[0m[0mOptimizer[0m[0;34m][0m [0;34m=[0m [0mbnb[0m[0;34m.[0m[0moptim[0m[0;34m.[0m[0mLAMB[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    626[0m [0;34m[0m[0m
[1;32m    627[0m     [0mtype[0m[0;34m:[0m [0mstr[0m [0;34m=[0m [0mschema_utils[0m[0;34m.[0m[0mProtectedString[0m[0;34m([0m[0;34m"lamb"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: module 'bitsandbytes.optim' has no attribute 'LAMB'

## === cell 17
def configs(par, sample_key, sample_item):
    config = {'combiner': {'dropout': 0.3,
                  'num_fc_layers': 3,
                  'output_size': 256,

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

     'trainer': {'epochs': 40, 'optimizer': {'type': 'adam'}},

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
            'search_alg': {'type': 'optuna', 'random_state': 42, },
            'goal': 'maximize',
            'metric': 'roc_auc',
            'output_feature': par,
            }        
             }
    return config

config1 = configs('EC1','oversample_minority', 0.5)

model = LudwigModel(config=config1, logging_level=logging.INFO)
