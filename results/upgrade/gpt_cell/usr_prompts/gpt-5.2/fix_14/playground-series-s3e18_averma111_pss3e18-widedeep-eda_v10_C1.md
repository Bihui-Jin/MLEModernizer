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
plotly==5.24.1
plotly-express==0.4.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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

## === cell 1
%%capture
!pip install pytorch-widedeep



## === cell 2
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<4"])

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.training import Trainer
from pytorch_widedeep.models import TabMlp, WideDeep

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic("matplotlib", "inline")
plt.style.use("fivethirtyeight")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass



## === cell 3
class Datapreparation(object):
    
    def __init__(self,root_path):
        self.root_path = root_path
        
    def get_dataframe(self,filename):
        return pd.read_csv(os.path.join(self.root_path,filename))
    
    def summary(self,text, df):
        summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
        summary['null'] = df.isnull().sum()
        summary['unique'] = df.nunique()
        summary['min'] = df.min(numeric_only=True)
        summary['median'] = df.median(numeric_only=True)
        summary['max'] = df.max(numeric_only=True)
        summary['mean'] = df.mean(numeric_only=True)
        summary['std'] = df.std(numeric_only=True)
        summary['duplicate'] = df.duplicated().sum()
        return summary
    
data = Datapreparation('/kaggle/input/playground-series-s3e18')
train = data.get_dataframe('train.csv')



## === cell 4
data.summary('train', train)



## === cell 5
train.head()



## === cell 6
train.drop(["id",'EC3','EC4','EC5','EC6'], axis=1, inplace=True)



## === cell 7
cat_embed_cols = ['fr_COO','fr_COO2','NumHeteroatoms']
continuous_cols = ['BertzCT', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3v',
       'Chi4n', 'EState_VSA1', 'EState_VSA2', 'ExactMolWt', 'FpDensityMorgan1',
       'FpDensityMorgan2', 'FpDensityMorgan3', 'HallKierAlpha',
       'HeavyAtomMolWt', 'Kappa3', 'MaxAbsEStateIndex', 'MinEStateIndex',
        'PEOE_VSA10', 'PEOE_VSA14', 'PEOE_VSA6', 'PEOE_VSA7',
       'PEOE_VSA8', 'SMR_VSA10', 'SMR_VSA5', 'SlogP_VSA3', 'VSA_EState9']

target_col = ['EC1','EC2']



## === cell 8
tab_preprocessor = TabPreprocessor(continuous_cols=continuous_cols, cat_embed_cols=cat_embed_cols)
X_tab = tab_preprocessor.fit_transform(train)
tab_preprocessor.cat_embed_input, X_tab.shape



## === cell 9
def train_binary_model(y_train: np.ndarray, tab_preprocessor: TabPreprocessor):
    """
    Train one binary model per target (same TabMlp architecture).
    """
    tab_mlp = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        cat_embed_dropout=0.,
        continuous_cols=continuous_cols,
        mlp_hidden_dims=[64, 32],
        mlp_dropout=0.5,
        mlp_activation="leaky_relu"
    )
    model = WideDeep(deeptabular=tab_mlp, pred_dim=1)

    trainer = Trainer(
        model=model,
        objective="binary",
        optimizers=torch.optim.Adam(model.parameters(), lr=0.0001)
    )
    trainer.fit(X_tab=X_tab, target=y_train, n_epochs=20, batch_size=64, val_split=0.2)
    return trainer

def get_pos_proba(trainer: Trainer, X_tab: np.ndarray) -> np.ndarray:
    proba = trainer.predict_proba(X_tab=X_tab)
    proba = np.asarray(proba)
    if proba.ndim == 1:
        return proba
    if proba.shape[1] == 1:
        return proba[:, 0]
    return proba[:, 1]



## === cell 10
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.use_deterministic_algorithms(False)
device = torch.device("cpu")

y_ec1 = train[["EC1"]].values.astype(float)
y_ec2 = train[["EC2"]].values.astype(float)

trainer_ec1 = train_binary_model(y_ec1, tab_preprocessor)
trainer_ec1.model.to(device)

trainer_ec2 = train_binary_model(y_ec2, tab_preprocessor)
trainer_ec2.model.to(device)


## === cell 11
test = data.get_dataframe('test.csv')
test.head()



## === cell 12
def train_binary_model(y_train: np.ndarray, tab_preprocessor: TabPreprocessor):
    """
    Train one binary model per target (same TabMlp architecture).
    """
    tab_mlp = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        cat_embed_dropout=0.0,
        continuous_cols=continuous_cols,
        mlp_hidden_dims=[64, 32],
        mlp_dropout=0.5,
        mlp_activation="leaky_relu",
    )
    model = WideDeep(deeptabular=tab_mlp, pred_dim=1)

    trainer = Trainer(
        model=model,
        objective="binary",
        optimizers=torch.optim.Adam(model.parameters(), lr=0.0001),
    )
    trainer.fit(X_tab=X_tab, target=y_train, n_epochs=20, batch_size=64, val_split=0.2)
    return trainer


def get_pos_proba(trainer: Trainer, X_tab: np.ndarray) -> np.ndarray:
    try:
        model_device = next(trainer.model.parameters()).device
        trainer.device = model_device
    except StopIteration:
        pass

    proba = trainer.predict_proba(X_tab=X_tab)
    proba = np.asarray(proba)
    if proba.ndim == 1:
        return proba
    if proba.shape[1] == 1:
        return proba[:, 0]
    return proba[:, 1]


## === cell 13
sample_sub = pd.read_csv('/kaggle/input/playground-series-s3e18/sample_submission.csv')
test_ids = test[['id']].copy()

df_pred = pd.DataFrame({'id': test_ids['id'].values, 'EC1': proba_ec1, 'EC2': proba_ec2})
df_submit = sample_sub[['id']].merge(df_pred, on='id', how='left')

df_submit['EC1'] = df_submit['EC1'].fillna(df_submit['EC1'].mean())
df_submit['EC2'] = df_submit['EC2'].fillna(df_submit['EC2'].mean())

df_submit.to_csv('submission.csv', index=False)
print('Submission Completed!!', df_submit.shape)



## --- ERROR in cell 13, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3405888988.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0mtest_ids[0m [0;34m=[0m [0mtest[0m[0;34m[[0m[0;34m[[0m[0;34m'id'[0m[0;34m][0m[0;34m][0m[0;34m.[0m[0mcopy[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;34m[0m[0m
[0;32m----> 4[0;31m [0mdf_pred[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mDataFrame[0m[0;34m([0m[0;34m{[0m[0;34m'id'[0m[0;34m:[0m [0mtest_ids[0m[0;34m[[0m[0;34m'id'[0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m,[0m [0;34m'EC1'[0m[0;34m:[0m [0mproba_ec1[0m[0;34m,[0m [0;34m'EC2'[0m[0;34m:[0m [0mproba_ec2[0m[0;34m}[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0mdf_submit[0m [0;34m=[0m [0msample_sub[0m[0;34m[[0m[0;34m[[0m[0;34m'id'[0m[0;34m][0m[0;34m][0m[0;34m.[0m[0mmerge[0m[0;34m([0m[0mdf_pred[0m[0;34m,[0m [0mon[0m[0;34m=[0m[0;34m'id'[0m[0;34m,[0m [0mhow[0m[0;34m=[0m[0;34m'left'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;34m[0m[0m

[0;31mNameError[0m: name 'proba_ec1' is not defined

## === cell 14
df_submit.head()
```
