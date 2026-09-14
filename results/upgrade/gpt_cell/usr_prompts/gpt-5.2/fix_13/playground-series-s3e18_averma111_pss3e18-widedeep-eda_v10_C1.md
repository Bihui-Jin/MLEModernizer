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
X_tab_te = tab_preprocessor.transform(test)

proba_ec1 = get_pos_proba(trainer_ec1, X_tab_te)
proba_ec2 = get_pos_proba(trainer_ec2, X_tab_te)

print("EC1 proba shape:", proba_ec1.shape, "EC2 proba shape:", proba_ec2.shape)



## --- ERROR in cell 12, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mRuntimeError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_10/3531987019.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0mX_tab_te[0m [0;34m=[0m [0mtab_preprocessor[0m[0;34m.[0m[0mtransform[0m[0;34m([0m[0mtest[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;34m[0m[0m
[0;32m----> 3[0;31m [0mproba_ec1[0m [0;34m=[0m [0mget_pos_proba[0m[0;34m([0m[0mtrainer_ec1[0m[0;34m,[0m [0mX_tab_te[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0mproba_ec2[0m [0;34m=[0m [0mget_pos_proba[0m[0;34m([0m[0mtrainer_ec2[0m[0;34m,[0m [0mX_tab_te[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_10/2259442963.py[0m in [0;36mget_pos_proba[0;34m(trainer, X_tab)[0m
[1;32m     24[0m [0;31m# Change (score -> target): robustly extract positive-class probability regardless of widedeep output shape[0m[0;34m[0m[0;34m[0m[0m
[1;32m     25[0m [0;32mdef[0m [0mget_pos_proba[0m[0;34m([0m[0mtrainer[0m[0;34m:[0m [0mTrainer[0m[0;34m,[0m [0mX_tab[0m[0;34m:[0m [0mnp[0m[0;34m.[0m[0mndarray[0m[0;34m)[0m [0;34m->[0m [0mnp[0m[0;34m.[0m[0mndarray[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 26[0;31m     [0mproba[0m [0;34m=[0m [0mtrainer[0m[0;34m.[0m[0mpredict_proba[0m[0;34m([0m[0mX_tab[0m[0;34m=[0m[0mX_tab[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     27[0m     [0mproba[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0masarray[0m[0;34m([0m[0mproba[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     28[0m     [0;32mif[0m [0mproba[0m[0;34m.[0m[0mndim[0m [0;34m==[0m [0;36m1[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_widedeep/training/trainer.py[0m in [0;36mpredict_proba[0;34m(self, X_wide, X_tab, X_text, X_img, X_test, batch_size)[0m
[1;32m    724[0m         """
[1;32m    725[0m [0;34m[0m[0m
[0;32m--> 726[0;31m         [0mpreds_l[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_predict[0m[0;34m([0m[0mX_wide[0m[0;34m,[0m [0mX_tab[0m[0;34m,[0m [0mX_text[0m[0;34m,[0m [0mX_img[0m[0;34m,[0m [0mX_test[0m[0;34m,[0m [0mbatch_size[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    727[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mmethod[0m [0;34m==[0m [0;34m"binary"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    728[0m             [0mpreds[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mvstack[0m[0;34m([0m[0mpreds_l[0m[0;34m)[0m[0;34m.[0m[0msqueeze[0m[0;34m([0m[0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_widedeep/training/trainer.py[0m in [0;36m_predict[0;34m(self, X_wide, X_tab, X_text, X_img, X_test, batch_size, uncertainty_granularity, uncertainty)[0m
[1;32m   1164[0m                                     [0mX[0m[0;34m[[0m[0mk[0m[0;34m][0m [0;34m=[0m [0mto_device[0m[0;34m([0m[0mv[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mdevice[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1165[0m                             preds = (
[0;32m-> 1166[0;31m                                 [0mself[0m[0;34m.[0m[0mmodel[0m[0;34m([0m[0mX[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1167[0m                                 [0;32mif[0m [0;32mnot[0m [0mself[0m[0;34m.[0m[0mis_model_tabnet[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1168[0m                                 [0;32melse[0m [0mself[0m[0;34m.[0m[0mmodel[0m[0;34m([0m[0mX[0m[0;34m)[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_wrapped_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1737[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_compiled_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m  [0;31m# type: ignore[misc][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1738[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1739[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1740[0m [0;34m[0m[0m
[1;32m   1741[0m     [0;31m# torchrec tests the code consistency with the following code[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1748[0m                 [0;32mor[0m [0m_global_backward_pre_hooks[0m [0;32mor[0m [0m_global_backward_hooks[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1749[0m                 or _global_forward_hooks or _global_forward_pre_hooks):
[0;32m-> 1750[0;31m             [0;32mreturn[0m [0mforward_call[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1751[0m [0;34m[0m[0m
[1;32m   1752[0m         [0mresult[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_widedeep/models/wide_deep.py[0m in [0;36mforward[0;34m(self, X, y)[0m
[1;32m    210[0m             [0mdeep[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_forward_deephead[0m[0;34m([0m[0mX[0m[0;34m,[0m [0mwide_out[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    211[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 212[0;31m             [0mdeep[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_forward_deep[0m[0;34m([0m[0mX[0m[0;34m,[0m [0mwide_out[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    213[0m [0;34m[0m[0m
[1;32m    214[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0menforce_positive[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_widedeep/models/wide_deep.py[0m in [0;36m_forward_deep[0;34m(self, X, wide_out)[0m
[1;32m    316[0m                 [0mwide_out[0m[0;34m.[0m[0madd_[0m[0;34m([0m[0mtab_out[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    317[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 318[0;31m                 wide_out = self._forward_component(
[0m[1;32m    319[0m                     [0mX[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mdeeptabular[0m[0;34m,[0m [0;34m"deeptabular"[0m[0;34m,[0m [0mwide_out[0m[0;34m[0m[0;34m[0m[0m
[1;32m    320[0m                 )

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_widedeep/models/wide_deep.py[0m in [0;36m_forward_component[0;34m(self, X, component, component_type, wide_out)[0m
[1;32m    384[0m             )
[1;32m    385[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 386[0;31m             [0mcomponent_out[0m [0;34m=[0m [0mcomponent[0m[0;34m([0m[0mX[0m[0;34m[[0m[0mcomponent_type[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    387[0m [0;34m[0m[0m
[1;32m    388[0m         [0;32mreturn[0m [0mwide_out[0m[0;34m.[0m[0madd_[0m[0;34m([0m[0mcomponent_out[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_wrapped_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1737[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_compiled_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m  [0;31m# type: ignore[misc][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1738[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1739[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1740[0m [0;34m[0m[0m
[1;32m   1741[0m     [0;31m# torchrec tests the code consistency with the following code[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1748[0m                 [0;32mor[0m [0m_global_backward_pre_hooks[0m [0;32mor[0m [0m_global_backward_hooks[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1749[0m                 or _global_forward_hooks or _global_forward_pre_hooks):
[0;32m-> 1750[0;31m             [0;32mreturn[0m [0mforward_call[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1751[0m [0;34m[0m[0m
[1;32m   1752[0m         [0mresult[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py[0m in [0;36mforward[0;34m(self, input)[0m
[1;32m    248[0m     [0;32mdef[0m [0mforward[0m[0;34m([0m[0mself[0m[0;34m,[0m [0minput[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    249[0m         [0;32mfor[0m [0mmodule[0m [0;32min[0m [0mself[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 250[0;31m             [0minput[0m [0;34m=[0m [0mmodule[0m[0;34m([0m[0minput[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    251[0m         [0;32mreturn[0m [0minput[0m[0;34m[0m[0;34m[0m[0m
[1;32m    252[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_wrapped_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1737[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_compiled_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m  [0;31m# type: ignore[misc][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1738[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1739[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1740[0m [0;34m[0m[0m
[1;32m   1741[0m     [0;31m# torchrec tests the code consistency with the following code[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1748[0m                 [0;32mor[0m [0m_global_backward_pre_hooks[0m [0;32mor[0m [0m_global_backward_hooks[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1749[0m                 or _global_forward_hooks or _global_forward_pre_hooks):
[0;32m-> 1750[0;31m             [0;32mreturn[0m [0mforward_call[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1751[0m [0;34m[0m[0m
[1;32m   1752[0m         [0mresult[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_widedeep/models/tabular/mlp/tab_mlp.py[0m in [0;36mforward[0;34m(self, X)[0m
[1;32m    204[0m [0;34m[0m[0m
[1;32m    205[0m     [0;32mdef[0m [0mforward[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mX[0m[0;34m:[0m [0mTensor[0m[0;34m)[0m [0;34m->[0m [0mTensor[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 206[0;31m         [0mx[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_get_embeddings[0m[0;34m([0m[0mX[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    207[0m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mencoder[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    208[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_widedeep/models/tabular/_base_tabular_model.py[0m in [0;36m_get_embeddings[0;34m(self, X)[0m
[1;32m    133[0m         [0mtensors_to_concat[0m[0;34m:[0m [0mList[0m[0;34m[[0m[0mTensor[0m[0;34m][0m [0;34m=[0m [0;34m[[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m    134[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mcat_embed_input[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 135[0;31m             [0mx_cat[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mcat_embed[0m[0;34m([0m[0mX[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    136[0m             [0mtensors_to_concat[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mx_cat[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    137[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_wrapped_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1737[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_compiled_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m  [0;31m# type: ignore[misc][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1738[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1739[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1740[0m [0;34m[0m[0m
[1;32m   1741[0m     [0;31m# torchrec tests the code consistency with the following code[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1748[0m                 [0;32mor[0m [0m_global_backward_pre_hooks[0m [0;32mor[0m [0m_global_backward_hooks[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1749[0m                 or _global_forward_hooks or _global_forward_pre_hooks):
[0;32m-> 1750[0;31m             [0;32mreturn[0m [0mforward_call[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1751[0m [0;34m[0m[0m
[1;32m   1752[0m         [0mresult[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_widedeep/models/tabular/embeddings_layers.py[0m in [0;36mforward[0;34m(self, X)[0m
[1;32m    383[0m [0;34m[0m[0m
[1;32m    384[0m     [0;32mdef[0m [0mforward[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mX[0m[0;34m:[0m [0mTensor[0m[0;34m)[0m [0;34m->[0m [0mTensor[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 385[0;31m         embed = [
[0m[1;32m    386[0m             self.embed_layers["emb_layer_" + self.embed_layers_names[col]](
[1;32m    387[0m                 [0mX[0m[0;34m[[0m[0;34m:[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mcolumn_idx[0m[0;34m[[0m[0mcol[0m[0;34m][0m[0;34m][0m[0;34m.[0m[0mlong[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_widedeep/models/tabular/embeddings_layers.py[0m in [0;36m<listcomp>[0;34m(.0)[0m
[1;32m    384[0m     [0;32mdef[0m [0mforward[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mX[0m[0;34m:[0m [0mTensor[0m[0;34m)[0m [0;34m->[0m [0mTensor[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    385[0m         embed = [
[0;32m--> 386[0;31m             self.embed_layers["emb_layer_" + self.embed_layers_names[col]](
[0m[1;32m    387[0m                 [0mX[0m[0;34m[[0m[0;34m:[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mcolumn_idx[0m[0;34m[[0m[0mcol[0m[0;34m][0m[0;34m][0m[0;34m.[0m[0mlong[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    388[0m             )

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_wrapped_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1737[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_compiled_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m  [0;31m# type: ignore[misc][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1738[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1739[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1740[0m [0;34m[0m[0m
[1;32m   1741[0m     [0;31m# torchrec tests the code consistency with the following code[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1748[0m                 [0;32mor[0m [0m_global_backward_pre_hooks[0m [0;32mor[0m [0m_global_backward_hooks[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1749[0m                 or _global_forward_hooks or _global_forward_pre_hooks):
[0;32m-> 1750[0;31m             [0;32mreturn[0m [0mforward_call[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1751[0m [0;34m[0m[0m
[1;32m   1752[0m         [0mresult[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/sparse.py[0m in [0;36mforward[0;34m(self, input)[0m
[1;32m    188[0m [0;34m[0m[0m
[1;32m    189[0m     [0;32mdef[0m [0mforward[0m[0;34m([0m[0mself[0m[0;34m,[0m [0minput[0m[0;34m:[0m [0mTensor[0m[0;34m)[0m [0;34m->[0m [0mTensor[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 190[0;31m         return F.embedding(
[0m[1;32m    191[0m             [0minput[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    192[0m             [0mself[0m[0;34m.[0m[0mweight[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/functional.py[0m in [0;36membedding[0;34m(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)[0m
[1;32m   2549[0m         [0;31m# remove once script supports set_grad_enabled[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2550[0m         [0m_no_grad_embedding_renorm_[0m[0;34m([0m[0mweight[0m[0;34m,[0m [0minput[0m[0;34m,[0m [0mmax_norm[0m[0;34m,[0m [0mnorm_type[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2551[0;31m     [0;32mreturn[0m [0mtorch[0m[0;34m.[0m[0membedding[0m[0;34m([0m[0mweight[0m[0;34m,[0m [0minput[0m[0;34m,[0m [0mpadding_idx[0m[0;34m,[0m [0mscale_grad_by_freq[0m[0;34m,[0m [0msparse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2552[0m [0;34m[0m[0m
[1;32m   2553[0m [0;34m[0m[0m

[0;31mRuntimeError[0m: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument index in method wrapper_CUDA__index_select)

## === cell 13
sample_sub = pd.read_csv('/kaggle/input/playground-series-s3e18/sample_submission.csv')
test_ids = test[['id']].copy()

df_pred = pd.DataFrame({'id': test_ids['id'].values, 'EC1': proba_ec1, 'EC2': proba_ec2})
df_submit = sample_sub[['id']].merge(df_pred, on='id', how='left')

df_submit['EC1'] = df_submit['EC1'].fillna(df_submit['EC1'].mean())
df_submit['EC2'] = df_submit['EC2'].fillna(df_submit['EC2'].mean())

df_submit.to_csv('submission.csv', index=False)
print('Submission Completed!!', df_submit.shape)
