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
y_ec1 = train[['EC1']].values.astype(float)
y_ec2 = train[['EC2']].values.astype(float)

trainer_ec1 = train_binary_model(y_ec1, tab_preprocessor)
trainer_ec2 = train_binary_model(y_ec2, tab_preprocessor)



## --- ERROR in cell 10, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mRuntimeError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3517397752.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0my_ec2[0m [0;34m=[0m [0mtrain[0m[0;34m[[0m[0;34m[[0m[0;34m'EC2'[0m[0;34m][0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mfloat[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;34m[0m[0m
[0;32m----> 4[0;31m [0mtrainer_ec1[0m [0;34m=[0m [0mtrain_binary_model[0m[0;34m([0m[0my_ec1[0m[0;34m,[0m [0mtab_preprocessor[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0mtrainer_ec2[0m [0;34m=[0m [0mtrain_binary_model[0m[0;34m([0m[0my_ec2[0m[0;34m,[0m [0mtab_preprocessor[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2259442963.py[0m in [0;36mtrain_binary_model[0;34m(y_train, tab_preprocessor)[0m
[1;32m     19[0m         [0moptimizers[0m[0;34m=[0m[0mtorch[0m[0;34m.[0m[0moptim[0m[0;34m.[0m[0mAdam[0m[0;34m([0m[0mmodel[0m[0;34m.[0m[0mparameters[0m[0;34m([0m[0;34m)[0m[0;34m,[0m [0mlr[0m[0;34m=[0m[0;36m0.0001[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     20[0m     )
[0;32m---> 21[0;31m     [0mtrainer[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX_tab[0m[0;34m=[0m[0mX_tab[0m[0;34m,[0m [0mtarget[0m[0;34m=[0m[0my_train[0m[0;34m,[0m [0mn_epochs[0m[0;34m=[0m[0;36m20[0m[0;34m,[0m [0mbatch_size[0m[0;34m=[0m[0;36m64[0m[0;34m,[0m [0mval_split[0m[0;34m=[0m[0;36m0.2[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     22[0m     [0;32mreturn[0m [0mtrainer[0m[0;34m[0m[0;34m[0m[0m
[1;32m     23[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_widedeep/utils/general_utils.py[0m in [0;36mwrapper[0;34m(*args, **kwargs)[0m
[1;32m     59[0m                     [0mkwargs[0m[0;34m[[0m[0moriginal_name[0m[0;34m][0m [0;34m=[0m [0mkwargs[0m[0;34m.[0m[0mpop[0m[0;34m([0m[0malt_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     60[0m                     [0;32mbreak[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 61[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     62[0m [0;34m[0m[0m
[1;32m     63[0m         [0;32mreturn[0m [0mwrapper[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_widedeep/utils/general_utils.py[0m in [0;36mwrapper[0;34m(*args, **kwargs)[0m
[1;32m     59[0m                     [0mkwargs[0m[0;34m[[0m[0moriginal_name[0m[0;34m][0m [0;34m=[0m [0mkwargs[0m[0;34m.[0m[0mpop[0m[0;34m([0m[0malt_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     60[0m                     [0;32mbreak[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 61[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     62[0m [0;34m[0m[0m
[1;32m     63[0m         [0;32mreturn[0m [0mwrapper[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_widedeep/utils/general_utils.py[0m in [0;36mwrapper[0;34m(*args, **kwargs)[0m
[1;32m     59[0m                     [0mkwargs[0m[0;34m[[0m[0moriginal_name[0m[0;34m][0m [0;34m=[0m [0mkwargs[0m[0;34m.[0m[0mpop[0m[0;34m([0m[0malt_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     60[0m                     [0;32mbreak[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 61[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     62[0m [0;34m[0m[0m
[1;32m     63[0m         [0;32mreturn[0m [0mwrapper[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_widedeep/training/trainer.py[0m in [0;36mfit[0;34m(self, X_wide, X_tab, X_text, X_img, X_train, X_val, val_split, target, n_epochs, validation_freq, batch_size, train_dataloader, eval_dataloader, feature_importance_sample_size, finetune, stop_after_finetuning, **kwargs)[0m
[1;32m    485[0m         )
[1;32m    486[0m         [0;32mfor[0m [0mepoch[0m [0;32min[0m [0mrange[0m[0;34m([0m[0mn_epochs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 487[0;31m             [0mepoch_logs[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_train_epoch[0m[0;34m([0m[0mtrain_loader[0m[0;34m,[0m [0mepoch[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    488[0m             if eval_loader is not None and epoch % validation_freq == (
[1;32m    489[0m                 [0mvalidation_freq[0m [0;34m-[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_widedeep/training/trainer.py[0m in [0;36m_train_epoch[0;34m(self, train_loader, epoch)[0m
[1;32m    956[0m             [0;32mfor[0m [0mbatch_idx[0m[0;34m,[0m [0;34m([0m[0mdata[0m[0;34m,[0m [0mtargett[0m[0;34m)[0m [0;32min[0m [0mzip[0m[0;34m([0m[0mt[0m[0;34m,[0m [0mtrain_loader[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    957[0m                 [0mt[0m[0;34m.[0m[0mset_description[0m[0;34m([0m[0;34m"epoch %i"[0m [0;34m%[0m [0;34m([0m[0mepoch[0m [0;34m+[0m [0;36m1[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 958[0;31m                 [0mtrain_score[0m[0;34m,[0m [0mtrain_loss[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_train_step[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mtargett[0m[0;34m,[0m [0mbatch_idx[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    959[0m                 [0mprint_loss_and_metric[0m[0;34m([0m[0mt[0m[0;34m,[0m [0mtrain_loss[0m[0;34m,[0m [0mtrain_score[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    960[0m                 [0mself[0m[0;34m.[0m[0mcallback_container[0m[0;34m.[0m[0mon_batch_end[0m[0;34m([0m[0mbatch[0m[0;34m=[0m[0mbatch_idx[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_widedeep/training/trainer.py[0m in [0;36m_train_step[0;34m(self, data, target, batch_idx)[0m
[1;32m    988[0m         [0mself[0m[0;34m.[0m[0moptimizer[0m[0;34m.[0m[0mzero_grad[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    989[0m [0;34m[0m[0m
[0;32m--> 990[0;31m         [0my_pred[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mmodel[0m[0;34m([0m[0mX[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    991[0m [0;34m[0m[0m
[1;32m    992[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mis_model_tabnet[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

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

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py[0m in [0;36mforward[0;34m(self, input)[0m
[1;32m    123[0m [0;34m[0m[0m
[1;32m    124[0m     [0;32mdef[0m [0mforward[0m[0;34m([0m[0mself[0m[0;34m,[0m [0minput[0m[0;34m:[0m [0mTensor[0m[0;34m)[0m [0;34m->[0m [0mTensor[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 125[0;31m         [0;32mreturn[0m [0mF[0m[0;34m.[0m[0mlinear[0m[0;34m([0m[0minput[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mweight[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mbias[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    126[0m [0;34m[0m[0m
[1;32m    127[0m     [0;32mdef[0m [0mextra_repr[0m[0;34m([0m[0mself[0m[0;34m)[0m [0;34m->[0m [0mstr[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mRuntimeError[0m: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

## === cell 11
test = data.get_dataframe('test.csv')
test.head()
