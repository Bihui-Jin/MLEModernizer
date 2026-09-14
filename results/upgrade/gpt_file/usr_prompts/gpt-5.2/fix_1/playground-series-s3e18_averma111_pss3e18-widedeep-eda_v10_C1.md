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
Predict values for synthetic data.

### Description
## Metric
Area under the ROC curve for each target, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict the value for the targets `EC1` and `EC2`. The file should contain a header and have the following format:

```
id,EC1,EC2
14838,0.22,0.71
14839,0.78,0.43
14840,0.53,0.11
etc.
```

## Dataset 
- **train.csv** - the training dataset; `[EC1 - EC6]` are the (binary) targets, although you are only asked to predict `EC1` and `EC2`.
- **test.csv** - the test dataset; your objective is to predict the probability of the two targets `EC1` and `EC2`
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

# 3. Installed packages

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

# 4. Data file paths

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

# 5. Target score

0.62396

# 6. Current score

0.56043

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
%%capture
!pip install pytorch-widedeep


## === cell 1
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import os


import torch
from pytorch_widedeep.preprocessing import WidePreprocessor, TabPreprocessor
from pytorch_widedeep.training import Trainer
from pytorch_widedeep.models import Wide, TabMlp, WideDeep
from pytorch_widedeep.metrics import Accuracy, Precision,F1Score,Recall

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler


import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')
import itertools


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
class Datapreparation(object):
    
    def __init__(self,root_path):
        self.root_path = root_path
        
    def get_dataframe(self,filename):
        return pd.read_csv(os.path.join(self.root_path,filename))
    
    def summary(self,text, df):
        summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
        summary['null'] = df.isnull().sum()
        summary['unique'] = df.nunique()
        summary['min'] = df.min()
        summary['median'] = df.median()
        summary['max'] = df.max()
        summary['mean'] = df.mean()
        summary['std'] = df.std()
        summary['duplicate'] = df.duplicated().sum()
        return summary
    
    

    
data = Datapreparation('/kaggle/input/playground-series-s3e18')
train=data.get_dataframe('train.csv')


## === cell 3
data.summary('train',train)


## === cell 4
train.head()


## === cell 5
train.drop(["id",'EC3','EC4','EC5','EC6'], axis=1, inplace=True)


## === cell 6
cat_embed_cols = ['fr_COO','fr_COO2','NumHeteroatoms']
continuous_cols = ['BertzCT', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3v',
       'Chi4n', 'EState_VSA1', 'EState_VSA2', 'ExactMolWt', 'FpDensityMorgan1',
       'FpDensityMorgan2', 'FpDensityMorgan3', 'HallKierAlpha',
       'HeavyAtomMolWt', 'Kappa3', 'MaxAbsEStateIndex', 'MinEStateIndex',
        'PEOE_VSA10', 'PEOE_VSA14', 'PEOE_VSA6', 'PEOE_VSA7',
       'PEOE_VSA8', 'SMR_VSA10', 'SMR_VSA5', 'SlogP_VSA3', 'VSA_EState9']

target_col = ['EC1','EC2']
target = train[target_col].values
target=target.astype(float)


## === cell 7
target.shape


## === cell 8
tab_preprocessor = TabPreprocessor(continuous_cols=continuous_cols,cat_embed_cols=cat_embed_cols)
X_tab = tab_preprocessor.fit_transform(train)
tab_preprocessor.cat_embed_input,X_tab.shape


## === cell 9
tab_mlp = TabMlp(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    cat_embed_dropout=0.,
    continuous_cols=continuous_cols,
    mlp_hidden_dims=[64, 32],
    mlp_dropout=0.5,
    mlp_activation="leaky_relu"
)

tab_model = WideDeep(deeptabular=tab_mlp,pred_dim=2)
tab_model


## === cell 10
tab_trainer = Trainer(
    model=tab_model,
    objective="multiclass",
    optimizers=torch.optim.Adam(tab_model.parameters(), lr=0.0001)
)


## === cell 11
tab_trainer.fit(X_tab=X_tab, target=target, n_epochs=20, batch_size=64, val_split=0.2)


## === cell 12
test=data.get_dataframe('test.csv')
test.head()


## === cell 13
X_tab_te = tab_preprocessor.transform(test)
preds = tab_trainer.predict_proba(X_tab=X_tab_te)
print(f'The  probabilities are {preds}')


## === cell 14
preds=preds.tolist()
ec1,ec2=[],[]
for element in preds:
    ec1.append(element[0])
    
for element in preds:
    ec2.append(element[1])


## === cell 15
class Submit:
    
    def submit_predictions(self,test_updated):
        df_submit = pd.DataFrame(data={'id': test['id'],'EC1':ec1,'EC2':ec2})
        df_submit.to_csv('submission.csv',index=False)
        print('Submission Completed!!')
        return df_submit
        
        
submit = Submit()
df_submit=submit.submit_predictions(test)


## === cell 16
df_submit.head()
