# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Predict likely degradation rates at each base of an RNA molecule.

## Metric
Mean columnwise root mean squared error:

$\textrm{MCRMSE} = \frac{1}{N_{t}}\sum_{j=1}^{N_{t}}\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{ij} - \hat{y}_{ij})^2}$

where $N_{t}$ is the number of scored ground truth target columns, and $y$ and $\hat{y}$ are the actual and predicted values, respectively.

There are multiple ground truth values provided in the training data. While the submission format requires all 5 to be predicted, only the following are scored: reactivity, deg_Mg_pH10, and deg_Mg_50C.

## Submission Formats
For each sample `id` in the test set, you must predict targets for *each* sequence position (`seqpos`), one per row. If the length of the `sequence` of an `id` is, e.g., 107, then you should make 107 predictions. Positions greater than the `seq_scored` value of a sample are not scored, but still need a value in the solution file.

```csv
id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C
id_d190610e8_0,0.1,0.3,0.2,0.5,0.4
id_d190610e8_1,0.3,0.2,0.5,0.4,0.2
id_d190610e8_2,0.5,0.4,0.2,0.1,0.2
etc.
```

## Dataset 
- **train.json** - the training data
- **test.json** - the test set, without any columns associated with the ground truth.
- **sample_submission.csv** - a sample submission file in the correct format

#### Columns
- `id` - An arbitrary identifier for each sample.
- `seq_scored` - (68 in Train and Public Test, 68 in Private Test) Integer value denoting the number of positions used in scoring with predicted values. This should match the length of `reactivity`, `deg_*` and `*_error_*` columns.
- `seq_length` - (107 in Train and Public Test, 107 in Private Test) Integer values, denotes the length of `sequence`.
- `sequence` - (1x107 string in Train and Public Test, 107 in Private Test) Describes the RNA sequence, a combination of `A`, `G`, `U`, and `C` for each sample. Should be 107 characters long, and the first 68 bases should correspond to the 68 positions specified in `seq_scored` (note: indexed starting at 0).
- `structure` - (1x107 string in Train and Public Test, 107 in Private Test) An array of `(`, `)`, and `.` characters that describe whether a base is estimated to be paired or unpaired. Paired bases are denoted by opening and closing parentheses e.g. (....) means that base 0 is paired to base 5, and bases 1-4 are unpaired.
- `reactivity` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likely secondary structure of the RNA sample.
- `deg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high pH (pH 10).
- `deg_Mg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium in high pH (pH 10).
- `deg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high temperature (50 degrees Celsius).
- `deg_Mg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium at high temperature (50 degrees Celsius).
- `*_error_*` - An array of floating point numbers, should have the same length as the corresponding `reactivity` or `deg_*` columns, calculated errors in experimental values obtained in `reactivity` and `deg_*` columns.
- `predicted_loop_type` - (1x107 string) Describes the structural context (also referred to as 'loop type')of each character in `sequence`. Loop types assigned by bpRNA from Vienna RNAfold 2 structure. From the bpRNA_documentation: S: paired "Stem" M: Multiloop I: Internal loop B: Bulge H: Hairpin loop E: dangling End X: eXternal loop
    - `S/N filter` Indicates if the sample passed filters described below in `Additional Notes`.

#### Additional Notes
At the beginning of the competition, Stanford scientists have data on 2400 RNA sequences of length 107. For technical reasons, measurements cannot be carried out on the final bases of these RNA sequences, so we have experimental data (ground truth) in 5 conditions for the first 68 bases.

We have split out 240 of these 2400 sequences for a public test set to allow for continuous evaluation through the competition, on the public leaderboard. These sequences, in `test.json`, have been additionally filtered based on three criteria detailed below to ensure that this subset is not dominated by any large cluster of RNA molecules with poor data, which might bias the public leaderboard. The remaining 2160 sequences for which we have data are in `train.json`.

For our final and most important scoring (the Private Leaderbooard), Stanford scientists are carrying out measurements on 240 new RNAs. For these data, we expect to have measurements for the first 68 bases, again missing the ends of the RNA. These sequences constitute the 240 sequences in `test.json`.

For those interested in how the sequences in `test.json` were filtered, here were the steps to ensure a diverse and high quality test set for public leaderboard scoring:

1. Minimum value across all 5 conditions must be greater than -0.5.
2. Mean signal/noise across all 5 conditions must be greater than 1.0. [Signal/noise is defined as mean( measurement value over 68 nts )/mean( statistical error in measurement value over 68 nts)]
3. To help ensure sequence diversity, the resulting sequences were clustered into clusters with less than 50% sequence similarity, and the 240 test set sequences were chosen from clusters with 3 or fewer members. That is, any sequence in the test set should be sequence similar to at most 2 other sequences.

Note that these filters have not been applied to the 2160 RNAs in the public training data `train.json` -- some of those measurements have negative values or poor signal-to-noise, or some RNA sequences have near-identical sequences in that set. But we are providing all those data in case competitors can squeeze out more signal.

# 2. Python version

3.8

# 3. Installed packages

catboost==1.2.8
geopandas==0.14.4
imbalanced-learn==0.13.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
numpy==1.26.4
optuna==4.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0
statsmodels==0.14.5
tqdm==4.67.1
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (125 lines)
            sample_submission.csv (25681 lines)
            sample_submission.csv.zip (74.8 kB)
            test.json (240 lines)
            train.json (2160 lines)
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
        input/
            description.md (125 lines)
            sample_submission.csv (25681 lines)
            sample_submission.csv.zip (74.8 kB)
            test.json (240 lines)
            train.json (2160 lines)
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
        working/
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
```

-> data/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> data/stanford-covid-vaccine/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> data/stanford-covid-vaccine/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    }
  },
  "required": [
    "id",
    "index",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
    "sequence",
    "structure"
  ]
}

-> data/stanford-covid-vaccine/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "signal_to_noise": {
      "type": "number"
    },
    "SN_filter": {
      "type": "integer"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    },
    "reactivity_error": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "reactivity": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    }
  },
  "required": [
    "SN_filter",
    "deg_50C",
    "deg_Mg_50C",
    "deg_Mg_pH10",
    "deg_error_50C",
    "deg_error_Mg_50C",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_pH10",
    "id",
    "index",
    "predicted_loop_type",
    "reactivity",
    "reactivity_error",
    "seq_length",
    "seq_scored",
    "sequence",
    "signal_to_noise",
    "structure"
  ]
}

-> data/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    }
  },
  "required": [
    "id",
    "index",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
    "sequence",
    "structure"
  ]
}

-> data/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "signal_to_noise": {
      "type": "number"
    },
    "SN_filter": {
      "type": "integer"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    },
    "reactivity_error": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "reactivity": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    }
  },
  "required": [
    "SN_filter",
    "deg_50C",
    "deg_Mg_50C",
    "deg_Mg_pH10",
    "deg_error_50C",
    "deg_error_Mg_50C",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_pH10",
    "id",
    "index",
    "predicted_loop_type",
    "reactivity",
    "reactivity_error",
    "seq_length",
    "seq_scored",
    "sequence",
    "signal_to_noise",
    "structure"
  ]
}

-> input/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> (stopped after 10 files for performance)

# 5. Target score

0.92973

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import catboost
import optuna
import imblearn
from catboost import CatBoostRegressor
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
from catboost import *
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import Pool
from datetime import datetime
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression,RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from scipy.stats import norm,skew
from scipy import stats
from sklearn.metrics import mean_squared_error,make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import pandas as pd
import nltk
import operator
import re
import sys
from scipy import stats
from nltk.corpus import stopwords
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download("stopwords")
nltk.download("punkt")
import statsmodels.api as sm
from statsmodels.formula.api import ols
import time


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1772480288.py in <cell line: 0>()
      3 import catboost
      4 import optuna
----> 5 import imblearn
      6 from catboost import CatBoostRegressor
      7 from imblearn.under_sampling import RandomUnderSampler

/usr/local/lib/python3.11/dist-packages/imblearn/__init__.py in <module>
     50     # process, as it may not be compiled yet
     51 else:
---> 52     from . import (
     53         combine,
     54         ensemble,

/usr/local/lib/python3.11/dist-packages/imblearn/combine/__init__.py in <module>
      3 """
      4 
----> 5 from ._smote_enn import SMOTEENN
      6 from ._smote_tomek import SMOTETomek
      7 

/usr/local/lib/python3.11/dist-packages/imblearn/combine/_smote_enn.py in <module>
     10 from sklearn.utils import check_X_y
     11 
---> 12 from ..base import BaseSampler
     13 from ..over_sampling import SMOTE
     14 from ..over_sampling.base import BaseOverSampler

/usr/local/lib/python3.11/dist-packages/imblearn/base.py in <module>
     10 from sklearn.base import BaseEstimator, OneToOneFeatureMixin
     11 from sklearn.preprocessing import label_binarize
---> 12 from sklearn.utils._metadata_requests import METHODS
     13 from sklearn.utils.multiclass import check_classification_targets
     14 

ModuleNotFoundError: No module named 'sklearn.utils._metadata_requests'

## === cell 2
train = pd.read_json('../input/stanford-covid-vaccine/train.json',lines=True)
test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
ss = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')
train.shape, test.shape, ss.shape


## === cell 3
train


## === cell 4
ss


## === cell 5
test['structure'].value_counts()


## === cell 6
train.columns


## === cell 7
train['deg_Mg_pH10']


## === cell 8
ss.columns


## === cell 10
test['E']=[sum([i=='E' for i in j])/len(j) for j in test['predicted_loop_type']]
test['S']=[sum([i=='S' for i in j])/len(j) for j in test['predicted_loop_type']]
test['B']=[sum([i=='B' for i in j])/len(j) for j in test['predicted_loop_type']]
test['H']=[sum([i=='H' for i in j])/len(j) for j in test['predicted_loop_type']]
test['I']=[sum([i=='I' for i in j])/len(j) for j in test['predicted_loop_type']]

test['G']=[sum([i=='G' for i in j])/len(j) for j in test['sequence']]
test['A']=[sum([i=='A' for i in j])/len(j) for j in test['sequence']]
test['C']=[sum([i=='C' for i in j])/len(j) for j in test['sequence']]
test['U']=[sum([i=='U' for i in j])/len(j) for j in test['sequence']]
test['Paired']=[sum([i=='(' or i==')' for i in j]) for j in test['structure']]
test['Unpaired']=[sum([i=='.' for i in j]) for j in test['structure']]


## === cell 11
train['E']=[sum([i=='E' for i in j])/len(j) for j in train['predicted_loop_type']]
train['S']=[sum([i=='S' for i in j])/len(j) for j in train['predicted_loop_type']]
train['B']=[sum([i=='B' for i in j])/len(j) for j in train['predicted_loop_type']]
train['H']=[sum([i=='H' for i in j])/len(j) for j in train['predicted_loop_type']]
train['I']=[sum([i=='I' for i in j])/len(j) for j in train['predicted_loop_type']]

train['G']=[sum([i=='G' for i in j])/len(j) for j in train['sequence']]
train['A']=[sum([i=='A' for i in j])/len(j) for j in train['sequence']]
train['C']=[sum([i=='C' for i in j])/len(j) for j in train['sequence']]
train['U']=[sum([i=='U' for i in j])/len(j) for j in train['sequence']]
train['Paired']=[sum([i=='(' or i==')' for i in j]) for j in train['structure']]
train['Unpaired']=[sum([i=='.' for i in j]) for j in train['structure']]


## === cell 12
train.columns


## === cell 16
for a in [ 'G', 'A', 'C', 'U']:
    train[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['sequence']]
    test[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in test['sequence']]


## === cell 17
for a in [ 'E', 'S', 'H',]:
    train[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['predicted_loop_type']]
    test[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in test['predicted_loop_type']]


## === cell 18
for a in [ 'E', 'S', 'H',]:
    train[a+'']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['predicted_loop_type']]
    test[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in test['predicted_loop_type']]


## === cell 19
a='S'
[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['predicted_loop_type']]


## === cell 21
train.head()


## === cell 22
test['predicted_loop_type'][0:100]


## === cell 23
train.columns
    


## === cell 24
train.head()


## === cell 25
test.columns


## === cell 26
train['seq_length'][0]


## === cell 27
train_ex=pd.DataFrame()
for index in train.index:
    temp=pd.DataFrame()
    temp['id_seqpos']=[str(str(train['id'][index])+'_'+str(i)) for i in range(train['seq_scored'][index])]
    temp['sequence']=[train['sequence'][index][i] for i in range(train['seq_scored'][index])]
    temp['structure']=[train['structure'][index][i] for i in range(train['seq_scored'][index])]
    temp['predicted_loop_type']=[train['predicted_loop_type'][index][i] for i in range(train['seq_scored'][index])]
    temp['E']=train['E'][index]
    temp['S']=train['S'][index]
    temp['B']=train['B'][index]
    temp['H']=train['H'][index]
    temp['I']=train['I'][index]
    temp['G']=train['G'][index]
    temp['A']=train['A'][index]
    temp['C']=train['C'][index]
    temp['U']=train['U'][index]
    temp['Paired']=train['Paired'][index]
    temp['Unpaired']=train['Unpaired'][index]
    temp['G_position']=train['G_position'][index]
    temp['A_position']=train['A_position'][index]
    temp['C_position']=train['C_position'][index]
    temp['U_position']=train['U_position'][index]
    temp['E_position']=train['E_position'][index]
    temp['S_position']=train['S_position'][index]
    temp['H_position']=train['H_position'][index]
    temp['G']=train['G'][index]
    temp['A']=train['A'][index]
    temp['C']=train['C'][index]
    temp['U']=train['U'][index]
    temp['reactivity']=[train['reactivity'][index][i] for i in range(train['seq_scored'][index])]
    temp['deg_Mg_pH10']=[train['deg_Mg_pH10'][index][i] for i in range(train['seq_scored'][index])]
    temp['deg_pH10']=[train['deg_pH10'][index][i] for i in range(train['seq_scored'][index])]
    temp['deg_Mg_50C']=[train['deg_Mg_50C'][index][i] for i in range(train['seq_scored'][index])]
    temp['deg_50C']=[train['deg_50C'][index][i] for i in range(train['seq_scored'][index])]
    train_ex=train_ex.append(temp)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/515778142.py in <cell line: 0>()
     33     temp['deg_Mg_50C']=[train['deg_Mg_50C'][index][i] for i in range(train['seq_scored'][index])]
     34     temp['deg_50C']=[train['deg_50C'][index][i] for i in range(train['seq_scored'][index])]
---> 35     train_ex=train_ex.append(temp)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

## === cell 29
test_ex=pd.DataFrame()
for index in test.index:
    temp=pd.DataFrame()
    temp['id_seqpos']=[str(str(test['id'][index])+'_'+str(i)) for i in range(test['seq_length'][index])]
    temp['sequence']=[test['sequence'][index][i] for i in range(test['seq_length'][index])]
    temp['structure']=[test['structure'][index][i] for i in range(test['seq_length'][index])]
    temp['predicted_loop_type']=[test['predicted_loop_type'][index][i] for i in range(test['seq_length'][index])]
    temp['E']=test['E'][index]
    temp['S']=test['S'][index]
    temp['B']=test['B'][index]
    temp['H']=test['H'][index]
    temp['I']=test['I'][index]
    temp['G']=test['G'][index]
    temp['A']=test['A'][index]
    temp['C']=test['C'][index]
    temp['U']=test['U'][index]
    temp['Paired']=test['Paired'][index]
    temp['Unpaired']=test['Unpaired'][index]
    temp['G_position']=test['G_position'][index]
    temp['A_position']=test['A_position'][index]
    temp['C_position']=test['C_position'][index]
    temp['U_position']=test['U_position'][index]
    temp['E_position']=test['E_position'][index]
    temp['S_position']=test['S_position'][index]
    temp['H_position']=test['H_position'][index]
    test_ex=test_ex.append(temp)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2312578695.py in <cell line: 0>()
     24     temp['S_position']=test['S_position'][index]
     25     temp['H_position']=test['H_position'][index]
---> 26     test_ex=test_ex.append(temp)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

## === cell 30
train_ex=train_ex.replace(-1,0)


## === cell 32
result=test_ex


## === cell 33
x_test=test_ex[[i for i in test_ex.columns if i!='id_seqpos']]
x_t=train_ex[x_test.columns]
x_test=pd.get_dummies(x_test)
x_t=pd.get_dummies(x_t)
y_t=train_ex[['reactivity', 'deg_Mg_pH10',
       'deg_pH10', 'deg_Mg_50C', 'deg_50C']]
x_t=x_t[x_test.columns]


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3475143165.py in <cell line: 0>()
      1 x_test=test_ex[[i for i in test_ex.columns if i!='id_seqpos']]
      2 x_t=train_ex[x_test.columns]
----> 3 x_test=pd.get_dummies(x_test)
      4 x_t=pd.get_dummies(x_t)
      5 y_t=train_ex[['reactivity', 'deg_Mg_pH10',

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/encoding.py in get_dummies(data, prefix, prefix_sep, dummy_na, columns, sparse, drop_first, dtype)
    222             )
    223             with_dummies.append(dummy)
--> 224         result = concat(with_dummies, axis=1)
    225     else:
    226         result = _get_dummies_1d(

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/concat.py in concat(objs, axis, join, ignore_index, keys, levels, names, verify_integrity, sort, copy)
    380         copy = False
    381 
--> 382     op = _Concatenator(
    383         objs,
    384         axis=axis,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/concat.py in __init__(self, objs, axis, join, keys, levels, names, ignore_index, verify_integrity, copy, sort)
    443         self.copy = copy
    444 
--> 445         objs, keys = self._clean_keys_and_objs(objs, keys)
    446 
    447         # figure out what our result ndim is going to be

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/concat.py in _clean_keys_and_objs(self, objs, keys)
    505 
    506         if len(objs_list) == 0:
--> 507             raise ValueError("No objects to concatenate")
    508 
    509         if keys is None:

ValueError: No objects to concatenate

## === cell 34
x_t.columns


## === cell 37
x_train,x_valid,y_train,y_valid=train_test_split(x_t,y_t,test_size=0.1,shuffle=True)


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3486734280.py in <cell line: 0>()
----> 1 x_train,x_valid,y_train,y_valid=train_test_split(x_t,y_t,test_size=0.1,shuffle=True)

NameError: name 'train_test_split' is not defined

## === cell 38
import optuna
import xgboost as xgb
import sklearn
column='reactivity'
def objective(trial):
    dtrain = xgb.DMatrix(x_train.values, label=y_train[column])
    dvalid = xgb.DMatrix(x_valid.values, label=y_valid[column])

    param = {
        "silent": 1,
          "eval_metric": "rmse",
        "booster": "gbtree",
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
        'tree_method' : 'gpu_hist'
        
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)


    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, str("validation-"+param["eval_metric"]))
    bst = xgb.train(param, dtrain, evals=[(dvalid, "validation")],callbacks=[pruning_callback])
    preds = bst.predict(dvalid)
    rmse=np.sqrt(sklearn.metrics.mean_squared_error(y_valid[column], preds))
    return rmse


## === cell 39
study = optuna.create_study()
study.optimize(objective, n_trials=100)


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3066454264.py in <cell line: 0>()
      1 study = optuna.create_study()
----> 2 study.optimize(objective, n_trials=100)

/usr/local/lib/python3.11/dist-packages/optuna/study/study.py in optimize(self, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)
    488                 If nested invocation of this method occurs.
    489         """
--> 490         _optimize(
    491             study=self,
    492             func=func,

/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py in _optimize(study, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)
     61     try:
     62         if n_jobs == 1:
---> 63             _optimize_sequential(
     64                 study,
     65                 func,

/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py in _optimize_sequential(study, func, n_trials, timeout, catch, callbacks, gc_after_trial, reseed_sampler_rng, time_start, progress_bar)
    158 
    159         try:
--> 160             frozen_trial_id = _run_trial(study, func, catch)
    161         finally:
    162             # The following line mitigates memory problems that can be occurred in some

/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py in _run_trial(study, func, catch)
    256         and not isinstance(func_err, catch)
    257     ):
--> 258         raise func_err
    259     return trial._trial_id
    260 

/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py in _run_trial(study, func, catch)
    199     with get_heartbeat_thread(trial._trial_id, study._storage):
    200         try:
--> 201             value_or_values = func(trial)
    202         except exceptions.TrialPruned as e:
    203             # TODO(mamu): Handle multi-objective cases.

/tmp/ipykernel_11/1640465053.py in objective(trial)
      4 column='reactivity'
      5 def objective(trial):
----> 6     dtrain = xgb.DMatrix(x_train.values, label=y_train[column])
      7     dvalid = xgb.DMatrix(x_valid.values, label=y_valid[column])
      8 

NameError: name 'x_train' is not defined

## === cell 40
print(study.best_params)


## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3696326561.py in <cell line: 0>()
----> 1 print(study.best_params)

/usr/local/lib/python3.11/dist-packages/optuna/study/study.py in best_params(self)
    118         """
    119 
--> 120         return self.best_trial.params
    121 
    122     @property

/usr/local/lib/python3.11/dist-packages/optuna/study/study.py in best_trial(self)
    154 
    155         """
--> 156         return self._get_best_trial(deepcopy=True)
    157 
    158     @property

/usr/local/lib/python3.11/dist-packages/optuna/study/study.py in _get_best_trial(self, deepcopy)
    306             )
    307 
--> 308         best_trial = self._storage.get_best_trial(self._study_id)
    309 
    310         # If the trial with the best value is infeasible, select the best trial from all feasible

/usr/local/lib/python3.11/dist-packages/optuna/storages/_in_memory.py in get_best_trial(self, study_id)
    250 
    251             if best_trial_id is None:
--> 252                 raise ValueError("No trials are completed yet.")
    253             elif len(self._studies[study_id].directions) > 1:
    254                 raise RuntimeError(

ValueError: No trials are completed yet.

## === cell 42
dtrain = xgb.DMatrix(x_train.values, label=y_train[column])
dvalid = xgb.DMatrix(x_valid.values, label=y_valid[column])
dtest = xgb.DMatrix(x_test.values)
bst = xgb.train( study.best_params,dtrain, evals=[(dvalid, "validation")])
preds = bst.predict(dtest)
result[column]=preds


## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4054688939.py in <cell line: 0>()
----> 1 dtrain = xgb.DMatrix(x_train.values, label=y_train[column])
      2 dvalid = xgb.DMatrix(x_valid.values, label=y_valid[column])
      3 dtest = xgb.DMatrix(x_test.values)
      4 bst = xgb.train( study.best_params,dtrain, evals=[(dvalid, "validation")])
      5 preds = bst.predict(dtest)

NameError: name 'x_train' is not defined

## === cell 43
import optuna
import xgboost as xgb
import sklearn
column='deg_Mg_pH10'
def objective(trial):
    column='deg_Mg_pH10'
    dtrain = xgb.DMatrix(x_train.values, label=y_train[column])
    dvalid = xgb.DMatrix(x_valid.values, label=y_valid[column])

    param = {
        "silent": 1,
          "eval_metric": "rmse",
        "booster": "gbtree",
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
        'tree_method' : 'gpu_hist'
        
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)


    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, str("validation-"+param["eval_metric"]))
    bst = xgb.train(param, dtrain, evals=[(dvalid, "validation")],callbacks=[pruning_callback])
    preds = bst.predict(dvalid)
    rmse=np.sqrt(sklearn.metrics.mean_squared_error(y_valid[column], preds))
    return rmse


## === cell 44
study = optuna.create_study()
study.optimize(objective, n_trials=100)


## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3066454264.py in <cell line: 0>()
      1 study = optuna.create_study()
----> 2 study.optimize(objective, n_trials=100)

/usr/local/lib/python3.11/dist-packages/optuna/study/study.py in optimize(self, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)
    488                 If nested invocation of this method occurs.
    489         """
--> 490         _optimize(
    491             study=self,
    492             func=func,

/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py in _optimize(study, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)
     61     try:
     62         if n_jobs == 1:
---> 63             _optimize_sequential(
     64                 study,
     65                 func,

/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py in _optimize_sequential(study, func, n_trials, timeout, catch, callbacks, gc_after_trial, reseed_sampler_rng, time_start, progress_bar)
    158 
    159         try:
--> 160             frozen_trial_id = _run_trial(study, func, catch)
    161         finally:
    162             # The following line mitigates memory problems that can be occurred in some

/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py in _run_trial(study, func, catch)
    256         and not isinstance(func_err, catch)
    257     ):
--> 258         raise func_err
    259     return trial._trial_id
    260 

/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py in _run_trial(study, func, catch)
    199     with get_heartbeat_thread(trial._trial_id, study._storage):
    200         try:
--> 201             value_or_values = func(trial)
    202         except exceptions.TrialPruned as e:
    203             # TODO(mamu): Handle multi-objective cases.

/tmp/ipykernel_11/2023147431.py in objective(trial)
      5 def objective(trial):
      6     column='deg_Mg_pH10'
----> 7     dtrain = xgb.DMatrix(x_train.values, label=y_train[column])
      8     dvalid = xgb.DMatrix(x_valid.values, label=y_valid[column])
      9 

NameError: name 'x_train' is not defined

## === cell 45
study.best_params


## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/919701543.py in <cell line: 0>()
----> 1 study.best_params

/usr/local/lib/python3.11/dist-packages/optuna/study/study.py in best_params(self)
    118         """
    119 
--> 120         return self.best_trial.params
    121 
    122     @property

/usr/local/lib/python3.11/dist-packages/optuna/study/study.py in best_trial(self)
    154 
    155         """
--> 156         return self._get_best_trial(deepcopy=True)
    157 
    158     @property

/usr/local/lib/python3.11/dist-packages/optuna/study/study.py in _get_best_trial(self, deepcopy)
    306             )
    307 
--> 308         best_trial = self._storage.get_best_trial(self._study_id)
    309 
    310         # If the trial with the best value is infeasible, select the best trial from all feasible

/usr/local/lib/python3.11/dist-packages/optuna/storages/_in_memory.py in get_best_trial(self, study_id)
    250 
    251             if best_trial_id is None:
--> 252                 raise ValueError("No trials are completed yet.")
    253             elif len(self._studies[study_id].directions) > 1:
    254                 raise RuntimeError(

ValueError: No trials are completed yet.

## === cell 46
dtrain = xgb.DMatrix(x_train.values, label=y_train[column])
dvalid = xgb.DMatrix(x_valid.values, label=y_valid[column])
dtest = xgb.DMatrix(x_test.values)
bst = xgb.train( study.best_params,dtrain, evals=[(dvalid, "validation")])
preds = bst.predict(dtest)
result[column]=preds


## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4054688939.py in <cell line: 0>()
----> 1 dtrain = xgb.DMatrix(x_train.values, label=y_train[column])
      2 dvalid = xgb.DMatrix(x_valid.values, label=y_valid[column])
      3 dtest = xgb.DMatrix(x_test.values)
      4 bst = xgb.train( study.best_params,dtrain, evals=[(dvalid, "validation")])
      5 preds = bst.predict(dtest)

NameError: name 'x_train' is not defined

## === cell 47
import optuna
import xgboost as xgb
import sklearn
column='deg_Mg_50C'
def objective(trial):
    column='deg_Mg_50C'
    dtrain = xgb.DMatrix(x_train.values, label=y_train[column])
    dvalid = xgb.DMatrix(x_valid.values, label=y_valid[column])

    param = {
        "silent": 1,
          "eval_metric": "rmse",
        "booster": "gbtree",
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
        'tree_method' : 'gpu_hist'
        
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)


    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, str("validation-"+param["eval_metric"]))
    bst = xgb.train(param, dtrain, evals=[(dvalid, "validation")],callbacks=[pruning_callback])
    preds = bst.predict(dvalid)
    rmse=np.sqrt(sklearn.metrics.mean_squared_error(y_valid[column], preds))
    return rmse


## === cell 48
study = optuna.create_study()
study.optimize(objective, n_trials=100)


## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3066454264.py in <cell line: 0>()
      1 study = optuna.create_study()
----> 2 study.optimize(objective, n_trials=100)

/usr/local/lib/python3.11/dist-packages/optuna/study/study.py in optimize(self, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)
    488                 If nested invocation of this method occurs.
    489         """
--> 490         _optimize(
    491             study=self,
    492             func=func,

/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py in _optimize(study, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)
     61     try:
     62         if n_jobs == 1:
---> 63             _optimize_sequential(
     64                 study,
     65                 func,

/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py in _optimize_sequential(study, func, n_trials, timeout, catch, callbacks, gc_after_trial, reseed_sampler_rng, time_start, progress_bar)
    158 
    159         try:
--> 160             frozen_trial_id = _run_trial(study, func, catch)
    161         finally:
    162             # The following line mitigates memory problems that can be occurred in some

/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py in _run_trial(study, func, catch)
    256         and not isinstance(func_err, catch)
    257     ):
--> 258         raise func_err
    259     return trial._trial_id
    260 

/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py in _run_trial(study, func, catch)
    199     with get_heartbeat_thread(trial._trial_id, study._storage):
    200         try:
--> 201             value_or_values = func(trial)
    202         except exceptions.TrialPruned as e:
    203             # TODO(mamu): Handle multi-objective cases.

/tmp/ipykernel_11/3947453071.py in objective(trial)
      5 def objective(trial):
      6     column='deg_Mg_50C'
----> 7     dtrain = xgb.DMatrix(x_train.values, label=y_train[column])
      8     dvalid = xgb.DMatrix(x_valid.values, label=y_valid[column])
      9 

NameError: name 'x_train' is not defined

## === cell 49
study.best_params


## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/919701543.py in <cell line: 0>()
----> 1 study.best_params

/usr/local/lib/python3.11/dist-packages/optuna/study/study.py in best_params(self)
    118         """
    119 
--> 120         return self.best_trial.params
    121 
    122     @property

/usr/local/lib/python3.11/dist-packages/optuna/study/study.py in best_trial(self)
    154 
    155         """
--> 156         return self._get_best_trial(deepcopy=True)
    157 
    158     @property

/usr/local/lib/python3.11/dist-packages/optuna/study/study.py in _get_best_trial(self, deepcopy)
    306             )
    307 
--> 308         best_trial = self._storage.get_best_trial(self._study_id)
    309 
    310         # If the trial with the best value is infeasible, select the best trial from all feasible

/usr/local/lib/python3.11/dist-packages/optuna/storages/_in_memory.py in get_best_trial(self, study_id)
    250 
    251             if best_trial_id is None:
--> 252                 raise ValueError("No trials are completed yet.")
    253             elif len(self._studies[study_id].directions) > 1:
    254                 raise RuntimeError(

ValueError: No trials are completed yet.

## === cell 50
dtrain = xgb.DMatrix(x_train.values, label=y_train[column])
dvalid = xgb.DMatrix(x_valid.values, label=y_valid[column])
dtest = xgb.DMatrix(x_test.values)
bst = xgb.train( study.best_params,dtrain, evals=[(dvalid, "validation")])
preds = bst.predict(dtest)
result[column]=preds


## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4054688939.py in <cell line: 0>()
----> 1 dtrain = xgb.DMatrix(x_train.values, label=y_train[column])
      2 dvalid = xgb.DMatrix(x_valid.values, label=y_valid[column])
      3 dtest = xgb.DMatrix(x_test.values)
      4 bst = xgb.train( study.best_params,dtrain, evals=[(dvalid, "validation")])
      5 preds = bst.predict(dtest)

NameError: name 'x_train' is not defined

## === cell 51
result.head()


## === cell 52
result['deg_pH10']=0
result['deg_50C']=0


## === cell 53
result[['id_seqpos','reactivity', 'deg_Mg_pH10',
       'deg_pH10', 'deg_Mg_50C', 'deg_50C']].to_csv('submission.csv',index=False)


## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/138898551.py in <cell line: 0>()
----> 1 result[['id_seqpos','reactivity', 'deg_Mg_pH10',
      2        'deg_pH10', 'deg_Mg_50C', 'deg_50C']].to_csv('submission.csv',index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['id_seqpos', 'reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'] not in index"

## === cell 54
result.shape


## === cell 55
xgb.plot_importance(bst)


## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1772087365.py in <cell line: 0>()
----> 1 xgb.plot_importance(bst)

NameError: name 'bst' is not defined
