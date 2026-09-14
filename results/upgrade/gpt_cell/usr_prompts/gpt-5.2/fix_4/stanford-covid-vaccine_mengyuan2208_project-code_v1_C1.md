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

3.10

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
xgboost==2.0.3

# 3. Data file paths

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

# 4. Code solution

## === cell 0
!pip3 install -q forgi[all]
!conda install -y -c bioconda viennarna


## === cell 1
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import forgi.graph.bulge_graph as fgb
import forgi.visual.mplotlib as fvm
sns.set(style='darkgrid')


## === cell 2
train = pd.read_json('../input/stanford-covid-vaccine/train.json',lines=True)
test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
sample_sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')


## === cell 3
print("train data shape: ", train.shape)
print("test data shape: ", test.shape)
print("sample submission shape: ", sample_sub.shape)


## === cell 4
train.head()


## === cell 5
test.head()


## === cell 6
sample_sub.head()


## === cell 7
train['seq_length'].value_counts()


## === cell 8
test['seq_length'].value_counts()


## === cell 9
fig, ax = plt.subplots(1, 2, figsize=(20,5))
sns.boxplot(data=train, x='signal_to_noise', ax=ax[0])
ax[0].set_title('Signal/Noise')
sns.countplot(data=train, y='SN_filter', ax=ax[1])
ax[1].set_title('SN_filter')
plt.show()


## === cell 10
avg_reactivity = np.array(list(map(np.array,train.reactivity))).mean(axis=0)
avg_deg_50C = np.array(list(map(np.array,train.deg_50C))).mean(axis=0)
avg_deg_pH10 = np.array(list(map(np.array,train.deg_pH10))).mean(axis=0)
avg_deg_Mg_50C = np.array(list(map(np.array,train.deg_Mg_50C))).mean(axis=0)
avg_deg_Mg_pH10 = np.array(list(map(np.array,train.deg_Mg_pH10))).mean(axis=0)


## === cell 11
plt.figure(figsize=(20,10))

sns.lineplot(x=range(68),y=avg_reactivity,label='avg_reactivity')
sns.lineplot(x=range(68),y=avg_deg_50C,label='avg_deg_50C')
sns.lineplot(x=range(68),y=avg_deg_pH10,label='avg_deg_ph10')
sns.lineplot(x=range(68),y=avg_deg_Mg_50C,label='avg_deg_Mg_50C')
sns.lineplot(x=range(68),y=avg_deg_Mg_pH10,label='avg_deg_Mg_pH10')

plt.xlabel('Positions on the RNA sequence')
plt.xticks(range(0,68))
plt.ylabel('Values')
plt.title('Average Target Values vs Positions')

plt.show()


## === cell 12
np.corrcoef(np.vstack((avg_reactivity, avg_deg_50C, avg_deg_pH10, avg_deg_Mg_50C, avg_deg_Mg_pH10)))


## === cell 13
def plot_sample(sample):
    
    """
    Reference: https://www.kaggle.com/erelin6613/openvaccine-rna-visualization
    Visualize RNA using viennarna
    Arguments:
    sample: pandas.series, a sample of RNA, must contain 'id', structure' and 'sequence'
    
    """
    struct = sample['structure']
    seq = sample['sequence']
    bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{struct}\n{seq}')[0]
    
    plt.figure(figsize=(20,8))
    fvm.plot_rna(bg)
    plt.title(f"RNA Structure (id: {sample.id})")
    plt.show()


## === cell 14
import importlib.util

sample = train.iloc[np.random.choice(len(train))]

if importlib.util.find_spec("RNA") is None:
    print(
        "Skipping RNA structure plot: ViennaRNA Python module 'RNA' is not available in this environment."
    )
else:
    plot_sample(sample)


## === cell 15
mask = train['SN_filter'] == 1
train = train[mask]


## === cell 16
train = train.drop(['signal_to_noise', 'SN_filter', 'reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10', 'deg_error_Mg_50C', 'deg_error_50C'], axis = 1)
train.shape


## === cell 17
train_data = []

for ID in train['id'].unique():
    entry = train.loc[train['id'] == ID]     
    for i in range(entry['seq_scored'].values[0]):
        sample_dict = {'id': entry['id'].values[0],
                       'id_seqpos': str(entry['id'].values[0]) + '_' + str(i),
                       'sequence': entry['sequence'].values[0][i],
                       'structure': entry['structure'].values[0][i],
                       'predicted_loop_type': entry['predicted_loop_type'].values[0][i],
                       'reactivity': entry['reactivity'].values[0][i],
                       'deg_Mg_pH10': entry['deg_Mg_pH10'].values[0][i],
                       'deg_pH10': entry['deg_pH10'].values[0][i],
                       'deg_Mg_50C': entry['deg_Mg_50C'].values[0][i],
                       'deg_50C': entry['deg_50C'].values[0][i]}
        train_data.append(sample_dict)
        
train_data = pd.DataFrame(train_data)
train_data.head()


## === cell 18
test_data = []

for ID in test['id'].unique():
    entry = test.loc[test['id'] == ID]     
    for i in range(entry['seq_length'].values[0]):
        sample_dict = {'id': entry['id'].values[0],
                       'id_seqpos': str(entry['id'].values[0]) + '_' + str(i),
                       'sequence': entry['sequence'].values[0][i],
                       'structure': entry['structure'].values[0][i],
                       'predicted_loop_type': entry['predicted_loop_type'].values[0][i]}
        test_data.append(sample_dict)
        
test_data = pd.DataFrame(test_data)
test_data.head()


## === cell 19
dict_sequence = {'A': 0, 'G' : 1, 'U' : 2, 'C' : 3}
dict_structure = {'(' : 0, ')' : 1, '.' : 2}
dict_looptype = {'S':0, 'M':1, 'I':2, 'B':3, 'H':4, 'E':5, 'X':6}

train_data['sequence'] = train_data['sequence'].replace(dict_sequence)
train_data['structure'] = train_data['structure'].replace(dict_structure)
train_data['predicted_loop_type'] = train_data['predicted_loop_type'].replace(dict_looptype)

test_data['sequence'] = test_data['sequence'].replace(dict_sequence)
test_data['structure'] = test_data['structure'].replace(dict_structure)
test_data['predicted_loop_type'] = test_data['predicted_loop_type'].replace(dict_looptype)

train_data.head()


## === cell 20
X_train = train_data.drop(['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C'], axis=1)
Y_train = train_data[['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']]


## === cell 21
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)
X_train.shape, X_val.shape, Y_train.shape, Y_val.shape


## === cell 22
def mcrmse_loss(y_true, y_pred, N = 5):
    """
    Calculates competition eval metric
    """
    n = len(y_true)
    return np.sum(np.sqrt(np.sum((y_true - y_pred)**2, axis = 0)/n)) / N


## === cell 23
xgb = XGBRegressor(
    n_estimators=800,
    eval_metric='rmse',
    learning_rate=0.1,
    subsample=0.8, # prevent overfitting
    colsample_bytree=0.8 # prevent overfitting
)


## === cell 24
features = ['sequence', 'structure', 'predicted_loop_type']
targets = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
sub = pd.DataFrame(test_data['id_seqpos'])
feature_importances = pd.DataFrame(index=features)
tr_X = X_train[features]
vl_X = X_val[features]
ts_X = test_data[features]
tr_X.shape, vl_X.shape, ts_X.shape


## === cell 25
for i in range(5):
    tr_Y, vl_Y = Y_train[targets[i]], Y_val[targets[i]]
    xgb.fit(tr_X, tr_Y)
    feature_importances.insert(i, targets[i], xgb.feature_importances_)
    vl_pred = xgb.predict(vl_X)
    loss = mcrmse_loss(vl_Y, vl_pred)
    print(f'{targets[i]} loss : {loss}')
    sub[targets[i]] = xgb.predict(ts_X)


## === cell 26
fig, ax = plt.subplots(3, 2, figsize = (12, 8))
fig.suptitle('Feature Importances Visualization')
for i in range(5):
    sns.barplot(x = features, y = targets[i], data=feature_importances, ax=ax[i // 2][i % 2])
plt.tight_layout()
plt.show()


## === cell 27
sub.to_csv('submission.csv', index=False)
sub.shape, sub.head()


## === cell 28
sub["id"], sub["seqpos"] = sub["id_seqpos"].str.rsplit("_", n=1).str
sub["seqpos"] = sub["seqpos"].astype(int)
sub = sub.sort_values(by=["id", "seqpos"]).reset_index(drop=True)
reac0 = sub.groupby("id")["reactivity"].apply(list)[0]
reac1 = sub.groupby("id")["reactivity"].apply(list)[1000]
reac2 = sub.groupby("id")["reactivity"].apply(list)[2000]
reac3 = sub.groupby("id")["reactivity"].apply(list)[3000]


## --- ERROR in cell 28, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1589917853.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;31m# Fix for pandas>=2.0 where Series.str.rsplit() requires keyword-only argument for n.[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;31m# Using n=1 preserves the original intent: split from the right into ['id', 'seqpos'].[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 3[0;31m [0msub[0m[0;34m[[0m[0;34m"id"[0m[0;34m][0m[0;34m,[0m [0msub[0m[0;34m[[0m[0;34m"seqpos"[0m[0;34m][0m [0;34m=[0m [0msub[0m[0;34m[[0m[0;34m"id_seqpos"[0m[0;34m][0m[0;34m.[0m[0mstr[0m[0;34m.[0m[0mrsplit[0m[0;34m([0m[0;34m"_"[0m[0;34m,[0m [0mn[0m[0;34m=[0m[0;36m1[0m[0;34m)[0m[0;34m.[0m[0mstr[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0msub[0m[0;34m[[0m[0;34m"seqpos"[0m[0;34m][0m [0;34m=[0m [0msub[0m[0;34m[[0m[0;34m"seqpos"[0m[0;34m][0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mint[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0msub[0m [0;34m=[0m [0msub[0m[0;34m.[0m[0msort_values[0m[0;34m([0m[0mby[0m[0;34m=[0m[0;34m[[0m[0;34m"id"[0m[0;34m,[0m [0;34m"seqpos"[0m[0;34m][0m[0;34m)[0m[0;34m.[0m[0mreset_index[0m[0;34m([0m[0mdrop[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/strings/accessor.py[0m in [0;36m__iter__[0;34m(self)[0m
[1;32m    251[0m [0;34m[0m[0m
[1;32m    252[0m     [0;32mdef[0m [0m__iter__[0m[0;34m([0m[0mself[0m[0;34m)[0m [0;34m->[0m [0mIterator[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 253[0;31m         [0;32mraise[0m [0mTypeError[0m[0;34m([0m[0;34mf"'{type(self).__name__}' object is not iterable"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    254[0m [0;34m[0m[0m
[1;32m    255[0m     def _wrap_result(

[0;31mTypeError[0m: 'StringMethods' object is not iterable

## === cell 29
fig, ax = plt.subplots(4, 1, sharex=True)
fig.suptitle('Predicted Reactivity vs Position')
sns.lineplot(data = reac0, ax=ax[0])
sns.lineplot(data = reac1, ax=ax[1])
sns.lineplot(data = reac2, ax=ax[2])
sns.lineplot(data = reac3, ax=ax[3])
plt.show()
