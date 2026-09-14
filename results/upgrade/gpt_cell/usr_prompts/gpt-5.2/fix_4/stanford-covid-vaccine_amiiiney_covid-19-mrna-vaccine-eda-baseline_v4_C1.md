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

3.8

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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore')

train = pd.read_json('../input/stanford-covid-vaccine/train.json',lines=True)
test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')

def plotd(f1,f2):
    plt.style.use('seaborn')
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(15,5))
    ax1 = plt.subplot2grid((1,2),(0,0))
    plt.hist(a[f1], bins=4, color='black',alpha=0.5)
    plt.title(f'{f1}',weight='bold', fontsize=18)
    ax1 = plt.subplot2grid((1,2),(0,1))
    plt.hist(a[f2], bins=4, color='crimson',alpha=0.5)
    plt.title(f'{f2}',weight='bold', fontsize=18)
 
    plt.show()

def plotc(f1,f2):
    plt.style.use('seaborn')
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(15,5))
    ax1 = plt.subplot2grid((1,2),(0,0))
    plt.hist(a[f1], bins=7, color='black',alpha=0.7)
    plt.title(f'{f1}',weight='bold', fontsize=18)
    ax1 = plt.subplot2grid((1,2),(0,1))
    plt.hist(a[f2], bins=5, color='crimson',alpha=0.7)
    plt.title(f'{f2}',weight='bold', fontsize=18)
    plt.xticks(weight='bold')
    plt.show()
    
def ploth(data, w=15, h=9):
    plt.figure(figsize=(w,h))
    sns.heatmap(data.corr(), cmap='hot', annot=True)
    plt.title('Correlation between the features', fontsize=18, weight='bold')
    plt.xticks(weight='bold')
    plt.yticks(weight='bold')
    return plt.show()


## === cell 1
train.head()


## === cell 2
def length(feature):
    column= train[[feature]]
    column['length']= column[feature].apply(len)
    return column.head()

length('sequence')


## === cell 3
length('reactivity')


## === cell 4
train_data = []
for mol_id in train['id'].unique():
    sample_data = train.loc[train['id'] == mol_id]
    for i in range(68):
        sample_tuple = (sample_data['id'].values[0], sample_data['sequence'].values[0][i],
                        sample_data['structure'].values[0][i], sample_data['predicted_loop_type'].values[0][i],
                        sample_data['reactivity'].values[0][i], sample_data['reactivity_error'].values[0][i],
                        sample_data['deg_Mg_pH10'].values[0][i], sample_data['deg_error_Mg_pH10'].values[0][i],
                        sample_data['deg_pH10'].values[0][i], sample_data['deg_error_pH10'].values[0][i],
                        sample_data['deg_Mg_50C'].values[0][i], sample_data['deg_error_Mg_50C'].values[0][i],
                        sample_data['deg_50C'].values[0][i], sample_data['deg_error_50C'].values[0][i])
        train_data.append(sample_tuple)


## === cell 5
a = pd.DataFrame(train_data, columns=['id', 'sequence', 'structure', 'predicted_loop_type', 'reactivity', 'reactivity_error', 'deg_Mg_pH10', 'deg_error_Mg_pH10',
                                  'deg_pH10', 'deg_error_pH10', 'deg_Mg_50C', 'deg_error_Mg_50C', 'deg_50C', 'deg_error_50C'])
a.head()


## === cell 6
plotd('reactivity', 'reactivity_error')


## === cell 7
plotd('deg_50C','deg_Mg_50C')


## === cell 8
plotd('deg_pH10','deg_Mg_pH10')


## === cell 9
plotc('predicted_loop_type', 'structure')


## === cell 10
sns.countplot(x=a["sequence"], palette="terrain", alpha=0.8)
plt.title("Nucleotides count per sequence", weight="bold", fontsize=12)
plt.show()


## === cell 11
b=a[['reactivity', 'deg_Mg_pH10',
                                  'deg_pH10', 'deg_Mg_50C', 'deg_50C']]

ploth(b, 10, 4)


## === cell 12
c = a[["id", "sequence", "structure", "predicted_loop_type"]]
c = pd.get_dummies(c, columns=["sequence", "structure", "predicted_loop_type"])

ploth(c.drop(columns=["id"]))


## === cell 13
public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()


public_data = []
for mol_id in public_df['id'].unique():
    sample_data = public_df.loc[public_df['id'] == mol_id]
    for i in range(68):
        sample_tuple = (sample_data['id'].values[0] + '_' + str(i),
                        sample_data['sequence'].values[0][i],
                        sample_data['structure'].values[0][i], 
                        sample_data['predicted_loop_type'].values[0][i],
                        )
        public_data.append(sample_tuple)

pudf=pd.DataFrame(public_data, columns=['id', 'sequence', 'structure', 'predicted_loop_type'])
        

private_data = []
for mol_id in private_df['id'].unique():
    sample_data = private_df.loc[private_df['id'] == mol_id]
    for i in range(91):
        sample_tuple = (sample_data['id'].values[0] + '_' + str(i),
                        sample_data['sequence'].values[0][i],
                        sample_data['structure'].values[0][i], sample_data['predicted_loop_type'].values[0][i],
                        )
        private_data.append(sample_tuple)
        
prdf=pd.DataFrame(private_data, columns=['id', 'sequence', 'structure', 'predicted_loop_type'])


X2= pd.get_dummies(pudf, columns=['sequence', 'structure', 'predicted_loop_type'])
X3= pd.get_dummies(prdf, columns=['sequence', 'structure', 'predicted_loop_type'])


X2= X2.drop('id', axis=1)
X3= X3.drop('id', axis=1)
X=c.drop('id', axis=1)


## === cell 14
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

X2 = X2.reindex(columns=X.columns, fill_value=0)
X3 = X3.reindex(columns=X.columns, fill_value=0)

model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42)).fit(X, b)

public_preds = model.predict(X2)

private_preds = model.predict(X3)

pu_predictions = pd.DataFrame(public_preds, columns=b.columns)
pr_predictions = pd.DataFrame(private_preds, columns=b.columns)


## --- ERROR in cell 14, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1167697287.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     11[0m [0mpublic_preds[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mX2[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m [0;34m[0m[0m
[0;32m---> 13[0;31m [0mprivate_preds[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mX3[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     14[0m [0;34m[0m[0m
[1;32m     15[0m [0mpu_predictions[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mDataFrame[0m[0;34m([0m[0mpublic_preds[0m[0;34m,[0m [0mcolumns[0m[0;34m=[0m[0mb[0m[0;34m.[0m[0mcolumns[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/multioutput.py[0m in [0;36mpredict[0;34m(self, X)[0m
[1;32m    246[0m             [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"The base estimator should implement a predict method"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    247[0m [0;34m[0m[0m
[0;32m--> 248[0;31m         y = Parallel(n_jobs=self.n_jobs)(
[0m[1;32m    249[0m             [0mdelayed[0m[0;34m([0m[0me[0m[0;34m.[0m[0mpredict[0m[0;34m)[0m[0;34m([0m[0mX[0m[0;34m)[0m [0;32mfor[0m [0me[0m [0;32min[0m [0mself[0m[0;34m.[0m[0mestimators_[0m[0;34m[0m[0;34m[0m[0m
[1;32m    250[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py[0m in [0;36m__call__[0;34m(self, iterable)[0m
[1;32m     61[0m             [0;32mfor[0m [0mdelayed_func[0m[0;34m,[0m [0margs[0m[0;34m,[0m [0mkwargs[0m [0;32min[0m [0miterable[0m[0;34m[0m[0;34m[0m[0m
[1;32m     62[0m         )
[0;32m---> 63[0;31m         [0;32mreturn[0m [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__call__[0m[0;34m([0m[0miterable_with_config[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     64[0m [0;34m[0m[0m
[1;32m     65[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/joblib/parallel.py[0m in [0;36m__call__[0;34m(self, iterable)[0m
[1;32m   1984[0m             [0moutput[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_get_sequential_output[0m[0;34m([0m[0miterable[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1985[0m             [0mnext[0m[0;34m([0m[0moutput[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1986[0;31m             [0;32mreturn[0m [0moutput[0m [0;32mif[0m [0mself[0m[0;34m.[0m[0mreturn_generator[0m [0;32melse[0m [0mlist[0m[0;34m([0m[0moutput[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1987[0m [0;34m[0m[0m
[1;32m   1988[0m         [0;31m# Let's create an ID that uniquely identifies the current call. If the[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/joblib/parallel.py[0m in [0;36m_get_sequential_output[0;34m(self, iterable)[0m
[1;32m   1912[0m                 [0mself[0m[0;34m.[0m[0mn_dispatched_batches[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1913[0m                 [0mself[0m[0;34m.[0m[0mn_dispatched_tasks[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1914[0;31m                 [0mres[0m [0;34m=[0m [0mfunc[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1915[0m                 [0mself[0m[0;34m.[0m[0mn_completed_tasks[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1916[0m                 [0mself[0m[0;34m.[0m[0mprint_progress[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py[0m in [0;36m__call__[0;34m(self, *args, **kwargs)[0m
[1;32m    121[0m             [0mconfig[0m [0;34m=[0m [0;34m{[0m[0;34m}[0m[0;34m[0m[0;34m[0m[0m
[1;32m    122[0m         [0;32mwith[0m [0mconfig_context[0m[0;34m([0m[0;34m**[0m[0mconfig[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 123[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mfunction[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py[0m in [0;36mpredict[0;34m(self, X)[0m
[1;32m   1796[0m             [0mThe[0m [0mpredicted[0m [0mvalues[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1797[0m         """
[0;32m-> 1798[0;31m         X = self._validate_data(
[0m[1;32m   1799[0m             [0mX[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mDTYPE[0m[0;34m,[0m [0morder[0m[0;34m=[0m[0;34m"C"[0m[0;34m,[0m [0maccept_sparse[0m[0;34m=[0m[0;34m"csr"[0m[0;34m,[0m [0mreset[0m[0;34m=[0m[0;32mFalse[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1800[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/base.py[0m in [0;36m_validate_data[0;34m(self, X, y, reset, validate_separately, **check_params)[0m
[1;32m    563[0m             [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"Validation should be done on X, y or both."[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    564[0m         [0;32melif[0m [0;32mnot[0m [0mno_val_X[0m [0;32mand[0m [0mno_val_y[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 565[0;31m             [0mX[0m [0;34m=[0m [0mcheck_array[0m[0;34m([0m[0mX[0m[0;34m,[0m [0minput_name[0m[0;34m=[0m[0;34m"X"[0m[0;34m,[0m [0;34m**[0m[0mcheck_params[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    566[0m             [0mout[0m [0;34m=[0m [0mX[0m[0;34m[0m[0;34m[0m[0m
[1;32m    567[0m         [0;32melif[0m [0mno_val_X[0m [0;32mand[0m [0;32mnot[0m [0mno_val_y[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py[0m in [0;36mcheck_array[0;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)[0m
[1;32m    929[0m         [0mn_samples[0m [0;34m=[0m [0m_num_samples[0m[0;34m([0m[0marray[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    930[0m         [0;32mif[0m [0mn_samples[0m [0;34m<[0m [0mensure_min_samples[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 931[0;31m             raise ValueError(
[0m[1;32m    932[0m                 [0;34m"Found array with %d sample(s) (shape=%s) while a"[0m[0;34m[0m[0;34m[0m[0m
[1;32m    933[0m                 [0;34m" minimum of %d is required%s."[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: Found array with 0 sample(s) (shape=(0, 14)) while a minimum of 1 is required by GradientBoostingRegressor.

## === cell 15
pu_predictions['id_seqpos']= pudf['id']
pr_predictions['id_seqpos']= prdf['id']
final= pd.concat([pu_predictions, pr_predictions])
sub1=sub.merge(final, on='id_seqpos', how='left')
sub1= sub1.drop(['reactivity_x', 'deg_Mg_pH10_x', 'deg_pH10_x', 'deg_Mg_50C_x', 'deg_50C_x'], axis=1)
sub1= sub1.rename(columns= {'reactivity_y':'reactivity','deg_Mg_pH10_y':'deg_Mg_pH10',
                     'deg_pH10_y':'deg_pH10', 'deg_Mg_50C_y':'deg_Mg_50C',
                     'deg_50C_y':'deg_50C'})
submission= sub1.fillna(0)
