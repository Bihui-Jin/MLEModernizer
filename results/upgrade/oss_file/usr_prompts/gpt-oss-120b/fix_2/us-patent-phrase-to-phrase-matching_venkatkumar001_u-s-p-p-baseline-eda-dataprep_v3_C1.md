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
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# 2. Python version

3.10

# 3. Installed packages

catboost==1.2.8
geopandas==0.14.4
lightgbm==4.6.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Target score

0.1518

# 6. Current score

0.34777

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
I will keep the overall pipeline but let CatBoost handle the original text columns as categorical features instead of converting them to integer codes and scaling them. This small change lets the model use proper categorical handling, which should raise the Pearson correlation from the current -0.0392 toward the target 0.1518. The rest of the code, including folding, training, and submission creation, remains unchanged.

```


## --- ERROR in cell 0, traceback:
  File "/tmp/ipykernel_11/3593632146.py", line 1
    I will keep the overall pipeline but let CatBoost handle the original text columns as categorical features instead of converting them to integer codes and scaling them. This small change lets the model use proper categorical handling, which should raise the Pearson correlation from the current -0.0392 toward the target 0.1518. The rest of the code, including folding, training, and submission creation, remains unchanged.
                                                                                                                                                                                                                                                                                                                                    ^
SyntaxError: invalid non-printable character U+202F


## === cell 1
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
import seaborn as sns
import datetime as dt
from sklearn import model_selection
from catboost import CatBoostRegressor



## === cell 2
start_time=dt.datetime.now()
print("started at",start_time)



## === cell 3
train = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/train.csv")
test = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/test.csv")
sample = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/sample_submission.csv")
print(f'train_shape: {train.shape},test_shape: {test.shape},sample_shape: {sample.shape}')
train.head()



## === cell 4
train.isnull().sum().plot()



## === cell 5
train.info()



## === cell 6
def create_folds(data,num_splits):
    data['kfold'] = -1
    data = data.sample(frac=1).reset_index(drop=True)
    num_bins = int(np.floor(1+np.log2(len(data))))
    data.loc[:,"bins"] = pd.cut(data["score"],bins=num_bins,labels=False)
    kf = model_selection.StratifiedKFold(n_splits= num_splits,shuffle=True,random_state=42)
    for f,(t_,v_) in enumerate(kf.split(X=data,y=data.bins.values)):
        data.loc[v_,'kfold']=f
    data = data.drop("bins",axis=1)
    return data



## === cell 7
df_t = create_folds(train,num_splits=5)
df_t.kfold.value_counts()



## === cell 8
df_t.to_csv("train_sfolds_5.csv",index=False)
print("successfully Skfold")



## === cell 9
df_t.head()



## === cell 10
train = pd.read_csv('./train_sfolds_5.csv')
test = test.copy()
useful_features = [c for c in train.columns if c not in ('id','score','kfold')]
test = test[useful_features]



## === cell 11
prediction = []
for fold in range (5):
    xtrain = train[train.kfold != fold].reset_index(drop=True)
    xvalid = train[train.kfold == fold].reset_index(drop=True)
    xtest = test.copy()
    
    ytrain = xtrain["score"]
    yvalid = xvalid["score"]
    
    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]
    
    cat_features = ['anchor','target','context']
    
    c_para = {
        'iterations':15000,
        'use_best_model':True,
        'early_stopping_rounds':300,
        'learning_rate':0.0011589,
        'border_count':32,
        'verbose':False,
        'random_state':228,
        'subsample': 0.95312,
        "max_depth": 3,
        "min_data_in_leaf":77,
        'l2_leaf_reg': 0.02247766515106271,
        'cat_features':cat_features
    }
    cat_boost = CatBoostRegressor(**c_para)
    cat_boost.fit(xtrain, ytrain, eval_set=[(xvalid, yvalid)], verbose=False)
    
    test_predict = cat_boost.predict(xtest)
    prediction.append(test_predict)
    print(f'complete fold:{fold}')



## === cell 12
final_predict = np.mean(np.column_stack(prediction),axis=1)
print(final_predict)
sample['score'] = final_predict
sample.to_csv("submission.csv",index=False)
print("Final submission written to submission.csv")



## === cell 13
sample.head(5)
```

## --- ERROR in cell 13, traceback:
  File "/tmp/ipykernel_11/3082945931.py", line 2
    ```
    ^
SyntaxError: invalid syntax
