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
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.6

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Target score

0.986164371770121

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import pandas as pd
import numpy as np
import os

DENSITY_COEFF = 0.1
assert DENSITY_COEFF >= 0.0 and DENSITY_COEFF <= 1.0

OVER_CORR_CUTOFF = 0.98
assert OVER_CORR_CUTOFF >= 0.0 and OVER_CORR_CUTOFF <= 1.0

INPUT_DIR = '../input/private-toxic-comment-sumbmissions/'

def load_submissions():
    files = os.listdir(INPUT_DIR)
    csv_files = []
    for f in files:
        if f.endswith(".csv"):
            csv_files.append(f)
    frames = {f:pd.read_csv(INPUT_DIR+f).sort_values('id') for f in csv_files}
    return frames


def get_corr_mat(col,frames):
    c = pd.DataFrame()
    for name,df in frames.items():
        c[name] = df[col]
    cor = c.corr()
    for name in cor.columns:
        cor.set_value(name,name,0.0)
    return cor


def highest_corr(mat,frames):
    n_cor = np.array(mat.values)
    corr = np.max(n_cor)
    idx = np.unravel_index(np.argmax(n_cor, axis=None), n_cor.shape)
    f1 = mat.columns[idx[0]]
    f2 = mat.columns[idx[1]]
    return corr,f1,f2


def get_merge_weights(m1,m2,densities):
    d1 = densities[m1]
    d2 = densities[m2]
    d_tot = d1 + d2
    weights1 = 0.5*DENSITY_COEFF + (d1/d_tot)*(1-DENSITY_COEFF)
    weights2 = 0.5*DENSITY_COEFF + (d2/d_tot)*(1-DENSITY_COEFF)
    return weights1, weights2


def ensemble_col(col,frames,densities):
    if len(frames) == 1:
        _, fr = frames.popitem()
        return fr[col]

    mat = get_corr_mat(col,frames)
    corr,merge1,merge2 = highest_corr(mat,frames)
    new_col_name = merge1 + '_' + merge2

    w1,w2 = get_merge_weights(merge1,merge2,densities)
    new_df = pd.DataFrame()
    new_df[col] = (frames[merge1][col]*w1) + (frames[merge2][col]*w2)
    del frames[merge1]
    del frames[merge2]
    frames[new_col_name] = new_df

    if corr >= OVER_CORR_CUTOFF:
        print('\t',merge1,merge2,'  (OVER CORR)')
        densities[new_col_name] = max(densities[merge1],densities[merge2])
    else:
        print('\t',merge1,merge2)
        densities[new_col_name] = densities[merge1] + densities[merge2]

    del densities[merge1]
    del densities[merge2]
    return ensemble_col(col,frames,densities)


ens_submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv').sort_values('id')

for col in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
    frames = load_submissions()
    print('\n\n',col)
    densities = {k:1.0 for k in frames.keys()}
    ens_submission[col] = ensemble_col(col,frames,densities)

print(ens_submission)    
ens_submission.to_csv('lazy_ensemble_submission.csv', index=False)


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2174047112.py in <cell line: 0>()
     87 
     88 for col in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
---> 89     frames = load_submissions()
     90     print('\n\n',col)
     91     densities = {k:1.0 for k in frames.keys()}

/tmp/ipykernel_11/2174047112.py in load_submissions()
     17 
     18 def load_submissions():
---> 19     files = os.listdir(INPUT_DIR)
     20     csv_files = []
     21     for f in files:

FileNotFoundError: [Errno 2] No such file or directory: '../input/private-toxic-comment-sumbmissions/'
