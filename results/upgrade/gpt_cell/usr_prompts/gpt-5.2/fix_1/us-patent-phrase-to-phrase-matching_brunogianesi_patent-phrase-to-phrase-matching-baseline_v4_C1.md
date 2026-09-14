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

fuzzywuzzy==0.18.0
geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scipy==1.15.3
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
ydata-profiling==4.17.0

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import pandas as pd

from pandas_profiling import ProfileReport


## === cell 1
train_df = pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv')
train_df['score'] = train_df['score'].astype(float)
test_df = pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv')
test_id = test_df['id']


## === cell 2
profile = ProfileReport(train_df)
profile


## === cell 3
def baseline_result(df):
  def even_odd(row):
    if row.number % 2 == 0:
      return 0.50
    else:
      return 0.25
  df['number'] = pd.Series(range(0,df.shape[0]))
  df['pred']  = df.apply(lambda x: even_odd(x), axis=1)
  correlation = df['pred'].corr(df['score'], method='pearson')
  return correlation
  
train_corr = baseline_result(train_df)

print(f'Train pearson correlation {train_corr:.4f}')


## === cell 4
from fuzzywuzzy import fuzz

def round_values(x):
    if x >= 0 and x < 0.125:
        return 0
    if x >= 0.125 and x < 0.375:
        return 0.25
    if x >= 0.375 and x < 0.625:
        return 0.5
    if x >= 0.625 and x < 0.875:
        return 0.75
    if x >= 0.875 and x <= 1:
        return 1.0

train_df['fuzzy'] = train_df.apply(lambda x: fuzz.partial_ratio(x['anchor'], x['target'])/100, axis=1)
train_df['fuzzy_rounded'] = train_df.apply(lambda x: round_values(x['fuzzy']), axis=1)

train_corr = train_df['fuzzy'].corr(train_df['score'], method='pearson')
print(f'Train pearson correlation {train_corr:.4f}')

train_corr = train_df['fuzzy_rounded'].corr(train_df['score'], method='pearson')
print(f'Train pearson correlation for rounded similarity {train_corr:.4f}')


## === cell 5
import spacy
from scipy.spatial import distance

nlp = spacy.load("en_core_web_lg")

train_df['anchor_embedding'] = train_df.apply(lambda x: nlp(x['anchor']).vector, axis=1)
train_df['target_embedding'] = train_df.apply(lambda x: nlp(x['target']).vector, axis=1)

test_df['anchor_embedding'] = test_df.apply(lambda x: nlp(x['anchor']).vector, axis=1)
test_df['target_embedding'] = test_df.apply(lambda x: nlp(x['target']).vector, axis=1)

train_df['similarity'] = train_df.apply(lambda x: distance.cosine(x['anchor_embedding'], x['target_embedding']), axis=1)

test_df['similarity'] = test_df.apply(lambda x: distance.cosine(x['anchor_embedding'], x['target_embedding']), axis=1)


## --- ERROR in cell 5, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mOSError[0m                                   Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2339239741.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0;32mfrom[0m [0mscipy[0m[0;34m.[0m[0mspatial[0m [0;32mimport[0m [0mdistance[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;34m[0m[0m
[0;32m----> 4[0;31m [0mnlp[0m [0;34m=[0m [0mspacy[0m[0;34m.[0m[0mload[0m[0;34m([0m[0;34m"en_core_web_lg"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0;34m[0m[0m
[1;32m      6[0m [0mtrain_df[0m[0;34m[[0m[0;34m'anchor_embedding'[0m[0;34m][0m [0;34m=[0m [0mtrain_df[0m[0;34m.[0m[0mapply[0m[0;34m([0m[0;32mlambda[0m [0mx[0m[0;34m:[0m [0mnlp[0m[0;34m([0m[0mx[0m[0;34m[[0m[0;34m'anchor'[0m[0;34m][0m[0;34m)[0m[0;34m.[0m[0mvector[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/spacy/__init__.py[0m in [0;36mload[0;34m(name, vocab, disable, enable, exclude, config)[0m
[1;32m     50[0m     [0mRETURNS[0m [0;34m([0m[0mLanguage[0m[0;34m)[0m[0;34m:[0m [0mThe[0m [0mloaded[0m [0mnlp[0m [0mobject[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m     51[0m     """
[0;32m---> 52[0;31m     return util.load_model(
[0m[1;32m     53[0m         [0mname[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     54[0m         [0mvocab[0m[0;34m=[0m[0mvocab[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/spacy/util.py[0m in [0;36mload_model[0;34m(name, vocab, disable, enable, exclude, config)[0m
[1;32m    482[0m     [0;32mif[0m [0mname[0m [0;32min[0m [0mOLD_MODEL_SHORTCUTS[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    483[0m         [0;32mraise[0m [0mIOError[0m[0;34m([0m[0mErrors[0m[0;34m.[0m[0mE941[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0mname[0m[0;34m=[0m[0mname[0m[0;34m,[0m [0mfull[0m[0;34m=[0m[0mOLD_MODEL_SHORTCUTS[0m[0;34m[[0m[0mname[0m[0;34m][0m[0;34m)[0m[0;34m)[0m  [0;31m# type: ignore[index][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 484[0;31m     [0;32mraise[0m [0mIOError[0m[0;34m([0m[0mErrors[0m[0;34m.[0m[0mE050[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0mname[0m[0;34m=[0m[0mname[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    485[0m [0;34m[0m[0m
[1;32m    486[0m [0;34m[0m[0m

[0;31mOSError[0m: [E050] Can't find model 'en_core_web_lg'. It doesn't seem to be a Python package or a valid path to a data directory.

## === cell 6
train_df['cosine_distance'] = train_df.apply(lambda x: 1 - distance.cosine(x['anchor_embedding'], x['target_embedding']), axis=1)
train_df['cosine_distance_rounded'] = train_df.apply(lambda x: round_values(x['cosine_distance']), axis=1)
train_df['euclidean_distance'] = train_df.apply(lambda x: distance.euclidean(x['anchor_embedding'], x['target_embedding']), axis=1)
train_df['euclidean_distance_rounded'] = train_df.apply(lambda x: round_values(x['euclidean_distance']), axis=1)

train_corr = train_df['cosine_distance'].corr(train_df['score'], method='pearson')
print(f'Train pearson correlation for cosine distance {train_corr:.4f}')

train_corr = train_df['cosine_distance_rounded'].corr(train_df['score'], method='pearson')
print(f'Train pearson correlation for cosine distance for rounded values {train_corr:.4f}')

train_corr = train_df['euclidean_distance'].corr(train_df['score'], method='pearson')
print(f'Train pearson correlation for euclidean distance {train_corr:.4f}')

train_corr = train_df['euclidean_distance_rounded'].corr(train_df['score'], method='pearson')
print(f'Train pearson correlation for euclidean distance for rounded values {train_corr:.4f}')


test_df['cosine_distance'] = test_df.apply(lambda x: 1 - distance.cosine(x['anchor_embedding'], x['target_embedding']), axis=1)
