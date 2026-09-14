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

0.4504

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/2339239741.py in <cell line: 0>()
      2 from scipy.spatial import distance
      3 
----> 4 nlp = spacy.load("en_core_web_lg")
      5 
      6 train_df['anchor_embedding'] = train_df.apply(lambda x: nlp(x['anchor']).vector, axis=1)

/usr/local/lib/python3.11/dist-packages/spacy/__init__.py in load(name, vocab, disable, enable, exclude, config)
     50     RETURNS (Language): The loaded nlp object.
     51     """
---> 52     return util.load_model(
     53         name,
     54         vocab=vocab,

/usr/local/lib/python3.11/dist-packages/spacy/util.py in load_model(name, vocab, disable, enable, exclude, config)
    482     if name in OLD_MODEL_SHORTCUTS:
    483         raise IOError(Errors.E941.format(name=name, full=OLD_MODEL_SHORTCUTS[name]))  # type: ignore[index]
--> 484     raise IOError(Errors.E050.format(name=name))
    485 
    486 

OSError: [E050] Can't find model 'en_core_web_lg'. It doesn't seem to be a Python package or a valid path to a data directory.

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


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'anchor_embedding'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3273336448.py in <cell line: 0>()
----> 1 train_df['cosine_distance'] = train_df.apply(lambda x: 1 - distance.cosine(x['anchor_embedding'], x['target_embedding']), axis=1)
      2 train_df['cosine_distance_rounded'] = train_df.apply(lambda x: round_values(x['cosine_distance']), axis=1)
      3 train_df['euclidean_distance'] = train_df.apply(lambda x: distance.euclidean(x['anchor_embedding'], x['target_embedding']), axis=1)
      4 train_df['euclidean_distance_rounded'] = train_df.apply(lambda x: round_values(x['euclidean_distance']), axis=1)
      5 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in apply(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)
  10372             kwargs=kwargs,
  10373         )
> 10374         return op.apply().__finalize__(self, method="apply")
  10375 
  10376     def map(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
    914             return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)
    915 
--> 916         return self.apply_standard()
    917 
    918     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1061     def apply_standard(self):
   1062         if self.engine == "python":
-> 1063             results, res_index = self.apply_series_generator()
   1064         else:
   1065             results, res_index = self.apply_series_numba()

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_series_generator(self)
   1079             for i, v in enumerate(series_gen):
   1080                 # ignore SettingWithCopy here in case the user mutates
-> 1081                 results[i] = self.func(v, *self.args, **self.kwargs)
   1082                 if isinstance(results[i], ABCSeries):
   1083                     # If we have a view on v, we need to make a copy because

/tmp/ipykernel_11/3273336448.py in <lambda>(x)
----> 1 train_df['cosine_distance'] = train_df.apply(lambda x: 1 - distance.cosine(x['anchor_embedding'], x['target_embedding']), axis=1)
      2 train_df['cosine_distance_rounded'] = train_df.apply(lambda x: round_values(x['cosine_distance']), axis=1)
      3 train_df['euclidean_distance'] = train_df.apply(lambda x: distance.euclidean(x['anchor_embedding'], x['target_embedding']), axis=1)
      4 train_df['euclidean_distance_rounded'] = train_df.apply(lambda x: round_values(x['euclidean_distance']), axis=1)
      5 

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in __getitem__(self, key)
   1119 
   1120         elif key_is_scalar:
-> 1121             return self._get_value(key)
   1122 
   1123         # Convert generator to list before going through hashable part

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _get_value(self, label, takeable)
   1235 
   1236         # Similar to Index.get_value, but we do not fall back to positional
-> 1237         loc = self.index.get_loc(label)
   1238 
   1239         if is_integer(loc):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'anchor_embedding'

## === cell 7
submission_df = pd.DataFrame()
submission_df['id'] = test_id
submission_df['score'] = test_df['cosine_distance']


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'cosine_distance'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2346789322.py in <cell line: 0>()
      1 submission_df = pd.DataFrame()
      2 submission_df['id'] = test_id
----> 3 submission_df['score'] = test_df['cosine_distance']

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'cosine_distance'

## === cell 8
submission_df.to_csv('submission.csv', index=False, header=True)


## --- ERROR in outputing the csv:
Invalid submission: Submission must have columns ['id', 'score'], got Index(['id'], dtype='object')
