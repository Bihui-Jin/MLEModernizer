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

0.4194

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import sys
import pandas as pd
import spacy
import time


## === cell 1
TRAIN_FILE_PATH = '../input/us-patent-phrase-to-phrase-matching/train.csv'
TEST_FILE_PATH = '../input/us-patent-phrase-to-phrase-matching/test.csv'
SAMPLE_SUBMISSION_PATH = '../input/us-patent-phrase-to-phrase-matching/sample_submission.csv'


## === cell 2
class config:
    PRINT_EVERY_N_WORD = 100
    BAR_LEN = 50


## === cell 3
train_df = pd.read_csv(TRAIN_FILE_PATH)
test_df = pd.read_csv(TEST_FILE_PATH)
submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)

print('train_df shape:', train_df.shape)
print('test_df shape:', test_df.shape)
print('submission_df shape:', submission_df.shape)


## === cell 4
similarity_score = []
n_words = train_df.shape[0]
start = time.time()
nlp = spacy.load('en_core_web_lg')

for i, row in train_df.iterrows():
    token1 = nlp(row.anchor)
    token2 = nlp(row.target)
    similarity_score.append(token1.similarity(token2))
    
    if ((i+1)%config.PRINT_EVERY_N_WORD == 0) | (i+1 == n_words):
        end = time.time()
        time_elapsed = end - start
        if i+1 == n_words:
            bar = '[' + '='*int((i+1)*config.BAR_LEN/n_words) + '.'*(config.BAR_LEN - int((i+1)*config.BAR_LEN/n_words) - 1) + ']'
        else:
            bar = '[' + '='*int((i+1)*config.BAR_LEN/n_words) + '>' + '.'*(config.BAR_LEN - int((i)*config.BAR_LEN/n_words) - 1) + ']'
        perc = (i+1)*100/n_words
        sys.stdout.write('\r')
        sys.stdout.write("%i/%i words completed %s %d%% %.1fs %.1fms/word" % (i+1, n_words, bar, perc, time_elapsed, time_elapsed*1000/(i+1)))
        sys.stdout.flush()

train_df['similarity_score'] = similarity_score


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/4181436809.py in <cell line: 0>()
      2 n_words = train_df.shape[0]
      3 start = time.time()
----> 4 nlp = spacy.load('en_core_web_lg')
      5 
      6 for i, row in train_df.iterrows():

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

## === cell 5
'''
0.000 - 0.125 -> 0.00
0.125 - 0.375 -> 0.25
0.375 - 0.625 -> 0.50
0.625 - 0.875 -> 0.75
0.875 - 1.000 -> 1.00
'''

mapping = {0.00: [0.000, 0.125],
           0.25: [0.125, 0.375],
           0.50: [0.375, 0.625],
           0.75: [0.625, 0.875],
           1.00: [0.875, 1.000]}

for key in mapping.keys():
    train_df['similarity_score'] = train_df['similarity_score'].mask((train_df['similarity_score'] >= mapping[key][0]) & (train_df['similarity_score'] < mapping[key][1]), key)


## --- ERROR in cell 5, traceback:
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

KeyError: 'similarity_score'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/201168253.py in <cell line: 0>()
     15 
     16 for key in mapping.keys():
---> 17     train_df['similarity_score'] = train_df['similarity_score'].mask((train_df['similarity_score'] >= mapping[key][0]) & (train_df['similarity_score'] < mapping[key][1]), key)

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

KeyError: 'similarity_score'

## === cell 6
from scipy.stats import pearsonr
corr, _ = pearsonr(train_df.score, train_df.similarity_score)
print('Training Pearson Correlation: %0.3f' % corr)


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2839801563.py in <cell line: 0>()
      1 from scipy.stats import pearsonr
----> 2 corr, _ = pearsonr(train_df.score, train_df.similarity_score)
      3 print('Training Pearson Correlation: %0.3f' % corr)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'similarity_score'

## === cell 7
similarity_score = []
n_words = test_df.shape[0]
start = time.time()
nlp = spacy.load('en_core_web_lg')

for i, row in test_df.iterrows():
    token1 = nlp(row.anchor)
    token2 = nlp(row.target)
    similarity_score.append(token1.similarity(token2))
    
    if ((i+1)%config.PRINT_EVERY_N_WORD == 0) | (i+1 == n_words):
        end = time.time()
        time_elapsed = end - start
        if i+1 == n_words:
            bar = '[' + '='*int((i+1)*config.BAR_LEN/n_words) + '.'*(config.BAR_LEN - int((i+1)*config.BAR_LEN/n_words) - 1) + ']'
        else:
            bar = '[' + '='*int((i+1)*config.BAR_LEN/n_words) + '>' + '.'*(config.BAR_LEN - int((i)*config.BAR_LEN/n_words) - 1) + ']'
        perc = (i+1)*100/n_words
        sys.stdout.write('\r')
        sys.stdout.write("%i/%i words completed %s %d%% %.1fs %.1fms/word" % (i+1, n_words, bar, perc, time_elapsed, time_elapsed*1000/(i+1)))
        sys.stdout.flush()

test_df['score'] = similarity_score


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/540326406.py in <cell line: 0>()
      2 n_words = test_df.shape[0]
      3 start = time.time()
----> 4 nlp = spacy.load('en_core_web_lg')
      5 
      6 for i, row in test_df.iterrows():

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

## === cell 8
for key in mapping.keys():
    test_df['score'] = test_df['score'].mask((test_df['score'] >= mapping[key][0]) & (test_df['score'] < mapping[key][1]), key)


## --- ERROR in cell 8, traceback:
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

KeyError: 'score'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2005454014.py in <cell line: 0>()
      1 for key in mapping.keys():
----> 2     test_df['score'] = test_df['score'].mask((test_df['score'] >= mapping[key][0]) & (test_df['score'] < mapping[key][1]), key)

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

KeyError: 'score'

## === cell 9
submission_df = test_df[['id', 'score']]
submission_df.to_csv('submission.csv', index = False)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2156943694.py in <cell line: 0>()
----> 1 submission_df = test_df[['id', 'score']]
      2 submission_df.to_csv('submission.csv', index = False)

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

KeyError: "['score'] not in index"
