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

geopandas==0.14.4
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0

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

0.4821241527019558

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy.linalg import norm

import os

## === cell 1
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag

## === cell 2
!cp -r /kaggle/input/train-baseline-bert-w-o-context/sentence-transformers /tmp/sentence-transformers
!pip install /tmp/sentence-transformers

## === cell 3
from sentence_transformers import SentenceTransformer

modelPath = "/kaggle/input/train-baseline-bert-w-o-context/bert-base/"

model = SentenceTransformer(modelPath)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
wnl = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

## === cell 6
def clean_text(corpus, remove_stop_words = True):
    '''
    Function to clean a given corpus - lower the words, strip of the spaces, remove stopwords and lemmatize the corpus
    Args:
        corpus: the text to be cleaned
        remove_stop_words: whether to remove stopwords
    Returns:
        filtered_sentence: cleaned corpus
    '''
    corpus = corpus.lower().strip()
    word_tokens = word_tokenize(corpus)
    if remove_stop_words:
        filtered_sentence = " ".join([wnl.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else wnl.lemmatize(i) for i,j in pos_tag(word_tokenize(corpus)) if i not in stop_words])
    else:
        filtered_sentence = " ".join([wnl.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else wnl.lemmatize(i) for i,j in pos_tag(word_tokenize(corpus))])
    return filtered_sentence

def cosine(a,b):
    '''
    Function to calculate cosine similarity of two vectors
    Args:
        a,b: vectors to calculate cosine between
    Returns:
        cosine similarity of the given vectors
    '''
    return np.dot(a,b)/(norm(a)*norm(b))

## === cell 7
test_df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv")

## === cell 8
test_df.head()

## === cell 9
test_df["anchor"] = test_df["anchor"].apply(lambda x: clean_text(x,False))
test_df["target"] = test_df["target"].apply(lambda x: clean_text(x,False))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
LookupError                               Traceback (most recent call last)
/tmp/ipykernel_11/4033337715.py in <cell line: 0>()
----> 1 test_df["anchor"] = test_df["anchor"].apply(lambda x: clean_text(x,False))
      2 test_df["target"] = test_df["target"].apply(lambda x: clean_text(x,False))

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in apply(self, func, convert_dtype, args, by_row, **kwargs)
   4922             args=args,
   4923             kwargs=kwargs,
-> 4924         ).apply()
   4925 
   4926     def _reindex_indexer(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
   1425 
   1426         # self.func is Callable
-> 1427         return self.apply_standard()
   1428 
   1429     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1505         #  Categorical (GH51645).
   1506         action = "ignore" if isinstance(obj.dtype, CategoricalDtype) else None
-> 1507         mapped = obj._map_values(
   1508             mapper=curried, na_action=action, convert=self.convert_dtype
   1509         )

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _map_values(self, mapper, na_action, convert)
    919             return arr.map(mapper, na_action=na_action)
    920 
--> 921         return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
    922 
    923     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/algorithms.py in map_array(arr, mapper, na_action, convert)
   1741     values = arr.astype(object, copy=False)
   1742     if na_action is None:
-> 1743         return lib.map_infer(values, mapper, convert=convert)
   1744     else:
   1745         return lib.map_infer_mask(

lib.pyx in pandas._libs.lib.map_infer()

/tmp/ipykernel_11/4033337715.py in <lambda>(x)
----> 1 test_df["anchor"] = test_df["anchor"].apply(lambda x: clean_text(x,False))
      2 test_df["target"] = test_df["target"].apply(lambda x: clean_text(x,False))

/tmp/ipykernel_11/3480022215.py in clean_text(corpus, remove_stop_words)
     13         filtered_sentence = " ".join([wnl.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else wnl.lemmatize(i) for i,j in pos_tag(word_tokenize(corpus)) if i not in stop_words])
     14     else:
---> 15         filtered_sentence = " ".join([wnl.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else wnl.lemmatize(i) for i,j in pos_tag(word_tokenize(corpus))])
     16     return filtered_sentence
     17 

/usr/local/lib/python3.11/dist-packages/nltk/tag/__init__.py in pos_tag(tokens, tagset, lang)
    166     :rtype: list(tuple(str, str))
    167     """
--> 168     tagger = _get_tagger(lang)
    169     return _pos_tag(tokens, tagset, tagger, lang)
    170 

/usr/local/lib/python3.11/dist-packages/nltk/tag/__init__.py in _get_tagger(lang)
    108         tagger = PerceptronTagger(lang=lang)
    109     else:
--> 110         tagger = PerceptronTagger()
    111     return tagger
    112 

/usr/local/lib/python3.11/dist-packages/nltk/tag/perceptron.py in __init__(self, load, lang, loc)
    178         )
    179         if load:
--> 180             self.load_from_json(lang, loc)
    181 
    182     def param_files(self, lang="eng"):

/usr/local/lib/python3.11/dist-packages/nltk/tag/perceptron.py in load_from_json(self, lang, loc)
    275         # Automatically find path to the tagger if location is not specified.
    276         if not loc:
--> 277             loc = find(f"taggers/averaged_perceptron_tagger_{lang}")
    278 
    279         def load_param(json_file):

/usr/local/lib/python3.11/dist-packages/nltk/data.py in find(resource_name, paths)
    577     sep = "*" * 70
    578     resource_not_found = f"\n{sep}\n{msg}\n{sep}\n"
--> 579     raise LookupError(resource_not_found)
    580 
    581 

LookupError: 
**********************************************************************
  Resource averaged_perceptron_tagger_eng not found.
  Please use the NLTK Downloader to obtain the resource:

  >>> import nltk
  >>> nltk.download('averaged_perceptron_tagger_eng')
  
  For more information see: https://www.nltk.org/data.html

  Attempted to load taggers/averaged_perceptron_tagger_eng

  Searched in:
    - '/root/nltk_data'
    - '/usr/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/lib/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/local/share/nltk_data'
    - '/usr/lib/nltk_data'
    - '/usr/local/lib/nltk_data'
**********************************************************************


## === cell 10
anchors = test_df["anchor"].to_list()
anchor_embed = model.encode(anchors,show_progress_bar=True, batch_size=256)
targets = test_df["target"].to_list()
target_embed = model.encode(targets,show_progress_bar=True, batch_size=256)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/184526644.py in <cell line: 0>()
      1 anchors = test_df["anchor"].to_list()
----> 2 anchor_embed = model.encode(anchors,show_progress_bar=True, batch_size=256)
      3 targets = test_df["target"].to_list()
      4 target_embed = model.encode(targets,show_progress_bar=True, batch_size=256)

NameError: name 'model' is not defined

## === cell 11
sims = [cosine(i[0],i[1]) for i in zip(anchor_embed,target_embed)]
max_val = max(sims)
min_val = min(sims)
sim_norm = (sims-min_val)/(max_val-min_val)
sim_norm = np.floor(sim_norm*4)/4

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2788545425.py in <cell line: 0>()
      1 # Calculating the cosine similarity between the anchir and target embeddings
----> 2 sims = [cosine(i[0],i[1]) for i in zip(anchor_embed,target_embed)]
      3 # Normalizing the similarity between the range 0 and 1 & categorizing them in 5 categories: 0, 0.25, 0.5, 0.75, 1
      4 max_val = max(sims)
      5 min_val = min(sims)

NameError: name 'anchor_embed' is not defined

## === cell 12
df = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/sample_submission.csv")

## === cell 13
df.head()

## === cell 14
submission_df = test_df.drop(columns = ["anchor","target","context"])

## === cell 15
submission_df["score"] = sims

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2673090411.py in <cell line: 0>()
----> 1 submission_df["score"] = sims

NameError: name 'sims' is not defined

## === cell 16
submission_df.to_csv("submission.csv",index=False)

## --- ERROR in outputing the csv:
Invalid submission: Submission must have columns ['id', 'score'], got Index(['id'], dtype='object')
