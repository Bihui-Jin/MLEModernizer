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
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

# 3. Installed packages

catboost==1.2.8
cudf-polars-cu12==25.6.0
geopandas==0.14.4
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
polars==1.25.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Target score

0.7995683527601928

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2
import os
import gc 
import ctypes
import random
import time
import string
import re
from tqdm import tqdm
import pickle

import pandas as pd, numpy as np
import polars as pl # For Feature Engineering

import matplotlib.pyplot as plt
import seaborn as sns 

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import words
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import VotingRegressor

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # For GPU T4x2

import warnings 
warnings.filterwarnings('ignore')

## === cell 4
class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = '/kaggle/input/aes2-cat/'
    LOAD_FEATURES_FROM = '/kaggle/input/aes2-cat/train_feats_1.csv'
    BASE_PATH = '/kaggle/input/learning-agency-lab-automated-essay-scoring-2/'

## === cell 6
Clean = True

def clean_memory():
    if Clean:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
        gc.collect()
        
clean_memory()  

## === cell 8
def seed_everything(): # To proudce simliar result in each run 
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    os.environ['PYTHONHASHSEED'] = str(CFG.SEED)
    
seed_everything()

## === cell 10
def seed_everything(): # To proudce simliar result in each run 
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    os.environ['PYTHONHASHSEED'] = str(CFG.SEED)
    
seed_everything()

## === cell 11
df_train = pd.read_csv(CFG.BASE_PATH + 'train.csv')
df_train = df_train.sort_values(by='essay_id')

print('Shape of Train: ', df_train.shape)
print(display(df_train.head()))

## === cell 12
df_test = pd.read_csv(CFG.BASE_PATH + 'test.csv')
df_test = df_test.sort_values(by='essay_id')

print('Shape of Test: ', df_test.shape)
print(display(df_test.head()))

## === cell 13

train = pl.from_pandas(df_train).with_columns(
    pl.col('full_text').str.split(by='\n\n').alias('paragraph'))
test = pl.from_pandas(df_test).with_columns(
    pl.col('full_text').str.split(by='\n\n').alias('paragraph')
)

schema_train = train.schema # MetaData
schema_test = test.schema # MetaData

## === cell 16
def removeHTML(x):
    html = re.compile(r'<.*?>')
    return html.sub(r'',x) # html -> ''

def dataPreprocessing(x):
    
    x = x.lower()
    x = removeHTML(x)
    
    x = re.sub("@\w+",'',x)
    
    x = re.sub("'\d+",'',x)
    x = re.sub("\d+",'',x)
    
    x = re.sub("http\w+",'',x)
    
    x = re.sub(r'\s+', " ", x)
    x = re.sub(r'[^\w\s.,;:"''?!]', '', x)
    x = re.sub("paragraph", "", x)
    x = re.sub(r'\.+', ".",x)
    x = re.sub(r'\,+', ",",x)
    x = x.strip()
    return x

## === cell 17
!pip install -q --no-index /kaggle/input/aes-2-misspelled-whl/pyspellchecker-0.8.1-py3-none-any.whl

## === cell 18
from spellchecker import SpellChecker

spell = SpellChecker()
def count_misspelled_words(text):
    misspelled_words = spell.unknown(text.split())
    return len(misspelled_words)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_10/2304523589.py in <cell line: 0>()
----> 1 from spellchecker import SpellChecker
      2 
      3 spell = SpellChecker()
      4 def count_misspelled_words(text):
      5     misspelled_words = spell.unknown(text.split())

ModuleNotFoundError: No module named 'spellchecker'

## === cell 19
paragraph_features = ['paragraph_len','paragraph_sentence_cnt','paragraph_word_cnt',"paragraph_comma_cnt",'paragraph_misspelled_cnt']

def Paragraph_Features(x):
    x = x.explode('paragraph') 
   
    print('Paragraph Preprocessing')
    x = x.with_columns(
         pl.col('paragraph').map_elements(dataPreprocessing)
    )
    print('Caculate the length of each paragraph')
    x = x.with_columns(
         pl.col('paragraph').map_elements(lambda x: len(x)).alias("paragraph_len")
    )
    x = x.with_columns(
         pl.col('paragraph').map_elements(lambda x: count_misspelled_words(x)).alias('paragraph_misspelled_cnt')
    )
    x = x.with_columns(
         pl.col('paragraph').map_elements(lambda x: x.count(',')).alias("paragraph_comma_cnt")
    )
    print('Caculate the number of sentences and words in each paragraph')
    x = x.with_columns(
         pl.col('paragraph').map_elements(lambda x: len(x.split('.'))).alias('paragraph_sentence_cnt'),
         pl.col('paragraph').map_elements(lambda x: len(x.split(' '))).alias('paragraph_word_cnt')
    )
    return x

def Paragraph_aggregation(x):    
    
    print('Aggregation')
    aggs = [
    *[pl.col('paragraph').filter(pl.col('paragraph_len') >= i).count().alias(f'paragraph_{i}_cnt') for i in [100,150, 200, 250, 300, 350, 400, 450, 500, 600, 800]],
    *[pl.col('paragraph').filter(pl.col('paragraph_len') <= i).count().alias(f'paragraph_{i}_cnt_v2') for i in [100,200]],
    *[pl.col('paragraph').filter((pl.col('paragraph_len') <= 300) & (pl.col('paragraph_len') > 100)).count().alias(f'short_paragraph_cnt')],    
    *[pl.col('paragraph').filter((pl.col('paragraph_len') <= 500) & (pl.col('paragraph_len') > 300)).count().alias(f'mid_paragraph_cnt')],
    *[pl.col('paragraph').filter((pl.col('paragraph_len') <= 700) & (pl.col('paragraph_len') > 500)).count().alias(f'long_paragraph_cnt')],
    *[pl.col('paragraph').filter(pl.col('paragraph_sentence_cnt') >= i).count().alias(f'paragraph_sentence_{i}_cnt') for i in [2,4,6,8,10]],
    *[pl.col('paragraph').filter((pl.col('paragraph_sentence_cnt') <= 4) & (pl.col('paragraph_sentence_cnt') > 2)).count().alias(f'short_paragraph_sentence_cnt')],
    *[pl.col('paragraph').filter((pl.col('paragraph_sentence_cnt') <= 8) & (pl.col('paragraph_sentence_cnt') > 4)).count().alias(f'mid_paragraph_sentence_cnt')],
    *[pl.col('paragraph').filter((pl.col('paragraph_sentence_cnt') <= 10) & (pl.col('paragraph_sentence_cnt') > 8)).count().alias(f'long_paragraph_sentence_cnt')],    
    *[pl.col('paragraph').filter(pl.col('paragraph_word_cnt') >= i).count().alias(f'paragraph_word_{i}_cnt') for i in [20,40,60,90,120]],
    *[pl.col('paragraph').filter((pl.col('paragraph_word_cnt') <= 40) & (pl.col('paragraph_word_cnt') > 20)).count().alias(f'short_paragraph_word_cnt')],
    *[pl.col('paragraph').filter((pl.col('paragraph_word_cnt') <= 90) & (pl.col('paragraph_word_cnt') > 40)).count().alias(f'mid_paragraph_word_cnt')],
    *[pl.col('paragraph').filter((pl.col('paragraph_word_cnt') <= 120) & (pl.col('paragraph_word_cnt') > 90)).count().alias(f'long_paragraph_word_cnt')],
    *[pl.col('paragraph').filter(pl.col('paragraph_comma_cnt') >= i).count().alias(f'paragraph_comma_{i}_cnt') for i in [1,2,3,4,5]],
    *[pl.col('paragraph').filter(pl.col('paragraph_misspelled_cnt') >= i).count().alias(f'paragraph_misspelled_{i}_cnt') for i in [4,8,12,16]],
    *[pl.col('paragraph').filter(pl.col('paragraph_misspelled_cnt') <= i).count().alias(f'paragraph_misspelled_{i}_cnt_v2') for i in [2,4]],
    *[pl.col('paragraph').filter((pl.col('paragraph_misspelled_cnt') <= 8) & (pl.col('paragraph_misspelled_cnt') > 4)).count().alias(f'short_paragraph_misspelled_cnt')],
    *[pl.col('paragraph').filter((pl.col('paragraph_misspelled_cnt') <= 12) & (pl.col('paragraph_misspelled_cnt') > 8)).count().alias(f'mid_paragraph_misspelled_cnt')],
    *[pl.col('paragraph').filter((pl.col('paragraph_misspelled_cnt') <= 16) & (pl.col('paragraph_misspelled_cnt') > 12)).count().alias(f'long_paragraph_misspelled_cnt')],
    *[pl.col('paragraph').count().alias('paragraph_cnt')], 
    *[pl.col(feat).max().alias(f'{feat}_max') for feat in paragraph_features],
    *[pl.col(feat).mean().alias(f'{feat}_mean') for feat in paragraph_features],
    *[pl.col(feat).min().alias(f'{feat}_min') for feat in paragraph_features],
    *[pl.col(feat).std().alias(f'{feat}_std') for feat in paragraph_features],
    *[pl.col(feat).sum().alias(f'{feat}_sum') for feat in paragraph_features],
    *[pl.col(feat).quantile(0.25).alias(f'{feat}_q1') for feat in paragraph_features],
    *[pl.col(feat).quantile(0.75).alias(f'{feat}_q3') for feat in paragraph_features],   
    ]
      
    df = x.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
    df = df.to_pandas() # polars -> pandas
    
    return df

## === cell 20
sentence_features = ['sentence_len','sentence_word_cnt']

def Sentence_Features(x):
    print('Preprocess full_text and use periods to segment sentences in the text')
    x = x.with_columns(
        pl.col('full_text').map_elements(lambda x: dataPreprocessing(x)).str.split(".").alias('sentence')
    )
    x = x.explode('sentence')
    
    print('Caculate the length of a sentence') 
    x = x.with_columns(
        pl.col('sentence').map_elements(lambda x: len(x)).alias('sentence_len'))
    x = x.filter(pl.col('sentence_len') > 3)
    x = x.with_columns(
       pl.col('sentence').map_elements(lambda x: len(x.replace(' ', ''))).alias('only_sentence_len'))
    print('Count the number of words in each sentence')
    x = x.with_columns(
        pl.col('sentence').map_elements(lambda x: len(x.split(" "))).alias("sentence_word_cnt"))
    return x

def Sentence_aggregation(x):    
    
    print('Aggregation')
    aggs = [
    *[pl.col('sentence').filter(pl.col('sentence_len') >= i).count().alias(f'sentence_{i}_cnt') for i in [40,60,70,80,100,120,140]],    
    *[pl.col('sentence').filter(pl.col('sentence_len') <= i).count().alias(f'sentence_{i}_cnt_v2') for i in [10,20,30]],
    *[pl.col('sentence').filter((pl.col('sentence_len') <= 70) & (pl.col('sentence_len') > 40)).count().alias(f'short_sentence_cnt')],    
    *[pl.col('sentence').filter((pl.col('sentence_len') <= 100) & (pl.col('sentence_len') > 70)).count().alias(f'mid_sentence_cnt')], 
    *[pl.col('sentence').filter((pl.col('sentence_len') <= 140) & (pl.col('sentence_len') > 100)).count().alias(f'long_sentence_cnt')],
    *[pl.col('sentence').filter(pl.col('only_sentence_len') >= i).count().alias(f'only_sentence_{i}_cnt') for i in [40,60,80,100,120]],    
    *[pl.col('sentence').filter((pl.col('only_sentence_len') <= 60) & (pl.col('only_sentence_len') > 40)).count().alias(f'short_only_sentence_cnt')],    
    *[pl.col('sentence').filter((pl.col('only_sentence_len') <= 100) & (pl.col('only_sentence_len') > 60)).count().alias(f'mid_only_sentence_cnt')], 
    *[pl.col('sentence').filter((pl.col('only_sentence_len') <= 120) & (pl.col('only_sentence_len') > 100)).count().alias(f'long_only_sentence_cnt')],
    *[pl.col('sentence').filter(pl.col('sentence_word_cnt') >= i).count().alias(f'sentence_word_{i}_cnt') for i in [10,15,20,25]], 
    *[pl.col('sentence').filter((pl.col('sentence_word_cnt') <= 15) & (pl.col('sentence_word_cnt') > 10)).count().alias(f'short_sentence_word_cnt')],    
    *[pl.col('sentence').filter((pl.col('sentence_word_cnt') <= 20) & (pl.col('sentence_word_cnt') > 15)).count().alias(f'mid_sentence_word_cnt')], 
    *[pl.col('sentence').filter((pl.col('sentence_word_cnt') <= 25) & (pl.col('sentence_word_cnt') > 20)).count().alias(f'long_sentence_word_cnt')],    
    *[pl.col('sentence').count().alias('sentence_cnt')],    
    *[pl.col(feat).max().alias(f'{feat}_max') for feat in sentence_features],
    *[pl.col(feat).mean().alias(f'{feat}_mean') for feat in sentence_features],
    *[pl.col(feat).min().alias(f'{feat}_min') for feat in sentence_features],
    *[pl.col(feat).std().alias(f'{feat}_std') for feat in sentence_features],
    *[pl.col(feat).sum().alias(f'{feat}_sum') for feat in sentence_features], 
    *[pl.col(feat).quantile(0.25).alias(f'{feat}_q1') for feat in sentence_features], 
    *[pl.col(feat).quantile(0.75).alias(f'{feat}_q3') for feat in sentence_features],     
    ]
    
    df = x.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
    df = df.with_columns(
     *[(pl.col(f'sentence_{i}_cnt')/pl.col('sentence_cnt')).alias(f'sentence_{i}_cnt_ratio') for i in [40,60,70,80,100,120,140]],    
    *[(pl.col('short_sentence_cnt')/pl.col('sentence_cnt')).alias(f'short_sentence_cnt_ratio')],
    *[(pl.col('mid_sentence_cnt')/pl.col('sentence_cnt')).alias(f'mid_sentence_cnt_ratio')],
    *[(pl.col('long_sentence_cnt')/pl.col('sentence_cnt')).alias(f'long_sentence_cnt_ratio')],    
    ).sort('essay_id')
    
    df = df.to_pandas() # polars -> pandas
    
    return df

## === cell 21
word_features = ['word_len',]

def Word_Features(x):
    print('Preprocess full_text and use spaces to seperate words fro the text')

    x = x.with_columns(
        pl.col('full_text').map_elements(lambda x: dataPreprocessing(x)).str.split(" ").alias('word')
    )
    x = x.explode('word')
    
    print('Caculate the length of a word') 
    x = x.with_columns(
        pl.col('word').map_elements(lambda x: len(x)).alias('word_len'))
    x = x.filter(pl.col('word_len')>0)

    return x

def Word_aggregation(x):    
    
    print('Aggregation')
    aggs = [
    *[pl.col('word').filter(pl.col('word_len') >= i).count().alias(f'word_{i}_cnt') for i in [3,4,5,6,7,8,10]],
    *[pl.col('word').filter(pl.col('word_len') <= i).count().alias(f'word_{i}_cnt_v2') for i in [1,2,3]],
    *[pl.col('word').filter((pl.col('word_len') <= 4) & (pl.col('word_len') > 2)).count().alias(f'short_word_cnt')],
    *[pl.col('word').filter((pl.col('word_len') <= 6) & (pl.col('word_len') > 4)).count().alias(f'mid_word_cnt')], 
    *[pl.col('word').filter((pl.col('word_len') <= 10) & (pl.col('word_len') > 6)).count().alias(f'long_word_cnt')],     
    *[pl.col('word').count().alias('word_cnt')],  
    *[pl.col(feat).max().alias(f'{feat}_max') for feat in word_features],    
    *[pl.col(feat).mean().alias(f'{feat}_mean') for feat in word_features],
    *[pl.col(feat).min().alias(f'{feat}_min') for feat in word_features],
    *[pl.col(feat).std().alias(f'{feat}_std') for feat in word_features],
    *[pl.col(feat).sum().alias(f'{feat}_sum') for feat in word_features],
    *[pl.col(feat).quantile(0.25).alias(f'{feat}_q1') for feat in word_features],
    *[pl.col(feat).quantile(0.75).alias(f'{feat}_q3') for feat in word_features],   
    ]
    
    df = x.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
    df = df.with_columns(
    *[(pl.col(f'word_{i}_cnt')/pl.col('word_cnt')).alias(f'word_{i}_cnt_ratio') for i in [3,4,5,6,7,8,10]], 
    *[(pl.col(f'word_{i}_cnt_v2')/pl.col('word_cnt')).alias(f'word_{i}_cnt_v2_ratio') for i in [1,2,3]], 
    *[(pl.col(f'word_{i}_cnt')/pl.col('word_2_cnt_v2')).alias(f'word_{i}_pre2_ratio') for i in [3,4,5,6,7,8,10]],    
    *[(pl.col(f'word_{i}_cnt')/pl.col('word_3_cnt_v2')).alias(f'word_{i}_pre3_ratio') for i in [3,4,5,6,7,8,10]],  
    *[(pl.col(f'short_word_cnt')/ pl.col(f'word_{i}_cnt_v2')).alias(f'short_word_ratio_{i}') for i in [1,2,3]], 
    *[(pl.col(f'mid_word_cnt')/ pl.col(f'word_{i}_cnt_v2')).alias(f'mid_word_ratio_{i}') for i in [1,2,3]], 
    *[(pl.col(f'long_word_cnt')/ pl.col(f'word_{i}_cnt_v2')).alias(f'long_word_ratio_{i}') for i in [1,2,3]],     
      
    ).sort("essay_id")
    
 
    df = df.to_pandas() # polars -> pandas
    
    return df

## === cell 22
vectorizer = TfidfVectorizer(
    tokenizer = lambda x: x,
    preprocessor = lambda x: x,
    token_pattern=None,
    strip_accents='unicode',
    analyzer= 'word',
    ngram_range = (1,4),
    min_df =0.05,
    max_df=0.95,
    sublinear_tf = True # Term Frequency Log Scaling 
    )

train_a = train.with_columns(
        pl.col('full_text').map_elements(lambda x: dataPreprocessing(x))
    )
train_tfid = vectorizer.fit_transform([i for i in train_a['full_text']])

              
dense_matrix = train_tfid.toarray()

df = pd.DataFrame(dense_matrix)
df.columns = [f'tfidf_{i}' for i in range(len(df.columns))]


df['essay_id'] = df_train['essay_id']

## === cell 23
vectorizer_cnt = CountVectorizer(
      tokenizer=lambda x: x,
      preprocessor=lambda x: x,
      token_pattern=None,
      strip_accents='unicode',
      analyzer = 'word',
      ngram_range=(2,4),
      min_df=0.10,
      max_df=0.85,)
    

train_b = train.with_columns(
        pl.col('full_text').map_elements(lambda x: dataPreprocessing(x))
     )
train_cnt = vectorizer_cnt.fit_transform([i for i in train_b['full_text']])

              
dense_matrix2 = train_cnt.toarray()

df2 = pd.DataFrame(dense_matrix2)
df2.columns = [f'cnt_{i}' for i in range(len(df2.columns))]


df2['essay_id'] = df_train['essay_id']

## === cell 24
if CFG.LOAD_FEATURES_FROM is None: 
    
    train_feats1 = Paragraph_Features(train)
    train_feats1 = Paragraph_aggregation(train_feats1)
    train_feats2 = Sentence_Features(train)
    train_feats2 = Sentence_aggregation(train_feats2)
    train_feats3 = Word_Features(train)
    train_feats3 = Word_aggregation(train_feats3)
    
    train_feats = train_feats1.merge(train_feats2, on='essay_id', how='left')
    train_feats = train_feats.merge(train_feats3, on='essay_id', how='left')
    train_feats = train_feats.merge(df, on='essay_id', how='left')
    train_feats = train_feats.merge(df2, on='essay_id', how='left')
    train_feats['score'] = df_train['score'].values
else:
    None

## === cell 25
if CFG.LOAD_FEATURES_FROM is None: 
    print('Save train_feats.csv')
    train_feats.to_csv(f'train_feats_{CFG.VER}.csv', index=False)
else: 
    print('Load train_feats.csv')
    train_feats = pd.read_csv(CFG.LOAD_FEATURES_FROM)

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_10/43728007.py in <cell line: 0>()
      4 else:
      5     print('Load train_feats.csv')
----> 6     train_feats = pd.read_csv(CFG.LOAD_FEATURES_FROM)

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/aes2-cat/train_feats_1.csv'

## === cell 26
display(train_feats.head())

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/1483965322.py in <cell line: 0>()
----> 1 display(train_feats.head())

NameError: name 'train_feats' is not defined

## === cell 28
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import cohen_kappa_score

## === cell 29
def quadratic_weighted_kappa(y_true, y_pred):
    y_true = y_true + a
    y_pred = (y_pred + a).clip(1, 6).round()
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return 'QWK', qwk, True
def qwk_obj(y_true, y_pred):
    labels = y_true + a
    preds = y_pred + a
    preds = preds.clip(1, 6)
    f = 1/2*np.sum((preds-labels)**2)
    g = 1/2*np.sum((preds-a)**2+b)
    df = preds - labels
    dg = preds - a
    grad = (df/g - f*dg/g**2)*len(labels)
    hess = np.ones(len(labels))
    return grad, hess
a = 2.948
b = 1.092

## === cell 30
from sklearn.model_selection import train_test_split
import optuna

import catboost 
from catboost import CatBoostRegressor, Pool
print('Catboost Version: ', catboost.__version__)

## === cell 33
'''
def cat_objective(trial):

    params = { 
          'verbose'      : 0,
          'random_state' : CFG.SEED, 
          'loss_function' : 'MultiClass', 
          'learning_rate' : trial.suggest_float('learning_rate', 0.001, 0.5), 
          'depth' : trial.suggest_int('depth', 5, 10),
    }

    train_x, valid_x, train_y, valid_y = train_test_split(train_feats[FEATURES], train_feats[TARGET], test_size=0.2, random_state=CFG.SEED)
    train_pool = Pool(
          data = train_x,
          label = train_y
    )    

    valid_pool = Pool(
          data = valid_x,
          label = valid_y
    )

    model  = CatBoostClassifier(**params)

    model.fit(train_pool,
          eval_set = valid_pool,  
           )
    oof = model.predict(valid_pool)
    cv = cohen_kappa_score(valid_y, oof, weights="quadratic")

    
    return cv '''

## === cell 34
'''
study = optuna.create_study(direction='minimize', study_name='Classification') 
study.optimize(cat_objective, n_trials=10, show_progress_bar=True)
'''

## === cell 36
categorical_columns = train_feats.select_dtypes(include=['object','category']).columns.tolist()
FEATURES = [col for col in train_feats.columns if col not in categorical_columns + ['score']]
TARGET = 'score'

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/1776365531.py in <cell line: 0>()
----> 1 categorical_columns = train_feats.select_dtypes(include=['object','category']).columns.tolist()
      2 FEATURES = [col for col in train_feats.columns if col not in categorical_columns + ['score']]
      3 TARGET = 'score'

NameError: name 'train_feats' is not defined

## === cell 37
def catboost():
    all_oof = [] 
    all_true = []
    
    
    skf = StratifiedKFold(n_splits=10, random_state=CFG.SEED, shuffle=True)
    for i, (train_index, valid_index) in enumerate(skf.split(train_feats, train_feats[TARGET])):

        print('#'*25)
        print(f'### Fold {i+1}')
        print(f'### train size {len(train_index)}, valid size {len(valid_index)}')
        print('#'*25)
        
        model = CatBoostRegressor(
        iterations=1000,
        learning_rate = 0.1,
        depth = 5,
        subsample=0.8,
        l2_leaf_reg = 1,
        task_type = 'CPU',
        objective = 'RMSE',
        eval_metric = 'RMSE',
        random_state = CFG.SEED,
        )

        train_pool = Pool(
          data = np.clip(train_feats.loc[train_index, FEATURES].fillna(0),
                      0, 10000),
          label = train_feats.loc[train_index, TARGET]
        )    

        valid_pool = Pool(
          data = np.clip(train_feats.loc[valid_index, FEATURES].fillna(0),
                      0,10000),
          label = train_feats.loc[valid_index, TARGET]
        )

        model.fit(train_pool, verbose=100,
              eval_set = valid_pool,
              early_stopping_rounds=75

           )

        pickle.dump(model, open(f'CAT_v{CFG.VER}_f{i}.pkl', 'wb'))

        oof = model.predict(valid_pool)
        all_oof.append(oof)
        all_true.append(train_feats.loc[valid_index, TARGET])

        del train_pool, valid_pool, oof, model
    clean_memory()

    all_oof = np.concatenate(all_oof)
    all_true = np.concatenate(all_true)

    oof = pd.DataFrame(all_oof.copy())
    oof['id'] = np.arange(len(oof))

    true = pd.DataFrame(all_true.copy())
    true['id'] = np.arange(len(true))

    cv = cohen_kappa_score(true[0], oof[0].clip(1,6).round(), weights="quadratic")
    print('CV Score for Low Catboost = ',cv)
    cm = confusion_matrix(true[0], oof[0].clip(1,6).round(), labels=[x for x in range(1,7)])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[x for x in range(1,7)])   
    disp.plot()
    plt.show()

## === cell 38
if CFG.LOAD_MODELS_FROM is None:
    print('Training CATBoost')
    catboost()
else: 
    None


## === cell 39
if CFG.LOAD_MODELS_FROM:
    model = pickle.load(open(f'{CFG.LOAD_MODELS_FROM}CAT_v{CFG.VER}_f0.pkl', 'rb'))
else: 
    model = pickle.load(open(f'CAT_v{CFG.VER}_f0.pkl', 'rb'))

df_importance = pd.DataFrame({
        'features_name': FEATURES,
        'importance': model.feature_importances_,
    })
df_importance = df_importance.sort_values(by='importance', ascending=False)

plt.figure(figsize=(12,6))
plt.bar(data=df_importance.head(30), x='features_name', height='importance', color='pink', edgecolor='black')
plt.title('Distribution of Feature Importance of Catboost')
plt.xticks(rotation=90)
plt.show()

## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_10/632611462.py in <cell line: 0>()
      1 if CFG.LOAD_MODELS_FROM:
----> 2     model = pickle.load(open(f'{CFG.LOAD_MODELS_FROM}CAT_v{CFG.VER}_f0.pkl', 'rb'))
      3 else:
      4     model = pickle.load(open(f'CAT_v{CFG.VER}_f0.pkl', 'rb'))
      5 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/aes2-cat/CAT_v1_f0.pkl'

## === cell 40
test_a = test.with_columns(
         pl.col('full_text').map_elements(lambda x: dataPreprocessing(x))
     )
test_tfid = vectorizer.transform([i for i in test_a['full_text']])
dense_matrix = test_tfid.toarray()
df3 = pd.DataFrame(dense_matrix)
tfid_columns = [ f'tfidf_{i}' for i in range(len(df3.columns))]
df3.columns = tfid_columns
df3['essay_id'] = df_test['essay_id']

## === cell 41
test_b = test.with_columns(
         pl.col('full_text').map_elements(lambda x: dataPreprocessing(x))
     )
test_cnt = vectorizer_cnt.transform([i for i in test_b['full_text']])
dense_matrix = test_cnt.toarray()
df4 = pd.DataFrame(dense_matrix)
cnt_columns = [ f'cnt_{i}' for i in range(len(df4.columns))]
df4.columns = cnt_columns
df4['essay_id'] = df_test['essay_id']

## === cell 42
%%time 
test_feats1 = Paragraph_Features(test)
test_feats1 = Paragraph_aggregation(test_feats1)
test_feats2 = Sentence_Features(test)
test_feats2 = Sentence_aggregation(test_feats2)
test_feats3 = Word_Features(test)
test_feats3 = Word_aggregation(test_feats3)

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
ComputeError                              Traceback (most recent call last)
<timed exec> in <module>

/tmp/ipykernel_10/984290812.py in Paragraph_Features(x)
     13          pl.col('paragraph').map_elements(lambda x: len(x)).alias("paragraph_len")
     14     )
---> 15     x = x.with_columns(
     16          pl.col('paragraph').map_elements(lambda x: count_misspelled_words(x)).alias('paragraph_misspelled_cnt')
     17     )

/usr/local/lib/python3.11/dist-packages/polars/dataframe/frame.py in with_columns(self, *exprs, **named_exprs)
   9803         └─────┴──────┴─────────────┘
   9804         """
-> 9805         return self.lazy().with_columns(*exprs, **named_exprs).collect(_eager=True)
   9806 
   9807     def with_columns_seq(

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
     86                 kwargs["engine"] = "old-streaming"
     87 
---> 88             return function(*args, **kwargs)
     89 
     90         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/lazyframe/frame.py in collect(self, type_coercion, _type_check, predicate_pushdown, projection_pushdown, simplify_expression, slice_pushdown, comm_subplan_elim, comm_subexpr_elim, cluster_with_columns, collapse_joins, no_optimization, engine, background, _check_order, _eager, **_kwargs)
   2186         # Only for testing purposes
   2187         callback = _kwargs.get("post_opt_callback", callback)
-> 2188         return wrap_df(ldf.collect(engine, callback))
   2189 
   2190     @overload

ComputeError: NameError: name 'count_misspelled_words' is not defined

## === cell 43
test_feats = test_feats1.merge(test_feats2, on='essay_id', how='left')
test_feats = test_feats.merge(test_feats3, on='essay_id', how='left')
test_feats = test_feats.merge(df3, on='essay_id', how='left')
test_feats = test_feats.merge(df4, on='essay_id', how='left')
print('Shape of test_feats:', test_feats.shape)
display(test_feats.head())

## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/463599271.py in <cell line: 0>()
----> 1 test_feats = test_feats1.merge(test_feats2, on='essay_id', how='left')
      2 test_feats = test_feats.merge(test_feats3, on='essay_id', how='left')
      3 test_feats = test_feats.merge(df3, on='essay_id', how='left')
      4 test_feats = test_feats.merge(df4, on='essay_id', how='left')
      5 print('Shape of test_feats:', test_feats.shape)

NameError: name 'test_feats1' is not defined

## === cell 44
preds = []
categorical_columns = test_feats.select_dtypes(include=['object','category']).columns.tolist()
FEATURES = [col for col in test_feats.columns if col not in categorical_columns]


for i in range(10):
    print(f'Fold {i+1}')
    if CFG.LOAD_MODELS_FROM:
        model = pickle.load(open(f'{CFG.LOAD_MODELS_FROM}CAT_v{CFG.VER}_f{i}.pkl', 'rb'))
    else: 
        model = pickle.load(open(f'CAT_v{CFG.VER}_f{i}.pkl', 'rb'))
        
        
    pred= model.predict(test_feats[FEATURES])

    
    preds.append(pred)
pred = np.mean(preds,axis=0)          
   

## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/2240989877.py in <cell line: 0>()
      1 preds = []
----> 2 categorical_columns = test_feats.select_dtypes(include=['object','category']).columns.tolist()
      3 FEATURES = [col for col in test_feats.columns if col not in categorical_columns]
      4 
      5 

NameError: name 'test_feats' is not defined

## === cell 45
sub = pd.DataFrame({'essay_id': df_test.essay_id.values})
sub[TARGET] = pred.clip(1,6).round() 
sub.to_csv('submission.csv',index=False)
print('Submission shape', sub.shape)
sub.head()

## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/2084156733.py in <cell line: 0>()
      1 sub = pd.DataFrame({'essay_id': df_test.essay_id.values})
----> 2 sub[TARGET] = pred.clip(1,6).round()
      3 sub.to_csv('submission.csv',index=False)
      4 print('Submission shape', sub.shape)
      5 sub.head()

NameError: name 'pred' is not defined
