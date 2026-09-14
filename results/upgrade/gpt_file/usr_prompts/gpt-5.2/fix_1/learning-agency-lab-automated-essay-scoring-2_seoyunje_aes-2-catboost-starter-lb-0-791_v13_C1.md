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

0.8003551363465263

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

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import VotingRegressor

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # For GPU T4x2

import warnings 
warnings.filterwarnings('ignore')

## === cell 4
class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = '/kaggle/input/aes2-catboost/'
    LOAD_FEATURES_FROM = '/kaggle/input/aes2-train-csv-and-model/train_feats_1.csv'
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

print('Shape of Train: ', df_train.shape)
print(display(df_train.head()))

## === cell 12
df_test = pd.read_csv(CFG.BASE_PATH + 'test.csv')

print('Shape of Test: ', df_test.shape)
print(display(df_test.head()))

## === cell 13

train = pl.read_csv(os.path.join(CFG.BASE_PATH,'train.csv')).with_columns(
    pl.col('full_text').str.split(by='\n\n').alias('paragraph'))
test = pl.read_csv(os.path.join(CFG.BASE_PATH,'test.csv')).with_columns(
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
    
    x = re.sub(r'\.+', ".",x)
    x = re.sub(r'\,+', ".",x)
    x = x.strip()
    return x

## === cell 17
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
    
    x = re.sub(r'\.+', ".",x)
    x = re.sub(r'\,+', ".",x)
    x = x.strip()
    return x

## === cell 18
paragraph_features = ['paragraph_len','paragraph_sentence_cnt','paragraph_word_cnt']

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
    
    print('Caculate the number of sentences and words in each paragraph')
    x = x.with_columns(
         pl.col('paragraph').map_elements(lambda x: len(x.split('.'))).alias('paragraph_sentence_cnt'),
         pl.col('paragraph').map_elements(lambda x: len(x.split(' '))).alias('paragraph_word_cnt')
    )
    return x

## === cell 19
def Paragraph_aggregation(x):    
    
    print('Aggregation')
    aggs = [
    *[pl.col('paragraph').filter(pl.col('paragraph_len') >= i).count().alias(f'paragraph_{i}_cnt') for i in [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]],
    
    *[pl.col(feat).max().alias(f'{feat}_max') for feat in paragraph_features],
    *[pl.col(feat).mean().alias(f'{feat}_mean') for feat in paragraph_features],
    *[pl.col(feat).min().alias(f'{feat}_min') for feat in paragraph_features],
    *[pl.col(feat).std().alias(f'{feat}_std') for feat in paragraph_features],
    *[pl.col(feat).sum().alias(f'{feat}_sum') for feat in paragraph_features],
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
    
    print('Count the number of words in each sentence')
    x = x.with_columns(
        pl.col('sentence').map_elements(lambda x: len(x.split(" "))).alias("sentence_word_cnt"))

    return x

## === cell 21
def Sentence_aggregation(x):    
    
    print('Aggregation')
    aggs = [
    *[pl.col('sentence').filter(pl.col('sentence_len') >= i).count().alias(f'sentence_{i}_cnt') for i in [5,10,15,25,30,40,50,60,70,80,90,100]],
    
    *[pl.col(feat).max().alias(f'{feat}_max') for feat in sentence_features],
    *[pl.col(feat).mean().alias(f'{feat}_mean') for feat in sentence_features],
    *[pl.col(feat).min().alias(f'{feat}_min') for feat in sentence_features],
    *[pl.col(feat).std().alias(f'{feat}_std') for feat in sentence_features],
    *[pl.col(feat).sum().alias(f'{feat}_sum') for feat in sentence_features],   
    ]
    
    df = x.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
    
    df = df.to_pandas() # polars -> pandas
    
    return df

## === cell 22
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
    

    return x

## === cell 23
def Word_aggregation(x):    
    
    print('Aggregation')
    aggs = [
    *[pl.col('word').filter(pl.col('word_len') >= i).count().alias(f'sentence_{i}_cnt') for i in [2.0,2.5,3.0,3.5,4.0,4.5,5.0,6.0]],
    
    *[pl.col(feat).max().alias(f'{feat}_max') for feat in word_features],
    *[pl.col(feat).mean().alias(f'{feat}_mean') for feat in word_features],
    *[pl.col(feat).min().alias(f'{feat}_min') for feat in word_features],
    *[pl.col(feat).std().alias(f'{feat}_std') for feat in word_features],
    *[pl.col(feat).sum().alias(f'{feat}_sum') for feat in word_features],   
    ]
    
    df = x.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
    
    df = df.to_pandas() # polars -> pandas
    
    return df

## === cell 24
vectorizer = TfidfVectorizer(
    tokenizer = lambda x: x,
    preprocessor = lambda x: x,
    token_pattern=None,
    strip_accents='unicode',
    analyzer= 'word',
    ngram_range = (1,3),
    min_df =0.05,
    max_df=0.95,
    sublinear_tf = True # Term Frequency Log Scaling 
    )

tmp = train.with_columns(pl.col('full_text').map_elements(lambda x: dataPreprocessing(x)))

train_tfid = vectorizer.fit_transform([i for i in tmp['full_text']])

              
dense_matrix = train_tfid.toarray()

df = pd.DataFrame(dense_matrix)
df.columns = [f'tfidf_{i}' for i in df.columns.tolist()]


df['essay_id'] = df_train['essay_id']

## === cell 25
def dataPreprocessing2(x):
    
    x = x.lower()
    x = removeHTML(x)
    
    x = re.sub("@\w+",'',x)
        
    x = re.sub("http\w+",'',x)
    
    x = re.sub(r'\s+', " ", x)
    
    x = x.strip()
    return x

## === cell 26
def Lexical_Features(x):
    
    print('Full text Preprocessing')
    x = x.with_columns(
         pl.col('full_text').map_elements(dataPreprocessing2)
    )
    
    print("Caculate number of punctuation")
    x = x.with_columns(
       pl.col('full_text').map_elements(lambda x: len([char for char in list(x) if char in string.punctuation])).alias('num_puncts_text')
    )
    
    print('Caculate number of unique words')
    x = x.with_columns(
       pl.col('full_text').map_elements(lambda x: len(set(word_tokenize(x)))).alias('unique_word_text')
    )

    
    return x

## === cell 27
lexical_features = ['unique_word_text','num_puncts_text']

def lexical_aggregation(x):    
    
    print('Aggregation')
    aggs = [
    *[pl.col('full_text').filter(pl.col('unique_word_text') >= i).count().alias(f'unique_word_{i}_cnt') for i in [50, 100,125,150,175,200]],
    *[pl.col('full_text').filter(pl.col('num_puncts_text') >= i).count().alias(f'num_puncts_{i}_cnt') for i in [20,30,35,40,45, 50,60]],    
    
    *[pl.col(feat).max().alias(f'{feat}_max') for feat in lexical_features],
    *[pl.col(feat).mean().alias(f'{feat}_mean') for feat in lexical_features],
    *[pl.col(feat).min().alias(f'{feat}_min') for feat in lexical_features],
    *[pl.col(feat).std().alias(f'{feat}_std') for feat in lexical_features],
    *[pl.col(feat).sum().alias(f'{feat}_sum') for feat in lexical_features],   
    ]
    
    df = x.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
    
    df = df.to_pandas() # polars -> pandas
    
    return df

## === cell 28
if CFG.LOAD_FEATURES_FROM is None: 
    print('Save train_feats.csv')
    train_feats.to_csv(f'train_feats_{CFG.VER}.csv', index=False)
else: 
    print('Load train_feats.csv')
    train_feats = pd.read_csv(CFG.LOAD_FEATURES_FROM)

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/43728007.py in <cell line: 0>()
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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/aes2-train-csv-and-model/train_feats_1.csv'

## === cell 29
display(train_feats.head())

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1483965322.py in <cell line: 0>()
----> 1 display(train_feats.head())

NameError: name 'train_feats' is not defined

## === cell 31
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import cohen_kappa_score

## === cell 32
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

## === cell 33
categorical_columns = train_feats.select_dtypes(include=['object','category']).columns.tolist()
FEATURES = [col for col in train_feats.columns if col not in categorical_columns + ['score']]
TARGET = 'score'

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1776365531.py in <cell line: 0>()
----> 1 categorical_columns = train_feats.select_dtypes(include=['object','category']).columns.tolist()
      2 FEATURES = [col for col in train_feats.columns if col not in categorical_columns + ['score']]
      3 TARGET = 'score'

NameError: name 'train_feats' is not defined

## === cell 34
from sklearn.model_selection import train_test_split
import optuna

import catboost 
from catboost import CatBoostRegressor, Pool
print('Catboost Version: ', catboost.__version__)

## === cell 37
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

## === cell 38
'''
study = optuna.create_study(direction='minimize', study_name='Classification') 
study.optimize(cat_objective, n_trials=10, show_progress_bar=True)
'''

## === cell 40
def catboost(): 
    all_oof = [] 
    all_true = []

    skf = StratifiedKFold(n_splits=5, random_state=CFG.SEED, shuffle=True)
    for i, (train_index, valid_index) in enumerate(skf.split(train_feats, train_feats[TARGET])):

        print('#'*25)
        print(f'### Fold {i+1}')
        print(f'### train size {len(train_index)}, valid size {len(valid_index)}')
        print('#'*25)
        
        qwk_obj.calc_ders_range = True 
        
        model = CatBoostRegressor(
        iterations=1000,
        learning_rate = 0.05,
        depth = 5,
        task_type = 'CPU',
        objective = 'RMSE',
        eval_metric = 'RMSE'
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
    print('CV Score for Catboost = ',cv)

## === cell 41
if CFG.LOAD_MODELS_FROM is None: 
    print('Training CatBoost')
    catboost() 
else: 
    None

## === cell 42
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

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/632611462.py in <cell line: 0>()
      1 if CFG.LOAD_MODELS_FROM:
----> 2     model = pickle.load(open(f'{CFG.LOAD_MODELS_FROM}CAT_v{CFG.VER}_f0.pkl', 'rb'))
      3 else:
      4     model = pickle.load(open(f'CAT_v{CFG.VER}_f0.pkl', 'rb'))
      5 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/aes2-catboost/CAT_v1_f0.pkl'

## === cell 43
test_tfid = vectorizer.transform([i for i in test['full_text']])
dense_matrix = test_tfid.toarray()
df2 = pd.DataFrame(dense_matrix)
tfid_columns = [ f'tfidf_{i}' for i in range(len(df2.columns))]
df2.columns = tfid_columns
df2['essay_id'] = df_test['essay_id']

## === cell 44
%%time 
test_feats1 = Paragraph_Features(test)
test_feats1 = Paragraph_aggregation(test_feats1)
test_feats2 = Sentence_Features(test)
test_feats2 = Sentence_aggregation(test_feats2)
test_feats3 = Word_Features(test)
test_feats3 = Word_aggregation(test_feats3)
test_feats4 = Lexical_Features(test)
test_feats4 = lexical_aggregation(test_feats4)

## === cell 45
test_feats = test_feats1.merge(test_feats2, on='essay_id', how='left')
test_feats = test_feats.merge(test_feats3, on='essay_id', how='left')
test_feats = test_feats.merge(test_feats4, on='essay_id', how='left')
test_feats = test_feats.merge(df2, on='essay_id', how='left')
print('Shape of test_feats:', test_feats.shape)
display(test_feats.head())

## === cell 46
preds = []
categorical_columns = test_feats.select_dtypes(include=['object','category']).columns.tolist()
FEATURES = [col for col in test_feats.columns if col not in categorical_columns]


for i in range(5):
    print(f'Fold {i+1}')
    if CFG.LOAD_MODELS_FROM:
        model = pickle.load(open(f'{CFG.LOAD_MODELS_FROM}CAT_v{CFG.VER}_f{i}.pkl', 'rb'))
    else: 
        model = pickle.load(open(f'CAT_v{CFG.VER}_f{i}.pkl', 'rb'))
        
    pred = model.predict(test_feats[FEATURES])
    preds.append(pred)
pred = np.mean(preds,axis=0)          
   

## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3233507134.py in <cell line: 0>()
      7     print(f'Fold {i+1}')
      8     if CFG.LOAD_MODELS_FROM:
----> 9         model = pickle.load(open(f'{CFG.LOAD_MODELS_FROM}CAT_v{CFG.VER}_f{i}.pkl', 'rb'))
     10     else:
     11         model = pickle.load(open(f'CAT_v{CFG.VER}_f{i}.pkl', 'rb'))

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/aes2-catboost/CAT_v1_f0.pkl'

## === cell 47
sub = pd.DataFrame({'essay_id': df_test.essay_id.values})
sub[TARGET] = pred.clip(1,6).round() 
sub.to_csv('submission.csv',index=False)
print('Submission shape', sub.shape)
sub.head()

## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2084156733.py in <cell line: 0>()
      1 sub = pd.DataFrame({'essay_id': df_test.essay_id.values})
----> 2 sub[TARGET] = pred.clip(1,6).round()
      3 sub.to_csv('submission.csv',index=False)
      4 print('Submission shape', sub.shape)
      5 sub.head()

NameError: name 'pred' is not defined
