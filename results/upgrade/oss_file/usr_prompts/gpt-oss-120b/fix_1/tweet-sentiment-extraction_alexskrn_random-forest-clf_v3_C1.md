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
Predict the word or phrase from tweets that exemplifies the labelled sentiment.

## Metric
Word-level Jaccard score.

## Submission Format
For each ID in the test set, you must predict the string that best supports the sentiment for the tweet in question. Note that the selected text _needs_ to be **quoted** and **complete** (include punctuation, etc. - the above code splits ONLY on whitespace) to work correctly. The file should contain a header and have the following format:
```
textID,selected_text
2,"very good"
5,"I don't care"
6,"bad"
8,"it was, yes"
etc.
```

## Dataset
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

- `textID` - unique ID for each piece of text
- `text` - the text of the tweet
- `sentiment` - the general sentiment of the tweet
- `selected_text` - [train only] the text that supports the tweet's sentiment

# 2. Python version

3.8

# 3. Installed packages

gensim==4.4.0
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        input/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        working/
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
```

-> data/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> data/tweet-sentiment-extraction/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/tweet-sentiment-extraction/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/tweet-sentiment-extraction/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> input/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> (stopped after 10 files for performance)

# 5. Target score

0.47893

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
from time import time

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

import gensim

import warnings
warnings.simplefilter(action='ignore')


## === cell 1
DATA_DIR = '../input/tweet-sentiment-extraction'
TRAIN_DATA_FILE = 'train.csv'
TEST_DATA_FILE = 'test.csv'
SUBMISSION_FILE = 'submission.csv'

RANDOM_STATE = 0


## === cell 2
train_data = pd.read_csv(os.path.join(DATA_DIR, TRAIN_DATA_FILE)).fillna('')
test_data = pd.read_csv(os.path.join(DATA_DIR, TEST_DATA_FILE)).fillna('')


## === cell 3
train_data = train_data[['textID', 'text', 'sentiment', 'selected_text']]
train_data[17:22]


## === cell 4
test_data.head()


## === cell 5
starts = []
ends = []
for text, selected_text in zip(train_data['text'], train_data['selected_text']):
  start = text.find(selected_text)
  starts.append(start)
  ends.append(start + len(selected_text))

train_data['start_idx'] = starts
train_data['end_idx'] = ends

train_data.head(3)


## === cell 6
DIM = 200  # vector size
print('Creating corpus for FastText model...')
t0 = time()
corpus = [text.lower().split() for text in train_data['text']]
corpus.extend([text.lower().split() for text in test_data['text']])
corpus.extend(train_data['sentiment'].unique().tolist())
print(f'Done in {time() - t0} seconds')

print('Building FastText model from corpus...')
t0 = time()
fast_text = gensim.models.FastText(corpus, size=DIM, min_count=1, min_n=1)
print(f'Done in {time() - t0} seconds')


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/507901250.py in <cell line: 0>()
      9 print('Building FastText model from corpus...')
     10 t0 = time()
---> 11 fast_text = gensim.models.FastText(corpus, size=DIM, min_count=1, min_n=1)
     12 print(f'Done in {time() - t0} seconds')

TypeError: FastText.__init__() got an unexpected keyword argument 'size'

## === cell 7
def get_embedding(text, model, dim):
    """Return FastText embeddings."""
    from collections import Counter
    text = text.lower().split()
    
    words = Counter(text)
    total = len(text)
    vectors = np.zeros((len(words), dim))
    
    for i,word in enumerate(words):
        try:
            v = model[word]
            vectors[i] = v*(words[word]/total)
        except (KeyError, ValueError):
            raise
    
    if vectors.any():
        vector = np.average(vectors, axis=0)
    else:
        vector = np.zeros((dim))
    
    return vector


## === cell 8
print('Create embeddings for \'text\' ...')
X_text_ft = np.zeros((train_data.shape[0], DIM))

t0 = time()
for i, text in enumerate(train_data['text'].values):
    X_text_ft[i] = get_embedding(text, fast_text, DIM)
print(f'Done in {time() - t0} seconds')

X_sentiment_ft = np.zeros((train_data.shape[0], DIM))

print('... and \'sentiment\' columns in train data')
t0 = time()
for i, text in enumerate(train_data['sentiment'].values):
    X_sentiment_ft[i] = get_embedding(text, fast_text, DIM)
print(f'Done in {time() - t0} seconds')

print(X_text_ft.shape, X_sentiment_ft.shape)

print('Concatenated text and sentiment vectors:')
train_data_ft = np.concatenate([X_text_ft, X_sentiment_ft], axis=1)
print(train_data_ft.shape)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2413325064.py in <cell line: 0>()
      4 t0 = time()
      5 for i, text in enumerate(train_data['text'].values):
----> 6     X_text_ft[i] = get_embedding(text, fast_text, DIM)
      7 print(f'Done in {time() - t0} seconds')
      8 

NameError: name 'fast_text' is not defined

## === cell 9
print('Create embeddings for \'text\' ...')
text_ft = np.zeros((test_data.shape[0], DIM))

t0 = time()
for i, text in enumerate(test_data['text'].values):
    text_ft[i] = get_embedding(text, fast_text, DIM)
print(f'Done in {time() - t0} seconds')

print('... and \'sentiment\' columns in test data')
sentiment_ft = np.zeros((test_data.shape[0], DIM))

t0 = time()
for i, text in enumerate(test_data['sentiment'].values):
    sentiment_ft[i] = get_embedding(text, fast_text, DIM)
print(f'Done in {time() - t0} seconds')

print(text_ft.shape, sentiment_ft.shape)

print('Concatenated text and sentiment vectors:')
kaggle_test_ft = np.concatenate([text_ft, sentiment_ft], axis=1)
print(kaggle_test_ft.shape)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/333500026.py in <cell line: 0>()
      4 t0 = time()
      5 for i, text in enumerate(test_data['text'].values):
----> 6     text_ft[i] = get_embedding(text, fast_text, DIM)
      7 print(f'Done in {time() - t0} seconds')
      8 

NameError: name 'fast_text' is not defined

## === cell 10
def create_model_starts():
    clf = RandomForestClassifier(
        max_depth=20,
        min_samples_leaf=2,
        min_samples_split=2,
        n_estimators=50,
        random_state=RANDOM_STATE
        )
    return clf

def create_model_ends():
    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=20,
        min_samples_leaf=17,
        min_samples_split=2,
        random_state=RANDOM_STATE
        )
    return clf


## === cell 12
def jaccard(top_selected):
    str1, str2 = top_selected
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    if (len(a) == 0) & (len(b) == 0):
        return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


## === cell 14
print('Training models on all data...')
t0 = time()
model_starts = create_model_starts()
model_starts.fit(train_data_ft, train_data['start_idx'])

model_ends = create_model_ends()
model_ends.fit(train_data_ft, train_data['end_idx'])

print(f'Done in {time() - t0} seconds')


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/531533174.py in <cell line: 0>()
      3 t0 = time()
      4 model_starts = create_model_starts()
----> 5 model_starts.fit(train_data_ft, train_data['start_idx'])
      6 
      7 model_ends = create_model_ends()

NameError: name 'train_data_ft' is not defined

## === cell 15
temp_df = pd.DataFrame()
temp_df['pred_starts'] = model_starts.predict(kaggle_test_ft)  # predict starts
temp_df['pred_ends'] = model_ends.predict(kaggle_test_ft)      # predict ends
    
temp_df['text'] = test_data['text']
print(temp_df.head())
temp_df['selected_text'] = temp_df[[
                          'text', 
                          'pred_starts', 
                          'pred_ends']
                                ].apply(lambda x: x[0][x[1]:x[2]], axis=1)

condition = temp_df['pred_starts'] >= temp_df['pred_ends']
temp_df.loc[:, 'selected_text'][condition] = temp_df.loc[:, 'text'][condition]

submission_df = pd.DataFrame() 
submission_df['textID'] = test_data['textID']
submission_df['selected_text'] = temp_df['selected_text']
submission_df.to_csv(SUBMISSION_FILE, index = False)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2555890665.py in <cell line: 0>()
      1 temp_df = pd.DataFrame()
----> 2 temp_df['pred_starts'] = model_starts.predict(kaggle_test_ft)  # predict starts
      3 temp_df['pred_ends'] = model_ends.predict(kaggle_test_ft)      # predict ends
      4 
      5 # columns = ['text', 'sentiment']

NameError: name 'kaggle_test_ft' is not defined

## === cell 16
pd.set_option('max_colwidth', 80)
test_data.head()


## === cell 17
submission_df.head(5)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2286490211.py in <cell line: 0>()
----> 1 submission_df.head(5)

NameError: name 'submission_df' is not defined
