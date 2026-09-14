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
Given some text, predict the author.

## Metric
Multi-class logarithmic loss. 

The submitted probabilities for a given sentences are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum).

In order to avoid the extremes of the log function, predicted probabilities are replaced with \\(max(min(p,1-10^{-15}),10^{-15})\\).

## Submission Format
You must submit a csv file with the id, and a probability for each of the three classes. The order of the rows does not matter. The file must have a header and should look like the following:

```
id,EAP,HPL,MWS
id07943,0.33,0.33,0.33
...
```

## Dataset 
### File descriptions
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Data fields
- **id** - a unique identifier for each sentence
- **text** - some text written by one of the authors
- **author** - the author of the sentence (EAP: Edgar Allan Poe, HPL: HP Lovecraft; MWS: Mary Wollstonecraft Shelley)

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
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
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        input/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        working/
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
```

-> data/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/spooky-author-identification/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> data/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> input/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> (stopped after 10 files for performance)

# 5. Target score

0.29678

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes
color = sns.color_palette()

%matplotlib inline

eng_stopwords = set(stopwords.words('english'))
pd.options.mode.chained_assignment = None


## === cell 1
train_df = pd.read_csv('../input/spooky-author-identification/train.zip')
test_df = pd.read_csv('../input/spooky-author-identification/test.zip')
print(f"Number of rows in train dataset : {train_df.shape[0]}")
print(f"Number of rows in test dataset : {test_df.shape[0]}")


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2084276435.py in <cell line: 0>()
----> 1 train_df = pd.read_csv('../input/spooky-author-identification/train.zip')
      2 test_df = pd.read_csv('../input/spooky-author-identification/test.zip')
      3 print(f"Number of rows in train dataset : {train_df.shape[0]}")
      4 print(f"Number of rows in test dataset : {test_df.shape[0]}")

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
    803                     raise ValueError(f"Zero files found in ZIP file {path_or_buf}")
    804                 else:
--> 805                     raise ValueError(
    806                         "Multiple files found in ZIP file. "
    807                         f"Only one file per ZIP: {zip_names}"

ValueError: Multiple files found in ZIP file. Only one file per ZIP: ['train.csv', 'sample_submission.csv']

## === cell 2
train_df.head()


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2663572906.py in <cell line: 0>()
----> 1 train_df.head()

NameError: name 'train_df' is not defined

## === cell 3
cnt_srs = train_df['author'].value_counts()

plt.figure(figsize=(8,4))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.ylabel('Number of Occurences', fontsize=12)
plt.xlabel('Author Name', fontsize=12)
plt.show()


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1742154077.py in <cell line: 0>()
----> 1 cnt_srs = train_df['author'].value_counts()
      2 
      3 plt.figure(figsize=(8,4))
      4 sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
      5 plt.ylabel('Number of Occurences', fontsize=12)

NameError: name 'train_df' is not defined

## === cell 4
grouped_df = train_df.groupby('author')

for name, group in grouped_df:
    print(f"Author name : {name}")
    cnt = 0
    
    for idx, row in group.iterrows():
        print(row['text'])
        cnt += 1
        
        if cnt == 5:
            break
    
    print("\n")


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4194115383.py in <cell line: 0>()
----> 1 grouped_df = train_df.groupby('author')
      2 
      3 for name, group in grouped_df:
      4     print(f"Author name : {name}")
      5     cnt = 0

NameError: name 'train_df' is not defined

## === cell 5
train_df['num_words'] = train_df['text'].apply(lambda x: len(str(x).split()))
test_df['num_words'] = test_df['text'].apply(lambda x: len(str(x).split()))

train_df['num_unique_words'] = train_df['text'].apply(lambda x: len(set(str(x).split())))
test_df['num_unique_words'] = test_df['text'].apply(lambda x: len(set(str(x).split())))

train_df['num_chars'] = train_df['text'].apply(lambda x: len(str(x)))
test_df['num_chars'] = test_df['text'].apply(lambda x: len(str(x)))

train_df['num_stopwrods'] = train_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test_df['num_stopwrods'] = test_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

train_df['num_punctuations'] = train_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
test_df['num_punctuations'] = test_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

train_df['num_words_upper'] = train_df['text'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test_df['num_words_upper'] = test_df['text'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

train_df['mean_word_len'] = train_df['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test_df['mean_word_len'] = test_df['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3429451871.py in <cell line: 0>()
      1 ## Number of words in the text ##
----> 2 train_df['num_words'] = train_df['text'].apply(lambda x: len(str(x).split()))
      3 test_df['num_words'] = test_df['text'].apply(lambda x: len(str(x).split()))
      4 
      5 ## Number of unique words in the text ##

NameError: name 'train_df' is not defined

## === cell 6
train_df['num_words'].loc[train_df['num_words']>80] = 80 # truncation for better visuals
plt.figure(figsize=(12, 8))
sns.violinplot(x='author', y='num_words', data=train_df)
plt.xlabel('Author Name', fontsize=12)
plt.ylabel('Number of words in text', fontsize=12)
plt.title('Number of words by author', fontsize=15)
plt.show()


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2459360702.py in <cell line: 0>()
----> 1 train_df['num_words'].loc[train_df['num_words']>80] = 80 # truncation for better visuals
      2 plt.figure(figsize=(12, 8))
      3 sns.violinplot(x='author', y='num_words', data=train_df)
      4 plt.xlabel('Author Name', fontsize=12)
      5 plt.ylabel('Number of words in text', fontsize=12)

NameError: name 'train_df' is not defined

## === cell 7
train_df['num_punctuations'].loc[train_df['num_punctuations']>10] = 10 # truncation for better visuals
plt.figure(figsize=(12, 8))
sns.violinplot(x='author', y='num_punctuations', data=train_df)
plt.xlabel('Author Name', fontsize=12)
plt.ylabel('Number of punctuations in text', fontsize=12)
plt.title('Number of punctuations by author', fontsize=15)
plt.show()


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2129473411.py in <cell line: 0>()
----> 1 train_df['num_punctuations'].loc[train_df['num_punctuations']>10] = 10 # truncation for better visuals
      2 plt.figure(figsize=(12, 8))
      3 sns.violinplot(x='author', y='num_punctuations', data=train_df)
      4 plt.xlabel('Author Name', fontsize=12)
      5 plt.ylabel('Number of punctuations in text', fontsize=12)

NameError: name 'train_df' is not defined

## === cell 8
author_mapping_dict = {'EAP':0, 'HPL':1, 'MWS':2}
train_y = train_df['author'].map(author_mapping_dict)
train_id = train_df['id'].values
test_id = test_df['id'].values

train_df['num_words'] = train_df['text'].apply(lambda x: len(str(x).split()))
test_df['num_words'] = test_df['text'].apply(lambda x: len(str(x).split()))
train_df['mean_word_len'] = train_df['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test_df['mean_word_len'] = test_df['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

cols_to_drop = ['id', 'text']
train_X = train_df.drop(cols_to_drop+['author'], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3116628808.py in <cell line: 0>()
      1 ## Prepare the data for modeling ##
      2 author_mapping_dict = {'EAP':0, 'HPL':1, 'MWS':2}
----> 3 train_y = train_df['author'].map(author_mapping_dict)
      4 train_id = train_df['id'].values
      5 test_id = test_df['id'].values

NameError: name 'train_df' is not defined

## === cell 9
def runXGB(train_X, train_y, test_X, test_y=None, test_X2=None, seed_val=0, child=1, colsample=0.3):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 3
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = 'mlogloss'
    param['min_child_weight'] = child
    param['subsample'] = 0.8
    param['colsample_bytree'] = colsample
    param['seed'] = seed_val
    num_rounds = 2000
    
    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    
    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [(xgtrain, 'train'), (xgtest, 'test')]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)
        
    pred_test_y = model.predict(xgtest, ntree_limit=model.best_ntree_limit)
    
    if test_X2 is not None:
        xgtest2 = xgb.DMatrix(test_X2)
        pred_test_y2 = model.predict(xgtest2, ntree_limit=model.best_ntree_limit)
    
    return pred_test_y, pred_test_y2, model


## === cell 10
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X, seed_val=0)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index, :] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    break

print(f'cv scores : ', cv_scores)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3409572944.py in <cell line: 0>()
      2 cv_scores = []
      3 pred_full_test = 0
----> 4 pred_train = np.zeros([train_df.shape[0], 3])
      5 
      6 for dev_index, val_index in kf.split(train_X):

NameError: name 'train_df' is not defined

## === cell 11
fig, ax = plt.subplots(figsize=(12, 12))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4165181355.py in <cell line: 0>()
      1 ### Plot the important variables ###
      2 fig, ax = plt.subplots(figsize=(12, 12))
----> 3 xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
      4 plt.show()

NameError: name 'model' is not defined

## === cell 12
tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist() + test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1059683008.py in <cell line: 0>()
      1 ### Fit transform the tfidf vectorizer ###
      2 tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
----> 3 full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist() + test_df['text'].values.tolist())
      4 train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
      5 test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())

NameError: name 'train_df' is not defined

## === cell 13
def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    
    return pred_test_y, pred_test_y2, model


## === cell 14
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index, :] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

print(f'Mean cv score : {np.mean(cv_scores)}')
pred_full_test = pred_full_test / 5.


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/822275280.py in <cell line: 0>()
      1 cv_scores = []
      2 pred_full_test = 0
----> 3 pred_train = np.zeros([train_df.shape[0], 3])
      4 kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
      5 

NameError: name 'train_df' is not defined

## === cell 15
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applies by setting 'normalize=True'
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment='center',
                color='white' if cm[i, j] > thresh else 'black')
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


## === cell 16
cnf_matrix = confusion_matrix(val_y, np.argmax(pred_val_y, axis=1))
np.set_printoptions(precision=2)

plt.figure(figsize=(8, 8))
plot_confusion_matrix(cnf_matrix, classes=['EAP', 'HPL', 'MWS'],
                     title='Confusion matrix, without normalization')
plt.show()


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1121859278.py in <cell line: 0>()
----> 1 cnf_matrix = confusion_matrix(val_y, np.argmax(pred_val_y, axis=1))
      2 np.set_printoptions(precision=2)
      3 
      4 # Plot non-normalized confusion matrix
      5 plt.figure(figsize=(8, 8))

NameError: name 'val_y' is not defined

## === cell 17
n_comp = 20
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)

train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))

train_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]
test_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)

del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3748376007.py in <cell line: 0>()
      1 n_comp = 20
      2 svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
----> 3 svd_obj.fit(full_tfidf)
      4 
      5 train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))

NameError: name 'full_tfidf' is not defined

## === cell 18
tfidf_vec = CountVectorizer(stop_words='english', ngram_range=(1,3))
tfidf_vec.fit(train_df['text'].values.tolist()+test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4121512007.py in <cell line: 0>()
      1 ### Fit transform the count vectorizer ###
      2 tfidf_vec = CountVectorizer(stop_words='english', ngram_range=(1,3))
----> 3 tfidf_vec.fit(train_df['text'].values.tolist()+test_df['text'].values.tolist())
      4 train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
      5 test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())

NameError: name 'train_df' is not defined

## === cell 19
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index, :] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    
print(f'Mean cv score : {np.mean(cv_scores)}')
pred_full_test = pred_full_test / 5.

train_df['nb_cvec_eap'] = pred_train[:,0]
train_df['nb_cvec_hpl'] = pred_train[:,1]
train_df['nb_cvec_mws'] = pred_train[:,2]
test_df['nb_cvec_eap'] = pred_full_test[:,0]
test_df['nb_cvec_hpl'] = pred_full_test[:,1]
test_df['nb_cvec_mws'] = pred_full_test[:,2]


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/818226617.py in <cell line: 0>()
      1 cv_scores = []
      2 pred_full_test = 0
----> 3 pred_train = np.zeros([train_df.shape[0], 3])
      4 kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
      5 for dev_index, val_index in kf.split(train_X):

NameError: name 'train_df' is not defined

## === cell 20
cnf_matrix = confusion_matrix(val_y, np.argmax(pred_val_y, axis=1))
np.set_printoptions(precision=2)

plt.figure(figsize=(8, 8))
plot_confusion_matrix(cnf_matrix, classes=['EAP', 'HPL', 'MWS'],
                     title='Confusion matrix of NB on word count, without normalization')
plt.show()


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1017010393.py in <cell line: 0>()
----> 1 cnf_matrix = confusion_matrix(val_y, np.argmax(pred_val_y, axis=1))
      2 np.set_printoptions(precision=2)
      3 
      4 # Plot non-normalized confusion matrix
      5 plt.figure(figsize=(8, 8))

NameError: name 'val_y' is not defined

## === cell 21
tfidf_vec = CountVectorizer(ngram_range=(1,7), analyzer='char')
tfidf_vec.fit(train_df['text'].values.tolist()+test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index, :] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

print(f'Mean cv score : {np.mean(cv_scores)}')
pred_full_test = pred_full_test / 5.

train_df['nb_cvec_char_eap'] = pred_train[:,0]
train_df['nb_cvec_char_hpl'] = pred_train[:,1]
train_df['nb_cvec_char_mws'] = pred_train[:,2]
test_df['nb_cvec_char_eap'] = pred_full_test[:,0]
test_df['nb_cvec_char_hpl'] = pred_full_test[:,1]
test_df['nb_cvec_char_mws'] = pred_full_test[:,2]


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3332767690.py in <cell line: 0>()
      1 ### Fit transform the tfidf vectorizer ###
      2 tfidf_vec = CountVectorizer(ngram_range=(1,7), analyzer='char')
----> 3 tfidf_vec.fit(train_df['text'].values.tolist()+test_df['text'].values.tolist())
      4 train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
      5 test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())

NameError: name 'train_df' is not defined

## === cell 22
tfidf_vec = TfidfVectorizer(ngram_range=(1,5), analyzer='char')
full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist()+test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

print(f'Mean cv score : {np.mean(cv_scores)}')
pred_full_test = pred_full_test / 5.

train_df['nb_tfidf_char_eap'] = pred_train[:,0]
train_df['nb_tfidf_char_hpl'] = pred_train[:,1]
train_df['nb_tfidf_char_mws'] = pred_train[:,2]
test_df['nb_tfidf_char_eap'] = pred_full_test[:,0]
test_df['nb_tfidf_char_hpl'] = pred_full_test[:,1]
test_df['nb_tfidf_char_mws'] = pred_full_test[:,2]


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/652766508.py in <cell line: 0>()
      1 #### Fit transform the tfidf vectorizer ###
      2 tfidf_vec = TfidfVectorizer(ngram_range=(1,5), analyzer='char')
----> 3 full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist()+test_df['text'].values.tolist())
      4 train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
      5 test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())

NameError: name 'train_df' is not defined

## === cell 23
n_comp = 20
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
    
train_svd.columns = ['svd_char_'+str(i) for i in range(n_comp)]
test_svd.columns = ['svd_char_'+str(i) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)

del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3267416998.py in <cell line: 0>()
      1 n_comp = 20
      2 svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
----> 3 svd_obj.fit(full_tfidf)
      4 train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
      5 test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))

NameError: name 'full_tfidf' is not defined

## === cell 24
cols_to_drop = ['id','text']
train_X = train_df.drop(cols_to_drop+['author'], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X, seed_val=0, colsample=0.7)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    break
print("cv scores : ", cv_scores)

out_df = pd.DataFrame(pred_full_test)
out_df.columns = ['EAP', 'HPL', 'MWS']
out_df.insert(0, 'id', test_id)
out_df.to_csv("sub_fe.csv", index=False)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2842100054.py in <cell line: 0>()
      1 cols_to_drop = ['id','text']
----> 2 train_X = train_df.drop(cols_to_drop+['author'], axis=1)
      3 test_X = test_df.drop(cols_to_drop, axis=1)
      4 
      5 kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

NameError: name 'train_df' is not defined

## === cell 25
fig, ax = plt.subplots(figsize=(12, 12))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4165181355.py in <cell line: 0>()
      1 ### Plot the important variables ###
      2 fig, ax = plt.subplots(figsize=(12, 12))
----> 3 xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
      4 plt.show()

NameError: name 'model' is not defined

## === cell 26
cnf_matrix = confusion_matrix(val_y, np.argmax(pred_val_y, axis=1))
np.set_printoptions(precision=2)

plt.figure(figsize=(8,8))
plot_confusion_matrix(cnf_matrix, classes=['EAP', 'HPL', 'MWS'],
                     title='Confusion matrix of XGB, without normalization')
plt.show()


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3010561474.py in <cell line: 0>()
----> 1 cnf_matrix = confusion_matrix(val_y, np.argmax(pred_val_y, axis=1))
      2 np.set_printoptions(precision=2)
      3 
      4 # Plot non-normalized confusio matrix
      5 plt.figure(figsize=(8,8))

NameError: name 'val_y' is not defined
