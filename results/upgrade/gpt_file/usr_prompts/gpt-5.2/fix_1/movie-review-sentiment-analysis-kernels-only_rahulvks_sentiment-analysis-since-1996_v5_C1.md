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
Predict the sentiment of phrases.

## Metric
Classification accuracy.

## Submission Format
For each phrase in the test set, predict a label for the sentiment. Your submission should have a header and look like the following:

```
PhraseId,Sentiment
156061,2
156062,2
156063,2
...
```

## Dataset
The dataset is comprised of tab-separated files with phrases. Each phrase has a PhraseId. Each sentence has a SentenceId.

The sentiment labels are:

0 - negative

1 - somewhat negative

2 - neutral

3 - somewhat positive

4 - positive

# 2. Python version

3.6

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        input/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        working/
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
```

-> data/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> data/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> (stopped after 10 files for performance)

# 5. Target score

0.62728

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import re
from nltk.tokenize import TweetTokenizer
import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
pd.set_option('max_colwidth',400)
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import cross_val_score
eng_stopwords = set(stopwords.words("english"))
import matplotlib.gridspec as gridspec 
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc

from sklearn.metrics import accuracy_score
import os
print(os.listdir("../input"))



## === cell 1
train = pd.read_csv('../input/train.tsv',delimiter='\t')
test = pd.read_csv('../input/test.tsv',delimiter='\t')
sub = pd.read_csv('../input/sampleSubmission.csv')


## === cell 2
print(train.shape, test.shape)


## === cell 3
train.head()


## === cell 4
train['Sentiment'].value_counts()


## === cell 5
train.describe()


## === cell 6
test.describe()


## === cell 7
x=train['Sentiment'].value_counts()
plt.figure(figsize=(15,6))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("# per class")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('Type ', fontsize=12)
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/323490745.py in <cell line: 0>()
      2 #plot
      3 plt.figure(figsize=(15,6))
----> 4 ax= sns.barplot(x.index, x.values, alpha=0.8)
      5 plt.title("# per class")
      6 plt.ylabel('# of Occurrences', fontsize=12)

TypeError: barplot() takes from 0 to 1 positional arguments but 2 were given

## === cell 8
def cleaning(s):
    
    s = str(s)
    s = s.lower()
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub(r'[^\w]', ' ', s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace(",","")
    s = s.replace("[\w*"," ")
    s = re.sub(r'https?:\/\/.*[\r\n]*', '', s, flags=re.MULTILINE)
    s = re.sub(r'\, ' ', s)
    s = re.sub(r'&', '', s) 
    s = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', s)
    s = re.sub(r'[^\x00-\x7f]',r'',s) #removes arabic
    s = re.sub(r'', ' ', s)
    s = re.sub(r'\'', ' ', s)
    
    return s


## --- ERROR in cell 8, traceback:
  File "/tmp/ipykernel_11/2838282383.py", line 15
    s = re.sub(r'\, ' ', s)
                      ^
SyntaxError: unterminated string literal (detected at line 15)


## === cell 9
train['Phrase_Clean'] = [cleaning(s) for s in train['Phrase']]


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1315917184.py in <cell line: 0>()
----> 1 train['Phrase_Clean'] = [cleaning(s) for s in train['Phrase']]

/tmp/ipykernel_11/1315917184.py in <listcomp>(.0)
----> 1 train['Phrase_Clean'] = [cleaning(s) for s in train['Phrase']]

NameError: name 'cleaning' is not defined

## === cell 10
APPLY_STEMMING = True

if APPLY_STEMMING:
    import nltk.stem as stm # Import stem class from nltk
    stemmer = stm.PorterStemmer()


## === cell 11
train.Phrase_Clean = train.Phrase_Clean.apply(lambda text: " ".join([stemmer.stem(word) for word in text.split(" ")]))


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1358962744.py in <cell line: 0>()
----> 1 train.Phrase_Clean = train.Phrase_Clean.apply(lambda text: " ".join([stemmer.stem(word) for word in text.split(" ")]))

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'Phrase_Clean'

## === cell 12
train['count_word']=train["Phrase_Clean"].apply(lambda x: len(str(x).split()))
train['count_unique_word']=train["Phrase_Clean"].apply(lambda x: len(set(str(x).split())))
train["count_stopwords"] = train["Phrase_Clean"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
train["mean_word_len"] = train["Phrase_Clean"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))


## --- ERROR in cell 12, traceback:
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

KeyError: 'Phrase_Clean'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2198565528.py in <cell line: 0>()
----> 1 train['count_word']=train["Phrase_Clean"].apply(lambda x: len(str(x).split()))
      2 train['count_unique_word']=train["Phrase_Clean"].apply(lambda x: len(set(str(x).split())))
      3 train["count_stopwords"] = train["Phrase_Clean"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
      4 train["mean_word_len"] = train["Phrase_Clean"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

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

KeyError: 'Phrase_Clean'

## === cell 13
plt.figure(figsize=(12,6))
plt.subplot(122)
sns.violinplot(y='count_word',x='Sentiment', data=train,split=True,inner="quart")
plt.xlabel('Words Vs Target', fontsize=12)
plt.ylabel('# of words', fontsize=12)
plt.title("Number of words in each comment", fontsize=15)

plt.show()

plt.figure(figsize=(12,6))

plt.subplot(122)
sns.violinplot(y='count_unique_word',x='Sentiment', data=train,split=True,inner="quart")
plt.xlabel('Unique Word Vs Target', fontsize=12)
plt.ylabel('# of words', fontsize=12)
plt.title("Number of words in each comment", fontsize=15)

plt.show()

plt.figure(figsize=(12,6))

plt.subplot(122)
sns.violinplot(y='count_stopwords',x='Sentiment', data=train,split=True,inner="quart")
plt.xlabel('count stopwords Word Vs Target', fontsize=12)
plt.ylabel('# of words', fontsize=12)
plt.title("Number of words in each comment", fontsize=15)

plt.show()

plt.figure(figsize=(12,6))

plt.subplot(122)
sns.violinplot(y='count_stopwords',x='Sentiment', data=train,split=True,inner="quart")
plt.xlabel('mean_word_len Vs Target', fontsize=12)
plt.ylabel('# of words', fontsize=12)
plt.title("Number of words in each comment", fontsize=15)

plt.show()


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3661347246.py in <cell line: 0>()
      2 # words
      3 plt.subplot(122)
----> 4 sns.violinplot(y='count_word',x='Sentiment', data=train,split=True,inner="quart")
      5 plt.xlabel('Words Vs Target', fontsize=12)
      6 plt.ylabel('# of words', fontsize=12)

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in violinplot(data, x, y, hue, order, hue_order, bw, cut, scale, scale_hue, gridsize, width, inner, split, dodge, orient, linewidth, color, palette, saturation, ax, **kwargs)
   2303 ):
   2304 
-> 2305     plotter = _ViolinPlotter(x, y, hue, data, order, hue_order,
   2306                              bw, cut, scale, scale_hue, gridsize,
   2307                              width, inner, split, dodge, orient, linewidth,

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in __init__(self, x, y, hue, data, order, hue_order, bw, cut, scale, scale_hue, gridsize, width, inner, split, dodge, orient, linewidth, color, palette, saturation)
    899                  color, palette, saturation):
    900 
--> 901         self.establish_variables(x, y, hue, data, orient, order, hue_order)
    902         self.establish_colors(color, palette, saturation)
    903         self.estimate_densities(bw, cut, scale, scale_hue, gridsize)

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in establish_variables(self, x, y, hue, data, orient, order, hue_order, units)
    539                 if isinstance(var, str):
    540                     err = f"Could not interpret input '{var}'"
--> 541                     raise ValueError(err)
    542 
    543             # Figure out the plotting orientation

ValueError: Could not interpret input 'count_word'

## === cell 14
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words = None,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1,3),
    dtype=np.float32,
    max_features=20000
)


char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words = None,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 4),
    dtype=np.float32,
    max_features=30000
)

word_vectorizer.fit(train['Phrase_Clean'])

char_vectorizer.fit(train['Phrase_Clean'])


## --- ERROR in cell 14, traceback:
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

KeyError: 'Phrase_Clean'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1398726032.py in <cell line: 0>()
     23 )
     24 
---> 25 word_vectorizer.fit(train['Phrase_Clean'])
     26 
     27 char_vectorizer.fit(train['Phrase_Clean'])

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

KeyError: 'Phrase_Clean'

## === cell 15
Target = train["Sentiment"]


## === cell 16
train_word_features = word_vectorizer.transform(train['Phrase_Clean'])
train_char_features = char_vectorizer.transform(train['Phrase_Clean'])


## --- ERROR in cell 16, traceback:
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

KeyError: 'Phrase_Clean'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/684233868.py in <cell line: 0>()
      1 # Train
----> 2 train_word_features = word_vectorizer.transform(train['Phrase_Clean'])
      3 train_char_features = char_vectorizer.transform(train['Phrase_Clean'])

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

KeyError: 'Phrase_Clean'

## === cell 17
train_features = hstack([
    train_char_features,
    train_word_features])


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2359110996.py in <cell line: 0>()
      1 train_features = hstack([
----> 2     train_char_features,
      3     train_word_features])

NameError: name 'train_char_features' is not defined

## === cell 18
from sklearn.naive_bayes import MultinomialNB
model_NB = MultinomialNB()
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(train_features, Target, train_size=0.75)
model_NB.fit(X_train_tfidf, y_train_tfidf)
predictions_tfidf = model_NB.predict(X_test_tfidf)
accuracy_tfidf = accuracy_score(y_test_tfidf, predictions_tfidf)
print(accuracy_tfidf)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/456146496.py in <cell line: 0>()
      1 from sklearn.naive_bayes import MultinomialNB
      2 model_NB = MultinomialNB()
----> 3 X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(train_features, Target, train_size=0.75)
      4 model_NB.fit(X_train_tfidf, y_train_tfidf)
      5 predictions_tfidf = model_NB.predict(X_test_tfidf)

NameError: name 'train_features' is not defined

## === cell 19
print("Auc Score: ",np.mean(cross_val_score(model_NB, train_features, Target, cv=5,)))


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2881918172.py in <cell line: 0>()
----> 1 print("Auc Score: ",np.mean(cross_val_score(model_NB, train_features, Target, cv=5,)))

NameError: name 'train_features' is not defined

## === cell 20
print("Modeling..")
loss = []
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(train_features, Target, train_size=0.75)

lr = LogisticRegression(solver="liblinear", max_iter=500,class_weight='balanced')
lr.fit(train_features,Target)
lr_pred=lr.predict(X_test_tfidf)
accuracy_tfidf =accuracy_score(y_test_tfidf,lr_pred)
print(accuracy_tfidf)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/521097607.py in <cell line: 0>()
      1 print("Modeling..")
      2 loss = []
----> 3 X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(train_features, Target, train_size=0.75)
      4 
      5 lr = LogisticRegression(solver="liblinear", max_iter=500,class_weight='balanced')

NameError: name 'train_features' is not defined

## === cell 21
print("Auc Score: ",np.mean(cross_val_score(lr, train_features, Target, cv=3,)))


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1133891187.py in <cell line: 0>()
----> 1 print("Auc Score: ",np.mean(cross_val_score(lr, train_features, Target, cv=3,)))

NameError: name 'lr' is not defined

## === cell 22
Target_Names = train['Sentiment'].unique()
Target_Names


## === cell 23
print(classification_report(y_test_tfidf,lr_pred))


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2753860942.py in <cell line: 0>()
----> 1 print(classification_report(y_test_tfidf,lr_pred))

NameError: name 'y_test_tfidf' is not defined

## === cell 24
test['Phrase_Clean'] = [cleaning(s) for s in test['Phrase']]
test.Phrase_Clean = test.Phrase_Clean.apply(lambda text: " ".join([stemmer.stem(word) for word in text.split(" ")]))


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3013395179.py in <cell line: 0>()
----> 1 test['Phrase_Clean'] = [cleaning(s) for s in test['Phrase']]
      2 test.Phrase_Clean = test.Phrase_Clean.apply(lambda text: " ".join([stemmer.stem(word) for word in text.split(" ")]))

/tmp/ipykernel_11/3013395179.py in <listcomp>(.0)
----> 1 test['Phrase_Clean'] = [cleaning(s) for s in test['Phrase']]
      2 test.Phrase_Clean = test.Phrase_Clean.apply(lambda text: " ".join([stemmer.stem(word) for word in text.split(" ")]))

NameError: name 'cleaning' is not defined

## === cell 25
test_word_features = word_vectorizer.transform(test['Phrase_Clean'])
test_char_features = char_vectorizer.transform(test['Phrase_Clean'])


## --- ERROR in cell 25, traceback:
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

KeyError: 'Phrase_Clean'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2704081325.py in <cell line: 0>()
      1 # test
----> 2 test_word_features = word_vectorizer.transform(test['Phrase_Clean'])
      3 test_char_features = char_vectorizer.transform(test['Phrase_Clean'])

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

KeyError: 'Phrase_Clean'

## === cell 26
test_features = hstack([
    test_char_features,
    test_word_features])


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/974808179.py in <cell line: 0>()
      1 test_features = hstack([
----> 2     test_char_features,
      3     test_word_features])

NameError: name 'test_char_features' is not defined

## === cell 27
predicted_values = lr.predict(test_features)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3287229785.py in <cell line: 0>()
----> 1 predicted_values = lr.predict(test_features)

NameError: name 'lr' is not defined

## === cell 28
test.head()


## === cell 29
test = test.drop('Phrase_Clean', 1)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4180013385.py in <cell line: 0>()
----> 1 test = test.drop('Phrase_Clean', 1)

TypeError: DataFrame.drop() takes from 1 to 2 positional arguments but 3 were given

## === cell 30
test['Sentiment'] = predicted_values


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/315835167.py in <cell line: 0>()
----> 1 test['Sentiment'] = predicted_values

NameError: name 'predicted_values' is not defined

## === cell 31
test.head()


## === cell 32
test[['PhraseId', 'Sentiment']].to_csv('submission_lr.csv', index=False)


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1738582557.py in <cell line: 0>()
----> 1 test[['PhraseId', 'Sentiment']].to_csv('submission_lr.csv', index=False)

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

KeyError: "['Sentiment'] not in index"

## === cell 33
test.shape
