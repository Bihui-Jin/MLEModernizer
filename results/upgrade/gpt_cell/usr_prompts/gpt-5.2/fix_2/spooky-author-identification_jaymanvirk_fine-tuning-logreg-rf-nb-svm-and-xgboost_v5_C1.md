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

3.12

# 2. Installed packages

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
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
xgboost==2.0.3

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os
import time
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from concurrent.futures import ProcessPoolExecutor
from sklearn.decomposition import TruncatedSVD

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb

from sklearn.model_selection import GridSearchCV


## === cell 1
list_l = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        list_l.append(os.path.join(dirname, filename))
list_l


## === cell 2
train_data = pd.read_csv(list_l[0])
test_data = pd.read_csv(list_l[1])
sample_data = pd.read_csv(list_l[2])


## === cell 3
del list_l


## === cell 4
def print_short_summary(name, data):
    """
    Print data head, shape and info.
    Args:
        name (str): name of dataset
        data (dataframe): dataset in a pd.DataFrame format
    """
    print(name)
    print('\n1. Data head:')
    print(data.head())
    print('\n2 Data shape: {}'.format(data.shape))
    print('\n3. Data info:')
    data.info()
    if 'text' in data.columns:
        avg = np.mean(np.vectorize(len)(data['text']))
        print('\n4. Average number of characters per text: {:.0f}'.format(avg))


## === cell 5
print_short_summary('Train data', train_data)


## === cell 6
print_short_summary('Test data', test_data)


## === cell 7
print_short_summary('Sample data', sample_data)


## === cell 8
del print_short_summary


## === cell 9
if "author" not in train_data.columns:
    candidate_paths = [
        "/kaggle/input/train.csv",
        "/kaggle/input/spooky-author-identification/train.csv",
    ]
    loaded = False
    for p in candidate_paths:
        if os.path.exists(p):
            train_data = pd.read_csv(p)
            loaded = True
            break
    if not loaded:
        raise KeyError(
            "'author' column not found in train_data, and could not locate train.csv "
            "in expected input paths."
        )

plt.figure(figsize=(16, 9))
tmp = train_data["author"].value_counts()
sns.barplot(y=tmp.index.values, x=tmp.values, orient="h")
plt.xlabel("Number of records")
plt.ylabel("Author")
plt.title("Number of records per author")
plt.show()


## === cell 10
del tmp


## === cell 11
def plot_word_dist_author(labels, top_n_words = 10):
    """
    Plot charts with word frequencies per author
    Args:
        labels: list of authors
        top_n_words (opt): how many top words to plot in one chart
    """
    n = len(labels)
    
    default_palette = sns.color_palette("deep")
    
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(16, 9))
    
    for i in range(n):
        col = i % n
        indexes = train_data['author'] == labels[i]
        w = train_data['text'][indexes].str.split(expand=True).unstack().value_counts()
        l = w[:top_n_words]/np.sum(w)*100
        axes[col].bar(l.index, l.values, color=default_palette[i])
        axes[col].set_title(labels[i])
        axes[col].set_xlabel('Words')
        axes[col].set_ylabel('Percentage of total word count (%)')

    plt.tight_layout()
    plt.show()


## === cell 12
plot_word_dist_author(['EAP', 'MWS', 'HPL'])


## === cell 13
del plot_word_dist_author


## === cell 14
spacy_process = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

pattern = re.compile(r'\b([a-zA-Z])\b|\d+|[.,!?()-:;]')

stop_words = set(stopwords.words('english'))


## === cell 15
def get_processed_text(text):
    """
    Return lemmatized text without single letters and digits.
    Everything is in the lower case register.
    Args:
        text (str): text of an article
    Returns:
        text (str): cleand text
    """
    text = pattern.sub('', text.lower())
    
    lemmas = spacy_process(text)
    lemmas = [token.lemma_ for token in lemmas if token.text not in stop_words]

    text = ' '.join(lemmas)
    
    return text

def get_clean_text(texts):
    """
    Return cleaned text.
    Execution in parallel.
    
    Args:
        texts: numpy array of string elements
    Returns:
        clean_texts: numpy array of cleaned string elements
    """
    with ProcessPoolExecutor() as executor:
        clean_texts = list(executor.map(get_processed_text, texts))
        
    return np.array(clean_texts)


## === cell 16
train_clean_data = get_clean_text(train_data['text'].values)
test_clean_data = get_clean_text(test_data['text'].values)


## --- ERROR in cell 16, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mKeyError[0m                                  Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py[0m in [0;36mget_loc[0;34m(self, key)[0m
[1;32m   3804[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3805[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_engine[0m[0;34m.[0m[0mget_loc[0m[0;34m([0m[0mcasted_key[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3806[0m         [0;32mexcept[0m [0mKeyError[0m [0;32mas[0m [0merr[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32mindex.pyx[0m in [0;36mpandas._libs.index.IndexEngine.get_loc[0;34m()[0m

[0;32mindex.pyx[0m in [0;36mpandas._libs.index.IndexEngine.get_loc[0;34m()[0m

[0;32mpandas/_libs/hashtable_class_helper.pxi[0m in [0;36mpandas._libs.hashtable.PyObjectHashTable.get_item[0;34m()[0m

[0;32mpandas/_libs/hashtable_class_helper.pxi[0m in [0;36mpandas._libs.hashtable.PyObjectHashTable.get_item[0;34m()[0m

[0;31mKeyError[0m: 'text'

The above exception was the direct cause of the following exception:

[0;31mKeyError[0m                                  Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1632344608.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;31m# Get cleaned train and test texts[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0mtrain_clean_data[0m [0;34m=[0m [0mget_clean_text[0m[0;34m([0m[0mtrain_data[0m[0;34m[[0m[0;34m'text'[0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 3[0;31m [0mtest_clean_data[0m [0;34m=[0m [0mget_clean_text[0m[0;34m([0m[0mtest_data[0m[0;34m[[0m[0;34m'text'[0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36m__getitem__[0;34m(self, key)[0m
[1;32m   4100[0m             [0;32mif[0m [0mself[0m[0;34m.[0m[0mcolumns[0m[0;34m.[0m[0mnlevels[0m [0;34m>[0m [0;36m1[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4101[0m                 [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_getitem_multilevel[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 4102[0;31m             [0mindexer[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mcolumns[0m[0;34m.[0m[0mget_loc[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   4103[0m             [0;32mif[0m [0mis_integer[0m[0;34m([0m[0mindexer[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4104[0m                 [0mindexer[0m [0;34m=[0m [0;34m[[0m[0mindexer[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py[0m in [0;36mget_loc[0;34m(self, key)[0m
[1;32m   3810[0m             ):
[1;32m   3811[0m                 [0;32mraise[0m [0mInvalidIndexError[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3812[0;31m             [0;32mraise[0m [0mKeyError[0m[0;34m([0m[0mkey[0m[0;34m)[0m [0;32mfrom[0m [0merr[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3813[0m         [0;32mexcept[0m [0mTypeError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3814[0m             [0;31m# If we have a listlike key, _check_indexing_error will raise[0m[0;34m[0m[0;34m[0m[0m

[0;31mKeyError[0m: 'text'

## === cell 17
train_clean_data[0]
