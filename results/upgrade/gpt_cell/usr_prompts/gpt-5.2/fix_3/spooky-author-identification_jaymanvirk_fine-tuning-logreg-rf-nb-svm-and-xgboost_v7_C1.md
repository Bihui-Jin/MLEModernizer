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
plt.figure(figsize=(16, 9))

if "author" not in train_data.columns and "target" not in train_data.columns:
    base_dir = "/kaggle/input"
    candidates = [
        (
            os.path.join(base_dir, "train.csv"),
            os.path.join(base_dir, "test.csv"),
            os.path.join(base_dir, "sample_submission.csv"),
        ),
        (
            os.path.join(base_dir, "spooky-author-identification", "train.csv"),
            os.path.join(base_dir, "spooky-author-identification", "test.csv"),
            os.path.join(
                base_dir, "spooky-author-identification", "sample_submission.csv"
            ),
        ),
    ]
    loaded = False
    for tr_path, te_path, ss_path in candidates:
        if (
            os.path.exists(tr_path)
            and os.path.exists(te_path)
            and os.path.exists(ss_path)
        ):
            train_data = pd.read_csv(tr_path)
            test_data = pd.read_csv(te_path)
            sample_data = pd.read_csv(ss_path)
            loaded = True
            break
    if not loaded:
        raise FileNotFoundError(
            "Could not find expected train/test/sample_submission CSVs under /kaggle/input."
        )

if "author" in train_data.columns:
    label_col = "author"
elif "target" in train_data.columns:
    label_col = "target"
else:
    raise KeyError(
        "Expected label column 'author' in train_data, but it was not found. "
        f"Available columns: {list(train_data.columns)}. "
        "This likely means train_data was read from the wrong CSV."
    )

tmp = train_data[label_col].value_counts()
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


## === cell 17
train_clean_data[0]


## === cell 18
del spacy_process, pattern, stop_words
del get_clean_text, get_processed_text


## === cell 19
vectorizer = TfidfVectorizer(sublinear_tf = True
                             , ngram_range = (1,2)
                             )
tfidf_vect = vectorizer.fit(train_clean_data)


## === cell 20
X_train_tfidf = tfidf_vect.transform(train_clean_data)
X_test_tfidf = tfidf_vect.transform(test_clean_data)


## === cell 21
del vectorizer, test_data, train_clean_data, test_clean_data


## === cell 22
svd = TruncatedSVD(n_components=100)
X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd = svd.transform(X_test_tfidf)


## === cell 23
scaler = StandardScaler(with_mean = False)
X_train_stand = scaler.fit_transform(X_train_tfidf)
X_test_stand = scaler.fit_transform(X_test_tfidf)

X_train_stand_svd = scaler.fit_transform(X_train_svd)
X_test_stand_svd = scaler.fit_transform(X_test_svd)


## === cell 24
del scaler


## === cell 25
dict_map = {'EAP':0, 'MWS':1, 'HPL':2}
y_train = train_data['author'].map(dict_map).values


## === cell 26
del label_encoder, train_data


## --- ERROR in cell 26, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/346813938.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;31m# Cleaning[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0;32mdel[0m [0mlabel_encoder[0m[0;34m,[0m [0mtrain_data[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;31mNameError[0m: name 'label_encoder' is not defined

## === cell 27
config_logreg = {
    'model': LogisticRegression()
    , 'name': 'Log Reg'
    , 'param_grid':
    {
        'class_weight': ['balanced']
        , 'C': [0.5, 1.0, 1.5]
        , 'solver': ['saga', 'lbfgs']
    }
}
