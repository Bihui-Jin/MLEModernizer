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
if "text" not in train_data.columns:
    candidate_train_paths = [
        "/kaggle/input/train.csv",
        "/kaggle/input/spooky-author-identification/train.csv",
    ]
    loaded = False
    for p in candidate_train_paths:
        if os.path.exists(p):
            train_data = pd.read_csv(p)
            loaded = True
            break
    if not loaded or "text" not in train_data.columns:
        raise KeyError(
            "'text' column not found in train_data, and could not locate a valid train.csv "
            "in expected input paths."
        )

if "text" not in test_data.columns:
    candidate_test_paths = [
        "/kaggle/input/test.csv",
        "/kaggle/input/spooky-author-identification/test.csv",
    ]
    loaded = False
    for p in candidate_test_paths:
        if os.path.exists(p):
            test_data = pd.read_csv(p)
            loaded = True
            break
    if not loaded or "text" not in test_data.columns:
        raise KeyError(
            "'text' column not found in test_data, and could not locate a valid test.csv "
            "in expected input paths."
        )

train_clean_data = get_clean_text(train_data["text"].values)
test_clean_data = get_clean_text(test_data["text"].values)


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
dict_map = {'EAP':0, 'MWS':1, 'HPL':2,}
y_train = train_data['author'].map(dict_map).values


## === cell 26
if "label_encoder" in globals():
    del label_encoder
del train_data


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


## === cell 28
config_rf = {
    'model': RandomForestClassifier()
    , 'name': 'Random Forest'
    , 'param_grid':
    {
        'n_estimators': [50, 75, 100]
        , 'max_depth': [10, 20]
    }
}


## === cell 29
config_nb = {
    'model': MultinomialNB()
    , 'name': 'Mult NB'
    , 'param_grid':
    {
        'alpha': [0.001, 0.01, 0.1, 0.2, 0.4]
    }
}


## === cell 30
config_svm = {
    'model': SVC()
    , 'name': 'SVM'
    , 'param_grid':
    {
        'class_weight': ['balanced']
        , 'probability': [True]
        , 'max_iter': [100]
        , 'C': [0.5, 1.0, 1.5]
        , 'kernel': ['rbf', 'sigmoid']
    }
}


## === cell 31
config_xgb = {
    'model': xgb.XGBClassifier()
    , 'name': 'XGBoost'
    , 'param_grid':
    {
        'objective': ['multi:softmax']
        , 'num_class': [3]
        , 'eval_metric': ['mlogloss']
        , 'n_estimators': [100]
        , 'eta': [0.2, 0.3, 0.4]
        , 'max_depth': [5, 7]
    }
}


## === cell 32
def get_grid(config_model, X_train, y_train):
    """
    Return grid of GridSearchCV results from selected models and their parameters.
    Execution in parallel.
    
    Args:
        config_model (dict): dictionary of model's parameters
        X_train (ndarray): data to train
        y_train (ndarray): data labels
    Returns:
        dict: a dictionary with the results from training via GridSeachCV
    """
    grid = GridSearchCV(config_model['model']
                        , config_model['param_grid']
                        , return_train_score = True
                        , scoring = 'neg_log_loss'
                        , cv = 5
                        , n_jobs = -1)

    grid = grid.fit(X_train, y_train)
        

    return grid


def get_model_results(model_name, grid_results, data_standardized):
    """
    Return a dictionary of model results.
    
    Args:
        model_name (str): name of a model
        grid_results (dict): GridSearchCV dictionary of model results
        data_standardized (1 or 0): data is stadardized
    Returns:
        dict: dictionary with updated columns
    """
    runtime = grid_results['mean_fit_time'] + grid_results['mean_score_time']
    model_results = {
        'model': model_name
        , 'params': grid_results['params']
        , 'data_std': data_standardized
        , 'mean_runtime (sec)': runtime
        , 'mean_train_score (logloss)': -grid_results['mean_train_score']
        , 'mean_test_score (logloss)': -grid_results['mean_test_score']
    }
    
    return model_results

def get_table_results_sorted(list_dict):
    """
    Convert list of dictionaries and sort by mean_test_score and mean_runtime.
    
    Args:
        list_dict (list): list of dictionaries from GridSearchCV
    Returns:
        DataFrame: sorted dataframe by mean_test_score and mean_runtime
    """
    table = [pd.DataFrame(results) for results in list_dict]   
    table = pd.concat(table)
    table = table.sort_values(by = ['mean_test_score (logloss)'
                                  ,'mean_runtime (sec)']
                        , ascending = [True, True])
    
    return table

def get_final_table_results(config_models, config_data):
    """
    Return table with every trained model and its results from GridSearchCV.
    
    Args:
        config_models (list): list of model configurations
        confg_data (list): list of data configurations
    Returns:
        DataFrame: sorted dataframe by mean_test_score and mean_runtime
    """
    n = len(config_models)
    m = len(config_data)
    table = []
    for i in range(n):
        for j in range(m):
            print('model: {} of {} | data: {} of {}'.format(i+1,n,j+1,m))
            grid = get_grid(config_models[i], config_data[j], y_train)
            model_results = get_model_results(config_models[i]['name']
                                              , grid.cv_results_
                                             , j)
            table.append(model_results)
    
    table = get_table_results_sorted(table)

    return table


## === cell 33
table_nb = get_final_table_results([config_nb], [X_train_tfidf, X_train_stand])


## === cell 34
config_models = [config_logreg, config_rf, config_svm, config_xgb]
config_data = [X_train_svd, X_train_stand_svd]


## === cell 35
table_results = get_final_table_results(config_models, config_data)


## === cell 36
table_results = get_table_results_sorted([table_results, table_nb])
table_results


## === cell 37
del config_models, config_data, table_results
del get_results_table, get_grid, get_final_table_results


## --- ERROR in cell 37, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/605265537.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;31m# Cleaning[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;32mdel[0m [0mconfig_models[0m[0;34m,[0m [0mconfig_data[0m[0;34m,[0m [0mtable_results[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 3[0;31m [0;32mdel[0m [0mget_results_table[0m[0;34m,[0m [0mget_grid[0m[0;34m,[0m [0mget_final_table_results[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;31mNameError[0m: name 'get_results_table' is not defined

## === cell 38
model = MultinomialNB(alpha = 0.01).fit(X_train_tfidf, y_train)
