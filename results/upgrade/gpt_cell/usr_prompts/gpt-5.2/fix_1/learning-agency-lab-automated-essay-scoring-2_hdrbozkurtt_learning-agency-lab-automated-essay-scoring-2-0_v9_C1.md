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

catboost==1.2.8
geopandas==0.14.4
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0
xgboost==2.0.3

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, cohen_kappa_score
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingRegressor
from catboost import CatBoostRegressor
import pickle
import lightgbm as lgb
import xgboost as xgb
import re


## === cell 1
train = pd.read_csv("/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv")
test = pd.read_csv("/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv")


## === cell 2
tfidf_vectorizer = CountVectorizer(stop_words='english')
train_matrix = tfidf_vectorizer.fit_transform(train['full_text'])
mtrain =  pd.DataFrame.sparse.from_spmatrix(train_matrix)
mtrain["full_text"] = train.full_text
mtrain["score"] = train.score


test_matrix = tfidf_vectorizer.transform(test['full_text'])
mtest =  pd.DataFrame.sparse.from_spmatrix(test_matrix)
mtest["full_text"] = test.full_text


## === cell 3
def quadratic_weighted_kappa(preds, data):
    y_true = data.get_label()
    y_pred = preds.clip(1, 6).round()
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return 'QWK', qwk, True


def count_uppercase(text):
    return sum(1 for char in text if char.isupper())


punctuation_marks = string.punctuation

def count_punctuation(text):
    return sum(1 for char in text if char in punctuation_marks)



def count_dot_punctuation(text):
    return sum(1 for char in text if char in ".")
def count_coma_punctuation(text):
    return sum(1 for char in text if char in ",")

def count_tırnak_punctuation(text):
    return sum(1 for char in text if char in "'")
def count_çç_punctuation(text):
    return sum(1 for char in text if char in '"')

def count_kısa_punctuation(text):
    return sum(1 for char in text if char in "-")
def count_soru_punctuation(text):
    return sum(1 for char in text if char in '?')





def count_words(text):
    words = text.split()  
    return len(words)


def count_p(text):
    words = text.split("\n\n")  
    return len(words)

def count_integers(text):
    integers = [char for char in text if char.isdigit()]
    return len(integers)


def count_integers2(text):
    integer_groups = re.findall(r'\d+', text)
    total_count = sum(1 for group in integer_groups)
    return total_count


def no_space_dot(text):
    count = 0
    for i in range(len(text)):
        if text[i] == '.' and (i == len(text) - 1 or text[i + 1] != ' '):
            count += 1
    return count


def no_space_coma(text):
    count = 0
    for i in range(len(text)):
        if text[i] == ',' and (i == len(text) - 1 or text[i + 1] != ' '):
            count += 1
    return count


## === cell 4
mtrain["essay_len"] =  mtrain.full_text.apply(lambda x : len(x))
mtrain["total_uppercase"] =  mtrain.full_text.apply(lambda x : count_uppercase(x))

mtrain["total_punctuation"] =  mtrain.full_text.apply(lambda x : count_punctuation(x))
mtrain["total_dot_punctuation"] =  mtrain.full_text.apply(lambda x : count_dot_punctuation(x))
mtrain["total_coma_punctuation"] =  mtrain.full_text.apply(lambda x : count_coma_punctuation(x))
mtrain["total_çç_punctuation"] =  mtrain.full_text.apply(lambda x : count_çç_punctuation(x))
mtrain["total_tırnak_punctuation"] =  mtrain.full_text.apply(lambda x : count_tırnak_punctuation(x))
mtrain["total_kısa_punctuation"] =  mtrain.full_text.apply(lambda x : count_kısa_punctuation(x))
mtrain["total_soru_punctuation"] =  mtrain.full_text.apply(lambda x : count_soru_punctuation(x))



mtrain["total_word"] =  mtrain.full_text.apply(lambda x : count_words(x))
mtrain["total_p"] =  mtrain.full_text.apply(lambda x : count_p(x))

mtrain["total_int"] =  mtrain.full_text.apply(lambda x : count_integers(x))
mtrain["total_num"] =  mtrain.full_text.apply(lambda x : count_integers2(x))

mtrain["no_space_dot"] =  mtrain.full_text.apply(lambda x : no_space_dot(x))
mtrain["no_space_coma"] =  mtrain.full_text.apply(lambda x : no_space_coma(x))










mtest["essay_len"] =  mtest.full_text.apply(lambda x : len(x))
mtest["total_uppercase"] =  mtest.full_text.apply(lambda x : count_uppercase(x))

mtest["total_punctuation"] =  mtest.full_text.apply(lambda x : count_punctuation(x))
mtest["total_dot_punctuation"] =  mtest.full_text.apply(lambda x : count_dot_punctuation(x))
mtest["total_coma_punctuation"] =  mtest.full_text.apply(lambda x : count_coma_punctuation(x))
mtest["total_çç_punctuation"] =  mtest.full_text.apply(lambda x : count_çç_punctuation(x))
mtest["total_tırnak_punctuation"] =  mtest.full_text.apply(lambda x : count_tırnak_punctuation(x))
mtest["total_kısa_punctuation"] =  mtest.full_text.apply(lambda x : count_kısa_punctuation(x))
mtest["total_soru_punctuation"] =  mtest.full_text.apply(lambda x : count_soru_punctuation(x))



mtest["total_word"] =  mtest.full_text.apply(lambda x : count_words(x))
mtest["total_p"] =  mtest.full_text.apply(lambda x : count_p(x))

mtest["total_int"] =  mtest.full_text.apply(lambda x : count_integers(x))
mtest["total_num"] =  mtest.full_text.apply(lambda x : count_integers2(x))

mtest["no_space_dot"] =  mtest.full_text.apply(lambda x : no_space_dot(x))
mtest["no_space_coma"] =  mtest.full_text.apply(lambda x : no_space_coma(x))


## === cell 5
model_train = mtrain.drop(["full_text","score",],axis=1)
model_train = csr_matrix(model_train.values.astype(float))


"""X_train, X_test, y_train, y_test = train_test_split(model_train , mtrain.score, test_size=0.3, random_state=45)



reg_xgb = xgb.XGBRegressor()
reg_cat = CatBoostRegressor(silent=True)
reg_lgb = lgb.LGBMRegressor(verbose=-1)

# Voting Regressor ile modelleri birleştirme
reg = VotingRegressor(estimators=[('xgb', reg_lgb),
                                  ('xg2b', reg_xgb),
                                  ('xg3b', reg_cat)
        
                                 
                                 ])




print('Model eğitiliyor...')
reg.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
pred = reg.predict(X_test)

# Performans metriklerini hesaplama
score = cohen_kappa_score(y_test, pred.clip(1, 6).round(), weights="quadratic")
f1 = f1_score(y_test, pred.clip(1, 6).round(),average="weighted")

# Sonuçları yazdırma
print(f'Test cohen kappa score: {score}')
print(f'F1 score: {f1}')"""


## === cell 6
model_train = mtrain.drop(["full_text","score",],axis=1)
model_train = csr_matrix(model_train.values.astype(float))

model_test = mtest.drop(["full_text",],axis=1)
model_test = csr_matrix(model_test.values.astype(float))



"""reg_test = VotingRegressor(estimators=[('xgb', reg_lgb),
                                  ('xg2b', reg_xgb),
                                  ('xg3b', reg_cat)
        
                                ])


reg_test.fit(model_train,mtrain.score)
preds = reg_test.predict(model_test)"""


## === cell 7
"""import pickle
pickle_file = "voting_regressor_model.pkl"
with open(pickle_file, 'wb') as file:
    pickle.dump(reg_test, file)"""


## === cell 8
with open("/kaggle/input/modellllllll/voting_regressor_model.pkl", 'rb') as file:
    loaded_model = pickle.load(file)


## --- ERROR in cell 8, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2605745620.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0;32mwith[0m [0mopen[0m[0;34m([0m[0;34m"/kaggle/input/modellllllll/voting_regressor_model.pkl"[0m[0;34m,[0m [0;34m'rb'[0m[0;34m)[0m [0;32mas[0m [0mfile[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m     [0mloaded_model[0m [0;34m=[0m [0mpickle[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mfile[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '/kaggle/input/modellllllll/voting_regressor_model.pkl'

## === cell 9
preds = loaded_model.predict(model_test)
sub = pd.read_csv("/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv")
sub["score"] =  preds.clip(1, 6).round()

sub.to_csv("submission.csv",index=False)
