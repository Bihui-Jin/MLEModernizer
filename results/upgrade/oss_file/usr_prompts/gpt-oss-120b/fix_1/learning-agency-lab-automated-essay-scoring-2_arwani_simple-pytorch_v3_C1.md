# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

geopandas==0.14.4
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

0.7135653160395936

# 6. Current score

0.63187

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

stop_words = stopwords.words('english')

## === cell 2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

## === cell 3
df = pd.read_csv("/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv")
df.head()

## === cell 4
df['full_text'] = df['full_text'].apply(lambda x: x.lower())
df.head()

## === cell 5
df_test = pd.read_csv("/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv")
df_test.head()

## === cell 6
def get_data(df, mode='train'):
    df = df.copy()
    res = []
    texts = df.full_text.values
    stop_words_set = set(stop_words)
    punc = set(punctuation)
    for data in tqdm(df.values.tolist()):
        essay_id = data[0]
        text = data[1]
        score = 0 if mode != 'train' else data[2]
        txt = word_tokenize(text)
        lt = len(txt)
        sw = len([word for word in txt if word in stop_words_set])
        pct = len([word for word in txt if word in punc])
        res.append([essay_id, lt, sw, pct, score])
    ndf = pd.DataFrame(res, columns=['essay_id','lt','sw','pct','score'])
    ndf = ndf.set_index('essay_id')
    return ndf
        

## === cell 7
data = get_data(df)
data.head()

## === cell 8
data.corr()

## === cell 9
X = data[['lt','sw','pct']]
y = data['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape   

## === cell 10
numerical_pipeline = Pipeline([
    ("scaler", RobustScaler())
])
preprocessor = ColumnTransformer([
    ("numeric", numerical_pipeline, X_train.columns),
])
pipeline = Pipeline([
    ("prep", preprocessor),
    ("algo", GradientBoostingClassifier())
])

## === cell 11
pipeline.get_params()

## === cell 12
parameter = {
    'algo__tol' : [.0001, .0003, .001, .003, .01, .03, .1, .3]
}
model = GridSearchCV(pipeline, parameter, cv=3, n_jobs=-1, verbose=1)
model.fit(X_train, y_train)

## === cell 13
print(model.score(X_train, y_train), model.score(X_test, y_test), model.best_params_, model.best_score_)

## === cell 14
data_test = get_data(df_test, 'test')
data_test.head()

## === cell 15
X_pred = data_test[['lt', 'sw', 'pct']]

## === cell 16
y_pred = model.predict(X_pred)

## === cell 17
data_test['score'] = y_pred
data_test.head()

## === cell 18
data_test.drop(columns=['lt','sw','pct']).to_csv('submission.csv')
