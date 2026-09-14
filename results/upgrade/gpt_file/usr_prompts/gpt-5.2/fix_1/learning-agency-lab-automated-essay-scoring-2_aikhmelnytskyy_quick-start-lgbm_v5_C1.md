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

geopandas==0.14.4
imbalanced-learn==0.13.0
lightgbm==4.6.0
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
sklearn-pandas==2.2.0
ydata-profiling==4.17.0

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

0.741

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import nltk
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import random
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB,MultinomialNB,ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import Pipeline
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
nltk.download('wordnet')


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2082754087.py in <cell line: 0>()
     16 from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
     17 from sklearn.pipeline import Pipeline
---> 18 from imblearn.ensemble import BalancedBaggingClassifier
     19 from sklearn.feature_selection import SelectFromModel
     20 import matplotlib.pyplot as plt

/usr/local/lib/python3.11/dist-packages/imblearn/__init__.py in <module>
     50     # process, as it may not be compiled yet
     51 else:
---> 52     from . import (
     53         combine,
     54         ensemble,

/usr/local/lib/python3.11/dist-packages/imblearn/combine/__init__.py in <module>
      3 """
      4 
----> 5 from ._smote_enn import SMOTEENN
      6 from ._smote_tomek import SMOTETomek
      7 

/usr/local/lib/python3.11/dist-packages/imblearn/combine/_smote_enn.py in <module>
     10 from sklearn.utils import check_X_y
     11 
---> 12 from ..base import BaseSampler
     13 from ..over_sampling import SMOTE
     14 from ..over_sampling.base import BaseOverSampler

/usr/local/lib/python3.11/dist-packages/imblearn/base.py in <module>
     10 from sklearn.base import BaseEstimator, OneToOneFeatureMixin
     11 from sklearn.preprocessing import label_binarize
---> 12 from sklearn.utils._metadata_requests import METHODS
     13 from sklearn.utils.multiclass import check_classification_targets
     14 

ModuleNotFoundError: No module named 'sklearn.utils._metadata_requests'

## === cell 1
train = pd.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv')

print(train.head())


## === cell 2
from ydata_profiling import ProfileReport

profile = ProfileReport(train, title="Pandas Profiling Report")

profile


## === cell 3
%%time

from nltk.stem import WordNetLemmatizer
import re
def remove_punctuations(data):
    punct_tag=re.compile(r'[^\w\s]')
    data=punct_tag.sub(r'',data)
    return data

def remove_html(data):
    html_tag=re.compile(r'<.*?>')
    data=html_tag.sub(r'',data)
    return data

def remove_url(data):
    url_clean= re.compile(r"https://\S+|www\.\S+")
    data=url_clean.sub(r'',data)
    return data

def remove_emoji(data):
    emoji_clean= re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    data=emoji_clean.sub(r'',data)
    url_clean= re.compile(r"https://\S+|www\.\S+")
    data=url_clean.sub(r'',data)
    return data.lower()
def lemma_traincorpus(data):
    lemmatizer=WordNetLemmatizer()
    out_data=""
    for words in data:
        out_data+= lemmatizer.lemmatize(words)
    return out_data

def tfidf(data):
    tfidfv = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=150000)
    fit_data_tfidf=tfidfv.fit_transform(data)
    return fit_data_tfidf

DESCRIPTION=train['full_text']
DESCRIPTION=DESCRIPTION.apply(lambda z: remove_punctuations(z))
DESCRIPTION=DESCRIPTION.apply(lambda z: remove_html(z))
DESCRIPTION=DESCRIPTION.apply(lambda z: remove_url(z))
DESCRIPTION=DESCRIPTION.apply(lambda z: remove_emoji(z))
train['text']=DESCRIPTION
train


## === cell 4
X = train['text'].astype(str)

y = train['score'].astype(int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    stratify=y,  # Ensures that the train and test sets have approximately the same proportion of each target class
    test_size=0.33,  # Setting the test set size to 33% of the original dataset
    random_state=0  # Setting a random state for reproducibility
)


## === cell 5
model = Pipeline([
    ('Vectorizer', TfidfVectorizer()),  # TF-IDF vectorizer for text feature extraction
    ('feature_selection', SelectFromModel(ExtraTreesClassifier())),  # Feature selection using ExtraTreesClassifier
    ('clf', LinearSVC())  # Linear Support Vector Classifier
])

model.fit(X_train, y_train)

predictions = model.predict(X_train)
f1 = f1_score(y_train, predictions, average='weighted')  # 'weighted' or 'macro', depending on your needs
print(f'F1 score: {f1}')
score = cohen_kappa_score(y_train, predictions)  # 'weighted' or 'macro', depending on your needs
print(f'cohen kappa score: {score}')


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3331698730.py in <cell line: 0>()
      2 model = Pipeline([
      3     ('Vectorizer', TfidfVectorizer()),  # TF-IDF vectorizer for text feature extraction
----> 4     ('feature_selection', SelectFromModel(ExtraTreesClassifier())),  # Feature selection using ExtraTreesClassifier
      5     ('clf', LinearSVC())  # Linear Support Vector Classifier
      6 ])

NameError: name 'SelectFromModel' is not defined

## === cell 6
predictions = model.predict(X_test)

cm = confusion_matrix(y_test, predictions, labels=model.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model.classes_)
disp.plot()
plt.show()

f1 = f1_score(y_test, predictions, average='weighted')  # 'weighted' or 'macro', depending on your needs
print(f'F1 score: {f1}')
score = cohen_kappa_score(y_test, predictions)  # 'weighted' or 'macro', depending on your needs
print(f'cohen kappa score: {score}')


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/146790262.py in <cell line: 0>()
      1 # Making predictions on the test data
----> 2 predictions = model.predict(X_test)
      3 
      4 # Calculating the confusion matrix
      5 cm = confusion_matrix(y_test, predictions, labels=model.classes_)

NameError: name 'model' is not defined

## === cell 7
model.steps.pop()

new_pipeline = Pipeline(model.steps)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1233835421.py in <cell line: 0>()
      1 # Remove the classifier from the original pipeline
----> 2 model.steps.pop()
      3 
      4 # Create a new pipeline using the remaining steps
      5 new_pipeline = Pipeline(model.steps)

NameError: name 'model' is not defined

## === cell 9
import lightgbm as lgb

params = {'n_estimators': 195,
 'num_leaves': 6,
 'min_child_samples': 3,
 'learning_rate': 0.06518970520093895,
 'max_bin': 511,
 'colsample_bytree': 0.9699010403795221,
 'reg_alpha': 0.028218191448367104,
 'reg_lambda': 0.855160569748305}


X_train_vector=new_pipeline.transform(X_train)
X_test_vector=new_pipeline.transform(X_test)
lgb_model = lgb.LGBMClassifier(**params)
lgb_model.fit(X_train_vector, y_train)

predictions = lgb_model.predict(X_train_vector)
f1 = f1_score(y_train, predictions, average='weighted')  # 'weighted' or 'macro', depending on your needs
print(f'F1 score: {f1}')
score = cohen_kappa_score(y_train, predictions)  # 'weighted' or 'macro', depending on your needs
print(f'cohen kappa score: {score}')


predictions = lgb_model.predict(X_test_vector)

cm = confusion_matrix(y_test, predictions, labels=[x for x in range(1,7)])

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[x for x in range(1,7)])
disp.plot()
plt.show()

f1 = f1_score(y_test, predictions, average='weighted')  # 'weighted' or 'macro', depending on your needs
print(f'F1 score: {f1}')
score = cohen_kappa_score(y_test, predictions)  # 'weighted' or 'macro', depending on your needs
print(f'cohen kappa score: {score}')


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/903925090.py in <cell line: 0>()
     11 
     12 
---> 13 X_train_vector=new_pipeline.transform(X_train)
     14 X_test_vector=new_pipeline.transform(X_test)
     15 lgb_model = lgb.LGBMClassifier(**params)

NameError: name 'new_pipeline' is not defined

## === cell 10
"""
# Define the number of splits for cross-validation
n_splits = 5

# Initialize StratifiedKFold with the specified number of splits
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

# Lists to store scores
f1_scores = []
kappa_scores = []
models = []
predictions = []
# Loop through each fold of the cross-validation
for train_index, test_index in skf.split(X, y):
    # Split the data into training and testing sets for this fold
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    
    # Fit the model on the training data for this fold
    
    model = Pipeline([
    ('Vectorizer', TfidfVectorizer()),  # TF-IDF vectorizer for text feature extraction
    ('feature_selection', SelectFromModel(ExtraTreesClassifier())),  # Feature selection using ExtraTreesClassifier
    ('clf', LinearSVC())  # Linear Support Vector Classifier
                    ])
    model.fit(X_train_fold, y_train_fold)
    models.append(model)
    # Make predictions on the test data for this fold
    predictions_fold = model.predict(X_test_fold)
    predictions.append(predictions_fold)
    # Calculate and store the F1 score for this fold
    f1_fold = f1_score(y_test_fold, predictions_fold, average='weighted')
    f1_scores.append(f1_fold)
    
    # Calculate and store the Cohen's kappa score for this fold
    kappa_fold = cohen_kappa_score(y_test_fold, predictions_fold)
    kappa_scores.append(kappa_fold)

    
    
# Calculate the mean scores across all folds
mean_f1_score = np.mean(f1_scores)
mean_kappa_score = np.mean(kappa_scores)

# Print the mean scores
print(f'Mean F1 score across {n_splits} folds: {mean_f1_score}')
print(f'Mean Cohen kappa score across {n_splits} folds: {mean_kappa_score}')
"""


## === cell 11
test=pd.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv')
DESCRIPTION=test['full_text']
DESCRIPTION=DESCRIPTION.apply(lambda z: remove_punctuations(z))
DESCRIPTION=DESCRIPTION.apply(lambda z: remove_html(z))
DESCRIPTION=DESCRIPTION.apply(lambda z: remove_url(z))
DESCRIPTION=DESCRIPTION.apply(lambda z: remove_emoji(z))
test['text']=DESCRIPTION
test


## === cell 12
test_vector=new_pipeline.transform(test['text'])
predictions=lgb_model.predict(test_vector)
predictions


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/151666565.py in <cell line: 0>()
----> 1 test_vector=new_pipeline.transform(test['text'])
      2 predictions=lgb_model.predict(test_vector)
      3 predictions

NameError: name 'new_pipeline' is not defined

## === cell 13
"""
probabilities = []

# Iterate over each model in the list of models
for model in models:
    # Make predictions using the current model
    print(model.predict(test['text']))
    
    # Compute decision scores for the current model
    decision_scores = model.decision_function(test['text'])
    
    # Convert decision scores to probabilities using the sigmoid function
    proba = 1 / (1 + np.exp(-decision_scores))
    # Append the probabilities to the list
    probabilities.append(proba)

# Compute the average probabilities across all models
probabilities = np.mean(probabilities, axis=0)

# Determine the predicted class by selecting the class with the highest probability
# Add 1 to the index to match the class labels (assuming classes start from 1)
predictions = np.argmax(probabilities, axis=1) + 1

# Print the predictions
print(predictions)
"""


## === cell 14
submission=pd.read_csv("/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv")
submission['score']=predictions
submission.to_csv("submission.csv",index=None)
display(submission.head())


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3310277098.py in <cell line: 0>()
      1 submission=pd.read_csv("/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv")
----> 2 submission['score']=predictions
      3 submission.to_csv("submission.csv",index=None)
      4 display(submission.head())

NameError: name 'predictions' is not defined
