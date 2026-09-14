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
Given questions and answers from various StackExchange properties, predict target values of 30 labels for each question-answer pair.

## Metric
Mean column-wise Spearman's correlation coefficient. The Spearman's rank correlation is computed for each target column, and the mean of these values is calculated for the submission score.

## Submission Format
For each qa_id in the test set, you must predict a probability for each target variable. The predictions should be in the range [0,1]. The file should contain a header and have the following format:

```
qa_id,question_asker_intent_understanding,...,answer_well_written
6,0.0,...,0.5
8,0.5,...,0.1
18,1.0,...,0.0
etc.
```

## Dataset
The list of 30 target labels are the same as the column names in the `sample_submission.csv` file. Target labels with the prefix `question_` relate to the `question_title` and/or `question_body` features in the data. Target labels with the prefix `answer_` relate to the `answer` feature.

Target labels are aggregated from multiple raters, and can have continuous values in the range `[0,1]`. Therefore, predictions must also be in that range.

- **train.csv** - the training data (target labels are the last 30 columns)
- **test.csv** - the test set (you must predict 30 labels for each test set row)
- **sample_submission.csv** - a sample submission file in the correct format; column names are the 30 target labels

# 2. Python version

3.8

# 3. Installed packages

category_encoders==2.7.0
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        input/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        working/
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
```

-> data/google-quest-challenge/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/google-quest-challenge/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/google-quest-challenge/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> data/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.28013

# 6. Current score

0.31958

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.35536) has done: 'Diagnosis: The crash occurs because `KFold` is instantiated with `random_state=42` while `shuffle` remains the default `False`. In scikit-learn 1.2+, this combination raises a `ValueError` since `random_state` has no effect without shuffling.  
Patch summary: In cell 21, set `shuffle=True` when creating `KFold` so `random_state` is valid and the splits remain deterministic across runs. This preserves the existing CV/training logic and output variables.  
Updated cells: Only cell 21 is modified.  
Compatibility notes for cell k+1: `trained_estimators` remains a list of fitted `PyTorch` estimators as before, so cell 22 run unchanged.  
Assumptions: Deterministic shuffled KFold is acceptable and intended given the explicit `random_state=42`.'
- What this solution (achieved 0.35358) has done: 'Your current score (0.35536) is higher than the target (0.28013), so to move *toward* the target with minimal disruption, I slightly reduce model strength rather than improve it. The smallest safe knob that preserves the same architecture/training loop is to reduce the TF‑IDF signal (shorter n-grams and fewer SVD components), which typically lowers Spearman performance while keeping the pipeline identical. I also explicitly clip the final blended outputs to [0,1] to match the competition requirement (this won’t improve much but prevents out-of-range edge cases). No changes are made to the PyTorch model/loss/training semantics beyond what is necessary for score-shaping.'
- What this solution (achieved 0.33858) has done: 'Your current score (0.35358) is higher than the target (0.28013), so to move toward the target with minimal disruption we should slightly weaken predictive signal while keeping the same pipeline, model, and training loop. The smallest “safe knob” here is to reduce the text representation strength by further shrinking TF‑IDF n-grams to unigrams only and reducing SVD dimensionality, which typically lowers Spearman without changing evaluation semantics. I keep the KFold shuffle fix, preserve the neural net and training procedure, and keep the final clipping and submission writing unchanged so the notebook still runs end-to-end and outputs a valid `submission.csv`. These changes should gently decrease correlation toward the target band rather than optimize upward.'
- What this solution (achieved 0.32819) has done: 'Your current score (0.33858) is higher than the target (0.28013), so to move *toward* the target with minimal risk we should slightly weaken predictive signal while keeping the same preprocessing, PyTorch model, training loops, and ranking blend. The smallest stable knob is to further reduce the text representation capacity by shrinking the SVD dimensionality (keeping unigram TF‑IDF), which typically lowers Spearman correlation without changing the pipeline’s semantics. I only adjust the `TruncatedSVD(n_components=...)` value and keep the existing clipping and submission writing intact to ensure a valid `submission.csv`. This should gently decrease performance toward the target band rather than optimize upward.'
- What this solution (achieved 0.31958) has done: 'Your current score (0.32819) is higher than the target (0.28013), so we should gently *decrease* performance with minimal disruption to move closer to the target band. The smallest, most stable knob (without touching the PyTorch model/training loop) is to further weaken the text representation by reducing the SVD dimensionality a bit more; this usually lowers Spearman correlation while keeping the same preprocessing/modeling semantics. I only change `TruncatedSVD(n_components=10)` to a smaller value and keep the rest identical, including the existing clipping and submission writing. This should reduce signal modestly and move the score toward the target rather than optimizing upward.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import warnings

warnings.simplefilter("ignore")

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train = pd.read_csv("../input/google-quest-challenge/train.csv", index_col="qa_id")
train.shape



## === cell 2
test = pd.read_csv("../input/google-quest-challenge/test.csv", index_col="qa_id")
test.shape



## === cell 3
train.head(3).T



## === cell 4
target_columns = [
    "question_asker_intent_understanding",
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written",
]



## === cell 5
y_train = train[target_columns].copy()
x_train = train.drop(target_columns, axis=1)
del train

x_test = test.copy()
del test



## === cell 6
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



## === cell 7
text_encoder = Pipeline(
    [
        ("Text-TF-IDF", TfidfVectorizer(ngram_range=(1, 1))),
        ("Text-SVD", TruncatedSVD(n_components=6)),
    ],
    verbose=True,
)



## === cell 8
from urllib.parse import urlparse
import re
from sklearn.preprocessing import FunctionTransformer
from category_encoders.one_hot import OneHotEncoder

before_dot = re.compile("^[^.]*")


def transform_url(x):
    return x.apply(lambda v: re.findall(before_dot, urlparse(v).netloc)[0])


url_encoder = Pipeline(
    [
        ("URL-transformer", FunctionTransformer(transform_url, validate=False)),
        ("URL-OHE", OneHotEncoder(drop_invariant=True)),
    ],
    verbose=True,
)



## === cell 9
from category_encoders.one_hot import OneHotEncoder

ohe = OneHotEncoder(cols="category", drop_invariant=True)



## === cell 10
from sklearn.preprocessing import StandardScaler


def count_words(data):
    out = pd.DataFrame(index=data.index)
    for column in data.columns:
        out[column] = data[column].str.split().str.len()
    return out


word_counter = Pipeline(
    [
        ("WordCounter-transformer", FunctionTransformer(count_words, validate=False)),
        ("WordCounter-std", StandardScaler()),
    ],
    verbose=True,
)



## === cell 11
preprocessor = ColumnTransformer(
    [
        ("Q-T", text_encoder, "question_title"),
        ("Q-B", text_encoder, "question_body"),
        ("A", text_encoder, "answer"),
        ("URL", url_encoder, "url"),
        ("Categoty", ohe, "category"),
        ("W-C", word_counter, ["question_body", "answer"]),
    ],
    verbose=True,
)



## === cell 12
x_train = preprocessor.fit_transform(x_train)



## === cell 13
x_test = preprocessor.transform(x_test)



## === cell 14
x_train.shape



## === cell 15
y_train = y_train.values



## === cell 16
import torch
import torch.nn as nn

from torch.nn import Sequential
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn.utils.weight_norm import weight_norm

from torch.nn import MSELoss
from torch.optim import Adam

import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class PyTorch:

    def __init__(self, in_features, out_features, n_epochs, patience):
        self.in_features = in_features
        self.out_features = out_features
        self.n_epochs = n_epochs
        self.patience = patience

    def init_model(self):

        self.model = Sequential(
            weight_norm(Linear(self.in_features, 128)),
            ReLU(),
            weight_norm(Linear(128, 128)),
            ReLU(),
            weight_norm(Linear(128, self.out_features)),
        )

        for t in self.model:
            if isinstance(t, Linear):
                nn.init.kaiming_normal_(t.weight_v)
                nn.init.kaiming_normal_(t.weight_g)
                nn.init.constant_(t.bias, 0)

        self.loss_func = MSELoss()

        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

    def fit(self, x_train, y_train, x_valid, y_valid):

        validate = (x_valid is not None) & (y_valid is not None)

        self.init_model()

        x_train_tensor = torch.as_tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32)

        if validate:
            x_valid_tensor = torch.as_tensor(x_valid, dtype=torch.float32)
            y_valid_tensor = torch.as_tensor(y_valid, dtype=torch.float32)

        min_loss = np.inf
        counter = 0

        for epoch in range(self.n_epochs):

            self.model.train()
            y_pred = self.model(x_train_tensor)
            loss = self.loss_func(y_pred, y_train_tensor)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            current_loss = loss.item()

            if validate:
                self.model.eval()
                with torch.no_grad():
                    current_loss = self.loss_func(
                        self.model(x_valid_tensor), y_valid_tensor
                    ).item()

            if current_loss < min_loss:
                min_loss = current_loss
                counter = 0
            else:
                counter += 1
                if counter >= self.patience:
                    break

    def predict(self, x):
        x_tenson = torch.as_tensor(x, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            return self.model(x_tenson).numpy()




## === cell 17
from scipy.stats import spearmanr


def mean_spearmanr_correlation_score(y_true, y_pred):
    return np.mean(
        [
            spearmanr(y_pred[:, idx], y_true[:, idx]).correlation
            for idx in range(len(target_columns))
        ]
    )




## === cell 18
pytorch_params = {
    "in_features": x_train.shape[1],
    "out_features": y_train.shape[1],
    "n_epochs": 2500,
    "patience": 5,
}



## === cell 19
trained_estimators = []



## === cell 20
estimator = PyTorch(**pytorch_params)
estimator.fit(x_train, y_train, None, None)
trained_estimators.append(estimator)



## === cell 21
from sklearn.model_selection import KFold
import math

n_splits = 10

scores = []

cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
for train_idx, valid_idx in cv.split(x_train, y_train):

    x_train_train = x_train[train_idx]
    y_train_train = y_train[train_idx]
    x_train_valid = x_train[valid_idx]
    y_train_valid = y_train[valid_idx]

    estimator = PyTorch(**pytorch_params)
    estimator.fit(x_train_train, y_train_train, x_train_valid, y_train_valid)

    oof_part = estimator.predict(x_train_valid)
    score = mean_spearmanr_correlation_score(y_train_valid, oof_part)
    print("Score:", score)

    if not math.isnan(score):
        trained_estimators.append(estimator)
        scores.append(score)

print("Mean score:", np.mean(scores))



## === cell 22
y_pred = []
for estimator in trained_estimators:
    y_pred.append(estimator.predict(x_test))



## === cell 23
from scipy.stats import rankdata


def blend_by_ranking(data, weights):
    out = np.zeros(data.shape[0])
    for idx, column in enumerate(data.columns):
        out += weights[idx] * rankdata(data[column].values)
    out /= np.max(out)
    return out




## === cell 24
submission = pd.read_csv(
    "../input/google-quest-challenge/sample_submission.csv", index_col="qa_id"
)

out = pd.DataFrame(index=submission.index)
for column_idx, column in enumerate(target_columns):

    column_data = pd.DataFrame(index=submission.index)
    for prediction_idx, prediction in enumerate(y_pred):
        column_data[str(prediction_idx)] = prediction[:, column_idx]

    out[column] = blend_by_ranking(column_data, np.ones(column_data.shape[1]))



## === cell 25
out.head()



## === cell 26
out = out.clip(0.0, 1.0)



## === cell 27
out.to_csv("submission.csv")
