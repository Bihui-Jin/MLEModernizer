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

3.7

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0
wordcloud==1.9.4

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

0.55514

# 6. Current score

0.48334

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.75816) has done: 'I fix the Keras import/runtime issues caused by the Kaggle environment having `keras==3.x` alongside `tf_keras`, by switching the model/Tokenizer utilities to `tf_keras` consistently (this resolves the protobuf `GetPrototype` crash and the missing `Tokenizer/to_categorical/EarlyStopping` symbols). I also make the data path robust by reading from the existing `/kaggle/input/...` location (with a safe fallback), so the notebook always finds `train.csv` and `test.csv`. Finally, I keep your exact model architecture/training loop intact and ensure the submission is written as `my_submission.csv` with the required columns `id,EAP,HPL,MWS`.'
- What this solution (achieved 0.48334) has done: 'I fix the crash in the Keras/TensorFlow import stack that triggers `MessageFactory.GetPrototype` by avoiding `tf_keras` entirely and using scikit-learn’s `LogisticRegression` with a `TfidfVectorizer`, which is stable in this environment and well-suited to this log-loss text task. This keeps the overall pipeline concept the same (tokenize text → train a classifier → output class probabilities) while removing the protobuf-dependent deep-learning runtime that is currently blocking execution. I also ensure the predicted probability columns are aligned to `EAP,HPL,MWS` and that a valid `my_submission.csv` is always written with the required header. This change should substantially reduce log-loss from 0.75816 toward the 0.55514 target.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

if os.path.exists("/kaggle/input"):
    print(os.listdir("/kaggle/input")[:20])
else:
    print(os.listdir("../input")[:20])



## === cell 1
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression



## === cell 2
base_dir = "/kaggle/input/spooky-author-identification"
if not os.path.exists(base_dir):
    base_dir = "/kaggle/input"

train_path = os.path.join(base_dir, "train.csv")
test_path = os.path.join(base_dir, "test.csv")

data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)



## === cell 3
data.head()



## === cell 4
data.author.value_counts().plot(kind="bar")



## === cell 5
data_length = data.text.apply(len)
data_length.head()



## === cell 6
plt.figure(figsize=(12, 5))
plt.hist(data_length, bins=20, range=[0, 500], color="r", alpha=0.3)
plt.show()



## === cell 7
data_split_length = data.text.apply(lambda x: len(x.split(" ")))
data_split_length.head()



## === cell 8
plt.figure(figsize=(12, 5))
plt.hist(data_split_length, bins=10, range=[0, 100], color="g", alpha=0.5)
plt.show()



## === cell 9
print("data_length max : ", np.max(data_length))
print("data_length min : ", np.min(data_length))
print("data_length mean : ", np.mean(data_length))
print("data_length 75% : ", np.percentile(data_length, 75))
print("data_length 90% : ", np.percentile(data_length, 90))



## === cell 10
print("data_split_length max : ", np.max(data_split_length))
print("data_split_length min : ", np.min(data_split_length))
print("data_split_length mean : ", np.mean(data_split_length))
print("data_split_length 75% : ", np.percentile(data_split_length, 75))
print("data_split_length 90% : ", np.percentile(data_split_length, 90))



## === cell 11
cloud = WordCloud(width=400, height=200).generate(" ".join(data.text))
plt.figure(figsize=(12, 5))
plt.imshow(cloud)
plt.axis("off")



## === cell 12
cloud = WordCloud(width=400, height=200).generate(
    " ".join(data[data["author"] == "HPL"]["text"])
)
plt.figure(figsize=(12, 5))
plt.imshow(cloud)
plt.axis("off")



## === cell 13
cloud = WordCloud(width=400, height=200).generate(
    " ".join(data[data["author"] == "MWS"]["text"])
)
plt.figure(figsize=(12, 5))
plt.imshow(cloud)
plt.axis("off")



## === cell 14
cloud = WordCloud(width=400, height=200).generate(
    " ".join(data[data["author"] == "EAP"]["text"])
)
plt.figure(figsize=(12, 5))
plt.imshow(cloud)
plt.axis("off")



## === cell 15
le = LabelEncoder()
le.fit(data.author)
y = le.transform(data.author)



## === cell 16
y[:10]



## === cell 17
y_onehot = np.eye(len(le.classes_), dtype=np.float32)[y]



## === cell 18
y_onehot[:2], le.classes_



## === cell 19
tfidf = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=50000,
    sublinear_tf=True,
)

X = tfidf.fit_transform(data["text"].astype(str))
X_test = tfidf.transform(test_data["text"].astype(str))

print("TF-IDF shapes:", X.shape, X_test.shape)



## === cell 20
clf = LogisticRegression(
    solver="lbfgs",
    multi_class="multinomial",
    max_iter=300,
    C=4.0,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

clf.fit(X, y)



## === cell 21
proba = clf.predict_proba(X_test)

class_order = list(le.classes_)
desired_order = ["EAP", "HPL", "MWS"]

if class_order != desired_order:
    idx_map = [class_order.index(c) for c in desired_order]
    proba = proba[:, idx_map]

results = pd.DataFrame(proba, columns=desired_order)
results.insert(0, "id", test_data["id"].values)

results.head()



## === cell 22
assert results.shape[0] == test_data.shape[0]
assert list(results.columns) == ["id", "EAP", "HPL", "MWS"]
assert results.isna().sum().sum() == 0

row_sums = results[["EAP", "HPL", "MWS"]].sum(axis=1).values
print(
    "Row sum stats (should be ~1.0):", row_sums.min(), row_sums.mean(), row_sums.max()
)



## === cell 23
results.to_csv("my_submission.csv", index=False)
print(
    "Wrote submission:", os.path.abspath("my_submission.csv"), "shape:", results.shape
)
print(results.head())
