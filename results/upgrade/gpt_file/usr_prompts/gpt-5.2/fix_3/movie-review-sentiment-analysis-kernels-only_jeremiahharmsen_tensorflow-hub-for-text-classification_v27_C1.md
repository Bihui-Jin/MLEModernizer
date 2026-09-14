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

3.7

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

0.65223

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

try:
    import tensorflow_estimator as tf_estimator  # noqa: F401

    _ESTIMATOR = tf_estimator.estimator
except Exception:
    _ESTIMATOR = tf.estimator

np.random.seed(0)
tf.set_random_seed(0)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
SENTIMENT_LABELS = [
    "negative",
    "somewhat negative",
    "neutral",
    "somewhat positive",
    "positive",
]


def add_readable_labels_column(df, sentiment_value_column):
    df["SentimentLabel"] = df[sentiment_value_column].replace(
        range(5), SENTIMENT_LABELS
    )


def get_data(validation_set_ratio=0.1):
    train_path_candidates = [
        "../input/train.tsv",
        "/kaggle/input/train.tsv",
        "/kaggle/data/train.tsv",
        "/kaggle/input/movie-review-sentiment-analysis-kernels-only/train.tsv",
        "/kaggle/data/movie-review-sentiment-analysis-kernels-only/train.tsv",
    ]
    test_path_candidates = [
        "../input/test.tsv",
        "/kaggle/input/test.tsv",
        "/kaggle/data/test.tsv",
        "/kaggle/input/movie-review-sentiment-analysis-kernels-only/test.tsv",
        "/kaggle/data/movie-review-sentiment-analysis-kernels-only/test.tsv",
    ]

    train_path = next((p for p in train_path_candidates if os.path.exists(p)), None)
    test_path = next((p for p in test_path_candidates if os.path.exists(p)), None)
    if train_path is None or test_path is None:
        raise FileNotFoundError(
            f"Could not find train/test TSV. Looked for train in {train_path_candidates} "
            f"and test in {test_path_candidates}."
        )

    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")

    add_readable_labels_column(train_df, "Sentiment")

    train_indices, validation_indices = model_selection.train_test_split(
        np.unique(train_df["SentenceId"]),
        test_size=validation_set_ratio,
        random_state=0,
    )

    validation_df = train_df[train_df["SentenceId"].isin(validation_indices)].copy()
    train_df = train_df[train_df["SentenceId"].isin(train_indices)].copy()
    print(
        "Split the training data into %d training and %d validation examples."
        % (len(train_df), len(validation_df))
    )

    return train_df, validation_df, test_df


train_df, validation_df, test_df = get_data()
train_df.head()



## === cell 2

MAX_FEATURES = 20000  # conservative for speed/memory within 600s
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=MAX_FEATURES,
    min_df=2,
)

X_train = vectorizer.fit_transform(train_df["Phrase"].astype(str).values)
X_val = vectorizer.transform(validation_df["Phrase"].astype(str).values)
X_test = vectorizer.transform(test_df["Phrase"].astype(str).values)

X_train_dense = X_train.astype(np.float32).toarray()
X_val_dense = X_val.astype(np.float32).toarray()
X_test_dense = X_test.astype(np.float32).toarray()

y_train = train_df["Sentiment"].astype(np.int32).values
y_val = validation_df["Sentiment"].astype(np.int32).values


def make_input_fn(X, y=None, num_epochs=1, shuffle=False, batch_size=256):
    def _fn():
        features = {"x": X}
        if y is None:
            return _ESTIMATOR.inputs.numpy_input_fn(
                x=features,
                num_epochs=num_epochs,
                shuffle=shuffle,
                batch_size=batch_size,
            )()
        return _ESTIMATOR.inputs.numpy_input_fn(
            x=features,
            y=y,
            num_epochs=num_epochs,
            shuffle=shuffle,
            batch_size=batch_size,
        )()

    return _fn


train_input_fn = make_input_fn(
    X_train_dense, y_train, num_epochs=None, shuffle=True, batch_size=256
)
predict_train_input_fn = make_input_fn(
    X_train_dense, y_train, num_epochs=1, shuffle=False, batch_size=256
)
predict_validation_input_fn = make_input_fn(
    X_val_dense, y_val, num_epochs=1, shuffle=False, batch_size=256
)
predict_test_input_fn = make_input_fn(
    X_test_dense, y=None, num_epochs=1, shuffle=False, batch_size=256
)

feature_columns = [
    tf.feature_column.numeric_column("x", shape=(X_train_dense.shape[1],))
]

run_config = _ESTIMATOR.RunConfig(keep_checkpoint_max=0)

estimator = _ESTIMATOR.DNNClassifier(
    hidden_units=[250, 50],
    feature_columns=feature_columns,
    n_classes=5,
    config=run_config,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003),
)

estimator.train(input_fn=train_input_fn, steps=10000)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1747591663.py in <cell line: 0>()
     65 ]
     66 
---> 67 run_config = _ESTIMATOR.RunConfig(keep_checkpoint_max=0)
     68 
     69 estimator = _ESTIMATOR.DNNClassifier(

NameError: name '_ESTIMATOR' is not defined

## === cell 3
train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
validation_eval_result = estimator.evaluate(input_fn=predict_validation_input_fn)

print("Training set accuracy: {accuracy}".format(**train_eval_result))
print("Validation set accuracy: {accuracy}".format(**validation_eval_result))




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2855871204.py in <cell line: 0>()
----> 1 train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
      2 validation_eval_result = estimator.evaluate(input_fn=predict_validation_input_fn)
      3 
      4 print("Training set accuracy: {accuracy}".format(**train_eval_result))
      5 print("Validation set accuracy: {accuracy}".format(**validation_eval_result))

NameError: name 'estimator' is not defined

## === cell 4
def get_predictions(estimator, input_fn):
    return [int(x["class_ids"][0]) for x in estimator.predict(input_fn=input_fn)]


with tf.Graph().as_default():
    cm = tf.math.confusion_matrix(
        labels=y_train,
        predictions=get_predictions(estimator, predict_train_input_fn),
        num_classes=5,
    )
    with tf.Session() as session:
        cm_out = session.run(cm)

cm_out = cm_out.astype(float)
cm_out = cm_out / cm_out.sum(axis=1, keepdims=True)

sns.heatmap(
    cm_out, annot=True, xticklabels=SENTIMENT_LABELS, yticklabels=SENTIMENT_LABELS
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4094663885.py in <cell line: 0>()
      6     cm = tf.math.confusion_matrix(
      7         labels=y_train,
----> 8         predictions=get_predictions(estimator, predict_train_input_fn),
      9         num_classes=5,
     10     )

NameError: name 'estimator' is not defined

## === cell 5
test_pred = get_predictions(estimator, predict_test_input_fn)

submission = pd.DataFrame(
    {
        "PhraseId": test_df["PhraseId"].astype(int).values,
        "Sentiment": np.array(test_pred, dtype=int),
    }
)

assert len(submission) == len(test_df), "Submission length mismatch with test set."

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1928604563.py in <cell line: 0>()
----> 1 test_pred = get_predictions(estimator, predict_test_input_fn)
      2 
      3 submission = pd.DataFrame(
      4     {
      5         "PhraseId": test_df["PhraseId"].astype(int).values,

NameError: name 'estimator' is not defined
