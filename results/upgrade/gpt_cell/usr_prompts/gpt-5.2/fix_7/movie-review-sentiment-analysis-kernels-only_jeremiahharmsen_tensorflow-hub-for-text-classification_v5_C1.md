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

3.7

# 2. Installed packages

No external packages required in the script and installed.

# 3. Data file paths

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

# 4. Code solution

## === cell 0
tf = None
hub = None

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import zipfile

from sklearn import model_selection


## === cell 1
SENTIMENT_LABELS = [
    "negative", "somewhat negative", "neutral", "somewhat positive", "positive"
]

def add_readable_labels_column(df, sentiment_value_column):
  df["SentimentLabel"] = df[sentiment_value_column].replace(
      range(5), SENTIMENT_LABELS)

def get_data(validation_set_ratio=0.1):
  train_df = pd.read_csv("../input/train.tsv", sep="\t")
  test_df = pd.read_csv("../input/test.tsv", sep="\t")

  add_readable_labels_column(train_df, "Sentiment")

  train_indices, validation_indices = model_selection.train_test_split(
      np.unique(train_df["SentenceId"]),
      test_size=validation_set_ratio,
      random_state=0)

  validation_df = train_df[train_df["SentenceId"].isin(validation_indices)]
  train_df = train_df[train_df["SentenceId"].isin(train_indices)]
  print("Split the training data into %d training and %d validation examples." %
        (len(train_df), len(validation_df)))

  return train_df, validation_df, test_df


train_df, validation_df, test_df = get_data()
train_df.head()


## === cell 2
if tf is None:

    class _StubInputs:
        @staticmethod
        def pandas_input_fn(x, y=None, num_epochs=1, shuffle=False):
            def _input_fn():
                return x, y

            return _input_fn

    class _StubEstimator:
        def __init__(self):
            self._majority_class = 2  # default neutral
            self._n_classes = 5

        def train(self, input_fn, steps=None):
            x, y = input_fn()
            if y is not None and len(y) > 0:
                vc = pd.Series(y).value_counts()
                self._majority_class = int(vc.index[0])
            return None

        def evaluate(self, input_fn):
            x, y = input_fn()
            if y is None or len(y) == 0:
                return {"accuracy": 0.0}
            y = np.asarray(y)
            acc = float(np.mean(y == self._majority_class))
            return {"accuracy": acc}

        def predict(self, input_fn):
            x, y = input_fn()
            n = len(x) if hasattr(x, "__len__") else 0
            for _ in range(n):
                probs = np.zeros(self._n_classes, dtype=float)
                probs[self._majority_class] = 1.0
                yield {
                    "class_ids": np.array([self._majority_class], dtype=np.int64),
                    "probabilities": probs,
                }

    class _StubEstimatorModule:
        inputs = _StubInputs()

        class DNNClassifier(_StubEstimator):
            def __init__(
                self, hidden_units, feature_columns, n_classes, optimizer=None
            ):
                super().__init__()
                self._n_classes = int(n_classes)

    class _StubTrainModule:
        class AdagradOptimizer:
            def __init__(self, learning_rate=0.1):
                self.learning_rate = learning_rate

    class _StubTF:
        estimator = _StubEstimatorModule()
        train = _StubTrainModule()

    tf = _StubTF()

if hub is None:

    class _StubHub:
        @staticmethod
        def text_embedding_column(key, module_spec):
            return {"key": key, "module_spec": module_spec}

    hub = _StubHub()

train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["Sentiment"], num_epochs=None, shuffle=True
)

predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["Sentiment"], shuffle=False
)
predict_validation_input_fn = tf.estimator.inputs.pandas_input_fn(
    validation_df, validation_df["Sentiment"], shuffle=False
)
predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(test_df, shuffle=False)

embedded_text_feature_column = hub.text_embedding_column(
    key="Phrase", module_spec="https://tfhub.dev/google/nnlm-en-dim128/1"
)

estimator = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=5,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003),
)

estimator.train(input_fn=train_input_fn, steps=10000)


## === cell 3
train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
validation_eval_result = estimator.evaluate(input_fn=predict_validation_input_fn)

print("Training set accuracy: {accuracy}".format(**train_eval_result))
print("Validation set accuracy: {accuracy}".format(**validation_eval_result))


## === cell 4
def get_predictions(estimator, input_fn):
    return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]


if hasattr(tf, "Graph") and hasattr(tf, "confusion_matrix") and hasattr(tf, "Session"):
    with tf.Graph().as_default():
        cm = tf.confusion_matrix(
            train_df["Sentiment"], get_predictions(estimator, predict_train_input_fn)
        )
        with tf.Session() as session:
            cm_out = session.run(cm)
else:
    y_true = np.asarray(train_df["Sentiment"], dtype=int)
    y_pred = np.asarray(get_predictions(estimator, predict_train_input_fn), dtype=int)
    n_classes = 5
    cm_out = np.bincount(
        y_true * n_classes + y_pred, minlength=n_classes * n_classes
    ).reshape(n_classes, n_classes)

cm_out = cm_out.astype(float) / cm_out.sum(axis=1)[:, np.newaxis]

sns.heatmap(
    cm_out, annot=True, xticklabels=SENTIMENT_LABELS, yticklabels=SENTIMENT_LABELS
)
plt.xlabel("Predicted")
plt.ylabel("True")


## === cell 5
test_df["Predictions"] = get_predictions(estimator, predict_test_input_fn)
test_df.to_csv(
    tf.gfile.GFile("submission.csv", "w"),
    columns=["PhraseId", "Predictions"],
    header=["PhraseId", "Sentiment"],
    index=False)


## --- ERROR in cell 5, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1461095577.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0mtest_df[0m[0;34m[[0m[0;34m"Predictions"[0m[0;34m][0m [0;34m=[0m [0mget_predictions[0m[0;34m([0m[0mestimator[0m[0;34m,[0m [0mpredict_test_input_fn[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m test_df.to_csv(
[0;32m----> 3[0;31m     [0mtf[0m[0;34m.[0m[0mgfile[0m[0;34m.[0m[0mGFile[0m[0;34m([0m[0;34m"submission.csv"[0m[0;34m,[0m [0;34m"w"[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m     [0mcolumns[0m[0;34m=[0m[0;34m[[0m[0;34m"PhraseId"[0m[0;34m,[0m [0;34m"Predictions"[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m     [0mheader[0m[0;34m=[0m[0;34m[[0m[0;34m"PhraseId"[0m[0;34m,[0m [0;34m"Sentiment"[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: '_StubTF' object has no attribute 'gfile'
