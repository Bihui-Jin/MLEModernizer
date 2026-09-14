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

No external packages required in the script and installed.

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

0.62585

# 6. Current score

0.51314

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.51314) has done: 'Diagnosis: Cell 5 crashes because the notebook runs with the provided `_StubTF` (since TensorFlow isn’t installed), and that stub does not implement `tf.gfile.GFile`. The code uses `tf.gfile.GFile("submission.csv","w")` as the file handle for `to_csv`, causing an `AttributeError`. The simplest deterministic fix is to fall back to Python’s built-in `open()` when `tf.gfile.GFile` is unavailable, while keeping the output filename and CSV schema unchanged.

Patch summary: Modify only cell 5 to select an output file handle using `tf.gfile.GFile` if present, otherwise `open()`. Keep `to_csv` arguments (columns/header/index) identical to preserve submission format and semantics.

Updated cells: Only cell 5 is changed below.

Compatibility notes for cell k+1: No variables or interfaces used by the next cell are changed; `test_df` and `submission.csv` are produced exactly as before.

Assumptions: Writing to the current working directory is permitted; TensorFlow is not available (hence the stub), so `open()` is the correct fallback.'

# 9. Code solution

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

_gfile = getattr(getattr(tf, "gfile", None), "GFile", None)
_open_fn = _gfile if _gfile is not None else open

with _open_fn("submission.csv", "w") as f:
    test_df.to_csv(
        f,
        columns=["PhraseId", "Predictions"],
        header=["PhraseId", "Sentiment"],
        index=False,
    )
