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
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import google.protobuf  # noqa: F401

from google.protobuf import message_factory as _message_factory

if not hasattr(_message_factory.MessageFactory, "GetPrototype"):

    def _GetPrototype(self, descriptor):
        return self.GetMessageClass(descriptor)

    _message_factory.MessageFactory.GetPrototype = _GetPrototype

import tensorflow as tf
import tensorflow_hub as hub
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
try:
    import tensorflow_estimator as estimator_api  # type: ignore
except Exception:
    estimator_api = getattr(tf, "estimator", None)
    if estimator_api is None:
        estimator_api = getattr(getattr(tf, "compat", None), "v1", None)
        estimator_api = getattr(estimator_api, "estimator", None)
    if estimator_api is None:
        raise ImportError(
            "No TensorFlow Estimator API found (tensorflow_estimator, tf.estimator, "
            "or tf.compat.v1.estimator)."
        )

train_input_fn = estimator_api.inputs.pandas_input_fn(
    train_df, train_df["Sentiment"], num_epochs=None, shuffle=True
)

predict_train_input_fn = estimator_api.inputs.pandas_input_fn(
    train_df, train_df["Sentiment"], shuffle=False
)
predict_validation_input_fn = estimator_api.inputs.pandas_input_fn(
    validation_df, validation_df["Sentiment"], shuffle=False
)
predict_test_input_fn = estimator_api.inputs.pandas_input_fn(test_df, shuffle=False)

embedded_text_feature_column = hub.text_embedding_column(
    key="Phrase",
    module_spec="https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1",
    trainable=True,
)

run_config = estimator_api.RunConfig(keep_checkpoint_max=0)
estimator = estimator_api.DNNClassifier(
    hidden_units=[250, 50],
    feature_columns=[embedded_text_feature_column],
    n_classes=5,
    config=run_config,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003),
)

estimator.train(input_fn=train_input_fn, steps=10000)


## --- ERROR in cell 2, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/897909429.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      3[0m [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m     [0;32mimport[0m [0mtensorflow_estimator[0m [0;32mas[0m [0mestimator_api[0m  [0;31m# type: ignore[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0;32mexcept[0m [0mException[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'tensorflow_estimator'

During handling of the above exception, another exception occurred:

[0;31mImportError[0m                               Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/897909429.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      9[0m         [0mestimator_api[0m [0;34m=[0m [0mgetattr[0m[0;34m([0m[0mestimator_api[0m[0;34m,[0m [0;34m"estimator"[0m[0;34m,[0m [0;32mNone[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m     [0;32mif[0m [0mestimator_api[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 11[0;31m         raise ImportError(
[0m[1;32m     12[0m             [0;34m"No TensorFlow Estimator API found (tensorflow_estimator, tf.estimator, "[0m[0;34m[0m[0;34m[0m[0m
[1;32m     13[0m             [0;34m"or tf.compat.v1.estimator)."[0m[0;34m[0m[0;34m[0m[0m

[0;31mImportError[0m: No TensorFlow Estimator API found (tensorflow_estimator, tf.estimator, or tf.compat.v1.estimator).

## === cell 3
train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
validation_eval_result = estimator.evaluate(input_fn=predict_validation_input_fn)

print("Training set accuracy: {accuracy}".format(**train_eval_result))
print("Validation set accuracy: {accuracy}".format(**validation_eval_result))
