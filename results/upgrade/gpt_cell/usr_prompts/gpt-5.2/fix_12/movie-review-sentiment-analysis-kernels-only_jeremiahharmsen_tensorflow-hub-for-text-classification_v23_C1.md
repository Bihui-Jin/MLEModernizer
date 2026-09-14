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

try:
    from google.protobuf.internal import api_implementation

    api_implementation._default_implementation_type = "python"
except Exception:
    pass


import numpy as np
import pandas as pd
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
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
try:
    from google.protobuf.internal import api_implementation

    api_implementation._default_implementation_type = "python"
except Exception:
    pass

try:
    from google.protobuf.message_factory import MessageFactory

    if not hasattr(MessageFactory, "GetPrototype"):
        if hasattr(MessageFactory, "GetMessageClass"):

            def _GetPrototype(self, descriptor):
                return self.GetMessageClass(descriptor)

        else:

            def _GetPrototype(self, descriptor):
                raise AttributeError(
                    "protobuf MessageFactory is missing GetPrototype and GetMessageClass"
                )

        MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import tensorflow as tf
import tensorflow_hub as hub

if not hasattr(tf, "estimator"):
    if (
        hasattr(tf, "compat")
        and hasattr(tf.compat, "v1")
        and hasattr(tf.compat.v1, "estimator")
    ):
        tf.estimator = tf.compat.v1.estimator
    else:
        try:
            import tensorflow_estimator as _tfe

            tf.estimator = _tfe.estimator
        except Exception:
            raise ImportError(
                "TensorFlow Estimator API is not available in this environment. "
                "This notebook requires a TensorFlow distribution that includes "
                "tf.estimator (or tf.compat.v1.estimator), or an installed "
                "tensorflow_estimator package."
            )

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
    key="Phrase",
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1",
    trainable=True,
)

run_config = tf.estimator.RunConfig(keep_checkpoint_max=0)
estimator = tf.estimator.DNNClassifier(
    hidden_units=[250, 50],
    feature_columns=[embedded_text_feature_column],
    n_classes=5,
    dropout=0.1,
    config=run_config,
    batch_norm=True,
    optimizer=tf.compat.v1.train.AdagradOptimizer(learning_rate=0.003),
)

estimator.train(input_fn=train_input_fn, steps=10000)


## --- ERROR in cell 2, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/268556740.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     46[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 47[0;31m             [0;32mimport[0m [0mtensorflow_estimator[0m [0;32mas[0m [0m_tfe[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     48[0m [0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'tensorflow_estimator'

During handling of the above exception, another exception occurred:

[0;31mImportError[0m                               Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/268556740.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     49[0m             [0mtf[0m[0;34m.[0m[0mestimator[0m [0;34m=[0m [0m_tfe[0m[0;34m.[0m[0mestimator[0m[0;34m[0m[0;34m[0m[0m
[1;32m     50[0m         [0;32mexcept[0m [0mException[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 51[0;31m             raise ImportError(
[0m[1;32m     52[0m                 [0;34m"TensorFlow Estimator API is not available in this environment. "[0m[0;34m[0m[0;34m[0m[0m
[1;32m     53[0m                 [0;34m"This notebook requires a TensorFlow distribution that includes "[0m[0;34m[0m[0;34m[0m[0m

[0;31mImportError[0m: TensorFlow Estimator API is not available in this environment. This notebook requires a TensorFlow distribution that includes tf.estimator (or tf.compat.v1.estimator), or an installed tensorflow_estimator package.

## === cell 3
train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
validation_eval_result = estimator.evaluate(input_fn=predict_validation_input_fn)

print("Training set accuracy: {accuracy}".format(**train_eval_result))
print("Validation set accuracy: {accuracy}".format(**validation_eval_result))
