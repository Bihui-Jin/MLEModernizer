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

0.64657

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection

tf.compat.v1.disable_eager_execution()



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
    train_df = pd.read_csv("./input/train.tsv", sep="\t")
    test_df = pd.read_csv("./input/test.tsv", sep="\t")

    add_readable_labels_column(train_df, "Sentiment")

    train_indices, validation_indices = model_selection.train_test_split(
        np.unique(train_df["SentenceId"]),
        test_size=validation_set_ratio,
        random_state=0,
    )

    validation_df = train_df[train_df["SentenceId"].isin(validation_indices)]
    train_df = train_df[train_df["SentenceId"].isin(train_indices)]

    print(
        "Split the training data into %d training and %d validation examples."
        % (len(train_df), len(validation_df))
    )

    return train_df, validation_df, test_df


train_df, validation_df, test_df = get_data()
train_df.head()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3407796564.py in <cell line: 0>()
     38 
     39 
---> 40 train_df, validation_df, test_df = get_data()
     41 train_df.head()
     42 

/tmp/ipykernel_11/3407796564.py in get_data(validation_set_ratio)
     16 def get_data(validation_set_ratio=0.1):
     17     # Adjust paths to match typical Kaggle layout
---> 18     train_df = pd.read_csv("./input/train.tsv", sep="\t")
     19     test_df = pd.read_csv("./input/test.tsv", sep="\t")
     20 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: './input/train.tsv'

## === cell 2
train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=train_df, y=train_df["Sentiment"], num_epochs=None, shuffle=True
)

predict_train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=train_df, y=train_df["Sentiment"], shuffle=False
)

predict_validation_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=validation_df, y=validation_df["Sentiment"], shuffle=False
)

predict_test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=test_df, shuffle=False
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4012665488.py in <cell line: 0>()
      1 # Input functions using TF‑v1 compatibility layer
----> 2 train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
      3     x=train_df, y=train_df["Sentiment"], num_epochs=None, shuffle=True
      4 )
      5 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/module_wrapper.py in _getattr(self, name)
    230     """
    231     try:
--> 232       attr = getattr(self._tfmw_wrapped_module, name)
    233     except AttributeError:
    234     # Placeholder for Google-internal contrib error

AttributeError: module 'tensorflow.compat.v1' has no attribute 'estimator'

## === cell 3
embedded_text_feature_column = hub.text_embedding_column(
    key="Phrase",
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1",
    trainable=True,
)

run_config = tf.compat.v1.estimator.RunConfig(keep_checkpoint_max=0)

estimator = tf.compat.v1.estimator.DNNClassifier(
    hidden_units=[250, 50],
    feature_columns=[embedded_text_feature_column],
    n_classes=5,
    dropout=0.2,
    config=run_config,
    batch_norm=True,
    optimizer=tf.compat.v1.train.AdagradOptimizer(learning_rate=0.003),
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1765952519.py in <cell line: 0>()
----> 1 embedded_text_feature_column = hub.text_embedding_column(
      2     key="Phrase",
      3     module_spec="https://tfhub.dev/google/nnlm-en-dim128/1",
      4     trainable=True,
      5 )

AttributeError: module 'tensorflow_hub' has no attribute 'text_embedding_column'

## === cell 4
estimator.train(input_fn=train_input_fn, steps=10000)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1104026880.py in <cell line: 0>()
      1 # Train the model
----> 2 estimator.train(input_fn=train_input_fn, steps=10000)
      3 

NameError: name 'estimator' is not defined

## === cell 5
train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
validation_eval_result = estimator.evaluate(input_fn=predict_validation_input_fn)

print("Training set accuracy: {accuracy}".format(**train_eval_result))
print("Validation set accuracy: {accuracy}".format(**validation_eval_result))




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/125861531.py in <cell line: 0>()
      1 # Evaluate on training and validation sets
----> 2 train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
      3 validation_eval_result = estimator.evaluate(input_fn=predict_validation_input_fn)
      4 
      5 print("Training set accuracy: {accuracy}".format(**train_eval_result))

NameError: name 'estimator' is not defined

## === cell 6
def get_predictions(estimator, input_fn):
    return [int(p["class_ids"][0]) for p in estimator.predict(input_fn=input_fn)]


with tf.Graph().as_default():
    cm = tf.math.confusion_matrix(
        train_df["Sentiment"], get_predictions(estimator, predict_train_input_fn)
    )
    with tf.compat.v1.Session() as sess:
        cm_out = sess.run(cm)

cm_norm = cm_out.astype(float) / cm_out.sum(axis=1)[:, np.newaxis]

sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    xticklabels=SENTIMENT_LABELS,
    yticklabels=SENTIMENT_LABELS,
    cmap="Blues",
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/862445113.py in <cell line: 0>()
      6 with tf.Graph().as_default():
      7     cm = tf.math.confusion_matrix(
----> 8         train_df["Sentiment"], get_predictions(estimator, predict_train_input_fn)
      9     )
     10     with tf.compat.v1.Session() as sess:

NameError: name 'train_df' is not defined

## === cell 7
test_df["Predictions"] = get_predictions(estimator, predict_test_input_fn)

test_df.to_csv(
    "submission.csv",
    columns=["PhraseId", "Predictions"],
    header=["PhraseId", "Sentiment"],
    index=False,
)

print("Submission file written to submission.csv")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1658249995.py in <cell line: 0>()
      1 # Generate predictions for the test set and write submission file
----> 2 test_df["Predictions"] = get_predictions(estimator, predict_test_input_fn)
      3 
      4 test_df.to_csv(
      5     "submission.csv",

NameError: name 'estimator' is not defined
