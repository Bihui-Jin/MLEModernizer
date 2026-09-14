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

0.64579

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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

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
    train_path = os.path.join(".", "input", "train.tsv")
    test_path = os.path.join(".", "input", "test.tsv")

    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")

    add_readable_labels_column(train_df, "Sentiment")

    train_indices, validation_indices = tf.compat.v1.keras.utils.train_test_split(
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
/tmp/ipykernel_11/3684081283.py in <cell line: 0>()
     40 
     41 
---> 42 train_df, validation_df, test_df = get_data()
     43 train_df.head()
     44 

/tmp/ipykernel_11/3684081283.py in get_data(validation_set_ratio)
     19     test_path = os.path.join(".", "input", "test.tsv")
     20 
---> 21     train_df = pd.read_csv(train_path, sep="\t")
     22     test_df = pd.read_csv(test_path, sep="\t")
     23 

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
    train_df, train_df["Sentiment"], num_epochs=None, shuffle=True, batch_size=128
)

predict_train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    train_df, train_df["Sentiment"], shuffle=False, batch_size=128
)

predict_validation_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    validation_df, validation_df["Sentiment"], shuffle=False, batch_size=128
)

predict_test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    test_df, shuffle=False, batch_size=128
)

embedded_text_feature_column = hub.text_embedding_column(
    key="Phrase",
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1",
    trainable=True,
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4141001288.py in <cell line: 0>()
      1 # Input functions – use the TF‑v1 compatibility layer
----> 2 train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
      3     train_df, train_df["Sentiment"], num_epochs=None, shuffle=True, batch_size=128
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
run_config = tf.estimator.RunConfig(keep_checkpoint_max=0)

estimator = tf.estimator.DNNClassifier(
    hidden_units=[250, 50],
    feature_columns=[embedded_text_feature_column],
    n_classes=5,
    dropout=0.1,
    optimizer=tf.compat.v1.train.AdagradOptimizer(learning_rate=0.003),
    config=run_config,
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1992770907.py in <cell line: 0>()
----> 1 run_config = tf.estimator.RunConfig(keep_checkpoint_max=0)
      2 
      3 estimator = tf.estimator.DNNClassifier(
      4     hidden_units=[250, 50],
      5     feature_columns=[embedded_text_feature_column],

AttributeError: module 'tensorflow' has no attribute 'estimator'

## === cell 4
estimator.train(input_fn=train_input_fn, steps=10000)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1548919889.py in <cell line: 0>()
----> 1 estimator.train(input_fn=train_input_fn, steps=10000)
      2 
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
/tmp/ipykernel_11/2855871204.py in <cell line: 0>()
----> 1 train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
      2 validation_eval_result = estimator.evaluate(input_fn=predict_validation_input_fn)
      3 
      4 print("Training set accuracy: {accuracy}".format(**train_eval_result))
      5 print("Validation set accuracy: {accuracy}".format(**validation_eval_result))

NameError: name 'estimator' is not defined

## === cell 6
def get_predictions(est, input_fn):
    return [int(pred["class_ids"][0]) for pred in est.predict(input_fn=input_fn)]


with tf.Graph().as_default():
    cm = tf.math.confusion_matrix(
        train_df["Sentiment"], get_predictions(estimator, predict_train_input_fn)
    )
    with tf.compat.v1.Session() as sess:
        cm_out = sess.run(cm)

cm_out = cm_out.astype(float) / cm_out.sum(axis=1)[:, np.newaxis]

sns.heatmap(
    cm_out,
    annot=True,
    fmt=".2f",
    xticklabels=SENTIMENT_LABELS,
    yticklabels=SENTIMENT_LABELS,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/546553973.py in <cell line: 0>()
      5 with tf.Graph().as_default():
      6     cm = tf.math.confusion_matrix(
----> 7         train_df["Sentiment"], get_predictions(estimator, predict_train_input_fn)
      8     )
      9     with tf.compat.v1.Session() as sess:

NameError: name 'train_df' is not defined

## === cell 7
test_df["Sentiment"] = get_predictions(estimator, predict_test_input_fn)

test_df.to_csv(
    "submission.csv",
    columns=["PhraseId", "Sentiment"],
    header=["PhraseId", "Sentiment"],
    index=False,
)

print("Submission file 'submission.csv' created.")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/657868771.py in <cell line: 0>()
----> 1 test_df["Sentiment"] = get_predictions(estimator, predict_test_input_fn)
      2 
      3 test_df.to_csv(
      4     "submission.csv",
      5     columns=["PhraseId", "Sentiment"],

NameError: name 'estimator' is not defined
