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

0.6463374162794907

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TFHUB_CACHE_DIR", "/kaggle/working/tfhub_cache")

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import model_selection

print("TF version:", tf.__version__)
print("TF Hub version:", getattr(hub, "__version__", "unknown"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR_CANDIDATES = [
    "/kaggle/input/movie-review-sentiment-analysis-kernels-only",
    "/kaggle/input",
    "../input/movie-review-sentiment-analysis-kernels-only",
    "../input",
]

DATA_DIR = None
for d in DATA_DIR_CANDIDATES:
    if os.path.exists(os.path.join(d, "train.tsv")) and os.path.exists(
        os.path.join(d, "test.tsv")
    ):
        DATA_DIR = d
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not locate train.tsv/test.tsv in expected Kaggle input directories."
    )

TRAIN_PATH = os.path.join(DATA_DIR, "train.tsv")
TEST_PATH = os.path.join(DATA_DIR, "test.tsv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sampleSubmission.csv")

print("Using DATA_DIR:", DATA_DIR)
print(
    "Train exists:",
    os.path.exists(TRAIN_PATH),
    "Test exists:",
    os.path.exists(TEST_PATH),
)



## === cell 2
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
    train_df = pd.read_csv(TRAIN_PATH, sep="\t")
    test_df = pd.read_csv(TEST_PATH, sep="\t")

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




## === cell 3
def make_features(df):
    return {"Phrase": df["Phrase"].astype(str)}


train_x = make_features(train_df)
val_x = make_features(validation_df)
test_x = make_features(test_df)

train_y = train_df["Sentiment"].astype(np.int32)
val_y = validation_df["Sentiment"].astype(np.int32)

print(train_x["Phrase"].head())
print(train_y.head())



## === cell 4
train_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=train_x, y=train_y, num_epochs=None, shuffle=True
)

predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=train_x, y=train_y, num_epochs=1, shuffle=False
)
predict_validation_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=val_x, y=val_y, num_epochs=1, shuffle=False
)
predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=test_x, num_epochs=1, shuffle=False
)

embedded_text_feature_column = hub.text_embedding_column(
    key="Phrase",
    module_spec=hub.Module("https://tfhub.dev/google/nnlm-en-dim128/1", trainable=True),
)

run_config = tf.estimator.RunConfig(keep_checkpoint_max=1, save_checkpoints_steps=500)

estimator = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=5,
    config=run_config,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003),
)

estimator.train(input_fn=train_input_fn, steps=10000)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/907072240.py in <cell line: 0>()
      1 # Fix: Use TF1 compat estimator input_fns + TF-Hub text embedding via hub.Module.
      2 # This preserves the original "hub text embedding column + DNNClassifier" logic.
----> 3 train_input_fn = tf.estimator.inputs.pandas_input_fn(
      4     x=train_x, y=train_y, num_epochs=None, shuffle=True
      5 )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/module_wrapper.py in _getattr(self, name)
    230     """
    231     try:
--> 232       attr = getattr(self._tfmw_wrapped_module, name)
    233     except AttributeError:
    234     # Placeholder for Google-internal contrib error

AttributeError: module 'tensorflow.compat.v1' has no attribute 'estimator'

## === cell 5
print("Training complete.")



## === cell 6
train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
validation_eval_result = estimator.evaluate(input_fn=predict_validation_input_fn)

print("Training set accuracy: {accuracy}".format(**train_eval_result))
print("Validation set accuracy: {accuracy}".format(**validation_eval_result))




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2855871204.py in <cell line: 0>()
----> 1 train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
      2 validation_eval_result = estimator.evaluate(input_fn=predict_validation_input_fn)
      3 
      4 print("Training set accuracy: {accuracy}".format(**train_eval_result))
      5 print("Validation set accuracy: {accuracy}".format(**validation_eval_result))

NameError: name 'estimator' is not defined

## === cell 7
def get_predictions(estimator, input_fn):
    return [int(x["class_ids"][0]) for x in estimator.predict(input_fn=input_fn)]




## === cell 8
pred_train = get_predictions(estimator, predict_train_input_fn)
with tf.Graph().as_default():
    cm = tf.math.confusion_matrix(
        labels=tf.constant(train_y.values, dtype=tf.int32),
        predictions=tf.constant(pred_train, dtype=tf.int32),
        num_classes=5,
    )
    with tf.Session() as session:
        cm_out = session.run(cm)

cm_out = cm_out.astype(float) / np.maximum(cm_out.sum(axis=1, keepdims=True), 1.0)

sns.heatmap(
    cm_out, annot=True, xticklabels=SENTIMENT_LABELS, yticklabels=SENTIMENT_LABELS
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1999045641.py in <cell line: 0>()
      1 # Fix: TF2 removed tf.confusion_matrix and tf.Session at top-level; use compat.v1 equivalents.
----> 2 pred_train = get_predictions(estimator, predict_train_input_fn)
      3 with tf.Graph().as_default():
      4     cm = tf.math.confusion_matrix(
      5         labels=tf.constant(train_y.values, dtype=tf.int32),

NameError: name 'estimator' is not defined

## === cell 9
if os.path.exists(SAMPLE_SUB_PATH):
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    print("Sample submission head:")
    print(sample_sub.head())
    print("Sample columns:", list(sample_sub.columns))



## === cell 10
test_df["Sentiment"] = get_predictions(estimator, predict_test_input_fn)

submission = test_df[["PhraseId", "Sentiment"]].copy()
submission["PhraseId"] = submission["PhraseId"].astype(np.int64)
submission["Sentiment"] = submission["Sentiment"].astype(np.int64)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())
print("Rows:", len(submission), "Cols:", submission.shape[1])

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4250696627.py in <cell line: 0>()
      1 # Fix: Write required submission columns exactly: PhraseId, Sentiment (integer class).
----> 2 test_df["Sentiment"] = get_predictions(estimator, predict_test_input_fn)
      3 
      4 submission = test_df[["PhraseId", "Sentiment"]].copy()
      5 submission["PhraseId"] = submission["PhraseId"].astype(np.int64)

NameError: name 'estimator' is not defined
