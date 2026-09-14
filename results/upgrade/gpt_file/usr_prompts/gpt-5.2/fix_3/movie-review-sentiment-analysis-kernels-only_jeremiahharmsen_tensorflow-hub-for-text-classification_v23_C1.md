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

0.64579

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow_estimator as tfes

from sklearn import model_selection

tf.compat.v1.disable_eager_execution()

np.random.seed(0)
tf.compat.v1.set_random_seed(0)

print("TensorFlow:", tf.__version__)
print("TF Hub:", hub.__version__)



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
    train_path = "/kaggle/input/movie-review-sentiment-analysis-kernels-only/train.tsv"
    test_path = "/kaggle/input/movie-review-sentiment-analysis-kernels-only/test.tsv"

    if not os.path.exists(train_path):
        train_path = "/kaggle/input/train.tsv"
    if not os.path.exists(test_path):
        test_path = "/kaggle/input/test.tsv"

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



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2020425651.py in <cell line: 0>()
     45 
     46 
---> 47 train_df, validation_df, test_df = get_data()
     48 train_df.head()
     49 

/tmp/ipykernel_11/2020425651.py in get_data(validation_set_ratio)
     28     add_readable_labels_column(train_df, "Sentiment")
     29 
---> 30     train_indices, validation_indices = model_selection.train_test_split(
     31         np.unique(train_df["SentenceId"]),
     32         test_size=validation_set_ratio,

NameError: name 'model_selection' is not defined

## === cell 2
try:
    pandas_input_fn = tf.estimator.inputs.pandas_input_fn
except Exception:
    pandas_input_fn = tfes.estimator.inputs.pandas_input_fn

train_input_fn = pandas_input_fn(
    x=train_df[["Phrase"]], y=train_df["Sentiment"], num_epochs=None, shuffle=True
)

predict_train_input_fn = pandas_input_fn(
    x=train_df[["Phrase"]], y=train_df["Sentiment"], shuffle=False, num_epochs=1
)

predict_validation_input_fn = pandas_input_fn(
    x=validation_df[["Phrase"]],
    y=validation_df["Sentiment"],
    shuffle=False,
    num_epochs=1,
)

predict_test_input_fn = pandas_input_fn(
    x=test_df[["Phrase"]], shuffle=False, num_epochs=1
)

embedded_text_feature_column = hub.text_embedding_column(
    key="Phrase",
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1",
    trainable=True,
)

run_config = tf.compat.v1.estimator.RunConfig(keep_checkpoint_max=0)

estimator = tfes.estimator.DNNClassifier(
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
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4232760309.py in <cell line: 0>()
      3 try:
----> 4     pandas_input_fn = tf.estimator.inputs.pandas_input_fn
      5 except Exception:

AttributeError: module 'tensorflow' has no attribute 'estimator'

During handling of the above exception, another exception occurred:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4232760309.py in <cell line: 0>()
      5 except Exception:
      6     # Fallback: some environments expose it under tensorflow_estimator
----> 7     pandas_input_fn = tfes.estimator.inputs.pandas_input_fn
      8 
      9 train_input_fn = pandas_input_fn(

NameError: name 'tfes' is not defined

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
    return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]


with tf.Graph().as_default():
    cm = tf.compat.v1.confusion_matrix(
        labels=train_df["Sentiment"].values,
        predictions=get_predictions(estimator, predict_train_input_fn),
        num_classes=5,
    )
    with tf.compat.v1.Session() as session:
        cm_out = session.run(cm)

cm_out = cm_out.astype(float) / cm_out.sum(axis=1)[:, np.newaxis]

sns.heatmap(
    cm_out, annot=True, xticklabels=SENTIMENT_LABELS, yticklabels=SENTIMENT_LABELS
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/747150605.py in <cell line: 0>()
      5 with tf.Graph().as_default():
      6     cm = tf.compat.v1.confusion_matrix(
----> 7         labels=train_df["Sentiment"].values,
      8         predictions=get_predictions(estimator, predict_train_input_fn),
      9         num_classes=5,

NameError: name 'train_df' is not defined

## === cell 5
test_df["Sentiment"] = get_predictions(estimator, predict_test_input_fn)

submission = test_df[["PhraseId", "Sentiment"]].copy()
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Submission columns:", submission.columns.tolist())
print("Sentiment value counts:\n", submission["Sentiment"].value_counts().sort_index())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2606381844.py in <cell line: 0>()
----> 1 test_df["Sentiment"] = get_predictions(estimator, predict_test_input_fn)
      2 
      3 submission = test_df[["PhraseId", "Sentiment"]].copy()
      4 submission.to_csv("submission.csv", index=False)
      5 

NameError: name 'estimator' is not defined
