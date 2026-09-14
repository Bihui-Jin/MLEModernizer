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
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    base_path = "/kaggle/input/movie-review-sentiment-analysis-kernels-only"
    train_path = os.path.join(base_path, "train.tsv")
    test_path = os.path.join(base_path, "test.tsv")

    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")

    add_readable_labels_column(train_df, "Sentiment")

    from sklearn.model_selection import train_test_split

    train_sids, val_sids = train_test_split(
        np.unique(train_df["SentenceId"]),
        test_size=validation_set_ratio,
        random_state=0,
    )

    validation_df = train_df[train_df["SentenceId"].isin(val_sids)]
    train_df = train_df[train_df["SentenceId"].isin(train_sids)]

    print(
        "Split the training data into %d training and %d validation examples."
        % (len(train_df), len(validation_df))
    )
    return train_df, validation_df, test_df




## === cell 2
train_df, validation_df, test_df = get_data()
train_df.head()




## === cell 3
train_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=train_df,
    y=train_df["Sentiment"],
    num_epochs=None,
    shuffle=True,
    batch_size=128,
)

predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=train_df,
    y=train_df["Sentiment"],
    shuffle=False,
    batch_size=128,
)

predict_validation_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=validation_df,
    y=validation_df["Sentiment"],
    shuffle=False,
    batch_size=128,
)

predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=test_df,
    shuffle=False,
    batch_size=128,
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/636845353.py in <cell line: 0>()
      1 # Input functions using the Estimator API (TensorFlow v2 compatible)
----> 2 train_input_fn = tf.estimator.inputs.pandas_input_fn(
      3     x=train_df,
      4     y=train_df["Sentiment"],
      5     num_epochs=None,

AttributeError: module 'tensorflow' has no attribute 'estimator'

## === cell 4
text_categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
    key="Phrase", hash_bucket_size=10000
)

embedded_text_feature_column = tf.feature_column.embedding_column(
    categorical_column=text_categorical_column,
    dimension=128,
    trainable=True,
)




## === cell 5
run_config = tf.estimator.RunConfig(keep_checkpoint_max=0)

estimator = tf.estimator.DNNClassifier(
    hidden_units=[250, 50],
    feature_columns=[embedded_text_feature_column],
    n_classes=5,
    dropout=0.1,
    optimizer=tf.compat.v1.train.AdagradOptimizer(learning_rate=0.003),
    config=run_config,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1992770907.py in <cell line: 0>()
----> 1 run_config = tf.estimator.RunConfig(keep_checkpoint_max=0)
      2 
      3 estimator = tf.estimator.DNNClassifier(
      4     hidden_units=[250, 50],
      5     feature_columns=[embedded_text_feature_column],

AttributeError: module 'tensorflow' has no attribute 'estimator'

## === cell 6
estimator.train(input_fn=train_input_fn, steps=10000)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1548919889.py in <cell line: 0>()
----> 1 estimator.train(input_fn=train_input_fn, steps=10000)
      2 
      3 

NameError: name 'estimator' is not defined

## === cell 7
train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
validation_eval_result = estimator.evaluate(input_fn=predict_validation_input_fn)

print("Training set accuracy: {accuracy:.4f}".format(**train_eval_result))
print("Validation set accuracy: {accuracy:.4f}".format(**validation_eval_result))




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/75113758.py in <cell line: 0>()
----> 1 train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
      2 validation_eval_result = estimator.evaluate(input_fn=predict_validation_input_fn)
      3 
      4 print("Training set accuracy: {accuracy:.4f}".format(**train_eval_result))
      5 print("Validation set accuracy: {accuracy:.4f}".format(**validation_eval_result))

NameError: name 'estimator' is not defined

## === cell 8
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




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/546553973.py in <cell line: 0>()
      5 with tf.Graph().as_default():
      6     cm = tf.math.confusion_matrix(
----> 7         train_df["Sentiment"], get_predictions(estimator, predict_train_input_fn)
      8     )
      9     with tf.compat.v1.Session() as sess:

NameError: name 'estimator' is not defined

## === cell 9
test_df["Sentiment"] = get_predictions(estimator, predict_test_input_fn)

test_df.to_csv(
    "submission.csv",
    columns=["PhraseId", "Sentiment"],
    header=["PhraseId", "Sentiment"],
    index=False,
)

print("Submission file 'submission.csv' created.")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/657868771.py in <cell line: 0>()
----> 1 test_df["Sentiment"] = get_predictions(estimator, predict_test_input_fn)
      2 
      3 test_df.to_csv(
      4     "submission.csv",
      5     columns=["PhraseId", "Sentiment"],

NameError: name 'estimator' is not defined
