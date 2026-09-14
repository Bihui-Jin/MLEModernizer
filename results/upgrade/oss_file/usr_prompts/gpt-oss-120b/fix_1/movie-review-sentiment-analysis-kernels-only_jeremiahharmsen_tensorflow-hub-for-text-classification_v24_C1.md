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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import zipfile

from sklearn import model_selection


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["Sentiment"], num_epochs=None, shuffle=True)

predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["Sentiment"], shuffle=False)
predict_validation_input_fn = tf.estimator.inputs.pandas_input_fn(
    validation_df, validation_df["Sentiment"], shuffle=False)
predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    test_df, shuffle=False)

embedded_text_feature_column = hub.text_embedding_column(
    key="Phrase", 
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1",
    trainable=True)

run_config = tf.estimator.RunConfig(keep_checkpoint_max=0)
estimator = tf.estimator.DNNClassifier(
    hidden_units=[250, 50],
    feature_columns=[embedded_text_feature_column],
    n_classes=5,
    dropout=0.2,
    config=run_config,
    batch_norm=True,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

estimator.train(input_fn=train_input_fn, steps=10000);


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/474938619.py in <cell line: 0>()
      1 # Training input on the whole training set with no limit on training epochs.
----> 2 train_input_fn = tf.estimator.inputs.pandas_input_fn(
      3     train_df, train_df["Sentiment"], num_epochs=None, shuffle=True)
      4 
      5 # Prediction on the whole training set.

AttributeError: module 'tensorflow' has no attribute 'estimator'

## === cell 3
train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
validation_eval_result = estimator.evaluate(input_fn=predict_validation_input_fn)

print("Training set accuracy: {accuracy}".format(**train_eval_result))
print("Validation set accuracy: {accuracy}".format(**validation_eval_result))


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1383778768.py in <cell line: 0>()
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
  cm = tf.confusion_matrix(train_df["Sentiment"],
                           get_predictions(estimator, predict_train_input_fn))
  with tf.Session() as session:
    cm_out = session.run(cm)

cm_out = cm_out.astype(float) / cm_out.sum(axis=1)[:, np.newaxis]

sns.heatmap(
    cm_out,
    annot=True,
    xticklabels=SENTIMENT_LABELS,
    yticklabels=SENTIMENT_LABELS)
plt.xlabel("Predicted")
plt.ylabel("True")


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4077222459.py in <cell line: 0>()
      4 # Create a confusion matrix on training data.
      5 with tf.Graph().as_default():
----> 6   cm = tf.confusion_matrix(train_df["Sentiment"],
      7                            get_predictions(estimator, predict_train_input_fn))
      8   with tf.Session() as session:

AttributeError: module 'tensorflow' has no attribute 'confusion_matrix'

## === cell 5
test_df["Predictions"] = get_predictions(estimator, predict_test_input_fn)
test_df.to_csv(
    'submission.csv',
    columns=["PhraseId", "Predictions"],
    header=["PhraseId", "Sentiment"],
    index=False)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2634373042.py in <cell line: 0>()
----> 1 test_df["Predictions"] = get_predictions(estimator, predict_test_input_fn)
      2 test_df.to_csv(
      3     'submission.csv',
      4     columns=["PhraseId", "Predictions"],
      5     header=["PhraseId", "Sentiment"],

NameError: name 'estimator' is not defined
