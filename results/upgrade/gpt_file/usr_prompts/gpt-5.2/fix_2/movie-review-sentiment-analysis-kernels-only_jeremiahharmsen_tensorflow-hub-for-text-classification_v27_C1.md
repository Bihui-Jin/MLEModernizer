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

0.65223

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import tensorflow_hub as hub

import tensorflow_estimator as tf_estimator



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
    train_path_candidates = [
        "../input/train.tsv",
        "/kaggle/input/train.tsv",
        "/kaggle/data/train.tsv",
        "/kaggle/input/movie-review-sentiment-analysis-kernels-only/train.tsv",
    ]
    test_path_candidates = [
        "../input/test.tsv",
        "/kaggle/input/test.tsv",
        "/kaggle/data/test.tsv",
        "/kaggle/input/movie-review-sentiment-analysis-kernels-only/test.tsv",
    ]

    train_path = next((p for p in train_path_candidates if os.path.exists(p)), None)
    test_path = next((p for p in test_path_candidates if os.path.exists(p)), None)
    if train_path is None or test_path is None:
        raise FileNotFoundError(
            f"Could not find train/test TSV. Looked for train in {train_path_candidates} "
            f"and test in {test_path_candidates}."
        )

    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")

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



## === cell 2
train_input_fn = tf_estimator.estimator.inputs.pandas_input_fn(
    x=train_df, y=train_df["Sentiment"], num_epochs=None, shuffle=True
)

predict_train_input_fn = tf_estimator.estimator.inputs.pandas_input_fn(
    x=train_df, y=train_df["Sentiment"], shuffle=False, num_epochs=1
)
predict_validation_input_fn = tf_estimator.estimator.inputs.pandas_input_fn(
    x=validation_df, y=validation_df["Sentiment"], shuffle=False, num_epochs=1
)
predict_test_input_fn = tf_estimator.estimator.inputs.pandas_input_fn(
    x=test_df, shuffle=False, num_epochs=1
)

embedded_text_feature_column = hub.text_embedding_column(
    key="Phrase",
    module_spec="https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1",
    trainable=True,
)

run_config = tf_estimator.estimator.RunConfig(keep_checkpoint_max=0)

estimator = tf_estimator.estimator.DNNClassifier(
    hidden_units=[250, 50],
    feature_columns=[embedded_text_feature_column],
    n_classes=5,
    config=run_config,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003),
)

estimator.train(input_fn=train_input_fn, steps=10000)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3914993511.py in <cell line: 0>()
      1 # Fix 3: Use tf_estimator (separate package) and TF1-compatible input_fns.
----> 2 train_input_fn = tf_estimator.estimator.inputs.pandas_input_fn(
      3     x=train_df, y=train_df["Sentiment"], num_epochs=None, shuffle=True
      4 )
      5 

NameError: name 'tf_estimator' is not defined

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
    return [int(x["class_ids"][0]) for x in estimator.predict(input_fn=input_fn)]


with tf.Graph().as_default():
    cm = tf.math.confusion_matrix(
        labels=train_df["Sentiment"].values,
        predictions=get_predictions(estimator, predict_train_input_fn),
        num_classes=5,
    )
    with tf.Session() as session:
        cm_out = session.run(cm)

cm_out = cm_out.astype(float)
cm_out = cm_out / cm_out.sum(axis=1, keepdims=True)

sns.heatmap(
    cm_out, annot=True, xticklabels=SENTIMENT_LABELS, yticklabels=SENTIMENT_LABELS
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2954617325.py in <cell line: 0>()
      8     cm = tf.math.confusion_matrix(
      9         labels=train_df["Sentiment"].values,
---> 10         predictions=get_predictions(estimator, predict_train_input_fn),
     11         num_classes=5,
     12     )

NameError: name 'estimator' is not defined

## === cell 5
test_pred = get_predictions(estimator, predict_test_input_fn)

submission = pd.DataFrame(
    {
        "PhraseId": test_df["PhraseId"].astype(int).values,
        "Sentiment": np.array(test_pred, dtype=int),
    }
)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2806148833.py in <cell line: 0>()
      1 # Fix 5: Produce a valid submission with required header: PhraseId,Sentiment
----> 2 test_pred = get_predictions(estimator, predict_test_input_fn)
      3 
      4 submission = pd.DataFrame(
      5     {

NameError: name 'estimator' is not defined
