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

0.64749

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn import model_selection

import matplotlib.pyplot as plt
import seaborn as sns

print("TensorFlow:", tf.__version__)



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


def _resolve_input_path(fname):
    candidates = [
        f"/kaggle/input/movie-review-sentiment-analysis-kernels-only/{fname}",
        f"/kaggle/input/{fname}",
        f"../input/movie-review-sentiment-analysis-kernels-only/{fname}",
        f"../input/{fname}",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {fname}. Tried: {candidates}")


def get_data(validation_set_ratio=0.1):
    train_path = _resolve_input_path("train.tsv")
    test_path = _resolve_input_path("test.tsv")

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



## === cell 2

tf.random.set_seed(0)
np.random.seed(0)

max_tokens = 30000
sequence_length = 40

vectorize = tf.keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    name="text_vectorization",
)

vectorize.adapt(train_df["Phrase"].astype(str).values)

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="Phrase")
x = vectorize(text_input)

x = tf.keras.layers.Embedding(
    input_dim=max_tokens,
    output_dim=128,
    name="token_embedding",
)(x)
x = tf.keras.layers.GlobalAveragePooling1D(name="avg_pool")(x)

x = tf.keras.layers.Dense(250, activation="relu", name="dense_250")(x)
x = tf.keras.layers.Dense(50, activation="relu", name="dense_50")(x)
logits = tf.keras.layers.Dense(5, activation="softmax", name="sentiment")(x)

model = tf.keras.Model(inputs=text_input, outputs=logits)
model.compile(
    optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.003),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## === cell 3
batch_size = 128
steps_target = 10000
steps_per_epoch = int(np.ceil(len(train_df) / batch_size))
epochs = int(np.ceil(steps_target / steps_per_epoch))

x_train = train_df["Phrase"].astype(str).values
y_train = train_df["Sentiment"].astype(np.int64).values

x_val = validation_df["Phrase"].astype(str).values
y_val = validation_df["Sentiment"].astype(np.int64).values

history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    verbose=2,
)



## === cell 4
train_loss, train_acc = model.evaluate(
    x_train, y_train, batch_size=batch_size, verbose=0
)
val_loss, val_acc = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)

print(f"Training set accuracy: {train_acc:.6f}")
print(f"Validation set accuracy: {val_acc:.6f}")




## === cell 5
def get_predictions_keras(model, phrases, batch_size=128):
    probs = model.predict(phrases.astype(str), batch_size=batch_size, verbose=0)
    return np.argmax(probs, axis=1)


train_preds = get_predictions_keras(
    model, train_df["Phrase"].values, batch_size=batch_size
)

cm = tf.math.confusion_matrix(
    labels=train_df["Sentiment"].astype(np.int64).values,
    predictions=train_preds,
    num_classes=5,
).numpy()

cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    xticklabels=SENTIMENT_LABELS,
    yticklabels=SENTIMENT_LABELS,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1135322977.py in <cell line: 0>()
      4 
      5 
----> 6 train_preds = get_predictions_keras(
      7     model, train_df["Phrase"].values, batch_size=batch_size
      8 )

/tmp/ipykernel_11/1135322977.py in get_predictions_keras(model, phrases, batch_size)
      1 def get_predictions_keras(model, phrases, batch_size=128):
----> 2     probs = model.predict(phrases.astype(str), batch_size=batch_size, verbose=0)
      3     return np.argmax(probs, axis=1)
      4 
      5 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/optree/ops.py in tree_map(func, tree, is_leaf, none_is_leaf, namespace, *rests)
    764     leaves, treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    765     flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
--> 766     return treespec.unflatten(map(func, *flat_args))
    767 
    768 

ValueError: Invalid dtype: str9056

## === cell 6
test_phrases = test_df["Phrase"].astype(str).values
test_preds = get_predictions_keras(model, test_phrases, batch_size=batch_size)

submission = pd.DataFrame(
    {
        "PhraseId": test_df["PhraseId"].astype(np.int64).values,
        "Sentiment": test_preds.astype(np.int64),
    }
)

submission = submission.sort_values("PhraseId").reset_index(drop=True)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())
print("Submission shape:", submission.shape)
print("Sentiment value counts:\n", submission["Sentiment"].value_counts().sort_index())

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/359476184.py in <cell line: 0>()
      1 test_phrases = test_df["Phrase"].astype(str).values
----> 2 test_preds = get_predictions_keras(model, test_phrases, batch_size=batch_size)
      3 
      4 submission = pd.DataFrame(
      5     {

/tmp/ipykernel_11/1135322977.py in get_predictions_keras(model, phrases, batch_size)
      1 def get_predictions_keras(model, phrases, batch_size=128):
----> 2     probs = model.predict(phrases.astype(str), batch_size=batch_size, verbose=0)
      3     return np.argmax(probs, axis=1)
      4 
      5 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/optree/ops.py in tree_map(func, tree, is_leaf, none_is_leaf, namespace, *rests)
    764     leaves, treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    765     flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
--> 766     return treespec.unflatten(map(func, *flat_args))
    767 
    768 

ValueError: Invalid dtype: str8576
