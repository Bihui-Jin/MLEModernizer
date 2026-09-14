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

try:
    import tensorflow_hub as hub
except Exception as e:
    raise RuntimeError(
        "tensorflow_hub failed to import due to an environment/protobuf mismatch. "
        "This solution requires TF Hub to load the NNLM embedding module."
    ) from e

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

hub_url = "https://tfhub.dev/google/nnlm-en-dim128/1"

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="Phrase")
embedding_layer = hub.KerasLayer(hub_url, trainable=True, name="nnlm_embedding")
x = embedding_layer(text_input)

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



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/198245126.py in <cell line: 0>()
     14 text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="Phrase")
     15 embedding_layer = hub.KerasLayer(hub_url, trainable=True, name="nnlm_embedding")
---> 16 x = embedding_layer(text_input)
     17 
     18 x = tf.keras.layers.Dense(250, activation="relu", name="dense_250")(x)

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/keras_layer.py in call(self, inputs, training)
    224     # These checks happen here and not in __init__, because self.trainable is
    225     # a mutable public attribute.
--> 226     self._check_trainability()
    227 
    228     # We basically want to call this...

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/keras_layer.py in _check_trainability(self)
    283     #   provide a trainable_variables attribute.
    284     if self._is_hub_module_v1:
--> 285       raise ValueError(
    286           "Setting hub.KerasLayer.trainable = True is unsupported when "
    287           "loading from the TF1 Hub format.")

ValueError: Exception encountered when calling layer 'nnlm_embedding' (type KerasLayer).

Setting hub.KerasLayer.trainable = True is unsupported when loading from the TF1 Hub format.

Call arguments received by layer 'nnlm_embedding' (type KerasLayer):
  • inputs=<KerasTensor shape=(None,), dtype=string, sparse=False, name=Phrase>
  • training=None

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



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3409076601.py in <cell line: 0>()
     13 y_val = validation_df["Sentiment"].astype(np.int64).values
     14 
---> 15 history = model.fit(
     16     x=x_train,
     17     y=y_train,

NameError: name 'model' is not defined

## === cell 4
train_loss, train_acc = model.evaluate(
    x_train, y_train, batch_size=batch_size, verbose=0
)
val_loss, val_acc = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)

print(f"Training set accuracy: {train_acc:.6f}")
print(f"Validation set accuracy: {val_acc:.6f}")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3251256581.py in <cell line: 0>()
      1 # Evaluate similarly to estimator.evaluate
----> 2 train_loss, train_acc = model.evaluate(
      3     x_train, y_train, batch_size=batch_size, verbose=0
      4 )
      5 val_loss, val_acc = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)

NameError: name 'model' is not defined

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
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3911498172.py in <cell line: 0>()
      6 
      7 train_preds = get_predictions_keras(
----> 8     model, train_df["Phrase"].values, batch_size=batch_size
      9 )
     10 

NameError: name 'model' is not defined

## === cell 6
test_phrases = test_df["Phrase"].astype(str).values
test_preds = get_predictions_keras(model, test_phrases, batch_size=batch_size)

submission = pd.DataFrame(
    {
        "PhraseId": test_df["PhraseId"].astype(np.int64).values,
        "Sentiment": test_preds.astype(np.int64),
    }
)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())
print("Submission shape:", submission.shape)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3774221754.py in <cell line: 0>()
      1 # Create submission with required columns: PhraseId, Sentiment
      2 test_phrases = test_df["Phrase"].astype(str).values
----> 3 test_preds = get_predictions_keras(model, test_phrases, batch_size=batch_size)
      4 
      5 submission = pd.DataFrame(

NameError: name 'model' is not defined
