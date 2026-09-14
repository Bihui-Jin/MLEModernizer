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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.58945

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, random
import numpy as np, pandas as pd
import tensorflow as tf, tensorflow_hub as hub
from sklearn.model_selection import train_test_split

seed = 197
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

print("TF version:", tf.__version__)
print("Contents of ../input:", os.listdir("../input"))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = "../input/train.tsv"
test_path = "../input/test.tsv"
train_df = pd.read_csv(train_path, sep="\t")
test_df = pd.read_csv(test_path, sep="\t")

print(train_df.head())
print(test_df.head())




## === cell 2
train_df, val_df = train_test_split(
    train_df, test_size=0.1, random_state=seed, stratify=train_df["Sentiment"]
)

train_text = train_df["Phrase"].astype(str).values
train_labels = train_df["Sentiment"].values
val_text = val_df["Phrase"].astype(str).values
val_labels = val_df["Sentiment"].values




## === cell 3
embedding_url = "https://tfhub.dev/google/nnlm-en-dim50-with-normalization/1"
embedding_layer = hub.KerasLayer(
    embedding_url,
    output_shape=[50],
    trainable=False,
    input_shape=[],
    dtype=tf.string,
    name="embedding",
)

inputs = tf.keras.Input(shape=(), dtype=tf.string, name="Phrase")
x = embedding_layer(inputs)
x = tf.keras.layers.Dense(500, activation="relu")(x)
x = tf.keras.layers.Dense(100, activation="relu")(x)
outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.003),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/434411455.py in <cell line: 0>()
     11 
     12 inputs = tf.keras.Input(shape=(), dtype=tf.string, name="Phrase")
---> 13 x = embedding_layer(inputs)
     14 x = tf.keras.layers.Dense(500, activation="relu")(x)
     15 x = tf.keras.layers.Dense(100, activation="relu")(x)

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/keras_layer.py in call(self, inputs, training)
    240     # or else Keras' global `learning_phase`, which might actually be a tensor.
    241     if not self._has_training_argument:
--> 242       result = f()
    243     else:
    244       if self.trainable:

TypeError: Exception encountered when calling layer 'embedding' (type KerasLayer).

Binding inputs to tf.function failed due to `too many positional arguments`. Received args: (<KerasTensor shape=(None,), dtype=string, sparse=False, name=Phrase>,) and kwargs: {} for signature: () -> Dict[['default', TensorSpec(shape=(None, 50), dtype=tf.float32, name=None)]].
Fallback to flat signature also failed due to: pruned(default): expected argument #0(zero-based) to be a Tensor; got KerasTensor (<KerasTensor shape=(None,), dtype=string, sparse=False, name=Phrase>).

Call arguments received by layer 'embedding' (type KerasLayer):
  • inputs=<KerasTensor shape=(None,), dtype=string, sparse=False, name=Phrase>
  • training=None

## === cell 4
batch_size = 128
epochs = 5  # modest number to stay within time limits
model.fit(
    x=train_text,
    y=train_labels,
    validation_data=(val_text, val_labels),
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1920576544.py in <cell line: 0>()
      2 batch_size = 128
      3 epochs = 5  # modest number to stay within time limits
----> 4 model.fit(
      5     x=train_text,
      6     y=train_labels,

NameError: name 'model' is not defined

## === cell 5
val_loss, val_acc = model.evaluate(x=val_text, y=val_labels, verbose=0)
print(f"Validation accuracy: {val_acc:.4f}")




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2303308969.py in <cell line: 0>()
      1 # Evaluate on validation set
----> 2 val_loss, val_acc = model.evaluate(x=val_text, y=val_labels, verbose=0)
      3 print(f"Validation accuracy: {val_acc:.4f}")
      4 
      5 

NameError: name 'model' is not defined

## === cell 6
test_text = test_df["Phrase"].astype(str).values
pred_probs = model.predict(test_text, batch_size=batch_size, verbose=0)
pred_classes = np.argmax(pred_probs, axis=1)

sub = pd.read_csv("../input/sampleSubmission.csv")
sub["Sentiment"] = pred_classes
sub.to_csv("sub_tfhub.csv", index=False)
print("Submission saved to sub_tfhub.csv")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1033890789.py in <cell line: 0>()
      1 # Predict on test set and create submission
      2 test_text = test_df["Phrase"].astype(str).values
----> 3 pred_probs = model.predict(test_text, batch_size=batch_size, verbose=0)
      4 pred_classes = np.argmax(pred_probs, axis=1)
      5 

NameError: name 'model' is not defined
