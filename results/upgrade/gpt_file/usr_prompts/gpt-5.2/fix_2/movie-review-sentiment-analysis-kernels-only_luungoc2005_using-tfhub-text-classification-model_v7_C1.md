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

0.58839

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print("Listing ../input (if exists):")
try:
    print(os.listdir("../input"))
except Exception as e:
    print("Could not list ../input:", e)
print("Listing /kaggle/input (if exists):")
try:
    print(os.listdir("/kaggle/input"))
except Exception as e:
    print("Could not list /kaggle/input:", e)



## === cell 1
seed = 197

import random
import tensorflow as tf
import tensorflow_hub as hub

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

print("TensorFlow:", tf.__version__)
print("TF Hub:", hub.__version__)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
import pandas as pd

train = pd.read_csv("../input/train.tsv", sep="\t")
test = pd.read_csv("../input/test.tsv", sep="\t")

print(train.shape, test.shape)
print(train.columns.tolist())



## === cell 3
train.head()



## === cell 4
train_df = train.sample(frac=1, random_state=seed).reset_index(drop=True)

x_train = train_df["Phrase"].astype(str).values
y_train = train_df["Sentiment"].astype(np.int64).values

x_test = test["Phrase"].astype(str).values

num_classes = 5



## === cell 5
hub_url = "https://tfhub.dev/google/nnlm-en-dim50/1"
hub_layer = hub.KerasLayer(
    hub_url, input_shape=[], dtype=tf.string, trainable=False, name="nnlm_en_dim50"
)

model = tf.keras.Sequential(
    [
        hub_layer,
        tf.keras.layers.Dense(500, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(num_classes),  # logits
    ]
)

optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.003)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.summary()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_10/3321108372.py in <cell line: 0>()
      7 )
      8 
----> 9 model = tf.keras.Sequential(
     10     [
     11         hub_layer,

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in __init__(self, layers, trainable, name)
     73         if layers:
     74             for layer in layers:
---> 75                 self.add(layer, rebuild=False)
     76             self._maybe_rebuild()
     77 

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in add(self, layer, rebuild)
     95                 layer = origin_layer
     96         if not isinstance(layer, Layer):
---> 97             raise ValueError(
     98                 "Only instances of `keras.Layer` can be "
     99                 f"added to a Sequential model. Received: {layer} "

ValueError: Only instances of `keras.Layer` can be added to a Sequential model. Received: <tensorflow_hub.keras_layer.KerasLayer object at 0x7f802b750d50> (of type <class 'tensorflow_hub.keras_layer.KerasLayer'>)

## === cell 6
batch_size = 256
steps = 10000
epochs = int(np.ceil((steps * batch_size) / len(x_train)))
epochs = max(1, epochs)

print("Train rows:", len(x_train), "Batch size:", batch_size, "Approx epochs:", epochs)



## === cell 7
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    verbose=2,
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/483119334.py in <cell line: 0>()
      1 # Train end-to-end.
----> 2 history = model.fit(
      3     x_train,
      4     y_train,
      5     batch_size=batch_size,

NameError: name 'model' is not defined

## === cell 8
train_loss, train_acc = model.evaluate(
    x_train, y_train, batch_size=batch_size, verbose=0
)
print(f"Training set accuracy: {train_acc:.6f}")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/58966626.py in <cell line: 0>()
      1 # Training accuracy (closest analogue to estimator.evaluate on training set).
----> 2 train_loss, train_acc = model.evaluate(
      3     x_train, y_train, batch_size=batch_size, verbose=0
      4 )
      5 print(f"Training set accuracy: {train_acc:.6f}")

NameError: name 'model' is not defined

## === cell 9
sub = pd.read_csv("../input/sampleSubmission.csv")
sub.head()



## === cell 10
logits = model.predict(x_test, batch_size=batch_size, verbose=0)
pred = np.argmax(logits, axis=1).astype(np.int64)

sub = sub.copy()
sub["Sentiment"] = pred

sub.to_csv("sub_tfhub.csv", index=False)
print("Wrote submission:", "sub_tfhub.csv", "shape:", sub.shape)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/2786842441.py in <cell line: 0>()
      1 # Predict sentiments for test set in the same order as sampleSubmission/test.
      2 # Ensure alignment by predicting on test['Phrase'] and then mapping to sub rows by PhraseId order.
----> 3 logits = model.predict(x_test, batch_size=batch_size, verbose=0)
      4 pred = np.argmax(logits, axis=1).astype(np.int64)
      5 

NameError: name 'model' is not defined

## === cell 11
sub.head()
