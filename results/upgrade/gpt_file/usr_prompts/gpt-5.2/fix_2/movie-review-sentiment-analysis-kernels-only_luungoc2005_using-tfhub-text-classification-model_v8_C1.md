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
import numpy as np
import pandas as pd
import os

print(os.listdir("../input"))



## === cell 1
seed = 197

import random
import tensorflow as tf
import tensorflow_hub as hub

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

print("TF:", tf.__version__)
print("TF Hub:", hub.__version__)

os.environ["PYTHONHASHSEED"] = str(seed)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train_df = pd.read_csv("../input/train.tsv", sep="\t")
test_df = pd.read_csv("../input/test.tsv", sep="\t")

train_df.head()



## === cell 3
x_text = train_df["Phrase"].astype(str).values
y = train_df["Sentiment"].astype("int32").values

x_test_text = test_df["Phrase"].astype(str).values

num_classes = 5



## === cell 4
hub_url = "https://tfhub.dev/google/nnlm-en-dim50-with-normalization/1"
text_embedding = hub.KerasLayer(
    hub_url, input_shape=[], dtype=tf.string, trainable=False, name="nnlm50"
)



## === cell 5
model = tf.keras.Sequential(
    [
        text_embedding,
        tf.keras.layers.Dense(500, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.003)

model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.summary()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2027550189.py in <cell line: 0>()
      1 # Model mirrors the original DNNClassifier hidden_units=[500, 100] on top of the embedding.
      2 # Use sparse_categorical_crossentropy since labels are integer class IDs 0..4.
----> 3 model = tf.keras.Sequential(
      4     [
      5         text_embedding,

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

ValueError: Only instances of `keras.Layer` can be added to a Sequential model. Received: <tensorflow_hub.keras_layer.KerasLayer object at 0x7f1e58f612d0> (of type <class 'tensorflow_hub.keras_layer.KerasLayer'>)

## === cell 6
steps = 10000
batch_size = 256

steps_per_epoch = int(np.ceil(len(x_text) / batch_size))
epochs = int(np.ceil(steps / steps_per_epoch))

print(
    "N:",
    len(x_text),
    "batch_size:",
    batch_size,
    "steps_per_epoch:",
    steps_per_epoch,
    "epochs:",
    epochs,
)



## === cell 7
history = model.fit(
    x_text, y, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3734557928.py in <cell line: 0>()
----> 1 history = model.fit(
      2     x_text, y, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2
      3 )
      4 

NameError: name 'model' is not defined

## === cell 8
train_loss, train_acc = model.evaluate(x_text, y, batch_size=batch_size, verbose=0)
print(f"Training set accuracy: {train_acc:.6f}")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3337224275.py in <cell line: 0>()
----> 1 train_loss, train_acc = model.evaluate(x_text, y, batch_size=batch_size, verbose=0)
      2 print(f"Training set accuracy: {train_acc:.6f}")
      3 

NameError: name 'model' is not defined

## === cell 9
sub = pd.read_csv("../input/sampleSubmission.csv")
sub.head()



## === cell 10
probs = model.predict(x_test_text, batch_size=batch_size, verbose=0)
pred = np.argmax(probs, axis=1).astype(int)

sub["Sentiment"] = pred

out_path = "sub_tfhub.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1938817004.py in <cell line: 0>()
      1 # Predict labels for the test set and write a valid Kaggle submission.
----> 2 probs = model.predict(x_test_text, batch_size=batch_size, verbose=0)
      3 pred = np.argmax(probs, axis=1).astype(int)
      4 
      5 # Ensure alignment by PhraseId (submission expects PhraseId from sampleSubmission)

NameError: name 'model' is not defined

## === cell 11
assert list(sub.columns) == ["PhraseId", "Sentiment"]
assert sub["Sentiment"].between(0, 4).all()
assert sub.shape[0] == test_df.shape[0]
sub.head()
