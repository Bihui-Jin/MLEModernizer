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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.60031

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(os.listdir("../input"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_path = os.path.join("../input", "train.tsv")
test_data_path = os.path.join("../input", "test.tsv")
data = pd.read_csv(data_path, sep="\t")
test_data = pd.read_csv(test_data_path, sep="\t")
data.describe()



## === cell 2
data.head()



## === cell 3
from sklearn.model_selection import train_test_split

train_texts = data["Phrase"]
train_labels = np.array(data["Sentiment"])
test_texts = test_data["Phrase"]

X_train, X_val, y_train, y_val = train_test_split(
    train_texts, train_labels, test_size=0.33, random_state=42, stratify=train_labels
)
data = ((X_train, np.array(y_train)), (X_val, np.array(y_val)))



## === cell 4
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

NGRAM_RANGE = (1, 3)
TOP_K = 5000
TOKEN_MODE = "word"
MIN_DOCUMENT_FREQUENCY = 2




## === cell 5
def ngram_vectorize(train_texts, train_labels, val_texts, test_texts):
    """Vectorizes texts as n-gram vectors."""
    kwargs = {
        "ngram_range": NGRAM_RANGE,
        "dtype": "int32",
        "strip_accents": "unicode",
        "decode_error": "replace",
        "analyzer": TOKEN_MODE,
        "min_df": MIN_DOCUMENT_FREQUENCY,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)
    x_test = vectorizer.transform(test_texts)

    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)

    x_train = selector.transform(x_train).astype("float32")
    x_val = selector.transform(x_val).astype("float32")
    x_test = selector.transform(x_test).astype("float32")
    return x_train, x_val, x_test




## === cell 6
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout




## === cell 7
def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    """Creates an instance of a multi-layer perceptron model."""
    op_units, op_activation = (
        5,
        "sigmoid",
    )  # keep original variables to preserve core logic intent
    model = models.Sequential()
    model.add(Dropout(dropout_rate, input_shape=input_shape))

    for _ in range(layers - 1):
        model.add(Dense(units=units, activation="relu", name="d1"))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation="softmax", name="d2"))
    return model




## === cell 8
def train_ngram_model(
    X_train,
    y_train,
    X_validate,
    y_validate,
    learning_rate=1e-3,
    epochs=20,
    batch_size=128,
    layers=2,
    units=64,
    dropout_rate=0.2,
):
    """Trains n-gram model on the given dataset."""
    num_classes = 5

    x_train, x_val = X_train, X_validate
    model = mlp_model(
        layers=layers,
        units=units,
        dropout_rate=dropout_rate,
        input_shape=x_train.shape[1:],
        num_classes=num_classes,
    )

    loss = "sparse_categorical_crossentropy"

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)]

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(x_val, y_validate),
        verbose=2,
        batch_size=batch_size,
    )

    h = history.history
    val_acc_key = (
        "val_accuracy"
        if "val_accuracy" in h
        else ("val_acc" if "val_acc" in h else None)
    )
    val_loss_key = "val_loss"
    if val_acc_key is not None:
        print(
            "Validation accuracy: {acc}, loss: {loss}".format(
                acc=h[val_acc_key][-1], loss=h[val_loss_key][-1]
            )
        )
    else:
        print("Validation loss: {loss}".format(loss=h[val_loss_key][-1]))

    return model




## === cell 9
(train_texts, train_labels), (val_texts, val_labels) = data
x_train, x_val, x_test = ngram_vectorize(
    train_texts, train_labels, val_texts, test_texts
)
model = train_ngram_model(x_train, train_labels, x_val, val_labels)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/313129574.py in <cell line: 0>()
      3     train_texts, train_labels, val_texts, test_texts
      4 )
----> 5 model = train_ngram_model(x_train, train_labels, x_val, val_labels)
      6 

/tmp/ipykernel_11/3624127907.py in train_ngram_model(X_train, y_train, X_validate, y_validate, learning_rate, epochs, batch_size, layers, units, dropout_rate)
     29 
     30     # Use 'accuracy' so history keys are stable across TF/Keras versions.
---> 31     model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
     32 
     33     callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/keras/engine/training.py in compile(self, optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)
    566       self._run_eagerly = run_eagerly
    567 
--> 568       self.optimizer = self._get_optimizer(optimizer)
    569       self.compiled_loss = compile_utils.LossesContainer(
    570           loss, loss_weights, output_names=self.output_names)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/keras/engine/training.py in _get_optimizer(self, optimizer)
    604       return opt
    605 
--> 606     return nest.map_structure(_get_single_optimizer, optimizer)
    607 
    608   @trackable.no_automatic_dependency_tracking

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/nest.py in map_structure(func, *structure, **kwargs)
    626     ValueError: If wrong keyword arguments are provided.
    627   """
--> 628   return nest_util.map_structure(
    629       nest_util.Modality.CORE, func, *structure, **kwargs
    630   )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/nest_util.py in map_structure(modality, func, *structure, **kwargs)
   1063   """
   1064   if modality == Modality.CORE:
-> 1065     return _tf_core_map_structure(func, *structure, **kwargs)
   1066   elif modality == Modality.DATA:
   1067     return _tf_data_map_structure(func, *structure, **kwargs)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/nest_util.py in _tf_core_map_structure(func, *structure, **kwargs)
   1103   return _tf_core_pack_sequence_as(
   1104       structure[0],
-> 1105       [func(*x) for x in entries],
   1106       expand_composites=expand_composites,
   1107   )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/nest_util.py in <listcomp>(.0)
   1103   return _tf_core_pack_sequence_as(
   1104       structure[0],
-> 1105       [func(*x) for x in entries],
   1106       expand_composites=expand_composites,
   1107   )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/keras/engine/training.py in _get_single_optimizer(opt)
    595 
    596     def _get_single_optimizer(opt):
--> 597       opt = optimizers.get(opt)
    598       if (loss_scale is not None and
    599           not isinstance(opt, lso.LossScaleOptimizer)):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/keras/optimizers.py in get(identifier)
    125     return deserialize(config)
    126   else:
--> 127     raise ValueError(
    128         'Could not interpret optimizer identifier: {}'.format(identifier))

ValueError: Could not interpret optimizer identifier: <keras.src.optimizers.adam.Adam object at 0x7f969fea8c10>

## === cell 10
y_test_pred = model.predict(x_test, batch_size=256, verbose=0)
y_class = np.argmax(y_test_pred, axis=1)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3287622573.py in <cell line: 0>()
----> 1 y_test_pred = model.predict(x_test, batch_size=256, verbose=0)
      2 y_class = np.argmax(y_test_pred, axis=1)
      3 

NameError: name 'model' is not defined

## === cell 11
my_submission = pd.DataFrame(
    {"PhraseId": test_data.PhraseId, "Sentiment": y_class.astype(int)}
)
my_submission.to_csv("submission.csv", index=False)
print(my_submission.head())
print("Wrote submission.csv with shape:", my_submission.shape)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/124651338.py in <cell line: 0>()
      1 my_submission = pd.DataFrame(
----> 2     {"PhraseId": test_data.PhraseId, "Sentiment": y_class.astype(int)}
      3 )
      4 my_submission.to_csv("submission.csv", index=False)
      5 print(my_submission.head())

NameError: name 'y_class' is not defined
