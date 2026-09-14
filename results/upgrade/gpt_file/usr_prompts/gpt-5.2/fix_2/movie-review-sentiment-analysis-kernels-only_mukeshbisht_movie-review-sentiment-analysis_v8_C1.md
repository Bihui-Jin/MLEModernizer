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

0.60769

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
import matplotlib.pyplot as plt

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf

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

X_train, X_test, y_train, y_test = train_test_split(
    train_texts, train_labels, test_size=0.33, random_state=42
)

y_temp = y_train

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

data = ((X_train, np.array(y_train)), (X_test, np.array(y_test)))



## === cell 4
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

NGRAM_RANGE = (1, 3)
TOP_K = 9000
TOKEN_MODE = "word"
MIN_DOCUMENT_FREQUENCY = 5




## === cell 5
def ngram_vectorize(train_texts, train_labels, val_texts, test_texts):
    """Vectorizes texts as n-gram vectors."""
    stop_words_lst = list(text.ENGLISH_STOP_WORDS.union(["'s"]))

    kwargs = {
        "ngram_range": NGRAM_RANGE,
        "dtype": "int32",
        "strip_accents": "unicode",
        "decode_error": "replace",
        "analyzer": TOKEN_MODE,
        "min_df": MIN_DOCUMENT_FREQUENCY,
        "stop_words": stop_words_lst,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)
    x_test = vectorizer.transform(test_texts)

    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, y_temp)

    x_train = selector.transform(x_train).astype("float32")
    x_val = selector.transform(x_val).astype("float32")
    x_test = selector.transform(x_test).astype("float32")
    return x_train, x_val, x_test




## === cell 6
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout




## === cell 7
def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    """Creates an instance of a multi-layer perceptron model."""
    print("input shape : ", input_shape)
    model = models.Sequential()
    model.add(Dropout(dropout_rate, input_shape=input_shape))

    for _ in range(layers - 1):
        model.add(Dense(units=units, activation="relu"))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=64, activation="relu"))
    model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=num_classes, activation="softmax", name="d2"))
    return model




## === cell 8
def train_ngram_model(
    X_train,
    X_label,
    X_validate,
    val_label,
    learning_rate=1e-3,
    epochs=50,
    batch_size=150,
    layers=3,
    units=128,
    dropout_rate=0.2,
):
    """Trains n-gram model on the given dataset."""
    num_classes = 5

    x_train, x_val = X_train, X_validate
    val_labels = val_label

    model = mlp_model(
        layers=layers,
        units=units,
        dropout_rate=dropout_rate,
        input_shape=x_train.shape[1:],
        num_classes=num_classes,
    )
    loss = "categorical_crossentropy"

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6)]
    history_obj = model.fit(
        x_train,
        X_label,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(x_val, val_labels),
        verbose=2,
        batch_size=batch_size,
    )

    history = history_obj.history
    val_acc_key = "val_acc" if "val_acc" in history else "val_accuracy"
    print(
        "Validation accuracy: {acc}, loss: {loss}".format(
            acc=history[val_acc_key][-1], loss=history["val_loss"][-1]
        )
    )
    return model




## === cell 9
(train_texts, train_labels), (val_texts, val_labels) = data
x_train, x_val, x_test = ngram_vectorize(
    train_texts, train_labels, val_texts, test_texts
)
model = train_ngram_model(x_train, train_labels, x_val, val_labels)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/313129574.py in <cell line: 0>()
      3     train_texts, train_labels, val_texts, test_texts
      4 )
----> 5 model = train_ngram_model(x_train, train_labels, x_val, val_labels)
      6 

/tmp/ipykernel_11/2897605255.py in train_ngram_model(X_train, X_label, X_validate, val_label, learning_rate, epochs, batch_size, layers, units, dropout_rate)
     31 
     32     callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6)]
---> 33     history_obj = model.fit(
     34         x_train,
     35         X_label,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

InvalidArgumentError: Graph execution error:

Detected at node RaggedGather_1/RaggedGather defined at (most recent call last):
<stack traces unavailable>
Detected at node RaggedGather_1/RaggedGather defined at (most recent call last):
<stack traces unavailable>
2 root error(s) found.
  (0) INVALID_ARGUMENT:  Error in user-defined function passed to ParallelMapDatasetV2:7 transformation with iterator: Iterator::Root::Prefetch::ParallelMapV2: indices[139] = 73191 is not in [0, 73191)
	 [[{{node RaggedGather_1/RaggedGather}}]]
	 [[IteratorGetNext]]
	 [[IteratorGetNext/_8]]
  (1) INVALID_ARGUMENT:  Error in user-defined function passed to ParallelMapDatasetV2:7 transformation with iterator: Iterator::Root::Prefetch::ParallelMapV2: indices[139] = 73191 is not in [0, 73191)
	 [[{{node RaggedGather_1/RaggedGather}}]]
	 [[IteratorGetNext]]
0 successful operations.
0 derived errors ignored. [Op:__inference_multi_step_on_iterator_2235]

## === cell 10
y_pred_proba = model.predict(x_test, verbose=0)
y_class = np.argmax(y_pred_proba, axis=1)

my_submission = pd.DataFrame({"PhraseId": test_data.PhraseId, "Sentiment": y_class})
my_submission.to_csv("submission.csv", index=False)
print(my_submission.head())
print("Wrote submission.csv with shape:", my_submission.shape)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2770800185.py in <cell line: 0>()
----> 1 y_pred_proba = model.predict(x_test, verbose=0)
      2 y_class = np.argmax(y_pred_proba, axis=1)
      3 
      4 my_submission = pd.DataFrame({"PhraseId": test_data.PhraseId, "Sentiment": y_class})
      5 my_submission.to_csv("submission.csv", index=False)

NameError: name 'model' is not defined
