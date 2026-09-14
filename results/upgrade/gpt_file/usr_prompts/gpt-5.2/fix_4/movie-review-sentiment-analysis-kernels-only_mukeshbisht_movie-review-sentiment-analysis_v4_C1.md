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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import random
import numpy as np
import pandas as pd

import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)
print("Input dir listing:", os.listdir("../input"))



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
from tensorflow import keras
from tensorflow.keras import layers




## === cell 7
def mlp_model(layers_count, units, dropout_rate, input_shape, num_classes):
    """Creates an instance of a multi-layer perceptron model."""
    model = keras.Sequential()
    model.add(layers.Dropout(dropout_rate, input_shape=input_shape, name="drop_input"))

    for i in range(layers_count - 1):
        model.add(layers.Dense(units=units, activation="relu", name=f"dense_{i+1}"))
        model.add(layers.Dropout(rate=dropout_rate, name=f"drop_{i+1}"))

    model.add(layers.Dense(units=num_classes, activation="softmax", name="output"))
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
    layers_count=2,
    units=64,
    dropout_rate=0.2,
):
    """Trains n-gram model on the given dataset."""
    num_classes = 5

    x_train, x_val = X_train, X_validate
    model = mlp_model(
        layers_count=layers_count,
        units=units,
        dropout_rate=dropout_rate,
        input_shape=x_train.shape[1:],
        num_classes=num_classes,
    )

    loss = "sparse_categorical_crossentropy"
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    y_train = np.asarray(y_train).astype(np.int64).reshape(-1)
    y_validate = np.asarray(y_validate).astype(np.int64).reshape(-1)

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
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
    if val_acc_key is not None:
        print(f"Validation accuracy: {h[val_acc_key][-1]}, loss: {h['val_loss'][-1]}")
    else:
        print(f"Validation loss: {h['val_loss'][-1]}")

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

/tmp/ipykernel_11/2733563572.py in train_ngram_model(X_train, y_train, X_validate, y_validate, learning_rate, epochs, batch_size, layers_count, units, dropout_rate)
     32     y_validate = np.asarray(y_validate).astype(np.int64).reshape(-1)
     33 
---> 34     history = model.fit(
     35         x_train,
     36         y_train,

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
  (0) INVALID_ARGUMENT:  Error in user-defined function passed to ParallelMapDatasetV2:22 transformation with iterator: Iterator::Root::Prefetch::ParallelMapV2: indices[81] = 36049 is not in [0, 36049)
	 [[{{node RaggedGather_1/RaggedGather}}]]
	 [[IteratorGetNext]]
	 [[DeserializeSparse/_2]]
  (1) INVALID_ARGUMENT:  Error in user-defined function passed to ParallelMapDatasetV2:22 transformation with iterator: Iterator::Root::Prefetch::ParallelMapV2: indices[81] = 36049 is not in [0, 36049)
	 [[{{node RaggedGather_1/RaggedGather}}]]
	 [[IteratorGetNext]]
0 successful operations.
0 derived errors ignored. [Op:__inference_multi_step_on_iterator_4101]

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
    {
        "PhraseId": test_data["PhraseId"].astype(np.int64),
        "Sentiment": y_class.astype(int),
    }
)

assert list(my_submission.columns) == ["PhraseId", "Sentiment"]
assert len(my_submission) == len(test_data), (len(my_submission), len(test_data))
assert my_submission["Sentiment"].between(0, 4).all()

my_submission.to_csv("submission.csv", index=False)
print(my_submission.head())
print("Wrote submission.csv with shape:", my_submission.shape)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2797558768.py in <cell line: 0>()
      2     {
      3         "PhraseId": test_data["PhraseId"].astype(np.int64),
----> 4         "Sentiment": y_class.astype(int),
      5     }
      6 )

NameError: name 'y_class' is not defined
