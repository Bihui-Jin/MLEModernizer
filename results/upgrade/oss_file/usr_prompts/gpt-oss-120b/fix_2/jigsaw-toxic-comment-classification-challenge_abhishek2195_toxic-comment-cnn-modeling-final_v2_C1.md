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
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Target score

0.96744

# 6. Current score

0.47965

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.47965) has done: 'Implemented fixes:
- Replaced incompatible `keras` imports with `tensorflow.keras` equivalents to resolve protobuf errors.
- Added proper import for `Tokenizer` and other utilities.
- Used absolute paths for train/test CSVs, removing reliance on unzip commands.
- Ensured tokenizer, sequences, padding, and vocabulary creation run after imports.
- Fixed multilabel stratified split, model construction, compilation, callbacks, and loading of best weights.
- Added explicit loading of the best model before predictions.
- Built the submission DataFrame with all required columns in correct order and saved it as `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import matplotlib.pyplot as plt
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Embedding,
    Dropout,
    Conv1D,
    GlobalMaxPooling1D,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from sklearn.metrics import roc_auc_score
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv"
test_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv"



## === cell 3
df_train = pd.read_csv(train_path)
print("Shape=>", df_train.shape)
df_train.head()



## === cell 4
df_test = pd.read_csv(test_path)
print("Shape=>", df_test.shape)
df_test.head()



## === cell 5
for col in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
    print(df_train[col].value_counts(normalize=True) * 100)



## === cell 6
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
for ax, class_name in zip(
    axes.flatten(),
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
):
    pd.value_counts(df_train[class_name], sort=True).plot(kind="bar", rot=0, ax=ax)
    ax.set_title(f"{class_name} Distribution")
    ax.set_xticks([0, 1])
    ax.set_xlabel("Labels")
    ax.set_ylabel("Frequency")
plt.show()




## === cell 7
def cleaner(text):
    text = text.lower()
    text = re.sub("[^a-z]+", " ", text)
    text = re.sub("[ ]+", " ", text)
    return text




## === cell 8
df_train["cleaned"] = df_train["comment_text"].apply(cleaner)



## === cell 9
df_test["cleaned"] = df_test["comment_text"].apply(cleaner)



## === cell 10
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train["cleaned"])
print("Vocabulary Size=>", len(tokenizer.word_index))



## === cell 11
train_seq = tokenizer.texts_to_sequences(df_train["cleaned"])
test_seq = tokenizer.texts_to_sequences(df_test["cleaned"])



## === cell 12
MAXLEN = 100
train_seq = pad_sequences(train_seq, maxlen=MAXLEN, padding="post")
test_seq = pad_sequences(test_seq, maxlen=MAXLEN, padding="post")
print("Shape of train_sequence=>", train_seq.shape)
print("Shape of test_sequence=>", test_seq.shape)



## === cell 13
y_train = df_train[
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
].values
print("Shape of Training Labels=>", y_train.shape)



## === cell 14
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
except Exception:
    import subprocess, sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "iterative-stratification"]
    )
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit



## === cell 15
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)



## === cell 16
for train_idx, val_idx in msss.split(train_seq, y_train):
    x_train_split, y_train_split = train_seq[train_idx], y_train[train_idx]
    x_valid_split, y_valid_split = train_seq[val_idx], y_train[val_idx]

print("Shape of Train Split=>", x_train_split.shape, y_train_split.shape)
print("Shape of Validation Split=>", x_valid_split.shape, y_valid_split.shape)



## === cell 17
vocabulary = len(tokenizer.word_index) + 1
print("Vocabulary Size=>", vocabulary)



## === cell 18
input_1 = Input(shape=(MAXLEN,))
embedding_1 = Embedding(vocabulary, 100)(input_1)
conv_1 = Conv1D(filters=352, kernel_size=7, padding="same")(embedding_1)
dropout_1 = Dropout(0.06675)(conv_1)
pool_1 = GlobalMaxPooling1D()(dropout_1)

dense = Dense(128, activation="relu")(pool_1)
output = Dense(6, activation="sigmoid")(dense)

model = Model(inputs=[input_1], outputs=output)
model.summary()



## === cell 19
plot_model(model, to_file="model.png", show_shapes=True)



## === cell 20
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])



## === cell 21
es = EarlyStopping(
    monitor="val_loss", mode="min", verbose=1, patience=3, min_delta=1e-5
)
mc = ModelCheckpoint(
    "/kaggle/working/model.hdf5",
    monitor="val_loss",
    verbose=0,
    save_best_only=True,
    mode="min",
)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2643520224.py in <cell line: 0>()
      2     monitor="val_loss", mode="min", verbose=1, patience=3, min_delta=1e-5
      3 )
----> 4 mc = ModelCheckpoint(
      5     "/kaggle/working/model.hdf5",
      6     monitor="val_loss",

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    192                 self.filepath.endswith(ext) for ext in (".keras", ".h5")
    193             ):
--> 194                 raise ValueError(
    195                     "The filepath provided must end in `.keras` "
    196                     "(Keras model format). Received: "

ValueError: The filepath provided must end in `.keras` (Keras model format). Received: filepath=/kaggle/working/model.hdf5

## === cell 22
model.fit(
    x_train_split,
    y_train_split,
    batch_size=512,
    epochs=20,
    verbose=1,
    validation_data=(x_valid_split, y_valid_split),
    callbacks=[es, mc],
)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3045891681.py in <cell line: 0>()
      6     verbose=1,
      7     validation_data=(x_valid_split, y_valid_split),
----> 8     callbacks=[es, mc],
      9 )
     10 

NameError: name 'mc' is not defined

## === cell 23
model.load_weights("/kaggle/working/model.hdf5")



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3093212727.py in <cell line: 0>()
      1 # Load best weights before evaluation
----> 2 model.load_weights("/kaggle/working/model.hdf5")
      3 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '/kaggle/working/model.hdf5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 24
train_pred = model.predict(x_train_split, verbose=0)
print("In-sample ROC-AUC:", roc_auc_score(y_train_split, train_pred, average="macro"))



## === cell 25
valid_pred = model.predict(x_valid_split, verbose=0)
print("Validation ROC-AUC:", roc_auc_score(y_valid_split, valid_pred, average="macro"))



## === cell 26
final_pred = model.predict(test_seq, verbose=0)



## === cell 27
prob = pd.DataFrame(df_test["id"])
prob.columns = ["id"]
for idx, col in enumerate(
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
):
    prob[col] = final_pred[:, idx]



## === cell 28
prob.head()



## === cell 29
prob.to_csv("/kaggle/working/submission.csv", index=False)
