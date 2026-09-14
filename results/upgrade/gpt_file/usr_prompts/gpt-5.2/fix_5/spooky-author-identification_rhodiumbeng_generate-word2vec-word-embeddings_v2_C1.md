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
Given some text, predict the author.

## Metric
Multi-class logarithmic loss. 

The submitted probabilities for a given sentences are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum).

In order to avoid the extremes of the log function, predicted probabilities are replaced with \\(max(min(p,1-10^{-15}),10^{-15})\\).

## Submission Format
You must submit a csv file with the id, and a probability for each of the three classes. The order of the rows does not matter. The file must have a header and should look like the following:

```
id,EAP,HPL,MWS
id07943,0.33,0.33,0.33
...
```

## Dataset 
### File descriptions
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Data fields
- **id** - a unique identifier for each sentence
- **text** - some text written by one of the authors
- **author** - the author of the sentence (EAP: Edgar Allan Poe, HPL: HP Lovecraft; MWS: Mary Wollstonecraft Shelley)

# 2. Python version

3.7

# 3. Installed packages

gensim==4.4.0
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
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        input/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        working/
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
```

-> data/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/spooky-author-identification/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> data/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> input/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> (stopped after 10 files for performance)

# 5. Target score

0.40443

# 6. Current score

1.28576

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.49359) has done: 'I fix the import/runtime issues that prevent the notebook from running in Kaggle (remove unused gensim/nltk pieces that trigger the protobuf `MessageFactory` error, and remove the IPython `%matplotlib inline` magic). I also fix the `re` scoping issue by keeping `re` properly imported where `clean_text()` is used, and fix the history keys (`accuracy` vs `acc`) to avoid the plotting KeyError in newer Keras. To nudge logloss toward your target (lower is better) without changing the model architecture or training approach, I make the train/dev split stratified so class balance is consistent and reduce training randomness via fixed seeds; this is typically a small, legitimate improvement. Paths be made robust to your provided `/kaggle/input/...` layout while still working if `../input/...` exists, and the script always write a valid `.csv` submission with the required columns.'
- What this solution (achieved 1.09875) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by avoiding the TensorFlow/`tf_keras` import path that triggers the protobuf incompatibility in this Kaggle image, and instead use `keras` (v3) with the NumPy backend, which keeps the same dense-network architecture and training loop semantics. I also keep the vectorization/training logic the same, but remove the redundant second model training (which was unintentionally degrading performance) by reusing the already-trained model for evaluation and prediction. Finally, I ensure the submission file is always written with the required `.csv` suffix and correct columns. These changes are minimal, unblock end-to-end execution, and should improve logloss toward your target by not throwing away the 20-epoch trained weights.'
- What this solution (achieved 1.30324) has done: 'We need to switch off the Keras NumPy backend because `model.fit()` is not implemented there, which is the direct cause of the runtime failure and prevents generating a valid submission. The minimal fix is to use the TensorFlow-backed Keras (`tf_keras`) for training while keeping the exact same vectorization, network architecture, loss, optimizer, epochs, and batch size. I also add a small safety fallback to ensure the correct train/test paths resolve to your provided `/kaggle/input/...` layout and keep the submission columns/order exactly as required. This should both run end-to-end and substantially improve logloss from the near-uniform 1.098 baseline toward your target.'
- What this solution (achieved 1.28576) has done: 'I fix the immediate runtime crash caused by importing `tf_keras` in this environment (protobuf `MessageFactory.GetPrototype` incompatibility) by switching to `keras` v3 with the TensorFlow backend, which keeps the exact same model architecture, optimizer/loss, epochs, and training loop. I also add a tiny backend check to ensure training actually runs on TensorFlow (since the NumPy backend can’t train). Finally, I keep your existing data loading/vectorization/prediction flow intact and ensure the submission is always written as a valid `.csv` with the required `id,EAP,HPL,MWS` columns.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import keras
from keras import models, layers

SEED = 123
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

print("Ready. Using keras version:", getattr(keras, "__version__", "unknown"))
try:
    print("Keras backend:", keras.backend.backend())
except Exception as e:
    print("Could not read Keras backend:", repr(e))

if hasattr(keras, "backend") and hasattr(keras.backend, "backend"):
    if keras.backend.backend() != "tensorflow":
        raise RuntimeError(
            f"Expected KERAS_BACKEND=tensorflow but got {keras.backend.backend()}. "
            "Set env KERAS_BACKEND=tensorflow before importing keras."
        )




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _resolve_input_path(filename: str) -> str:
    candidates = [
        os.path.join("/kaggle/input", filename),
        os.path.join("/kaggle/input/spooky-author-identification", filename),
        os.path.join("../input", filename),
        os.path.join("../input/spooky-author-identification", filename),
        os.path.join("..", "input", filename),
        os.path.join("/kaggle/data", filename),
        os.path.join("/kaggle/data/spooky-author-identification", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


train_path = _resolve_input_path("train.csv")
test_path = _resolve_input_path("test.csv")
sample_path = _resolve_input_path("sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print(train_df.shape, test_df.shape)
print("Train columns:", train_df.columns.tolist())
print("Test columns:", test_df.columns.tolist())



## === cell 2
train_df.head()



## === cell 3
train_df["author"].value_counts()



## === cell 4
train_df["text_length"] = train_df["text"].str.len()



## === cell 5
ax = train_df["text_length"].hist(bins=50)
plt.title("Train text_length")
plt.show()



## === cell 6
test_df["text_length"] = test_df["text"].str.len()
ax = test_df["text_length"].hist(bins=50)
plt.title("Test text_length")
plt.show()



## === cell 7
train_df["author"] = pd.Categorical(train_df["author"])
df_Dummies = pd.get_dummies(train_df["author"], prefix="author")
train_df = pd.concat([train_df, df_Dummies], axis=1)
train_df.head()




## === cell 8
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\W", " ", text)
    return text




## === cell 9
train_df["text"] = train_df["text"].map(lambda com: clean_text(com))



## === cell 10
test_df["text"] = test_df["text"].map(lambda com: clean_text(com))



## === cell 11
X = train_df["text"]
y = train_df[["author_EAP", "author_HPL", "author_MWS"]]



## === cell 12
from sklearn.model_selection import train_test_split

strat = train_df["author"]
X_train, X_dev, y_train, y_dev = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=strat
)
print(X_train.shape, y_train.shape, X_dev.shape, y_dev.shape)



## === cell 13
print(y_train.sum(axis=0), "\n", y_dev.sum(axis=0))



## === cell 14
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vect = CountVectorizer()
vect



## === cell 15
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm = X_train_dtm.toarray()
X_train_dtm



## === cell 16
print(X_train_dtm.shape)



## === cell 17
X_dev_dtm = vect.transform(X_dev)
X_dev_dtm = X_dev_dtm.toarray()
X_dev_dtm



## === cell 18
print(X_train_dtm.shape, y_train.shape)
print(X_dev_dtm.shape, y_dev.shape)



## === cell 19
num_input_shape = X_train_dtm.shape[1]
num_class = y_train.shape[1]



## === cell 20
model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(num_input_shape,)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(num_class, activation="softmax"))



## === cell 21
model.summary()



## === cell 22
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)



## === cell 23
history = model.fit(
    X_train_dtm,
    y_train,
    epochs=20,
    batch_size=512,
    validation_data=(X_dev_dtm, y_dev),
    verbose=2,
)



## === cell 24
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.plot(epochs, val_loss, "b", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()



## === cell 25
plt.clf()
acc = history.history.get("accuracy", None)
val_acc = history.history.get("val_accuracy", None)

if acc is not None and val_acc is not None:
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, "bo", label="Training Accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
else:
    print(
        "Accuracy history keys not found. Available keys:", list(history.history.keys())
    )



## === cell 26
results = model.evaluate(X_dev_dtm, y_dev, verbose=0)
print(results)



## === cell 27
test = test_df["text"]
test_dtm = vect.transform(test)
test_dtm = test_dtm.toarray()
test_dtm



## === cell 28
print(test_dtm.shape)



## === cell 29
dnn_predictions = model.predict(test_dtm, verbose=0)
print(dnn_predictions.shape)



## === cell 30
print(dnn_predictions[:10])



## === cell 31
result = pd.DataFrame(dnn_predictions, columns=["EAP", "HPL", "MWS"])
result.insert(0, "id", test_df["id"].values)
result.head()



## === cell 32
out_path = "rhodium_submission_17.csv"
result.to_csv(out_path, index=False, float_format="%.20f")
print("Wrote submission:", out_path, "shape:", result.shape)
print(result.head())
