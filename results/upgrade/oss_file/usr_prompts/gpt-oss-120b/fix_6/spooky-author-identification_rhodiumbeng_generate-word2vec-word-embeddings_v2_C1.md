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

1.43187

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.96439) has done: 'I replace the incompatible Keras imports with TensorFlow’s Keras, fix the missing `re` import by ensuring the first cell runs, correct the history keys for accuracy, switch to a TF‑IDF vectorizer with bi‑grams (which usually improves text classification), and increase training epochs slightly to lower the log‑loss toward the target. All other logic remains unchanged, and the script now writes a proper `*.csv` submission file.'
- What this solution (achieved 0.726) has done: 'Implemented a fix for the TensorFlow import error by switching to the standalone Keras package, and enhanced the model capacity and training regime to improve validation loss. Adjustments include increasing TF‑IDF features to 50 000, expanding Dense layers to 128 units, using the Adam optimizer, and training for 60 epochs. These changes keep the original workflow intact while aiming to lower the log‑loss toward the target score.'
- What this solution (achieved 0.74931) has done: 'I fix the import error by switching to TensorFlow’s Keras (`tensorflow.keras`) which is compatible with the installed packages, and I extend the training epochs from 60 to 80 to modestly lower the validation log‑loss and move the score toward the target. All other logic remains unchanged.'
- What this solution (achieved 1.44724) has done: 'I removed the problematic `tensorflow` import that caused the protobuf error and switched to using only Keras modules. The train‑validation split now uses `stratify` to keep class distribution balanced, which improves validation loss. I enlarged the hidden layers to 256 units, added Dropout layers for regularization, and extended training to 120 epochs to let the model converge better. These minimal, targeted changes fix the runtime error and are expected to lower the log‑loss toward the target score while keeping the original workflow intact.'
- What this solution (achieved 1.43187) has done: 'I replace the failing TensorFlow Keras import with the standalone Keras package (which is installed) so the notebook runs without the protobuf error. No other logic is altered, preserving the existing model and workflow and ensuring a proper CSV submission is created.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

from keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print(train_df.shape, test_df.shape)




## === cell 2
print(train_df.head())




## === cell 3
print(train_df["author"].value_counts())




## === cell 4
train_df["text_length"] = train_df["text"].str.len()




## === cell 5
train_df.hist()
plt.show()




## === cell 6
test_df["text_length"] = test_df["text"].str.len()
test_df.hist()
plt.show()




## === cell 7
train_df["author"] = pd.Categorical(train_df["author"])
df_Dummies = pd.get_dummies(train_df["author"], prefix="author")
train_df = pd.concat([train_df, df_Dummies], axis=1)




## === cell 8
def clean_text(text):
    text = text.lower()
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
X_train, X_dev, y_train, y_dev = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=train_df["author"]
)
print(X_train.shape, y_train.shape, X_dev.shape, y_dev.shape)




## === cell 13
print(y_train.sum(axis=0), "\n", y_dev.sum(axis=0))




## === cell 14
vect = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
X_train_dtm = vect.fit_transform(X_train).toarray()
print(X_train_dtm.shape)




## === cell 15
X_dev_dtm = vect.transform(X_dev).toarray()
print(X_dev_dtm.shape)




## === cell 16
num_input_shape = X_train_dtm.shape[1]
num_class = y_train.shape[1]




## === cell 17
model = models.Sequential()
model.add(layers.Dense(256, activation="relu", input_shape=(num_input_shape,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_class, activation="softmax"))




## === cell 18
model.summary()




## === cell 19
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])




## === cell 20
history = model.fit(
    X_train_dtm,
    y_train,
    epochs=120,
    batch_size=512,
    validation_data=(X_dev_dtm, y_dev),
    verbose=2,
)




## === cell 21
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




## === cell 22
plt.clf()
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, "bo", label="Training Accuracy")
plt.plot(epochs, val_acc, "b", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()




## === cell 23
results = model.evaluate(X_dev_dtm, y_dev, verbose=0)
print("Validation loss, accuracy:", results)




## === cell 24
test_text = test_df["text"]
test_dtm = vect.transform(test_text).toarray()
print(test_dtm.shape)




## === cell 25
dnn_predictions = model.predict(test_dtm)
print(dnn_predictions.shape)




## === cell 26
print(dnn_predictions[:10])




## === cell 27
result = pd.DataFrame(dnn_predictions, columns=["EAP", "HPL", "MWS"])
result.insert(0, "id", test_df["id"])
result.head()




## === cell 28
result.to_csv("rhodium_submission_17.csv", index=False, float_format="%.20f")
