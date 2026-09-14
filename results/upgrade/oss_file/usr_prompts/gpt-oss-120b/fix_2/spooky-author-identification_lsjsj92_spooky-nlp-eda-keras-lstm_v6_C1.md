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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0
wordcloud==1.9.4

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

0.48211

# 6. Current score

0.54233

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.54233) has done: 'The fix updates the Keras imports to use TensorFlow’s stable `tf.keras` API, which resolves the protobuf‑related import error and makes the tokenizer, callbacks, and utility functions available. No core modeling logic is changed, so the training and prediction pipeline remains the same, and a valid CSV submission file is written.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


import os

print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns




## === cell 1
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Input,
    Dropout,
    Bidirectional,
    GlobalMaxPool1D,
    Embedding,
)
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")




## === cell 3
data.head()




## === cell 4
data.author.value_counts().plot(kind="bar")




## === cell 5
data_length = data.text.apply(len)
data_length.head()




## === cell 6
plt.figure(figsize=(12, 5))
plt.hist(data_length, bins=20, range=[0, 500], color="r", alpha=0.3)
plt.show()




## === cell 7
data_split_length = data.text.apply(lambda x: len(x.split(" ")))
data_split_length.head()




## === cell 8
plt.figure(figsize=(12, 5))
plt.hist(data_split_length, bins=10, range=[0, 100], color="g", alpha=0.5)
plt.show()




## === cell 9
print("data_length max : ", np.max(data_length))
print("data_length min : ", np.min(data_length))
print("data_length mean : ", np.mean(data_length))
print("data_length 75% : ", np.percentile(data_length, 75))
print("data_length 90% : ", np.percentile(data_length, 90))




## === cell 10
print("data_split_length max : ", np.max(data_split_length))
print("data_split_length min : ", np.min(data_split_length))
print("data_split_length mean : ", np.mean(data_split_length))
print("data_split_length 75% : ", np.percentile(data_split_length, 75))
print("data_split_length 90% : ", np.percentile(data_split_length, 90))




## === cell 11
cloud = WordCloud(width=400, height=200).generate(" ".join(data.text))
plt.figure(figsize=(12, 5))
plt.imshow(cloud)
plt.axis("off")




## === cell 12
cloud = WordCloud(width=400, height=200).generate(
    " ".join(data[data["author"] == "HPL"]["text"])
)
plt.figure(figsize=(12, 5))
plt.imshow(cloud)
plt.axis("off")




## === cell 13
cloud = WordCloud(width=400, height=200).generate(
    " ".join(data[data["author"] == "MWS"]["text"])
)
plt.figure(figsize=(12, 5))
plt.imshow(cloud)
plt.axis("off")




## === cell 14
cloud = WordCloud(width=400, height=200).generate(
    " ".join(data[data["author"] == "EAP"]["text"])
)
plt.figure(figsize=(12, 5))
plt.imshow(cloud)
plt.axis("off")




## === cell 15
le = LabelEncoder()
le.fit(data.author)
y = le.transform(data.author)




## === cell 16
y[:10]




## === cell 17
y = to_categorical(y)




## === cell 18
y[:10]




## === cell 19
num_words = 5000
max_len = 50
emb_size = 64




## === cell 20
tok = Tokenizer(num_words=num_words)
tok.fit_on_texts(list(data.text))




## === cell 21
X = tok.texts_to_sequences(data.text)
X_test = tok.texts_to_sequences(test_data.text)




## === cell 22
X = sequence.pad_sequences(X, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)




## === cell 23
X[0]




## === cell 24
def model():
    inp = Input(shape=(max_len,))
    layer = Embedding(num_words, emb_size)(inp)
    layer = Bidirectional(LSTM(50, return_sequences=True, recurrent_dropout=0.2))(layer)
    layer = GlobalMaxPool1D()(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(16, activation="relu")(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(3, activation="softmax")(layer)
    model = Model(inputs=inp, outputs=layer)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model




## === cell 25
model = model()
model.summary()




## === cell 26
early_stop = EarlyStopping(monitor="val_loss", patience=1)




## === cell 27
hist = model.fit(
    X, y, batch_size=32, epochs=3, validation_split=0.2, callbacks=[early_stop]
)




## === cell 28
vloss = hist.history["val_loss"]
loss = hist.history["loss"]

x_len = np.arange(len(loss))

plt.plot(x_len, vloss, marker=".", color="r", label="val_loss")
plt.plot(x_len, loss, marker=".", color="b", label="loss")
plt.legend()
plt.grid()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()




## === cell 29
results = model.predict(X_test)
ids = test_data["id"]
results = pd.DataFrame(results, columns=["EAP", "HPL", "MWS"])
results.insert(0, "id", ids)
results.head()




## === cell 30
results.to_csv("my_submission.csv", index=False)
