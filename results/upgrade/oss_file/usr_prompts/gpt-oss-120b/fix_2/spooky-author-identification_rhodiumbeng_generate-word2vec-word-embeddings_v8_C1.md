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

0.5357

# 6. Current score

0.87389

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.87389) has done: 'I remove the invalid `%matplotlib inline` line, update the Word2Vec constructor to use the correct `vector_size` argument, and switch the Keras imports to TensorFlow’s Keras to avoid the protobuf error. These fixes let the pipeline run, produce embeddings, train the neural network, and finally write a proper `submission.csv` file.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
import gensim
from gensim.models import Word2Vec



## === cell 1
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print(train_df.shape, test_df.shape)




## === cell 2
def clean_text(text):
    """
    Convert all to lowercase and remove punctuations
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove everything that isn't word or space
    text = re.sub(r"_", "", text)  # remove underscore
    return text




## === cell 3
train_df["text"] = train_df["text"].map(lambda x: clean_text(x))
train_df["text"] = train_df["text"].map(lambda x: x.strip().split())
train_df.head()



## === cell 4
test_df["text"] = test_df["text"].map(lambda x: clean_text(x))
test_df["text"] = test_df["text"].map(lambda x: x.strip().split())
test_df.head()



## === cell 5
data = []
for i in range(len(train_df)):
    data.append(train_df["text"][i])
for j in range(len(test_df)):
    data.append(test_df["text"][j])



## === cell 6
embedding = Word2Vec(
    sentences=data, vector_size=50, window=10, min_count=1, sg=0  # CBOW
)



## === cell 7
print(embedding)



## === cell 8
pass



## === cell 9
words = list(embedding.wv.key_to_index.keys())
print(f"Vocabulary size: {len(words)}")



## === cell 10
print(embedding.wv["capered"][:5])  # show first five components as a sanity check



## === cell 11
print(embedding.wv.most_similar("dark", topn=5))



## === cell 12
print(embedding.wv.most_similar("shocked", topn=5))



## === cell 13
print(embedding.wv.most_similar("sprang", topn=5))



## === cell 14
print(embedding.wv.most_similar("pride", topn=5))



## === cell 15
train_df["author"] = pd.Categorical(train_df["author"])
df_Dummies = pd.get_dummies(train_df["author"], prefix="author")
train_df = pd.concat([train_df, df_Dummies], axis=1)
train_df.head()



## === cell 16
X = train_df["text"].str[:50]
Y = train_df[["author_EAP", "author_HPL", "author_MWS"]].values
print(X.shape, X.iloc[0], Y.shape, Y[0])



## === cell 17
X_test = test_df["text"].str[:50]
print(X_test.shape, X_test.iloc[0])




## === cell 18
def text_to_avg(text):
    """
    Given a list of words, extract the respective Word2Vec vectors
    and average them into a single 50‑dimensional representation.
    """
    avg = np.zeros((50,))
    for w in text:
        avg += embedding.wv[w]
    avg = avg / len(text)
    return avg




## === cell 19
X_avg = np.zeros((X.shape[0], 50))
for i in range(X.shape[0]):
    X_avg[i] = text_to_avg(X[i])



## === cell 20
print(X_avg.shape)
print(X_avg[0][:5])



## === cell 21
X_test_avg = np.zeros((X_test.shape[0], 50))
for i in range(X_test.shape[0]):
    X_test_avg[i] = text_to_avg(X_test[i])



## === cell 22
print(X_test_avg.shape)
print(X_test_avg[0][:5])



## === cell 23
from sklearn.model_selection import train_test_split

X_train, X_dev, Y_train, Y_dev = train_test_split(
    X_avg, Y, test_size=0.2, random_state=123
)
print(X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape)



## === cell 24
from tensorflow.keras import models, layers

model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(50,)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(3, activation="softmax"))

model.summary()



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 25
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



## === cell 26
epochs = 50
history = model.fit(
    X_train,
    Y_train,
    epochs=epochs,
    batch_size=128,
    validation_data=(X_dev, Y_dev),
    verbose=0,
)



## === cell 27
loss = history.history["loss"]
dev_loss = history.history["val_loss"]
epoch_range = range(1, len(loss) + 1)
plt.plot(epoch_range, loss, "bo", label="training loss")
plt.plot(epoch_range, dev_loss, "b", label="validation loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()



## === cell 28
model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(50,)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(3, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



## === cell 29
model.fit(X_avg, Y, epochs=10, batch_size=128, verbose=0)



## === cell 30
preds = model.predict(X_test_avg)
print(preds.shape)
print(preds[7])



## === cell 31
pred_labels = [np.argmax(p) for p in preds]



## === cell 32
print(pred_labels[7])



## === cell 33
result = pd.DataFrame(preds, columns=["EAP", "HPL", "MWS"])
result.insert(0, "id", test_df["id"])
result.head()



## === cell 34
result.to_csv("submission.csv", index=False, float_format="%.20f")
