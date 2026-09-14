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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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

0.3106106317504374

# 6. Current score

0.51119

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.51119) has done: 'I fix the broken file paths so the notebook reads the provided `train.tsv` and `test.tsv` from `/kaggle/input` correctly, which unblocks all downstream cells. I also fix the TensorFlow/Keras import issue by using `tensorflow.keras` only (avoids the `MessageFactory/GetPrototype` crash), while keeping the same model architecture and training loop. Next, I remove hard-coded reshapes that break when the training size changes and ensure inputs to the tokenizer are 1D lists of strings (your current `reshape` on a pandas Series is invalid). Finally, I ensure the submission is written as `submission.csv` with exactly `PhraseId,Sentiment` and aligned row counts.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
TRAIN_PATH = "/kaggle/input/train.tsv"
TEST_PATH = "/kaggle/input/test.tsv"
SAMPLE_SUB_PATH = "/kaggle/input/sampleSubmission.csv"

data = pd.read_csv(TRAIN_PATH, sep="\t")
data.head()



## === cell 2
data.info()



## === cell 3
data.dtypes



## === cell 4
try:
    import seaborn as sns

    sns.catplot(
        y="Sentiment", kind="count", palette="pastel", edgecolor=".6", data=data
    )
except Exception as e:
    print("Skipping seaborn plot due to:", repr(e))



## === cell 5
data["Sentiment"].value_counts()



## === cell 6
y_train = data["Sentiment"]



## === cell 7
first_class = y_train[y_train == 0]
second_class = y_train[y_train == 1]
third_class = y_train[y_train == 2]
forth_class = y_train[y_train == 3]
fifth_class = y_train[y_train == 4]

print(
    "",
    len(first_class),
    "\n",
    len(second_class),
    "\n",
    len(third_class),
    "\n",
    len(forth_class),
    "\n",
    len(fifth_class),
)



## === cell 9
second_class = second_class[0 : len(first_class)]
third_class = third_class[0 : len(first_class)]
forth_class = forth_class[0 : len(first_class)]
fifth_class = fifth_class[0 : len(first_class)]



## === cell 11
text_first_class = data[["Phrase", "Sentiment"]][y_train == 0]
text_second_class = data[["Phrase", "Sentiment"]][y_train == 1]
text_third_class = data[["Phrase", "Sentiment"]][y_train == 2]
text_forth_class = data[["Phrase", "Sentiment"]][y_train == 3]
text_fifth_class = data[["Phrase", "Sentiment"]][y_train == 4]

print(
    "",
    len(text_first_class),
    "\n",
    len(text_second_class),
    "\n",
    len(text_third_class),
    "\n",
    len(text_forth_class),
    "\n",
    len(text_fifth_class),
)



## === cell 12
text_second_class = text_second_class[0 : len(text_first_class)]
text_third_class = text_third_class[0 : len(text_fifth_class)]
text_forth_class = text_forth_class[0 : len(text_first_class)]
text_fifth_class = text_fifth_class[0 : len(text_first_class)]



## === cell 13
frames = [
    text_first_class,
    text_second_class,
    text_third_class,
    text_forth_class,
    text_fifth_class,
]
new_train = pd.concat(frames, axis=0).reset_index(drop=True)



## === cell 14
try:
    import seaborn as sns

    sns.catplot(
        y="Sentiment", kind="count", palette="pastel", edgecolor=".6", data=new_train
    )
except Exception as e:
    print("Skipping seaborn plot due to:", repr(e))



## === cell 15
new_train.head()



## === cell 16
X = new_train["Phrase"].astype(str).values
type(X), X.shape



## === cell 17
y = new_train["Sentiment"].astype(np.int64).values
y.shape



## === cell 18
y.shape, X.shape



## === cell 19
import tensorflow as tf

print("TensorFlow version:", tf.__version__)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 20
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



## === cell 21
vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"



## === cell 22
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X.tolist())

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(X.tolist())
training_padded = pad_sequences(
    training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
)



## === cell 23
training_padded = np.asarray(training_padded, dtype=np.int32)
y = np.asarray(
    y, dtype=np.int64
)  # sparse_categorical_crossentropy expects integer class labels
training_padded.shape, y.shape



## === cell 24
model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(5, activation="softmax"),
    ]
)
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
model.summary()



## === cell 25
num_epochs = 50
history = model.fit(training_padded, y, epochs=num_epochs, verbose=2)



## === cell 26
try:
    import matplotlib.pyplot as plt

    acc = history.history.get("accuracy", [])
    epochs = range(len(acc))

    plt.plot(list(epochs), acc, "r", label="Training accuracy")
    plt.title("Training accuracy")
    plt.legend()
    plt.figure()

    loss = history.history.get("loss", [])
    plt.plot(list(epochs), loss, "b", label="Training Loss")
    plt.title("Training loss")
    plt.legend()

    plt.show()
except Exception as e:
    print("Skipping plots due to:", repr(e))



## === cell 27
test_data = pd.read_csv(TEST_PATH, sep="\t")
test_data.head()



## === cell 28
test_data.info()



## === cell 29
X_test = test_data["Phrase"].astype(str).values
X_test.shape[0]



## === cell 30
type(X_test), X_test.shape



## === cell 31
testing_sequences = tokenizer.texts_to_sequences(X_test.tolist())
testing_padded = pad_sequences(
    testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
)



## === cell 32
testing_padded.shape



## === cell 33
predictions = model.predict(testing_padded, verbose=0)



## === cell 34
predictions[:2]



## === cell 35
prediction = np.argmax(predictions, axis=1).astype(int).tolist()
len(prediction), prediction[:10]



## === cell 36
submission = pd.DataFrame(
    {
        "PhraseId": test_data["PhraseId"].astype(np.int64),
        "Sentiment": np.asarray(prediction, dtype=np.int64),
    }
)

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath("submission.csv"))
