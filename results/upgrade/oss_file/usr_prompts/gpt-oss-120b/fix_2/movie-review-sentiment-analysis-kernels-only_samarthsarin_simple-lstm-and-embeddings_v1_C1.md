# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os, numpy as np, pandas as pd

print(os.listdir("../input"))



## === cell 1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm



## === cell 2
train = pd.read_csv(
    "../input/movie-review-sentiment-analysis-kernels-only/train.tsv", sep="\t"
)
test = pd.read_csv(
    "../input/movie-review-sentiment-analysis-kernels-only/test.tsv", sep="\t"
)
sub = pd.read_csv(
    "../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv"
)



## === cell 3
df = train[["Phrase", "Sentiment"]]



## === cell 4
token = Tokenizer()
x_text = df["Phrase"].astype(str).tolist()
token.fit_on_texts(x_text)
seq = token.texts_to_sequences(x_text)
pad_seq = pad_sequences(seq, maxlen=300)

y = to_categorical(df["Sentiment"].values, num_classes=5)

vocab_size = len(token.word_index) + 1
print("Vocab size:", vocab_size)



## === cell 5
embedding_dim = 300
embedding_layer = Embedding(
    input_dim=vocab_size, output_dim=embedding_dim, input_length=300, trainable=True
)



## === cell 6
x_train, x_val, y_train, y_val = train_test_split(
    pad_seq, y, test_size=0.3, random_state=42
)



## === cell 7
model = Sequential()
model.add(embedding_layer)
model.add(LSTM(75))  # CuDNNLSTM replaced by regular LSTM (CPU compatible)
model.add(Dense(128, activation="relu"))
model.add(Dense(5, activation="softmax"))



## === cell 8
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    x_train, y_train, batch_size=32, epochs=4, validation_data=(x_val, y_val), verbose=2
)



## === cell 9
test_phrases = test["Phrase"].astype(str).tolist()
test_seq = token.texts_to_sequences(test_phrases)
pad_test_seq = pad_sequences(test_seq, maxlen=300)

pred_probs = model.predict(pad_test_seq, batch_size=32, verbose=0)
predict = np.argmax(pred_probs, axis=1)



## === cell 10
submission = pd.DataFrame({"PhraseId": test["PhraseId"], "Sentiment": predict})
submission.to_csv("Submission.csv", index=False)
print("Submission saved to Submission.csv")
