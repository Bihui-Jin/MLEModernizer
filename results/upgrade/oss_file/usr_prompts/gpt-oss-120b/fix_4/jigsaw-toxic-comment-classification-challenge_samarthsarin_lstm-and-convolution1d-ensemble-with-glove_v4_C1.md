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

3.7

# 3. Installed packages

No external packages required in the script and installed.

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

0.7678853274770532

# 6. Current score

0.93699

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.93355) has done: 'The updates import TensorFlow Keras (instead of the legacy keras), download needed NLTK resources, add a safe fallback when the GloVe file is missing, replace CuDNNLSTM with standard LSTM, and switch the output layer to sigmoid with binary‑crossentropy (the correct loss for multi‑label toxicity). These fixes unblock all imports, allow tokenisation, model creation and training to run, and finally write a properly‑formatted `submission.csv` file.'
- What this solution (achieved 0.93081) has done: 'I set the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` environment variable before importing TensorFlow to avoid the protobuf “MessageFactory” error, and I renumbered the notebook cells to start at 1 so the script conforms to the required format. No other logic changes are made; the model, training, and submission steps remain identical, ensuring the same high score while now producing a valid `submission.csv`.'
- What this solution (achieved 0.93699) has done: 'The fix adds a second setting of the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` environment variable right before importing TensorFlow/Keras in cell 1, guaranteeing the protobuf implementation is forced to the pure‑Python version even if any earlier import triggered TensorFlow loading. This resolves the `'MessageFactory' object has no attribute 'GetPrototype'` error while keeping the original model, training, and submission logic unchanged, so the high score remains unchanged and a proper `submission.csv` is produced.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import numpy as np, pandas as pd

print(os.listdir("../input"))




## === cell 1
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm
import nltk

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")




## === cell 3
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = df[list_classes].values




## === cell 4
stop_words = list(set(stopwords.words("english"))) + list(punctuation) + ["\n"]
lem = WordNetLemmatizer()




## === cell 5
tqdm.pandas()




## === cell 6
embedding_path = "../input/gloveembeddings/glove.6B.50d.txt"
embedding_values = {}
if os.path.exists(embedding_path):
    with open(embedding_path, "r", encoding="utf8") as f:
        for line in tqdm(f, desc="Loading GloVe"):
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.array(parts[1:], dtype="float32")
            embedding_values[word] = vec
else:
    print("GloVe file not found – using random embeddings.")




## === cell 7
token = Tokenizer()
token.fit_on_texts(df["comment_text"])
seq = token.texts_to_sequences(df["comment_text"])
pad_seq = pad_sequences(seq, maxlen=100)




## === cell 8
vocab_size = len(token.word_index) + 1
print("Vocab size:", vocab_size)




## === cell 9
embedding_dim = 50
embedding_matrix = np.random.normal(size=(vocab_size, embedding_dim)).astype("float32")
if embedding_values:
    for word, i in token.word_index.items():
        vec = embedding_values.get(word)
        if vec is not None:
            embedding_matrix[i] = vec




## === cell 10
model = Sequential()
model.add(
    Embedding(
        vocab_size,
        embedding_dim,
        input_length=100,
        weights=[embedding_matrix],
        trainable=False,
    )
)
model.add(Bidirectional(LSTM(50, return_sequences=False)))
model.add(Dense(50, activation="relu"))
model.add(Dense(6, activation="sigmoid"))  # sigmoid for multi‑label




## === cell 11
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(pad_seq, y, epochs=2, batch_size=32, validation_split=0.1)




## === cell 12
test = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv")
x_test = test["comment_text"]
test_seq = token.texts_to_sequences(x_test)
test_pad_seq = pad_sequences(test_seq, maxlen=100)




## === cell 13
predict = model.predict(test_pad_seq, batch_size=256)




## === cell 14
sample_submission = pd.read_csv(
    "../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
)
sample_submission[list_classes] = predict
sample_submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
