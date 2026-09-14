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

0.32619

# 6. Current score

0.63296

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.63296) has done: 'I fixed the import errors, replaced the deprecated `sklearn.cross_validation` with `sklearn.model_selection`, switched to the modern TensorFlow Keras API, and ensured all preprocessing functions are defined before they are used. The script now loads the training data, preprocesses the text (tokenization, stemming, bigram/trigram generation), splits it, builds and trains a simple CNN model, predicts sentiments for the test set, and writes a valid `output.csv` with the required `PhraseId,Sentiment` columns.'

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd, nltk
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gensim
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

nltk.download("punkt", quiet=True)

print("Files in input folder:", os.listdir("../input"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = "../input/train.tsv"
cols = ["PhraseId", "SentenceId", "Phrase", "Sentiment"]
train_df = pd.read_csv(train_path, sep="\t", header=0, names=cols)
print("Train shape:", train_df.shape)
X_raw = train_df["Phrase"].astype(str).tolist()
Y_raw = train_df["Sentiment"].astype(int).tolist()



## === cell 2
ps = PorterStemmer()


def stem_text(text):
    tokens = nltk.word_tokenize(text.lower())
    return " ".join([ps.stem(tok) for tok in tokens])


X_stemmed = [stem_text(txt) for txt in X_raw]



## === cell 3
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X_stemmed, Y_raw, test_size=0.2, random_state=1, stratify=Y_raw
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.25, random_state=1, stratify=y_tmp
)  # 0.25 * 0.8 = 0.2

print(len(X_train), len(X_val), len(X_test))




## === cell 4
def build_phrase_models(texts):
    tokenized = [nltk.word_tokenize(t) for t in texts]
    bigram = gensim.models.Phrases(tokenized, min_count=5, threshold=10)
    trigram = gensim.models.Phrases(bigram[tokenized], threshold=10)
    return gensim.models.phrases.Phraser(bigram), gensim.models.phrases.Phraser(trigram)


bigram_mod, trigram_mod = build_phrase_models(X_train)


def apply_ngrams(texts):
    tokenized = [nltk.word_tokenize(t) for t in texts]
    trigged = [trigram_mod[bigram_mod[doc]] for doc in tokenized]
    return [" ".join(doc) for doc in trigged]


X_train_ng = apply_ngrams(X_train)
X_val_ng = apply_ngrams(X_val)
X_test_ng = apply_ngrams(X_test)



## === cell 5
max_len = max(len(s.split()) for s in X_train_ng)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train_ng)


def texts_to_padded(texts):
    seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=max_len, padding="post")


train_seq = texts_to_padded(X_train_ng)
val_seq = texts_to_padded(X_val_ng)
test_seq = texts_to_padded(X_test_ng)



## === cell 6
le = LabelEncoder()
le.fit(y_train)
y_train_enc = to_categorical(le.transform(y_train), num_classes=5)
y_val_enc = to_categorical(le.transform(y_val), num_classes=5)

vocab_size = len(tokenizer.word_index) + 1



## === cell 7
model = Sequential(
    [
        Embedding(
            input_dim=vocab_size, output_dim=100, input_length=max_len, trainable=True
        ),
        Conv1D(128, kernel_size=2, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(5, activation="softmax"),
    ]
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
model.fit(
    train_seq,
    y_train_enc,
    validation_data=(val_seq, y_val_enc),
    epochs=5,
    batch_size=128,
    verbose=1,
)



## === cell 8
val_pred = np.argmax(model.predict(val_seq), axis=1)
val_acc = np.mean(val_pred == y_val)
print(f"Validation accuracy: {val_acc:.4f}")



## === cell 9
test_path = "../input/test.tsv"
test_cols = ["PhraseId", "SentenceId", "Phrase"]
test_df = pd.read_csv(test_path, sep="\t", header=0, names=test_cols)

test_stem = [stem_text(txt) for txt in test_df["Phrase"].astype(str).tolist()]
test_ng = apply_ngrams(test_stem)
test_seq = texts_to_padded(test_ng)



## === cell 10
test_pred = np.argmax(model.predict(test_seq), axis=1)
submission = pd.DataFrame({"PhraseId": test_df["PhraseId"], "Sentiment": test_pred})
submission.to_csv("output.csv", index=False)
print("Submission saved to output.csv, head:")
print(submission.head())
