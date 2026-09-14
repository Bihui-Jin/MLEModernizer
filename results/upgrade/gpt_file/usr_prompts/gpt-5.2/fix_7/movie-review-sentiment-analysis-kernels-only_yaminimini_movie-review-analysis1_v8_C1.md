# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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

0.17406

# 6. Current score

0.58994

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50372) has done: 'I fix the runtime errors by avoiding `todense()` (which creates `np.matrix` and breaks scikit-learn) and by keeping the TF-IDF/CountVectorizer outputs as sparse CSR matrices end-to-end. I also stop the later Keras/LSTM section from crashing the run (it currently has multiple incompatible API calls and a catastrophic `pad_sequences` memory blow-up) while preserving the earlier core modeling approach (TF-IDF + MultinomialNB) so we reliably produce a valid submission CSV. Finally, I ensure the output file is written as a proper `.csv` with exactly the required `PhraseId,Sentiment` columns and aligned row order.'
- What this solution (achieved 0.51711) has done: 'I remove the intentional `sys.exit(0)` so the notebook finishes cleanly, but I keep the TF‑IDF + MultinomialNB pipeline as the final submission writer to preserve the current (already strong) score behavior. I also prevent the later experimental sections (CountVectorizer dense conversion and broken Keras API calls) from executing by guarding them, since they currently crash due to `todense()`/memory blow-ups and Keras 3 API incompatibilities. Finally, I ensure the submission is written exactly once as `submission.csv` with `PhraseId,Sentiment` aligned to `test.tsv` order.'
- What this solution (achieved 0.58994) has done: 'I fix the pipeline break at `TfidfVectorizer.fit()` by correcting the overly aggressive `min_df/max_df` settings that prune all terms, which currently prevents the model from fitting and cascades into later NameErrors. I keep the core approach identical (TF‑IDF features + MultinomialNB classifier) and ensure we transform both train/test with the same fitted vectorizer. Finally, I guarantee a valid `submission.csv` is written with exactly `PhraseId,Sentiment` in the same row order as `test.tsv`, while leaving the later experimental blocks disabled as they are.'
- What this solution (achieved 0.58994) has done: 'I fix the failure where TF‑IDF prunes all terms by making the fallback progressively less aggressive until a vocabulary is learned, instead of having two settings that can both fail. This unblocks the downstream cells (transform/train/predict) without changing the core modeling approach (TF‑IDF features + MultinomialNB). I also make the file paths robust to either `../input` or `/kaggle/input` layouts so it runs in your provided environment and always writes `submission.csv` with the required `PhraseId,Sentiment` columns. The experimental blocks stay disabled exactly as before.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import os

CANDIDATE_INPUT_DIRS = [
    "../input",
    "/kaggle/input",
    "/kaggle/data",
]


def _resolve_input_dir(candidates):
    for d in candidates:
        if os.path.isdir(d):
            return d
    return "."


INPUT_DIR = _resolve_input_dir(CANDIDATE_INPUT_DIRS)
print("Using INPUT_DIR =", INPUT_DIR)
print("Top-level entries:", os.listdir(INPUT_DIR)[:20])



## === cell 1
import pandas as pd




## === cell 2
def _find_file(filename):
    direct = os.path.join(INPUT_DIR, filename)
    if os.path.isfile(direct):
        return direct
    for entry in os.listdir(INPUT_DIR):
        p = os.path.join(INPUT_DIR, entry, filename)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Could not find {filename} under {INPUT_DIR}")


train_path = _find_file("train.tsv")
train = pd.read_csv(train_path, sep="\t")
train_path



## === cell 3
train.head()



## === cell 4
test_path = _find_file("test.tsv")
test = pd.read_csv(test_path, sep="\t")
test_path



## === cell 5
test.head()



## === cell 6
train["Sentiment"].unique()



## === cell 7
train.shape



## === cell 8
test.shape



## === cell 9
train.isnull().sum(axis=0)



## === cell 10
test.isnull().sum(axis=0)



## === cell 11
train["SentenceId"].value_counts()[0:5]



## === cell 12
test["SentenceId"].value_counts()[0:5]



## === cell 13
len(train["SentenceId"].unique()) + len(test["SentenceId"].unique())



## === cell 14
len(train["PhraseId"].unique()) + len(test["PhraseId"].unique())



## === cell 15
from sklearn.feature_extraction.text import TfidfVectorizer



## === cell 16
tfidf_candidates = [
    dict(
        analyzer="word",
        stop_words="english",
        min_df=0.30,
        max_df=0.60,
        ngram_range=(1, 1),
    ),
    dict(
        analyzer="word",
        stop_words="english",
        min_df=0.20,
        max_df=0.70,
        ngram_range=(1, 1),
    ),
    dict(
        analyzer="word",
        stop_words="english",
        min_df=0.10,
        max_df=0.85,
        ngram_range=(1, 1),
    ),
    dict(
        analyzer="word", stop_words="english", min_df=2, max_df=0.90, ngram_range=(1, 1)
    ),
    dict(
        analyzer="word", stop_words="english", min_df=1, max_df=1.00, ngram_range=(1, 1)
    ),
]

tfidf = None
last_err = None
for i, params in enumerate(tfidf_candidates, start=1):
    try:
        tfidf = TfidfVectorizer(**params)
        tfidf.fit(train["Phrase"].astype(str))
        print(f"TF-IDF fit succeeded with candidate {i}: {params}")
        break
    except ValueError as e:
        last_err = e
        print(f"TF-IDF candidate {i} failed: {e}")

if tfidf is None:
    raise last_err



## === cell 17
train_tfidf = tfidf.transform(train["Phrase"].astype(str))



## === cell 18
X_train = train_tfidf  # CSR sparse matrix



## === cell 19
Y_train = train["Sentiment"]



## === cell 20
from sklearn.naive_bayes import MultinomialNB



## === cell 21
NB = MultinomialNB()



## === cell 22
NB.fit(X_train, Y_train)



## === cell 23
test_tfidf = tfidf.transform(test["Phrase"].astype(str))



## === cell 24
x_test = test_tfidf  # CSR sparse matrix



## === cell 25
x_test.shape



## === cell 26
y_pred = NB.predict(x_test)



## === cell 27
type(y_pred)



## === cell 28
y_pred_df = pd.DataFrame(y_pred, columns=["Sentiment"])



## === cell 29
y_pred_df.head()



## === cell 30
sub = pd.concat([test["PhraseId"], y_pred_df], axis=1)



## === cell 31
sub.head()



## === cell 32
sub = sub.rename(columns={"PhraseId": "PhraseId", "Sentiment": "Sentiment"})
sub["PhraseId"] = sub["PhraseId"].astype(int)
sub["Sentiment"] = sub["Sentiment"].astype(int)
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())



## === cell 33
RUN_EXPERIMENTS = False



## === cell 34
if RUN_EXPERIMENTS:
    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.tokenize import RegexpTokenizer



## === cell 35
if RUN_EXPERIMENTS:
    pattern = RegexpTokenizer(r"[a-zA-Z0-9]+")



## === cell 36
if RUN_EXPERIMENTS:
    cv = CountVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 1),
        tokenizer=pattern.tokenize,
    )



## === cell 37
if RUN_EXPERIMENTS:
    cv.fit(train["Phrase"])



## === cell 38
if RUN_EXPERIMENTS:
    train_cv = cv.transform(train["Phrase"])



## === cell 39
if RUN_EXPERIMENTS:
    train_cv



## === cell 40
if RUN_EXPERIMENTS:
    X_train2 = train_cv.todense()



## === cell 41
if RUN_EXPERIMENTS:
    Y_train2 = train["Sentiment"]



## === cell 42
if RUN_EXPERIMENTS:
    from sklearn.linear_model import SGDClassifier

    sv = SGDClassifier()



## === cell 43
if RUN_EXPERIMENTS:
    from sklearn.linear_model import SGDClassifier



## === cell 44
if RUN_EXPERIMENTS:
    sv = SGDClassifier(max_iter=200)



## === cell 45
if RUN_EXPERIMENTS:
    sv.fit(X_train, Y_train)



## === cell 46
if RUN_EXPERIMENTS:
    y_pred2 = sv.predict(x_test)



## === cell 47
if RUN_EXPERIMENTS:
    y_pred2_df = pd.DataFrame(y_pred2, columns=["Sentiment"])



## === cell 48
if RUN_EXPERIMENTS:
    sub2 = pd.concat([test["PhraseId"], y_pred2_df], axis=1)



## === cell 49
if RUN_EXPERIMENTS:
    sub2.head()



## === cell 50
if RUN_EXPERIMENTS:
    from keras.preprocessing.text import Tokenizer



## === cell 51
if RUN_EXPERIMENTS:
    X_train = train["Phrase"]



## === cell 52
if RUN_EXPERIMENTS:
    train.dtypes



## === cell 53
if RUN_EXPERIMENTS:
    from keras.utils import to_categorical



## === cell 54
if RUN_EXPERIMENTS:
    Y_train = to_categorical(train["Sentiment"].values)



## === cell 55
if RUN_EXPERIMENTS:
    Y_train.shape



## === cell 56
if RUN_EXPERIMENTS:
    tz = Tokenizer(num_words=10000, lower=True)



## === cell 57
if RUN_EXPERIMENTS:
    tz.fit_on_texts(list(X_train))



## === cell 58
if RUN_EXPERIMENTS:
    X_train2 = tz.texts_to_sequences(X_train)



## === cell 59
if RUN_EXPERIMENTS:
    type(X_train2)



## === cell 60
if RUN_EXPERIMENTS:
    len(X_train2)



## === cell 61
if RUN_EXPERIMENTS:
    from keras.preprocessing.sequence import pad_sequences

    X_train2 = pad_sequences(X_train2, maxlen=100)



## === cell 62
if RUN_EXPERIMENTS:
    X.shape



## === cell 63
if RUN_EXPERIMENTS:
    random = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]



## === cell 64
if RUN_EXPERIMENTS:
    tz.fit_on_texts(random)



## === cell 65
if RUN_EXPERIMENTS:
    random



## === cell 66
if RUN_EXPERIMENTS:
    r = tz.texts_to_sequences(random)



## === cell 67
if RUN_EXPERIMENTS:
    r



## === cell 68
if RUN_EXPERIMENTS:
    pad_sequences(r, maxlen=4)



## === cell 69
if RUN_EXPERIMENTS:
    X_test2 = tz.texts_to_sequences(test["Phrase"])



## === cell 70
if RUN_EXPERIMENTS:
    X_test2 = pad_sequences(X_test2, maxlen=100)



## === cell 71
if RUN_EXPERIMENTS:
    from sklearn.model_selection import train_test_split

    seed = 42
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y_train, test_size=0.20, random_state=seed
    )



## === cell 72
if RUN_EXPERIMENTS:
    from keras.layers import Dense, Dropout, Embedding, LSTM
    from keras.losses import categorical_crossentropy
    from keras.optimizers import Adam
    from keras.models import Sequential



## === cell 73
if RUN_EXPERIMENTS:
    model = Sequential()
    model.add(Embedding(10000, 100, mask_zero=True))
    model.add(LSTM(64, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
    model.add(LSTM(32, dropout=0.5, recurrent_dropout=0.5, return_sequences=False))
    model.add(Dense(5, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )



## === cell 74
if RUN_EXPERIMENTS:
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=4, batch_size=32)



## === cell 75
if RUN_EXPERIMENTS:
    sub2["Sentiment"] = np.argmax(
        model.predict(X_test2, batch_size=32, verbose=1), axis=1
    )



## === cell 76
if RUN_EXPERIMENTS:
    sub2.head()



## === cell 77
if RUN_EXPERIMENTS:
    sub2.to_csv("submission3.csv", index=False)
