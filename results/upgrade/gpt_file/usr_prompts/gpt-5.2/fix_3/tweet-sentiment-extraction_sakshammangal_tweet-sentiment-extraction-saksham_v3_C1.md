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
Predict the word or phrase from tweets that exemplifies the labelled sentiment.

## Metric
Word-level Jaccard score.

## Submission Format
For each ID in the test set, you must predict the string that best supports the sentiment for the tweet in question. Note that the selected text _needs_ to be **quoted** and **complete** (include punctuation, etc. - the above code splits ONLY on whitespace) to work correctly. The file should contain a header and have the following format:
```
textID,selected_text
2,"very good"
5,"I don't care"
6,"bad"
8,"it was, yes"
etc.
```

## Dataset
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

- `textID` - unique ID for each piece of text
- `text` - the text of the tweet
- `sentiment` - the general sentiment of the tweet
- `selected_text` - [train only] the text that supports the tweet's sentiment

# 2. Python version

3.9

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        input/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        working/
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
```

-> data/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> data/tweet-sentiment-extraction/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/tweet-sentiment-extraction/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/tweet-sentiment-extraction/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> input/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> (stopped after 10 files for performance)

# 5. Target score

0.54158

# 6. Current score

0.39171

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.39171) has done: 'I fix the runtime error caused by indexing a pandas Series with `docs[0]` inside `TfidfEmbeddingVectorizer.transform`, by switching to position-safe access and converting Series to a list/array before inspection. This allow cell 32 to complete, which in turn defines `nb`, `vectorizer`, and downstream variables so cells 33–35 stop raising `NameError`. I also ensure the submission uses the required columns (`textID`, `selected_text`), writes to `submission.csv`, and that test text is cleaned consistently with training so inference matches the trained representation (score-neutral-to-slightly-positive but still aligned with the existing approach). No changes are made to the model/training core logic (Word2Vec + TFIDF-weighted embeddings + custom NB-like selector); only bug fixes and consistency fixes are applied.'

# 9. Code solution

## === cell 0
import os, re, math, random, string, warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases

import matplotlib.pyplot as plt
import seaborn as sns

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:50]:
        print(os.path.join(dirname, filename))



## === cell 1
train_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
train_df.head(8)



## === cell 2
print(train_df.shape)
print(test_df.shape)



## === cell 3
train_df.info()



## === cell 4
print(train_df[train_df["text"].isnull()])
print(train_df[train_df["selected_text"].isnull()])



## === cell 5
train_df.dropna(inplace=True)
train_df = train_df.reset_index(drop=True)



## === cell 6
train_df.info()



## === cell 7
train_df.describe()



## === cell 8
train_df["N_text_words"] = train_df["text"].apply(lambda tweet: len(str(tweet).split()))
train_df["N_selected_text_words"] = train_df["selected_text"].apply(
    lambda tweet: len(str(tweet).split())
)
train_df["N_words_difference"] = (
    train_df["N_text_words"] - train_df["N_selected_text_words"]
)
train_df.head(8)



## === cell 9
print(
    "There are {0} unique sentiments having values {1}".format(
        train_df["sentiment"].nunique(), train_df["sentiment"].unique()
    )
)



## === cell 10
n_neutral = train_df["sentiment"].loc[train_df["sentiment"] == "neutral"].count()
n_positive = train_df["sentiment"].loc[train_df["sentiment"] == "positive"].count()
n_negative = train_df["sentiment"].loc[train_df["sentiment"] == "negative"].count()

print(f"Neutral tweets : {n_neutral}")
print(f"Positive tweets : {n_positive}")
print(f"Negative tweets : {n_negative}")



## === cell 11
try:
    import plotly.graph_objects as go

    sentiments = ["Neutral", "Positive", "Negative"]
    fig = go.Figure(
        data=[go.Pie(labels=sentiments, values=[n_neutral, n_positive, n_negative])]
    )
    fig.show()
except Exception as e:
    print("Plotly unavailable or failed; skipping pie chart. Error:", str(e)[:200])




## === cell 12
def jaccard(str1, str2):
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    if len(a) == 0 and len(b) == 0:
        return 1.0
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


jaccard_score = []
for i in range(train_df.shape[0]):
    str1 = str(train_df["text"][i]).strip()
    str2 = str(train_df["selected_text"][i]).strip()
    jaccard_score.append(jaccard(str1, str2))

train_df["Jaccard_score"] = jaccard_score
train_df.head(8)




## === cell 13
def clean_text(text):
    """Lowercase, remove bracketed text, URLs, HTML, punctuation, newlines, and alphanumeric tokens with digits."""
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text




## === cell 14
try:
    import nltk
    from nltk.corpus import stopwords

    STOPWORDS = stopwords.words("english")
except Exception:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    STOPWORDS = list(ENGLISH_STOP_WORDS)


def remove_stopwords(text):
    return [word for word in str(text).split() if word not in STOPWORDS]


train_df["text_cleaned"] = train_df["text"].apply(
    lambda x: remove_stopwords(clean_text(x))
)
train_df["selected_text_cleaned"] = train_df["selected_text"].apply(
    lambda x: remove_stopwords(clean_text(x))
)



## === cell 15
train_df.head(8)




## === cell 16
def get_all_words(df_col):
    all_words_text = []
    for row in df_col:
        for word in row:
            all_words_text.append(word)
    return all_words_text


all_words_text = get_all_words(train_df["text_cleaned"])
all_words_selected_text = get_all_words(train_df["selected_text_cleaned"])



## === cell 17
all_words_neutral = get_all_words(
    train_df[train_df["sentiment"] == "neutral"]["text_cleaned"]
)
all_words_positive = get_all_words(
    train_df[train_df["sentiment"] == "positive"]["text_cleaned"]
)
all_words_negative = get_all_words(
    train_df[train_df["sentiment"] == "negative"]["text_cleaned"]
)




## === cell 18
def plot_wordcloud(all_words):
    try:
        from wordcloud import WordCloud

        stopwords_set = set(STOPWORDS)
        more_stopwords = {"u", "im"}
        stopwords_set = stopwords_set.union(more_stopwords)
        all_words_str = " ".join(all_words)
        wordcloud = WordCloud(
            width=400,
            height=200,
            background_color="white",
            max_words=200,
            stopwords=stopwords_set,
            min_font_size=10,
        ).generate(all_words_str)
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()
    except Exception as e:
        print("WordCloud unavailable or failed; skipping. Error:", str(e)[:200])




## === cell 19
plot_wordcloud(all_words_positive)



## === cell 20
plot_wordcloud(all_words_negative)



## === cell 21
plot_wordcloud(all_words_neutral)



## === cell 22
df_train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
df_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
df_submission = pd.read_csv(
    "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"
)




## === cell 23
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word_model):
        self.word_model = word_model
        self.word_idf_weight = None
        self.vector_size = word_model.wv.vector_size

    def fit(self, docs):
        text_docs = [" ".join(doc) for doc in docs]
        tfidf = TfidfVectorizer(stop_words="english", max_features=300)
        tfidf.fit(text_docs)
        max_idf = float(max(tfidf.idf_))
        self.word_idf_weight = defaultdict(
            lambda: max_idf,
            [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()],
        )
        self.vocabulary_ = tfidf.vocabulary_
        return self

    def transform(self, docs):
        if isinstance(docs, pd.Series):
            docs_list = docs.astype(str).tolist()
        elif isinstance(docs, np.ndarray):
            docs_list = docs.tolist()
        else:
            docs_list = docs

        if isinstance(docs_list, list) and len(docs_list) > 0:
            first = docs_list[0]
            if isinstance(first, str):
                docs_list = [clean_text(t).split() for t in docs_list]

        return self.word_average_list(docs_list)

    def word_average(self, sent):
        mean = []
        for word in sent:
            if word in self.word_model.wv.key_to_index:
                mean.append(
                    self.word_model.wv.get_vector(word) * self.word_idf_weight[word]
                )
        if not mean:
            return np.zeros(self.vector_size)
        return np.array(mean).mean(axis=0)

    def word_average_list(self, docs):
        return np.vstack([self.word_average(sent) for sent in docs])




## === cell 24
class MultinomialNBClassifier:
    def __init__(self, alpha=0):
        self.prob_w_given_pos = None
        self.prob_w_given_neut = None
        self.prob_w_given_neg = None
        self.prob_pos = None
        self.prob_neut = None
        self.prob_neg = None

    def fit(self, X_pos, X_neut, X_neg, alpha=0):
        num_features = X_pos.shape[1]
        prob_w_given_pos = np.zeros(num_features)
        prob_w_given_neut = np.zeros(num_features)
        prob_w_given_neg = np.zeros(num_features)

        all_feature_sum_pos = X_pos.sum()
        all_feature_sum_neut = X_neut.sum()
        all_feature_sum_neg = X_neg.sum()

        for feature in range(num_features):
            feature_sum_pos = X_pos[:, feature].sum()
            feature_sum_neut = X_neut[:, feature].sum()
            feature_sum_neg = X_neg[:, feature].sum()

            prob_w_given_pos[feature] = (feature_sum_pos + alpha) / (
                all_feature_sum_pos + num_features * alpha
            )
            prob_w_given_neut[feature] = (feature_sum_neut + alpha) / (
                all_feature_sum_neut + num_features * alpha
            )
            prob_w_given_neg[feature] = (feature_sum_neg + alpha) / (
                all_feature_sum_neg + num_features * alpha
            )

        self.prob_w_given_pos = prob_w_given_pos - (
            prob_w_given_neut + prob_w_given_neg
        )
        self.prob_w_given_neut = prob_w_given_neut - (
            prob_w_given_neg + prob_w_given_pos
        )
        self.prob_w_given_neg = prob_w_given_neg - (
            prob_w_given_neut + prob_w_given_pos
        )

        total = X_pos.shape[0] + X_neut.shape[0] + X_neg.shape[0]
        self.prob_pos = X_pos.shape[0] / total
        self.prob_neut = X_neut.shape[0] / total
        self.prob_neg = X_neg.shape[0] / total

    def predict_selected_text(self, vocab_to_index, text, sentiments):
        predictions = []
        num_examples = len(text)
        for i in range(num_examples):
            tweet = str(text[i])
            sentiment = sentiments[i]

            if sentiment == "neutral":
                predictions.append(tweet)
                continue
            elif sentiment == "positive":
                weights_to_use = self.prob_w_given_pos
            else:  # negative
                weights_to_use = self.prob_w_given_neg

            words_in_tweet = tweet.split()
            word_subsets = [
                words_in_tweet[i : j + 1]
                for i in range(len(words_in_tweet))
                for j in range(i, len(words_in_tweet))
            ]
            lst = sorted(word_subsets, key=len)

            max_weight_sum = 0.0
            selected_text = None

            for word_subset in lst:
                weight_sum = 0.0
                for word in word_subset:
                    translated_word = word.translate(
                        str.maketrans("", "", string.punctuation)
                    )
                    if translated_word in vocab_to_index:
                        weight_sum += weights_to_use[vocab_to_index[translated_word]]

                if weight_sum > max_weight_sum:
                    max_weight_sum = weight_sum
                    selected_text = word_subset

            predictions.append(
                tweet if selected_text is None else " ".join(selected_text)
            )
        return predictions




## === cell 25
def load_data(rootdir="./"):
    print("load data \n")
    train = pd.read_csv(os.path.join(rootdir, "train.csv"))
    test = pd.read_csv(os.path.join(rootdir, "test.csv"))
    sample = pd.read_csv(os.path.join(rootdir, "sample_submission.csv"))
    return train, test, sample




## === cell 26
def jaccard_metric(str1, str2):
    if len(str(str1)) == 0 and len(str(str2)) == 0:
        return 1.0
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    if len(a) == 0 and len(b) == 0:
        return 1.0
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))




## === cell 27
def clean_text_v2(text):
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text




## === cell 28
def convert_data(input_data):
    converted_data = [clean_text_v2(tweet).split() for tweet in input_data["text"]]
    converted_data = [tweet for tweet in converted_data if tweet != []]
    return converted_data




## === cell 29
def train_data(input_data):
    print("Training data using Word2Vec model \n")
    phrases = Phrases(input_data)
    bigram = Phraser(phrases)
    input_data = list(bigram[input_data])

    try:
        model = Word2Vec(
            input_data,
            min_count=3,
            vector_size=300,
            workers=5,
            window=5,
            epochs=30,
            sg=1,
        )
    except TypeError:
        model = Word2Vec(
            input_data, min_count=3, size=300, workers=5, window=5, iter=30, sg=1
        )
    return model




## === cell 30
def examples(model):
    vocab_size = (
        len(getattr(model.wv, "key_to_index", {}))
        if hasattr(model.wv, "key_to_index")
        else len(model.wv.vocab)
    )
    print(vocab_size)
    for w in ["happy", "funny", "danger"]:
        try:
            print(
                f"Looking into similarities to the word {w}:", model.wv.most_similar(w)
            )
        except Exception:
            print(f"most_similar failed for {w} (word may be OOV).")
    try:
        print(
            "Looking at the similarity distance between happy and weekend:",
            model.wv.similarity("happy", "weekend"),
        )
    except Exception:
        pass
    try:
        print(
            "Looking at the similarity distance between alright and disappointed:",
            model.wv.similarity("alright", "disappointed"),
        )
    except Exception:
        pass
    try:
        print(
            "Looking at the similarity distance between sniffle and sob:",
            model.wv.similarity("sniffle", "sob"),
        )
    except Exception:
        pass




## === cell 31
def predict_selected_text(df, vocab_to_index, pos_w, neut_w, neg_w):
    predictions = []
    for _, row in df.iterrows():
        tweet = row["text"]
        sentiment = row["sentiment"]

        if sentiment == "neutral":
            predictions.append(tweet)
            continue
        elif sentiment == "positive":
            weights_to_use = pos_w
        else:
            weights_to_use = neg_w

        words_in_tweet = tweet.split()
        word_subsets = [
            words_in_tweet[i : j + 1]
            for i in range(len(words_in_tweet))
            for j in range(i, len(words_in_tweet))
        ]
        lst = sorted(word_subsets, key=len)

        max_weight_sum = 0.0
        selected_text = None

        for word_subset in lst:
            weight_sum = 0.0
            for word in word_subset:
                translated_word = word.translate(
                    str.maketrans("", "", string.punctuation)
                )
                if translated_word in vocab_to_index:
                    weight_sum += weights_to_use[vocab_to_index[translated_word]]

            if weight_sum > max_weight_sum:
                max_weight_sum = weight_sum
                selected_text = word_subset

        predictions.append(tweet if selected_text is None else " ".join(selected_text))
    return predictions




## === cell 32
train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")

train.dropna(inplace=True)
train = train.reset_index(drop=True)

converted_data = convert_data(train)
w2v_model = train_data(converted_data)
examples(w2v_model)

train["text"] = train["text"].apply(lambda x: clean_text_v2(x))
train["selected_text"] = train["selected_text"].apply(lambda x: clean_text_v2(x))

test["text"] = test["text"].apply(lambda x: clean_text_v2(x))

X_train, X_val = train_test_split(train, train_size=0.80, random_state=0)

positive_train = X_train[X_train["sentiment"] == "positive"]
neutral_train = X_train[X_train["sentiment"] == "neutral"]
negative_train = X_train[X_train["sentiment"] == "negative"]

vectorizer = TfidfEmbeddingVectorizer(w2v_model)
vectorizer.fit(converted_data)

X_positive = vectorizer.transform(positive_train["text"])
X_neutral = vectorizer.transform(neutral_train["text"])
X_negative = vectorizer.transform(negative_train["text"])

nb = MultinomialNBClassifier()
nb.fit(X_positive, X_neutral, X_negative, alpha=4)



## === cell 33
vocab_to_index = {k: v for k, v in vectorizer.vocabulary_.items()}
predicted_text = nb.predict_selected_text(
    vocab_to_index, X_val["text"].to_numpy(), X_val["sentiment"].to_numpy()
)



## === cell 34
X_val = X_val.assign(predicted_text=predicted_text)
X_val["jaccard"] = X_val.apply(
    lambda x: jaccard_metric(x["selected_text"], x["predicted_text"]), axis=1
)
print("Word2Vec + Tfidf + MultiNB Jaccard Score: {}".format(np.mean(X_val["jaccard"])))

submission_predicted_text = nb.predict_selected_text(
    vocab_to_index, test["text"].astype(str).to_numpy(), test["sentiment"].to_numpy()
)

submission_df = pd.DataFrame(
    {"textID": test["textID"], "selected_text": submission_predicted_text}
)
submission_df["selected_text"] = submission_df["selected_text"].fillna("").astype(str)

submission_path = os.path.join("./", "submission.csv")
submission_df.to_csv(submission_path, index=False)
print("Saved submission to:", submission_path)
submission_df.head()
