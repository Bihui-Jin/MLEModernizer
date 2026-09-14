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

No external packages required in the script and installed.

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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
pass



## === cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import plotly
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)
import string
from wordcloud import WordCloud, STOPWORDS
import spacy
from tqdm import tqdm
import random
from spacy.util import compounding
from spacy.util import minibatch

import os
import re
import math
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.phrases import Phraser, Phrases

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

import warnings

warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 2
train_df = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
test_df = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
train_df.head(8)



## === cell 3
print(train_df.shape)
print(test_df.shape)



## === cell 4
train_df.info()



## === cell 5
print(train_df[train_df["text"].isnull()])
print(train_df[train_df["selected_text"].isnull()])



## === cell 6
train_df.dropna(inplace=True)
train_df = train_df.reset_index(drop=True)



## === cell 7
train_df.info()



## === cell 8
train_df.describe()



## === cell 9
train_df["N_text_words"] = train_df["text"].apply(lambda tweet: len(tweet.split()))
train_df["N_selected_text_words"] = train_df["selected_text"].apply(
    lambda tweet: len(tweet.split())
)
train_df["N_words_difference"] = (
    train_df["N_text_words"] - train_df["N_selected_text_words"]
)
train_df.head(8)



## === cell 10
print(
    "There are {0} unique sentiments having values {1}".format(
        train_df["sentiment"].nunique(), train_df["sentiment"].unique()
    )
)



## === cell 11
n_neutral = train_df["sentiment"].loc[train_df["sentiment"] == "neutral"].count()
n_positive = train_df["sentiment"].loc[train_df["sentiment"] == "positive"].count()
n_negative = train_df["sentiment"].loc[train_df["sentiment"] == "negative"].count()
print(f"Neutral tweets : {n_neutral}")
print(f"Positive tweets : {n_positive}")
print(f"Negative tweets : {n_negative}")



## === cell 12
sentiments = ["Neutral", "Positive", "Negative"]
fig = go.Figure(
    data=[go.Pie(labels=sentiments, values=[n_neutral, n_positive, n_negative])]
)
fig.show()




## === cell 13
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


jaccard_score = []
for i in range(train_df.shape[0]):
    str1 = train_df["text"][i].strip()
    str2 = train_df["selected_text"][i].strip()
    jaccard_score.append(jaccard(str1, str2))

train_df["Jaccard_score"] = jaccard_score
train_df.head(8)




## === cell 14
def clean_text(text):
    """Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers."""
    text = str(text).lower()
    text = re.sub("\[.*?\]", "", text)
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\w*\d\w*", "", text)
    return text




## === cell 15
train_df["text_cleaned"] = train_df["text"].apply(lambda x: clean_text(x))
train_df["selected_text_cleaned"] = train_df["selected_text"].apply(
    lambda x: clean_text(x)
)

STOPWORDS = stopwords.words("english")


def remove_stopwords(text):
    return [word for word in text.split() if word not in STOPWORDS]


train_df["text_cleaned"] = train_df["text_cleaned"].apply(lambda x: remove_stopwords(x))
train_df["selected_text_cleaned"] = train_df["selected_text_cleaned"].apply(
    lambda x: remove_stopwords(x)
)



## === cell 16
train_df.head(8)




## === cell 17
def get_all_words(df_col):
    all_words_text = []
    for row in df_col:
        for word in row:
            all_words_text.append(word)
    return all_words_text


all_words_text = get_all_words(train_df["text_cleaned"])
all_words_selected_text = get_all_words(train_df["selected_text_cleaned"])



## === cell 18
all_words_neutral = get_all_words(
    train_df[train_df["sentiment"] == "neutral"]["text_cleaned"]
)
all_words_positive = get_all_words(
    train_df[train_df["sentiment"] == "positive"]["text_cleaned"]
)
all_words_negative = get_all_words(
    train_df[train_df["sentiment"] == "negative"]["text_cleaned"]
)




## === cell 19
def plot_wordcloud(all_words):
    stopwords_set = set(STOPWORDS)
    more_stopwords = {"u", "im"}
    stopwords_set = stopwords_set.union(more_stopwords)
    all_words = " ".join(all_words)
    wordcloud = WordCloud(
        width=400,
        height=200,
        background_color="white",
        max_words=200,
        stopwords=stopwords_set,
        min_font_size=10,
    )
    wordcloud = wordcloud.generate(all_words)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()




## === cell 20
plot_wordcloud(all_words_positive)



## === cell 21
plot_wordcloud(all_words_negative)



## === cell 22
plot_wordcloud(all_words_neutral)



## === cell 23
df_train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
df_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
df_submission = pd.read_csv(
    "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"
)




## === cell 24
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word_model):
        self.word_model = word_model
        self.word_idf_weight = None
        self.vector_size = word_model.wv.vector_size

    def fit(self, docs):
        text_docs = [" ".join(doc) for doc in docs]
        tfidf = TfidfVectorizer(stop_words="english", max_features=300)
        tfidf.fit(text_docs)
        max_idf = max(tfidf.idf_)
        self.word_idf_weight = defaultdict(
            lambda: max_idf,
            [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()],
        )
        self.vocabulary_ = tfidf.vocabulary_
        return self

    def transform(self, docs):
        return self.word_average_list(docs)

    def word_average(self, sent):
        mean = []
        for word in sent:
            if word in self.word_model.wv.key_to_index:
                mean.append(
                    self.word_model.wv.get_vector(word) * self.word_idf_weight[word]
                )
        if not mean:
            return np.zeros(self.vector_size)
        else:
            return np.array(mean).mean(axis=0)

    def word_average_list(self, docs):
        return np.vstack([self.word_average(sent) for sent in docs])




## === cell 25
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
        for tweet, sentiment in zip(text, sentiments):
            if sentiment == "neutral":
                predictions.append(tweet)
                continue
            weights_to_use = (
                self.prob_w_given_pos
                if sentiment == "positive"
                else self.prob_w_given_neg
            )

            words_in_tweet = tweet.split()
            word_subsets = [
                words_in_tweet[i : j + 1]
                for i in range(len(words_in_tweet))
                for j in range(i, len(words_in_tweet))
            ]
            word_subsets = sorted(word_subsets, key=len)

            max_weight_sum = 0
            selected_text = None
            for subset in word_subsets:
                weight_sum = 0
                for w in subset:
                    w_clean = w.translate(
                        str.maketrans("", "", string.punctuation)
                    ).lower()
                    if w_clean in vocab_to_index:
                        weight_sum += weights_to_use[vocab_to_index[w_clean]]
                if weight_sum > max_weight_sum:
                    max_weight_sum = weight_sum
                    selected_text = subset
            predictions.append(" ".join(selected_text) if selected_text else tweet)
        return predictions




## === cell 26
def load_data(rootdir="./"):
    print("load data \\n")
    train = pd.read_csv(os.path.join(rootdir, "train.csv"))
    test = pd.read_csv(os.path.join(rootdir, "test.csv"))
    sample = pd.read_csv(os.path.join(rootdir, "sample_submission.csv"))
    return train, test, sample




## === cell 27
def convert_data(input_data):
    converted = [clean_text(tweet).split() for tweet in input_data["text"]]
    return [tweet for tweet in converted if tweet]




## === cell 28
def train_data(input_data):
    phrases = Phrases(input_data, min_count=5, threshold=10)
    bigram = Phraser(phrases)
    input_data = list(bigram[input_data])
    model = Word2Vec(
        sentences=input_data,
        vector_size=300,
        min_count=3,
        workers=5,
        window=5,
        epochs=30,
        sg=1,
    )
    return model




## === cell 29
def examples(model):
    print(len(model.wv.key_to_index))

    def safe_most_similar(word):
        if word in model.wv.key_to_index:
            sims = model.wv.most_similar(word)[:3]
            print(f"Most similar to {word}:", sims)
        else:
            print(f"Word '{word}' not in vocabulary.")

    def safe_similarity(w1, w2):
        if w1 in model.wv.key_to_index and w2 in model.wv.key_to_index:
            print(f"Similarity {w1}-{w2}:", model.wv.similarity(w1, w2))
        else:
            print(
                f"Cannot compute similarity for '{w1}' and/or '{w2}' (missing in vocab)."
            )

    for w in ["happy", "funny", "danger"]:
        safe_most_similar(w)
    safe_similarity("happy", "weekend")
    safe_similarity("alright", "disappointed")
    safe_similarity("sniffle", "sob")




## === cell 30
if __name__ == "__main__":
    train = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
    test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
    train.dropna(inplace=True)

    converted_data = convert_data(train)
    w2v_model = train_data(converted_data)
    examples(w2v_model)

    train["text"] = train["text"].apply(lambda x: clean_text(x))
    train["selected_text"] = train["selected_text"].apply(lambda x: clean_text(x))

    X_train, X_val = train_test_split(train, train_size=0.80, random_state=0)

    pos_train = X_train[X_train["sentiment"] == "positive"]
    neu_train = X_train[X_train["sentiment"] == "neutral"]
    neg_train = X_train[X_train["sentiment"] == "negative"]

    vectorizer = TfidfEmbeddingVectorizer(w2v_model)
    vectorizer.fit(converted_data)

    X_pos = vectorizer.transform(pos_train["text"])
    X_neu = vectorizer.transform(neu_train["text"])
    X_neg = vectorizer.transform(neg_train["text"])

    nb = MultinomialNBClassifier()
    nb.fit(X_pos, X_neu, X_neg, alpha=4)

    vocab_to_index = {k: v for k, v in vectorizer.vocabulary_.items()}
    val_predictions = nb.predict_selected_text(
        vocab_to_index, X_val["text"].to_numpy(), X_val["sentiment"].to_numpy()
    )
    X_val = X_val.assign(predicted_text=val_predictions)
    X_val["jaccard"] = X_val.apply(
        lambda x: jaccard(x["selected_text"], x["predicted_text"]), axis=1
    )
    print("Validation Jaccard mean:", X_val["jaccard"].mean())

    test_predictions = nb.predict_selected_text(
        vocab_to_index, test["text"].to_numpy(), test["sentiment"].to_numpy()
    )
    submission_df = pd.DataFrame(
        {"textID": test["textID"], "selected_text": test_predictions}
    )
    submission_path = os.path.join("./", "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")
    print(submission_df.head())

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
/tmp/ipykernel_11/1433929364.py in <cell line: 0>()
     33     )
     34     X_val = X_val.assign(predicted_text=val_predictions)
---> 35     X_val["jaccard"] = X_val.apply(
     36         lambda x: jaccard(x["selected_text"], x["predicted_text"]), axis=1
     37     )

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in apply(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)
  10372             kwargs=kwargs,
  10373         )
> 10374         return op.apply().__finalize__(self, method="apply")
  10375 
  10376     def map(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
    914             return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)
    915 
--> 916         return self.apply_standard()
    917 
    918     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1061     def apply_standard(self):
   1062         if self.engine == "python":
-> 1063             results, res_index = self.apply_series_generator()
   1064         else:
   1065             results, res_index = self.apply_series_numba()

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_series_generator(self)
   1079             for i, v in enumerate(series_gen):
   1080                 # ignore SettingWithCopy here in case the user mutates
-> 1081                 results[i] = self.func(v, *self.args, **self.kwargs)
   1082                 if isinstance(results[i], ABCSeries):
   1083                     # If we have a view on v, we need to make a copy because

/tmp/ipykernel_11/1433929364.py in <lambda>(x)
     34     X_val = X_val.assign(predicted_text=val_predictions)
     35     X_val["jaccard"] = X_val.apply(
---> 36         lambda x: jaccard(x["selected_text"], x["predicted_text"]), axis=1
     37     )
     38     print("Validation Jaccard mean:", X_val["jaccard"].mean())

/tmp/ipykernel_11/3018747336.py in jaccard(str1, str2)
      3     b = set(str2.lower().split())
      4     c = a.intersection(b)
----> 5     return float(len(c)) / (len(a) + len(b) - len(c))
      6 
      7 

ZeroDivisionError: float division by zero
