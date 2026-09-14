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

3.8

# 3. Installed packages

geopandas==0.14.4
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
pillow==11.3.0
plotly==5.24.1
plotly-express==0.4.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
tqdm==4.67.1
wordcloud==1.9.4

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

0.6491307616233826

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import re
import string
import random
from tqdm import tqdm

try:
    import seaborn as sns
except ImportError:

    class _DummySeaborn:
        @staticmethod
        def set_theme(*args, **kwargs):
            pass

        @staticmethod
        def displot(*args, **kwargs):
            pass

        @staticmethod
        def barplot(*args, **kwargs):
            pass

    sns = _DummySeaborn()




## === cell 1
def color_generator(number_of_colors):
    return [
        "#" + "".join(random.choice("0123456789ABCDEF") for _ in range(6))
        for _ in range(number_of_colors)
    ]




## === cell 2
train_path = "/kaggle/input/tweet-sentiment-extraction/train.csv"
test_path = "/kaggle/input/tweet-sentiment-extraction/test.csv"
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)




## === cell 3
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )  # remove punctuation
    text = text.strip()
    text = re.sub(r"\b\w{1,3}\b", "", text)  # remove very short words
    return text




## === cell 4
from sklearn.feature_extraction.text import TfidfVectorizer


def build_weight_dict(sentiment_mask):
    tfv = TfidfVectorizer(
        max_df=0.95, min_df=2, max_features=10000, stop_words="english", use_idf=True
    )
    tfv.fit(train_data.loc[sentiment_mask, "text"])
    feat_names = tfv.get_feature_names_out()
    weights = 1 / (2 ** np.array(tfv.idf_))
    return dict(zip(map(str, feat_names), weights))


pos_words = build_weight_dict(train_data["sentiment"] == "positive")
neg_words = build_weight_dict(train_data["sentiment"] == "negative")
neutral_words = build_weight_dict(train_data["sentiment"] == "neutral")

pos_words_new = {}
neg_words_new = {}
neutral_words_new = {}

for w in pos_words:
    neg_w = neg_words.get(w, 0)
    neu_w = neutral_words.get(w, 0)
    pos_words_new[w] = pos_words[w] - (neg_w + neu_w)

for w in neg_words:
    if neg_words[w] == 0:
        continue
    pos_w = pos_words.get(w, 0)
    neu_w = neutral_words.get(w, 0)
    neg_words_new[w] = neg_words[w] - (pos_w + neu_w)

for w in neutral_words:
    if neutral_words[w] == 0:
        continue
    pos_w = pos_words.get(w, 0)
    neg_w = neg_words.get(w, 0)
    neutral_words_new[w] = neutral_words[w] - (pos_w + neg_w)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2020942260.py in <cell line: 0>()
     17 pos_words = build_weight_dict(train_data["sentiment"] == "positive")
     18 neg_words = build_weight_dict(train_data["sentiment"] == "negative")
---> 19 neutral_words = build_weight_dict(train_data["sentiment"] == "neutral")
     20 
     21 # create contrast dictionaries

/tmp/ipykernel_11/2020942260.py in build_weight_dict(sentiment_mask)
      7         max_df=0.95, min_df=2, max_features=10000, stop_words="english", use_idf=True
      8     )
----> 9     tfv.fit(train_data.loc[sentiment_mask, "text"])
     10     # newer sklearn uses get_feature_names_out()
     11     feat_names = tfv.get_feature_names_out()

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in fit(self, raw_documents, y)
   2101             sublinear_tf=self.sublinear_tf,
   2102         )
-> 2103         X = super().fit_transform(raw_documents)
   2104         self._tfidf.fit(X)
   2105         return self

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in fit_transform(self, raw_documents, y)
   1386                     break
   1387 
-> 1388         vocabulary, X = self._count_vocab(raw_documents, self.fixed_vocabulary_)
   1389 
   1390         if self.binary:

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in _count_vocab(self, raw_documents, fixed_vocab)
   1273         for doc in raw_documents:
   1274             feature_counter = {}
-> 1275             for feature in analyze(doc):
   1276                 try:
   1277                     feature_idx = vocabulary[feature]

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in _analyze(doc, analyzer, tokenizer, ngrams, preprocessor, decoder, stop_words)
    104 
    105     if decoder is not None:
--> 106         doc = decoder(doc)
    107     if analyzer is not None:
    108         doc = analyzer(doc)

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in decode(self, doc)
    237 
    238         if doc is np.nan:
--> 239             raise ValueError(
    240                 "np.nan is an invalid document, expected byte or unicode string."
    241             )

ValueError: np.nan is an invalid document, expected byte or unicode string.

## === cell 5
def calculate_selected_text(df_row, tol=0.001):
    tweet = df_row["text"]
    sentiment = df_row["sentiment"]

    if sentiment == "neutral" or len(tweet.split()) <= 3:
        return tweet

    words = tweet.lower().split()
    subsets = [
        words[i : j + 1] for i in range(len(words)) for j in range(i, len(words))
    ]

    dict_to_use = pos_words_new if sentiment == "positive" else neg_words_new

    best_score = 0
    best_subset = []

    for sub in subsets:
        cur_score = 0
        for token in sub:
            token_clean = token.translate(str.maketrans("", "", string.punctuation))
            cur_score += dict_to_use.get(token_clean, 0)
        if cur_score > best_score + tol:
            best_score = cur_score
            best_subset = sub

    if not best_subset:
        best_subset = words
    return " ".join(best_subset)




## === cell 6
test_data["selected_text"] = test_data.apply(calculate_selected_text, axis=1)
submission_path = "submission.csv"
test_data[["textID", "selected_text"]].to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2435405786.py in <cell line: 0>()
      1 # generate predictions for the test set and write submission
----> 2 test_data["selected_text"] = test_data.apply(calculate_selected_text, axis=1)
      3 submission_path = "submission.csv"
      4 test_data[["textID", "selected_text"]].to_csv(submission_path, index=False)
      5 print(f"Submission file written to {submission_path}")

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

/tmp/ipykernel_11/2951983719.py in calculate_selected_text(df_row, tol)
     11     ]
     12 
---> 13     dict_to_use = pos_words_new if sentiment == "positive" else neg_words_new
     14 
     15     best_score = 0

NameError: name 'pos_words_new' is not defined
