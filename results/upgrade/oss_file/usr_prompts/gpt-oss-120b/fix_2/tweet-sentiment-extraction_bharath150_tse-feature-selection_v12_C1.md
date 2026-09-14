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

geopandas==0.14.4
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

0.5621

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np



## === cell 1
data = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
data["target"] = 0
data.loc[data["sentiment"] == "positive", "target"] = 1
data.loc[data["sentiment"] == "negative", "target"] = 2



## === cell 2
data["text"].replace("", np.nan, inplace=True)
data.dropna(subset=["text"], inplace=True)
data.reset_index(drop=True, inplace=True)

x_train, x_cv, y_train, y_cv = train_test_split(
    data.drop(["sentiment"], axis=1), data["target"], test_size=0.2, random_state=30
)



## === cell 3
use_tokenizer = False
if use_tokenizer:
    tokenizer = WordPunctTokenizer()
    vectorizer = CountVectorizer(
        tokenizer=tokenizer.tokenize, max_features=10000, min_df=2, max_df=0.95
    )
else:
    vectorizer = CountVectorizer(max_features=10000, min_df=2, max_df=0.95)

x_train_text = vectorizer.fit_transform(x_train["text"])
x_cv_text = vectorizer.transform(x_cv["text"])



## === cell 4
alphas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
cv_scores = []
for a in alphas:
    model = MultinomialNB(alpha=a)
    model.fit(x_train_text, y_train)
    acc = accuracy_score(y_cv, model.predict(x_cv_text))
    print("accuracy for alpha =", a, "is:", acc)
    cv_scores.append(acc)

best_alpha = alphas[cv_scores.index(max(cv_scores))]
print("Best alpha selected:", best_alpha)




## === cell 5
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))




## === cell 6
clf_split = MultinomialNB(alpha=best_alpha)
clf_split.fit(x_train_text, y_train)

feature_names = vectorizer.get_feature_names_out()
dict0 = dict(zip(feature_names, clf_split.feature_log_prob_[0]))
dict1 = dict(zip(feature_names, clf_split.feature_log_prob_[1]))
dict2 = dict(zip(feature_names, clf_split.feature_log_prob_[2]))




## === cell 7
def threshold_preds(k, use_tok, df):
    preds = []
    for _, row in df.iterrows():
        if row.target != 0:
            tokens = (
                WordPunctTokenizer().tokenize(row["text"])
                if use_tok
                else row["text"].split()
            )
            sentiment = row["target"]
            probs = dict1 if sentiment == 1 else dict2
            temp_score = sum(probs.get(tok.lower(), 0) for tok in tokens)
            avg_score = temp_score / max(len(tokens), 1)
            selected = [
                tok for tok in tokens if probs.get(tok.lower(), 0) > k * avg_score
            ]
            preds.append(" ".join(selected) if selected else row["text"])
        else:
            preds.append(row["text"])
    return preds


k_vals = np.linspace(0, 11, 20)
k_scores = []
for k in k_vals:
    preds = threshold_preds(
        k, use_tokenizer, x_cv.assign(selected_text=x_cv["selected_text"])
    )
    score = np.mean(
        [
            jaccard(row["selected_text"], pred)
            for row, pred in zip(x_cv.itertuples(), preds)
        ]
    )
    k_scores.append(score)
    print("jaccard score for threshold", k, "is:", score)

best_k = k_vals[k_scores.index(max(k_scores))]
print("Best k selected:", best_k)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/361841521.py in <cell line: 0>()
     31     )
     32     score = np.mean(
---> 33         [
     34             jaccard(row["selected_text"], pred)
     35             for row, pred in zip(x_cv.itertuples(), preds)

/tmp/ipykernel_11/361841521.py in <listcomp>(.0)
     32     score = np.mean(
     33         [
---> 34             jaccard(row["selected_text"], pred)
     35             for row, pred in zip(x_cv.itertuples(), preds)
     36         ]

TypeError: tuple indices must be integers or slices, not str

## === cell 8
full_vectorizer = (
    CountVectorizer(max_features=10000, min_df=2, max_df=0.95)
    if not use_tokenizer
    else CountVectorizer(
        tokenizer=WordPunctTokenizer().tokenize,
        max_features=10000,
        min_df=2,
        max_df=0.95,
    )
)

x_full_text = full_vectorizer.fit_transform(data["text"])
y_full = data["target"]

clf_full = MultinomialNB(alpha=best_alpha)
clf_full.fit(x_full_text, y_full)



## === cell 9
full_feature_names = full_vectorizer.get_feature_names_out()
dict0 = dict(zip(full_feature_names, clf_full.feature_log_prob_[0]))
dict1 = dict(zip(full_feature_names, clf_full.feature_log_prob_[1]))
dict2 = dict(zip(full_feature_names, clf_full.feature_log_prob_[2]))



## === cell 10
test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
test["target"] = 0
test.loc[test["sentiment"] == "positive", "target"] = 1
test.loc[test["sentiment"] == "negative", "target"] = 2

preds = threshold_preds(best_k, use_tokenizer, test)

submission = pd.read_csv(
    "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"
)
submission["selected_text"] = preds
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3289309666.py in <cell line: 0>()
      5 test.loc[test["sentiment"] == "negative", "target"] = 2
      6 
----> 7 preds = threshold_preds(best_k, use_tokenizer, test)
      8 
      9 submission = pd.read_csv(

NameError: name 'best_k' is not defined
