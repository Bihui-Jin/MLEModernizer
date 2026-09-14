# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.8

# 2. Installed packages

geopandas==0.14.4
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

# 3. Data file paths

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

# 4. Code solution

## === cell 1
import pandas as pd
import numpy as np
import pprint as pp
from sklearn.feature_extraction.text import CountVectorizer
import string
from sklearn.model_selection import train_test_split


def build_vocab(cv, count_df, train):
    neg_count_df, pos_count_df, neutral_count_df = count_df
    neg_train, pos_train, neutral_train = train

    pos_words = {}
    neutral_words = {}
    neg_words = {}

    for k in cv.get_feature_names():
        pos = pos_count_df[k].sum()
        neutral = neutral_count_df[k].sum()
        neg = neg_count_df[k].sum()

        pos_words[k] = pos / pos_train.shape[0]
        neutral_words[k] = neutral / neutral_train.shape[0]
        neg_words[k] = neg / neg_train.shape[0]

    neg_words_adj = {}
    pos_words_adj = {}
    neutral_words_adj = {}

    for key, value in neg_words.items():
        neg_words_adj[key] = neg_words[key] - (neutral_words[key] + pos_words[key])

    for key, value in pos_words.items():
        pos_words_adj[key] = pos_words[key] - (neutral_words[key] + neg_words[key])

    for key, value in neutral_words.items():
        neutral_words_adj[key] = neutral_words[key] - (neg_words[key] + pos_words[key])

    return (neg_words_adj, pos_words_adj, neutral_words_adj)


def calculate_selected_text(pos_words_adj, neg_words_adj, df_row, tol=0):
    tweet = df_row["text"]
    sentiment = df_row["sentiment"]

    if sentiment == "neutral":
        return tweet
    elif sentiment == "positive":
        dict_to_use = (
            pos_words_adj  # Calculate word weights using the pos_words dictionary
        )
    elif sentiment == "negative":
        dict_to_use = (
            neg_words_adj  # Calculate word weights using the neg_words dictionary
        )

    words = tweet.split()
    words_len = len(words)
    subsets = [words[i : j + 1] for i in range(words_len) for j in range(i, words_len)]

    score = 0
    selection_str = ""  # This will be our choice
    lst = sorted(subsets, key=len)  # Sort candidates by length

    for i in range(len(subsets)):

        new_sum = 0  # Sum for the current substring

        for p in range(len(lst[i])):
            if (
                lst[i][p].translate(str.maketrans("", "", string.punctuation))
                in dict_to_use.keys()
            ):
                new_sum += dict_to_use[
                    lst[i][p].translate(str.maketrans("", "", string.punctuation))
                ]

        if new_sum > score + tol:
            score = new_sum
            selection_str = lst[i]

    if len(selection_str) == 0:
        selection_str = words

    return " ".join(selection_str)


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def which_longer(truth, prediction):
    truth_set = set(truth.lower().split())
    pred_set = set(prediction.lower().split())

    return float(len(pred_set) - len(truth_set))


def load_data():
    train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
    test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
    sample = pd.read_csv(
        "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"
    )

    train[train["text"].isna()]

    train.drop(314, inplace=True)

    train = train.dropna(subset=["text"]).copy()
    test = test.dropna(subset=["text"]).copy()

    train["text"] = train["text"].apply(lambda x: x.lower())
    test["text"] = test["text"].apply(lambda x: x.lower())
    return train, test, sample


def main():
    tol = 0.001

    train, test, sample = load_data()
    K = 10
    jaccard_scores = []
    vocab_weights = []
    indexes = np.arange(train.shape[0])
    np.random.shuffle(indexes)
    print("Indexes shape: " + str(indexes.shape))
    print("Train shape: " + str(train.shape))

    for group in range(K):
        group_start = int(group * (indexes.shape[0] / K))
        group_end = int((group + 1) * (indexes.shape[0] / K))
        X_train_part_1 = train.iloc[indexes[:group_start]]  # from 0 to start of group
        X_train_part_2 = train.iloc[
            indexes[group_end:]
        ]  # from end of group to end of data
        X_train = pd.concat([X_train_part_1, X_train_part_2])
        X_val = train.iloc[group_start:group_end]

        pos_train = X_train[X_train["sentiment"] == "positive"]
        neutral_train = X_train[X_train["sentiment"] == "neutral"]
        neg_train = X_train[X_train["sentiment"] == "negative"]

        cv = CountVectorizer(
            max_df=0.95, min_df=2, max_features=10000, stop_words="english"
        )

        cv.fit_transform(X_train["text"])

        X_pos = cv.transform(pos_train["text"])
        X_neutral = cv.transform(neutral_train["text"])
        X_neg = cv.transform(neg_train["text"])

        pos_count_df = pd.DataFrame(X_pos.toarray(), columns=cv.get_feature_names())
        neutral_count_df = pd.DataFrame(
            X_neutral.toarray(), columns=cv.get_feature_names()
        )
        neg_count_df = pd.DataFrame(X_neg.toarray(), columns=cv.get_feature_names())

        count_df = (neg_count_df, pos_count_df, neutral_count_df)
        train_df = (neg_train, pos_train, neutral_train)
        neg_words_adj, pos_words_adj, neutral_words_adj = build_vocab(
            cv, count_df, train_df
        )

        vocab_weights.append(
            {"neg": neg_words_adj, "pos": pos_words_adj, "neu": neutral_words_adj}
        )
        pd.options.mode.chained_assignment = None

        X_val["predicted_selection"] = ""

        for index, row in X_val.iterrows():
            selected_text = calculate_selected_text(
                pos_words_adj, neg_words_adj, row, tol
            )
            X_val.loc[X_val["textID"] == row["textID"], ["predicted_selection"]] = (
                selected_text
            )

        X_val["which_longer"] = X_val.apply(
            lambda x: which_longer(x["selected_text"], x["predicted_selection"]), axis=1
        )
        X_val["jaccard"] = X_val.apply(
            lambda x: jaccard(x["selected_text"], x["predicted_selection"]), axis=1
        )
        jaccard_scores.append(np.mean(X_val["jaccard"]))
        print(
            "-------------------------------- K = "
            + str(group + 1)
            + " ---------------------------------------------"
        )
        print("The jaccard score for the validation set is:", np.mean(X_val["jaccard"]))
        print(
            "The selected text for negative is on average {} words smaller".format(
                str(np.mean((X_val[X_val["sentiment"] == "negative"])["which_longer"]))
            )
        )
        print(
            "The selected text for positive is on average {} words smaller".format(
                str(np.mean((X_val[X_val["sentiment"] == "positive"])["which_longer"]))
            )
        )
        print(
            "The selected text for neutral is on average {} words smaller".format(
                str(np.mean((X_val[X_val["sentiment"] == "neutral"])["which_longer"]))
            )
        )

    max_vocab_weights = vocab_weights[jaccard_scores.index(max(jaccard_scores))]
    neg_words_adj = max_vocab_weights["neg"]
    pos_words_adj = max_vocab_weights["pos"]
    neutral_words_adj = max_vocab_weights["neu"]

    for index, row in test.iterrows():
        selected_text = calculate_selected_text(pos_words_adj, neg_words_adj, row, tol)
        sample.loc[sample["textID"] == row["textID"], ["selected_text"]] = selected_text

    sample.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()


## --- ERROR in cell 1, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3794160674.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    225[0m [0;34m[0m[0m
[1;32m    226[0m [0;32mif[0m [0m__name__[0m [0;34m==[0m [0;34m"__main__"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 227[0;31m     [0mmain[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/tmp/ipykernel_11/3794160674.py[0m in [0;36mmain[0;34m()[0m
[1;32m    156[0m         [0mX_neg[0m [0;34m=[0m [0mcv[0m[0;34m.[0m[0mtransform[0m[0;34m([0m[0mneg_train[0m[0;34m[[0m[0;34m"text"[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    157[0m [0;34m[0m[0m
[0;32m--> 158[0;31m         [0mpos_count_df[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mDataFrame[0m[0;34m([0m[0mX_pos[0m[0;34m.[0m[0mtoarray[0m[0;34m([0m[0;34m)[0m[0;34m,[0m [0mcolumns[0m[0;34m=[0m[0mcv[0m[0;34m.[0m[0mget_feature_names[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    159[0m         neutral_count_df = pd.DataFrame(
[1;32m    160[0m             [0mX_neutral[0m[0;34m.[0m[0mtoarray[0m[0;34m([0m[0;34m)[0m[0;34m,[0m [0mcolumns[0m[0;34m=[0m[0mcv[0m[0;34m.[0m[0mget_feature_names[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'CountVectorizer' object has no attribute 'get_feature_names'
