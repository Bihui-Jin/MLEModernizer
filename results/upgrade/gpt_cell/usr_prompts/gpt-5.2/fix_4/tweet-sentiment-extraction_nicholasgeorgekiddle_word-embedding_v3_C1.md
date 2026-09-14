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

gensim==4.4.0
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
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import string
from sklearn.model_selection import train_test_split


class LogisticRegressionCustom:

    def __init__(self, H=0.00001, batch_size=-1, iters=200, verbose=False):
        self.H = H
        self.batch_size = batch_size
        self.iters = iters
        self.verbose = False

    def score(self, X, Y):
        predictors = np.matmul(X, self.W)

        predictors = np.where(predictors < 0, 0, 1)

        falses = np.sum(np.abs(Y - predictors.T))

        1 - (falses / (len(Y)))
        return 1 - (falses / (len(Y)))

    def predict(self, X):
        return np.reciprocal((np.exp(-1 * np.matmul(X, self.W)) + 1))

    def fit(self, X, Y):
        X = np.nan_to_num(X)
        Y = np.nan_to_num(Y)

        W = np.zeros(X.shape[1])
        batch_num = 0
        grad = float("inf")

        if self.batch_size == -1:
            self.batch_size = X.shape[0]

        for j in range(int(self.iters * (X.shape[0] / self.batch_size))):

            batch = X[
                batch_num * self.batch_size : (batch_num + 1) * self.batch_size, :
            ]

            estimated = np.reciprocal((np.exp(-1 * np.matmul(batch, W)) + 1))

            actual = Y[batch_num * self.batch_size : (batch_num + 1) * self.batch_size]
            diffs = estimated.T - actual

            grad = np.sum(np.multiply(diffs, batch.T), axis=1)

            W = W - (self.H * grad)
            grad = np.linalg.norm(grad)

            batch_num += 1
            if (batch_num + 1) * self.batch_size - 1 > X.shape[0]:
                batch_num = 0

            self.W = W
            if self.verbose is True:
                print(self.score(X, Y))

        if self.verbose is True:
            print("Found weights are:")
            pp.pprint(W)


def apply_model(model, x):
    sum = np.zeros(50)
    count = 0
    for i in range(len(x)):
        if x[i] in model.wv.key_to_index:
            sum += model.wv[x[i]]
            count += 1

    return np.divide(sum, count)


def is_positive(x):
    if x == "positive":
        return 1
    else:
        return 0


def is_negative(x):
    if x == "negative":
        return 1
    else:
        return 0


def get_logistic(full_train):
    text = full_train.apply(lambda x: x["text"].split(), axis=1)

    print("Training word embedding...")
    model = Word2Vec(
        text.values,
        min_count=2,
        window=2,
        vector_size=50,
        sample=6e-5,
        alpha=0.03,
        min_alpha=0.0007,
        negative=20,
    )

    applied_model = text.apply(lambda x: apply_model(model, x))

    X = np.array(list(applied_model.values))
    print("Done with that!\n")

    Ypos = full_train["sentiment"].apply(lambda x: is_positive(x))
    Ypos = Ypos.values.ravel()

    Yneg = full_train["sentiment"].apply(lambda x: is_negative(x))
    Yneg = Yneg.values.ravel()

    X = np.nan_to_num(X)
    Ypos = np.nan_to_num(Ypos)
    Yneg = np.nan_to_num(Yneg)

    print("Fitting logistic regression models...")
    posLogisticRegr = LogisticRegression()
    posLogisticRegr.fit(X, Ypos)

    negLogisticRegr = LogisticRegression()
    negLogisticRegr.fit(X, Yneg)
    pos_score = posLogisticRegr.score(X, Ypos)
    neg_score = negLogisticRegr.score(X, Yneg)
    print("Done!\n")
    print("Positive Score: " + str(pos_score))
    print("Negative Score: " + str(neg_score))

    return (posLogisticRegr, negLogisticRegr, model)


def calculate_selected_text(df_row, positiveModel, negativeModel, word2vecModel, tol=0):
    tweet = df_row["text"]
    sentiment = df_row["sentiment"]

    if sentiment == "neutral":
        return tweet
    elif sentiment == "positive":
        logisticToUse = (
            positiveModel  # Calculate word weights using the pos_words dictionary
        )
    elif sentiment == "negative":
        logisticToUse = (
            negativeModel  # Calculate word weights using the neg_words dictionary
        )

    words = tweet.split()
    words_len = len(words)
    subsets = [words[i : j + 1] for i in range(words_len) for j in range(i, words_len)]

    score = 0
    selection_str = ""  # This will be our choice
    lst = sorted(subsets, key=len)  # Sort candidates by length

    failed_subsets = 0
    for i in range(len(subsets)):

        sum = np.zeros(50)
        count = 0
        for j in range(len(lst[i])):
            if (lst[i][j]) in word2vecModel.wv.key_to_index:
                sum += word2vecModel.wv[lst[i][j]]
                count += 1
        if count > 0:
            new_score = logisticToUse.predict([np.divide(sum, count)])
        else:
            new_score = 0
            failed_subsets += 1

        if new_score > score + tol:
            score = new_score
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
    intersect = truth_set.intersection(pred_set)

    return float(len(truth_set - intersect) + len(pred_set - intersect))


def load_data():
    train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
    test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
    sample = pd.read_csv(
        "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"
    )

    train[train["text"].isna()]

    train.drop(314, inplace=True)

    train["text"] = train["text"].fillna("").apply(lambda x: x.lower())
    test["text"] = test["text"].fillna("").apply(lambda x: x.lower())
    return train, test, sample


def main():
    tol = 0.001

    train, test, sample = load_data()
    K = 5
    indexes = np.arange(train.shape[0])
    np.random.shuffle(indexes)
    bestScore = 0
    bestModel = None

    for group in range(K):
        group_start = int(group * (indexes.shape[0] / K))
        group_end = int((group + 1) * (indexes.shape[0] / K))
        X_train_part_1 = train.iloc[indexes[:group_start]]  # from 0 to start of group
        X_train_part_2 = train.iloc[
            indexes[group_end:]
        ]  # from end of group to end of data
        X_train = pd.concat([X_train_part_1, X_train_part_2])
        X_val = train.iloc[group_start:group_end]

        posLogisticModel, negLogisticModel, word2vecModel = get_logistic(X_train)

        pd.options.mode.chained_assignment = None

        X_val["predicted_selection"] = ""

        for index, row in X_val.iterrows():
            selected_text = calculate_selected_text(
                row, posLogisticModel, negLogisticModel, word2vecModel
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
        print("The jaccard score for the validation set is:", np.mean(X_val["jaccard"]))
        print(
            "The selected text for negative is on average {} words different".format(
                str(np.mean((X_val[X_val["sentiment"] == "negative"])["which_longer"]))
            )
        )
        print(
            "The selected text for positive is on average {} words different".format(
                str(np.mean((X_val[X_val["sentiment"] == "positive"])["which_longer"]))
            )
        )
        print(
            "The selected text for neutral is on average {} words different".format(
                str(np.mean((X_val[X_val["sentiment"] == "neutral"])["which_longer"]))
            )
        )

        if np.mean(X_val["jaccard"]) > bestScore:
            bestModel = (posLogisticModel, negLogisticModel, word2vecModel)

    (posLogisticModel, negLogisticModel, word2vecModel) = bestModel

    for index, row in test.iterrows():
        selected_text = calculate_selected_text(
            row, posLogisticModel, negLogisticModel, word2vecModel
        )
        sample.loc[sample["textID"] == row["textID"], ["selected_text"]] = selected_text

    sample.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()


## --- ERROR in cell 1, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1255865017.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    293[0m [0;34m[0m[0m
[1;32m    294[0m [0;32mif[0m [0m__name__[0m [0;34m==[0m [0;34m"__main__"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 295[0;31m     [0mmain[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/tmp/ipykernel_11/1255865017.py[0m in [0;36mmain[0;34m()[0m
[1;32m    255[0m             )
[1;32m    256[0m [0;34m[0m[0m
[0;32m--> 257[0;31m         X_val["which_longer"] = X_val.apply(
[0m[1;32m    258[0m             [0;32mlambda[0m [0mx[0m[0;34m:[0m [0mwhich_longer[0m[0;34m([0m[0mx[0m[0;34m[[0m[0;34m"selected_text"[0m[0;34m][0m[0;34m,[0m [0mx[0m[0;34m[[0m[0;34m"predicted_selection"[0m[0;34m][0m[0;34m)[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m    259[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36mapply[0;34m(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)[0m
[1;32m  10372[0m             [0mkwargs[0m[0;34m=[0m[0mkwargs[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m  10373[0m         )
[0;32m> 10374[0;31m         [0;32mreturn[0m [0mop[0m[0;34m.[0m[0mapply[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__finalize__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mmethod[0m[0;34m=[0m[0;34m"apply"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m  10375[0m [0;34m[0m[0m
[1;32m  10376[0m     def map(

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py[0m in [0;36mapply[0;34m(self)[0m
[1;32m    914[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mapply_raw[0m[0;34m([0m[0mengine[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mengine[0m[0;34m,[0m [0mengine_kwargs[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mengine_kwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    915[0m [0;34m[0m[0m
[0;32m--> 916[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mapply_standard[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    917[0m [0;34m[0m[0m
[1;32m    918[0m     [0;32mdef[0m [0magg[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py[0m in [0;36mapply_standard[0;34m(self)[0m
[1;32m   1061[0m     [0;32mdef[0m [0mapply_standard[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1062[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mengine[0m [0;34m==[0m [0;34m"python"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1063[0;31m             [0mresults[0m[0;34m,[0m [0mres_index[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mapply_series_generator[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1064[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1065[0m             [0mresults[0m[0;34m,[0m [0mres_index[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mapply_series_numba[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py[0m in [0;36mapply_series_generator[0;34m(self)[0m
[1;32m   1079[0m             [0;32mfor[0m [0mi[0m[0;34m,[0m [0mv[0m [0;32min[0m [0menumerate[0m[0;34m([0m[0mseries_gen[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1080[0m                 [0;31m# ignore SettingWithCopy here in case the user mutates[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1081[0;31m                 [0mresults[0m[0;34m[[0m[0mi[0m[0;34m][0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mfunc[0m[0;34m([0m[0mv[0m[0;34m,[0m [0;34m*[0m[0mself[0m[0;34m.[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mself[0m[0;34m.[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1082[0m                 [0;32mif[0m [0misinstance[0m[0;34m([0m[0mresults[0m[0;34m[[0m[0mi[0m[0;34m][0m[0;34m,[0m [0mABCSeries[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1083[0m                     [0;31m# If we have a view on v, we need to make a copy because[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1255865017.py[0m in [0;36m<lambda>[0;34m(x)[0m
[1;32m    256[0m [0;34m[0m[0m
[1;32m    257[0m         X_val["which_longer"] = X_val.apply(
[0;32m--> 258[0;31m             [0;32mlambda[0m [0mx[0m[0;34m:[0m [0mwhich_longer[0m[0;34m([0m[0mx[0m[0;34m[[0m[0;34m"selected_text"[0m[0;34m][0m[0;34m,[0m [0mx[0m[0;34m[[0m[0;34m"predicted_selection"[0m[0;34m][0m[0;34m)[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    259[0m         )
[1;32m    260[0m         X_val["jaccard"] = X_val.apply(

[0;32m/tmp/ipykernel_11/1255865017.py[0m in [0;36mwhich_longer[0;34m(truth, prediction)[0m
[1;32m    198[0m [0;34m[0m[0m
[1;32m    199[0m [0;32mdef[0m [0mwhich_longer[0m[0;34m([0m[0mtruth[0m[0;34m,[0m [0mprediction[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 200[0;31m     [0mtruth_set[0m [0;34m=[0m [0mset[0m[0;34m([0m[0mtruth[0m[0;34m.[0m[0mlower[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0msplit[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    201[0m     [0mpred_set[0m [0;34m=[0m [0mset[0m[0;34m([0m[0mprediction[0m[0;34m.[0m[0mlower[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0msplit[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    202[0m     [0mintersect[0m [0;34m=[0m [0mtruth_set[0m[0;34m.[0m[0mintersection[0m[0;34m([0m[0mpred_set[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'float' object has no attribute 'lower'
