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

3.8

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

0.26029

# 6. Current score

0.29036

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.17626) has done: 'I first fix why your notebook doesn’t yield a Kaggle score: it currently never trains a model, never generates predictions for the test set, and never writes `submission.csv`. I keep your existing core logic (CountVectorizer + either MultinomialNB or linear SVM, and your heuristic selected_text reconstruction), and add a minimal “main” execution cell that (a) creates a safe train/validation split from the provided train data, (b) trains the chosen classifier, (c) builds a properly formatted submission with exactly `textID,selected_text`, and (d) writes `submission.csv`. I also make two tiny compatibility fixes that otherwise can crash on modern NumPy/Matplotlib: remove deprecated `np.int` usage and make `plotIt` robust when `y2=None`. These changes are directly aimed at producing a valid submission and getting you a measurable score toward the target.'
- What this solution (achieved 0.29036) has done: 'Your current approach mainly loses Jaccard because the selected-text reconstruction often returns an empty string or misses punctuation/spacing; that tanks word overlap even when sentiment is right. I keep your exact model/feature core (CountVectorizer + linear SVM and the same word-by-word scoring idea), but add a minimal “safe fallback” so we never output empty selections and we preserve the original tweet substring formatting when possible. I also fix a small indexing bug in `yPrep` (`> trainIdx` should be `>= trainIdx`) that can mis-handle null-text rows at the train/test boundary and corrupt alignment. These changes should improve the score toward your target without changing the modeling approach.'

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

trainFile = "/kaggle/input/tweet-sentiment-extraction/train.csv"
testFile = "/kaggle/input/tweet-sentiment-extraction/test.csv"
sample = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"
outTrain = "train_submission.csv"
outFile = "submission.csv"

df_train = pd.read_csv(trainFile, delimiter=",", dtype=str)
df_test = pd.read_csv(testFile, delimiter=",", dtype=str)

df_list = [df_train[["textID", "text", "sentiment"]], df_test]
df_all = pd.concat(df_list, ignore_index=True)
sentList = df_all["sentiment"].unique()



## === cell 1
trainIdx = 27481
max_feat = 25000
max_df = 0.11
min_df = 1
alpha = 1.0
c = 3.0
tune = False
test = False
method = False  # True selects MultinomialNB, False selects SVM




## === cell 2
def cleanText(text):
    """
    Some useful hints of cleaning text up using regex:
    https://www.kaggle.com/raenish/cheatsheet-text-helper-functions
    """
    import re
    import string

    text = str(text).strip().lower()
    text = re.sub(r"http[s]?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text




## === cell 3
def vectorizeIt(df, textCol, trainIdx, maxFeat=2000, maxDF=1.0, minDF=0.0):
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    vectorizer = CountVectorizer(
        stop_words="english",
        preprocessor=cleanText,
        max_features=maxFeat,
        max_df=maxDF,
        min_df=minDF,
    )
    index = list(np.where(df[textCol].isnull())[0])
    new_df = df[textCol].dropna(axis=0)
    vectorizer.fit(new_df)
    inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]
    wc = vectorizer.fit_transform(new_df)
    del new_df
    idxTr = [i for i in index if i < trainIdx]
    sz = trainIdx - len(idxTr)
    xTrain = wc[:sz]
    xTest = wc[sz:]
    return index, vocabulary, xTrain, xTest




## === cell 4
def sentArray(df, textCol):
    import numpy as np

    y = np.zeros(shape=(df[textCol].size), dtype=int)
    words = df[textCol].unique()
    for i in range(len(df)):
        s = df[textCol].iloc[i]
        t = np.where(words == s)
        y[i] = t[0]
    return y




## === cell 5
def jaccard(str1, str2):
    import re

    str1 = str(str1)
    str2 = str(str2)
    if len(str1) > 0:
        str1 = re.sub(r"\"", "", str1)
    if len(str2) > 0:
        str2 = re.sub(r"\"", "", str2)
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    if len(c) > 0:
        jacc = float(len(c)) / (len(a) + len(b) - len(c))
    else:
        jacc = 0.0
    return jacc




## === cell 6
def plotIt(x, y, name, title, xLabel, yLabel, label1, y2=None, label2=None):
    import matplotlib.pyplot as plt

    if y2 is not None:
        plt.plot(x, y, "bo-", label=label1)
        plt.plot(x, y2, "rs-", label=label2)
        plt.legend()
    else:
        plt.plot(x, y, "bo-")
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.savefig(name + ".png", format="png")
    plt.close()




## === cell 7
def yPrep(index, trainIdx, train, test):
    import numpy as np

    idxTr = [i for i in index if i < trainIdx]
    yTr = sentArray(train, "sentiment")
    yTrain = np.delete(yTr, idxTr)

    idxTe = [i - trainIdx for i in index if i >= trainIdx]
    yTe = sentArray(test, "sentiment")
    yTest = np.delete(yTe, idxTe)
    trainIdx = trainIdx - len(idxTr)
    return trainIdx, yTrain, yTest, idxTr, idxTe




## === cell 8
def multiNB(alpha, xTrain, xTest, yTrain, yTest):
    from sklearn.naive_bayes import MultinomialNB

    clf = MultinomialNB(alpha=alpha)
    clf.fit(xTrain, yTrain)
    accTrain = clf.score(xTrain, yTrain)
    accTest = clf.score(xTest, yTest)
    return clf, accTrain, accTest




## === cell 9
def tuneNB(all, train, test, trainIdx, max_feat, max_df, min_df, alpha):
    import numpy as np

    print("Tune MultinomialNB...")
    max_feat = 28500
    max_df = 1.0
    min_df = 0.0
    alpha = 1.0
    alpha = np.arange(start=40.0, stop=60.0, step=2.0)
    accTest = np.zeros(shape=(len(alpha)))
    accTrain = np.zeros(shape=(len(alpha)))
    for j in range(len(alpha)):
        tIdx = trainIdx
        index, bow, xTrain, xTest = vectorizeIt(
            all, "text", trainIdx, maxFeat=max_feat, maxDF=max_df, minDF=min_df
        )
        tIdx, yTrain, yTest, idxTr, idxTe = yPrep(index, tIdx, train, test)
        clf, accTrain[j], accTest[j] = multiNB(alpha[j], xTrain, xTest, yTrain, yTest)
    name = "alpha_v_acc"
    title = "alpha vs. Accuracy"
    xLabel = "alpha"
    yLabel = "Accuracy"
    label1 = "Train"
    label2 = "Test"
    plotIt(alpha, accTrain, name, title, xLabel, yLabel, label1, accTest, label2)
    alpha = alpha[np.argmax(accTest)]
    print("alpha: " + "{:.2f}".format(alpha))
    max_feat = np.arange(start=23000, stop=30000, step=500)
    accTest = np.zeros(shape=(len(max_feat)))
    accTrain = np.zeros(shape=(len(max_feat)))
    for j in range(0, len(max_feat)):
        tIdx = trainIdx
        index, bow, xTrain, xTest = vectorizeIt(
            all, "text", trainIdx, maxFeat=max_feat[j], maxDF=max_df, minDF=min_df
        )
        tIdx, yTrain, yTest, idxTr, idxTe = yPrep(index, tIdx, train, test)
        clf, accTrain[j], accTest[j] = multiNB(alpha, xTrain, xTest, yTrain, yTest)
    name = "feat_v_acc"
    title = "max_feat vs. Accuracy"
    xLabel = "max_feat"
    yLabel = "Accuracy"
    label1 = "Train"
    label2 = "Test"
    plotIt(max_feat, accTrain, name, title, xLabel, yLabel, label1, accTest, label2)
    max_feat = max_feat[np.argmax(accTest)]
    print("max_feat: " + "{:.2f}".format(max_feat))
    min_df = np.arange(start=0.000, stop=0.010, step=0.001)
    accTest = np.zeros(shape=(len(min_df)))
    accTrain = np.zeros(shape=(len(min_df)))
    for j in range(0, len(min_df)):
        tIdx = trainIdx
        index, bow, xTrain, xTest = vectorizeIt(
            all, "text", trainIdx, maxFeat=max_feat, maxDF=max_df, minDF=min_df[j]
        )
        tIdx, yTrain, yTest, idxTr, idxTe = yPrep(index, tIdx, train, test)
        clf, accTrain[j], accTest[j] = multiNB(alpha, xTrain, xTest, yTrain, yTest)
    name = "min_df_v_acc"
    title = "min_df vs. Accuracy"
    xLabel = "min_df"
    yLabel = "Accuracy"
    label1 = "Train"
    label2 = "Test"
    plotIt(min_df, accTrain, name, title, xLabel, yLabel, label1, accTest, label2)
    min_df = min_df[np.argmax(accTest)]
    print("min_df: " + "{:.3f}".format(min_df))
    max_df = np.arange(start=1.0, stop=max(min_df + 0.1, 0.1), step=-0.1)
    accTest = np.zeros(shape=(len(max_df)))
    accTrain = np.zeros(shape=(len(max_df)))
    for j in range(0, len(max_df)):
        tIdx = trainIdx
        index, bow, xTrain, xTest = vectorizeIt(
            all, "text", trainIdx, maxFeat=max_feat, maxDF=max_df[j], minDF=min_df
        )
        tIdx, yTrain, yTest, idxTr, idxTe = yPrep(index, tIdx, train, test)
        clf, accTrain[j], accTest[j] = multiNB(alpha, xTrain, xTest, yTrain, yTest)
    name = "max_df_v_acc"
    title = "max_df vs. Accuracy"
    xLabel = "max_df"
    yLabel = "Accuracy"
    label1 = "Train"
    label2 = "Test"
    plotIt(max_df, accTrain, name, title, xLabel, yLabel, label1, accTest, label2)
    max_df = max_df[np.argmax(accTest)]
    print("max_df: " + "{:.3f}".format(max_df))




## === cell 10
def _best_substring_from_words(original_text, chosen_words):
    """
    Minimal post-processing to improve word-level Jaccard without changing core logic:
    - If chosen_words is empty, return full original_text (prevents catastrophic 0 Jaccard).
    - Otherwise, find the shortest substring in the original tweet that contains the chosen words
      in order, preserving punctuation/spacing as Kaggle expects.
    """
    import re

    text = "" if original_text is None else str(original_text)
    if text == "nan":
        return ""

    chosen_words = [w for w in chosen_words if str(w).strip() != ""]
    if len(chosen_words) == 0:
        return text.strip()

    low = text.lower()
    positions = []
    start = 0
    for w in chosen_words:
        wl = str(w).lower()
        m = re.search(r"(?<!\w)" + re.escape(wl) + r"(?!\w)", low[start:])
        if not m:
            return " ".join(chosen_words).strip()
        s = start + m.start()
        e = start + m.end()
        positions.append((s, e))
        start = e

    s0 = positions[0][0]
    e0 = positions[-1][1]
    return text[s0:e0].strip()




## === cell 11
def buildSelTextNB(df, test, idxList, pred, bow, featLogProb, sentList):
    import pandas as pd

    """
    neutral == 0
    negative == 1
    positive == 2
    """
    print("Build MultinomialNB output...")
    if test:
        retDF = pd.DataFrame(columns=["textID", "selected_text", "sentiment"])
        retDF["textID"] = df["textID"]
    else:
        retDF = pd.DataFrame(columns=["textID", "selected_text"])
        retDF["textID"] = df["textID"]
    j = 0
    for i in range(len(retDF)):
        if i in idxList:
            aword = str(df["text"].iloc[i])
            if aword == "nan":
                retDF["selected_text"].iloc[i] = '"' + '"'  # keep original behavior
            else:
                retDF["selected_text"].iloc[i] = '"' + aword + '"'
            if test:
                retDF["sentiment"].iloc[i] = "neutral"
        elif pred[j] == 0:
            retDF["selected_text"].iloc[i] = '"' + df["text"].iloc[i] + '"'
            if test:
                retDF["sentiment"].iloc[i] = "neutral"
            j = j + 1  # advance the prediction index
        else:  # Else add selected_text back in based on probability of feature
            idxA = pred[j]
            words = df["text"].iloc[i].split()
            chosen = []
            for word in words:
                wordCheck = cleanText(word)
                if wordCheck in bow:
                    probWord = (featLogProb)[idxA, bow.index(wordCheck)]
                    probNeg = (featLogProb)[1, bow.index(wordCheck)]
                    probPos = (featLogProb)[2, bow.index(wordCheck)]
                    if ((idxA == 1) and (probWord > probPos)) or (
                        (idxA == 2) and (probWord > probNeg)
                    ):
                        chosen.append(word)

            outstring = _best_substring_from_words(df["text"].iloc[i], chosen)
            retDF["selected_text"].iloc[i] = '"' + outstring + '"'
            if test:
                retDF["sentiment"].iloc[i] = sentList[idxA]
            j = j + 1  # advance the prediction index
    return retDF




## === cell 12
def svmClass(xTrain, xTest, yTrain, yTest, c=1.0):
    from sklearn import svm

    clf = svm.SVC(C=c, kernel="linear")
    clf.fit(xTrain, yTrain)
    accTrain = clf.score(xTrain, yTrain)
    accTest = clf.score(xTest, yTest)
    return clf, accTrain, accTest




## === cell 13
def tuneSVM(xTrain, xTest, yTrain, yTest):
    import numpy as np

    print("Tune SVM...")
    c = np.arange(start=0.01, stop=0.11, step=0.01)
    accTest = np.zeros(shape=(len(c)))
    accTrain = np.zeros(shape=(len(c)))
    for j in range(len(c)):
        clf, accTrain[j], accTest[j] = svmClass(xTrain, xTest, yTrain, yTest, c[j])
    name = "c_v_acc"
    title = "c vs. Accuracy"
    xLabel = "c"
    yLabel = "Accuracy"
    label1 = "Train"
    label2 = "Test"
    plotIt(c, accTrain, name, title, xLabel, yLabel, label1, accTest, label2)
    c = c[np.argmax(accTest)]
    print("c: " + "{:.4f}".format(c))




## === cell 14
def buildSelTextSVM(df, test, idxList, pred, bow, sentList, y, w):
    import pandas as pd
    import numpy as np

    """
    neutral == 0
    negative == 1
    positive == 2
    """
    print("Build SVM output...")
    if test:
        retDF = pd.DataFrame(columns=["textID", "selected_text", "sentiment"])
        retDF["textID"] = df["textID"]
    else:
        retDF = pd.DataFrame(columns=["textID", "selected_text"])
        retDF["textID"] = df["textID"]
    j = 0
    for i in range(len(retDF)):
        if i in idxList:
            aword = str(df["text"].iloc[i])
            if aword == "nan":
                retDF["selected_text"].iloc[i] = '"' + '"'
            else:
                retDF["selected_text"].iloc[i] = '"' + aword + '"'
            if test:
                retDF["sentiment"].iloc[i] = "neutral"
        elif pred[j] == 0:
            retDF["selected_text"].iloc[i] = '"' + df["text"].iloc[i] + '"'
            if test:
                retDF["sentiment"].iloc[i] = "neutral"
            j = j + 1  # advance the prediction index
        else:  # Else add selected_text back in based on distance from hyperplane
            idxA = pred[j]
            words = df["text"].iloc[i].split()
            chosen = []
            for word in words:
                wordCheck = cleanText(word)
                if wordCheck in bow:
                    d = np.zeros(shape=(3))
                    dWord = 0.0
                    yVal = y[j, idxA]
                    iBow = bow.index(wordCheck)
                    if w[idxA, iBow] != 0.0:
                        dWord = yVal / abs(w[idxA, iBow])
                    for k in range(3):
                        if w[k, iBow] != 0.0:
                            d[k] = y[j, k] / abs(w[k, iBow])
                    if (
                        (
                            ((idxA == 1) and (dWord > d[2]))
                            or ((idxA == 2) and (dWord > d[1]))
                        )
                        and (dWord > 0.0)
                        and (dWord > d[0])
                    ):
                        chosen.append(word)

            outstring = _best_substring_from_words(df["text"].iloc[i], chosen)
            retDF["selected_text"].iloc[i] = '"' + outstring + '"'
            if test:
                retDF["sentiment"].iloc[i] = sentList[idxA]
            j = j + 1  # advance the prediction index
    return retDF




## === cell 15
def sentArray(df, textCol):
    import numpy as np

    y = np.zeros(shape=(df[textCol].size), dtype=int)
    words = df[textCol].unique()
    for i in range(len(df)):
        s = df[textCol].iloc[i]
        t = np.where(words == s)
        y[i] = t[0]
    return y




## === cell 16
from sklearn.model_selection import train_test_split

df_train = df_train.copy()
df_test = df_test.copy()
df_train["text"] = df_train["text"].astype(str)
df_train["sentiment"] = df_train["sentiment"].astype(str)
df_train["selected_text"] = df_train["selected_text"].astype(str)
df_test["text"] = df_test["text"].astype(str)
df_test["sentiment"] = df_test["sentiment"].astype(str)

tr_df, va_df = train_test_split(
    df_train[["textID", "text", "sentiment", "selected_text"]],
    test_size=0.1,
    random_state=42,
    stratify=df_train["sentiment"],
)

df_all_full = pd.concat(
    [
        df_train[["textID", "text", "sentiment"]],
        df_test[["textID", "text", "sentiment"]],
    ],
    ignore_index=True,
)
trainIdx_full = len(df_train)

index, bow, xTrain, xTest = vectorizeIt(
    df_all_full, "text", trainIdx_full, maxFeat=max_feat, maxDF=max_df, minDF=min_df
)

_, yTrain, yTest, idxTr, idxTe = yPrep(
    index, trainIdx_full, df_train[["sentiment"]], df_test[["sentiment"]]
)

if method:  # MultinomialNB
    clf, accTrain, accTest = multiNB(alpha, xTrain, xTest, yTrain, yTest)
    print(f"NB sentiment accuracy (train/test proxy): {accTrain:.4f}/{accTest:.4f}")
    pred_test = clf.predict(xTest)
    sub_full = buildSelTextNB(
        df_test[["textID", "text"]],
        True,
        idxTe,
        pred_test,
        bow,
        clf.feature_log_prob_,
        sentList,
    )
else:  # SVM
    clf, accTrain, accTest = svmClass(xTrain, xTest, yTrain, yTest, c=c)
    print(f"SVM sentiment accuracy (train/test proxy): {accTrain:.4f}/{accTest:.4f}")
    pred_test = clf.predict(xTest)
    y_dist = clf.decision_function(xTest)  # shape (n_samples, 3) for 3 classes
    w = clf.coef_  # shape (3, n_features)
    sub_full = buildSelTextSVM(
        df_test[["textID", "text"]], True, idxTe, pred_test, bow, sentList, y_dist, w
    )

submission = sub_full[["textID", "selected_text"]].copy()
submission.to_csv(outFile, index=False)

print("Wrote:", outFile)
print(submission.head(10))
