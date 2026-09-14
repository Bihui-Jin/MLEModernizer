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

0.39408

# 6. Current score

0.26044

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.07369) has done: 'Your notebook currently can’t yield a score because it never reaches a “train → predict → write submission.csv” step, and it also has a few issues that crash on modern NumPy/Pandas (e.g., `np.int`, `len(y2)` when `y2=None`, and an invalid `trainIdx`). I keep your core approach (CountVectorizer + sentiment classifier + heuristic token selection) and add only the missing end-to-end pipeline to generate `submission.csv`. I also make minimal “correctness/stability” fixes so the code runs reliably and the output rows align exactly with `sample_submission.csv`. Finally, I switch the default `trainIdx` to the actual train size so train/test splitting works as intended (this should improve score from “no submission” to a meaningful baseline toward your target).'
- What this solution (achieved 0.15687) has done: 'Your score is far below the target (0.07369 vs 0.39408), so we should improve performance with minimal, low-risk changes that keep your approach (CountVectorizer + sentiment classifier + your word-picking heuristic) intact. The biggest gain without changing the “core model” is to stop training the sentiment classifier on the concatenated train+test vocabulary fit (leakage/shift isn’t helping here) and instead fit the vectorizer on train only, then transform test—this usually improves generalization and downstream selected-text quality. Next, your SVM is currently extremely over-regularized (`C=1e-5`), which tends to predict majority/neutral and yields very poor Jaccard; nudging `C` to a still-conservative value (e.g., 0.1) is a minimal hyperparameter change that should move the score substantially toward your target band. Finally, we keep your submission quoting behavior but add a safe fallback: if the heuristic selects nothing, return the full tweet (a common baseline for this competition that avoids empty strings tanking Jaccard).'
- What this solution (achieved 0.26035) has done: 'Your current baseline is far below the target, so the smallest safe way to move the Jaccard score upward (without changing your overall “vectorize → sentiment-classify → heuristic word-pick” approach) is to (1) use a tweet-extraction-friendly classifier that still matches your SVM logic (LinearSVC) and (2) make your token selection use signed class weights directly (rather than the current distance/ratio heuristic which tends to fall back to full text too often). These are minimal, local changes: same bag-of-words features, same linear SVM family, same sentiment prediction, same “select subset of original tokens and fall back to full tweet”. I also make the sentiment label encoding deterministic (`neutral/negative/positive`) so class indices align with your downstream selection logic. This should increase score materially toward your target while keeping runtime well under the limit.'
- What this solution (achieved 0.26044) has done: 'Your current score (0.26035) is well below the target (0.39408), so we should improve Jaccard with the smallest changes that keep your exact “CountVectorizer → linear SVM sentiment → heuristic token selection” pipeline intact. The biggest low-risk gain is to make token selection use the SVM margin strengths (per-class decision values) instead of just positive weights, so we only include words that actively push the tweet toward the predicted sentiment more than toward neutral/opposite. To avoid O(V) lookups per token (and keep runtime <600s), we also replace `bow.index()` with a one-time `{token: index}` map; this is performance-only and doesn’t change semantics. Finally, we add a sentiment-aware fallback consistent with common baselines for this competition: for neutral predictions, return the full tweet; for positive/negative with empty selection, still fall back to full tweet (as you already do), preserving submission validity.'

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

df_all = pd.concat(
    [df_train[["textID", "text", "sentiment"]], df_test], ignore_index=True
)

sentList = np.array(["neutral", "negative", "positive"], dtype=object)



## === cell 1
tune = False
test = False
method = False  # True selects MultinomialNB, False selects SVM

trainIdx = len(df_train)

if method:
    max_df = 1.0
    min_df = 9  # maybe try higher?
    max_feat = 2000
    alpha = 6.0
else:
    max_df = 1.0  # 0.1
    min_df = 1  # 14
    max_feat = 25000  # 2000
    c = 0.1  # was 0.00001




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
def vectorizeTrainTest(train_df, test_df, textCol, maxFeat=2000, maxDF=1.0, minDF=0.0):
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    vectorizer = CountVectorizer(
        stop_words="english",
        preprocessor=cleanText,
        max_features=maxFeat,
        max_df=maxDF,
        min_df=minDF,
    )

    tr_text = train_df[textCol]
    te_text = test_df[textCol]

    idxTrNull = list(np.where(tr_text.isnull())[0])
    idxTeNull = list(np.where(te_text.isnull())[0])

    tr_nonnull = tr_text.dropna(axis=0)
    vectorizer.fit(tr_nonnull)

    inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    bow = [inv_vocab[i] for i in range(len(inv_vocab))]

    xTrain_full = vectorizer.transform(tr_nonnull)
    te_nonnull = te_text.dropna(axis=0)
    xTest_full = vectorizer.transform(te_nonnull)

    return idxTrNull, idxTeNull, bow, xTrain_full, xTest_full




## === cell 5
def sentArray(df, textCol):
    import numpy as np

    mapping = {"neutral": 0, "negative": 1, "positive": 2}
    y = np.zeros(shape=(df[textCol].size), dtype=int)
    for i in range(len(df)):
        s = str(df[textCol].iloc[i])
        y[i] = mapping.get(s, 0)
    return y




## === cell 6
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




## === cell 7
def plotIt(x, y, name, title, xLabel, yLabel, label1, y2=None, label2=None):
    import matplotlib.pyplot as plt

    if y2 is not None:
        plt.plot(x, y, "bo-", label=label1)
        plt.plot(x, y2, "rs-", label=label2)
        plt.legend()
    else:
        plt.plot(x, y, "bo-")
        plt.legend([label1])
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.savefig(name + ".png", format="png")
    plt.close()




## === cell 8
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




## === cell 9
def yPrepTrainTest(idxTrNull, idxTeNull, train, test):
    import numpy as np

    yTr = sentArray(train, "sentiment")
    yTrain = np.delete(yTr, idxTrNull)

    yTe = sentArray(test, "sentiment")
    yTest = np.delete(yTe, idxTeNull)

    return yTrain, yTest




## === cell 10
def multiNB(alpha, xTrain, xTest, yTrain, yTest):
    from sklearn.naive_bayes import MultinomialNB

    clf = MultinomialNB(alpha=alpha)
    clf.fit(xTrain, yTrain)
    accTrain = clf.score(xTrain, yTrain)
    accTest = clf.score(xTest, yTest)
    return clf, accTrain, accTest




## === cell 11
def tuneNB(all, train, test, trainIdx, max_feat, max_df, min_df, alpha):
    import numpy as np
    from sklearn.model_selection import train_test_split

    print("Tune MultinomialNB...")
    cvSz = 5  # iterations of cross validation
    tune_max_df = True
    tune_min_df = True
    tune_max_feat = True
    tune_alpha = True
    if tune_max_df:
        max_df = np.arange(start=1.0, stop=0.0, step=-0.1)
        accTrainAll = np.zeros(shape=(cvSz, len(max_df)))
        accValidAll = np.zeros(shape=(cvSz, len(max_df)))
        for j in range(len(max_df)):
            for i in range(cvSz):
                tIdx = trainIdx
                index, bow, xTrain, xTest = vectorizeIt(
                    all,
                    "text",
                    trainIdx,
                    maxFeat=max_feat,
                    maxDF=max_df[j],
                    minDF=min_df,
                )
                tIdx, yTrain, yTest, idxTr, idxTe = yPrep(index, tIdx, train, test)
                xTr, xVld, yTr, yVld = train_test_split(
                    xTrain, yTrain, test_size=0.33, random_state=i, shuffle=True
                )
                clf, accTrainAll[i, j], accValidAll[i, j] = multiNB(
                    alpha, xTr, xVld, yTr, yVld
                )
        accTrain = accTrainAll.mean(axis=0)
        accValid = accValidAll.mean(axis=0)
        name = "NB_max_df_v_acc"
        title = "MultinomialNB: max_df vs. Accuracy"
        xLabel = "max_df"
        yLabel = "Accuracy"
        label1 = "Train"
        label2 = "Valid"
        plotIt(max_df, accTrain, name, title, xLabel, yLabel, label1, accValid, label2)
        max_df = max_df[np.argmax(accValid)]
        print("max_df: " + "{:.3f}".format(max_df))
    if tune_min_df:
        min_df = np.arange(start=1, stop=20, step=1)
        accTrainAll = np.zeros(shape=(cvSz, len(min_df)))
        accValidAll = np.zeros(shape=(cvSz, len(min_df)))
        for j in range(len(min_df)):
            for i in range(cvSz):
                tIdx = trainIdx
                index, bow, xTrain, xTest = vectorizeIt(
                    all,
                    "text",
                    trainIdx,
                    maxFeat=max_feat,
                    maxDF=max_df,
                    minDF=min_df[j],
                )
                tIdx, yTrain, yTest, idxTr, idxTe = yPrep(index, tIdx, train, test)
                xTr, xVld, yTr, yVld = train_test_split(
                    xTrain, yTrain, test_size=0.33, random_state=i, shuffle=True
                )
                clf, accTrainAll[i, j], accValidAll[i, j] = multiNB(
                    alpha, xTr, xVld, yTr, yVld
                )
        accTrain = accTrainAll.mean(axis=0)
        accValid = accValidAll.mean(axis=0)
        name = "NB_min_df_v_acc"
        title = "MultinomialNB: min_df vs. Accuracy"
        xLabel = "min_df"
        yLabel = "Accuracy"
        label1 = "Train"
        label2 = "Valid"
        plotIt(min_df, accTrain, name, title, xLabel, yLabel, label1, accValid, label2)
        min_df = min_df[np.argmax(accValid)]
        print("min_df: " + "{:.3f}".format(min_df))
    if tune_max_feat:
        max_feat = np.arange(start=1000, stop=26000, step=1000)
        accTrainAll = np.zeros(shape=(cvSz, len(max_feat)))
        accValidAll = np.zeros(shape=(cvSz, len(max_feat)))
        for j in range(len(max_feat)):
            for i in range(cvSz):
                tIdx = trainIdx
                index, bow, xTrain, xTest = vectorizeIt(
                    all,
                    "text",
                    trainIdx,
                    maxFeat=max_feat[j],
                    maxDF=max_df,
                    minDF=min_df,
                )
                tIdx, yTrain, yTest, idxTr, idxTe = yPrep(index, tIdx, train, test)
                xTr, xVld, yTr, yVld = train_test_split(
                    xTrain, yTrain, test_size=0.33, random_state=i, shuffle=True
                )
                clf, accTrainAll[i, j], accValidAll[i, j] = multiNB(
                    alpha, xTr, xVld, yTr, yVld
                )
        accTrain = accTrainAll.mean(axis=0)
        accValid = accValidAll.mean(axis=0)
        name = "NB_feat_v_acc"
        title = "MultinomialNB: max_feat vs. Accuracy"
        xLabel = "max_feat"
        yLabel = "Accuracy"
        label1 = "Train"
        label2 = "Valid"
        plotIt(
            max_feat, accTrain, name, title, xLabel, yLabel, label1, accValid, label2
        )
        max_feat = max_feat[np.argmax(accValid)]
        print("max_feat: " + "{:.2f}".format(max_feat))
    if tune_alpha:
        alpha = np.arange(start=1.0, stop=21.0, step=1.0)
        accTrainAll = np.zeros(shape=(cvSz, len(alpha)))
        accValidAll = np.zeros(shape=(cvSz, len(alpha)))
        for j in range(len(alpha)):
            for i in range(cvSz):
                tIdx = trainIdx
                index, bow, xTrain, xTest = vectorizeIt(
                    all, "text", trainIdx, maxFeat=max_feat, maxDF=max_df, minDF=min_df
                )
                tIdx, yTrain, yTest, idxTr, idxTe = yPrep(index, tIdx, train, test)
                xTr, xVld, yTr, yVld = train_test_split(
                    xTrain, yTrain, test_size=0.33, random_state=i, shuffle=True
                )
                clf, accTrainAll[i, j], accValidAll[i, j] = multiNB(
                    alpha[j], xTr, xVld, yTr, yVld
                )
        accTrain = accTrainAll.mean(axis=0)
        accValid = accValidAll.mean(axis=0)
        name = "NB_alpha_v_acc"
        title = "MultinomialNB: alpha vs. Accuracy"
        xLabel = "alpha"
        yLabel = "Accuracy"
        label1 = "Train"
        label2 = "Valid"
        plotIt(alpha, accTrain, name, title, xLabel, yLabel, label1, accValid, label2)
        alpha = alpha[np.argmax(accValid)]
        print("alpha: " + "{:.2f}".format(alpha))




## === cell 12
def buildSelTextNB(df, test, idxList, pred, bow, featLogProb, sentList):
    import pandas as pd

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
                retDF["selected_text"].iloc[i] = '"' + '"'
            else:
                retDF["selected_text"].iloc[i] = '"' + aword + '"'
            if test:
                retDF["sentiment"].iloc[i] = "neutral"
        elif pred[j] == 0:
            retDF["selected_text"].iloc[i] = '"' + df["text"].iloc[i] + '"'
            if test:
                retDF["sentiment"].iloc[i] = "neutral"
            j = j + 1
        else:
            outstring = ""
            idxA = pred[j]
            words = df["text"].iloc[i].split()
            for word in words:
                wordCheck = cleanText(word)
                if wordCheck in bow:
                    probWord = (featLogProb)[idxA, bow.index(wordCheck)]
                    probNeg = (featLogProb)[1, bow.index(wordCheck)]
                    probPos = (featLogProb)[2, bow.index(wordCheck)]
                    if ((idxA == 1) and (probWord > probPos)) or (
                        (idxA == 2) and (probWord > probNeg)
                    ):
                        if len(outstring) == 0:
                            outstring = word
                        else:
                            outstring = outstring + " " + word
            if len(outstring.strip()) == 0:
                outstring = str(df["text"].iloc[i])
            outstring = '"' + outstring + '"'
            retDF["selected_text"].iloc[i] = outstring
            if test:
                retDF["sentiment"].iloc[i] = sentList[idxA]
            j = j + 1
    return retDF




## === cell 13
def svmClass(xTrain, xTest, yTrain, yTest, c=1.0):
    from sklearn.svm import LinearSVC

    clf = LinearSVC(C=c)
    clf.fit(xTrain, yTrain)
    accTrain = clf.score(xTrain, yTrain)
    accTest = clf.score(xTest, yTest)
    return clf, accTrain, accTest




## === cell 14
def tuneSVM(xTrain, xTest, yTrain, yTest):
    import numpy as np
    from sklearn.model_selection import train_test_split

    print("Tune SVM...")
    cvSz = 5
    tune_max_df = True
    tune_min_df = True
    tune_max_feat = True
    tune_c = True
    if tune_max_df:
        max_df = np.arange(start=1.0, stop=0.0, step=-0.1)
        accTrainAll = np.zeros(shape=(cvSz, len(max_df)))
        accValidAll = np.zeros(shape=(cvSz, len(max_df)))
        for j in range(len(max_df)):
            for i in range(cvSz):
                tIdx = trainIdx
                index, bow, xTrain, xTest = vectorizeIt(
                    all,
                    "text",
                    trainIdx,
                    maxFeat=max_feat,
                    maxDF=max_df[j],
                    minDF=min_df,
                )
                tIdx, yTrain, yTest, idxTr, idxTe = yPrep(index, tIdx, train, test)
                xTr, xVld, yTr, yVld = train_test_split(
                    xTrain, yTrain, test_size=0.33, random_state=i, shuffle=True
                )
                clf, accTrainAll[i, j], accValidAll[i, j] = svmClass(
                    xTr, xVld, yTr, yVld, c
                )
        accTrain = accTrainAll.mean(axis=0)
        accValid = accValidAll.mean(axis=0)
        plotIt(
            max_df,
            accTrain,
            "SVM_max_df_v_acc",
            "SVM: max_df vs. Accuracy",
            "max_df",
            "Accuracy",
            "Train",
            accValid,
            "Valid",
        )
        max_df = max_df[np.argmax(accValid)]
        print("max_df: " + "{:.3f}".format(max_df))
    if tune_min_df:
        min_df = np.arange(start=1, stop=20, step=1)
        accTrainAll = np.zeros(shape=(cvSz, len(min_df)))
        accValidAll = np.zeros(shape=(cvSz, len(min_df)))
        for j in range(len(min_df)):
            for i in range(cvSz):
                tIdx = trainIdx
                index, bow, xTrain, xTest = vectorizeIt(
                    all,
                    "text",
                    trainIdx,
                    maxFeat=max_feat,
                    maxDF=max_df,
                    minDF=min_df[j],
                )
                tIdx, yTrain, yTest, idxTr, idxTe = yPrep(index, tIdx, train, test)
                xTr, xVld, yTr, yVld = train_test_split(
                    xTrain, yTrain, test_size=0.33, random_state=i, shuffle=True
                )
                clf, accTrainAll[i, j], accValidAll[i, j] = svmClass(
                    xTr, xVld, yTr, yVld, c
                )
        accTrain = accTrainAll.mean(axis=0)
        accValid = accValidAll.mean(axis=0)
        plotIt(
            min_df,
            accTrain,
            "SVM_min_df_v_acc",
            "SVM: min_df vs. Accuracy",
            "min_df",
            "Accuracy",
            "Train",
            accValid,
            "Valid",
        )
        min_df = min_df[np.argmax(accValid)]
        print("min_df: " + "{:.3f}".format(min_df))
    if tune_max_feat:
        max_feat = np.arange(start=1000, stop=26000, step=1000)
        accTrainAll = np.zeros(shape=(cvSz, len(max_feat)))
        accValidAll = np.zeros(shape=(cvSz, len(max_feat)))
        for j in range(len(max_feat)):
            for i in range(cvSz):
                tIdx = trainIdx
                index, bow, xTrain, xTest = vectorizeIt(
                    all,
                    "text",
                    trainIdx,
                    maxFeat=max_feat[j],
                    maxDF=max_df,
                    minDF=min_df,
                )
                tIdx, yTrain, yTest, idxTr, idxTe = yPrep(index, tIdx, train, test)
                xTr, xVld, yTr, yVld = train_test_split(
                    xTrain, yTrain, test_size=0.33, random_state=i, shuffle=True
                )
                clf, accTrainAll[i, j], accValidAll[i, j] = svmClass(
                    xTr, xVld, yTr, yVld, c
                )
        accTrain = accTrainAll.mean(axis=0)
        accValid = accValidAll.mean(axis=0)
        plotIt(
            max_feat,
            accTrain,
            "SVM_feat_v_acc",
            "SVM: max_feat vs. Accuracy",
            "max_feat",
            "Accuracy",
            "Train",
            accValid,
            "Valid",
        )
        max_feat = max_feat[np.argmax(accValid)]
        print("max_feat: " + "{:.2f}".format(max_feat))
    if tune_c:
        c = np.arange(start=0.5, stop=5.0, step=0.5)
        accTrainAll = np.zeros(shape=(cvSz, len(c)))
        accValidAll = np.zeros(shape=(cvSz, len(c)))
        for j in range(len(c)):
            for i in range(cvSz):
                tIdx = trainIdx
                index, bow, xTrain, xTest = vectorizeIt(
                    all, "text", trainIdx, maxFeat=max_feat, maxDF=max_df, minDF=min_df
                )
                tIdx, yTrain, yTest, idxTr, idxTe = yPrep(index, tIdx, train, test)
                from sklearn.model_selection import train_test_split

                xTr, xVld, yTr, yVld = train_test_split(
                    xTrain, yTrain, test_size=0.33, random_state=i, shuffle=True
                )
                clf, accTrainAll[i, j], accValidAll[i, j] = svmClass(
                    xTr, xVld, yTr, yVld, c[j]
                )
        accTrain = accTrainAll.mean(axis=0)
        accValid = accValidAll.mean(axis=0)
        plotIt(
            c,
            accTrain,
            "SVM_c_v_acc",
            "SVM: c vs. Accuracy",
            "c",
            "Accuracy",
            "Train",
            accValid,
            "Valid",
        )
        c = c[np.argmax(accValid)]
        print("c: " + "{:.2f}".format(c))




## === cell 15
def buildSelTextSVM(df, test, idxList, pred, bow, sentList, y_dec, w, bow2idx):
    import pandas as pd
    import numpy as np

    print("Build SVM output...")
    if test:
        retDF = pd.DataFrame(columns=["textID", "selected_text", "sentiment"])
        retDF["textID"] = df["textID"]
    else:
        retDF = pd.DataFrame(columns=["textID", "selected_text"])
        retDF["textID"] = df["textID"]

    y_dec = np.asarray(y_dec)
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
            continue

        idxA = int(pred[j])

        if idxA == 0:
            retDF["selected_text"].iloc[i] = '"' + str(df["text"].iloc[i]) + '"'
            if test:
                retDF["sentiment"].iloc[i] = "neutral"
            j += 1
            continue

        out_tokens = []
        words = str(df["text"].iloc[i]).split()

        opp = 2 if idxA == 1 else 1
        w_a = w[idxA]
        w_0 = w[0]
        w_o = w[opp]

        for word in words:
            wordCheck = cleanText(word)
            iBow = bow2idx.get(wordCheck, None)
            if iBow is None:
                continue
            contrib_a_vs_0 = w_a[iBow] - w_0[iBow]
            contrib_a_vs_o = w_a[iBow] - w_o[iBow]
            if (contrib_a_vs_0 > 0.0) and (contrib_a_vs_o > 0.0):
                out_tokens.append(word)

        if len(out_tokens) == 0:
            outstring = str(df["text"].iloc[i])
        else:
            outstring = " ".join(out_tokens)

        retDF["selected_text"].iloc[i] = '"' + outstring + '"'
        if test:
            retDF["sentiment"].iloc[i] = sentList[idxA]
        j += 1

    return retDF




## === cell 16
from sklearn.model_selection import train_test_split

df_sample = pd.read_csv(sample, dtype=str)

idxTrNull, idxTeNull, bow, xTrain, xTest = vectorizeTrainTest(
    df_train, df_test, "text", maxFeat=max_feat, maxDF=max_df, minDF=min_df
)
yTrain, yTest = yPrepTrainTest(idxTrNull, idxTeNull, df_train, df_test)

bow2idx = {t: i for i, t in enumerate(bow)}

if tune:
    if method:
        tuneNB(df_all, df_train, df_test, trainIdx, max_feat, max_df, min_df, alpha)
    else:
        tuneSVM(xTrain, xTest, yTrain, yTest)

if method:
    clf, accTrain, _ = multiNB(alpha, xTrain, xTrain, yTrain, yTrain)
    print(f"Train accuracy (NB, on train): {accTrain:.4f}")
    pred_test = clf.predict(xTest)
    featLogProb = clf.feature_log_prob_
    predDF = buildSelTextNB(
        df_test, True, idxTeNull, pred_test, bow, featLogProb, sentList
    )
else:
    clf, accTrain, _ = svmClass(xTrain, xTrain, yTrain, yTrain, c=c)
    print(f"Train accuracy (SVM, on train): {accTrain:.4f}")
    pred_test = clf.predict(xTest)

    y_dec = clf.decision_function(xTest)
    w = clf.coef_
    predDF = buildSelTextSVM(
        df_test, True, idxTeNull, pred_test, bow, sentList, y_dec, w, bow2idx
    )

sub = df_sample[["textID"]].merge(
    predDF[["textID", "selected_text"]], on="textID", how="left"
)

missing = sub["selected_text"].isna()
if missing.any():
    fill_map = df_test.set_index("textID")["text"].fillna("").to_dict()
    sub.loc[missing, "selected_text"] = sub.loc[missing, "textID"].map(
        lambda tid: '"' + str(fill_map.get(tid, "")) + '"'
    )

sub["selected_text"] = sub["selected_text"].fillna('""').astype(str)

sub.to_csv(outFile, index=False)
print(f"Wrote {outFile} with shape {sub.shape} and columns {list(sub.columns)}")
print(sub.head())
