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

0.26964

# 6. Current score

0.16461

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.14549) has done: 'I fix the crashing length-mismatch by replacing the hard-coded `trainIdx` with the true size of `df_train`, so `xTrain` and `yTrain` are built over the same number of rows. I also fix an off-by-one bug in `yPrep` (`>= trainIdx` instead of `> trainIdx`) that can silently misalign test sentiment labels when null rows exist. To ensure a valid submission is always produced, I keep the write path and filename the same and add a small safety fallback so `selected_text` is never empty (empty predictions hurt Jaccard and can cause formatting issues). These changes preserve the core model/training logic and only address alignment + output robustness.'
- What this solution (achieved 0.10477) has done: 'To move the public Jaccard score up toward your 0.26964 target (from 0.14549) without changing the overall approach, I make two minimal, high-impact adjustments that align better with the competition metric: (1) stop stripping punctuation/quotes during cleaning so the predicted spans can better match the ground-truth text (Jaccard is whitespace-based but exact text content matters), and (2) add a simple, sentiment-aware post-processing rule: for neutral predict full tweet; for positive/negative, select the longest contiguous run of “sentiment-indicative” tokens based on the classifier’s per-token class evidence, rather than a sparse set of tokens. This keeps your same CountVectorizer + SVM/NB classifier and training flow, but improves the span construction which is the main driver of Jaccard. I also keep the existing non-empty fallback and ensure the submission CSV stays correctly formatted and quoted.'
- What this solution (achieved 0.16461) has done: 'Your current score (0.10477) is far below the target (0.26964), so we should cautiously improve without changing the core “CountVectorizer + linear SVM/NB classifier + heuristic span reconstruction” approach. The biggest score drag in your current code is a logic bug in SVM span selection: it compares the full tweet’s class score (per-class decision value) but then divides by a single word weight, which doesn’t measure that word’s contribution; this makes the “keep” mask mostly noise. I fix this by using the linear SVM’s actual per-token contribution for the predicted-vs-opposite class (a minimal, directly relevant correction that keeps the same model/training) and then still selecting the best contiguous span as you already do. I also remove an unnecessary second `fit` in `vectorizeIt` (same semantics, less inconsistency risk) and keep the submission formatting unchanged (quoted text).'

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
    max_df = 0.1
    min_df = 14
    max_feat = 2000
    c = 0.5




## === cell 2
def cleanText(text):
    """
    Keep punctuation/quotes in the original tweet; only normalize case and remove obvious noise.
    This improves exact-match overlap when reconstructing selected_text (word-level Jaccard).
    """
    import re

    text = str(text).strip().lower()
    text = re.sub(r"http[s]?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\s+", " ", text).strip()
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

    wc = vectorizer.fit_transform(new_df)

    inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]

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
    else:
        plt.plot(x, y, "bo-")
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.legend()
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

    """
    https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB
    alpha:  float, default=1.0
            Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    """
    clf = MultinomialNB(alpha=alpha)
    clf.fit(xTrain, yTrain)
    accTrain = clf.score(xTrain, yTrain)
    accTest = clf.score(xTest, yTest)
    return clf, accTrain, accTest




## === cell 9
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
        plotIt(
            max_df,
            accTrain,
            "NB_max_df_v_acc",
            "MultinomialNB: max_df vs. Accuracy",
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
                clf, accTrainAll[i, j], accValidAll[i, j] = multiNB(
                    alpha, xTr, xVld, yTr, yVld
                )
        accTrain = accTrainAll.mean(axis=0)
        accValid = accValidAll.mean(axis=0)
        plotIt(
            min_df,
            accTrain,
            "NB_min_df_v_acc",
            "MultinomialNB: min_df vs. Accuracy",
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
                clf, accTrainAll[i, j], accValidAll[i, j] = multiNB(
                    alpha, xTr, xVld, yTr, yVld
                )
        accTrain = accTrainAll.mean(axis=0)
        accValid = accValidAll.mean(axis=0)
        plotIt(
            max_feat,
            accTrain,
            "NB_feat_v_acc",
            "MultinomialNB: max_feat vs. Accuracy",
            "max_feat",
            "Accuracy",
            "Train",
            accValid,
            "Valid",
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
        plotIt(
            alpha,
            accTrain,
            "NB_alpha_v_acc",
            "MultinomialNB: alpha vs. Accuracy",
            "alpha",
            "Accuracy",
            "Train",
            accValid,
            "Valid",
        )
        alpha = alpha[np.argmax(accValid)]
        print("alpha: " + "{:.2f}".format(alpha))




## === cell 10
def _best_contiguous_span(tokens, keep_mask):
    """
    Select a contiguous span (ground-truth selected_text is almost always contiguous).
    """
    best_len = 0
    best_i = 0
    best_j = -1
    cur_i = None
    cur_len = 0
    for k, keep in enumerate(keep_mask):
        if keep:
            if cur_i is None:
                cur_i = k
                cur_len = 1
            else:
                cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_i = cur_i
                best_j = k
        else:
            cur_i = None
            cur_len = 0
    if best_len <= 0:
        return ""
    return " ".join(tokens[best_i : best_j + 1])




## === cell 11
def buildSelTextNB(df, test, idxList, pred, bow, featLogProb, sentList):
    import pandas as pd
    import numpy as np

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
                retDF.loc[i, "selected_text"] = '""'
            else:
                retDF.loc[i, "selected_text"] = '"' + aword + '"'
            if test:
                retDF.loc[i, "sentiment"] = "neutral"
        elif pred[j] == 0:
            retDF.loc[i, "selected_text"] = '"' + str(df["text"].iloc[i]) + '"'
            if test:
                retDF.loc[i, "sentiment"] = "neutral"
            j = j + 1
        else:
            idxA = int(pred[j])
            tokens = str(df["text"].iloc[i]).split()
            keep = []
            for word in tokens:
                w = cleanText(word)
                if w in bow:
                    bi = bow.index(w)
                    p_word = featLogProb[idxA, bi]
                    p_neg = featLogProb[1, bi]
                    p_pos = featLogProb[2, bi]
                    if idxA == 1:
                        keep.append(p_word > p_pos)
                    else:
                        keep.append(p_word > p_neg)
                else:
                    keep.append(False)

            outstring = _best_contiguous_span(tokens, keep)

            if len(outstring.strip()) == 0:
                outstring = str(df["text"].iloc[i])

            retDF.loc[i, "selected_text"] = '"' + outstring + '"'
            if test:
                retDF.loc[i, "sentiment"] = sentList[idxA]
            j = j + 1
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
def tuneSVM(all, train, test, trainIdx, max_feat, max_df, min_df, c):
    import numpy as np
    from sklearn.model_selection import train_test_split

    print("Tune SVM...")
    cvSz = 5  # iterations of cross validation
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
        c_arr = np.arange(start=0.5, stop=5.0, step=0.5)
        accTrainAll = np.zeros(shape=(cvSz, len(c_arr)))
        accValidAll = np.zeros(shape=(cvSz, len(c_arr)))
        for j in range(len(c_arr)):
            for i in range(cvSz):
                tIdx = trainIdx
                index, bow, xTrain, xTest = vectorizeIt(
                    all, "text", trainIdx, maxFeat=max_feat, maxDF=max_df, minDF=min_df
                )
                tIdx, yTrain, yTest, idxTr, idxTe = yPrep(index, tIdx, train, test)
                xTr, xVld, yTr, yVld = train_test_split(
                    xTrain, yTrain, test_size=0.33, random_state=i, shuffle=True
                )
                clf, accTrainAll[i, j], accValidAll[i, j] = svmClass(
                    xTr, xVld, yTr, yVld, c_arr[j]
                )
        accTrain = accTrainAll.mean(axis=0)
        accValid = accValidAll.mean(axis=0)
        plotIt(
            c_arr,
            accTrain,
            "SVM_c_v_acc",
            "SVM: c vs. Accuracy",
            "c",
            "Accuracy",
            "Train",
            accValid,
            "Valid",
        )
        c = c_arr[np.argmax(accValid)]
        print("c: " + "{:.2f}".format(c))




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
                retDF.loc[i, "selected_text"] = '""'
            else:
                retDF.loc[i, "selected_text"] = '"' + aword + '"'
            if test:
                retDF.loc[i, "sentiment"] = "neutral"
        elif pred[j] == 0:
            retDF.loc[i, "selected_text"] = '"' + str(df["text"].iloc[i]) + '"'
            if test:
                retDF.loc[i, "sentiment"] = "neutral"
            j = j + 1
        else:
            idxA = int(pred[j])
            tokens = str(df["text"].iloc[i]).split()
            keep = []

            if idxA == 1:
                opp = 2
            else:
                opp = 1

            w_diff = w[idxA, :] - w[opp, :]

            for word in tokens:
                wc = cleanText(word)
                if wc in bow:
                    iBow = bow.index(wc)
                    contrib = w_diff[iBow]
                    keep.append(contrib > 0.0)
                else:
                    keep.append(False)

            outstring = _best_contiguous_span(tokens, keep)

            if len(outstring.strip()) == 0:
                outstring = str(df["text"].iloc[i])

            retDF.loc[i, "selected_text"] = '"' + outstring + '"'
            if test:
                retDF.loc[i, "sentiment"] = sentList[idxA]
            j = j + 1
    return retDF




## === cell 15
if method:
    print("Multinomial Naive Bayes Classifier")
    if tune:
        tuneNB(df_all, df_train, df_test, trainIdx, max_feat, max_df, min_df, alpha)
    else:
        index, bow, xTrain, xTest = vectorizeIt(
            df_all, "text", trainIdx, maxFeat=max_feat, maxDF=max_df, minDF=min_df
        )
        trainIdx2, yTrain, yTest, idxTr, idxTe = yPrep(
            index, trainIdx, df_train, df_test
        )
        clf, accTrain, accTest = multiNB(alpha, xTrain, xTest, yTrain, yTest)
        predTrain = clf.predict(xTrain)
        predTest = clf.predict(xTest)
        print("Training Accuracy: " + "{:.4f}".format(accTrain))
        print("Testing Accuracy: " + "{:.4f}".format(accTest))
        testRet_df = buildSelTextNB(
            df_test, test, idxTe, predTest, bow, clf.feature_log_prob_, sentList
        )
        if test:
            trainRet_df = buildSelTextNB(
                df_train, test, idxTr, predTrain, bow, clf.feature_log_prob_, sentList
            )

        testRet_df[["textID", "selected_text"]].to_csv(outFile, index=False)

        if test:
            trainRet_df.to_csv(outTrain, index=False)
            sz = len(df_train)
            scoreArr = np.zeros(shape=(sz))
            for i in range(len(df_train)):
                scoreArr[i] = jaccard(
                    df_train["selected_text"].iloc[i],
                    trainRet_df["selected_text"].iloc[i],
                )
            print("Jaccard Score: " + "{:.4f}".format(np.mean(scoreArr)))
else:
    print("Support Vector Machine Classifier")
    if tune:
        tuneSVM(df_all, df_train, df_test, trainIdx, max_feat, max_df, min_df, c)
    else:
        index, bow, xTrain, xTest = vectorizeIt(
            df_all, "text", trainIdx, maxFeat=max_feat, maxDF=max_df, minDF=min_df
        )
        trainIdx2, yTrain, yTest, idxTr, idxTe = yPrep(
            index, trainIdx, df_train, df_test
        )
        clf, accTrain, accTest = svmClass(xTrain, xTest, yTrain, yTest, c)
        predTrain = clf.predict(xTrain)
        predTest = clf.predict(xTest)
        print("Training Accuracy: " + "{:.4f}".format(accTrain))
        print("Testing Accuracy: " + "{:.4f}".format(accTest))
        coef = clf.coef_.toarray()
        w_norm = np.linalg.norm(coef)
        y = clf.decision_function(xTest)
        dist = y / w_norm  # kept for parity with original logic (even if unused)
        testRet_df = buildSelTextSVM(
            df_test, test, idxTe, predTest, bow, sentList, y, coef
        )
        if test:
            y = clf.decision_function(xTrain)
            dist = y / w_norm
            trainRet_df = buildSelTextSVM(
                df_train, test, idxTr, predTrain, bow, sentList, y, coef
            )

        testRet_df[["textID", "selected_text"]].to_csv(outFile, index=False)

        if test:
            trainRet_df.to_csv(outTrain, index=False)
            sz = len(df_train)
            scoreArr = np.zeros(shape=(sz))
            for i in range(len(df_train)):
                scoreArr[i] = jaccard(
                    df_train["selected_text"].iloc[i],
                    trainRet_df["selected_text"].iloc[i],
                )
            print("Jaccard Score: " + "{:.4f}".format(np.mean(scoreArr)))
