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
    Keep punctuation (important for exact token matching in the selected span),
    but normalize case and remove urls/digits/noise for more stable vocabulary.
    """
    import re

    text = str(text).strip().lower()
    text = re.sub(r"http[s]?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"\d+", "", text)
    return text




## === cell 3
def vectorizeIt(df, textCol, trainIdx, maxFeat=2000, maxDF=1.0, minDF=0.0):
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    vectorizer = CountVectorizer(
        stop_words=None,
        preprocessor=cleanText,
        max_features=maxFeat,
        max_df=maxDF,
        min_df=minDF,
        sublinear_tf=True,
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
        stop_words=None,
        preprocessor=cleanText,
        max_features=maxFeat,
        max_df=maxDF,
        min_df=minDF,
        sublinear_tf=True,
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

    clf = LinearSVC(C=c, class_weight="balanced")
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

        row = y_dec[j]
        if np.ndim(row) == 0:
            thr = 0.0
        else:
            best = float(np.max(row))
            second = float(np.partition(row, -2)[-2]) if row.shape[0] >= 2 else 0.0
            margin = best - second
            thr = 0.005 + 0.03 * margin  # keep your current tuned thresholding

        out_tokens = []
        token_scores = []  # (score, token) for fallback
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
            score = min(contrib_a_vs_0, contrib_a_vs_o)
            token_scores.append((score, word))
            if (contrib_a_vs_0 > thr) and (contrib_a_vs_o > thr):
                out_tokens.append(word)

        if len(out_tokens) == 0:
            outstring = str(df["text"].iloc[i])
        else:
            outstring = " ".join(out_tokens)

        if outstring.strip() == str(df["text"].iloc[i]).strip():
            if len(token_scores) > 0:
                best_word = max(token_scores, key=lambda x: x[0])[1]
                if str(best_word).strip():
                    outstring = str(best_word)

        retDF["selected_text"].iloc[i] = '"' + outstring + '"'
        if test:
            retDF["sentiment"].iloc[i] = sentList[idxA]
        j += 1

    return retDF




## === cell 16
from sklearn.model_selection import train_test_split

idxTrNull, idxTeNull, bow, xTrain, xTest = vectorizeTrainTest(
    df_train, df_test, "text", maxFeat=max_feat, maxDF=max_df, minDF=min_df
)

yTrain, yTest = yPrepTrainTest(idxTrNull, idxTeNull, df_train, df_test)

if method:
    clf, accTrain, accValid = multiNB(alpha, xTrain, xTrain, yTrain, yTrain)
    pred_test = clf.predict(xTest)
    out_test = buildSelTextNB(
        df_test, False, idxTeNull, pred_test, bow, clf.feature_log_prob_, sentList
    )
else:
    clf, accTrain, _ = svmClass(xTrain, xTrain, yTrain, yTrain, c=c)

    y_dec_test = clf.decision_function(xTest)

    w = clf.coef_
    bow2idx = {tok: i for i, tok in enumerate(bow)}

    pred_test = clf.predict(xTest)
    out_test = buildSelTextSVM(
        df_test, False, idxTeNull, pred_test, bow, sentList, y_dec_test, w, bow2idx
    )

sub = pd.read_csv(sample, dtype=str)[["textID"]].merge(
    out_test[["textID", "selected_text"]], on="textID", how="left"
)

if sub["selected_text"].isnull().any():
    fallback = df_test.set_index("textID")["text"].to_dict()
    sub["selected_text"] = sub.apply(
        lambda r: (
            r["selected_text"]
            if pd.notnull(r["selected_text"])
            else '"' + str(fallback.get(r["textID"], "")) + '"'
        ),
        axis=1,
    )

sub.to_csv(outFile, index=False)
print("Wrote:", outFile, "rows:", len(sub), "cols:", list(sub.columns))
print(sub.head())

## --- ERROR in cell 16, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4119810370.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      4[0m [0;34m[0m[0m
[1;32m      5[0m [0;31m# Vectorize using train-only fit (as already intended by your pipeline)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 6[0;31m idxTrNull, idxTeNull, bow, xTrain, xTest = vectorizeTrainTest(
[0m[1;32m      7[0m     [0mdf_train[0m[0;34m,[0m [0mdf_test[0m[0;34m,[0m [0;34m"text"[0m[0;34m,[0m [0mmaxFeat[0m[0;34m=[0m[0mmax_feat[0m[0;34m,[0m [0mmaxDF[0m[0;34m=[0m[0mmax_df[0m[0;34m,[0m [0mminDF[0m[0;34m=[0m[0mmin_df[0m[0;34m[0m[0;34m[0m[0m
[1;32m      8[0m )

[0;32m/tmp/ipykernel_11/3013170184.py[0m in [0;36mvectorizeTrainTest[0;34m(train_df, test_df, textCol, maxFeat, maxDF, minDF)[0m
[1;32m      5[0m     [0;32mimport[0m [0mnumpy[0m [0;32mas[0m [0mnp[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;34m[0m[0m
[0;32m----> 7[0;31m     vectorizer = CountVectorizer(
[0m[1;32m      8[0m         [0mstop_words[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m         [0mpreprocessor[0m[0;34m=[0m[0mcleanText[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: CountVectorizer.__init__() got an unexpected keyword argument 'sublinear_tf'
