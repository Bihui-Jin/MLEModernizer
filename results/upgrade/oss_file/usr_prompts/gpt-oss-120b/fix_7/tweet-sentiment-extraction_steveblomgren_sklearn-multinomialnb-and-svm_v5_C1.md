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

0.16798

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.19691) has done: 'I replace the deprecated `np.int` with the built‑in `int` type and correctly set `trainIdx` to the actual number of training rows (instead of a hard‑coded value) so the train‑test split works. These fixes eliminate the runtime error and ensure the model trains on the proper data, enabling a valid submission file and moving the score toward the target.'
- What this solution (achieved 0.1115) has done: 'I increase the vocabulary size and include bi‑grams in the CountVectorizer, which usually boosts text‑classification performance and therefore should raise the Jaccard score toward the target. The changes are limited to the hyper‑parameter definitions and the vectorizer construction, preserving all other logic.'
- What this solution (achieved 0.17744) has done: 'I switch the pipeline to use the Multinomial Naive Bayes classifier (which performed better in earlier runs) and adjust a few lightweight hyper‑parameters (reduce max features, tighten max_df, and lower α) to improve the Jaccard score toward the target while keeping the core logic unchanged.'
- What this solution (achieved 0.11192) has done: 'The changes raise the feature set size and loosen the document‑frequency cut‑off, then switch the classifier from MultinomialNB to the Linear SVM (with a slightly larger C). This keeps the overall pipeline unchanged while improving classification accuracy, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.11197) has done: 'I slightly adjust the SVM hyper‑parameters to improve classification (increase C to 10.0, tighten max_df to 0.80) and add balanced class weighting, which should raise the Jaccard score toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.16798) has done: 'Implemented minimal hyper‑parameter tweaks to move the validation Jaccard score closer to the target.  
- Switched the classifier back to Multinomial Naive Bayes (`method=True`) since it previously yielded higher scores.  
- Adjusted `max_df` to `1.0` (no document‑frequency cut‑off) and set `alpha` to `1.0`, values that have shown better performance for this task.  
These small changes keep the overall pipeline intact while improving sentiment prediction and consequently the Jaccard metric.'

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
trainIdx = df_train.shape[0]
max_feat = 50000  # larger vocabulary
max_df = 1.0  # use full document frequency range (improved NB performance)
min_df = 1
alpha = 1.0  # NB smoothing parameter tuned for better Jaccard
c = 10.0  # regularization strength for LinearSVC (kept for completeness)
tune = False
test = False
method = True  # switch to Multinomial Naive Bayes classifier (better baseline)




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
    import pandas as pd
    import numpy as np

    vectorizer = CountVectorizer(
        stop_words="english",
        preprocessor=cleanText,
        max_features=maxFeat,
        max_df=maxDF,
        min_df=minDF,
        ngram_range=(1, 2),
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
    import pandas as pd

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

    if y2 is not None and len(y2) > 0:
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
    idxTe = [i - trainIdx for i in index if i > trainIdx]
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
        else:  # Else add selected_text back in based on probability of feature
            outstring = ""
            idxA = pred[j]
            words = df["text"].iloc[i].split()
            for word in words:
                wordCheck = cleanText(word)
                if wordCheck in bow:
                    probWord = (featLogProb)[idxA, bow.index(wordCheck)]
                    probNeu = (featLogProb)[0, bow.index(wordCheck)]
                    probNeg = (featLogProb)[1, bow.index(wordCheck)]
                    probPos = (featLogProb)[2, bow.index(wordCheck)]
                    if ((idxA == 1) and (probWord > probPos)) or (
                        (idxA == 2) and (probWord > probNeg)
                    ):
                        if len(outstring) == 0:
                            outstring = word
                        else:
                            outstring = outstring + " " + word
            outstring = '"' + outstring + '"'
            retDF["selected_text"].iloc[i] = outstring
            if test:
                retDF["sentiment"].iloc[i] = sentList[idxA]
            j = j + 1  # advance the prediction index
    return retDF




## === cell 11
def svmClass(xTrain, xTest, yTrain, yTest, c=1.0):
    from sklearn.svm import LinearSVC

    """
    LinearSVC works with sparse matrices and provides a linear kernel.
    Adding balanced class weighting helps with any sentiment imbalance.
    """
    clf = LinearSVC(C=c, max_iter=10000, class_weight="balanced")
    clf.fit(xTrain, yTrain)
    accTrain = clf.score(xTrain, yTrain)
    accTest = clf.score(xTest, yTest)
    return clf, accTrain, accTest




## === cell 12
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




## === cell 13
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
            outstring = ""
            idxA = pred[j]
            words = df["text"].iloc[i].split()
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
                            d[k] = yVal / abs(w[k, iBow])
                    if (
                        (
                            ((idxA == 1) and (dWord > d[2]))
                            or ((idxA == 2) and (dWord > d[1]))
                        )
                        and (dWord > 0.0)
                        and (dWord > d[0])
                    ):
                        if len(outstring) == 0:
                            outstring = word
                        else:
                            outstring = outstring + " " + word
            outstring = '"' + outstring + '"'
            retDF["selected_text"].iloc[i] = outstring
            if test:
                retDF["sentiment"].iloc[i] = sentList[idxA]
            j = j + 1  # advance the prediction index
    return retDF




## === cell 14
if method:
    print("Multinomial Naive Bayes Classifier")
    if tune:
        tuneNB(df_all, df_train, df_test, trainIdx, max_feat, max_df, min_df, alpha)
    else:
        index, bow, xTrain, xTest = vectorizeIt(
            df_all, "text", trainIdx, maxFeat=max_feat, maxDF=max_df, minDF=min_df
        )
        trainIdx, yTrain, yTest, idxTr, idxTe = yPrep(
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
        testRet_df.to_csv(outFile, index=False)
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
    index, bow, xTrain, xTest = vectorizeIt(
        df_all, "text", trainIdx, maxFeat=max_feat, maxDF=max_df, minDF=min_df
    )
    trainIdx, yTrain, yTest, idxTr, idxTe = yPrep(index, trainIdx, df_train, df_test)
    if tune:
        tuneSVM(xTrain, xTest, yTrain, yTest)
    else:
        clf, accTrain, accTest = svmClass(xTrain, xTest, yTrain, yTest, c)
        predTrain = clf.predict(xTrain)
        predTest = clf.predict(xTest)
        print("Training Accuracy: " + "{:.4f}".format(accTrain))
        print("Testing Accuracy: " + "{:.4f}".format(accTest))
        coef = clf.coef_
        if hasattr(coef, "toarray"):
            coef = coef.toarray()
        w_norm = np.linalg.norm(coef)
        y = clf.decision_function(xTest)
        testRet_df = buildSelTextSVM(
            df_test, test, idxTe, predTest, bow, sentList, y, coef
        )
        if test:
            y = clf.decision_function(xTrain)
            dist = y / w_norm
            trainRet_df = buildSelTextSVM(
                df_train, test, idxTr, predTrain, bow, sentList, y, coef
            )
        testRet_df.to_csv(outFile, index=False)
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
