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

0.39374

# 6. Current score

0.2472

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.07084) has done: 'I fix the runtime error by replacing deprecated `np.int` with the builtin `int` (NumPy 1.26+ compatibility). I also correct `trainIdx` to match the actual train size so the train/test split inside `vectorizeIt()` is aligned with the concatenated dataframe (this is required for the pipeline to run end-to-end and produce a valid submission). Additionally, I fix a few small logic/robustness issues that can crash execution (e.g., `plotIt` checking `len(y2)` when `y2=None`, and incorrect test-index handling in `yPrep`). These changes keep the core modeling logic intact (CountVectorizer + SVM/NB + heuristic word selection) while ensuring a proper `submission.csv` is written.'
- What this solution (achieved 0.1073) has done: 'Your score gap to the target is large (0.07084 vs 0.39374, higher-is-better), so we need a modest but meaningful improvement while keeping the same overall pipeline (CountVectorizer + SVM/NB + heuristic word selection). The biggest issue is that the current SVM path is using `decision_function` outputs incorrectly (multiclass OVR returns shape (n_samples, n_classes)), and the word-selection heuristic ends up essentially random/empty, collapsing Jaccard. I keep the same SVM model and the same “select words based on class signal” idea, but fix the multiclass handling by using the proper per-class weights and per-sample class scores; this preserves core logic but makes it actually correspond to the intended heuristic. I also add a minimal, metric-aligned fallback: if the heuristic selects nothing, return the full tweet (and for neutral sentiment always return the full tweet), which is standard for this competition and should move the score toward the target band without changing the model.'
- What this solution (achieved 0.25162) has done: 'Your current gap to the target is large (0.1073 vs 0.39374), so we need a meaningful but still “same-core-logic” improvement. The biggest score drag left is that the SVM is trained on the full tweet sentiment labels, but the selection heuristic doesn’t know which tokens are sentiment-bearing; we can keep the same CountVectorizer+linear SVM approach and improve the heuristic by using the model’s per-token contribution (coef[class, token] * count) to rank and select words. I also make the selection metric-aligned by adding a tiny post-processing step to trim leading/trailing punctuation around the selected span (without changing tokenization or the model). Finally, I keep the existing safe fallbacks (neutral/full tweet; empty->full tweet) to avoid catastrophic Jaccard failures.'
- What this solution (achieved 0.25114) has done: 'Your current score (0.25162) is well below the target (0.39374), so we should improve performance with minimal, metric-aligned adjustments while keeping your CountVectorizer + linear SVM + “pick words by per-token contribution” core logic. The biggest remaining drag is that you drop all punctuation in `cleanText`, so the model’s vocabulary can’t match the raw tweet tokens that include punctuation, causing many sentiment-bearing tokens to be missed during selection; I keep the same pipeline but make `cleanText` preserve punctuation as separators (not delete it) so token matching works. I also remove the expensive `bow.index(wc)` lookup by building a `token->vocab_id` map once, which doesn’t change semantics but makes the selection step reliable within the time limit. Finally, I keep your existing neutral/full-tweet and empty->full-tweet fallbacks and apply the same trimming you already do, ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.2472) has done: 'We’re still far below the target (0.25114 vs 0.39374, higher-is-better), so we need a modest but meaningful improvement while keeping your exact pipeline (CountVectorizer + linear SVM + per-token contribution heuristic). The main remaining issue is that you currently drop any token whose contribution isn’t positive, which often yields either empty selections (forcing full tweet) or overly long bag-of-words selections; replacing that with a minimal “best contiguous span by token score” keeps the same per-token contribution logic but aligns much better with the word-level Jaccard metric. I also ensure the SVM coefficient matrix is aligned to sentiment class indices (neutral/negative/positive) using `clf.classes_` so the right class weights are used. All other training, vectorization, and fallbacks (neutral/full tweet, empty->full tweet) remain unchanged, and the script still writes a valid `submission.csv`.'

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

trainIdx = len(df_train)



## === cell 1
tune = False
test = False
method = False  # True selects MultinomialNB, False selects SVM

if method:
    max_df = 1.0
    min_df = 9  # maybe try higher?
    max_feat = 2000
    alpha = 6.0
else:
    max_df = 1.0  # 0.1
    min_df = 1  # 14
    max_feat = 25000  # 2000
    c = 0.0003  # 0.7




## === cell 2
def cleanText(text):
    """
    Change rationale (score-improvement, same core logic):
    - Previously, we removed ALL punctuation. That makes the vectorizer vocabulary (trained on cleaned text)
      mismatch the raw tweet tokens we later iterate over (which include punctuation like "good!", ":( ", etc.).
      This causes many sentiment-bearing words to be missed during selection, hurting Jaccard.
    - Keep the same CountVectorizer(preprocessor=cleanText) approach, but preserve punctuation as token separators
      by translating punctuation to spaces instead of deleting it. This improves alignment without changing the
      overall model/training logic.
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

    trans = str.maketrans({ch: " " for ch in string.punctuation})
    text = text.translate(trans)

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
    else:
        plt.plot(x, y, "bo-")
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    if y2 is not None:
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
                outstring = df["text"].iloc[i]
            outstring = '"' + outstring + '"'
            retDF["selected_text"].iloc[i] = outstring
            if test:
                retDF["sentiment"].iloc[i] = sentList[idxA]
            j = j + 1
    return retDF




## === cell 11
def svmClass(xTrain, xTest, yTrain, yTest, c=1.0):
    from sklearn import svm

    """
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """
    clf = svm.SVC(C=c, kernel="linear")
    clf.fit(xTrain, yTrain)
    accTrain = clf.score(xTrain, yTrain)
    accTest = clf.score(xTest, yTest)
    return clf, accTrain, accTest




## === cell 12
def tuneSVM(xTrain, xTest, yTrain, yTest):
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




## === cell 13
def _trim_selected_text_like_metric(s):
    import re

    s = str(s)
    if len(s) == 0:
        return s
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    s = re.sub(r"^[\s\.,;:!\?\-_\(\)\[\]\{\}]+", "", s)
    s = re.sub(r"[\s\.,;:!\?\-_\(\)\[\]\{\}]+$", "", s)
    return s.strip()




## === cell 14
def _best_contiguous_span(words, scores):
    """
    Change rationale (score-improvement, same core logic):
    - We still compute per-token contributions from the same linear SVM weights.
    - Instead of selecting an unordered set of positive tokens (which can be too long or disjoint),
      choose the best *contiguous* span by maximum subarray (Kadane) on token scores.
      This better matches the competition's "selected_text" being a substring/span and improves Jaccard.
    """
    if not words:
        return ""
    best_sum = None
    best_l = 0
    best_r = 0
    cur_sum = 0.0
    cur_l = 0
    for r, sc in enumerate(scores):
        if cur_sum <= 0.0:
            cur_sum = float(sc)
            cur_l = r
        else:
            cur_sum += float(sc)
        if best_sum is None or cur_sum > best_sum:
            best_sum = cur_sum
            best_l = cur_l
            best_r = r
    return " ".join(words[best_l : best_r + 1]).strip()




## === cell 15
def buildSelTextSVM(df, test, idxList, pred, bow, sentList, x_mat, w):
    import pandas as pd
    import numpy as np

    """
    Change rationale (score-improvement, same core logic):
    - Preserve linear SVM + per-token contribution selection.
    - Improve selection to output a contiguous span with highest summed token contribution (Kadane),
      which is metric-aligned for word-level Jaccard on substrings.
    - Keep O(1) token->vocab-id lookup via dict for reliability/time.
    """
    print("Build SVM output...")
    if test:
        retDF = pd.DataFrame(columns=["textID", "selected_text", "sentiment"])
        retDF["textID"] = df["textID"]
    else:
        retDF = pd.DataFrame(columns=["textID", "selected_text"])
        retDF["textID"] = df["textID"]

    w = np.asarray(w)
    if w.ndim == 1:
        w = w.reshape(1, -1)

    bow_to_id = {tok: i for i, tok in enumerate(bow)}

    j = 0
    for i in range(len(retDF)):
        if i in idxList:
            aword = str(df["text"].iloc[i])
            if aword == "nan":
                retDF.loc[i, "selected_text"] = ""
            else:
                retDF.loc[i, "selected_text"] = aword
            if test:
                retDF.loc[i, "sentiment"] = "neutral"
            continue

        idxA = int(pred[j])

        if idxA == 0:
            retDF.loc[i, "selected_text"] = str(df["text"].iloc[i])
            if test:
                retDF.loc[i, "sentiment"] = "neutral"
            j += 1
            continue

        class_coef = w[idxA, :]
        neutral_coef = w[0, :] if w.shape[0] > 1 else 0.0

        raw_words = str(df["text"].iloc[i]).split()
        token_scores = []
        scored_words = []

        for word in raw_words:
            wc = cleanText(word)
            v_id = bow_to_id.get(wc, None)
            if v_id is None:
                continue
            sc = float(
                class_coef[v_id]
                - (neutral_coef[v_id] if np.ndim(neutral_coef) else 0.0)
            )
            scored_words.append(word)
            token_scores.append(sc)

        outstring = ""
        if scored_words:
            outstring = _best_contiguous_span(scored_words, token_scores)

            if sum(token_scores) <= 0.0 and max(token_scores) <= 0.0:
                outstring = ""

        if len(outstring.strip()) == 0:
            outstring = str(df["text"].iloc[i])

        outstring = _trim_selected_text_like_metric(outstring)
        if len(outstring) == 0:
            outstring = str(df["text"].iloc[i])

        retDF.loc[i, "selected_text"] = outstring
        if test:
            retDF.loc[i, "sentiment"] = (
                sentList[idxA] if idxA < len(sentList) else sentList[-1]
            )
        j += 1

    return retDF




## === cell 16
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

        coef_raw = clf.coef_.toarray()
        coef = np.zeros((len(sentList), coef_raw.shape[1]), dtype=coef_raw.dtype)
        for k, cls in enumerate(clf.classes_):
            if int(cls) < coef.shape[0]:
                coef[int(cls), :] = coef_raw[k, :]

        testRet_df = buildSelTextSVM(
            df_test, test, idxTe, predTest, bow, sentList, xTest, coef
        )

        if test:
            trainRet_df = buildSelTextSVM(
                df_train, test, idxTr, predTrain, bow, sentList, xTrain, coef
            )

        sub = testRet_df[["textID", "selected_text"]].copy()
        sub.to_csv(outFile, index=False)
        print(f"Wrote submission to: {outFile} with shape {sub.shape}")

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
