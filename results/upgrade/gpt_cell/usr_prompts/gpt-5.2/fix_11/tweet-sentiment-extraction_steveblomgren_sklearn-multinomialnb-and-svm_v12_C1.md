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

0.52952

# 6. Current score

0.35471

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.34544) has done: 'Diagnosis: The crash happens in cell 14 when fitting the SVM because `xTrain` and `yTrain` have different lengths (`27480` vs `24731`). This comes from an incorrect `trainIdx` split point: it is set to `27481`, but the actual training set has 24732 rows, so the vectorized matrix is split at the wrong boundary while `yPrep()` uses `df_train`/`df_test` sizes. This mismatch propagates into `svmClass().fit()` and triggers `check_consistent_length`.  
Patch summary: In cell 14 only, recompute `trainIdx` from `len(df_train)` right before vectorization/training so the split point always matches the real training size and keeps downstream logic unchanged. Also keep the existing `np.int` compatibility shim.  
Updated cells: Only cell 14 is modified as below.  
Compatibility notes for cell k+1: No interface/variable names change; `trainIdx`, `index`, `bow`, `xTrain/xTest`, and downstream outputs keep the same semantics, but now have consistent sample counts so training proceeds.  
Assumptions: `df_train` is already loaded in earlier cells and corresponds to the intended training dataset; the correct split boundary is exactly `len(df_train)`.'
- What this solution (achieved 0.15891) has done: 'Your current pipeline is learning a *sentiment classifier* and then heuristically picking words, but the Kaggle metric is word-level Jaccard on the extracted span; the biggest low-risk gain without changing your model/training is to make your post-processing closer to “extract exact words from the tweet.” I keep the SVM/BoW training exactly as-is, and only adjust the `buildSelTextSVM` reconstruction so it selects words that are most supportive *for the predicted class* (by using the class-specific SVM weights) rather than selecting every word with nonnegative value in the sparse vector. I also ensure the output CSV matches the required format by dropping the extra `sentiment` column when `test=False` is not used, and by quoting the `selected_text` like the competition expects. These are minimal, metric-aligned changes that should improve Jaccard toward your target without altering the core learning logic.'
- What this solution (achieved 0.20932) has done: 'Your current score is far below the target, so we should improve extraction quality without changing the core SVM/BoW training. The biggest issue is that `buildSelTextSVM` uses `bow.index()` inside nested loops (very slow) and uses only a “positive weight” rule; we keep the same idea but make it more faithful to the linear SVM decision by scoring words via class-vs-neutral weight differences and selecting the top-contributing contiguous span. This remains pure post-processing (no model/loop/feature/loss changes) but better matches the word-level Jaccard metric. We also avoid converting sparse matrices to dense arrays (unnecessary and can hurt runtime/memory) while keeping outputs identical in meaning and ensuring the submission CSV format stays correct.'
- What this solution (achieved 0.21197) has done: 'Your current score (0.20932) is far below the target (0.52952), so we should improve the extraction post-processing while keeping the same SVM+BoW training. The biggest low-risk gain for Jaccard is to stop using `cleanText()` for token alignment during span selection (it removes punctuation/apostrophes, which the metric expects you to preserve) and instead map model vocab to *raw tweet tokens* via a lightweight normalization that matches the CountVectorizer’s default tokenization more closely. We also special-case `neutral` to return the full tweet (a common strong baseline for this competition) while keeping your classifier intact. Finally, we keep the same CSV writing but add a safety step to ensure non-null strings and correct quoting.'
- What this solution (achieved 0.20932) has done: 'We keep your SVM + CountVectorizer training exactly the same and only adjust the SVM span reconstruction to better match the Jaccard metric: (1) apply the same `cleanText()` normalization to each raw word (so token matching aligns with the vectorizer), (2) use class-vs-rest weights (for binary coef_ case) or class-vs-neutral (for multiclass) consistently, and (3) extract a high-scoring contiguous span but fall back safely to the full tweet for neutral/low-signal cases. This is a minimal post-processing change aimed at improving extracted-text overlap without changing the model, features, or training loop. We also keep submission formatting identical (quoted strings) and ensure it always writes a valid `submission.csv`.'
- What this solution (achieved 0.20932) has done: 'Your current score is far below the target, so the smallest meaningful improvement is to adjust only the extraction post-processing to better match the word-level Jaccard metric while keeping the same SVM+BoW training intact. The biggest issue is that the extractor currently selects spans based on token-level SVM weights but has no strong “neutral/low-signal fallback” and can output overly long/poor spans when confidence is weak. I keep your model and vectorizer unchanged, and only (1) force a strong baseline behavior for predicted `neutral` (return full tweet), and (2) add a minimal “margin gate” using the existing `decision_function` so that low-confidence predictions also return the full tweet, improving average overlap. This should move the score upward toward your target without changing training, features, or loss.'
- What this solution (achieved 0.21077) has done: 'Your score gap to the target is large, so we make a minimal metric-aligned change only in the SVM span reconstruction (post-processing) while keeping your SVM/BoW training and parameters intact. The main issue is that span selection currently can output punctuation-stripped/misaligned spans because it matches words via `cleanText(word)` (which deletes apostrophes/punctuation) while the competition scoring expects exact raw substrings. I keep the same per-word scoring idea but switch token alignment to mirror CountVectorizer’s *default* tokenization (`(?u)\b\w\w+\b`) by extracting raw “word tokens” with positions from the original tweet and selecting a contiguous span over those tokens, then slicing the original tweet by character indices to preserve punctuation/spacing exactly. Neutral still return the full tweet (as you already effectively do), and the low-margin fallback remains to avoid over-aggressive extraction. This should move Jaccard upward toward your target without changing the core model/training logic.'
- What this solution (achieved 0.35471) has done: 'Your current score is far below the target, so we make a minimal, metric-aligned improvement only in the SVM post-processing (span reconstruction) without changing the SVM/BoW training, vectorizer settings, or any training loops. The biggest low-risk gain for Jaccard here is to handle the very common “sentiment == neutral” case explicitly by returning the full tweet (this is a known strong baseline for this competition), because neutral spans frequently equal the whole text. Additionally, we extend the token regex to include single-character tokens so outputs can better match short selected spans like “!”/“I”/“a”, while still slicing the original tweet to preserve exact punctuation/spacing. These changes keep your core logic intact and should move the score upward toward the target.'
- What this solution (achieved 0.35471) has done: 'We keep your SVM + CountVectorizer training and parameters exactly the same, and only make a minimal, metric-aligned adjustment to the SVM span reconstruction to better match the Kaggle word-level Jaccard. Specifically, we (1) ensure `neutral` predictions always return the full tweet (already done, kept), (2) add the standard competition rule that for `positive`/`negative` tweets where the raw tweet has only 1 token we return the full tweet (prevents bad empty/partial spans), and (3) slightly tighten the “low-confidence fallback to full tweet” using a sentiment-dependent margin threshold (more conservative extraction tends to improve overlap for this baseline). This should move the score upward from 0.35471 toward your 0.52952 target without changing model architecture, features, or training loops, and still writes a valid `submission.csv`.'

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
trainIdx = 27481

if method:
    max_df = 1.0
    min_df = 9  # maybe try higher?
    max_feat = 2000
    alpha = 6.0
else:
    max_df = 1.0  # 0.1
    min_df = 12  # 14
    max_feat = 3000  # 2000
    c = 0.7  # 0.7




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

    y = np.zeros(shape=(df[textCol].size), dtype=np.int)
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
                retDF.loc[i, "selected_text"] = '"' + '"'  # keep quoted empty
            else:
                retDF.loc[i, "selected_text"] = '"' + aword + '"'
            if test:
                retDF.loc[i, "sentiment"] = "neutral"
        elif pred[j] == 0:
            retDF.loc[i, "selected_text"] = '"' + df["text"].iloc[i] + '"'
            if test:
                retDF.loc[i, "sentiment"] = "neutral"
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
                        outstring = (
                            word if len(outstring) == 0 else (outstring + " " + word)
                        )
            retDF.loc[i, "selected_text"] = '"' + outstring + '"'
            if test:
                retDF.loc[i, "sentiment"] = sentList[idxA]
            j = j + 1
    return retDF




## === cell 11
def svmClass(xTrain, xTest, yTrain, yTest, c=1.0):
    from sklearn import svm

    clf = svm.SVC(C=c, kernel="linear")
    clf.fit(xTrain, yTrain)
    accTrain = clf.score(xTrain, yTrain)
    accTest = clf.score(xTest, yTest)
    return clf, accTrain, accTest




## === cell 12
def tuneSVM(xTrain, xTest, yTrain, yTest):
    import numpy as np
    from math import floor
    from sklearn.model_selection import train_test_split

    print("Tune SVM...")
    vldSz = 0.129
    cvSz = floor(1.0 / vldSz)
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
                    xTrain, yTrain, test_size=vldSz, random_state=i, shuffle=True
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
                    xTrain, yTrain, test_size=vldSz, random_state=i, shuffle=True
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
                    xTrain, yTrain, test_size=vldSz, random_state=i, shuffle=True
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
        c = np.arange(start=0.1, stop=1.1, step=0.1)
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
                    xTrain, yTrain, test_size=vldSz, random_state=i, shuffle=True
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
def buildSelTextSVM(df, test, idxList, pred, bow, sentList, y, w, b, x):
    import pandas as pd
    import numpy as np
    import re

    print("Build SVM output...")

    bow2idx = {tok: k for k, tok in enumerate(bow)}

    if test:
        retDF = pd.DataFrame(columns=["textID", "selected_text", "sentiment"])
        retDF["textID"] = df["textID"]
    else:
        retDF = pd.DataFrame(columns=["textID", "selected_text"])
        retDF["textID"] = df["textID"]

    def best_span(scores):
        best_sum = -1e18
        best_l = 0
        best_r = 0
        cur_sum = 0.0
        cur_l = 0
        for k in range(len(scores)):
            if cur_sum <= 0.0:
                cur_sum = scores[k]
                cur_l = k
            else:
                cur_sum += scores[k]
            if cur_sum > best_sum:
                best_sum = cur_sum
                best_l = cur_l
                best_r = k
        return best_sum, best_l, best_r

    def get_margin(dec_row):
        if dec_row is None:
            return 0.0
        dr = np.asarray(dec_row)
        if dr.ndim == 0:
            return float(abs(dr))
        if dr.ndim == 1 and dr.size > 1:
            s = np.sort(dr)
            return float(s[-1] - s[-2])
        return float(abs(dr).ravel()[0])

    token_re = re.compile(r"(?u)\b\w+\b")

    j = 0
    for i in range(len(retDF)):
        txt = "" if pd.isna(df["text"].iloc[i]) else str(df["text"].iloc[i])
        raw_sent = (
            ""
            if pd.isna(df["sentiment"].iloc[i])
            else str(df["sentiment"].iloc[i]).strip().lower()
        )

        if i in idxList:
            retDF.loc[i, "selected_text"] = '"' + txt + '"'
            if test:
                retDF.loc[i, "sentiment"] = "neutral"
            continue

        cls = int(pred[j])

        if raw_sent == "neutral":
            retDF.loc[i, "selected_text"] = '"' + txt + '"'
            if test:
                retDF.loc[i, "sentiment"] = "neutral"
            j += 1
            continue

        if len(txt.split()) <= 1:
            retDF.loc[i, "selected_text"] = '"' + txt + '"'
            if test:
                retDF.loc[i, "sentiment"] = (
                    sentList[cls] if cls < len(sentList) else raw_sent
                )
            j += 1
            continue

        if cls == 0:
            retDF.loc[i, "selected_text"] = '"' + txt + '"'
            if test:
                retDF.loc[i, "sentiment"] = "neutral"
            j += 1
            continue

        dec_row = None
        if y is not None:
            try:
                dec_row = y[j]
            except Exception:
                dec_row = None

        margin = get_margin(dec_row)

        margin_thr = 0.20 if raw_sent in ("positive", "negative") else 0.15
        if margin < margin_thr:
            retDF.loc[i, "selected_text"] = '"' + txt + '"'
            if test:
                retDF.loc[i, "sentiment"] = sentList[cls]
            j += 1
            continue

        matches = list(token_re.finditer(txt))
        if len(matches) == 0:
            retDF.loc[i, "selected_text"] = '"' + txt + '"'
            if test:
                retDF.loc[i, "sentiment"] = sentList[cls]
            j += 1
            continue

        W = w
        if W.ndim == 1:
            W = W.reshape(1, -1)

        scores = np.zeros(len(matches), dtype=float)
        for k, m in enumerate(matches):
            raw_tok = m.group(0)
            tok = cleanText(raw_tok)
            idx = bow2idx.get(tok, None)
            if idx is None:
                continue

            if W.shape[0] == 1:
                scores[k] = float(W[0, idx]) if cls == 1 else float(-W[0, idx])
            else:
                scores[k] = float(W[cls, idx] - W[0, idx])

        best_sum, best_l, best_r = best_span(scores)

        if (not np.isfinite(best_sum)) or best_sum <= 0.0:
            outstring = txt
        else:
            start = matches[best_l].start()
            end = matches[best_r].end()
            outstring = txt[start:end].strip()
            if len(outstring) == 0:
                outstring = txt

        retDF.loc[i, "selected_text"] = '"' + outstring + '"'
        if test:
            retDF.loc[i, "sentiment"] = sentList[cls]
        j += 1

    return retDF




## === cell 14
import numpy as np

trainIdx = len(df_train)

if not hasattr(np, "int"):
    np.int = int

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

        testRet_df["selected_text"] = (
            testRet_df["selected_text"].fillna('""').astype(str)
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
    print("max_df:   " + "{:3.5f}".format(max_df))
    print("min_df:   " + "{:8d}".format(min_df))
    print("max_feat: " + "{:7.1f}".format(max_feat))
    print("c:        " + "{:3.5f}".format(c))
    if tune:
        tuneSVM(df_all, df_train, df_test, trainIdx, max_feat, max_df, min_df, c)
    else:
        index, bow, xTrain, xTest = vectorizeIt(
            df_all, "text", trainIdx, maxFeat=max_feat, maxDF=max_df, minDF=min_df
        )
        trainIdx, yTrain, yTest, idxTr, idxTe = yPrep(
            index, trainIdx, df_train, df_test
        )
        clf, accTrain, accTest = svmClass(xTrain, xTest, yTrain, yTest, c)
        predTrain = clf.predict(xTrain)
        predTest = clf.predict(xTest)
        print("Training Accuracy: " + "{:.4f}".format(accTrain))
        print("Testing Accuracy: " + "{:.4f}".format(accTest))

        coef = clf.coef_.toarray()
        b = clf.intercept_

        y = clf.decision_function(xTest)
        testRet_df = buildSelTextSVM(
            df_test, test, idxTe, predTest, bow, sentList, y, coef, b, None
        )
        if test:
            y = clf.decision_function(xTrain)
            trainRet_df = buildSelTextSVM(
                df_train,
                test,
                idxTr,
                predTrain,
                bow,
                sentList,
                y,
                coef,
                b,
                None,
            )

        testRet_df["selected_text"] = (
            testRet_df["selected_text"].fillna('""').astype(str)
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
