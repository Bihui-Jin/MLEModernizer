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

0.29648

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.07369) has done: 'Implemented two critical fixes:
1. Replaced deprecated `np.int` with built‑in `int` in `sentArray` to stop the AttributeError.
2. Made the train‑test split robust by setting `trainIdx` dynamically to the actual number of training rows instead of a hard‑coded value.

These changes allow the script to run end‑to‑end and generate a valid `submission.csv` while keeping the core modeling logic unchanged.'
- What this solution (achieved 0.27825) has done: 'I adjust the vectorizer to keep all words (remove English stop‑word filtering) and raise the SVM regularisation parameter `c` from 0.00001 to 1.0, giving the model stronger discriminative power. I also relax the strict “greater‑than” checks when choosing words for the final snippet, allowing ties to be kept, which typically yields longer, more accurate selected texts and moves the Jaccard score closer to the target.'
- What this solution (achieved 0.279) has done: 'I add a simple fallback in the text‑selection functions so that if no word meets the probability criteria the whole tweet is returned (instead of an empty quoted string). This small change keeps the model untouched but usually yields a higher Jaccard overlap, moving the score toward the target. The modification is applied to both the Naive‑Bayes and SVM output builders.'
- What this solution (achieved 0.33854) has done: 'I fixed the crash caused by an unsupported LinearSVC configuration: the original code used `loss='hinge'` together with `dual=False`, which is invalid. I changed the loss to `'squared_hinge'`, which works with `dual=False` while keeping the rest of the model unchanged. This minimal fix lets the pipeline run end‑to‑end and generate a valid `submission.csv`. No other logic is altered, preserving the original approach and allowing the score to move toward the target.'
- What this solution (achieved 0.29673) has done: 'We slightly strengthen the SVM (increase C to 5.0), allow a larger vocabulary (max feat = 35000) and keep very‑frequent words (max df = 0.99).  
We also modify cleanText to retain punctuation so that words with punctuation can be matched in the bag‑of‑words lookup, improving the selected‑text reconstruction and thus the Jaccard score.'
- What this solution (achieved 0.29789) has done: 'Implemented fixes and modest enhancements to lift the score toward the target:

1. Removed the unsupported `sublinear_tf` argument from `CountVectorizer`.
2. Renamed the returned vocabulary to `bow` for consistency with downstream code.
3. Adjusted SVM hyper‑parameters: increased `max_df` to keep more frequent terms, lowered `min_df` to retain rare words, and raised the regularisation constant `c` for stronger discrimination.

These changes resolve the runtime error, keep the core modeling logic unchanged, and are expected to improve the Jaccard score without extensive retraining modifications.'
- What this solution (achieved 0.29648) has done: 'I make two minimal adjustments that are expected to raise the Jaccard score toward the target:  
1. Make the text cleaning less aggressive (keep punctuation and most characters) so the bag‑of‑words better aligns with the original tweet substrings used for evaluation.  
2. Set `min_df` to 0 so even very rare words are kept in the vectorizer, giving the model more expressive power.  
Both changes preserve the overall modeling pipeline and only tweak preprocessing hyper‑parameters.'

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g., pd.read_csv)
import re, string  # static imports for cleanText
from functools import lru_cache

np.random.seed(42)  # deterministic seed

_bow_lookup = None

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

trainIdx = df_train.shape[0]

tune = False
test = False
method = False  # True selects MultinomialNB, False selects SVM

if method:
    max_df = 1.0
    min_df = 9  # maybe try higher?
    max_feat = 2000
    alpha = 6.0
else:
    max_df = 0.99  # keep more frequent terms
    min_df = 0  # retain even the rarest terms (previously 1)
    max_feat = 60000  # allow a larger feature space
    c = 20.0  # stronger regularisation for SVM




## === cell 1
@lru_cache(maxsize=None)
def cleanText(text):
    """
    Light cleaning: only lower‑case and strip whitespace.
    Keeping punctuation and most characters helps the model select
    substrings that match the original tweet text, which improves Jaccard.
    """
    text = str(text).strip().lower()
    return text




## === cell 2
def vectorizeIt(df, textCol, trainIdx, maxFeat=2000, maxDF=1.0, minDF=0.0):
    from sklearn.feature_extraction.text import CountVectorizer

    global _bow_lookup

    index = list(np.where(df[textCol].isnull())[0])

    new_df = df[textCol].dropna()

    vectorizer = CountVectorizer(
        stop_words=None,
        preprocessor=cleanText,
        max_features=maxFeat,
        max_df=maxDF,
        min_df=minDF,
        ngram_range=(1, 2),  # added bigrams
    )
    bow_matrix = vectorizer.fit_transform(new_df)  # single fit‑transform
    bow = vectorizer.get_feature_names_out().tolist()

    _bow_lookup = {word: i for i, word in enumerate(bow)}

    idxTr = [i for i in index if i < trainIdx]
    sz = trainIdx - len(idxTr)  # effective train rows after NaN removal
    xTrain = bow_matrix[:sz]
    xTest = bow_matrix[sz:]

    return index, bow, xTrain, xTest




## === cell 3
def sentArray(df, textCol):
    codes, _ = pd.factorize(df[textCol])
    return np.array(codes, dtype=int)




## === cell 4
def jaccard(str1, str2):
    import re

    str1 = str(str1)
    str2 = str(str2)
    if len(str1) > 0:
        str1 = re.sub(r'"', "", str1)
    if len(str2) > 0:
        str2 = re.sub(r'"', "", str2)
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    if len(c) > 0:
        jacc = float(len(c)) / (len(a) + len(b) - len(c))
    else:
        jacc = 0.0
    return jacc




## === cell 5
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




## === cell 6
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




## === cell 7
def multiNB(alpha, xTrain, xTest, yTrain, yTest):
    from sklearn.naive_bayes import MultinomialNB

    clf = MultinomialNB(alpha=alpha)
    clf.fit(xTrain, yTrain)
    accTrain = clf.score(xTrain, yTrain)
    accTest = clf.score(xTest, yTest)
    return clf, accTrain, accTest




## === cell 8
def tuneNB(all, train, test, trainIdx, max_feat, max_df, min_df, alpha):
    import numpy as np
    from sklearn.model_selection import train_test_split

    print("Tune MultinomialNB...")
    cvSz = 5




## === cell 9
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
                retDF.at[i, "selected_text"] = '""'
            else:
                retDF.at[i, "selected_text"] = '"' + aword + '"'
            if test:
                retDF.at[i, "sentiment"] = "neutral"
        elif pred[j] == 0:
            retDF.at[i, "selected_text"] = '"' + df["text"].iloc[i] + '"'
            if test:
                retDF.at[i, "sentiment"] = "neutral"
            j += 1
        else:
            outstring = ""
            idxA = pred[j]
            words = df["text"].iloc[i].split()
            for word in words:
                wordCheck = cleanText(word)
                idx_word = _bow_lookup.get(wordCheck)
                if idx_word is not None:
                    probWord = featLogProb[idxA, idx_word]
                    probNeu = featLogProb[0, idx_word]
                    probNeg = featLogProb[1, idx_word]
                    probPos = featLogProb[2, idx_word]
                    if (idxA == 1 and probWord >= probPos) or (
                        idxA == 2 and probWord >= probNeg
                    ):
                        outstring = word if not outstring else outstring + " " + word
            if outstring == "":
                outstring = df["text"].iloc[i]
            retDF.at[i, "selected_text"] = '"' + outstring + '"'
            if test:
                retDF.at[i, "sentiment"] = sentList[idxA]
            j += 1
    return retDF




## === cell 10
def svmClass(xTrain, xTest, yTrain, yTest, c=1.0):
    from sklearn.svm import LinearSVC

    clf = LinearSVC(
        C=c, loss="squared_hinge", class_weight="balanced", dual=False, max_iter=10000
    )
    clf.fit(xTrain, yTrain)
    accTrain = clf.score(xTrain, yTrain)
    accTest = clf.score(xTest, yTest)
    return clf, accTrain, accTest




## === cell 11
def tuneSVM(xTrain, xTest, yTrain, yTest):
    import numpy as np
    from sklearn.model_selection import train_test_split

    print("Tune SVM...")
    cvSz = 5




## === cell 12
def buildSelTextSVM(df, test, idxList, pred, bow, sentList, y, w):
    import pandas as pd
    import numpy as np

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
                retDF.at[i, "selected_text"] = '""'
            else:
                retDF.at[i, "selected_text"] = '"' + aword + '"'
            if test:
                retDF.at[i, "sentiment"] = "neutral"
        elif pred[j] == 0:
            retDF.at[i, "selected_text"] = '"' + df["text"].iloc[i] + '"'
            if test:
                retDF.at[i, "sentiment"] = "neutral"
            j += 1
        else:
            outstring = ""
            idxA = pred[j]
            words = df["text"].iloc[i].split()
            for word in words:
                wordCheck = cleanText(word)
                idx_word = _bow_lookup.get(wordCheck)
                if idx_word is not None:
                    d = np.zeros(3)
                    dWord = 0.0
                    yVal = y[j, idxA]
                    iBow = idx_word
                    if w[idxA, iBow] != 0.0:
                        dWord = yVal / abs(w[idxA, iBow])
                    for k in range(3):
                        if w[k, iBow] != 0.0:
                            d[k] = y[j, k] / abs(w[k, iBow])
                    if (
                        ((idxA == 1 and dWord >= d[2]) or (idxA == 2 and dWord >= d[1]))
                        and dWord > 0.0
                        and dWord >= d[0]
                    ):
                        outstring = word if not outstring else outstring + " " + word
            if outstring == "":
                outstring = df["text"].iloc[i]
            retDF.at[i, "selected_text"] = '"' + outstring + '"'
            if test:
                retDF.at[i, "sentiment"] = sentList[idxA]
            j += 1
    return retDF




## === cell 13
if method:
    print("Multinomial Naive Bayes Classifier")
    if tune:
        tuneNB(df_all, df_train, df_test, trainIdx, max_feat, max_df, min_df, alpha)
    else:
        index, bow, xTrain, xTest = vectorizeIt(
            df_all, "text", trainIdx, maxFeat=max_feat, maxDF=max_df, minDF=min_df
        )
        trainIdx_adj, yTrain, yTest, idxTr, idxTe = yPrep(
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
            scoreArr = np.zeros(sz)
            for i in range(sz):
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
        trainIdx_adj, yTrain, yTest, idxTr, idxTe = yPrep(
            index, trainIdx, df_train, df_test
        )
        clf, accTrain, accTest = svmClass(xTrain, xTest, yTrain, yTest, c)
        predTrain = clf.predict(xTrain)
        predTest = clf.predict(xTest)
        print("Training Accuracy: " + "{:.4f}".format(accTrain))
        print("Testing Accuracy: " + "{:.4f}".format(accTest))
        coef = clf.coef_
        y_dec = clf.decision_function(xTest)
        testRet_df = buildSelTextSVM(
            df_test, test, idxTe, predTest, bow, sentList, y_dec, coef
        )
        if test:
            y_dec_train = clf.decision_function(xTrain)
            trainRet_df = buildSelTextSVM(
                df_train, test, idxTr, predTrain, bow, sentList, y_dec_train, coef
            )
        testRet_df.to_csv(outFile, index=False)
        if test:
            trainRet_df.to_csv(outTrain, index=False)
            sz = len(df_train)
            scoreArr = np.zeros(sz)
            for i in range(sz):
                scoreArr[i] = jaccard(
                    df_train["selected_text"].iloc[i],
                    trainRet_df["selected_text"].iloc[i],
                )
            print("Jaccard Score: " + "{:.4f}".format(np.mean(scoreArr)))
