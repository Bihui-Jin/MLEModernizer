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

0.12176

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.17625) has done: 'I fix the runtime error caused by the removed `np.int` alias by switching it to the builtin `int` (score-neutral, but unblocks execution). I also correct the hard-coded `trainIdx` so the train/test split matches the actual concatenated dataframe boundary, preventing misalignment bugs and ensuring predictions map to the right rows. Finally, I make the output submission strictly match Kaggle’s required two-column format (`textID,selected_text`) and always write `submission.csv` successfully. Core model logic (CountVectorizer + SVM/NB flow and selected-text reconstruction) is preserved.'
- What this solution (achieved 0.17625) has done: 'To move your score upward toward the 0.26029 target (current 0.17625), I’m keeping the same CountVectorizer + linear SVM pipeline and the same “token filtering” reconstruction logic, but fixing two issues that suppress quality: (1) `sentList = df_all["sentiment"].unique()` creates a non-deterministic class↔label mapping that can misalign with `sentArray()`’s mapping; we hard-fix the sentiment order to `["neutral","negative","positive"]` to match the rest of your code. (2) In `vectorizeIt()`, you accidentally `fit()` twice and rebuild the vocabulary between them; we fit once and reuse the same fitted vocabulary/transform, which improves feature consistency without changing the core approach. These are minimal, semantics-preserving corrections that typically improve Jaccard by making classes and features consistent across train/test.'
- What this solution (achieved 0.17525) has done: 'We need to move your score up (0.17625 → target 0.26029), so the smallest legitimate gains here usually come from fixing the reconstruction step that’s currently discarding many correct spans. I keep your exact CountVectorizer + linear SVM pipeline and the same “token filtering” approach, but (1) make the SVM span-building use the *signed* class weights (your current `abs(w)` makes the word-importance test largely meaningless), and (2) add a safe fallback so if the filter selects nothing for non-neutral tweets, we output the full tweet (a common baseline that improves Jaccard vs empty strings). These are minimal, local changes that preserve the overall method and should raise the score toward the target without changing the model/training loop. The script still run end-to-end and write a valid `submission.csv` with `textID,selected_text`.'
- What this solution (achieved 0.18421) has done: 'I make two minimal, score-relevant fixes that preserve your CountVectorizer + linear SVM pipeline and the same token-filter reconstruction idea. First, your SVM reconstruction currently compares “per-class distances” using the raw `w[idxA, iBow]` which can be negative and flips the inequality logic; switching these comparisons to use `abs(w)` restores a consistent “importance magnitude” test and typically selects more correct spans. Second, when `cleanText(word)` becomes empty (common for punctuation/quotes), the current code may wrongly match empty tokens; skipping empty cleaned tokens avoids spurious selections and improves Jaccard stability. Everything else (data loading, vectorization, training, prediction, submission format/path) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.18421) has done: 'Your current score (0.18421) is below the target (0.26029), so we should improve it with minimal, score-relevant changes while preserving your CountVectorizer + linear SVM and the same token-filter reconstruction idea. The biggest suppressor here is that `cleanText()` removes apostrophes/quotes, but the reconstruction checks membership against the *raw* vocabulary tokens (which still include `'` etc.), causing many true sentiment words (e.g., “don't”, “can't”, “it's”) to never match and thus be dropped from `selected_text`. I make the vocabulary consistent with your preprocessor by rebuilding `bow` from `vectorizer.get_feature_names_out()` (already preprocessed) in `vectorizeIt()`, and I add a tiny performance-neutral speed fix by using a dict mapping token→index to avoid repeated `bow.index()` (this does not change semantics, just avoids timeouts). Everything else (architecture, training loop, vectorizer settings, output format/path) stays the same and still writes `submission.csv`.'
- What this solution (achieved 0.12073) has done: 'We need to move the public score up (0.18421 → 0.26029), so the smallest legitimate gain is to make the token filtering select *contiguous spans* instead of scattered words, because the competition metric rewards matching the exact excerpt. I keep your exact CountVectorizer + linear SVM training/prediction pipeline and the same “use SVM weights + decision_function to score tokens” logic, but I (1) score each token with the same dWord criterion you already use, then (2) pick the single best contiguous window of “kept” tokens (max sum of token scores) as `selected_text`. I also add a minimal sentiment-aware fallback for cases where no token passes (use full text for non-neutral, as you already do; and keep the neutral path unchanged). This preserves core semantics, still writes `submission.csv`, and should improve Jaccard by producing cleaner, more excerpt-like predictions.'
- What this solution (achieved 0.12073) has done: 'Your current score (0.12073) is far below the target (0.26029), so we should improve it with minimal, local changes that preserve your SVM + token-filter reconstruction approach. The biggest quality bug is that `y = clf.decision_function(xTest)` is for the test split only, but `buildSelTextSVM()` indexes it by `j` over the full `df_test` rows (including null-text rows you skipped during vectorization), which misaligns decision scores with tweets and wrecks selection. I fix this by passing a full-length `y_full` aligned to original row indices (filling skipped rows with zeros), and by computing `decision_function` on the correct matrix for each dataframe while keeping everything else identical. This is score-relevant, minimal, and should move your score back up toward (and likely beyond) your earlier ~0.18 baseline without changing model/feature/core logic.'
- What this solution (achieved 0.12073) has done: 'Your current score (0.12073) is far below the target (0.26029), so we should increase performance with the smallest changes that don’t alter your core “CountVectorizer + linear SVM + token scoring” approach. The main quality bug is still an index misalignment inside `buildSelTextSVM()`: it uses `j` (compressed index over non-null rows) to index `y`, but `y` is now full-length (original row indices), which makes token scoring effectively random for many tweets. I fix this by indexing `y` with the original row index `i` (full alignment), while keeping the exact scoring logic and contiguous-span selection unchanged. This is a local, semantics-preserving fix that should bring you back toward your earlier ~0.18 baseline and closer to the target, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.12073) has done: 'Your current score (0.12073) is far below the target (0.26029), so we should improve with the smallest fixes that keep your CountVectorizer + linear SVM + token-scoring/span selection core logic intact. The biggest quality suppressor is that `buildSelTextSVM()` currently indexes the decision scores `y` by the full row index `i`, but `y_test_full` is filled using `idxTe` values that are in the concatenated df_all coordinate system (train+test), causing many test rows to get zeroed decision scores and breaking excerpt selection. I fix this alignment by converting concatenated indices to test-local indices consistently (same for train when `test=True`) and by correcting an off-by-one in `yPrep()` (`>= trainIdx` instead of `> trainIdx`) so dropped test rows are computed correctly. These are minimal, score-relevant alignment corrections that should move you back toward your earlier ~0.18 baseline and closer to the 0.26 target, while still writing a valid `submission.csv`.'
- What this solution (achieved 0.12073) has done: 'Your current score (0.12073) is well below the target (0.26029), so we should raise performance with the smallest change that fixes a clear quality bug without altering the SVM/CountVectorizer core. The biggest suppressor is that the SVM excerpt builder uses `y[i, k]` even though the decision-function matrix `y` is only meaningful for the **compressed** non-null rows, and `j` is already tracking that compressed index; this mis-scores tokens for most tweets. I make `buildSelTextSVM()` consistently index decision scores by `j` (compressed row) while keeping the rest of the token scoring + contiguous-span selection logic identical. This should recover much of the lost performance back toward your earlier ~0.18 baseline and move closer to the 0.26 target, while still producing a valid `submission.csv`.'
- What this solution (achieved 0.12073) has done: 'Your current score (0.12073) is far below the target (0.26029), and the biggest suppressor is still a row-index alignment bug in the SVM excerpt builder: `buildSelTextSVM()` uses `j` (compressed index over non-null rows) to index `pred`, but your `y_test_full` is built in **full test row space**, so using `y[j, :]` mis-scores most tweets. I fix this minimally by making `buildSelTextSVM()` use `y[i, :]` (full row index) while keeping `j` only for indexing `pred`, preserving the same CountVectorizer+linear SVM and the same token scoring + contiguous-span selection logic. This should recover performance back toward your earlier ~0.18 baseline and move closer to the 0.26 target without changing the overall approach. The script still run end-to-end and write a valid `submission.csv` with `textID,selected_text`.'
- What this solution (achieved 0.12073) has done: 'We need to move your score up from 0.12073 toward 0.26029, so we should recover the earlier ~0.18–0.19 behavior with the smallest possible fix. The biggest score suppressor in your current SVM path is that `buildSelTextSVM()` decides “neutral => full text” based on `pred[j]`, but `pred` is produced from `xTest` (compressed, NaN-rows removed) while `j` also advances on full rows; this misaligns predictions with tweets and makes excerpt selection effectively random. I fix this by expanding `predTest` (and `predTrain` when `test=True`) back to full row length using `idxTe/idxTr`, so `pred[i]` always matches `df.iloc[i]`; then `buildSelTextSVM()` uses row index `i` (not `j`) consistently for both `pred` and `y`. Core logic (CountVectorizer + linear SVM + token scoring + contiguous-span selection) is preserved; this is purely an alignment correction and should increase Jaccard back toward your prior baseline.'
- What this solution (achieved 0.12176) has done: 'Your current public score (0.12073) is far below the target (0.26029), so we should increase performance with minimal, local fixes that keep your CountVectorizer + linear SVM and token-based reconstruction approach intact. The biggest quality issue is that you train the sentiment classifier on `df_all["text"]` only, but later try to extract the sentiment span from the raw tweet text: this mismatch makes the model ignore the provided `sentiment` signal that is crucial for this competition. I minimally incorporate sentiment into the same text field used for vectorization (by prefixing it into the tweet text for both train and test before vectorization) without changing the model type, training loop, or selection logic. This typically yields a meaningful jump in Jaccard while preserving your overall pipeline and still producing a valid `submission.csv`.'
- What this solution (achieved 0.12176) has done: 'I keep your current CountVectorizer + linear SVM pipeline and the same token-scoring/contiguous-span reconstruction, but fix a score-suppressing mismatch you introduced when you prefixed sentiment into the text used for vectorization. Right now, the excerpt builder cleans tokens from the *raw tweet* and looks them up in a vocabulary learned from *“sentiment + tweet”*, so the important tokens `"positive"`, `"negative"`, `"neutral"` (and their learned weights) are never considered, and the decision scores become less informative for span selection. The minimal fix is to make the reconstruction operate on the exact same “sentiment + tweet” string while still returning substrings from the original tweet (so submission semantics stay correct). This typically increases Jaccard toward your target without changing model/feature extraction/training logic.'

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

trainIdx = len(df_train)

df_list = [df_train[["textID", "text", "sentiment"]], df_test]
df_all = pd.concat(df_list, ignore_index=True)

sentList = np.array(["neutral", "negative", "positive"], dtype=object)



## === cell 1
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

    wc = vectorizer.fit_transform(new_df)

    vocabulary = list(vectorizer.get_feature_names_out())

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

    bow2idx = {t: k for k, t in enumerate(bow)}

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
                retDF["selected_text"].iloc[i] = '"' + '"'  # empty quoted string
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
                if wordCheck in bow2idx:
                    bi = bow2idx[wordCheck]
                    probWord = (featLogProb)[idxA, bi]
                    probNeg = (featLogProb)[1, bi]
                    probPos = (featLogProb)[2, bi]
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
def buildSelTextSVM(df, test, idxList, pred_full, bow, sentList, y_full, w):
    import pandas as pd
    import numpy as np

    """
    neutral == 0
    negative == 1
    positive == 2
    """
    print("Build SVM output...")

    bow2idx = {t: k for k, t in enumerate(bow)}

    def best_contiguous_span(words, keep_mask, token_scores):
        kept_positions = [k for k, m in enumerate(keep_mask) if m]
        if len(kept_positions) == 0:
            return ""

        best_sum = None
        best_l = kept_positions[0]
        best_r = kept_positions[0]

        cur_sum = 0.0
        cur_l = kept_positions[0]
        prev = None
        for pos in kept_positions:
            if prev is None or pos != prev + 1:
                cur_sum = token_scores[pos]
                cur_l = pos
            else:
                if cur_sum + token_scores[pos] >= token_scores[pos]:
                    cur_sum = cur_sum + token_scores[pos]
                else:
                    cur_sum = token_scores[pos]
                    cur_l = pos

            if (best_sum is None) or (cur_sum > best_sum):
                best_sum = cur_sum
                best_l = cur_l
                best_r = pos

            prev = pos

        return " ".join(words[best_l : best_r + 1])

    if test:
        retDF = pd.DataFrame(columns=["textID", "selected_text", "sentiment"])
        retDF["textID"] = df["textID"]
    else:
        retDF = pd.DataFrame(columns=["textID", "selected_text"])
        retDF["textID"] = df["textID"]

    for i in range(len(retDF)):
        if i in idxList:
            aword = str(df["text"].iloc[i])
            if aword == "nan":
                retDF["selected_text"].iloc[i] = '"' + '"'  # empty quoted string
            else:
                retDF["selected_text"].iloc[i] = '"' + aword + '"'
            if test:
                retDF["sentiment"].iloc[i] = "neutral"
            continue

        idxA = int(pred_full[i])

        if idxA == 0:
            retDF["selected_text"].iloc[i] = '"' + str(df["text"].iloc[i]) + '"'
            if test:
                retDF["sentiment"].iloc[i] = "neutral"
            continue

        words = (str(df["sentiment"].iloc[i]) + " " + str(df["text"].iloc[i])).split()

        keep_mask = [False] * len(words)
        token_scores = [0.0] * len(words)

        y_row = y_full[i, :]

        for t, word in enumerate(words):
            wordCheck = cleanText(word)
            if len(wordCheck) == 0:
                continue

            if wordCheck in bow2idx:
                iBow = bow2idx[wordCheck]

                yVal = y_row[idxA]

                d = np.zeros(shape=(3), dtype=float)
                dWord = 0.0

                denom = abs(w[idxA, iBow])
                if denom != 0.0:
                    dWord = yVal / denom

                for k in range(3):
                    denom_k = abs(w[k, iBow])
                    if denom_k != 0.0:
                        d[k] = y_row[k] / denom_k

                if (
                    (
                        ((idxA == 1) and (dWord > d[2]))
                        or ((idxA == 2) and (dWord > d[1]))
                    )
                    and (dWord > 0.0)
                    and (dWord > d[0])
                ):
                    keep_mask[t] = True
                    token_scores[t] = float(dWord)

        outstring = best_contiguous_span(words, keep_mask, token_scores)

        out_tokens = outstring.split()
        if len(out_tokens) > 0 and out_tokens[0].lower() in (
            "neutral",
            "negative",
            "positive",
        ):
            outstring = " ".join(out_tokens[1:])

        if len(outstring.strip()) == 0:
            outstring = str(df["text"].iloc[i])

        outstring = '"' + outstring + '"'
        retDF["selected_text"].iloc[i] = outstring
        if test:
            retDF["sentiment"].iloc[i] = sentList[idxA]

    return retDF




## === cell 14
df_all_vec = df_all.copy()
df_all_vec["text"] = (
    df_all_vec["sentiment"].astype(str).fillna("")
    + " "
    + df_all_vec["text"].astype(str).fillna("")
)

if method:
    print("Multinomial Naive Bayes Classifier")
    if tune:
        tuneNB(df_all_vec, df_train, df_test, trainIdx, max_feat, max_df, min_df, alpha)
    else:
        index, bow, xTrain, xTest = vectorizeIt(
            df_all_vec, "text", trainIdx, maxFeat=max_feat, maxDF=max_df, minDF=min_df
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

        submission = testRet_df[["textID", "selected_text"]].copy()
        submission.to_csv(outFile, index=False)

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
        df_all_vec, "text", trainIdx, maxFeat=max_feat, maxDF=max_df, minDF=min_df
    )
    trainIdx_adj, yTrain, yTest, idxTr, idxTe = yPrep(
        index, trainIdx, df_train, df_test
    )
    if tune:
        tuneSVM(xTrain, xTest, yTrain, yTest)
    else:
        clf, accTrain, accTest = svmClass(xTrain, xTest, yTrain, yTest, c)
        predTrain = clf.predict(xTrain)
        predTest = clf.predict(xTest)
        print("Training Accuracy: " + "{:.4f}".format(accTrain))
        print("Testing Accuracy: " + "{:.4f}".format(accTest))

        coef = clf.coef_.toarray()
        w_norm = np.linalg.norm(coef)

        y_xTest = clf.decision_function(xTest)

        y_test_full = np.zeros((len(df_test), y_xTest.shape[1]), dtype=float)
        keep_rows_test = [i for i in range(len(df_test)) if i not in idxTe]
        y_test_full[keep_rows_test, :] = y_xTest

        pred_test_full = np.zeros((len(df_test),), dtype=int)
        pred_test_full[keep_rows_test] = predTest

        dist = y_xTest / w_norm  # parity with original (unused later)
        testRet_df = buildSelTextSVM(
            df_test, test, idxTe, pred_test_full, bow, sentList, y_test_full, coef
        )

        if test:
            y_xTrain = clf.decision_function(xTrain)
            y_train_full = np.zeros((len(df_train), y_xTrain.shape[1]), dtype=float)
            keep_rows_train = [i for i in range(len(df_train)) if i not in idxTr]
            y_train_full[keep_rows_train, :] = y_xTrain

            pred_train_full = np.zeros((len(df_train),), dtype=int)
            pred_train_full[keep_rows_train] = predTrain

            dist = y_xTrain / w_norm
            trainRet_df = buildSelTextSVM(
                df_train,
                test,
                idxTr,
                pred_train_full,
                bow,
                sentList,
                y_train_full,
                coef,
            )

        submission = testRet_df[["textID", "selected_text"]].copy()
        submission.to_csv(outFile, index=False)

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
