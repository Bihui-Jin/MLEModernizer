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

0.30507

# 6. Current score

0.24379

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.13708) has done: 'The fix sets the correct training size (`trainIdx`) based on the actual training dataframe and switches to the Naive Bayes classifier (which has a complete implementation), preventing the sample‑size mismatch and the missing SVM post‑processing function.'
- What this solution (achieved 0.13708) has done: 'I fix the path‑loading error, ensure the train / test DataFrames are correctly defined, run the Naive Bayes pipeline that was already written, and finally build a proper submission CSV containing the quoted `selected_text`. The changes are minimal, keep the original modelling logic, and make the script produce a valid `submission.csv` that can achieve a score closer to the target.'
- What this solution (achieved 0.1491) has done: 'I slightly adjust the hyper‑parameters to give the Naive Bayes model a richer feature set (lower `min_df`, a larger vocabulary, and bi‑grams) and use a smaller smoothing `alpha`. These tweaks keep the overall pipeline unchanged while expectedly boosting the Jaccard score toward the target.'
- What this solution (achieved 0.16608) has done: 'The changes lower the minimum document frequency, enlarge the vocabulary, use a smaller smoothing `alpha`, and extend n‑grams to trigrams – all aimed at improving the Naive Bayes classifier (and thus the Jaccard score) while keeping the original pipeline untouched.'
- What this solution (achieved 0.1693) has done: 'I slightly tune the hyper‑parameters of the existing Naive Bayes pipeline – lower the minimum document frequency to keep all words, raise the maximum feature count for a richer vocabulary, and reduce the Laplace smoothing (alpha). These minimal changes keep the core logic unchanged while giving the model more expressive power, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.17368) has done: 'I slightly tighten the Naive Bayes configuration to give it a richer vocabulary and much less smoothing (alpha = 0.001, max_feat = 40000, max_df = 0.90). These are minimal, focused tweaks that keep the overall pipeline unchanged while expectedly improving the classifier’s ability to pick the correct sentiment spans, moving the Jaccard score closer to the target.'
- What this solution (achieved 0.24288) has done: 'I increase the vocabulary size to capture more informative tokens and remove the English stop‑word filter (which drops words like “not” that are crucial for sentiment spans). These minimal tweaks keep the original Naive Bayes pipeline intact while giving the model richer features, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.24516) has done: 'I slightly adjust the feature extraction hyper‑parameters to make the Naïve Bayes model a bit less over‑fitted and give it a richer but cleaner vocabulary: lower the `max_df` to drop more very‑common tokens and raise the Laplace smoothing `alpha` a little. I also switch the CountVectorizer to binary counts (presence/absence) which often works better for text classification with NB. These changes are minimal, keep the original pipeline intact, and are expected to raise the Jaccard score toward the target without over‑optimising.'
- What this solution (achieved 0.24379) has done: 'I slightly adjust the Naïve Bayes hyper‑parameters to give the model a richer yet less noisy vocabulary: lower the Laplace smoothing `alpha` to 0.001, raise `max_df` to 0.85, and limit n‑grams to bigrams (1‑2) which is known to improve Jaccard performance for this task. These changes keep the original pipeline intact while nudging the score upward toward the target.'

# 9. Code solution

## === cell 0
tune = False
test = False
method = True  # Use MultinomialNB (True) instead of SVM (False)
if method:
    max_df = 0.85  # keep slightly more frequent terms than before
    min_df = 0  # keep all words (no minimum document frequency)
    max_feat = 80000  # larger vocabulary for richer representation
    alpha = 0.001  # much less smoothing for more discriminative probabilities
else:
    max_df = 1.0  # 0.1
    min_df = 12  # 14
    max_feat = 3000  # 2000
    c = 0.7  # 0.7



## === cell 1
import pandas as pd
import os

possible_dirs = [
    "input",  # default from original code
    "/kaggle/input/tweet-sentiment-extraction",  # typical Kaggle input path
    "/kaggle/input",  # generic kaggle input root
]
data_dir = None
for d in possible_dirs:
    if os.path.isdir(d) and os.path.isfile(os.path.join(d, "train.csv")):
        data_dir = d
        break
if data_dir is None:
    raise FileNotFoundError("train.csv not found in any expected directory.")

df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
df_test = pd.read_csv(os.path.join(data_dir, "test.csv"))

df_all = pd.concat([df_train, df_test], ignore_index=True)

trainIdx = len(df_train)  # index where test data starts in the combined df
outFile = "submission.csv"

sentList = df_train["sentiment"].unique().tolist()




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
        stop_words=None,
        preprocessor=cleanText,
        max_features=maxFeat,
        max_df=maxDF,
        min_df=minDF,
        ngram_range=(1, 2),  # use unigrams and bigrams only (less sparse than trigrams)
        binary=True,  # use presence/absence instead of raw counts
    )
    index = list(np.where(df[textCol].isnull())[0])
    new_df = df[textCol].dropna(axis=0)
    vectorizer.fit(new_df)
    inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]
    wc = vectorizer.transform(new_df)
    sz = trainIdx - len([i for i in index if i < trainIdx])
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
def buildSelTextNB(df, test, idxList, pred, bow, featLogProb, sentList):
    import pandas as pd

    """
    neutral == 0
    negative == 1
    positive == 2
    """
    if test:
        retDF = pd.DataFrame(columns=["textID", "selected_text", "sentiment"])
        retDF["textID"] = df["textID"]
    else:
        retDF = pd.DataFrame(columns=["textID", "selected_text"])
        retDF["textID"] = df["textID"]
    j = 0
    for i in range(len(retDF)):
        if i in idxList:  # rows where original text was NaN
            aword = str(df["text"].iloc[i])
            if aword == "nan":
                retDF.at[i, "selected_text"] = '""'
            else:
                retDF.at[i, "selected_text"] = f'"{aword}"'
            if test:
                retDF.at[i, "sentiment"] = "neutral"
        elif pred[j] == 0:  # neutral class – return whole tweet
            retDF.at[i, "selected_text"] = f'"{df["text"].iloc[i]}"'
            if test:
                retDF.at[i, "sentiment"] = "neutral"
            j += 1
        else:  # negative / positive – try to pick informative words
            outstring = ""
            idxA = pred[j]
            words = df["text"].iloc[i].split()
            for word in words:
                wordCheck = cleanText(word)
                if wordCheck in bow:
                    probWord = featLogProb[idxA, bow.index(wordCheck)]
                    probNeu = featLogProb[0, bow.index(wordCheck)]
                    probNeg = featLogProb[1, bow.index(wordCheck)]
                    probPos = featLogProb[2, bow.index(wordCheck)]
                    if (idxA == 1 and probWord > probPos) or (
                        idxA == 2 and probWord > probNeg
                    ):
                        outstring = word if not outstring else outstring + " " + word
            retDF.at[i, "selected_text"] = f'"{outstring}"'
            if test:
                retDF.at[i, "sentiment"] = sentList[idxA]
            j += 1
    return retDF




## === cell 10
def svmClass(xTrain, xTest, yTrain, yTest, c=1.0):
    from sklearn import svm

    clf = svm.SVC(C=c, kernel="linear")
    clf.fit(xTrain, yTrain)
    accTrain = clf.score(xTrain, yTrain)
    accTest = clf.score(xTest, yTest)
    return clf, accTrain, accTest




## === cell 11
index, vocab, xTrain, xTest = vectorizeIt(
    df_all, "text", trainIdx, maxFeat=max_feat, maxDF=max_df, minDF=min_df
)

trainIdx_adj, yTrain, yTest, idxTr, idxTe = yPrep(index, trainIdx, df_train, df_test)

if method:
    clf, accTrain, accTest = multiNB(alpha, xTrain, xTest, yTrain, yTest)
else:
    clf, accTrain, accTest = svmClass(xTrain, xTest, yTrain, yTest, c)

print(f"Training accuracy: {accTrain:.4f}, test accuracy: {accTest:.4f}")

pred_test = clf.predict(xTest)

submission_df = buildSelTextNB(
    df_test,
    test=True,
    idxList=idxTe,
    pred=pred_test,
    bow=vocab,
    featLogProb=clf.feature_log_prob_,
    sentList=sentList,
)

submission_final = submission_df[["textID", "selected_text"]]

submission_final.to_csv(outFile, index=False)
print(f"Submission saved to {outFile}")
