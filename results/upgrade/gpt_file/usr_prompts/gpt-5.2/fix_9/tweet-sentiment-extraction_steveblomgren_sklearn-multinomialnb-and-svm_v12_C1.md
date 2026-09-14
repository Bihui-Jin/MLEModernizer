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

0.36899

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.34544) has done: 'I fix the inconsistent train/test split that causes `xTrain` and `yTrain` to have different lengths by deriving `trainIdx` directly from `len(df_train)` instead of using a hardcoded value. I also correct the off-by-one bug in `yPrep` where test null indices at exactly `trainIdx` were mishandled, which can silently misalign labels. Finally, I make the submission writer robust by enforcing string dtype and filling missing predictions with the original tweet text so a valid `submission.csv` is always produced without changing your model choice or training approach.'
- What this solution (achieved 0.09026) has done: 'Your score gap to the target is large (0.34544 vs 0.52952, higher-is-better), so the smallest safe lift is to fix the submission post-processing so it matches the competition’s quoted-string requirement and avoids empty predictions. Concretely, I (1) stop stripping punctuation/quotes when building the final `selected_text` (your current `cleanText` removes punctuation, making it hard to output exact spans), and (2) add a minimal, deterministic fallback that chooses the best contiguous phrase from the original tweet based on the model’s word evidence—so you return an actual span from the tweet (quoted) instead of sometimes returning empty strings. This keeps your core model (CountVectorizer + linear SVM / NB) and training loop intact, but aligns the output format and extraction with the Jaccard metric so the score moves upward toward the target. The script still runs end-to-end and always writes `submission.csv` with correct columns.'
- What this solution (achieved 0.09026) has done: 'I make two minimal, score-aligned fixes without changing your model/training: (1) ensure the CountVectorizer vocabulary is consistent with the transformed matrix (your current code fits twice and builds `bow` from a different vocab than the one used in `wc`, which makes token scoring nearly random and tanks Jaccard), and (2) speed up/clean up span scoring by using a `token->vocab_index` dict (no `bow.index()` linear scans), which also reduces subtle mismatches from duplicates. These changes keep the same SVM/NB core logic, but make the “evidence-based span extraction” actually correspond to the trained model’s features, which should move the score upward toward your 0.52952 target. Submission writing and quoting remain intact and the script still produces `submission.csv` end-to-end.'
- What this solution (achieved 0.13246) has done: 'Your current score is far below the target (0.09026 vs 0.52952, higher-is-better), so we need a minimal but directly score-aligned fix. The main issue is that your classifier is trained to predict *sentiment*, but your extraction logic then tries to use class-specific weights/probabilities to pick words—this is fundamentally misaligned and produces near-random spans. Without changing the overall model family or training loop, we can instead extract spans using simple, deterministic heuristics that are known to jump this competition’s baseline: for non-neutral sentiments, return the original tweet with common stopwords removed (as an exact span), and for neutral return the full text. This preserves your vectorizer+SVM/NB pipeline (so “core logic” remains), but makes the selected_text construction metric-aligned and should move the score substantially toward the target while still always producing a valid quoted `submission.csv`.'
- What this solution (achieved 0.09267) has done: 'Your current score (0.13246) is far below the target (0.52952), so we need a minimal, metric-aligned change that improves span extraction without changing your vectorizer/SVM training core. The biggest direct issue is that `buildSelTextSVM/NB` almost always returns either the full tweet or a stopword-stripped chunk, which is not sentiment-specific and badly mismatches the labeled “supporting span” objective. I keep the same sentiment classifier, but change only the post-processing to extract a contiguous span from the *original tweet* by selecting tokens whose class-weight contribution supports the predicted sentiment (using `MultinomialNB.feature_log_prob_` or `LinearSVC/SVC.coef_`), with a safe fallback to your current heuristic. This preserves your model/training approach and output semantics, but makes the predicted `selected_text` much more correlated with the sentiment evidence, which should move the public score upward toward the target band while still always producing a valid quoted `submission.csv`.'
- What this solution (achieved 0.29825) has done: 'We need to move your public Jaccard score up toward 0.52952 (current 0.09267), so the smallest score-aligned change is to stop using the sentiment classifier’s feature weights to “extract” spans (it’s not trained for span selection) and instead use a deterministic, competition-aligned baseline: for neutral return the full tweet; for positive/negative return the longest matching contiguous span from the original tweet using the tweet-sentiment-extraction “spaces-only” tokenization plus a tiny punctuation trim. This preserves your core pipeline (CountVectorizer + SVM/NB sentiment classifier and the same training loop), but replaces only the post-processing span selection to one that is known to be much more correlated with the Jaccard metric. I also keep your robust quoting/fallbacks so the submission is always valid. These changes should materially increase score while staying minimal and within the 600s runtime.'
- What this solution (achieved 0.36899) has done: 'Your current extraction is still mostly a generic baseline (neutral→full text; pos/neg→stopword span), which caps Jaccard well below the target. Without changing your model/training, we can move the score upward by making the span selection use the **tweet’s own token overlap with a learned pos/neg lexicon** derived from the training `selected_text` (a lightweight, deterministic post-process aligned to Jaccard). Concretely: build per-sentiment token score dictionaries from training spans, then for each test tweet choose the best contiguous span whose tokens maximize those scores (with a safe fallback to your existing baseline). This keeps your SVM/NB pipeline intact and only adjusts post-processing in a metric-aligned way, which should improve toward 0.52952 while staying fast and stable.'

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
    max_df = 1.0  # 0.1
    min_df = 12  # 14
    max_feat = 3000  # 2000
    c = 0.7  # 0.7




## === cell 2
def cleanText(text):
    """
    IMPORTANT for score: keep this as the model preprocessor only.
    We will NOT use this cleaned string as the final selected_text output,
    because Kaggle scoring expects an exact span (with punctuation/case).
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
        plt.legend()
    else:
        plt.plot(x, y, "bo-")
        plt.legend([label1])
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




## === cell 10
def _tokenize_keep_original(text):
    text = "" if text is None else str(text)
    if text == "nan":
        text = ""
    return text.split()


def _best_span_from_mask(tokens, keep_mask):
    best_s = -1
    best_len = 0
    cur_s = -1
    cur_len = 0
    for i, k in enumerate(keep_mask):
        if k:
            if cur_s < 0:
                cur_s = i
                cur_len = 1
            else:
                cur_len += 1
        else:
            if cur_len > best_len:
                best_len = cur_len
                best_s = cur_s
            cur_s = -1
            cur_len = 0
    if cur_len > best_len:
        best_len = cur_len
        best_s = cur_s
    if best_s < 0 or best_len <= 0:
        return ""
    return " ".join(tokens[best_s : best_s + best_len])


_STOPWORDS = set(
    """
a about above after again against all am an and any are arent as at be because been before being below between
both but by cant cannot could couldnt did didnt do does doesnt doing dont down during each few for from further
had hadnt has hasnt have havent having he hed hell hes her here heres hers herself him himself his how hows
i id ill im ive if in into is isnt it its its itself lets me more most mustnt my myself no nor not of off on
once only or other ought our ours ourselves out over own same shant she shed shell shes should shouldnt so some
such than that thats the their theirs them themselves then there theres these they theyd theyll theyre theyve
this those through to too under until up very was wasnt we wed well were weve were werent what whats when whens
where wheres which while who whos whom why whys with wont would wouldnt you youd youll youre youve your yours yourself yourselves
""".split()
)


def _remove_stopwords_span(raw_text):
    tokens = _tokenize_keep_original(raw_text)
    if len(tokens) == 0:
        return ""
    keep = []
    for tok in tokens:
        core = tok.strip(".,!?;:\"'()[]{}").lower()
        keep.append(core != "" and core not in _STOPWORDS)
    span = _best_span_from_mask(tokens, keep)
    if span == "":
        for tok, k in zip(tokens, keep):
            if k:
                return tok
        return tokens[0]
    return span


def _trim_punct(token):
    return token.strip(".,!?;:\"'()[]{}")


def _build_sentiment_token_scores(df_train):
    import re
    from collections import Counter, defaultdict

    def norm(tok):
        t = "" if tok is None else str(tok)
        t = t.strip().lower()
        t = re.sub(r"^[\W_]+|[\W_]+$", "", t)
        return t

    text_counts = Counter()
    sel_counts = defaultdict(Counter)  # sentiment -> Counter(token)
    for i in range(len(df_train)):
        sent = (
            str(df_train["sentiment"].iloc[i])
            if "sentiment" in df_train.columns
            else "neutral"
        )
        txt = str(df_train["text"].iloc[i]) if "text" in df_train.columns else ""
        sel = (
            str(df_train["selected_text"].iloc[i])
            if "selected_text" in df_train.columns
            else ""
        )
        if txt == "nan":
            txt = ""
        if sel == "nan":
            sel = ""
        for tok in txt.split():
            t = norm(tok)
            if t:
                text_counts[t] += 1
        for tok in sel.split():
            t = norm(tok)
            if t:
                sel_counts[sent][t] += 1

    scores = {}
    for sent, ctr in sel_counts.items():
        sd = {}
        for t, c in ctr.items():
            if t in _STOPWORDS:
                continue
            denom = 1.0 + np.log1p(text_counts.get(t, 0))
            sd[t] = float(np.log1p(c) / denom)
        scores[sent] = sd
    return scores


def _best_scoring_span(raw_text, sentiment, token_scores):
    """
    Find contiguous span maximizing sum(token_scores) over tokens in the tweet.
    Uses original whitespace tokenization so output is exact-span-compatible.
    """
    import re

    raw_text = "" if raw_text is None else str(raw_text)
    if raw_text == "nan":
        raw_text = ""
    tokens = _tokenize_keep_original(raw_text)
    if len(tokens) == 0:
        return ""

    sd = token_scores.get(str(sentiment), {})
    if not sd:
        return ""

    def norm(tok):
        t = tok.strip().lower()
        t = re.sub(r"^[\W_]+|[\W_]+$", "", t)
        return t

    best_sum = 0.0
    best_i = -1
    best_j = -1
    cur_sum = 0.0
    cur_i = 0
    for j, tok in enumerate(tokens):
        sc = sd.get(norm(tok), 0.0)
        if cur_sum + sc <= 0.0:
            cur_sum = 0.0
            cur_i = j + 1
        else:
            cur_sum += sc
            if cur_sum > best_sum:
                best_sum = cur_sum
                best_i = cur_i
                best_j = j

    if best_i < 0:
        return ""
    return " ".join(tokens[best_i : best_j + 1])


def _baseline_selected_text(
    raw_text, sentiment, fallback_non_neutral=None, token_scores=None
):
    raw_text = "" if raw_text is None else str(raw_text)
    if raw_text == "nan":
        raw_text = ""
    sentiment = "" if sentiment is None else str(sentiment)

    if sentiment.lower() == "neutral":
        return raw_text

    if token_scores is not None:
        span = _best_scoring_span(raw_text, sentiment, token_scores)
        if span != "":
            return span

    if not fallback_non_neutral:
        return _remove_stopwords_span(raw_text)

    tokens = _tokenize_keep_original(raw_text)
    if len(tokens) == 0:
        return ""

    target_set = set(
        [
            _trim_punct(t).lower()
            for t in str(fallback_non_neutral).split()
            if _trim_punct(t) != ""
        ]
    )
    if len(target_set) == 0:
        return _remove_stopwords_span(raw_text)

    keep = []
    for tok in tokens:
        core = _trim_punct(tok).lower()
        keep.append(core in target_set and core != "")

    span = _best_span_from_mask(tokens, keep)
    if span == "":
        span = _remove_stopwords_span(raw_text)
    return span


def _quote_selected(s):
    s = "" if s is None else str(s)
    if s == "nan":
        s = ""
    return '"' + s + '"'


def buildSelTextNB(
    df, test, idxList, pred, bow, featLogProb, sentList, token_scores=None
):
    import pandas as pd

    print("Build MultinomialNB output...")
    if test:
        retDF = pd.DataFrame(columns=["textID", "selected_text", "sentiment"])
        retDF["textID"] = df["textID"]
    else:
        retDF = pd.DataFrame(columns=["textID", "selected_text"])
        retDF["textID"] = df["textID"]

    for i in range(len(retDF)):
        raw_text = str(df["text"].iloc[i]) if "text" in df.columns else ""
        if raw_text == "nan":
            raw_text = ""

        if i in idxList:
            sel = raw_text
            if test:
                retDF.loc[i, "sentiment"] = "neutral"
        else:
            if test:
                cls = int(pred[i - sum([1 for z in idxList if z < i])])
                sent = sentList[cls]
                retDF.loc[i, "sentiment"] = sent
                sel = _baseline_selected_text(
                    raw_text, sent, fallback_non_neutral=None, token_scores=token_scores
                )
            else:
                sent = (
                    df["sentiment"].iloc[i] if "sentiment" in df.columns else "neutral"
                )
                gold = (
                    df["selected_text"].iloc[i] if "selected_text" in df.columns else ""
                )
                sel = _baseline_selected_text(
                    raw_text, sent, fallback_non_neutral=gold, token_scores=token_scores
                )

        retDF.loc[i, "selected_text"] = _quote_selected(sel)

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
def tuneSVM(all, train, test, trainIdx, max_feat, max_df, min_df, c):
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
        c_arr = np.arange(start=0.1, stop=1.1, step=0.1)
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
                    xTrain, yTrain, test_size=vldSz, random_state=i, shuffle=True
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




## === cell 13
def buildSelTextSVM(
    df, test, idxList, pred, bow, sentList, y, w, b, x, token_scores=None
):
    import pandas as pd

    print("Build SVM output...")
    if test:
        retDF = pd.DataFrame(columns=["textID", "selected_text", "sentiment"])
        retDF["textID"] = df["textID"]
    else:
        retDF = pd.DataFrame(columns=["textID", "selected_text"])
        retDF["textID"] = df["textID"]

    j = 0
    for i in range(len(retDF)):
        raw_text = str(df["text"].iloc[i]) if "text" in df.columns else ""
        if raw_text == "nan":
            raw_text = ""

        if i in idxList:
            sel = raw_text
            if test:
                retDF.loc[i, "sentiment"] = "neutral"
        else:
            if test:
                cls = int(pred[j])
                sent = sentList[cls]
                retDF.loc[i, "sentiment"] = sent
                sel = _baseline_selected_text(
                    raw_text, sent, fallback_non_neutral=None, token_scores=token_scores
                )
                j += 1
            else:
                sent = (
                    df["sentiment"].iloc[i] if "sentiment" in df.columns else "neutral"
                )
                gold = (
                    df["selected_text"].iloc[i] if "selected_text" in df.columns else ""
                )
                sel = _baseline_selected_text(
                    raw_text, sent, fallback_non_neutral=gold, token_scores=token_scores
                )

        retDF.loc[i, "selected_text"] = _quote_selected(sel)

    return retDF




## === cell 14
token_scores = _build_sentiment_token_scores(df_train)

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
            df_test,
            test,
            idxTe,
            predTest,
            bow,
            clf.feature_log_prob_,
            sentList,
            token_scores=token_scores,
        )
        if test:
            trainRet_df = buildSelTextNB(
                df_train,
                test,
                idxTr,
                predTrain,
                bow,
                clf.feature_log_prob_,
                sentList,
                token_scores=token_scores,
            )

        sub = testRet_df[["textID", "selected_text"]].copy()
        fallback_full = '"' + df_test["text"].fillna("").astype(str) + '"'
        sub["selected_text"] = sub["selected_text"].fillna(fallback_full).astype(str)
        sub.loc[
            sub["selected_text"].str.replace('"', "", regex=False).str.strip() == "",
            "selected_text",
        ] = fallback_full
        sub.to_csv(outFile, index=False)

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

        if xTrain.shape[0] != len(yTrain):
            raise ValueError(
                f"xTrain/yTrain length mismatch: {xTrain.shape[0]} vs {len(yTrain)}"
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
            df_test,
            test,
            idxTe,
            predTest,
            bow,
            sentList,
            y,
            coef,
            b,
            xTest.toarray(),
            token_scores=token_scores,
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
                xTrain.toarray(),
                token_scores=token_scores,
            )

        sub = testRet_df[["textID", "selected_text"]].copy()
        fallback_full = '"' + df_test["text"].fillna("").astype(str) + '"'
        sub["selected_text"] = sub["selected_text"].fillna(fallback_full).astype(str)
        sub.loc[
            sub["selected_text"].str.replace('"', "", regex=False).str.strip() == "",
            "selected_text",
        ] = fallback_full
        sub.to_csv(outFile, index=False)

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
