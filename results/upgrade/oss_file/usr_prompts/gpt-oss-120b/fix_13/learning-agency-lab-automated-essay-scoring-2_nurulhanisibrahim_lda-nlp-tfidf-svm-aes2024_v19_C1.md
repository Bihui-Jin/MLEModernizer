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
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
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
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Target score

0.72618

# 6. Current score

0.63069

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.4759) has done: 'The fixes address the main errors: 
* `y` is now taken directly from the `score` column. 
* The pipeline no longer uses `MaxAbsScaler` (which fails on sparse COO matrices) and replaces the `SVC` (which cannot handle sparse input) with a `LinearSVC` that works with the TF‑IDF sparse matrix. 
* The model is trained, validated, and then used to predict the test set, creating a proper `submission.csv` with integer scores. 
* All subsequent cells that depended on undefined variables are updated accordingly.'
- What this solution (achieved 0.62293) has done: 'I keep the overall pipeline unchanged but improve the text representation and the classifier regularisation, which are known to raise quadratic weighted kappa for this task.  
- In **cell 7** the TF‑IDF vectoriser now use sub‑linear TF scaling, IDF weighting, L2‑norm, and include bigrams while lowering the minimum document frequency so more informative features are kept.  
- In **cell 8** the `LinearSVC` receive a slightly larger C value and `class_weight='balanced'` to better handle the imbalanced score distribution.  
These small, targeted tweaks should move the validation kappa closer to the target 0.726 while preserving the original model structure and end‑to‑end workflow.'
- What this solution (achieved 0.60379) has done: 'I tune the TF‑IDF vectoriser to keep more informative n‑grams (including trigrams and a lower min_df) and expand the feature space, then increase the LinearSVC regularisation strength C and iterations. These small hyper‑parameter adjustments keep the original model intact while aiming to raise the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.6048) has done: 'I slightly adjust the TF‑IDF settings to keep rarer terms (min_df=1) and strengthen the LinearSVC regularisation (increase C, enable the appropriate dual formulation and allow more iterations). These modest tweaks keep the original pipeline intact while aiming to raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.58931) has done: 'The fix adds a safe default when mapping predictions (so unseen labels fall back to themselves) and bumps the LinearSVC regularisation slightly to nudge the validation kappa upward. This resolves the TypeError, ensures `test_pred` is defined, and keeps the core pipeline unchanged while modestly improving the score.'
- What this solution (achieved 0.59786) has done: 'I slightly increase the regularisation strength (C) of the LinearSVC and simplify the post‑processing: the model already predicts the correct score labels, so the extra “mode‑mapping” step can be removed. This reduces unnecessary label distortion and should raise the quadratic weighted kappa toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.59435) has done: 'I make a very small, targeted tweak to the classifier: increase the regularisation strength (C) and switch to the primal optimisation (`dual=False`). These changes keep the overall pipeline unchanged while giving the model a bit more flexibility, which should push the quadratic weighted kappa upward toward the target score.'
- What this solution (achieved 0.59045) has done: 'I make two very small, targeted tweaks that keep the overall pipeline unchanged:  
1. Increase the TF‑IDF feature space (`max_features`) to capture more informative terms.  
2. Strengthen the LinearSVC regularisation (`C`) and allow more iterations so the model can converge better on this richer representation. These adjustments are expected to raise the validation quadratic weighted kappa and move the score closer to the target without altering the core logic.'
- What this solution (achieved 0.65315) has done: 'I keep the overall TF‑IDF + LinearSVC pipeline but tune it to be less over‑regularised and add a lightweight calibration step.  
- Reduce the regularisation strength (C) so the model does not over‑fit the high‑dimensional sparse vectors.  
- Limit the n‑gram range to unigrams + bigrams and shrink `max_features` a bit to improve signal‑to‑noise.  
- Wrap the trained `LinearSVC` in `CalibratedClassifierCV` (5‑fold Platt scaling) and use the calibrated model for validation and test predictions; this gives better‑aligned probability estimates that, after rounding, usually raise the quadratic weighted kappa.  
These changes stay within the original workflow while nudging the score toward the target.'
- What this solution (achieved 0.63847) has done: 'I slightly expand the TF‑IDF representation to include trigrams and a few more features, and increase the LinearSVC regularisation strength (C) so the model captures more signal without changing the overall pipeline. These modest adjustments are expected to raise the quadratic weighted kappa toward the target while keeping the core workflow intact.'
- What this solution (achieved 0.63069) has done: 'I slightly adjust the TF‑IDF vectoriser to drop very rare tokens (min_df = 2) and limit n‑grams to unigrams + bigrams with a smaller feature cap, which reduces noise and often improves kappa. I also lower the LinearSVC regularisation (C = 2.0) and remove the extra calibration step, keeping the same overall pipeline but giving the classifier a better bias‑variance balance. These minimal changes should raise the quadratic weighted kappa toward the target while still producing a valid submission file.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.svm import LinearSVC
from collections import Counter



## === cell 1
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 2
train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)



## === cell 3
cList = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "when's": "when is",
    "where's": "where is",
    "who's": "who is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you'd": "you had",
    "you're": "you are",
    "you've": "you have",
}
c_re = re.compile("(%s)" % "|".join(cList.keys()))




## === cell 4
def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]

    return c_re.sub(replace, text)


def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub("", x)


def dataPreprocessing(series):
    series = series.str.lower()
    series = series.apply(removeHTML)
    series = series.apply(lambda s: re.sub("@\\w+", "", s))
    series = series.apply(lambda s: re.sub("'\\d+", "", s))
    series = series.apply(lambda s: re.sub("\\d+", "", s))
    series = series.apply(lambda s: re.sub("http\\w+", "", s))
    series = series.apply(lambda s: re.sub(r"\\s+", " ", s))
    series = series.apply(expandContractions)
    series = series.apply(lambda s: re.sub(r"\\.+", ".", s))
    series = series.apply(lambda s: re.sub(r"\\,+", ",", s))
    series = series.apply(lambda s: re.sub("\\n", "", s))
    series = series.apply(lambda s: re.sub("[^\\w\\s]", "", s))
    series = series.str.strip()
    return series




## === cell 5
X_full = dataPreprocessing(train_df["full_text"])
X_test_full = dataPreprocessing(test_df["full_text"])
y = train_df["score"]  # correct target series



## === cell 6
X_train, X_valid, y_train, y_valid = train_test_split(
    X_full, y, test_size=0.1, random_state=123, stratify=y
)



## === cell 7
vectorizer = TfidfVectorizer(
    stop_words="english",
    sublinear_tf=True,
    use_idf=True,
    norm="l2",
    strip_accents="unicode",
    binary=False,
    analyzer="word",
    token_pattern=r"\w{2,}",
    ngram_range=(1, 2),  # unigrams + bigrams only
    max_features=250000,  # reduced vocab size
    min_df=2,  # ignore extremely rare terms
)

X_train_vec = vectorizer.fit_transform(X_train)
X_valid_vec = vectorizer.transform(X_valid)



## === cell 8
base_clf = LinearSVC(
    C=2.0,  # lower C than before
    dual=False,
    max_iter=20000,
    class_weight="balanced",
    random_state=123,
)
clf = base_clf  # no calibration step
clf.fit(X_train_vec, y_train)



## === cell 9
y_valid_pred = clf.predict(X_valid_vec)

print("Confusion Matrix:")
print(confusion_matrix(y_valid, y_valid_pred))

print("\nClassification Report:")
print(classification_report(y_valid, y_valid_pred, digits=4))

kappa = cohen_kappa_score(y_valid, y_valid_pred, weights="quadratic")
print(f"\nQuadratic Weighted Kappa (raw): {kappa:.5f}")



## === cell 10
X_test_vec = vectorizer.transform(X_test_full)
test_pred_raw = clf.predict(X_test_vec)

test_pred = np.clip(np.rint(test_pred_raw), 1, 6).astype(int)



## === cell 11
submission_path = "/kaggle/working/submission.csv"
submission = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)
submission["score"] = test_pred
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
display(submission.head())
