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

0.56375

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.48391) has done: 'I fixed the pipeline by removing the `MaxAbsScaler` (which caused a fit error on sparse data) and replacing the `SVC` with `LinearSVC`, which natively supports sparse TF‑IDF matrices. The rest of the workflow (loading data, preprocessing, TF‑IDF vectorization, train‑validation split, evaluation with quadratic weighted kappa, and writing a proper `submission.csv`) is kept unchanged, so the script now runs end‑to‑end and produces a valid submission file.'
- What this solution (achieved 0.63305) has done: 'I enable true TF‑IDF weighting and sublinear term frequency, broaden the n‑gram range to include bigrams, and add balanced class weights to the LinearSVC model. These small tweaks keep the overall pipeline unchanged while giving the classifier richer features and better handling of class imbalance, which should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.60463) has done: 'I slightly strengthen the TF‑IDF features (include trigrams, lower `min_df`, allow more max features) and increase the LinearSVC regularisation strength (`C`) and iteration limit. These adjustments keep the original pipeline intact while giving the model a bit more expressive power, which should raise the quadratic weighted kappa toward the target. I also clip the final predictions to the valid 1‑6 range for safety.'
- What this solution (achieved 0.58399) has done: 'I slightly strengthen the TF‑IDF features by lowering `min_df` to capture rarer informative words, and increase the LinearSVC regularisation strength (`C`) and iteration limit so the classifier can fit the richer feature set better. These small parameter tweaks keep the overall pipeline unchanged while aiming to raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.59636) has done: 'I slightly enrich the TF‑IDF representation (lower min_df, expand ngram_range to include 4‑grams, increase max_features) and give the LinearSVC a bit more flexibility by raising C. These modest adjustments keep the overall pipeline unchanged while providing the model with richer features and a stronger fit, which should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.57138) has done: 'I slightly broaden the TF‑IDF features by allowing single‑document terms (min_df = 1) and give the LinearSVC a bit more regularisation power (C = 20) with a higher iteration limit so it can fully converge on the richer feature set. These minimal tweaks keep the overall pipeline unchanged while expected to raise the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.59524) has done: 'I slightly simplify the TF‑IDF representation to reduce noisy rare terms and temper the classifier regularisation. The vectorizer now use up to 3‑grams, a higher `min_df` (2) and a smaller feature cap (300 k). The LinearSVC regularisation strength is lowered to `C=5.0`. These modest tweaks keep the overall pipeline unchanged while giving the model a cleaner feature set and a better‑balanced fit, which should move the quadratic weighted kappa closer to the target score.'
- What this solution (achieved 0.60033) has done: 'I slightly expand the TF‑IDF feature set (lower min_df, larger max_features, include 4‑grams) and give the LinearSVC a bit more regularisation strength and iterations. These small, targeted tweaks keep the overall pipeline unchanged while providing richer text representations and a stronger classifier, which should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.60107) has done: 'I add a second LinearSVC model with a different regularisation strength (C=5.0) and combine its predictions with the original C=10.0 model by averaging the integer scores (rounded). This small ensemble often improves robustness and should lift the quadratic weighted kappa toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.59996) has done: 'I increase the regularisation strength of the primary LinearSVC (C = 15) and give it more iterations (max_iter = 40000) so it can better fit the richer TF‑IDF features. The secondary model remains unchanged; the ensemble still averages the two predictions. These minimal parameter tweaks keep the overall pipeline intact while aiming to raise the validation QWK toward the target score.'
- What this solution (achieved 0.59049) has done: 'I add a simple post‑processing step that remaps each predicted score to the most frequent true label observed for that prediction on the validation split. This tiny adjustment keeps the exact same TF‑IDF + LinearSVC pipeline while often boosting quadratic weighted kappa, moving the score closer to the target. The mapping is learned from the validation data and then applied to the test predictions before saving the submission.'
- What this solution (achieved 0.56375) has done: 'I replace the mode‑based post‑processing with a mean‑based mapping: for each ensemble‑predicted class we compute the average true score on the validation split and round it to the nearest integer. This simple calibration often aligns predictions more closely with the true distribution and tends to raise the quadratic weighted kappa while keeping the original TF‑IDF + LinearSVC pipeline unchanged.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.svm import LinearSVC




## === cell 1
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 2
train_df = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)
test_df = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
sample_submission = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)




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
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
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
    series = series.apply(lambda s: re.sub(r",+", ",", s))
    series = series.apply(lambda s: re.sub("\n", "", s))
    series = series.apply(lambda s: re.sub("[^\\w\\s]", "", s))
    series = series.str.strip()
    return series




## === cell 5
X_text = dataPreprocessing(train_df["full_text"])
X_test_text = dataPreprocessing(test_df["full_text"])
y = train_df["score"]




## === cell 6
X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_text, y, test_size=0.1, random_state=123, stratify=y
)




## === cell 7
vectorizer = TfidfVectorizer(
    stop_words="english",
    sublinear_tf=True,
    strip_accents="unicode",
    binary=False,
    analyzer="word",
    token_pattern=r"\w{2,}",
    ngram_range=(1, 4),  # up to 4‑grams
    norm="l2",
    use_idf=True,
    smooth_idf=True,
    max_features=500000,  # larger vocabulary
    min_df=1,  # keep rare terms
)

X_train_vec = vectorizer.fit_transform(X_train_text)
X_val_vec = vectorizer.transform(X_val_text)




## === cell 8
clf = LinearSVC(
    C=15.0,  # increased from 10.0
    dual=False,
    random_state=123,
    max_iter=40000,  # increased from 20000
    class_weight="balanced",
)
clf.fit(X_train_vec, y_train)

clf_alt = LinearSVC(
    C=5.0,
    dual=False,
    random_state=124,
    max_iter=20000,
    class_weight="balanced",
)
clf_alt.fit(X_train_vec, y_train)

val_pred = clf.predict(X_val_vec)
val_pred_alt = clf_alt.predict(X_val_vec)

val_pred_ens = np.rint((val_pred + val_pred_alt) / 2).astype(int)

mapping = {}
for pred_class in range(1, 7):
    mask = val_pred_ens == pred_class
    if mask.sum() > 0:
        avg_score = y_val[mask].mean()
        mapping[pred_class] = int(round(avg_score))
    else:
        mapping[pred_class] = pred_class

val_pred_mapped = np.vectorize(mapping.get)(val_pred_ens)

kappa_primary = cohen_kappa_score(y_val, val_pred, weights="quadratic")
kappa_ensemble = cohen_kappa_score(y_val, val_pred_ens, weights="quadratic")
kappa_mapped = cohen_kappa_score(y_val, val_pred_mapped, weights="quadratic")
print("Validation QWK (primary model):", kappa_primary)
print("Validation QWK (ensemble):", kappa_ensemble)
print("Validation QWK (ensemble + mean mapping):", kappa_mapped)




## === cell 9
X_full_vec = vectorizer.fit_transform(X_text)

clf_full = LinearSVC(
    C=15.0,  # match primary model above
    dual=False,
    random_state=123,
    max_iter=40000,
    class_weight="balanced",
)
clf_full.fit(X_full_vec, y)

clf_full_alt = LinearSVC(
    C=5.0,
    dual=False,
    random_state=124,
    max_iter=20000,
    class_weight="balanced",
)
clf_full_alt.fit(X_full_vec, y)




## === cell 10
test_vec = vectorizer.transform(X_test_text)

pred_primary = clf_full.predict(test_vec)
pred_alt = clf_full_alt.predict(test_vec)

test_predictions = np.rint((pred_primary + pred_alt) / 2).astype(int)

test_predictions = np.vectorize(mapping.get)(test_predictions)

test_predictions = np.clip(test_predictions, 1, 6)




## === cell 11
submission = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)
submission["score"] = test_predictions
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
display(submission.head())
