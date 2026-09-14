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

0.59127

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.47987) has done: 'I correct the label selection, replace the incompatible scaler‑SVC pipeline with a sparse‑compatible LinearSVC, and ensure the model is fitted before any predictions. These fixes resolve the AttributeErrors, allow successful training/evaluation, and produce a proper `submission.csv` with the required columns.'
- What this solution (achieved 0.61405) has done: 'I slightly adjust the text vectorizer to use true TF‑IDF weighting, include bigrams, lower the document‑frequency cut‑off, and remove the binary flag so word counts are retained. I also give the LinearSVC a balanced class weight and a modestly larger C. These targeted tweaks keep the original pipeline but should improve the quadratic weighted kappa, moving the score closer to the target.'
- What this solution (achieved 0.61021) has done: 'I slightly broaden the TF‑IDF representation (remove the hard feature cap and allow rarer terms) and give the LinearSVC a larger regularisation parameter C, which usually improves the quadratic weighted‑kappa for this dataset. These are minimal tweaks that keep the original pipeline intact while moving the validation score upward toward the target.'
- What this solution (achieved 0.58895) has done: 'I slightly broaden the TF‑IDF representation by allowing single‑occurrence terms (set `min_df=1`) and increase the regularization strength of the LinearSVC (`C=10.0`). These minimal tweaks keep the original pipeline but give the model more expressive features and a slightly looser margin, which should raise the quadratic weighted kappa toward the target without altering the overall architecture.'
- What this solution (achieved 0.6296) has done: 'I keep the overall LinearSVC + TF‑IDF pipeline but adjust the text vectorizer to capture more signal (no stop‑word removal and allowing single‑character tokens) and raise the regularisation parameter slightly. These small tweaks usually improve the discriminative power on this dataset and raise the quadratic weighted‑kappa without altering the core model architecture.'
- What this solution (achieved 0.59826) has done: 'I slightly broaden the TF‑IDF representation by adding trigrams (ngram_range = (1, 3)) to capture more contextual information and increase the LinearSVC regularisation strength (C = 30.0). These modest adjustments keep the original pipeline intact while giving the model a bit more expressive power, which is expected to raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.6014) has done: 'I adjust the TF‑IDF vectorizer to reduce noise and improve representation (limit very rare terms, cap feature count, and restrict n‑grams to bi‑grams) and increase the SVC regularisation strength slightly (C = 60). These modest tweaks keep the original LinearSVC + TF‑IDF pipeline intact while aiming to raise the quadratic weighted kappa toward the target. I also clip the final predictions to the valid score range (1‑6) before writing the submission.'
- What this solution (achieved 0.59127) has done: 'I keep the LinearSVC + TF‑IDF pipeline but give the text vectoriser a richer representation (add trigrams, keep single‑occurrence terms and raise the feature cap) and lower the SVC regularisation strength slightly (C = 30). These minimal tweaks preserve the original model while providing more signal, which should push the quadratic weighted kappa closer to the target score.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC  # LinearSVC works with sparse matrices




## === cell 1
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_df1 = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)




## === cell 2
train_df1["score"].dtypes




## === cell 3
train_df1.head(10)




## === cell 4
train_df1["score"].value_counts()




## === cell 5
submission = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)
submission




## === cell 6
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
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
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
    "you'll": "you you will",
    "you'll've": "you you will have",
    "you're": "you are",
    "you've": "you have",
}




## === cell 7
c_re = re.compile("(%s)" % "|".join(cList.keys()))




## === cell 8
def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]

    return c_re.sub(replace, text)




## === cell 9
def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)




## === cell 10
def dataPreprocessing(x):
    x = x.apply(lambda s: s.lower())
    x = x.apply(removeHTML)
    x = x.apply(lambda s: re.sub("@\\w+", "", s))
    x = x.apply(lambda s: re.sub("'\\d+", "", s))
    x = x.apply(lambda s: re.sub("\\d+", "", s))
    x = x.apply(lambda s: re.sub("http\\w+", "", s))
    x = x.apply(lambda s: re.sub(r"\\s+", " ", s))
    x = x.apply(expandContractions)
    x = x.apply(lambda s: re.sub(r"\\.+", ".", s))
    x = x.apply(lambda s: re.sub(r"\\,+", ",", s))
    x = x.apply(lambda s: re.sub("\\n", "", s))
    x = x.apply(lambda s: re.sub("[^\\w\\s]", "", s))
    x = x.apply(lambda s: s.strip())
    return x




## === cell 11
x = dataPreprocessing(train_df1["full_text"])




## === cell 12
test_df1 = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)




## === cell 13
x0 = dataPreprocessing(test_df1["full_text"])




## === cell 14
y = train_df1["score"]




## === cell 15
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=123, stratify=y  # preserve class distribution
)




## === cell 16
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)




## === cell 17
text_vectorizer = TfidfVectorizer(
    stop_words=None,
    sublinear_tf=True,
    strip_accents="unicode",
    binary=False,
    analyzer="word",
    token_pattern=r"\w{1,}",
    ngram_range=(1, 3),  # add trigrams
    norm="l2",
    use_idf=True,
    smooth_idf=True,
    max_features=120000,  # larger feature space
    min_df=1,  # keep rare words
)




## === cell 18
X_train_features = text_vectorizer.fit_transform(X_train)




## === cell 19
clf = LinearSVC(C=30.0, dual=False, random_state=123, tol=1e-5, class_weight="balanced")
clf.fit(X_train_features, y_train)

test_features = text_vectorizer.transform(X_test)
y_pred = clf.predict(test_features)




## === cell 20
print(confusion_matrix(y_test.values, y_pred))




## === cell 21
print(classification_report(y_test.values, y_pred))




## === cell 22
kappa = cohen_kappa_score(y_test.values, y_pred, weights="quadratic")
print("Cohen's kappa score:", kappa)




## === cell 23
st_features = text_vectorizer.transform(x0)
test_predictions = clf.predict(st_features)




## === cell 24
submission = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)
submission["score"] = np.clip(test_predictions.astype(int), 1, 6)
submission.to_csv("submission.csv", index=False)
display(submission)
