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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
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

0.75195

# 6. Current score

0.67603

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.67603) has done: 'I fix your validation/evaluation to match the competition’s quadratic weighted kappa by keeping the target as the original 1–6 score scale (right now you’re mixing label-encoded and original labels, which makes the printed kappa misleading and can hide issues). Then I make two minimal, core-logic-preserving improvements that typically raise QWK for this setup: stratified splitting (so the class distribution is stable) and a light hyperparameter calibration of the same `GradientBoostingClassifier` (more trees + smaller learning rate) to improve ordinal separation without changing the modeling approach. Finally, I ensure predictions are clipped to the valid 1–6 range and that the submission is aligned and valid.'

# 9. Code solution

## === cell 0
import nltk

try:
    _ = nltk.corpus.stopwords.words("english")
except LookupError:
    nltk.download("stopwords")



## === cell 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



## === cell 2
train = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv",
    index_col="essay_id",
)
test = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv",
    index_col="essay_id",
)



## === cell 3
train



## === cell 4
test



## === cell 5
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
_ = le.fit_transform(train["score"])  # fitted but not used as target
y = train["score"].astype(int).values



## === cell 6
y



## === cell 7
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
train["tokens"] = train["full_text"].apply(
    lambda x: [word for word in x.split() if word.isalpha()]
)
test["tokens"] = test["full_text"].apply(
    lambda x: [word for word in x.split() if word.isalpha()]
)



## === cell 8
stop_words



## === cell 9
train



## === cell 10
test




## === cell 11
class FeatureExtract:
    def __init__(self):
        pass

    def word_count(self, text):
        return len(text.split())

    def sen_count(self, text):
        return len(text.split("."))

    def ave_word_length(self, text):
        words = text.split()
        total_length = 0
        for word in words:
            total_length += len(word)
        if len(words) == 0:
            return 0
        else:
            return total_length / len(words)

    def total_stopwords(self, text):
        words = text.split()
        stopwords_ = [
            word
            for word in words
            if len(word) > 1 and word.isalpha() and word.lower() in stop_words
        ]
        return len(stopwords_)

    def count_spelling_errors(self, text):
        nlp = spacy.load("en_core_web_sm")
        spell_checker = SpellChecker()
        text = """many people have car where they live. the thing they do not know is that when you use a car alot of thing can happen like you can get in accidet or the smoke that the car has is bad to breath on if someone is walk but in vauban,germany they dont have that proble because percent of vaubans families do not own cars,and percent sold a car to move there. street parkig ,driveways and home garages are forbidden on the outskirts of freiburd that near the french and swiss borders. you probaly will not see a car in vaubans streets because they are completely "car free" but if some that lives in vauban that owns a car ownership is allowed,but there are only two places that you can park a large garages at the edge of the development,where a car owner buys a space but it not cheap to buy one they sell the space for you car for , along with a home. the vauban people completed this in ,they said that this an example of a growing trend in europe,the untile states and some where else are suburban life from auto use this is called "smart planning". the current efforts to drastically reduce greenhouse gas emissions from tailes the passengee cars are responsible for percent of greenhouse gas emissions in europe and up to percent in some car intensive in the united states. i honeslty think that good idea that they did that is vaudan because that makes cities denser and better for walking and in vauban there are , residents within a rectangular square mile. in the artical david gold berg said that "all of our development since world war has been centered on the cars,and that will have to change" and i think that was very true what david gold said because alot thing we need cars to do we can go anyway were with out cars beacuse some people are a very lazy to walk to place thats why they alot of people use car and i think that it was a good idea that that they did that in vauban so people can see how we really do not need car to go to place from place because we can walk from were we need to go or we can ride bycles with out the use of a car. it good that they are doing that if you thik about your help the earth in way and thats a very good thing to. in the united states ,the environmental protection agency is promoting what is called "car reduced"communtunties,and the legislators are starting to act,if cautiously. maany experts expect pubic transport serving suburbs to play a much larger role in a new six years federal transportation bill to approved this year. in previous bill, percent of appropriations have by law gone to highways and only percent to other transports. there many good reason why they should do this."""
        doc = nlp(text)
        tokens = [token.text for token in doc if token.is_alpha and not token.ent_type_]
        misspelled = [
            token for token in tokens if spell_checker.correction(token) != token
        ]
        return len(misspelled)

    def lexical_diversity(self, text):
        words = text.split()
        if len(words) == 0:
            return 0
        return len(set(words)) / len(words)

    def sentiment(self, text):
        if len(text) == 0:
            return 0.0
        return sum(ord(c) for c in text) / len(text)

    def extract_features(self, text):
        features = {
            "word_count": self.word_count(text),
            "sen_count": self.sen_count(text),
            "ave_word_length": self.ave_word_length(text),
            "total_stopwords": self.total_stopwords(text),
            "count_spelling_errors": self.count_spelling_errors(text),
            "lexical_diversity": self.lexical_diversity(text),
            "sentiment": self.sentiment(text),
        }
        return features




## === cell 12
def insert_features(df, text):
    extractor = FeatureExtract()
    for feature in [
        "word_count",
        "sen_count",
        "ave_word_length",
        "total_stopwords",
        "lexical_diversity",
        "sentiment",
    ]:
        df[feature] = df[text].apply(lambda x: getattr(extractor, feature)(x))


insert_features(train, "full_text")
insert_features(test, "full_text")



## === cell 13
train



## === cell 14
test



## === cell 15
features = [
    "word_count",
    "sen_count",
    "ave_word_length",
    "total_stopwords",
    "lexical_diversity",
    "sentiment",
]
X = train[features]
X_test = test[features]



## === cell 16
print("Value of X:")
print(X.head())



## === cell 17
print("\nValue of X_test:")
print(X_test.head())



## === cell 18
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



## === cell 19
print("Value of X_train:")
print(X_train.head())



## === cell 20
print("\nValue of X_valid:")
print(X_valid.head())



## === cell 21
print("\nValue of y_train:")
print(y_train)



## === cell 22
print("\nValue of y_valid:")
print(y_valid)



## === cell 23
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
)



## === cell 24
model.fit(X_train, y_train)



## === cell 25
y_pred = model.predict(X_valid)



## === cell 26
print("Predicted values on the validation set:")
print(y_pred)



## === cell 27
y_pred = np.clip(y_pred.astype(int), 1, 6)



## === cell 28
from sklearn.metrics import cohen_kappa_score, accuracy_score

kappa_score = cohen_kappa_score(y_valid, y_pred, weights="quadratic")
print("Cohen's Kappa Score:", kappa_score)



## === cell 29
accuracy = accuracy_score(y_valid, y_pred)
print("Accuracy:", accuracy)



## === cell 30
plt.figure(figsize=(8, 6))
plt.scatter(y_valid, y_pred, color="blue", alpha=0.3, s=10)
plt.plot(
    [min(y_valid), max(y_valid)],
    [min(y_valid), max(y_valid)],
    color="red",
    linestyle="--",
)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()



## === cell 31
plt.figure(figsize=(8, 6))
plt.hist(y_valid, bins=np.arange(0.5, 7.5, 1), alpha=0.5, label="Actual", color="blue")
plt.hist(
    y_pred, bins=np.arange(0.5, 7.5, 1), alpha=0.5, label="Predicted", color="orange"
)
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Distribution of Actual and Predicted Scores")
plt.legend()
plt.grid(True)
plt.show()



## === cell 32
predictions = model.predict(X_test)



## === cell 33
print("Predicted values on the test set:")
print(predictions[:20])



## === cell 34
predictions = np.clip(predictions.astype(int), 1, 6)



## === cell 35
predictions



## === cell 36
score_counts = pd.Series(predictions).value_counts().sort_index()
plt.figure(figsize=(8, 6))
plt.pie(score_counts, labels=score_counts.index, autopct="%1.1f%%", startangle=140)
plt.title("Distribution of Predicted Scores")
plt.axis("equal")
plt.show()



## === cell 37
submission = pd.DataFrame(
    {"essay_id": test.index.astype(str), "score": predictions.astype(int)}
)
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
