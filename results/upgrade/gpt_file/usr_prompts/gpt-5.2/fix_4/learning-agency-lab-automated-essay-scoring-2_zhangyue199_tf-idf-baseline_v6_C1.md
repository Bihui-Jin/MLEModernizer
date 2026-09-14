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

0.48775

# 6. Current score

0.62622

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.65334) has done: 'Your current score (0.54935) is better than the target (0.48775), so the goal is to *reduce* performance slightly into the ±10% target band (0.43898–0.53653) with the smallest, safest change. The minimal way to do this without changing model architecture/training logic is to slightly reduce TF‑IDF expressiveness by capping the vocabulary size and dropping very rare terms, which typically lowers QWK a bit while keeping semantics intact. I keep the same SGD+soft-voting setup and the same argmax-over-probabilities decoding, and still write a valid `submission.csv`. I’m also fixing the notebook `%%time` magic so the script runs as a plain `.py` cell runner.'
- What this solution (achieved 0.64154) has done: 'Your current score (0.65334) is above the target (0.48775), so we should *slightly reduce* model performance to move closer to the ±10% target band (0.43898–0.53653) with the smallest, safest change. The most minimal way (without touching the model/training loop) is to further reduce TF‑IDF expressiveness by tightening `max_features` and increasing `min_df`, which typically lowers QWK while keeping the same overall approach and semantics. I keep the exact same SGD+soft-voting classifier and the same argmax decoding, and ensure the script still writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.62622) has done: 'Your current score (0.64154) is above the target (0.48775), so the goal is to *reduce* performance into the ±10% band (0.43898–0.53653) with the smallest, safest change. Without changing the model/training/prediction semantics, the cleanest lever is to further constrain TF‑IDF so it captures less signal (fewer/rarer n-grams), which typically lowers QWK. I keep the same SGD + soft-voting setup and the same argmax-over-probabilities decoding, and still write a valid `submission.csv`. The only functional change is tightening `max_features` and increasing `min_df` in `TfidfVectorizer`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import cohen_kappa_score
import gc



## === cell 1
SEED = 1124
ONLINE = True
CV = 3



## === cell 2
data = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)




## === cell 3
def init_model():
    clf_sgd = SGDClassifier(
        max_iter=10000, tol=1e-4, loss="modified_huber", random_state=SEED
    )

    ensemble = VotingClassifier(
        estimators=[
            ("sgd", clf_sgd),  # core logic unchanged
        ],
        voting="soft",
        n_jobs=-1,
    )
    return ensemble




## === cell 4
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    lowercase=False,
    sublinear_tf=True,
    analyzer="word",
    strip_accents="unicode",
    stop_words="english",
    max_features=20000,  # tighter cap than 50k -> usually lowers QWK
    min_df=20,  # drop more rare terms -> usually lowers QWK
)

tf_data = vectorizer.fit_transform(data["full_text"])



## === cell 5
if not ONLINE:
    metrics = pd.DataFrame(
        np.zeros((1, CV)), columns=[f"kappa_{i}" for i in range(1, CV + 1)]
    )
    kf = KFold(n_splits=CV, shuffle=True, random_state=SEED)

    for i, (train_idx, val_idx) in enumerate(kf.split(tf_data)):
        print("Training fold", i + 1)
        train_X = tf_data[train_idx]
        train_y = data["score"].iloc[train_idx]
        val_X = tf_data[val_idx]
        val_y = data["score"].iloc[val_idx]

        model = init_model()
        model.fit(train_X, train_y)
        preds = np.argmax(model.predict_proba(val_X), axis=1) + 1

        kappa_score = cohen_kappa_score(val_y, preds, weights="quadratic")
        metrics.loc[0, f"kappa_{i + 1}"] = kappa_score

    metrics["kappa_mean"] = metrics.mean(axis=1)
    print(metrics)



## === cell 6
test = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
sub = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)



## === cell 7
if ONLINE:
    print("Online submission")
    tf_test = vectorizer.transform(test["full_text"])
    model = init_model()
    model.fit(tf_data, data["score"])

    print(gc.collect())
    preds = np.argmax(model.predict_proba(tf_test), axis=1) + 1

    sub["score"] = preds
    sub.to_csv("submission.csv", index=False)
    print(sub.head())
