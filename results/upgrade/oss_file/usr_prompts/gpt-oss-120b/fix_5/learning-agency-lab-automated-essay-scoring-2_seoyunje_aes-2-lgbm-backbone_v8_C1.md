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

cudf-polars-cu12==25.6.0
geopandas==0.14.4
lightgbm==4.6.0
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
polars==1.25.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tqdm==4.67.1

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

0.813007506495721

# 6. Current score

None

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = None
    LOAD_FEATURES_FROM = None
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"




## === cell 1
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score
import lightgbm as lgb

try:
    from spellchecker import SpellChecker

    spell = SpellChecker()

    def count_misspelled_words(text):
        """Return the number of misspelled words in `text`."""
        return len(spell.unknown(text.split()))

except Exception:  # pragma: no cover

    def count_misspelled_words(text):
        """Fallback: assume no misspellings (cost‑free)."""
        return 0




## === cell 2
train_path = os.path.join(CFG.BASE_PATH, "train.csv")
test_path = os.path.join(CFG.BASE_PATH, "test.csv")
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)


def basic_text_features(df):
    df_feat = pd.DataFrame()
    df_feat["essay_id"] = df["essay_id"]
    df_feat["char_len"] = df["full_text"].str.len()
    df_feat["word_cnt"] = df["full_text"].str.split().apply(len)
    df_feat["misspell_cnt"] = df["full_text"].apply(count_misspelled_words)
    return df_feat


tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
all_text = pd.concat([df_train["full_text"], df_test["full_text"]])
tfidf_matrix = tfidf.fit_transform(all_text)

train_tfidf = tfidf_matrix[: len(df_train)]
test_tfidf = tfidf_matrix[len(df_train) :]

tfidf_feat_names = [f"tfidf_{c}" for c in tfidf.get_feature_names_out()]

train_tfidf_df = pd.DataFrame(train_tfidf.toarray(), columns=tfidf_feat_names)
train_tfidf_df["essay_id"] = df_train["essay_id"].values

test_tfidf_df = pd.DataFrame(test_tfidf.toarray(), columns=tfidf_feat_names)
test_tfidf_df["essay_id"] = df_test["essay_id"].values

train_feats = basic_text_features(df_train).merge(
    train_tfidf_df, on="essay_id", how="left"
)
test_feats = basic_text_features(df_test).merge(
    test_tfidf_df, on="essay_id", how="left"
)

train_feats["score"] = df_train["score"].values

FEATURES = [col for col in train_feats.columns if col not in ("essay_id", "score")]




## === cell 3
if CFG.LOAD_FEATURES_FROM is None:
    print("Saving train_feats.csv")
    train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)
else:
    print("Loading train_feats.csv")
    train_feats = pd.read_csv(CFG.LOAD_FEATURES_FROM)




## === cell 4
def lightgbm_train():
    """Train a LightGBM model on the prepared features."""
    X = train_feats[FEATURES]
    y = train_feats["score"]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=CFG.SEED, stratify=y
    )
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=800,
        learning_rate=0.05,
        random_state=CFG.SEED,
        n_jobs=-1,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
    )
    val_pred = np.rint(model.predict(X_val)).clip(1, 6)
    qwk = cohen_kappa_score(y_val, val_pred, weights="quadratic")
    print(f"Validation QWK: {qwk:.5f}")

    model_path = f"LGB_v{CFG.VER}_f0.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model, qwk


if CFG.LOAD_MODELS_FROM:
    model_path = os.path.join(CFG.LOAD_MODELS_FROM, f"LGB_v{CFG.VER}_f0.pkl")
else:
    model_path = f"LGB_v{CFG.VER}_f0.pkl"

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Model loaded from disk.")
else:
    print("Model not found – training now.")
    model, _ = lightgbm_train()

df_importance = pd.DataFrame(
    {
        "features_name": FEATURES,
        "importance": model.feature_importances_,
    }
)
df_importance = df_importance.sort_values(by="importance", ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(
    x=df_importance["features_name"].head(30),
    height=df_importance["importance"].head(30),
    color="pink",
    edgecolor="black",
)
plt.title("Top 30 Feature Importances (LightGBM)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()




## === cell 5
preds = model.predict(test_feats[FEATURES])
preds = np.rint(preds).clip(1, 6)  # ensure integer 1‑6

sub = pd.DataFrame({"essay_id": df_test["essay_id"], "score": preds.astype(int)})
sub.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv – shape:", sub.shape)
sub.head()
