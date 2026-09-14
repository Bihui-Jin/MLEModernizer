# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Given questions and answers from various StackExchange properties, predict target values of 30 labels for each question-answer pair.

## Metric
Mean column-wise Spearman's correlation coefficient. The Spearman's rank correlation is computed for each target column, and the mean of these values is calculated for the submission score.

## Submission Format
For each qa_id in the test set, you must predict a probability for each target variable. The predictions should be in the range [0,1]. The file should contain a header and have the following format:

```
qa_id,question_asker_intent_understanding,...,answer_well_written
6,0.0,...,0.5
8,0.5,...,0.1
18,1.0,...,0.0
etc.
```

## Dataset
The list of 30 target labels are the same as the column names in the `sample_submission.csv` file. Target labels with the prefix `question_` relate to the `question_title` and/or `question_body` features in the data. Target labels with the prefix `answer_` relate to the `answer` feature.

Target labels are aggregated from multiple raters, and can have continuous values in the range `[0,1]`. Therefore, predictions must also be in that range.

- **train.csv** - the training data (target labels are the last 30 columns)
- **test.csv** - the test set (you must predict 30 labels for each test set row)
- **sample_submission.csv** - a sample submission file in the correct format; column names are the 30 target labels

# 2. Python version

3.8

# 3. Installed packages

cufflinks==0.17.3
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
plotly==5.24.1
plotly-express==0.4.1
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        input/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        working/
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
```

-> data/google-quest-challenge/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/google-quest-challenge/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/google-quest-challenge/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> data/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.11905

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.18118) has done: 'I fix the import-time crash caused by an incompatibility between `protobuf==6.x` and some TensorFlow/Keras dependencies by forcing Python protobuf parsing (a standard Kaggle workaround) before importing TensorFlow. I also remove notebook-only plotting initializations that can trigger extra side effects in script mode, since they aren’t used for training/inference and can break execution. Then I make sure the code actually trains a minimal text model end-to-end and writes a valid `submission.csv` with the exact `sample_submission.csv` columns and `[0,1]`-clipped predictions. These changes keep the core idea (TF-IDF + SVD + multi-output regression) while ensuring the pipeline runs and produces a valid submission file.'
- What this solution (achieved 0.1631) has done: 'I fix the import-time crash caused by a protobuf/TensorFlow incompatibility by pinning protobuf to the Python implementation *and* forcing the pure-Python backend before TensorFlow imports (this avoids the `MessageFactory.GetPrototype` error). I keep the modeling pipeline (TF-IDF → SVD → dense sigmoid multi-output regressor) unchanged, only ensuring the environment is stable and deterministic. Since your current score (0.18118) is already above the target (0.11905) and within the ±10% tolerance band, I not make any score-seeking changes—just correctness and runtime stability. The script still write a valid `submission.csv` matching `sample_submission.csv` columns and clip predictions into `[0,1]`.'
- What this solution (achieved 0.17195) has done: 'I fix the import-time crash caused by the protobuf 6.x / TensorFlow interaction by avoiding the problematic `google.protobuf.internal.api_implementation` call and forcing the pure-Python protobuf backend in a safer way before importing TensorFlow. The rest of the pipeline (TF‑IDF → SVD → scaling → dense sigmoid multi-output regressor trained with MSE) stay unchanged to preserve core logic and keep the score from drifting further away from your target (you’re currently above target and outside the ±10% band). I also keep deterministic seeds and ensure the submission is written with the exact `sample_submission.csv` column order and `[0,1]` clipping.'
- What this solution (achieved 0.18332) has done: 'The crash happens before training because TensorFlow (via Keras/TF-Hub internals) pulls in protobuf APIs that are incompatible with protobuf 6.x unless we force the pure‑Python protobuf backend *and* disable the C++ implementation early enough. I fix this by setting both the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` env var and the protobuf internal implementation type before importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error. Since your current score (0.17195) is above the target (0.11905), I not change the model/pipeline in any way that would intentionally improve score; the rest of the code stays the same aside from stability-only import ordering. The script then run end-to-end and write a valid `submission.csv` with the exact `sample_submission.csv` column order and predictions clipped to `[0,1]`.'
- What this solution (achieved 0.17157) has done: 'I fix the import-time crash caused by the protobuf 6.x incompatibility by preventing TensorFlow from importing the standalone `protobuf` package (which triggers the missing `MessageFactory.GetPrototype` API) and instead forcing TensorFlow to use its bundled protobuf implementation. This is done by temporarily removing `google/protobuf` from `sys.path` only during the TensorFlow import, then restoring it immediately after so the rest of the environment remains normal. No model/data logic is changed, so the score behavior should remain essentially the same while the pipeline becomes reliably runnable end-to-end. The script still train the same TF‑IDF→SVD→Dense-sigmoid model and write a valid `submission.csv` with the exact `sample_submission.csv` column order.'
- What this solution (achieved 0.18337) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing protobuf to use the pure-Python implementation *and* pre-importing `google.protobuf` before importing TensorFlow (this is the most reliable ordering under protobuf 6.x). I remove the brittle `sys.path` surgery (it doesn’t reliably prevent the protobuf mismatch and is causing the crash) while keeping the model/pipeline and training logic unchanged. Since your current score (0.17157) is already above the target (0.11905) and you did not ask to degrade performance, I not make any score-tuning changes—only stability/correctness fixes. The script still train end-to-end and write a valid `submission.csv` with the exact `sample_submission.csv` column order and predictions clipped to `[0,1]`.'
- What this solution (achieved 0.1801) has done: 'We need to fix the TensorFlow import crash caused by the protobuf 6.x `MessageFactory.GetPrototype` incompatibility. The most reliable minimal fix in Kaggle script environments is to force the pure-Python protobuf backend *and* ensure it is selected before any TensorFlow-related imports by setting env vars and importing `google.protobuf` after setting them; additionally, we avoid importing TensorFlow at all until after data/feature prep so any latent protobuf initialization happens as late as possible. No modeling/feature logic is changed; only import ordering and a small safety fallback are added so the notebook runs end-to-end and still writes a valid `submission.csv` with correct columns and `[0,1]` clipping. This should keep score behavior essentially unchanged while eliminating the runtime error.'
- What this solution (achieved 0.16173) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory.GetPrototype`) by forcing the pure‑Python protobuf backend *and* explicitly disabling the C++ protobuf implementation before importing TensorFlow. This is a minimal, stability-only change that preserves your exact TF‑IDF → SVD → scaler → dense-sigmoid multi-output regressor pipeline and training loop. I also keep submission column order identical to `sample_submission.csv` and ensure predictions are clipped to `[0,1]` so the file is always valid. Since your current score (0.1801) is above the target (0.11905), I not make any score-improving changes—only runtime correctness.'
- What this solution (achieved 0.19681) has done: 'We fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf backend *before any protobuf/TensorFlow-related imports* and by importing `google.protobuf` immediately after setting env vars (this ordering is the reliable workaround for the `MessageFactory.GetPrototype` error under protobuf 6.x). This is a stability-only change: the TF‑IDF → SVD → scaling → dense-sigmoid multi-output regressor and its training/inference stay the same, so score behavior should remain essentially unchanged (and still above your target band). We also keep the submission column order exactly matching `sample_submission.csv` and ensure predictions are clipped to `[0,1]`. The script run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.17034) has done: 'I fix the TensorFlow import-time crash caused by the protobuf 6.x incompatibility (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf backend and explicitly disabling the C++ implementation before importing TensorFlow/Keras. This change is isolated to the import section and keeps your TF‑IDF → SVD → scaling → dense-sigmoid multi-output regressor pipeline unchanged, so score behavior should remain essentially the same (and since you’re already above the target, we avoid any score-seeking modifications). I also ensure the script always writes a valid `submission.csv` with the exact `sample_submission.csv` column order and predictions clipped into `[0,1]`.'
- What this solution (achieved 0.17385) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing protobuf to use the pure-Python implementation *before any protobuf/TensorFlow import happens*, and by eagerly importing `google.protobuf` right after setting the env vars (this ordering is the most reliable workaround under protobuf 6.x). This is a stability-only change: the TF‑IDF → SVD → scaling → dense-sigmoid multi-output regressor pipeline, training loop, and inference remain identical, so score behavior should stay essentially unchanged (and since you’re already above the target, we avoid score-seeking tweaks). We also keep the submission column order exactly matching `sample_submission.csv` and ensure a valid `submission.csv` is always written.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import google.protobuf  # noqa: F401

import gc
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)



## === cell 1
BASE = "/kaggle/input/google-quest-challenge"
if not os.path.exists(BASE):
    BASE = "/kaggle/data/google-quest-challenge"  # fallback for the provided environment mirror

train_path = os.path.join(BASE, "train.csv")
test_path = os.path.join(BASE, "test.csv")
sample_path = os.path.join(BASE, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

target_cols = [c for c in sample.columns if c != "qa_id"]
assert len(target_cols) == 30, f"Expected 30 targets, got {len(target_cols)}"

print("train:", train.shape, "test:", test.shape, "sample:", sample.shape)
print("targets[0:5]:", target_cols[:5])




## === cell 2
def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_text(df):
    qt = df["question_title"].map(clean_text)
    qb = df["question_body"].map(clean_text)
    an = df["answer"].map(clean_text)
    return (qt + " " + qb + " " + an).astype(str)


x_train_text = build_text(train)
x_test_text = build_text(test)

y = train[target_cols].astype(np.float32).values

print("Text example:", x_train_text.iloc[0][:200])
print("y shape:", y.shape)



## === cell 3
max_features = 50000
svd_components = 256

tfidf = TfidfVectorizer(
    max_features=max_features,
    ngram_range=(1, 2),
    min_df=2,
    strip_accents="unicode",
)

X_train_tfidf = tfidf.fit_transform(x_train_text)
X_test_tfidf = tfidf.transform(x_test_text)

svd = TruncatedSVD(n_components=svd_components, random_state=42)
X_train = svd.fit_transform(X_train_tfidf)
X_test = svd.transform(X_test_tfidf)

scaler = StandardScaler(with_mean=True, with_std=True)
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

print("X_train:", X_train.shape, "X_test:", X_test.shape)

del X_train_tfidf, X_test_tfidf, x_train_text, x_test_text
gc.collect()



## === cell 4
import sys

_saved_sys_path = list(sys.path)
try:
    filtered = []
    for p in sys.path:
        p_low = (p or "").lower()
        if ("site-packages" in p_low) or ("dist-packages" in p_low):
            continue
        filtered.append(p)
    sys.path = filtered

    import tensorflow as tf
    from tensorflow import keras
finally:
    sys.path = _saved_sys_path

print("TF:", tf.__version__)

tf.random.set_seed(42)
np.random.seed(42)

inputs = keras.Input(shape=(X_train.shape[1],), name="svd_features")
x = keras.layers.Dense(256, activation="relu")(inputs)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(128, activation="relu")(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(len(target_cols), activation="sigmoid", name="targets")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
)

model.summary()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3667795582.py in <cell line: 0>()
     16     sys.path = filtered
     17 
---> 18     import tensorflow as tf
     19     from tensorflow import keras
     20 finally:

ModuleNotFoundError: No module named 'tensorflow'

## === cell 5
batch_size = 256
epochs = 3

history = model.fit(
    X_train,
    y,
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4108402150.py in <cell line: 0>()
      2 epochs = 3
      3 
----> 4 history = model.fit(
      5     X_train,
      6     y,

NameError: name 'model' is not defined

## === cell 6
pred = model.predict(X_test, batch_size=1024, verbose=0)
pred = np.clip(pred, 0.0, 1.0)

submission = pd.DataFrame(pred, columns=target_cols)
submission.insert(0, "qa_id", test["qa_id"].values)

submission = submission[sample.columns]

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path, "shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1104622540.py in <cell line: 0>()
----> 1 pred = model.predict(X_test, batch_size=1024, verbose=0)
      2 pred = np.clip(pred, 0.0, 1.0)
      3 
      4 submission = pd.DataFrame(pred, columns=target_cols)
      5 submission.insert(0, "qa_id", test["qa_id"].values)

NameError: name 'model' is not defined
