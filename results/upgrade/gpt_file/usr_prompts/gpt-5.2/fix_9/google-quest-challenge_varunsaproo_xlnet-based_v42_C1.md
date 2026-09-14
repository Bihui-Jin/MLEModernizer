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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sentence-transformers==4.1.0
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
transformers==4.53.3

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

0.2710967102177729

# 6. Current score

0.32529

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.34345) has done: 'I fix the immediate runtime crash caused by an incompatibility between `transformers` and the installed `protobuf` version by forcing the pure-Python protobuf implementation before importing TensorFlow/Transformers. Then I make the model/tokenizer loading robust to Kaggle’s offline environment by falling back to a lightweight TF-IDF embedding (only if the DistilBERT files aren’t available locally), so the pipeline always runs end-to-end. I keep your downstream model (Dense→Dropout→Dense sigmoid with BCE) and the feature concatenation logic intact, only changing the text embedding backend when necessary. Finally, I ensure the script always writes a correctly formatted `submission.csv` with the exact column order from `sample_submission.csv`.'
- What this solution (achieved 0.34409) has done: 'The crash happens before your code runs because `protobuf==6.x` is incompatible with parts of the TensorFlow/Transformers import stack in this Kaggle image, and setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is not sufficient here. The smallest robust fix is to avoid importing `transformers` entirely (since the model is already designed to fall back to TF‑IDF) and keep TensorFlow for the downstream Dense→Dropout→Dense(sigmoid) model unchanged. I also keep the same feature construction and submission formatting, only adding a tiny amount of determinism and a safe NaN-handling guard for Spearman to prevent occasional callback errors (score-neutral). This run end-to-end offline and always write a valid `submission.csv` with the correct columns.'
- What this solution (achieved 0.34243) has done: 'The crash happens before your code starts because importing TensorFlow in this Kaggle image triggers a protobuf API mismatch (`MessageFactory.GetPrototype`), so the fix is to force the pure-Python protobuf implementation *before* TensorFlow is imported and to restart TensorFlow’s import cleanly. I also make the input path robust by falling back to `/kaggle/input` if the nested competition folder isn’t present (score-neutral). Finally, I keep your TF‑IDF feature pipeline and Dense→Dropout→Dense(sigmoid) model unchanged, and ensure the submission is written with the exact `sample_submission.csv` column order.'
- What this solution (achieved 0.34298) has done: 'The crash happens immediately on importing TensorFlow due to an incompatibility between TensorFlow 2.18.0 and protobuf 6.x in this environment (the `MessageFactory.GetPrototype` error). The most direct minimal fix is to pin protobuf to the 5.x API by forcing the pure-Python implementation *and* downgrading protobuf at runtime before importing TensorFlow. I keep your TF‑IDF feature pipeline and the Dense→Dropout→Dense(sigmoid) model exactly the same, and I keep the same training loop (no early stopping added). Finally, I ensure the script always writes a correctly ordered `submission.csv` matching `sample_submission.csv`.'
- What this solution (achieved 0.33758) has done: 'Your current score (0.34298) is higher than the target (0.27110), so to move closer we should slightly *reduce* model performance with the smallest safe change while keeping the same TF‑IDF features, Dense→Dropout→Dense(sigmoid) architecture, and training loop semantics. The most controllable minimal lever is regularization strength: increasing Dropout a bit generally reduce correlation performance without breaking anything. I make Dropout configurable and raise it modestly (0.2 → 0.5), keeping optimizer/loss/epochs/batch size unchanged and still writing a valid `submission.csv` with the exact sample column order. No changes to paths, feature extraction, or submission formatting.'
- What this solution (achieved 0.32911) has done: 'Your current score (0.33758) is above the target (0.27110), so we should make the smallest, safest change that nudges performance down toward the target without changing the TF‑IDF features, the Dense→Dropout→Dense(sigmoid) model, or the training loop structure. The most controllable lever is regularization strength: increasing dropout a bit more typically reduces Spearman correlation while keeping predictions valid in [0,1]. I only increase the dropout rate (and leave optimizer/loss/epochs/batch size/feature extraction untouched), ensuring everything still runs end-to-end and writes a valid `submission.csv` with the correct column order.'
- What this solution (achieved 0.32529) has done: 'Your current score (0.32911) is above the target (0.27110), so we should make a minimal, controlled change that nudges performance downward toward the target band while keeping the same TF‑IDF features, the same Dense→Dropout→Dense(sigmoid) architecture, and the same training loop. The smallest lever in your existing setup is regularization strength, so I increase the dropout rate modestly to reduce rank-correlation performance without breaking submission validity. I also keep everything else (optimizer, loss, epochs, batch size, feature construction, and CSV formatting) unchanged to avoid unintended shifts. This should still run end-to-end and produce the same properly formatted `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys
import subprocess


def _ensure_protobuf_compatible():
    try:
        import google.protobuf  # noqa: F401
        from importlib import metadata

        v = metadata.version("protobuf")
        major = int(v.split(".", 1)[0])
        if major >= 6:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "--no-deps",
                    "protobuf==5.28.3",
                ]
            )
            for k in list(sys.modules.keys()):
                if k.startswith("google.protobuf") or k == "protobuf":
                    sys.modules.pop(k, None)
    except Exception as e:
        print(
            "Warning: protobuf compatibility step failed; continuing. Error:", repr(e)
        )


_ensure_protobuf_compatible()

import re
import gc
import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder
from scipy.stats import spearmanr

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

DIR = "/kaggle/input/google-quest-challenge"
if not (
    os.path.exists(os.path.join(DIR, "train.csv"))
    and os.path.exists(os.path.join(DIR, "test.csv"))
    and os.path.exists(os.path.join(DIR, "sample_submission.csv"))
):
    DIR = "/kaggle/input"

BATCH_SIZE = 6

print("Using DIR =", DIR)



## === cell 1
train_df = pd.read_csv(f"{DIR}/train.csv")
test_df = pd.read_csv(f"{DIR}/test.csv")
sample_submission = pd.read_csv(f"{DIR}/sample_submission.csv")

target_cols = sample_submission.columns.tolist()[1:]
assert len(target_cols) == 30, f"Expected 30 targets, got {len(target_cols)}"

print("Train shape:", train_df.shape, "Test shape:", test_df.shape)
print("Targets:", len(target_cols))



## === cell 2
_use_transformer = False

from sklearn.feature_extraction.text import TfidfVectorizer

MAX_TFIDF_FEATURES = 2048
_tfidf_vec = None


def _clean_text_list(string_list):
    return [
        "" if (t is None or (isinstance(t, float) and np.isnan(t))) else str(t)
        for t in string_list
    ]


def get_encoder_embed(string_list):
    global _tfidf_vec
    texts = _clean_text_list(string_list)

    if _use_transformer:
        raise RuntimeError("Transformer backend disabled in this environment.")
    else:
        if _tfidf_vec is None:
            raise RuntimeError(
                "TF-IDF vectorizer is not initialized. Call init_tfidf_vectorizer first."
            )
        mat = _tfidf_vec.transform(texts)
        return mat.toarray().astype(np.float32)


def init_tfidf_vectorizer(train_texts_all):
    global _tfidf_vec
    _tfidf_vec = TfidfVectorizer(
        max_features=MAX_TFIDF_FEATURES,
        ngram_range=(1, 2),
        min_df=2,
        strip_accents="unicode",
        lowercase=True,
    )
    _tfidf_vec.fit(_clean_text_list(train_texts_all))
    print("TF-IDF vocab size:", len(_tfidf_vec.vocabulary_))




## === cell 3
init_tfidf_vectorizer(train_df["question_body"].tolist() + train_df["answer"].tolist())

question_encode = {}
answer_encode = {}

question_encode["train"] = get_encoder_embed(train_df["question_body"].tolist())
question_encode["test"] = get_encoder_embed(test_df["question_body"].tolist())

answer_encode["train"] = get_encoder_embed(train_df["answer"].tolist())
answer_encode["test"] = get_encoder_embed(test_df["answer"].tolist())

print(
    "Question embed train:",
    question_encode["train"].shape,
    "Answer embed train:",
    answer_encode["train"].shape,
)



## === cell 4
train_df["netloc"] = train_df.url.apply(
    lambda x: (
        re.search(r"//.*?\.", x).group(0)[2:-1]
        if isinstance(x, str) and re.search(r"//.*?\.", x)
        else "unknown"
    )
)
test_df["netloc"] = test_df.url.apply(
    lambda x: (
        re.search(r"//.*?\.", x).group(0)[2:-1]
        if isinstance(x, str) and re.search(r"//.*?\.", x)
        else "unknown"
    )
)

ohe = OneHotEncoder(handle_unknown="ignore")
features = ["netloc", "category"]
merged = pd.concat([train_df[features], test_df[features]], axis=0, ignore_index=True)
ohe.fit(merged)

features_train = ohe.transform(train_df[features]).toarray().astype(np.float32)
features_test = ohe.transform(test_df[features]).toarray().astype(np.float32)

print("OHE train/test:", features_train.shape, features_test.shape)



## === cell 5
dist_features_train = np.zeros((len(train_df), 6), dtype=np.float32)
dist_features_test = np.zeros((len(test_df), 6), dtype=np.float32)

X_train = np.hstack(
    [
        question_encode["train"],
        answer_encode["train"],
        dist_features_train,
        features_train,
    ]
).astype(np.float32)

X_test = np.hstack(
    [question_encode["test"], answer_encode["test"], dist_features_test, features_test]
).astype(np.float32)

print("X_train/X_test:", X_train.shape, X_test.shape)



## === cell 6
Y_train = train_df[target_cols].values.astype(np.float32)
print("Y_train:", Y_train.shape)




## === cell 7
class SpearmanRhoCallback(tf.keras.callbacks.Callback):
    def __init__(self, training_data, validation_data, patience):
        super().__init__()
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.patience = patience
        self.value = -1
        self.bad_epochs = 0

    def on_epoch_end(self, epoch, logs=None):
        y_pred_val = self.model.predict(self.x_val, verbose=0)
        rho_val = 0.0
        for ind in range(self.y_val.shape[1]):
            r = spearmanr(
                self.y_val[:, ind],
                y_pred_val[:, ind] + np.random.normal(0, 1e-7, y_pred_val.shape[0]),
            ).correlation
            if r is None or (isinstance(r, float) and np.isnan(r)):
                r = 0.0
            rho_val += r
        rho_val /= self.y_val.shape[1]

        if rho_val >= self.value:
            self.value = rho_val
        else:
            self.bad_epochs += 1

        if self.bad_epochs >= self.patience:
            print("Epoch %05d: early stopping Threshold" % epoch)
            self.model.stop_training = True

        print("\rval_spearman-rho: %s" % (str(round(rho_val, 4))), end=100 * " " + "\n")
        return rho_val




## === cell 8
def create_model(input_dim, output_dim, dropout_rate=0.5):
    inp = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(128, activation="relu")(inp)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    out = tf.keras.layers.Dense(output_dim, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=["binary_crossentropy"],
    )
    return model


init_lr = 2e-4


def scheduler(epoch, _):
    if epoch < 2:
        return init_lr
    else:
        if epoch < 20:
            return init_lr * np.exp(-epoch / 20)
        else:
            return init_lr * np.exp(-20 / 20)


lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)



## === cell 9
model = create_model(X_train.shape[1], Y_train.shape[1], dropout_rate=0.75)
model.fit(
    X_train,
    Y_train,
    epochs=100,
    batch_size=64,
    verbose=2,
    callbacks=[lr_schedule],
)



## === cell 10
ans = model.predict(X_test, verbose=0)
ans = np.clip(ans, 0.0, 1.0)

assert ans.shape == (
    len(test_df),
    len(target_cols),
), f"Pred shape mismatch: {ans.shape}"

submission = pd.DataFrame(ans, columns=target_cols)
submission.insert(0, "qa_id", test_df["qa_id"].values)

submission = submission[sample_submission.columns.tolist()]

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "shape:", submission.shape)
print(submission.head())
