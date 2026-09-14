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

0.3173781628771867

# 6. Current score

0.37518

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.37578) has done: 'I fix the protobuf/transformers import crash and the missing tokenizer/transformer that prevents any embeddings from being computed, by removing the incompatible `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override and switching to a locally-available TF-Hub Universal Sentence Encoder so the notebook works fully offline in Kaggle. I also correct the `DIR` path to the actual dataset location under `/kaggle/input/` so the CSVs load reliably. The rest of the pipeline (feature construction, dense+dropout+sigmoid model, KFold training with Spearman callback, and submission writing) is kept the same. Finally, I ensure the submission uses the test `qa_id` order (sample submission has only 608 rows) so the output CSV has the correct 19,550 rows and 31 columns.'
- What this solution (achieved 0.37278) has done: 'I fix the immediate runtime crash caused by importing `transformers`, which is triggering a protobuf incompatibility (`MessageFactory.GetPrototype`) in this Kaggle environment. Since `transformers` is not used after switching embeddings to TF-Hub USE, removing that import is the smallest score-neutral change that unblocks execution while preserving the rest of the pipeline unchanged. I also make the TF-Hub load use the local cached module if available (and fall back to the URL) to keep it offline-safe without changing features or training logic. The script still train the same 5-fold dense+dropout+sigmoid model and write a valid `submission.csv` with 19,550 rows and the 30 required target columns.'
- What this solution (achieved 0.37124) has done: 'The crash comes from a protobuf/TensorFlow Hub compatibility issue during `import tensorflow_hub as hub`, which triggers `MessageFactory.GetPrototype` on this runtime; the smallest fix is to avoid TF-Hub entirely and instead compute text embeddings with the already-installed `sentence-transformers` (no `transformers` import needed). This keeps the same overall pipeline structure: build question/answer embeddings, add distance and OHE features, train the same Dense+Dropout+Sigmoid model with the same KFold loop and Spearman callback, then write `submission.csv` with 19,550 rows and the 30 target columns. I also keep predictions clipped to `[0,1]` and preserve the test `qa_id` order to ensure a valid submission. No changes are made to the model architecture or training loop beyond swapping the embedding backend to unblock execution.'
- What this solution (achieved 0.37524) has done: 'I fix two root-cause crashes that prevent any submission from being generated: the wrong dataset path (causing `FileNotFoundError`) and the `protobuf`/`sentence-transformers` import failure (causing `MessageFactory.GetPrototype`). To keep the core pipeline identical (same feature construction, same dense+dropout+sigmoid model, same KFold training loop and loss), I only swap the embedding backend to a TF‑IDF based embedder when `SentenceTransformer` cannot be imported, preserving shapes and downstream logic. I also make the input directory auto-resolve to the existing `/kaggle/input/google-quest-challenge/` layout so CSVs load reliably. Finally, I ensure the produced `submission.csv` has 19,550 rows (matching test) and the exact 31 columns (qa_id + 30 targets) with predictions clipped to `[0,1]`.'
- What this solution (achieved 0.37563) has done: 'I fix the protobuf-related crash happening at `import tensorflow as tf` by forcing the pure-Python protobuf implementation before TensorFlow is imported, which avoids the missing `MessageFactory.GetPrototype` attribute in this environment. I keep the embedding fallback and the dense+dropout+sigmoid KFold training loop unchanged, only making the import order/env handling robust so the pipeline runs end-to-end. I also add a small Spearman callback safeguard for rare constant-target columns (avoids NaN correlations) without changing training semantics, and keep the submission writing exactly in the required 19,550-row format.'
- What this solution (achieved 0.37465) has done: 'I fix the immediate protobuf/TensorFlow import crash by removing the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override (it’s causing the `MessageFactory.GetPrototype` AttributeError in this environment) and instead force TensorFlow to use the Python protobuf implementation via `TF_USE_C_API_PROTOBUF=0` before importing TensorFlow. This is a minimal stability fix that preserves your existing feature pipeline and KFold dense+dropout+sigmoid training loop unchanged. Since your current score (0.37563) is already above the target (0.31738) and within the ±10% tolerance band, I won’t make any score-changing modeling/calibration adjustments—only correctness/stability changes. The script run end-to-end and still write `submission.csv` with 19,550 rows and the required 31 columns.'
- What this solution (achieved 0.37463) has done: 'I fix the protobuf/TensorFlow import crash that stops execution by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow (this is the direct cause of the `MessageFactory.GetPrototype` error in this environment). I keep the rest of the pipeline (feature building, model, KFold training, prediction averaging, clipping, and submission writing) identical to avoid unnecessary score changes since your current score is already within the ±10% target band. I also remove the now-harmful `TF_USE_C_API_PROTOBUF=0` override to prevent re-triggering the same protobuf path. The script run end-to-end and write a valid `submission.csv` with 19,550 rows and the required 31 columns.'
- What this solution (achieved 0.37418) has done: 'I fix the TensorFlow/protobuf import crash by removing the problematic `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override and instead forcing TensorFlow to use the pure-Python protobuf backend via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` **only when it is safe** is not true here; the stable fix in this environment is to set `TF_USE_C_API_PROTOBUF=0` before importing TensorFlow and to avoid overriding protobuf implementation. I also renumber the cells to start from 1 (your current script starts at cell 0) while keeping the code order and core modeling/training logic identical. No score-tuning changes are introduced because your current score (0.37463) is already above the target (0.31738) and within the ±10% tolerance band; the edits are stability-only so it runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.37437) has done: 'We fix the runtime crash that happens when importing TensorFlow due to an incompatible protobuf C++ implementation in this environment by forcing the pure-Python protobuf backend *before* TensorFlow is imported (and not using the conflicting `TF_USE_C_API_PROTOBUF` override). This is a stability-only change that preserves your existing feature pipeline, model, KFold training loop, and submission formatting, so the score should remain in the same band (and you’re already within ±10% of the target). We also keep the directory auto-detection intact and ensure the notebook still writes a valid `submission.csv` with 19,550 rows and 31 columns. No modeling/calibration changes are introduced.'
- What this solution (achieved 0.37553) has done: 'I fix the TensorFlow/protobuf import crash by removing the problematic `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override that triggers `MessageFactory.GetPrototype` in this environment, and instead set `TF_USE_C_API_PROTOBUF=0` before importing TensorFlow (a stability-only change). I keep the rest of the pipeline (embeddings backend selection, feature construction, dense+dropout+sigmoid model, 5-fold training loop, and submission formatting) identical to avoid unnecessary score shifts since your current score is already above the target and within the ±10% tolerance band. I also keep the directory auto-detection and ensure the script runs end-to-end to write a valid `submission.csv` with 19,550 rows and the required 31 columns.'
- What this solution (achieved 0.37518) has done: 'I fix the TensorFlow/protobuf import crash by changing the environment variable handling to force the pure-Python protobuf implementation (the only reliably compatible option in this environment) and removing the conflicting `TF_USE_C_API_PROTOBUF` override. This is a stability-only change that keeps your embedding backend, feature construction, model architecture, training loop, and submission formatting identical, so your score should remain in the same range (already within the ±10% band around the target). I also renumber cells to start at 1 to match the required format, without altering execution order. The script then run end-to-end and write a valid `submission.csv` with 19,550 rows and the required 31 columns.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import os

_CANDIDATE_DIRS = [
    "/kaggle/input/google-quest-challenge/google-quest-challenge",
    "/kaggle/input/google-quest-challenge",
]
DIR = None
for _d in _CANDIDATE_DIRS:
    if os.path.exists(os.path.join(_d, "train.csv")):
        DIR = _d
        break
if DIR is None:
    for root, _, files in os.walk("/kaggle/input"):
        if (
            "train.csv" in files
            and "test.csv" in files
            and "sample_submission.csv" in files
        ):
            DIR = root
            break

if DIR is None:
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv/sample_submission.csv under /kaggle/input"
    )

BATCH_SIZE = 6

print("Using DIR:", DIR)



## === cell 2
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.pop("TF_USE_C_API_PROTOBUF", None)

import re
import gc
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

np.random.seed(10)
tf.random.set_seed(10)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer

_USE_ST = False
_ST_MODEL_NAME = "all-MiniLM-L6-v2"
_st_model = None

try:
    from sentence_transformers import (
        SentenceTransformer,
    )  # may still trigger protobuf crash

    _st_model = SentenceTransformer(_ST_MODEL_NAME)
    _USE_ST = True
    print("Embedding backend: sentence-transformers:", _ST_MODEL_NAME)
except Exception as e:
    _USE_ST = False
    _st_model = None
    _st_import_error = repr(e)
    print(
        "Embedding backend: TF-IDF fallback (SentenceTransformer unavailable):",
        _st_import_error,
    )

_tfidf_vectorizer = None  # fit later if needed


def _clean_text_list(text_batch: List):
    return [
        "" if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)
        for x in text_batch
    ]


def _st_embed_batch(text_batch):
    text_batch = _clean_text_list(text_batch)
    emb = _st_model.encode(
        text_batch,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return emb.astype(np.float32)


def _tfidf_embed_batch(text_batch):
    global _tfidf_vectorizer
    text_batch = _clean_text_list(text_batch)
    mat = _tfidf_vectorizer.transform(text_batch)
    return mat.toarray().astype(np.float32)


def _embed_batch(text_batch):
    if _USE_ST:
        return _st_embed_batch(text_batch)
    return _tfidf_embed_batch(text_batch)




## === cell 4
train_df = pd.read_csv(DIR + "/train.csv")
test_df = pd.read_csv(DIR + "/test.csv")
sample_submission = pd.read_csv(DIR + "/sample_submission.csv")
TARGET_COLS = sample_submission.columns.tolist()[1:]  # 30 targets

if not _USE_ST:
    all_text = pd.concat(
        [
            train_df["question_title"].fillna("").astype(str),
            train_df["question_body"].fillna("").astype(str),
            train_df["answer"].fillna("").astype(str),
            test_df["question_title"].fillna("").astype(str),
            test_df["question_body"].fillna("").astype(str),
            test_df["answer"].fillna("").astype(str),
        ],
        axis=0,
        ignore_index=True,
    )
    _tfidf_vectorizer = TfidfVectorizer(
        max_features=5120,  # keeps runtime/memory bounded while producing stable dense embeddings
        ngram_range=(1, 2),
        min_df=2,
    )
    _tfidf_vectorizer.fit(all_text.tolist())
    print("TF-IDF vocab size:", len(_tfidf_vectorizer.vocabulary_))




## === cell 5
def get_encoder_embed(string_list):
    batch_size = 32
    n = len(string_list)

    string_list = _clean_text_list(string_list)

    embs = []
    for i in range(0, n, batch_size):
        batch_text = string_list[i : i + batch_size]
        batch_emb = _embed_batch(batch_text)
        embs.append(batch_emb.astype(np.float32))

    dummy_token_batches = None
    dummy_masks = None
    return (
        dummy_token_batches,
        dummy_masks,
        tf.convert_to_tensor(np.vstack(embs), dtype=tf.float32),
    )




## === cell 6
question_ids, question_masks = {}, {}
answer_ids, answer_masks = {}, {}
question_encode = {}
answer_encode = {}

question_ids["train"], question_masks["train"], question_encode["train"] = (
    get_encoder_embed(train_df.question_body.tolist())
)
question_ids["test"], question_masks["test"], question_encode["test"] = (
    get_encoder_embed(test_df.question_body.tolist())
)

answer_ids["train"], answer_masks["train"], answer_encode["train"] = get_encoder_embed(
    train_df.answer.tolist()
)
answer_ids["test"], answer_masks["test"], answer_encode["test"] = get_encoder_embed(
    test_df.answer.tolist()
)

question_encode["train"] = question_encode["train"].numpy()
question_encode["test"] = question_encode["test"].numpy()
answer_encode["train"] = answer_encode["train"].numpy()
answer_encode["test"] = answer_encode["test"].numpy()

gc.collect()




## === cell 7
def get_universal_encoder(df):
    cols = ["question_title", "question_body", "answer"]
    universal_embed = {}
    for col in cols:
        x = (
            df[col]
            .fillna("")
            .astype(str)
            .str.replace("?", ".", regex=False)
            .str.replace("!", ".", regex=False)
            .tolist()
        )
        _, _, emb = get_encoder_embed(x)
        universal_embed[col] = emb.numpy()
    return universal_embed




## === cell 8
train_df["netloc"] = train_df.url.apply(
    lambda x: (
        re.search(r"//.*?\.", x).group(0)[2:-1]
        if isinstance(x, str) and re.search(r"//.*?\.", x)
        else ""
    )
)
test_df["netloc"] = test_df.url.apply(
    lambda x: (
        re.search(r"//.*?\.", x).group(0)[2:-1]
        if isinstance(x, str) and re.search(r"//.*?\.", x)
        else ""
    )
)

ohe = OneHotEncoder(handle_unknown="ignore")
features = ["netloc", "category"]
merged = pd.concat([train_df[features], test_df[features]], axis=0)
ohe.fit(merged)

features_train = ohe.transform(train_df[features]).toarray().astype(np.float32)
features_test = ohe.transform(test_df[features]).toarray().astype(np.float32)



## === cell 9
train_universal_embed = get_universal_encoder(train_df)
test_universal_embed = get_universal_encoder(test_df)
tf.keras.backend.clear_session()
gc.collect()



## === cell 10
l2_dist = lambda x, y: np.power(x - y, 2).sum(axis=1)
cos_dist = lambda x, y: (x * y).sum(axis=1)

dist_features_train = np.array(
    [
        l2_dist(
            train_universal_embed["question_title"], train_universal_embed["answer"]
        ),
        l2_dist(
            train_universal_embed["question_body"], train_universal_embed["answer"]
        ),
        l2_dist(
            train_universal_embed["question_body"],
            train_universal_embed["question_title"],
        ),
        cos_dist(
            train_universal_embed["question_title"], train_universal_embed["answer"]
        ),
        cos_dist(
            train_universal_embed["question_body"], train_universal_embed["answer"]
        ),
        cos_dist(
            train_universal_embed["question_body"],
            train_universal_embed["question_title"],
        ),
    ]
).T.astype(np.float32)

dist_features_test = np.array(
    [
        l2_dist(test_universal_embed["question_title"], test_universal_embed["answer"]),
        l2_dist(test_universal_embed["question_body"], test_universal_embed["answer"]),
        l2_dist(
            test_universal_embed["question_body"],
            test_universal_embed["question_title"],
        ),
        cos_dist(
            test_universal_embed["question_title"], test_universal_embed["answer"]
        ),
        cos_dist(test_universal_embed["question_body"], test_universal_embed["answer"]),
        cos_dist(
            test_universal_embed["question_body"],
            test_universal_embed["question_title"],
        ),
    ]
).T.astype(np.float32)



## === cell 11
X_train = np.hstack(
    [
        question_encode["train"].astype(np.float32),
        answer_encode["train"].astype(np.float32),
        dist_features_train,
        features_train,
    ]
    + [v.astype(np.float32) for _, v in train_universal_embed.items()]
).astype(np.float32)

X_test = np.hstack(
    [
        question_encode["test"].astype(np.float32),
        answer_encode["test"].astype(np.float32),
        dist_features_test,
        features_test,
    ]
    + [v.astype(np.float32) for _, v in test_universal_embed.items()]
).astype(np.float32)



## === cell 12
Y_train = train_df[TARGET_COLS].values.astype(np.float32)




## === cell 13
class SpearmanRhoCallback(tf.keras.callbacks.Callback):
    def __init__(self, training_data, validation_data, patience):
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
            corr = spearmanr(
                self.y_val[:, ind],
                y_pred_val[:, ind] + np.random.normal(0, 1e-7, y_pred_val.shape[0]),
            ).correlation
            if corr is None or np.isnan(corr):
                corr = 0.0
            rho_val += corr
        rho_val /= self.y_val.shape[1]
        if rho_val >= self.value:
            self.value = rho_val
        else:
            self.bad_epochs += 1

        print("\rval_spearman-rho: %s" % (str(round(rho_val, 4))), end=100 * " " + "\n")
        return rho_val




## === cell 14
def create_model():
    inp = tf.keras.Input(shape=(X_train.shape[1],))
    x = tf.keras.layers.Dense(128, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(Y_train.shape[1], activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inp, outputs=x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=["binary_crossentropy"],
    )
    return model




## === cell 15
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



## === cell 16
kf = KFold(n_splits=5, random_state=10, shuffle=True).split(X=X_train)
models = []
for fold, (train_idx, valid_idx) in enumerate(kf):
    tf.keras.backend.clear_session()
    model = create_model()
    model.fit(
        X_train[train_idx],
        Y_train[train_idx],
        validation_data=(X_train[valid_idx], Y_train[valid_idx]),
        epochs=100,
        batch_size=64,
        verbose=0,
        callbacks=[
            SpearmanRhoCallback(
                training_data=(X_train[train_idx], Y_train[train_idx]),
                validation_data=(X_train[valid_idx], Y_train[valid_idx]),
                patience=5,
            ),
            lr_schedule,
        ],
    )
    models.append(model)
    print("##########################################################")



## === cell 17
test_preds = np.zeros((X_test.shape[0], Y_train.shape[1]), dtype=np.float32)
for m in models:
    test_preds += m.predict(X_test, batch_size=256, verbose=0).astype(np.float32)
test_preds /= len(models)

test_preds = np.clip(test_preds, 0.0, 1.0)



## === cell 18
submission = pd.DataFrame({"qa_id": test_df["qa_id"].values})
for i, col in enumerate(TARGET_COLS):
    submission[col] = test_preds[:, i]

submission = submission[["qa_id"] + TARGET_COLS]
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)



## === cell 19
submission.head()
