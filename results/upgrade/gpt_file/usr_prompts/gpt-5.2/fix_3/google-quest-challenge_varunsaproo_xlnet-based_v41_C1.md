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

0.37578

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.37578) has done: 'I fix the protobuf/transformers import crash and the missing tokenizer/transformer that prevents any embeddings from being computed, by removing the incompatible `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override and switching to a locally-available TF-Hub Universal Sentence Encoder so the notebook works fully offline in Kaggle. I also correct the `DIR` path to the actual dataset location under `/kaggle/input/` so the CSVs load reliably. The rest of the pipeline (feature construction, dense+dropout+sigmoid model, KFold training with Spearman callback, and submission writing) is kept the same. Finally, I ensure the submission uses the test `qa_id` order (sample submission has only 608 rows) so the output CSV has the correct 19,550 rows and 31 columns.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
DIR = "/kaggle/input/google-quest-challenge"
BATCH_SIZE = 6



## === cell 2
import os
import re
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import transformers  # kept for minimal change; not used after switching to TF-Hub encoder

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

np.random.seed(10)
tf.random.set_seed(10)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
USE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
use_model = hub.load(USE_URL)



## === cell 4
train_df = pd.read_csv(DIR + "/train.csv")
test_df = pd.read_csv(DIR + "/test.csv")
sample_submission = pd.read_csv(DIR + "/sample_submission.csv")
TARGET_COLS = sample_submission.columns.tolist()[1:]  # 30 targets




## === cell 5
def get_encoder_embed(string_list):
    batch_size = 64
    embed = []
    n = len(string_list)

    string_list = [
        "" if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)
        for x in string_list
    ]

    for i in range(0, n, batch_size):
        batch_text = string_list[i : i + batch_size]
        out = use_model(batch_text)  # (batch, 512)
        embed.append(out)

    dummy_token_batches = None
    dummy_masks = None
    return (dummy_token_batches, dummy_masks, tf.concat(embed, axis=0))




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
            rho_val += spearmanr(
                self.y_val[:, ind],
                y_pred_val[:, ind] + np.random.normal(0, 1e-7, y_pred_val.shape[0]),
            ).correlation
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

submission.to_csv("submission.csv", index=False)



## === cell 19
submission.head()
