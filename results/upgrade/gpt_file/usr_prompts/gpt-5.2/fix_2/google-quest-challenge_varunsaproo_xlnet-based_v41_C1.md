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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import re
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub  # kept for minimal change; not used after USE removal
import transformers

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
MODEL_NAME = "distilbert-base-uncased"
tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(
    MODEL_NAME, local_files_only=True
)
transformer = transformers.TFDistilBertModel.from_pretrained(
    MODEL_NAME, local_files_only=True
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2072102964.py in <cell line: 0>()
      2 # Use an offline-available pretrained DistilBERT (will load from local cache in Kaggle).
      3 MODEL_NAME = "distilbert-base-uncased"
----> 4 tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(
      5     MODEL_NAME, local_files_only=True
      6 )

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, trust_remote_code, *init_inputs, **kwargs)
   2012                 logger.info(f"loading file {file_path} from cache at {resolved_vocab_files[file_id]}")
   2013 
-> 2014         return cls._from_pretrained(
   2015             resolved_vocab_files,
   2016             pretrained_model_name_or_path,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in _from_pretrained(cls, resolved_vocab_files, pretrained_model_name_or_path, init_configuration, token, cache_dir, local_files_only, _commit_hash, _is_local, trust_remote_code, *init_inputs, **kwargs)
   2050         # loaded directly from the GGUF file.
   2051         if (from_slow or not has_tokenizer_file) and cls.slow_tokenizer_class is not None and not gguf_file:
-> 2052             slow_tokenizer = (cls.slow_tokenizer_class)._from_pretrained(
   2053                 copy.deepcopy(resolved_vocab_files),
   2054                 pretrained_model_name_or_path,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in _from_pretrained(cls, resolved_vocab_files, pretrained_model_name_or_path, init_configuration, token, cache_dir, local_files_only, _commit_hash, _is_local, trust_remote_code, *init_inputs, **kwargs)
   2258         # Instantiate the tokenizer.
   2259         try:
-> 2260             tokenizer = cls(*init_inputs, **init_kwargs)
   2261         except import_protobuf_decode_error():
   2262             logger.info(

/usr/local/lib/python3.11/dist-packages/transformers/models/distilbert/tokenization_distilbert.py in __init__(self, vocab_file, do_lower_case, do_basic_tokenize, never_split, unk_token, sep_token, pad_token, cls_token, mask_token, tokenize_chinese_chars, strip_accents, clean_up_tokenization_spaces, **kwargs)
    115         **kwargs,
    116     ):
--> 117         if not os.path.isfile(vocab_file):
    118             raise ValueError(
    119                 f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"

/usr/lib/python3.11/genericpath.py in isfile(path)

TypeError: stat: path should be string, bytes, os.PathLike or integer, not NoneType

## === cell 4
train_df = pd.read_csv(DIR + "/train.csv")
test_df = pd.read_csv(DIR + "/test.csv")
sample_submission = pd.read_csv(DIR + "/sample_submission.csv")
TARGET_COLS = sample_submission.columns.tolist()[1:]  # 30 targets




## === cell 5
def get_encoder_embed(string_list):
    batch_size = 4
    max_len = 512
    embed = []
    n = len(string_list)

    string_list = [
        "" if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)
        for x in string_list
    ]

    for i in range(0, n, batch_size):
        batch_text = string_list[i : i + batch_size]

        enc = tokenizer(
            batch_text,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="np",
        )
        input_ids = enc["input_ids"].astype(np.int32)
        attention_mask = enc["attention_mask"].astype(np.int32)

        out = transformer({"input_ids": input_ids, "attention_mask": attention_mask})[
            0
        ][:, 0, :]
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




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/766435041.py in <cell line: 0>()
      6 # Body/answer CLS embeddings (as in original intent)
      7 question_ids["train"], question_masks["train"], question_encode["train"] = (
----> 8     get_encoder_embed(train_df.question_body.tolist())
      9 )
     10 question_ids["test"], question_masks["test"], question_encode["test"] = (

/tmp/ipykernel_55/86847138.py in get_encoder_embed(string_list)
     15         batch_text = string_list[i : i + batch_size]
     16 
---> 17         enc = tokenizer(
     18             batch_text,
     19             max_length=max_len,

NameError: name 'tokenizer' is not defined

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



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3232560156.py in <cell line: 0>()
----> 1 train_universal_embed = get_universal_encoder(train_df)
      2 test_universal_embed = get_universal_encoder(test_df)
      3 tf.keras.backend.clear_session()
      4 gc.collect()
      5 

/tmp/ipykernel_55/3830352418.py in get_universal_encoder(df)
     15             .tolist()
     16         )
---> 17         _, _, emb = get_encoder_embed(x)
     18         universal_embed[col] = emb.numpy()
     19     return universal_embed

/tmp/ipykernel_55/86847138.py in get_encoder_embed(string_list)
     15         batch_text = string_list[i : i + batch_size]
     16 
---> 17         enc = tokenizer(
     18             batch_text,
     19             max_length=max_len,

NameError: name 'tokenizer' is not defined

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



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1366543855.py in <cell line: 0>()
      5     [
      6         l2_dist(
----> 7             train_universal_embed["question_title"], train_universal_embed["answer"]
      8         ),
      9         l2_dist(

NameError: name 'train_universal_embed' is not defined

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



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/2208099330.py in <cell line: 0>()
      2 X_train = np.hstack(
      3     [
----> 4         question_encode["train"].astype(np.float32),
      5         answer_encode["train"].astype(np.float32),
      6         dist_features_train,

KeyError: 'train'

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



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1228171710.py in <cell line: 0>()
----> 1 kf = KFold(n_splits=5, random_state=10, shuffle=True).split(X=X_train)
      2 models = []
      3 for fold, (train_idx, valid_idx) in enumerate(kf):
      4     tf.keras.backend.clear_session()
      5     model = create_model()

NameError: name 'X_train' is not defined

## === cell 17
test_preds = np.zeros((X_test.shape[0], Y_train.shape[1]), dtype=np.float32)
for m in models:
    test_preds += m.predict(X_test, batch_size=256, verbose=0).astype(np.float32)
test_preds /= len(models)

test_preds = np.clip(test_preds, 0.0, 1.0)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2712127409.py in <cell line: 0>()
      1 # Predict with fold-ensemble (mean). Keeps core training approach; improves stability vs last-fold only.
----> 2 test_preds = np.zeros((X_test.shape[0], Y_train.shape[1]), dtype=np.float32)
      3 for m in models:
      4     test_preds += m.predict(X_test, batch_size=256, verbose=0).astype(np.float32)
      5 test_preds /= len(models)

NameError: name 'X_test' is not defined

## === cell 18
submission = pd.read_csv(DIR + "/sample_submission.csv")
submission.iloc[:, 1:] = test_preds
submission.to_csv("submission.csv", index=False)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2546494216.py in <cell line: 0>()
      1 submission = pd.read_csv(DIR + "/sample_submission.csv")
----> 2 submission.iloc[:, 1:] = test_preds
      3 submission.to_csv("submission.csv", index=False)
      4 

NameError: name 'test_preds' is not defined

## === cell 19
submission.head()
