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

0.2710967102177729

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import re
import gc
import numpy as np
import pandas as pd

import tensorflow as tf
import transformers

from sklearn.preprocessing import OneHotEncoder
from scipy.stats import spearmanr

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DIR = "/kaggle/input/google-quest-challenge"
BATCH_SIZE = 6

print("Listing /kaggle/input (trimmed):")
n_show = 0
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        if n_show < 50:
            print(os.path.join(dirname, filename))
        n_show += 1
print(f"Total files found: {n_show}")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

train_df = pd.read_csv(f"{DIR}/train.csv")
test_df = pd.read_csv(f"{DIR}/test.csv")
sample_submission = pd.read_csv(f"{DIR}/sample_submission.csv")

target_cols = sample_submission.columns.tolist()[1:]
assert len(target_cols) == 30, f"Expected 30 targets, got {len(target_cols)}"

print("Train shape:", train_df.shape, "Test shape:", test_df.shape)
print("Targets:", len(target_cols))



## === cell 2
MODEL_NAME = "distilbert-base-uncased"

tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(
    MODEL_NAME, local_files_only=True
)
transformer = transformers.TFDistilBertModel.from_pretrained(
    MODEL_NAME, local_files_only=True
)

transformer.trainable = False




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2365906233.py in <cell line: 0>()
      4 MODEL_NAME = "distilbert-base-uncased"
      5 
----> 6 tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(
      7     MODEL_NAME, local_files_only=True
      8 )

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

## === cell 3
def get_encoder_embed(string_list):
    batch_size = 4
    max_len = 512
    embed_batches = []
    n = len(string_list)

    for i in range(0, n, batch_size):
        texts = [
            "" if (t is None or (isinstance(t, float) and np.isnan(t))) else str(t)
            for t in string_list[i : i + batch_size]
        ]

        enc = tokenizer(
            texts,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="tf",
        )
        out = transformer(enc, training=False)[0]  # (bs, seq, hidden)
        cls = out[:, 0, :]  # (bs, hidden)
        embed_batches.append(cls)

    return tf.concat(embed_batches, axis=0).numpy()




## === cell 4
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



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2961343553.py in <cell line: 0>()
      3 answer_encode = {}
      4 
----> 5 question_encode["train"] = get_encoder_embed(train_df["question_body"].tolist())
      6 question_encode["test"] = get_encoder_embed(test_df["question_body"].tolist())
      7 

/tmp/ipykernel_55/711481714.py in get_encoder_embed(string_list)
     11         ]
     12 
---> 13         enc = tokenizer(
     14             texts,
     15             max_length=max_len,

NameError: name 'tokenizer' is not defined

## === cell 5
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

features_train = ohe.transform(train_df[features]).toarray()
features_test = ohe.transform(test_df[features]).toarray()

print("OHE train/test:", features_train.shape, features_test.shape)



## === cell 6
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



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/2656394283.py in <cell line: 0>()
      7 X_train = np.hstack(
      8     [
----> 9         question_encode["train"],
     10         answer_encode["train"],
     11         dist_features_train,

KeyError: 'train'

## === cell 7
Y_train = train_df[target_cols].values.astype(np.float32)
print("Y_train:", Y_train.shape)




## === cell 8
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




## === cell 9
def create_model(input_dim, output_dim):
    inp = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(128, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
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



## === cell 10
model = create_model(X_train.shape[1], Y_train.shape[1])
model.fit(X_train, Y_train, epochs=100, batch_size=64, verbose=2)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/183147797.py in <cell line: 0>()
      1 # Train (no CV in original provided code; keep simple fit)
----> 2 model = create_model(X_train.shape[1], Y_train.shape[1])
      3 model.fit(X_train, Y_train, epochs=100, batch_size=64, verbose=2)
      4 

NameError: name 'X_train' is not defined

## === cell 11
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

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/536589562.py in <cell line: 0>()
      1 # Predict and write submission
----> 2 ans = model.predict(X_test, verbose=0)
      3 ans = np.clip(ans, 0.0, 1.0)
      4 
      5 # Ensure correct shape/columns

NameError: name 'model' is not defined
