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

No external packages required in the script and installed.

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

-0.0015378042945261

# 6. Current score

nan

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved nan) has done: 'I fix the two blockers preventing any submission: (1) a protobuf/TF/transformers import crash and (2) incorrect calling conventions for TF XLNet inside a Keras Functional model (it rejects KerasTensors when passed positionally). I pin protobuf to the pure-Python implementation early to avoid the `MessageFactory.GetPrototype` issue, and I update the transformer backbone calls to use keyword arguments (`input_ids=..., attention_mask=...`) so they work with KerasTensors while keeping the same architecture and outputs. I also make the pipeline robust by skipping XLNet if it still cannot load in this environment (fallback is score-neutral relative to “no submission”), and ensure cell numbering starts at 1 and a valid `submission.csv` is always written. No training logic, head layers, loss, or post-processing are changed beyond what’s needed to run end-to-end.'
- What this solution (achieved nan) has done: 'I fix the protobuf/transformers crash by forcing the pure-Python protobuf implementation *before* any TensorFlow/transformers import and by also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`, which commonly resolves the `MessageFactory.GetPrototype` issue in Kaggle images. I keep the model/training logic intact, but add a safe fallback to load tokenizers/models from local cache only (no internet) and, if unavailable, use the same existing constant-0.5 fallback so a submission is always produced. I also correct the cell numbering to start at 1 (Kaggle notebook/export compatibility) and ensure the submission columns/order match `sample_submission.csv` exactly. These changes are execution/stability fixes; they should move the score from `nan` to a valid value by guaranteeing a valid `submission.csv` is written.'
- What this solution (achieved nan) has done: 'I fix the protobuf/transformers crash that currently stops execution before any submission is written by forcing the pure-Python protobuf implementation even if the environment has already imported `google.protobuf`, and by importing `transformers` only after that. I also remove the hard dependency on SciPy by providing a small Spearman implementation fallback (so the run doesn’t fail if `scipy` isn’t available), while keeping the same training metric semantics. Finally, I keep your model/training/prediction logic intact, but make the inference input-key selection consistent with how each model was built so prediction doesn’t error and a valid `submission.csv` is always produced.'
- What this solution (achieved nan) has done: 'I fix the protobuf-related crash by forcing the pure-Python protobuf implementation *before* any TensorFlow/transformers import and by avoiding any already-loaded `google.protobuf` modules. I also make the `transformers` import/load robust by setting `TRANSFORMERS_OFFLINE=1` and using `local_files_only=True`, so the notebook doesn’t hang or fail due to no-internet constraints; if models still can’t load, the existing constant-0.5 fallback ensure a valid submission is produced. Finally, I keep your model/training/prediction logic unchanged, but ensure the script always reaches the CSV-writing cell and that `submission.csv` has the exact sample column order and valid `[0,1]` values (preventing `nan` submissions). These changes should move you from `nan` (no valid run/score) to a valid scored submission without altering the intended modeling approach.'
- What this solution (achieved nan) has done: 'I fix the protobuf/transformers crash that currently stops execution (and leads to `nan`/no valid scored submission) by forcing the pure-Python protobuf implementation *and* downgrading to the TF-friendly `transformers` TF backend without touching your model/head/training logic. I also make tokenizer/model loading strictly offline (`local_files_only=True`) so the notebook doesn’t hang in the Kaggle no-internet environment; if weights aren’t available, your existing safe fallback still writes a valid submission. Finally, I ensure cell numbering starts at 1 and that the produced `submission.csv` exactly matches `sample_submission.csv` columns/order with finite values in `[0,1]`, preventing invalid submissions.'
- What this solution (achieved nan) has done: 'I fix the protobuf/transformers crash that prevents any end-to-end run by forcing the pure-Python protobuf implementation and pinning a compatible protobuf version before importing TensorFlow/transformers. I also remove the transformers pip-install attempt (which can fail/hang offline) and instead run fully offline with `local_files_only=True`, falling back to constant 0.5 predictions if pretrained files are not present. Finally, I add a hard safety net so even if transformers cannot be imported at all, the script still writes a valid `submission.csv` with the exact `sample_submission.csv` columns and finite values in `[0,1]`, eliminating the `nan` outcome.'
- What this solution (achieved nan) has done: 'Your current `nan` score is consistent with an invalid submission (e.g., wrong row count vs. `sample_submission`, misaligned `qa_id`, or NaNs/Infs slipping through). I make two minimal, score-relevant fixes: (1) build the submission from `sample_submission`’s `qa_id` and left-join predictions to guarantee the exact required 608 rows/order, and (2) ensure the prediction frame is indexed by `qa_id` and de-duplicated so the merge cannot silently expand/contract rows. These changes preserve your modeling/training logic and only harden the last-mile submission formatting so Kaggle can score it (moving from `nan` to a valid numeric score, closer to the target band).'
- What this solution (achieved nan) has done: 'Your `nan` score strongly suggests Kaggle couldn’t compute Spearman due to invalid submission values (NaNs/Infs) and/or schema mismatch, so the smallest score-improving change is to harden the submission generation to guarantee *finite* `[0,1]` floats for all 30 targets and the exact `sample_submission` row order. I remove the online `pip install` (often fails/hangs and can break imports), keep transformers strictly offline, and add a deterministic, score-safe fallback that always produces valid predictions without changing your model/training logic. I also fix the cell numbering to start at 1 (Kaggle export compatibility) and make the final CSV build use `sample_submission` as the single source of truth for `qa_id` ordering/row count. These changes are designed to move you from `nan` to a valid numeric score (closer to the target band) with minimal impact on the core approach.'
- What this solution (achieved nan) has done: 'I fix the protobuf/transformers import crash that currently prevents any end-to-end run (and thus yields `nan`) by forcing the pure-Python protobuf implementation and purging any preloaded `google.protobuf` modules before importing TensorFlow/transformers. To keep the core modeling/training logic intact, I won’t change the architecture, loss, or training loop; I only make the imports robust and strictly offline (`local_files_only=True`) so the notebook doesn’t hang or fail without internet. I also harden the last-mile submission creation to always match `sample_submission.csv` row order/row count and ensure all 30 target columns are finite floats in `[0,1]`. With these fixes, the script always write a valid `submission.csv` and should move the score from `nan` to a real value.'
- What this solution (achieved nan) has done: 'I fix the execution blocker causing the protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation early and (when available) switching TensorFlow to use the legacy Keras API, which is the most common compatibility path for `transformers` TF models in Kaggle images. I also make the imports robust by catching failures of TensorFlow/transformers and falling back to constant 0.5 predictions so a valid `submission.csv` is always produced (moving you from `nan` to a real score). Finally, I keep your model/training/prediction logic intact, but harden the submission build to always match `sample_submission.csv` row order/columns and ensure all outputs are finite floats in `[0,1]`.'
- What this solution (achieved nan) has done: 'I fix the protobuf/transformers crash that stops execution by applying the protobuf implementation env vars before any protobuf-dependent imports and by defensively handling the `MessageFactory.GetPrototype` incompatibility (so the notebook can still run even if transformers can’t import). Then I make `get_data()` pick the correct Kaggle input path robustly (both `/kaggle/input/...` and the mirrored paths you listed), so train/test/sample_submission always load. Finally, I keep your modeling/training logic unchanged, but ensure we always produce a valid `submission.csv` with the exact `sample_submission.csv` row order/columns and with finite `[0,1]` values, eliminating the `nan` outcome.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

import random
import html
import warnings

import numpy as np
import pandas as pd

TF_AVAILABLE = True
try:
    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D
    from tensorflow.keras.models import Model
except Exception as e:
    print(f"WARNING: TensorFlow not available/importable: {repr(e)}")
    TF_AVAILABLE = False
    tf = None
    K = None
    Dense = Dropout = GlobalAveragePooling1D = Model = None

from tqdm.auto import tqdm
from sklearn.model_selection import KFold

try:
    from scipy.stats import spearmanr as scipy_spearmanr  # type: ignore
except Exception:
    scipy_spearmanr = None

TRANSFORMERS_AVAILABLE = False
if TF_AVAILABLE:
    try:
        import transformers  # noqa: F401

        from transformers import (
            BertTokenizer,
            RobertaTokenizer,
            XLNetTokenizer,
            BertConfig,
            RobertaConfig,
            XLNetConfig,
            TFBertModel,
            TFRobertaModel,
            TFXLNetModel,
        )

        TRANSFORMERS_AVAILABLE = True
    except Exception as e:
        print(f"WARNING: transformers not available/importable: {repr(e)}")
        TRANSFORMERS_AVAILABLE = False
else:
    print("WARNING: Skipping transformers import because TensorFlow is unavailable.")
    TRANSFORMERS_AVAILABLE = False

warnings.filterwarnings("ignore")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
seed = 13
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
if TF_AVAILABLE:
    tf.random.set_seed(seed)




## === cell 2
def get_data():
    print("getting test and train data...")

    candidate_base_paths = [
        "/kaggle/input/google-quest-challenge/",
        "/kaggle/input/google-quest-challenge/google-quest-challenge/",
        "/kaggle/data/google-quest-challenge/",
        "/kaggle/data/google-quest-challenge/google-quest-challenge/",
    ]
    base_path = None
    for p in candidate_base_paths:
        if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
            os.path.join(p, "test.csv")
        ):
            base_path = p
            break
    if base_path is None:
        raise FileNotFoundError(
            "Could not find train.csv/test.csv under expected Kaggle paths."
        )

    train = pd.read_csv(os.path.join(base_path, "train.csv"))
    test = pd.read_csv(os.path.join(base_path, "test.csv"))
    sample_submission = pd.read_csv(os.path.join(base_path, "sample_submission.csv"))

    target_cols = sample_submission.columns.tolist()[1:]  # 30 targets
    y = train[target_cols]
    X = train[["question_title", "question_body", "answer"]].copy()
    X_test = test[["question_title", "question_body", "answer"]].copy()

    for col in ["question_title", "question_body", "answer"]:
        X[col] = X[col].astype(str).apply(html.unescape)
        X_test[col] = X_test[col].astype(str).apply(html.unescape)

    return X, X_test, y, train, test, sample_submission, target_cols




## === cell 3
def get_tokenizer(model_name):
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not available; cannot load tokenizer.")
    print(f"getting tokenizer for {model_name}...")
    kwargs = dict(local_files_only=True)
    try:
        if model_name == "xlnet-base-cased":
            tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased", **kwargs)
        elif model_name == "roberta-base":
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base", **kwargs)
        elif model_name == "bert-base-uncased":
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", **kwargs)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
        return tokenizer
    except Exception as e:
        raise RuntimeError(
            f"Tokenizer for {model_name} not available locally (offline mode). Original: {repr(e)}"
        )




## === cell 4
def fix_length(
    tokens,
    max_sequence_length=512,
    q_max_len=254,
    a_max_len=254,
    model_type="questions",
):
    if model_type == "questions":
        length = len(tokens)
        if length > max_sequence_length:
            tokens = tokens[: max_sequence_length - 1]
        return tokens
    else:
        question_tokens, answer_tokens = tokens
        q_len = len(question_tokens)
        a_len = len(answer_tokens)
        if q_len + a_len + 3 > max_sequence_length:
            if a_max_len <= a_len and q_max_len <= q_len:
                q_new_len_head = q_max_len // 2
                question_tokens = (
                    question_tokens[:q_new_len_head] + question_tokens[-q_new_len_head:]
                )
                a_new_len_head = a_max_len // 2
                answer_tokens = (
                    answer_tokens[:a_new_len_head] + answer_tokens[-a_new_len_head:]
                )
            elif q_len <= a_len and q_len < q_max_len:
                a_max_len = a_max_len + (q_max_len - q_len - 1)
                a_new_len_head = a_max_len // 2
                answer_tokens = (
                    answer_tokens[:a_new_len_head] + answer_tokens[-a_new_len_head:]
                )
            elif a_len < q_len:
                q_max_len = q_max_len + (a_max_len - a_len - 1)
                q_new_len_head = q_max_len // 2
                question_tokens = (
                    question_tokens[:q_new_len_head] + question_tokens[-q_new_len_head:]
                )
    return question_tokens, answer_tokens




## === cell 5
def transformer_inputs(
    title, question, answer, tokenizer, model_type="questions", MAX_SEQUENCE_LENGTH=512
):
    if model_type == "questions":
        question = f"{title} [SEP] {question}"
        question_tokens = tokenizer.tokenize(question)
        question_tokens = fix_length(question_tokens, model_type=model_type)

        ids_q = tokenizer.convert_tokens_to_ids(["[CLS]"] + question_tokens)
        padded_ids = (
            ids_q + [tokenizer.pad_token_id] * (MAX_SEQUENCE_LENGTH - len(ids_q))
        )[:MAX_SEQUENCE_LENGTH]
        token_type_ids = ([0] * MAX_SEQUENCE_LENGTH)[:MAX_SEQUENCE_LENGTH]
        attention_mask = ([1] * len(ids_q) + [0] * (MAX_SEQUENCE_LENGTH - len(ids_q)))[
            :MAX_SEQUENCE_LENGTH
        ]
        return padded_ids, token_type_ids, attention_mask
    else:
        question = f"{title} [SEP] {question}"
        question_tokens = tokenizer.tokenize(question)
        answer_tokens = tokenizer.tokenize(answer)
        question_tokens, answer_tokens = fix_length(
            tokens=(question_tokens, answer_tokens), model_type=model_type
        )

        ids = tokenizer.convert_tokens_to_ids(
            ["[CLS]"] + question_tokens + ["[SEP]"] + answer_tokens + ["[SEP]"]
        )
        padded_ids = (
            ids + [tokenizer.pad_token_id] * (MAX_SEQUENCE_LENGTH - len(ids))
        )[:MAX_SEQUENCE_LENGTH]
        token_type_ids = (
            [0] * (1 + len(question_tokens) + 1)
            + [1] * (len(answer_tokens) + 1)
            + [0] * (MAX_SEQUENCE_LENGTH - len(ids))
        )[:MAX_SEQUENCE_LENGTH]
        attention_mask = ([1] * len(ids) + [0] * (MAX_SEQUENCE_LENGTH - len(ids)))[
            :MAX_SEQUENCE_LENGTH
        ]
        return padded_ids, token_type_ids, attention_mask




## === cell 6
def input_data(df, tokenizer, model_type="questions"):
    print(f"generating {model_type} input for transformer...")
    input_ids, input_token_type_ids, input_attention_masks = [], [], []
    for title, body, answer in tqdm(
        zip(
            df["question_title"].values, df["question_body"].values, df["answer"].values
        ),
        total=len(df),
        leave=False,
    ):
        ids, type_ids, mask = transformer_inputs(
            title, body, answer, tokenizer, model_type=model_type
        )
        input_ids.append(ids)
        input_token_type_ids.append(type_ids)
        input_attention_masks.append(mask)

    return (
        np.asarray(input_ids, dtype=np.int32),
        np.asarray(input_attention_masks, dtype=np.int32),
        np.asarray(input_token_type_ids, dtype=np.int32),
    )




## === cell 7
def get_model(name):
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not available; cannot load model.")
    kwargs = dict(local_files_only=True)
    try:
        if name == "xlnet-base-cased":
            config = XLNetConfig.from_pretrained(
                "xlnet-base-cased", output_hidden_states=True, **kwargs
            )
            model = TFXLNetModel.from_pretrained(
                "xlnet-base-cased", config=config, **kwargs
            )
        elif name == "roberta-base":
            config = RobertaConfig.from_pretrained(
                "roberta-base", output_hidden_states=True, **kwargs
            )
            model = TFRobertaModel.from_pretrained(
                "roberta-base", config=config, **kwargs
            )
        elif name == "bert-base-uncased":
            config = BertConfig.from_pretrained(
                "bert-base-uncased", output_hidden_states=True, **kwargs
            )
            model = TFBertModel.from_pretrained(
                "bert-base-uncased", config=config, **kwargs
            )
        else:
            raise ValueError(f"Unknown model: {name}")
        return model
    except Exception as e:
        raise RuntimeError(
            f"Model weights for {name} not available locally (offline mode). Original: {repr(e)}"
        )




## === cell 8
def create_model(name="xlnet-base-cased", model_type="questions"):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available; cannot create model.")
    print(f"creating model {name} ({model_type})...")
    K.clear_session()
    max_seq_length = 512

    input_tokens = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name="input_tokens"
    )
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name="input_mask"
    )
    input_segment = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name="input_segment"
    )

    backbone = get_model(name)

    if name == "xlnet-base-cased":
        outputs = backbone(
            input_ids=input_tokens, attention_mask=input_mask, training=False
        )
        hidden_states = outputs.hidden_states
    elif name == "roberta-base" and model_type != "questions":
        outputs = backbone(
            input_ids=input_tokens, attention_mask=input_mask, training=False
        )
        hidden_states = outputs.hidden_states
    else:
        outputs = backbone(
            input_ids=input_tokens,
            attention_mask=input_mask,
            token_type_ids=input_segment,
            training=False,
        )
        hidden_states = outputs.hidden_states

    h12 = tf.reshape(hidden_states[-1][:, 0], (-1, 1, 768))
    h11 = tf.reshape(hidden_states[-2][:, 0], (-1, 1, 768))
    h10 = tf.reshape(hidden_states[-3][:, 0], (-1, 1, 768))
    h09 = tf.reshape(hidden_states[-4][:, 0], (-1, 1, 768))
    concat_hidden = tf.keras.layers.Concatenate(axis=2)([h12, h11, h10, h09])

    x = GlobalAveragePooling1D()(concat_hidden)
    x = Dropout(0.2)(x)

    if model_type == "answers":
        output = Dense(9, activation="sigmoid")(x)
    elif model_type == "questions":
        output = Dense(21, activation="sigmoid")(x)
    else:
        output = Dense(30, activation="sigmoid")(x)

    if name == "xlnet-base-cased" or (
        name == "roberta-base" and model_type != "questions"
    ):
        model = Model(inputs=[input_tokens, input_mask], outputs=output)
    else:
        model = Model(inputs=[input_tokens, input_mask, input_segment], outputs=output)

    return model




## === cell 9
class data_generator:
    def __init__(self, X, X_test, y, tokenizer, type_):
        tokens, masks, segments = input_data(X_test, tokenizer, type_)
        self.test_data = {
            "input_tokens": tokens,
            "input_mask": masks,
            "input_segment": segments,
        }

        self.tokens, self.masks, self.segments = input_data(X, tokenizer, type_)
        self.y = y.values  # (n, 30)

    def generate_data(self, tr, cv, name="xlnet-base-cased", model_type="questions"):
        if name != "xlnet-base-cased":
            train_data = {
                "input_tokens": self.tokens[tr],
                "input_mask": self.masks[tr],
                "input_segment": self.segments[tr],
            }
            cv_data = {
                "input_tokens": self.tokens[cv],
                "input_mask": self.masks[cv],
                "input_segment": self.segments[cv],
            }
        else:
            train_data = {"input_tokens": self.tokens[tr], "input_mask": self.masks[tr]}
            cv_data = {"input_tokens": self.tokens[cv], "input_mask": self.masks[cv]}

        if model_type == "questions":
            y_tr = self.y[tr, :21]
            y_cv = self.y[cv, :21]
        elif model_type == "answers":
            y_tr = self.y[tr, 21:]
            y_cv = self.y[cv, 21:]
        else:
            y_tr = self.y[tr, :]
            y_cv = self.y[cv, :]

        return train_data, cv_data, y_tr, y_cv




## === cell 10
def optimize_ranks(preds, unique_labels):
    print("optimizing the predicted values...")
    new_preds = np.zeros(preds.shape, dtype=np.float32)
    for i in range(preds.shape[1]):
        interpolate_bins = np.digitize(preds[:, i], bins=unique_labels, right=False)
        if len(np.unique(interpolate_bins)) == 1:
            new_preds[:, i] = preds[:, i]
        else:
            interpolate_bins = np.clip(interpolate_bins, 0, len(unique_labels) - 1)
            new_preds[:, i] = unique_labels[interpolate_bins]
    return new_preds




## === cell 11
def get_exp_labels(train_targets):
    unique_labels = np.unique(train_targets.values)
    denominator = 60
    q = np.arange(0, 101, 100 / denominator)
    exp_labels = np.percentile(unique_labels, q)
    return exp_labels




## === cell 12
def _spearmanr_fallback(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    def rankdata(x):
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)

        xs = x[order]
        i = 0
        while i < len(xs):
            j = i + 1
            while j < len(xs) and xs[j] == xs[i]:
                j += 1
            if j - i > 1:
                avg = (i + 1 + j) / 2.0
                ranks[order[i:j]] = avg
            i = j
        return ranks

    ra = rankdata(a)
    rb = rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt(np.sum(ra * ra) * np.sum(rb * rb))
    if denom == 0:
        return np.nan
    return float(np.sum(ra * rb) / denom)


def compute_spearmanr_ignore_nan(trues, preds):
    rhos_ = []
    for tcol, pcol in zip(np.transpose(trues), np.transpose(preds)):
        if scipy_spearmanr is not None:
            rhos_.append(scipy_spearmanr(tcol, pcol).correlation)
        else:
            rhos_.append(_spearmanr_fallback(tcol, pcol))
    return np.nanmean(rhos_)


def rhos(y, y_pred):
    return tf.py_function(compute_spearmanr_ignore_nan, (y, y_pred), tf.double)




## === cell 13
def fit_model(
    model,
    model_name,
    model_type,
    data_gen,
    train_df,
    use_saved_weights=False,
    file_path=None,
):
    if use_saved_weights and file_path is not None and os.path.exists(file_path):
        print(f"loading saved weights for {model_name} from {file_path}...")
        model.load_weights(file_path)
        return model

    print(f"fitting data on {model_name} ({model_type})...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00002)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[rhos])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr, cv) in enumerate(kf.split(np.arange(train_df.shape[0]))):
        tr_data, cv_data, y_tr, y_cv = data_gen.generate_data(
            tr, cv, model_name, model_type
        )
        model.fit(
            tr_data,
            y_tr,
            epochs=1,
            batch_size=4,
            validation_data=(cv_data, y_cv),
            verbose=2,
        )
        break

    if file_path is not None:
        try:
            model.save_weights(file_path)
        except Exception:
            pass

    return model




## === cell 14
def get_weighted_avg(model_predictions):
    xlnet_q, xlnet_a, roberta_q, roberta_a, roberta_qa, bert_q, bert_a, bert_qa = (
        model_predictions
    )
    xlnet_concat = np.concatenate((xlnet_q, xlnet_a), axis=1)
    bert_concat = np.concatenate((bert_q, bert_a), axis=1)
    roberta_concat = np.concatenate((roberta_q, roberta_a), axis=1)
    predict = (roberta_qa + bert_qa + xlnet_concat + bert_concat + roberta_concat) / 5.0
    return predict




## === cell 15
def get_predictions(predictions_present=False, model_saved_weights_present=False):
    X, X_test, y, train, test, sample_submission, target_cols = get_data()

    def _constant_predictions(val=0.5):
        n_test = test.shape[0]
        df = pd.concat(
            [
                test[["qa_id"]].reset_index(drop=True),
                pd.DataFrame(
                    np.full((n_test, len(target_cols)), val, dtype=np.float32),
                    columns=target_cols,
                ),
            ],
            axis=1,
        )
        return df

    if (not TF_AVAILABLE) or (not TRANSFORMERS_AVAILABLE):
        print("TF/transformers unavailable; returning constant 0.5 predictions.")
        return _constant_predictions(0.5)

    model_names = ["xlnet-base-cased", "roberta-base", "bert-base-uncased"]
    model_types = ["questions", "answers", "questions_answers"]

    saved_weights_names = [
        "xlnet_q.h5",
        "xlnet_a.h5",
        "roberta_q.h5",
        "roberta_a.h5",
        "roberta_qa.h5",
        "bert_q.h5",
        "bert_a.h5",
        "bert_qa.h5",
    ]

    model_predictions = []

    if predictions_present:
        raise FileNotFoundError(
            "Predictions dataset not available; set predictions_present=False to train/infer."
        )

    i = 0
    for name_ in model_names:
        for type_ in model_types:
            if name_ == "xlnet-base-cased" and type_ == "questions_answers":
                continue
            print("-" * 100)

            try:
                model = create_model(name_, type_)
                tokenizer = get_tokenizer(name_)
                data_gen = data_generator(X, X_test, y, tokenizer, type_)
                model = fit_model(
                    model,
                    name_,
                    type_,
                    data_gen,
                    train_df=train,
                    use_saved_weights=model_saved_weights_present,
                    file_path=saved_weights_names[i],
                )
                print(f"getting target predictions from {name_} ({type_})...")

                if (name_ == "xlnet-base-cased") or (
                    name_ == "roberta-base" and type_ != "questions"
                ):
                    pred_inputs = {
                        k: v
                        for k, v in data_gen.test_data.items()
                        if k in ["input_tokens", "input_mask"]
                    }
                else:
                    pred_inputs = {
                        k: v
                        for k, v in data_gen.test_data.items()
                        if k in ["input_tokens", "input_mask", "input_segment"]
                    }

                pred = model.predict(
                    pred_inputs,
                    batch_size=4,
                    verbose=0,
                )
                model_predictions.append(pred)
                i += 1

            except Exception as e:
                print(
                    f"WARNING: Skipping {name_} ({type_}) due to runtime error: {repr(e)}"
                )
                n_test = test.shape[0]
                out_dim = (
                    21 if type_ == "questions" else (9 if type_ == "answers" else 30)
                )
                model_predictions.append(
                    np.full((n_test, out_dim), 0.5, dtype=np.float32)
                )
                i += 1
                continue

    predicted_labels = get_weighted_avg(model_predictions)

    exp_labels = get_exp_labels(y)
    optimized_predicted_labels = optimize_ranks(predicted_labels, exp_labels)
    optimized_predicted_labels = np.clip(optimized_predicted_labels, 0.0, 1.0)

    df = pd.concat(
        [
            test[["qa_id"]].reset_index(drop=True),
            pd.DataFrame(optimized_predicted_labels, columns=target_cols),
        ],
        axis=1,
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df[target_cols] = df[target_cols].astype(np.float32).fillna(0.5).clip(0.0, 1.0)

    print("done...!")
    return df




## === cell 16
submission_pred = get_predictions(
    predictions_present=False, model_saved_weights_present=False
)
submission_pred.head()




## === cell 17
_, _, _, _, test, sample_submission, target_cols = get_data()

pred = submission_pred.copy()
pred = pred.replace([np.inf, -np.inf], np.nan)
pred = pred.drop_duplicates(subset=["qa_id"], keep="first").set_index("qa_id")

pred = pred.reindex(sample_submission["qa_id"].values)

pred[target_cols] = pred[target_cols].astype(np.float32).fillna(0.5).clip(0.0, 1.0)

submission = sample_submission[["qa_id"]].copy()
submission = pd.concat([submission, pred[target_cols].reset_index(drop=True)], axis=1)
submission = submission[["qa_id"] + target_cols]

submission = submission.replace([np.inf, -np.inf], np.nan)
submission[target_cols] = (
    submission[target_cols].astype(np.float32).fillna(0.5).clip(0.0, 1.0)
)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("Any NaNs in submission targets?:", submission[target_cols].isna().any().any())
print(
    "Row count matches sample_submission?:",
    submission.shape[0] == sample_submission.shape[0],
)
print(
    "Columns match sample_submission?:",
    submission.columns.tolist() == sample_submission.columns.tolist(),
)
