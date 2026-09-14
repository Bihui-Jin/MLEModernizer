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

3.12

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

0.3404916644532028

# 6. Current score

nan

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved nan) has done: 'I fix the immediate runtime blocker coming from `transformers` on Python 3.12 by removing the dependency on `BertTokenizer/BertForSequenceClassification` and instead loading the same saved fold models directly from the provided `.pt` checkpoints (the rest of the pipeline—tokenization → ids/masks/segments → fold ensembling → sigmoid → rounding—stays intact). I also make the data paths robust to both `/kaggle/input/google-quest-challenge/...` and `/kaggle/input/...` layouts, and add clear guards if checkpoints are missing. Finally, I ensure `submission.csv` is always written with the correct `qa_id` alignment and 30 target columns in `[0,1]`, so you get a valid file end-to-end.'
- What this solution (achieved nan) has done: 'I fix the immediate runtime blocker by making the vocab loading robust: if the BERT vocab file isn’t present in this Kaggle environment, the code automatically fall back to a small, self-contained tokenizer/vocab so the pipeline can still run end-to-end. I also make checkpoint discovery robust by searching under `/kaggle/input` for `best_param_score_*.pt` files and, if none exist, fall back to a deterministic baseline prediction (column-wise means from train) so a valid submission is always produced (this also avoids NaN score from missing/invalid submissions). Finally, I fix the `NameError` cascade caused by the earlier failure (by ensuring the tokenizer/input functions are defined before use) and keep the rest of the modeling/prediction logic unchanged when checkpoints are found.'

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd


def _resolve_path(*candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the candidate paths exist: {candidates}")


TRAIN_PATH = _resolve_path(
    "/kaggle/input/google-quest-challenge/train.csv",
    "/kaggle/input/train.csv",
)
TEST_PATH = _resolve_path(
    "/kaggle/input/google-quest-challenge/test.csv",
    "/kaggle/input/test.csv",
)
SAMPLE_SUB_PATH = _resolve_path(
    "/kaggle/input/google-quest-challenge/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
)

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

target_cols = [c for c in sample_sub.columns if c != "qa_id"]
assert (
    len(target_cols) == 30
), f"Expected 30 targets from sample_submission, got {len(target_cols)}"

input_categories = list(
    train.columns[[1, 2, 5]]
)  # ['question_title','question_body','answer']
print("Input text columns:", input_categories)
print("Train shape:", train.shape, "Test shape:", test.shape)
print("Sample submission shape:", sample_sub.shape)



## === cell 1
from math import floor, ceil
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


def _find_bert_vocab():
    candidates = [
        "/kaggle/input/bert-base-uncased/vocab.txt",
        "/kaggle/input/bert-base-uncased/bert-base-uncased/vocab.txt",
        "/kaggle/input/google-quest-challenge/vocab.txt",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    for root, _, files in os.walk("/kaggle/input"):
        if "vocab.txt" in files:
            return os.path.join(root, "vocab.txt")
    return None


VOCAB_PATH = _find_bert_vocab()
print("VOCAB_PATH:", VOCAB_PATH)


class SimpleBertTokenizer:
    def __init__(self, vocab_path=None, do_lower_case=True):
        self.do_lower_case = do_lower_case

        if vocab_path is None:
            vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        else:
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = [line.strip() for line in f if line.strip()]
            self.token_to_id = {tok: i for i, tok in enumerate(vocab)}

        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.unk_id = self.token_to_id.get(self.unk_token, 1)
        self.pad_id = self.token_to_id.get(self.pad_token, 0)
        self.vocab_size = max(self.token_to_id.values()) + 1

    def _is_whitespace(self, ch):
        return ch.isspace()

    def _is_punct(self, ch):
        o = ord(ch)
        if (33 <= o <= 47) or (58 <= o <= 64) or (91 <= o <= 96) or (123 <= o <= 126):
            return True
        import unicodedata

        return unicodedata.category(ch).startswith("P")

    def _basic_tokenize(self, text):
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return []
        text = str(text)
        if self.do_lower_case:
            text = text.lower()
        tokens = []
        buff = []
        for ch in text:
            if self._is_whitespace(ch):
                if buff:
                    tokens.append("".join(buff))
                    buff = []
            elif self._is_punct(ch):
                if buff:
                    tokens.append("".join(buff))
                    buff = []
                tokens.append(ch)
            else:
                buff.append(ch)
        if buff:
            tokens.append("".join(buff))
        return tokens

    def _wordpiece_tokenize(self, token, max_input_chars_per_word=100):
        if self.vocab_size <= 10:
            return [token]

        if len(token) > max_input_chars_per_word:
            return [self.unk_token]
        if token in self.token_to_id:
            return [token]
        chars = list(token)
        sub_tokens = []
        start = 0
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                piece = "".join(chars[start:end])
                if start > 0:
                    piece = "##" + piece
                if piece in self.token_to_id:
                    cur_substr = piece
                    break
                end -= 1
            if cur_substr is None:
                return [self.unk_token]
            sub_tokens.append(cur_substr)
            start = end
        return sub_tokens

    def tokenize(self, text):
        out = []
        for tok in self._basic_tokenize(text):
            out.extend(self._wordpiece_tokenize(tok))
        return out

    def convert_tokens_to_ids(self, tokens):
        if self.vocab_size <= 10:
            ids = []
            for t in tokens:
                if t in self.token_to_id:
                    ids.append(self.token_to_id[t])
                else:
                    h = (abs(hash(t)) % 20000) + 5
                    ids.append(h)
            return ids
        return [self.token_to_id.get(t, self.unk_id) for t in tokens]


tokenizer = SimpleBertTokenizer(VOCAB_PATH, do_lower_case=True)


def _get_masks(tokens, max_seq_length):
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def _get_segments(tokens, max_seq_length):
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    first_sep = True
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def _get_ids(tokens, tokenizer, max_seq_length):
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
    return input_ids


def _trim_input(
    title,
    question,
    answer,
    max_sequence_length=512,
    t_max_len=30,
    q_max_len=239,
    a_max_len=239,
):
    question = "" if pd.isna(question) else question
    answer = "" if pd.isna(answer) else answer
    title = "" if pd.isna(title) else title

    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)

    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len + q_len + a_len + 4) > max_sequence_length:
        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len) / 2)
            q_max_len = q_max_len + ceil((t_max_len - t_len) / 2)
        else:
            t_new_len = t_max_len

        if a_max_len > a_len:
            a_new_len = a_len
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len

        total = t_new_len + a_new_len + q_new_len + 4
        if total > max_sequence_length:
            overflow = total - max_sequence_length
            reduce_a = min(overflow, a_new_len)
            a_new_len -= reduce_a
            overflow -= reduce_a
            if overflow > 0:
                q_new_len = max(0, q_new_len - overflow)

        t = t[:t_new_len]
        q = q[:q_new_len]
        a = a[:a_new_len]

    return t, q, a


def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
    stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]
    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)
    return [input_ids, input_masks, input_segments]


def compute_input_arays(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm(df[columns].iterrows(), total=len(df)):
        t, q, a = instance.question_title, instance.question_body, instance.answer
        t, q, a = _trim_input(t, q, a, max_sequence_length)
        ids, masks, segments = _convert_to_bert_inputs(
            t, q, a, tokenizer, max_sequence_length
        )
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
    return [
        torch.from_numpy(np.asarray(input_ids, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_masks, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_segments, dtype=np.int32)).long(),
    ]




## === cell 2
def predict_result(model, test_loader, batch_size=64):
    test_preds = np.zeros((len(test_loader.dataset), 30), dtype=np.float32)

    model.eval()
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    for idx, x_batch in tk0:
        with torch.no_grad():
            try:
                outputs = model(
                    input_ids=x_batch[0].to(device),
                    labels=None,
                    attention_mask=x_batch[1].to(device),
                    token_type_ids=x_batch[2].to(device),
                )
                logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            except TypeError:
                logits = model(
                    x_batch[0].to(device),
                    x_batch[1].to(device),
                    x_batch[2].to(device),
                )

            batch_np = logits.detach().cpu().numpy()
            start = idx * batch_size
            end = start + batch_np.shape[0]
            test_preds[start:end] = batch_np

    output = torch.sigmoid(torch.from_numpy(test_preds)).numpy()
    return output




## === cell 3
class QuestDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, lengths, labels=None):
        self.inputs = inputs
        self.labels = labels if labels is not None else None
        self.lengths = lengths

    def __getitem__(self, idx):
        input_ids = self.inputs[0][idx]
        input_masks = self.inputs[1][idx]
        input_segments = self.inputs[2][idx]
        lengths = self.lengths[idx]
        if self.labels is not None:
            labels = self.labels[idx]
            return input_ids, input_masks, input_segments, labels, lengths
        return input_ids, input_masks, input_segments, lengths

    def __len__(self):
        return len(self.inputs[0])




## === cell 4
def _find_ckpt_dir():
    if os.path.isdir("/kaggle/input/pt-params"):
        return "/kaggle/input/pt-params"
    best_dir = None
    best_count = 0
    pat = re.compile(r"best_param_score_(\d+)\.pt$")
    for dirname, _, filenames in os.walk("/kaggle/input"):
        hits = [fn for fn in filenames if pat.match(fn)]
        if len(hits) > best_count:
            best_count = len(hits)
            best_dir = dirname
    return best_dir


CKPT_DIR = _find_ckpt_dir()
print("Checkpoint dir:", CKPT_DIR)

test_inputs = compute_input_arays(
    test, input_categories, tokenizer, max_sequence_length=512
)
lengths_test = np.argmax(test_inputs[0].numpy() == 0, axis=1)
lengths_test[lengths_test == 0] = test_inputs[0].shape[1]
test_set = QuestDataset(inputs=test_inputs, lengths=lengths_test, labels=None)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)



## === cell 5
NUM_FOLDS = 5
result = None


def _ckpt_paths(ckpt_dir, num_folds=5):
    if ckpt_dir is None:
        return []
    paths = []
    for fold in range(1, num_folds + 1):
        p = os.path.join(ckpt_dir, f"best_param_score_{fold}.pt")
        if os.path.exists(p):
            paths.append(p)
    return paths


ckpt_paths = _ckpt_paths(CKPT_DIR, NUM_FOLDS)
print("Found checkpoints:", len(ckpt_paths), ckpt_paths[:3])

if len(ckpt_paths) == NUM_FOLDS:
    result = np.zeros((len(test), 30), dtype=np.float32)

    with torch.no_grad():
        for fold in range(1, NUM_FOLDS + 1):
            ckpt_path = os.path.join(CKPT_DIR, f"best_param_score_{fold}.pt")

            try:
                model = torch.load(ckpt_path, map_location=device, weights_only=False)
            except TypeError:
                model = torch.load(ckpt_path, map_location=device)

            if hasattr(model, "to"):
                model.to(device)
            model.eval()

            fold_result = predict_result(model, test_loader, batch_size=64)
            result += fold_result.astype(np.float32)

    result /= NUM_FOLDS
    print(
        "Ensemble preds shape:",
        result.shape,
        "min/max:",
        float(result.min()),
        float(result.max()),
    )
else:
    train_targets = train[target_cols].astype(np.float32)
    col_means = train_targets.mean(axis=0).to_numpy()
    result = np.tile(col_means.reshape(1, -1), (len(test), 1)).astype(np.float32)
    result = np.clip(result, 0.0, 1.0)
    print(
        "Fallback mean preds used. shape:",
        result.shape,
        "min/max:",
        float(result.min()),
        float(result.max()),
    )



## === cell 6
submission = pd.read_csv(SAMPLE_SUB_PATH)

raters = np.array(
    [
        18,
        18,
        6,
        6,
        6,
        6,
        18,
        18,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        3,
        18,
        18,
        18,
        18,
        18,
        90,
        6,
        6,
        6,
        18,
    ],
    dtype=np.float32,
)

mins = np.min(result, axis=0)
maxs = np.max(result, axis=0)
den = maxs - mins
den[den == 0] = 1.0
result_norm = (result - mins) / den
result_norm = np.clip(result_norm, 0.0, 1.0)

result_rounded = (np.round(raters * result_norm).astype(np.float32) / raters).astype(
    np.float32
)
result_rounded = np.clip(result_rounded, 0.0, 1.0)

if "qa_id" not in submission.columns:
    raise ValueError("sample_submission.csv missing qa_id column.")

pred_df = pd.DataFrame(result_rounded, columns=target_cols)
pred_df["qa_id"] = test["qa_id"].values

if len(submission) == len(test) and submission["qa_id"].astype(str).equals(
    test["qa_id"].astype(str)
):
    submission.loc[:, target_cols] = result_rounded
else:
    submission = submission.drop(columns=target_cols, errors="ignore").merge(
        pred_df, on="qa_id", how="left"
    )
    submission = submission[["qa_id"] + target_cols]

submission[target_cols] = submission[target_cols].astype(np.float32).clip(0.0, 1.0)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
assert submission.shape[1] == 31 and submission.shape[0] == len(test)
assert (
    submission[target_cols].min().min() >= 0.0
    and submission[target_cols].max().max() <= 1.0
)
