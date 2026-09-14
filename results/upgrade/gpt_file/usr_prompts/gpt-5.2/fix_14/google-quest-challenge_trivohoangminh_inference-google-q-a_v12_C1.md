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

0.00265

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved nan) has done: 'I fix the immediate runtime blocker coming from `transformers` on Python 3.12 by removing the dependency on `BertTokenizer/BertForSequenceClassification` and instead loading the same saved fold models directly from the provided `.pt` checkpoints (the rest of the pipeline—tokenization → ids/masks/segments → fold ensembling → sigmoid → rounding—stays intact). I also make the data paths robust to both `/kaggle/input/google-quest-challenge/...` and `/kaggle/input/...` layouts, and add clear guards if checkpoints are missing. Finally, I ensure `submission.csv` is always written with the correct `qa_id` alignment and 30 target columns in `[0,1]`, so you get a valid file end-to-end.'
- What this solution (achieved nan) has done: 'I fix the immediate runtime blocker by making the vocab loading robust: if the BERT vocab file isn’t present in this Kaggle environment, the code automatically fall back to a small, self-contained tokenizer/vocab so the pipeline can still run end-to-end. I also make checkpoint discovery robust by searching under `/kaggle/input` for `best_param_score_*.pt` files and, if none exist, fall back to a deterministic baseline prediction (column-wise means from train) so a valid submission is always produced (this also avoids NaN score from missing/invalid submissions). Finally, I fix the `NameError` cascade caused by the earlier failure (by ensuring the tokenizer/input functions are defined before use) and keep the rest of the modeling/prediction logic unchanged when checkpoints are found.'
- What this solution (achieved nan) has done: 'Your current “nan” score is most plausibly coming from invalid predictions for Spearman (e.g., NaNs/inf) produced by the min-max normalization step when any fold output contains NaNs, which then propagates through rounding into the submission. I keep your pipeline identical (same tokenization → fold ensembling → sigmoid → rounding), but add a minimal sanitization step right after each fold prediction and after ensembling to replace any non-finite values with safe finite numbers and clip to [0,1]. I also make the length computation robust (avoid the “all zeros” edge case) without changing semantics, and keep the submission alignment logic unchanged so you always get a valid `submission.csv`. These changes should convert the “nan” submission into a valid one and typically improve the score toward your target by enabling real evaluation.'
- What this solution (achieved nan) has done: 'Your “nan” score most likely comes from producing a submission that Kaggle can’t score correctly (row count mismatch) because you start from `sample_submission.csv` (608 rows) while the provided `test.csv` has 19,550 rows, so many `qa_id`s end up missing predictions after the merge. I keep your modeling/prediction pipeline intact and only change submission construction to always use `test[['qa_id']]` as the base, guaranteeing exactly one prediction row per test row in the right order. I also enforce a final safety pass that fills any missing target values (from any unexpected join issues) with 0.5 and clips to `[0,1]`, which keeps the evaluation semantics but prevents invalid scoring. These minimal fixes should convert the “nan” into a valid scored submission, moving you toward the target score.'
- What this solution (achieved 0.00265) has done: 'Your “nan” score is almost certainly because the current pipeline can silently produce an unscorable submission when the fold checkpoints aren’t found (it falls back to constant column means, which yields undefined Spearman on Kaggle for constant columns), or because the min-max normalization step can collapse a column to a constant after clipping/rounding. I keep the model/tokenization/inference core logic intact, but make two minimal changes: (1) if checkpoints are missing, use a deterministic per-target *rank-based* baseline (still in [0,1]) derived from simple text-length signals so predictions aren’t constant; (2) make the post-processing strictly monotonic per column by using per-column rank normalization instead of min-max, which avoids zero-variance columns and prevents NaN Spearman. These changes are aimed at converting “nan” into a valid scored submission and moving the score upward toward your target without changing the architecture or training logic.'
- What this solution (achieved 0.00265) has done: 'Your score is far below the target, so we should improve it with minimal, low-risk changes that keep your overall pipeline intact. The biggest current score-killer is the final per-column rank-normalization plus rater-rounding: it destroys cross-sample relative structure produced by the model and can drastically reduce Spearman. I keep your tokenization, checkpoint loading, fold ensembling, and sigmoid exactly as-is, but change post-processing to (1) remove the rank-normalization and rounding when real model checkpoints are used, and (2) keep the rank-based fallback only for the “no checkpoints” case to avoid constant predictions. This should move the score substantially upward toward your target while preserving the core inference logic and still writing a valid `submission.csv`.'
- What this solution (achieved 0.00265) has done: 'Your current score (0.00265) is far below the target (0.34049), and the most likely cause is that you are not actually using the real BERT checkpoints (so you’re falling back to a weak length-based rank baseline), and/or the loaded `.pt` objects are state_dicts that you treat as full models (leading to effectively broken predictions). I keep your tokenization, dataset, inference loop, sigmoid, and fold ensembling intact, but make checkpoint discovery/load robust: (1) search for both `best_param_score_*.pt` and common BERT fold filenames, and (2) if a checkpoint is a `state_dict`, load it into a small, compatible `BertLike` module that matches the expected forward signature and produces 30 logits. This should make `use_model_preds=True` in the intended environment and dramatically lift score toward the target, while keeping the overall pipeline semantics unchanged and still writing a valid `submission.csv`.'
- What this solution (achieved 0.00265) has done: 'Your score is far below the target, so the safest way to move toward it is to ensure you actually use a meaningful model rather than the weak fallback and to stop distorting model outputs. I keep your tokenization, dataset, inference loop, sigmoid, and fold ensembling intact, but make checkpoint discovery/load more compatible with common Kaggle notebooks by (1) recognizing more fold checkpoint filename patterns and (2) robustly extracting a `state_dict` even when it’s nested. I also enforce `strict=True` when we successfully detect a BERT-like checkpoint so we don’t silently run a randomly-initialized `BertLikeForQuest` (which would explain the ~0.0026 score), and only fall back if loading truly fails. These are minimal changes aimed at making `use_model_preds=True` with real weights and preserving the raw model ranking signal that Spearman rewards.'
- What this solution (achieved 0.00265) has done: 'Your score is far below the target, so the smallest likely fix is to ensure you’re not unintentionally using the weak fallback path or producing near-constant predictions due to checkpoint-loading mismatches. I keep your entire tokenization → dataloader → fold loop → sigmoid → averaging pipeline, but make checkpoint discovery more permissive (while still requiring exactly 5 folds) and make checkpoint loading stricter/safer by (a) correctly handling common nesting keys like `model_state_dict`, and (b) refusing to run a randomly-initialized `BertLikeForQuest` if weights don’t match (so we fall back to the non-constant rank baseline instead of bad model preds). Finally, I add a tiny verification that fold logits are non-degenerate (not all equal) before accepting model predictions; otherwise we fallback, which should move you upward from ~0.00265 toward your target by avoiding “broken model” outputs.'
- What this solution (achieved 0.00265) has done: 'Your score is extremely far below the target, so the most plausible minimal fix is to stop falling into the weak fallback path and/or producing effectively untrained outputs due to strict checkpoint loading mismatches. I (1) make checkpoint discovery more likely to find the common Google QUEST fold `.pt` files, and (2) make checkpoint loading compatible with the real architecture by allowing a safe “head-only mismatch” load (still using your same inference pipeline) instead of failing and falling back. To avoid silently using random weights (which would tank Spearman), I add a strict verification that key encoder weights actually loaded; otherwise we still fall back. These changes keep your tokenization → dataloader → fold ensembling → sigmoid → submission format intact, but should move you materially upward toward the target by enabling real checkpoint predictions.'
- What this solution (achieved 0.00265) has done: 'Your score is far below the target, so we should make the smallest changes that plausibly unlock “real” model signal for Spearman without changing your overall inference pipeline. The biggest likely issue is that your custom `BertLikeForQuest` is not actually compatible with typical Google QUEST BERT checkpoints (hidden size/layer types differ), so you either fall back or produce near-random outputs; instead, we load the actual `BertForSequenceClassification` from the checkpoint if `transformers` is available, and only use `BertLikeForQuest` as a last resort. We also remove the “degenerate prediction” fallback trigger for the model path (it can incorrectly reject a valid model ensemble) while keeping NaN/inf sanitization and `[0,1]` clipping intact. These are minimal, score-relevant changes that keep the same tokenization → ids/masks/segments → fold ensembling → sigmoid → submission semantics and should move you materially toward the target.'
- What this solution (achieved 0.00265) has done: 'Your score is far below target, so the smallest likely improvement is to ensure you actually use meaningful fold checkpoints when they exist and avoid silently running a partially/randomly loaded “BertLike” model. I (1) tighten checkpoint selection so we prefer the canonical `best_param_score_1..5.pt` set when available (instead of grabbing the first matching fold-like file from anywhere), (2) make the custom-model load stricter by requiring essential embedding+encoder weights to be present (otherwise we fall back to the non-constant rank baseline), and (3) keep model predictions un-distorted (no rank-normalize/rounding) as you already do for `use_model_preds=True`. These are minimal, score-relevant changes that should move you materially upward toward the target by preventing broken/near-random model inference. The script still runs end-to-end and always writes a valid `submission.csv`.'

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
    "/kaggle/data/google-quest-challenge/train.csv",
    "/kaggle/data/train.csv",
)
TEST_PATH = _resolve_path(
    "/kaggle/input/google-quest-challenge/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/data/google-quest-challenge/test.csv",
    "/kaggle/data/test.csv",
)
SAMPLE_SUB_PATH = _resolve_path(
    "/kaggle/input/google-quest-challenge/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/google-quest-challenge/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
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
        "/kaggle/data/google-quest-challenge/vocab.txt",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    for root, _, files in os.walk("/kaggle/input"):
        if "vocab.txt" in files:
            return os.path.join(root, "vocab.txt")
    for root, _, files in os.walk("/kaggle/data"):
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
import torch.nn as nn


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
    for dirname, _, filenames in os.walk("/kaggle/data"):
        hits = [fn for fn in filenames if pat.match(fn)]
        if len(hits) > best_count:
            best_count = len(hits)
            best_dir = dirname
    return best_dir


def _discover_fold_ckpts():
    cd = _find_ckpt_dir()
    if cd:
        canonical = [os.path.join(cd, f"best_param_score_{i}.pt") for i in range(1, 6)]
        if all(os.path.exists(p) for p in canonical):
            return canonical

    dirs = []
    if cd:
        dirs.append(cd)
    dirs.extend(
        [
            "/kaggle/input/google-quest-challenge",
            "/kaggle/input",
            "/kaggle/data/google-quest-challenge",
            "/kaggle/data",
        ]
    )
    dirs = [d for d in dirs if d and os.path.isdir(d)]

    fold_to_path = {}

    expected = re.compile(r"best_param_score_(\d+)\.pt$")
    common = re.compile(r".*(fold|_f|f)(\d+).*\.pt$", re.IGNORECASE)
    kfold = re.compile(r".*kfold[_-]?(\d+).*\.pt$", re.IGNORECASE)
    fold_plain = re.compile(r"^fold[_-]?(\d+)\.(pt|pth|bin)$", re.IGNORECASE)
    quest_common = re.compile(
        r".*(quest|google[-_]?quest).*?(fold|_f|f)(\d+).*\.pt$", re.IGNORECASE
    )

    for base in dirs:
        for root, _, files in os.walk(base):
            for fn in files:
                full = os.path.join(root, fn)

                m = expected.match(fn)
                if m:
                    fold = int(m.group(1))
                    if 1 <= fold <= 5:
                        fold_to_path.setdefault(fold, full)
                    continue

                m = quest_common.match(fn)
                if m:
                    f = int(m.group(3))
                    if 0 <= f <= 4:
                        fold_to_path.setdefault(f + 1, full)
                    elif 1 <= f <= 5:
                        fold_to_path.setdefault(f, full)
                    continue

                m = fold_plain.match(fn)
                if m:
                    f0 = int(m.group(1))
                    if 0 <= f0 <= 4:
                        fold_to_path.setdefault(f0 + 1, full)
                    elif 1 <= f0 <= 5:
                        fold_to_path.setdefault(f0, full)
                    continue

                m = kfold.match(fn)
                if m:
                    f0 = int(m.group(1))
                    if 0 <= f0 <= 4:
                        fold_to_path.setdefault(f0 + 1, full)
                    continue

                m = common.match(fn)
                if m:
                    f = int(m.group(2))
                    if 0 <= f <= 4:
                        fold_to_path.setdefault(f + 1, full)
                    elif 1 <= f <= 5:
                        fold_to_path.setdefault(f, full)

    paths = [fold_to_path.get(i) for i in range(1, 6)]
    if all(p is not None for p in paths):
        return paths
    return []


class BertLikeForQuest(nn.Module):
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_labels=30,
        max_position_embeddings=512,
        type_vocab_size=2,
    ):
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=0
        )
        self.embeddings.position_embeddings = nn.Embedding(
            max_position_embeddings, hidden_size
        )
        self.embeddings.token_type_embeddings = nn.Embedding(
            type_vocab_size, hidden_size
        )
        self.embeddings.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.embeddings.dropout = nn.Dropout(0.1)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=12,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=12)
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.pooler_activation = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None
    ):
        if input_ids is None:
            raise TypeError("input_ids required")

        bsz, seqlen = input_ids.shape
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        pos_ids = (
            torch.arange(seqlen, device=input_ids.device)
            .unsqueeze(0)
            .expand(bsz, seqlen)
        )
        x = (
            self.embeddings.word_embeddings(input_ids)
            + self.embeddings.position_embeddings(pos_ids)
            + self.embeddings.token_type_embeddings(token_type_ids)
        )
        x = self.embeddings.LayerNorm(x)
        x = self.embeddings.dropout(x)

        src_key_padding_mask = attention_mask.eq(0)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        cls = x[:, 0, :]
        pooled = self.pooler_activation(self.pooler(cls))
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return (logits,)


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "checkpoint",
        ):
            if k in obj:
                v = obj[k]
                if isinstance(v, dict) and any(
                    isinstance(t, torch.Tensor) for t in v.values()
                ):
                    return v
                if hasattr(v, "state_dict"):
                    try:
                        sd = v.state_dict()
                        if isinstance(sd, dict) and any(
                            isinstance(t, torch.Tensor) for t in sd.values()
                        ):
                            return sd
                    except Exception:
                        pass
        if all(isinstance(k, str) for k in obj.keys()) and any(
            isinstance(v, torch.Tensor) for v in obj.values()
        ):
            return obj
    return None


def _infer_vocab_size_from_state(state):
    vocab_size = 30522
    for k in (
        "embeddings.word_embeddings.weight",
        "bert.embeddings.word_embeddings.weight",
    ):
        if k in state and getattr(state[k], "ndim", 0) == 2:
            vocab_size = int(state[k].shape[0])
            break
    return vocab_size


def _try_load_transformers_model(path: str):
    try:
        from transformers import BertConfig, BertForSequenceClassification  # type: ignore
    except Exception:
        return None

    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    state = None
    if (
        hasattr(obj, "state_dict")
        and callable(getattr(obj, "state_dict"))
        and not isinstance(obj, dict)
    ):
        try:
            state = obj.state_dict()
        except Exception:
            state = None
    if state is None:
        state = _extract_state_dict(obj)
    if state is None:
        return None

    stripped = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk.replace("module.", "", 1)
        if nk.startswith("bert."):
            nk = nk.replace("bert.", "", 1)
        stripped[nk] = v

    has_bert_prefix = any(k.startswith("bert.") for k in state.keys())
    if has_bert_prefix:
        tf_state = {
            k.replace("module.", "", 1): v
            for k, v in state.items()
            if isinstance(k, str)
        }
    else:
        tf_state = {"bert." + k: v for k, v in stripped.items()}
        for k, v in stripped.items():
            if k.startswith("classifier.") or k.startswith("dropout."):
                tf_state[k] = v

    cfg = BertConfig(
        vocab_size=int(_infer_vocab_size_from_state(state)),
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        num_labels=30,
    )
    model = BertForSequenceClassification(cfg)

    incompatible = model.load_state_dict(tf_state, strict=False)

    loaded_any_encoder = any(
        ("bert.encoder.layer.0" in k) for k in tf_state.keys()
    ) and not any(
        k.startswith("bert.encoder.layer.0") for k in incompatible.missing_keys
    )
    loaded_embeddings = any(
        k.startswith("bert.embeddings.word_embeddings") for k in tf_state.keys()
    ) and (
        "bert.embeddings.word_embeddings.weight" not in set(incompatible.missing_keys)
    )

    if not (loaded_any_encoder and loaded_embeddings):
        return None

    if len(incompatible.unexpected_keys) > 0:
        print(
            f"Note: transformers unexpected keys in ckpt ({len(incompatible.unexpected_keys)}); continuing (strict=False)."
        )
    if len(incompatible.missing_keys) > 0:
        print(
            f"Note: transformers missing keys in ckpt ({len(incompatible.missing_keys)}); continuing (strict=False)."
        )
    return model


def _load_ckpt_as_model(path: str):
    m = _try_load_transformers_model(path)
    if m is not None:
        return m

    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=device)

    if (
        hasattr(obj, "eval")
        and hasattr(obj, "to")
        and callable(getattr(obj, "forward", None))
    ):
        return obj

    state = _extract_state_dict(obj)
    if state is None:
        raise TypeError(f"Unsupported checkpoint object type at {path}: {type(obj)}")

    vocab_size = _infer_vocab_size_from_state(state)
    model = BertLikeForQuest(vocab_size=vocab_size, hidden_size=768, num_labels=30)

    stripped = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("bert."):
            nk = nk.replace("bert.", "", 1)
        if nk.startswith("module."):
            nk = nk.replace("module.", "", 1)
        stripped[nk] = v

    incompatible = model.load_state_dict(stripped, strict=False)

    missing = set(incompatible.missing_keys)

    required_any = [
        "embeddings.word_embeddings.weight",
        "encoder.layers.0.linear1.weight",
        "encoder.layers.0.linear2.weight",
    ]
    if not any(k in stripped for k in required_any):
        raise RuntimeError(
            "Checkpoint missing essential backbone tensors; refusing to run random/incompatible model."
        )
    if "embeddings.word_embeddings.weight" in missing:
        raise RuntimeError(
            "Embeddings not loaded; refusing to run random/incompatible model."
        )
    if ("encoder.layers.0.linear1.weight" in missing) or (
        "encoder.layers.0.linear2.weight" in missing
    ):
        raise RuntimeError(
            "Encoder layer weights missing; refusing to run random/incompatible model."
        )

    if len(incompatible.unexpected_keys) > 0:
        print(
            f"Note: unexpected keys in ckpt ({len(incompatible.unexpected_keys)}); continuing (strict=False)."
        )
    if len(incompatible.missing_keys) > 0:
        print(
            f"Note: missing keys in ckpt ({len(incompatible.missing_keys)}); continuing (strict=False)."
        )

    return model


CKPT_DIR = _find_ckpt_dir()
print("Checkpoint dir (best guess):", CKPT_DIR)

test_inputs = compute_input_arays(
    test, input_categories, tokenizer, max_sequence_length=512
)

pad_pos = test_inputs[0].numpy() == 0
lengths_test = pad_pos.argmax(axis=1)
no_pad = ~pad_pos.any(axis=1)
lengths_test[no_pad] = test_inputs[0].shape[1]
lengths_test[lengths_test == 0] = test_inputs[0].shape[1]

test_set = QuestDataset(inputs=test_inputs, lengths=lengths_test, labels=None)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)




## === cell 5
NUM_FOLDS = 5
result = None


def _rank_normalize_cols(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x, dtype=np.float32)
    n = x.shape[0]
    if n <= 1:
        return np.zeros_like(x, dtype=np.float32) + 0.5
    denom = float(n - 1)
    for j in range(x.shape[1]):
        col = x[:, j]
        order = np.argsort(col, kind="mergesort")
        ranks = np.empty(n, dtype=np.float32)
        ranks[order] = np.arange(n, dtype=np.float32)
        out[:, j] = ranks / denom
    return out


ckpt_paths = _discover_fold_ckpts()
print("Found fold checkpoints:", len(ckpt_paths), ckpt_paths[:3])

use_model_preds = len(ckpt_paths) == NUM_FOLDS

if use_model_preds:
    result = np.zeros((len(test), 30), dtype=np.float32)

    ok = True
    with torch.no_grad():
        for fold_idx, ckpt_path in enumerate(ckpt_paths, start=1):
            try:
                model = _load_ckpt_as_model(ckpt_path)
            except Exception as e:
                print(
                    f"Fold {fold_idx}: checkpoint load failed, falling back. Path={ckpt_path}. Error={repr(e)}"
                )
                ok = False
                break

            if hasattr(model, "to"):
                model.to(device)
            model.eval()

            fold_result = predict_result(model, test_loader, batch_size=64).astype(
                np.float32
            )
            fold_result = np.nan_to_num(
                fold_result, nan=0.5, posinf=1.0, neginf=0.0
            ).astype(np.float32)
            fold_result = np.clip(fold_result, 0.0, 1.0)

            if not np.isfinite(fold_result).all():
                print(
                    f"Fold {fold_idx}: non-finite predictions detected, falling back."
                )
                ok = False
                break

            result += fold_result

    if ok:
        result /= NUM_FOLDS
        result = np.nan_to_num(result, nan=0.5, posinf=1.0, neginf=0.0).astype(
            np.float32
        )
        result = np.clip(result, 0.0, 1.0)
        if not np.isfinite(result).all():
            print("Ensemble predictions non-finite; switching to fallback baseline.")
            use_model_preds = False

    if use_model_preds:
        print(
            "Ensemble preds shape:",
            result.shape,
            "min/max:",
            float(result.min()),
            float(result.max()),
        )

if not use_model_preds:
    qt = (
        test["question_title"]
        .fillna("")
        .astype(str)
        .str.len()
        .to_numpy(dtype=np.float32)
    )
    qb = (
        test["question_body"]
        .fillna("")
        .astype(str)
        .str.len()
        .to_numpy(dtype=np.float32)
    )
    ans = test["answer"].fillna("").astype(str).str.len().to_numpy(dtype=np.float32)
    base_signal = 0.25 * qt + 0.5 * qb + 0.25 * ans
    scales = (np.arange(30, dtype=np.float32) + 1.0) / 30.0
    raw = base_signal.reshape(-1, 1) * scales.reshape(1, -1)
    result = _rank_normalize_cols(raw).astype(np.float32)
    result = np.clip(result, 0.0, 1.0)
    print(
        "Fallback rank-baseline preds used. shape:",
        result.shape,
        "min/max:",
        float(result.min()),
        float(result.max()),
    )




## === cell 6
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

result = np.nan_to_num(result, nan=0.5, posinf=1.0, neginf=0.0).astype(np.float32)
result = np.clip(result, 0.0, 1.0)

if use_model_preds:
    final_pred = result
else:
    result_norm = _rank_normalize_cols(result)
    result_norm = np.clip(result_norm, 0.0, 1.0)
    final_pred = (np.round(raters * result_norm).astype(np.float32) / raters).astype(
        np.float32
    )
    final_pred = np.clip(final_pred, 0.0, 1.0)

submission = pd.DataFrame({"qa_id": test["qa_id"].values})
for j, col in enumerate(target_cols):
    submission[col] = final_pred[:, j].astype(np.float32)

submission[target_cols] = submission[target_cols].replace([np.inf, -np.inf], np.nan)
submission[target_cols] = (
    submission[target_cols].fillna(0.5).astype(np.float32).clip(0.0, 1.0)
)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

assert submission.shape[1] == 31 and submission.shape[0] == len(test)
assert submission["qa_id"].astype(str).equals(test["qa_id"].astype(str))
assert (
    submission[target_cols].min().min() >= 0.0
    and submission[target_cols].max().max() <= 1.0
)
assert np.isfinite(submission[target_cols].to_numpy()).all()
