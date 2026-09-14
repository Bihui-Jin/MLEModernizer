# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
For each article + question pair, you must predict / select long and short form answers to the question drawn *directly from the article*.

- A long answer would be a longer section of text that answers the question - several sentences or a paragraph.
- A short answer might be a sentence or phrase, or even in some cases a YES/NO. The short answers are always contained within / a subset of one of the plausible long answers.
- A given article can (and very often will) allow for both long *and* short answers, depending on the question.

There is more detail about the data and what you're predicting [on the Github page for the Natural Questions dataset](https://github.com/google-research-datasets/natural-questions/blob/master/README.md). This page also contains helpful utilities and scripts. Note that we are using the simplified text version of the data - most of the HTML tags have been removed, and only those necessary to break up paragraphs / sections are included.

## Metric
Micro F1. Predicted long and short answers must match exactly the token indices of one of the ground truth labels ((or match YES/NO if the question has a yes/no short answer). There may be up to five labels for long answers, and more for short. If no answer applies, leave the prediction blank/null.

## Submission Format
For each ID in the test set, you must predict a) a set of start:end token indices, b) a YES/NO answer if applicable (short answers ONLY), or c) a BLANK answer if no prediction can be made. The file should contain a header and have the following format:

```
-7853356005143141653_long,6:18
-7853356005143141653_short,YES
-545833482873225036_long,105:200
-545833482873225036_short,
-6998273848279890840_long,
-6998273848279890840_short,NO
```
`
## Data
Each sample contains a Wikipedia article, a related question, and the candidate long form answers. The training examples also provide the correct long and short form answer or answers for the sample, if any exist.

- **simplified-nq-train.jsonl** - the training data, in newline-delimited JSON format.
- **simplified-nq-kaggle-test.jsonl** - the test data, in newline-delimited JSON format.
- **sample_submission.csv** - a sample submission file in the correct format

### Data fields
- **document_text** - the text of the article in question (with some HTML tags to provide document structure). The text can be tokenized by splitting on whitespace.
- **question_text** - the question to be answered
- **long_answer_candidates** - a JSON array containing all of the plausible long answers.
- **annotations** - a JSON array containing all of the correct long + short answers. Only provided for train.
- **document_url** - the URL for the full article. Provided for informational purposes only. This is NOT the simplified version of the article so indices from this cannot be used directly. The content may also no longer match the html used to generate document_text. Only provided for train.
- **example_id** - unique ID for the sample.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (173 lines)
            sample_submission.csv (61477 lines)
            sample_submission.csv.zip (460.6 kB)
            simplified-nq-test.jsonl (1.7 GB)
            simplified-nq-train.jsonl (15.7 GB)
            tensorflow2-question-answering/
                description.md (173 lines)
                sample_submission.csv (61477 lines)
                ... and 3 other files
                tensorflow2-question-answering/
        input/
            description.md (173 lines)
            sample_submission.csv (61477 lines)
            sample_submission.csv.zip (460.6 kB)
            simplified-nq-test.jsonl (1.7 GB)
            simplified-nq-train.jsonl (15.7 GB)
            tensorflow2-question-answering/
                description.md (173 lines)
                sample_submission.csv (61477 lines)
                ... and 3 other files
                tensorflow2-question-answering/
        working/
            tensorflow2-question-answering/
                description.md (173 lines)
                sample_submission.csv (61477 lines)
                ... and 3 other files
                tensorflow2-question-answering/
```

-> data/sample_submission.csv has 61476 rows and 2 columns.
Here is some information about the columns:
PredictionString (float64) has 0 unique values: [nan]
example_id (object) has 2000 unique values. Some example values: ['-4639148749459150090_long', '-257485578111885185_short', '-257485578111885185_long', '-9110190923673509457_short']

-> data/tensorflow2-question-answering/sample_submission.csv has 61476 rows and 2 columns.
Here is some information about the columns:
PredictionString (float64) has 0 unique values: [nan]
example_id (object) has 2000 unique values. Some example values: ['-4639148749459150090_long', '-257485578111885185_short', '-257485578111885185_long', '-9110190923673509457_short']

-> input/sample_submission.csv has 61476 rows and 2 columns.
Here is some information about the columns:
PredictionString (float64) has 0 unique values: [nan]
example_id (object) has 2000 unique values. Some example values: ['-4639148749459150090_long', '-257485578111885185_short', '-257485578111885185_long', '-9110190923673509457_short']

-> input/tensorflow2-question-answering/sample_submission.csv has 61476 rows and 2 columns.
Here is some information about the columns:
PredictionString (float64) has 0 unique values: [nan]
example_id (object) has 2000 unique values. Some example values: ['-4639148749459150090_long', '-257485578111885185_short', '-257485578111885185_long', '-9110190923673509457_short']

-> working/tensorflow2-question-answering/sample_submission.csv has 61476 rows and 2 columns.
Here is some information about the columns:
PredictionString (float64) has 0 unique values: [nan]
example_id (object) has 2000 unique values. Some example values: ['-4639148749459150090_long', '-257485578111885185_short', '-257485578111885185_long', '-9110190923673509457_short']

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

print("Environment prepared: protobuf=python; HF telemetry disabled.")



## === cell 1
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import re
import json
import tensorflow as tf
import zlib

random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.get_logger().setLevel("ERROR")

print("Imports OK (using local tokenizer/encoder; no transformers).")



## === cell 2
debug = False

DATA_DIR = "/kaggle/input/tensorflow2-question-answering"
f_train = os.path.join(DATA_DIR, "simplified-nq-train.jsonl")
f_test = os.path.join(DATA_DIR, "simplified-nq-test.jsonl")

if not os.path.isfile(f_train):
    alt = "/kaggle/input/simplified-nq-train.jsonl"
    if os.path.isfile(alt):
        f_train = alt
if not os.path.isfile(f_test):
    alt = "/kaggle/input/simplified-nq-test.jsonl"
    if os.path.isfile(alt):
        f_test = alt

num_train_samples = 44943
num_test_samples = 346

AnswerType = {"NO_ANSWER": 0, "YES": 1, "NO": 2, "SHORT": 3, "LONG": 4}
AnswerTypeRev = {0: "NO_ANSWER", 1: "YES", 2: "NO", 3: "SHORT", 4: "LONG"}

_CLEANR = re.compile(r"<.*?>")


def clean_html(raw_html: str) -> str:
    return _CLEANR.sub("<tag>", raw_html)


def _prefix_tag_counts(doc_text_split):
    n = len(doc_text_split)
    pref = np.empty(n + 1, dtype=np.int32)
    c = 0
    pref[0] = 0
    for i, tok in enumerate(doc_text_split, start=1):
        if tok == "<tag>":
            c += 1
        pref[i] = c
    return pref


def parseData(
    filename, drop_noanswer_rate=0.8, drop_null_instances_rate=0.5, split=1.0
):
    INSTANCE_WORDS_LEN = 500
    STRIDE = 128
    if "train" in os.path.basename(filename):
        print("Parsing training file")
        n_sam = num_train_samples
        if debug:
            n_sam = 200
        is_train = True
    else:
        print("Parsing testing file")
        n_sam = num_test_samples
        if debug:
            n_sam = 2
        is_train = False

    list_instances = []
    second_list_instances = []

    with open(filename) as f:
        progress = tqdm(f)
        count_no_ans_ins = 0
        count_yes_no_ans_ins = 0
        count_short_ans_ins = 0
        count_long_ans_ins = 0

        count_drop_tlan_sam = 0
        count_num_yes_no_sam = 0
        count_num_san_sam = 0
        count_num_lan_sam = 0

        for sam_count, line in enumerate(progress):
            to_second_list = False
            rand_split = random.uniform(0, 1)
            if rand_split > split:  # this document goes to the second list
                to_second_list = True

            data = json.loads(line)

            if is_train:
                ans_id = data["annotations"][0]["long_answer"]["candidate_index"]
                if ans_id == -1:
                    noans_rand = random.uniform(0, 1)
                    if noans_rand < drop_noanswer_rate:  # drop this sample
                        continue

            example_id = data["example_id"]  # example id
            question = data["question_text"]  # question

            q_tokens = question.split()
            len_ques = len(q_tokens)
            is_keep = True

            doc_text_raw = data["document_text"]
            doc_text_raw = clean_html(doc_text_raw)  # change all html tags to <tag>
            doc_text_split = doc_text_raw.split()

            tag_pref = _prefix_tag_counts(doc_text_split)

            if is_train:  # get answer for training file
                if ans_id > -1:
                    count_num_lan_sam += 1
                    lan_start = data["long_answer_candidates"][ans_id]["start_token"]
                    lan_stop = data["long_answer_candidates"][ans_id]["end_token"]
                    long_answer_raw = doc_text_split[lan_start:lan_stop]

                    list_sans = data["annotations"][0]["short_answers"]
                    yes_no = data["annotations"][0]["yes_no_answer"]

                    if yes_no == "NONE":
                        is_yes_no = False
                    else:
                        count_num_yes_no_sam += 1
                        is_yes_no = True

                    san_start = lan_stop
                    san_stop = lan_start
                    for san in list_sans:
                        this_start = san["start_token"]
                        this_stop = san["end_token"]
                        if san_start > this_start:
                            san_start = this_start
                        if san_stop < this_stop:
                            san_stop = this_stop

                    if san_start < san_stop:
                        is_san = True
                        count_num_san_sam += 1
                        num_tag_doc_before_sans = int(tag_pref[san_start])
                        num_tag_san = int(tag_pref[san_stop] - tag_pref[san_start])

                        new_san_start = san_start - num_tag_doc_before_sans
                        new_san_stop = (
                            new_san_start + (san_stop - san_start) - num_tag_san
                        )
                    else:
                        is_san = False

                    num_tag_doc_before_lan = int(tag_pref[lan_start])
                    num_tag_lan = long_answer_raw.count("<tag>")

                    new_lan_start = lan_start - num_tag_doc_before_lan
                    new_lan_stop = new_lan_start + (lan_stop - lan_start) - num_tag_lan

            clean_doc = []
            for tok in doc_text_split:
                if tok != "<tag>":
                    clean_doc.append(tok)

            part_len = INSTANCE_WORDS_LEN - len_ques
            num_ins = (len(clean_doc) - part_len) // STRIDE
            for part_id in range(num_ins + 1):
                part_start = part_id * STRIDE
                part_end = min(len(clean_doc), part_id * STRIDE + part_len)

                part_tokens = clean_doc[part_start:part_end]

                target_ans_ins = "NO_ANSWER"
                an_start_ins = 0
                an_stop_ins = 0

                if is_train:
                    if ans_id > -1:  # if there is long answer
                        if is_san:  # if there is short answer (short but no yes/no)
                            if new_san_start >= part_start and new_san_stop < part_end:
                                san_start_ins = new_san_start - part_start
                                san_stop_ins = new_san_stop - part_start

                                an_start_ins = san_start_ins
                                an_stop_ins = san_stop_ins

                                target_ans_ins = "SHORT"
                                count_short_ans_ins += 1
                        else:
                            if new_lan_stop - new_lan_start <= part_len:
                                if (
                                    new_lan_start >= part_start
                                    and new_lan_stop < part_end
                                ):
                                    lan_start_ins = new_lan_start - part_start
                                    lan_stop_ins = new_lan_stop - part_start
                                    target_ans_ins = "LONG"
                                    an_start_ins = lan_start_ins
                                    an_stop_ins = lan_stop_ins

                                    if is_yes_no:
                                        target_ans_ins = yes_no
                                        count_yes_no_ans_ins += 1
                                    else:
                                        count_long_ans_ins += 1
                            else:
                                is_keep = False

                instance = {}
                instance["question"] = question
                instance["question_tokens"] = q_tokens
                instance["context_tokens"] = part_tokens
                instance["example_id"] = str(example_id)
                instance["part_id"] = part_id
                instance["target"] = target_ans_ins
                instance["start"] = an_start_ins
                instance["stop"] = an_stop_ins

                if is_train:
                    if target_ans_ins != "NO_ANSWER":
                        if to_second_list:
                            second_list_instances.append(instance)
                        else:
                            list_instances.append(instance)
                    else:
                        ins_rand = random.uniform(0, 1)
                        if ins_rand > drop_null_instances_rate:  # keep this instance
                            count_no_ans_ins += 1
                            if to_second_list:
                                second_list_instances.append(instance)
                            else:
                                list_instances.append(instance)
                else:  # test file: only first list
                    list_instances.append(instance)

            if not is_keep:
                count_drop_tlan_sam += 1

        if is_train:
            print("NUM YES NO ANSWER INSTANCES = ", count_yes_no_ans_ins)
            print("NUM SHORT ANSWER INSTANCES = ", count_short_ans_ins)
            print("NUM LONG ANSWER INSTANCES = ", count_long_ans_ins)
            print("NUM NO ANSWER INSTANCES = ", count_no_ans_ins)

            print("NUM YES NO SAMPLE = ", count_num_yes_no_sam)
            print("NUM SHORT ANSWER SAMPLE = ", count_num_san_sam)
            print("NUM LONG ANSWER SAMPLE = ", count_num_lan_sam)
            print("NUM TOO LONG ANSWER DROPPED SAMPLE = ", count_drop_tlan_sam)

    return list_instances, second_list_instances


def parseTestOnce(filename):
    INSTANCE_WORDS_LEN = 500
    STRIDE = 128

    list_instances = []
    list_sample = []

    with open(filename) as f:
        progress = tqdm(f)
        for sam_count, line in enumerate(progress):
            data = json.loads(line)
            example_id = data["example_id"]
            question = data["question_text"]

            q_tokens = question.split()
            len_ques = len(q_tokens)

            doc_text_raw = clean_html(data["document_text"])
            doc_text_split = doc_text_raw.split()

            tag_pref = _prefix_tag_counts(doc_text_split)
            n_doc_tokens = len(doc_text_split)

            clean_doc = []
            for tok in doc_text_split:
                if tok != "<tag>":
                    clean_doc.append(tok)

            list_candidates = data["long_answer_candidates"]
            list_new_candidates = []
            for cand in list_candidates:
                cand_start = int(cand.get("start_token", 0))
                cand_stop = int(cand.get("end_token", 0))

                cand_start = max(0, min(cand_start, n_doc_tokens))
                cand_stop = max(0, min(cand_stop, n_doc_tokens))

                num_tag_bef_start = int(tag_pref[cand_start])
                num_tag_bef_stop = int(tag_pref[cand_stop])
                new_start = cand_start - num_tag_bef_start
                new_stop = cand_stop - num_tag_bef_stop

                new_cand = {
                    "end_token": new_stop,
                    "start_token": new_start,
                    "top_level": cand["top_level"],
                }
                list_new_candidates.append([new_cand, cand])

            sample = {
                "example_id": str(example_id),
                "question": question,
                "document": " ".join(clean_doc),
                "raw_document": doc_text_raw,
                "candidates": list_new_candidates,
            }
            list_sample.append(sample)

            part_len = INSTANCE_WORDS_LEN - len_ques
            num_ins = (len(clean_doc) - part_len) // STRIDE
            for part_id in range(num_ins + 1):
                part_start = part_id * STRIDE
                part_end = min(len(clean_doc), part_id * STRIDE + part_len)
                part_tokens = clean_doc[part_start:part_end]

                instance = {
                    "question": question,
                    "question_tokens": q_tokens,
                    "context_tokens": part_tokens,
                    "example_id": str(example_id),
                    "part_id": part_id,
                    "target": "NO_ANSWER",
                    "start": 0,
                    "stop": 0,
                }
                list_instances.append(instance)

    return list_instances, list_sample


def mergeInstanceResult(test_res, list_test_ins):
    start_idx = np.argmax(test_res[:, 0, :], axis=1).astype(np.int32)
    stop_idx = np.argmax(test_res[:, 1, :], axis=1).astype(np.int32)
    target_idx = np.argmax(test_res[:, 2, :], axis=1).astype(np.int32)

    rows = np.arange(test_res.shape[0])
    start_score = test_res[rows, 0, start_idx]
    stop_score = test_res[rows, 1, stop_idx]
    target_score = test_res[rows, 2, target_idx]
    start_cls = test_res[:, 0, 0]
    stop_cls = test_res[:, 1, 0]

    for i in range(len(list_test_ins)):
        d = list_test_ins[i]
        d["start"] = int(start_idx[i])
        d["stop"] = int(stop_idx[i])
        d["target"] = int(target_idx[i])

        d["start_score"] = float(start_score[i])
        d["stop_score"] = float(stop_score[i])
        d["target_score"] = float(target_score[i])

        d["start_CLS"] = float(start_cls[i])
        d["stop_CLS"] = float(stop_cls[i])
    return list_test_ins


def mergeDocumentResult(doc_df, ins_df):
    list_doc_id = doc_df["example_id"].unique()
    list_doc_result_lan = []

    doc_records = doc_df.to_dict("records")
    doc_map = {r["example_id"]: r for r in doc_records}

    ex_ids = ins_df["example_id"].to_numpy()
    part_ids = ins_df["part_id"].to_numpy(np.int32)
    start = ins_df["start"].to_numpy(np.int32)
    stop = ins_df["stop"].to_numpy(np.int32)
    target = ins_df["target"].to_numpy(np.int32)
    start_score = ins_df["start_score"].to_numpy(np.float32)
    stop_score = ins_df["stop_score"].to_numpy(np.float32)
    start_cls = ins_df["start_CLS"].to_numpy(np.float32)
    stop_cls = ins_df["stop_CLS"].to_numpy(np.float32)

    valid = (start != 0) | (stop != 0) | (target != 0)
    valid &= stop > start
    vote = (start_score - start_cls) + (stop_score - stop_cls)

    idx_by_doc = {}
    for i, eid in enumerate(ex_ids):
        if valid[i]:
            idx_by_doc.setdefault(eid, []).append(i)

    for doc_id in list_doc_id:
        idx = idx_by_doc.get(doc_id)
        samrow = doc_map.get(doc_id)
        if samrow is None:
            continue

        if not idx:
            list_doc_result_lan.append(
                {
                    "example_id": doc_id,
                    "question": samrow["question"],
                    "raw_document": samrow["raw_document"],
                    "start_token": -1,
                    "stop_token": -1,
                    "target": AnswerTypeRev[0],
                }
            )
            continue

        idx = np.asarray(idx, dtype=np.int32)
        best_pos = idx[np.argmax(vote[idx])]
        real_start = int(start[best_pos] + part_ids[best_pos] * 128)
        real_stop = int(stop[best_pos] + part_ids[best_pos] * 128 + 1)
        real_target = AnswerTypeRev[int(target[best_pos])]

        lan_start_raw = -1
        lan_stop_raw = -1

        if real_start != 0:
            list_cands = samrow["candidates"]

            best_score = 0
            best_ratio = 0.0
            best_match_id = 1

            real_a = int(real_start)
            real_b = int(real_stop)

            for cand_id, cand in enumerate(list_cands):
                lan_start = int(cand[0]["start_token"])
                lan_stop = int(cand[0]["end_token"])
                lan_len = lan_stop - lan_start + 1
                if lan_len <= 0:
                    continue

                inter = max(0, min(real_b, lan_stop) - max(real_a, lan_start) + 1)
                ratio = float(inter) / float(lan_len) if lan_len > 0 else 0.0

                if inter > best_score:
                    best_score = inter
                    best_match_id = cand_id
                    best_ratio = ratio
                elif inter > 0 and inter == best_score:
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match_id = cand_id

            lan_raw_res = list_cands[best_match_id][1]
            lan_start_raw = int(lan_raw_res["start_token"])
            lan_stop_raw = int(lan_raw_res["end_token"])

        doc_result_lan = {
            "example_id": doc_id,
            "question": samrow["question"],
            "raw_document": samrow["raw_document"],
            "start_token": lan_start_raw,
            "stop_token": lan_stop_raw,
            "target": real_target,
        }
        list_doc_result_lan.append(doc_result_lan)
    return list_doc_result_lan


class SimpleTokenizer:
    def __init__(self, vocab_size=30522, max_length=512):
        self.vocab_size = int(vocab_size)
        self.max_length = int(max_length)
        self.cls_id = 101
        self.sep_id = 102
        self.pad_id = 0
        self.unk_id = 100

        self._cache = {}
        self._cache_get = self._cache.get
        self._cache_set = self._cache.__setitem__

        self._base = 999
        self._span = max(1, self.vocab_size - self._base)

    def add_special_tokens(self, *args, **kwargs):
        return

    def _stable_hash_token(self, tok: str) -> int:
        v = self._cache_get(tok)
        if v is not None:
            return v
        b = str(tok).encode("utf-8", "ignore")
        h = zlib.crc32(b) & 0xFFFFFFFF
        out = self._base + (h % self._span)
        self._cache_set(tok, out)
        return out

    def encode_plus(
        self,
        text_a,
        text_b=None,
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=True,
    ):
        max_length = int(max_length)

        if isinstance(text_a, (list, tuple)):
            a_toks = text_a
        else:
            a_toks = str(text_a).split()

        if text_b is None:
            b_toks = []
        elif isinstance(text_b, (list, tuple)):
            b_toks = text_b
        else:
            b_toks = str(text_b).split()

        input_ids = [self.cls_id]
        input_ids.extend(self._stable_hash_token(t) for t in a_toks)
        input_ids.append(self.sep_id)
        token_type_ids = [0] * len(input_ids)

        if b_toks:
            b_ids = [self._stable_hash_token(t) for t in b_toks]
            b_ids.append(self.sep_id)
            input_ids += b_ids
            token_type_ids += [1] * len(b_ids)

        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            token_type_ids = token_type_ids[:max_length]

        attention_mask = [1] * len(input_ids)

        if padding == "max_length" and len(input_ids) < max_length:
            pad_len = max_length - len(input_ids)
            input_ids += [self.pad_id] * pad_len
            token_type_ids += [0] * pad_len
            attention_mask += [0] * pad_len

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }


def preprocess_data(data, tokenizer):
    n = len(data)
    x1 = np.zeros((n, 512), dtype=np.int32)
    x2 = np.zeros((n, 512), dtype=np.int32)
    x3 = np.zeros((n, 512), dtype=np.int32)
    y = np.zeros((n, 3), dtype=np.int32)

    q_cache = {}  # tuple(tokens) -> list[int] (already hashed, excluding CLS/SEP)
    progress = tqdm(range(n), total=n)
    for i in progress:
        one_sam = data[i]
        q = one_sam.get("question_tokens", one_sam.get("question", ""))
        c = one_sam.get(
            "context_tokens",
            one_sam.get("context", one_sam.get("context_text", "")),
        )

        if not isinstance(q, (list, tuple)):
            q = str(q).split()
        if not isinstance(c, (list, tuple)):
            c = str(c).split()

        q_key = tuple(q)
        q_ids = q_cache.get(q_key)
        if q_ids is None:
            q_ids = [tokenizer._stable_hash_token(t) for t in q]
            q_cache[q_key] = q_ids

        input_ids = [tokenizer.cls_id]
        input_ids.extend(q_ids)
        input_ids.append(tokenizer.sep_id)
        token_type_ids = [0] * len(input_ids)

        if c:
            c_ids = [tokenizer._stable_hash_token(t) for t in c]
            c_ids.append(tokenizer.sep_id)
            input_ids += c_ids
            token_type_ids += [1] * len(c_ids)

        if len(input_ids) > 512:
            input_ids = input_ids[:512]
            token_type_ids = token_type_ids[:512]

        attn_len = len(input_ids)
        attention_mask = [1] * attn_len
        if attn_len < 512:
            pad_len = 512 - attn_len
            input_ids += [tokenizer.pad_id] * pad_len
            token_type_ids += [0] * pad_len
            attention_mask += [0] * pad_len

        x1[i, :] = input_ids
        x2[i, :] = token_type_ids
        x3[i, :] = attention_mask

        y[i, 0] = int(one_sam.get("start", 0))
        y[i, 1] = int(one_sam.get("stop", 0))
        y[i, 2] = int(AnswerType.get(one_sam.get("target", "NO_ANSWER"), 0))
    return x1, x2, x3, y


def get_strategy():
    try:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        print(
            "Running on TPU ", tpu_cluster_resolver.cluster_spec().as_dict()["worker"]
        )
        tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
        strategy = tf.distribute.experimental.TPUStrategy(tpu_cluster_resolver)
    except Exception as e:
        print("No TPU detected:", repr(e))
        strategy = tf.distribute.get_strategy()
    return strategy


def _find_weight_file(filename):
    candidates = [
        os.path.join(DATA_DIR, filename),
        os.path.join("/kaggle/input/tensorflow2-question-answering", filename),
        os.path.join("/kaggle/input", filename),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    if os.path.isdir(DATA_DIR):
        for dirpath, _, filenames in os.walk(DATA_DIR):
            if filename in filenames:
                return os.path.join(dirpath, filename)
    return None


def build_model(model_name_ignored=None):
    NUM_TARGET = 5
    VOCAB_SIZE = 30522
    HIDDEN = 768

    class LocalBertLike(tf.keras.layers.Layer):
        def __init__(self, vocab_size=VOCAB_SIZE, hidden_size=HIDDEN, **kwargs):
            super().__init__(**kwargs)
            self.emb = tf.keras.layers.Embedding(
                vocab_size, hidden_size, mask_zero=False
            )
            self.proj = tf.keras.layers.Dense(hidden_size, activation="tanh")

        def call(self, input_ids, token_type_ids=None, attention_mask=None):
            x = self.emb(input_ids)  # [B, 512, H]
            cls = x[:, 0, :]
            pooled = self.proj(cls)  # [B, H]
            return x, pooled

    class MyQAModel(tf.keras.Model):
        def __init__(self, *inputs, **kwargs):
            super().__init__(*inputs, **kwargs)
            self.bert = LocalBertLike()
            self.dropout1 = tf.keras.layers.Dropout(0.2)
            self.dropout2 = tf.keras.layers.Dropout(0.2)

            self.start_logits = tf.keras.layers.Dense(1)
            self.stop_logits = tf.keras.layers.Dense(1)

            self.target = tf.keras.layers.Dense(NUM_TARGET)

        def call(self, inputs, **kwargs):
            seq_out, pooled_out = self.bert(
                inputs[0], token_type_ids=inputs[1], attention_mask=inputs[2]
            )

            dropout_res0 = self.dropout1(seq_out)
            start_logits = tf.squeeze(self.start_logits(dropout_res0), -1)
            stop_logits = tf.squeeze(self.stop_logits(dropout_res0), -1)

            dropout_res1 = self.dropout1(pooled_out)
            targets = self.target(dropout_res1)  # [B, NUM_TARGET]
            paddings = tf.constant([[0, 0], [0, 512 - NUM_TARGET]])
            targets = tf.pad(targets, paddings)  # [B, 512]

            res = tf.stack([start_logits, stop_logits, targets], axis=1)  # [B, 3, 512]
            return res

    return MyQAModel()


def getRawInstanceResults(list_test, verbose=True, debug=False):
    if verbose:
        print("Getting raw result for all the instances generated from test file")

    tokenizer = SimpleTokenizer()

    x_test1, x_test2, x_test3, y_test = preprocess_data(list_test, tokenizer)
    if verbose:
        print("Finish tokenizing ", len(list_test), " data for the first model")

    if verbose:
        print("Preparing model")

    strategy = get_strategy()
    with strategy.scope():
        testModel = build_model(None)
        optAdam = tf.keras.optimizers.Adam(learning_rate=0.00005)
        lossSCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metricSCA = tf.keras.metrics.SparseCategoricalAccuracy()
        testModel.compile(optimizer=optAdam, loss=lossSCE, metrics=[metricSCA])

    testModel.evaluate(
        x=[x_test1[0:2], x_test2[0:2], x_test3[0:2]], y=y_test[0:2], verbose=0
    )

    wpath = _find_weight_file("weights-02-0.555.h5")
    if wpath is not None:
        try:
            with strategy.scope():
                testModel.load_weights(wpath)
            if verbose:
                print("Finish loading pretrained weights for the model:", wpath)
        except Exception as e:
            print(
                "Warning: could not load weights-02-0.555.h5 (architecture mismatch). Using initialized weights.",
                repr(e),
            )
    else:
        print("Warning: weights-02-0.555.h5 not found. Using initialized weights.")

    res = testModel.predict([x_test1, x_test2, x_test3], batch_size=256, verbose=0)
    res = tf.nn.softmax(res, axis=-1).numpy()

    if verbose:
        print("Finish calculating raw result, get an array of size: ", res.shape)
    return res


def getListShortSample(doc_lan_df):
    INSTANCE_WORDS_LEN = 500
    STRIDE = 384
    list_instances = []
    for _, row in doc_lan_df.iterrows():
        question = row["question"]
        long_start = row["start_token"]
        long_stop = row["stop_token"]
        raw_doc = row["raw_document"]
        example_id = row["example_id"]
        if long_start != -1 and long_stop != -1:  # if there is long answer
            lan_split = raw_doc.split()[long_start:long_stop]
            q_tokens = question.split()
            len_ques = len(q_tokens)
            len_lan = len(lan_split)
            part_len = INSTANCE_WORDS_LEN - len_ques

            if len_lan > part_len:
                num_ins = (len_lan - part_len) // STRIDE + 2
            else:
                num_ins = 1

            for part_id in range(num_ins):  # create instance
                part_start = part_id * STRIDE
                part_end = min(len(lan_split), part_id * STRIDE + part_len)

                part_tokens = lan_split[part_start:part_end]

                instance = {
                    "question": question,
                    "question_tokens": q_tokens,
                    "context_tokens": part_tokens,
                    "example_id": example_id,
                    "part_id": part_id,
                }
                list_instances.append(instance)
    return list_instances


def preprocess_san(list_san_ins):
    tokenizer = SimpleTokenizer()
    tokenizer.add_special_tokens({"unk_token": "<tag>"})

    n = len(list_san_ins)
    x1 = np.zeros((n, 512), dtype=np.int32)
    x2 = np.zeros((n, 512), dtype=np.int32)
    x3 = np.zeros((n, 512), dtype=np.int32)

    q_cache = {}
    progress = tqdm(range(n), total=n)
    for i in progress:
        one_sam = list_san_ins[i]
        q = one_sam.get("question_tokens", one_sam.get("question", ""))
        c = one_sam.get(
            "context_tokens", one_sam.get("context", one_sam.get("context_text", ""))
        )

        if not isinstance(q, (list, tuple)):
            q = str(q).split()
        if not isinstance(c, (list, tuple)):
            c = str(c).split()

        q_key = tuple(q)
        q_ids = q_cache.get(q_key)
        if q_ids is None:
            q_ids = [tokenizer._stable_hash_token(t) for t in q]
            q_cache[q_key] = q_ids

        input_ids = [tokenizer.cls_id]
        input_ids.extend(q_ids)
        input_ids.append(tokenizer.sep_id)
        token_type_ids = [0] * len(input_ids)

        if c:
            c_ids = [tokenizer._stable_hash_token(t) for t in c]
            c_ids.append(tokenizer.sep_id)
            input_ids += c_ids
            token_type_ids += [1] * len(c_ids)

        if len(input_ids) > 512:
            input_ids = input_ids[:512]
            token_type_ids = token_type_ids[:512]

        attn_len = len(input_ids)
        attention_mask = [1] * attn_len
        if attn_len < 512:
            pad_len = 512 - attn_len
            input_ids += [tokenizer.pad_id] * pad_len
            token_type_ids += [0] * pad_len
            attention_mask += [0] * pad_len

        x1[i, :] = input_ids
        x2[i, :] = token_type_ids
        x3[i, :] = attention_mask

    return x1, x2, x3


def getSanInstanceResults(list_san_ins, verbose=True):
    if verbose:
        print(
            "Calculating raw result for ", len(list_san_ins), " short answer instances"
        )

    if len(list_san_ins) == 0:
        return tf.zeros([0, 3, 512], dtype=tf.float32)

    x_san1, x_san2, x_san3 = preprocess_san(list_san_ins)
    if verbose:
        print("Finish tokenizing test instance for the second model")

    strategy = get_strategy()
    with strategy.scope():
        testModel = build_model(None)
        optAdam = tf.keras.optimizers.Adam(learning_rate=0.00005)
        lossSCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metricSCA = tf.keras.metrics.SparseCategoricalAccuracy()
        testModel.compile(optimizer=optAdam, loss=lossSCE, metrics=[metricSCA])

    tem_y = tf.convert_to_tensor([[0, 0, 0]], dtype=tf.int32)
    testModel.evaluate(x=[x_san1[0:1], x_san2[0:1], x_san3[0:1]], y=tem_y, verbose=0)

    wpath = _find_weight_file("weights-02-0.701.h5")
    if wpath is not None:
        try:
            with strategy.scope():
                testModel.load_weights(wpath)
            if verbose:
                print("Finish loading the weights for the second model:", wpath)
        except Exception as e:
            print(
                "Warning: could not load weights-02-0.701.h5 (architecture mismatch). Using initialized weights.",
                repr(e),
            )
    else:
        print("Warning: weights-02-0.701.h5 not found. Using initialized weights.")

    san_res = testModel.predict([x_san1, x_san2, x_san3], batch_size=256, verbose=0)
    san_res = tf.nn.softmax(san_res, axis=-1)

    if verbose:
        print("Finish calculating, get result of size: ", san_res.shape)

    return san_res


def mergeSanInstanceResult(san_res, list_san_ins, debug=True):
    if len(list_san_ins) == 0:
        return list_san_ins
    arr = san_res.numpy() if isinstance(san_res, tf.Tensor) else np.asarray(san_res)

    start_idx = np.argmax(arr[:, 0, :], axis=1).astype(np.int32)
    stop_idx = np.argmax(arr[:, 1, :], axis=1).astype(np.int32)
    target_idx = np.argmax(arr[:, 2, :], axis=1).astype(np.int32)

    rows = np.arange(arr.shape[0])
    start_score = arr[rows, 0, start_idx]
    stop_score = arr[rows, 1, stop_idx]
    target_score = arr[rows, 2, target_idx]
    start_cls = arr[:, 0, 0]
    stop_cls = arr[:, 1, 0]

    for i in range(len(list_san_ins)):
        d = list_san_ins[i]
        d["start"] = int(start_idx[i])
        d["stop"] = int(stop_idx[i])
        t = int(target_idx[i])
        d["target"] = AnswerTypeRev[t]

        d["start_score"] = float(start_score[i])
        d["stop_score"] = float(stop_score[i])
        d["target_score"] = float(target_score[i])

        d["start_CLS"] = float(start_cls[i])
        d["stop_CLS"] = float(stop_cls[i])
    return list_san_ins


def mergeSanLan(doc_lan_df, san_ins_res_df, verbose=True):
    SAN_STRIDE = 384
    list_san_lan_res_doc = []
    if verbose:
        print(
            "Merging ",
            san_ins_res_df.shape[0],
            " short answer result to ",
            doc_lan_df.shape[0],
            " documents",
        )

    san_groups = (
        {k: v for k, v in san_ins_res_df.groupby("example_id", sort=False)}
        if san_ins_res_df.shape[0]
        else {}
    )

    for row in doc_lan_df.to_dict("records"):
        doc_id = row["example_id"]

        lan_start = row["start_token"]
        lan_stop = row["stop_token"]
        lan_target = row["target"]

        san_start = -1
        san_stop = -1
        final_target = AnswerTypeRev[0]

        san_list = san_groups.get(doc_id)

        max_score = 0.00001
        if san_list is not None:
            for _, san in san_list.iterrows():
                part_id = int(san["part_id"])
                part_start = part_id * SAN_STRIDE
                start_in_part = int(san["start"])
                stop_in_part = int(san["stop"])

                start_in_lan = start_in_part + part_start
                stop_in_lan = stop_in_part + part_start

                if start_in_part > 0 and stop_in_part > start_in_part:
                    vote = (
                        float(san["start_score"])
                        - float(san["start_CLS"])
                        + float(san["stop_score"])
                        - float(san["stop_CLS"])
                    )
                    if vote > max_score:
                        san_start = start_in_lan + int(lan_start)
                        san_stop = stop_in_lan + int(lan_start)
                        max_score = vote
                final_target = san["target"]

        if lan_target != "NO_ANSWER" and lan_target != "SHORT" and lan_target != "LONG":
            if final_target == "NO_ANSWER" or final_target == "LONG":
                final_target = lan_target

        san_lan_res_doc = {
            "example_id": doc_id,
            "question": row["question"],
            "raw_document": row["raw_document"],
            "lan_start": lan_start,
            "lan_stop": lan_stop,
            "san_start": san_start,
            "san_stop": san_stop,
            "target": final_target,
        }

        list_san_lan_res_doc.append(san_lan_res_doc)
    return list_san_lan_res_doc


def getFinalResult(f_test):
    list_all_ins, list_doc = parseTestOnce(f_test)

    all_ins_res = getRawInstanceResults(list_all_ins)
    list_fine_res_all_ins = mergeInstanceResult(all_ins_res, list_all_ins)
    fine_res_all_ins_df = pd.DataFrame(list_fine_res_all_ins)

    doc_df = pd.DataFrame(list_doc)
    doc_res_lan = mergeDocumentResult(doc_df, fine_res_all_ins_df)
    doc_res_lan_df = pd.DataFrame(doc_res_lan)

    list_san_ins = getListShortSample(doc_res_lan_df)
    raw_san_ins_res = getSanInstanceResults(list_san_ins)
    list_san_res = mergeSanInstanceResult(raw_san_ins_res, list_san_ins, debug=False)
    list_san_res_df = pd.DataFrame(list_san_res)

    san_lan_doc = mergeSanLan(doc_res_lan_df, list_san_res_df)
    return san_lan_doc


def getLines(san_lan_doc):
    lines = []
    for doc in san_lan_doc:
        line1 = {}
        line2 = {}

        doc_id = doc["example_id"]

        long_text = doc_id + "_long"
        line1["example_id"] = long_text

        short_text = doc_id + "_short"
        line2["example_id"] = short_text

        if doc["lan_start"] != -1 and doc["lan_stop"] != -1:
            long_res = str(doc["lan_start"]) + ":" + str(doc["lan_stop"])
            line1["PredictionString"] = long_res
        else:
            line1["PredictionString"] = ""

        if doc["san_start"] != -1 and doc["san_stop"] != -1:
            short_res = str(doc["san_start"]) + ":" + str(doc["san_stop"])
            line2["PredictionString"] = short_res
        else:
            line2["PredictionString"] = ""

        if doc["target"] == "YES" or doc["target"] == "NO":
            line2["PredictionString"] = doc["target"]

        lines.append(line1)
        lines.append(line2)
    return lines


def getSubmission():
    san_lan_doc = getFinalResult(f_test)
    lines = getLines(san_lan_doc)
    pred_df = pd.DataFrame(lines)

    if "example_id" not in pred_df.columns or "PredictionString" not in pred_df.columns:
        raise ValueError("Submission dataframe missing required columns.")

    pred_df["PredictionString"] = pred_df["PredictionString"].fillna("").astype(str)

    sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
    if not os.path.isfile(sample_path):
        sample_path = "/kaggle/input/sample_submission.csv"
    if not os.path.isfile(sample_path):
        sample_path = (
            "/kaggle/input/tensorflow2-question-answering/sample_submission.csv"
        )
    sample_df = pd.read_csv(sample_path)

    out_df = sample_df[["example_id"]].merge(
        pred_df[["example_id", "PredictionString"]],
        on="example_id",
        how="left",
    )
    out_df["PredictionString"] = out_df["PredictionString"].fillna("").astype(str)

    out_df.to_csv(
        "./submission.csv", index=False, columns=["example_id", "PredictionString"]
    )
    print("Wrote ./submission.csv with shape:", out_df.shape)
    return out_df




## === cell 3
getSubmission()
