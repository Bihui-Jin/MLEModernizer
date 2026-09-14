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

# 5. Target score

0.3773075191355245

# 6. Current score

0.57117

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.57117) has done: 'I fix the protobuf/transformers crash by forcing the Python protobuf implementation (this avoids the `MessageFactory.GetPrototype` error that prevents any submission from being written). Then I make the weight-file resolver robust to the actual Kaggle dataset layout by searching for any `.h5` weights in the provided input tree and selecting the intended ones by best filename match, so the pipeline can load models successfully. Finally, I ensure `getSubmission()` always writes `submission.csv` with the exact sample order and required columns, and add a safe fallback to output a blank submission if model assets are still unavailable (so you always get a valid CSV).'
- What this solution (achieved 0.57117) has done: 'I fix the `MessageFactory.GetPrototype` crash by forcing a compatible protobuf runtime before TensorFlow/transformers import, and by defensively patching `google.protobuf.message_factory.MessageFactory` to provide `GetPrototype` when the installed protobuf version only exposes `GetMessageClass`. This is a correctness/stability fix that should restore end-to-end execution and submission writing. Because your current score (0.57117) is already well above the target (0.3773) and within the allowed tolerance band, I not change any model logic or post-processing that would intentionally move the score. I also keep the existing robust weight-file resolution and ensure `submission.csv` is always produced with the required columns and row order.'
- What this solution (achieved 0.57117) has done: 'Your current score (0.57117) is higher than the target (0.3773), so to move toward the target we should *slightly reduce* predictive aggressiveness without changing the model, architecture, or inference pipeline. The smallest safe lever is the existing “vote” thresholds used when selecting long/short spans from instance predictions; increasing these thresholds makes the system output more blanks, lowering recall and thus reducing F1 toward the target. I add two tiny constants to control these thresholds and raise them moderately, keeping everything else identical (same weights, tokenization, merging logic, and submission formatting). The script still run end-to-end and write a valid `submission.csv` with the exact sample order/columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np
import pandas as pd
import sys
import random
from tqdm import tqdm
import re
import json

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(descriptor)
            if hasattr(_message_factory, "GetMessageClass"):
                return _message_factory.GetMessageClass(descriptor)
            raise AttributeError(
                "Neither GetPrototype nor GetMessageClass is available"
            )

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import tensorflow as tf
from transformers import AutoTokenizer, TFBertModel

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass
try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

debug = False

f_train = "../input/tensorflow2-question-answering/simplified-nq-train.jsonl"
f_test = "../input/tensorflow2-question-answering/simplified-nq-test.jsonl"
num_train_samples = 44943
num_test_samples = 346

AnswerType = {"NO_ANSWER": 0, "YES": 1, "NO": 2, "SHORT": 3, "LONG": 4}
AnswerTypeRev = {0: "NO_ANSWER", 1: "YES", 2: "NO", 3: "SHORT", 4: "LONG"}

cleanr = re.compile("<.*?>")

_TRANS_TABLE = str.maketrans({ch: " " for ch in "<>/=\"'\\\n\r\t"})

LAN_VOTE_THRESHOLD = 0.75  # was effectively ~0.0001
SAN_VOTE_THRESHOLD = 0.75  # was effectively ~0.00001


def clean_html(raw_html: str) -> str:
    if "<" not in raw_html:
        return raw_html
    out = []
    i = 0
    n = len(raw_html)
    while i < n:
        lt = raw_html.find("<", i)
        if lt == -1:
            out.append(raw_html[i:])
            break
        out.append(raw_html[i:lt])
        gt = raw_html.find(">", lt + 1)
        if gt == -1:
            out.append(raw_html[lt:])
            break
        out.append(" <tag> ")
        i = gt + 1
    return "".join(out)


def parseData(
    filename, drop_noanswer_rate=0.8, drop_null_instances_rate=0.5, split=1.0
):
    INSTANCE_WORDS_LEN = 500
    STRIDE = 128
    if "train" in filename:
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
            if rand_split > split:
                to_second_list = True

            data = json.loads(line)

            if is_train:
                ans_id = data["annotations"][0]["long_answer"]["candidate_index"]
                if ans_id == -1:
                    noans_rand = random.uniform(0, 1)
                    if noans_rand < drop_noanswer_rate:
                        continue

            example_id = data["example_id"]
            question = data["question_text"]
            is_keep = True

            doc_text_raw = clean_html(data["document_text"])
            doc_text_split = doc_text_raw.split()

            if is_train:
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
                        num_tag_doc_before_sans = doc_text_split[0:san_start].count(
                            "<tag>"
                        )
                        num_tag_san = doc_text_split[san_start:san_stop].count("<tag>")

                        new_san_start = san_start - num_tag_doc_before_sans
                        new_san_stop = (
                            new_san_start + san_stop - san_start - num_tag_san
                        )
                    else:
                        is_san = False

                    num_tag_doc_before_lan = doc_text_split[0:lan_start].count("<tag>")
                    num_tag_lan = long_answer_raw.count("<tag>")

                    new_lan_start = lan_start - num_tag_doc_before_lan
                    new_lan_stop = new_lan_start + lan_stop - lan_start - num_tag_lan

            clean_doc = list(filter(("<tag>").__ne__, doc_text_split))

            len_ques = len(question.split())
            part_len = INSTANCE_WORDS_LEN - len_ques

            num_ins = (len(clean_doc) - part_len) // STRIDE
            for part_id in range(num_ins + 1):
                part_start = part_id * STRIDE
                part_end = min(len(clean_doc), part_id * STRIDE + part_len)

                part = clean_doc[part_start:part_end]

                target_ans_ins = "NO_ANSWER"
                an_start_ins = 0
                an_stop_ins = 0

                if is_train:
                    if ans_id > -1:
                        if is_san:
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

                part = " ".join(part)
                instance = {
                    "question": question,
                    "context": part,
                    "example_id": str(example_id),
                    "part_id": part_id,
                    "target": target_ans_ins,
                    "start": an_start_ins,
                    "stop": an_stop_ins,
                }

                if is_train:
                    if target_ans_ins != "NO_ANSWER":
                        (
                            second_list_instances if to_second_list else list_instances
                        ).append(instance)
                    else:
                        ins_rand = random.uniform(0, 1)
                        if ins_rand > drop_null_instances_rate:
                            count_no_ans_ins += 1
                            (
                                second_list_instances
                                if to_second_list
                                else list_instances
                            ).append(instance)
                else:
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


def parseTestAll(filename):
    INSTANCE_WORDS_LEN = 500
    STRIDE = 128
    if debug:
        n_sam = 2
    else:
        n_sam = num_test_samples

    list_instances = []
    list_sample = []

    with open(filename) as f:
        progress = tqdm(f)
        for sam_count, line in enumerate(progress):
            data = json.loads(line)
            example_id = data["example_id"]
            question = data["question_text"]

            doc_text_raw = clean_html(data["document_text"])
            doc_text_split = doc_text_raw.split()

            tag_prefix = np.empty(len(doc_text_split) + 1, dtype=np.int32)
            clean_doc = []
            tcount = 0
            tag_prefix[0] = 0
            for i, tok in enumerate(doc_text_split, start=1):
                if tok == "<tag>":
                    tcount += 1
                else:
                    clean_doc.append(tok)
                tag_prefix[i] = tcount

            clean_doc_str = " ".join(clean_doc)
            if clean_doc:
                lens = np.fromiter((len(t) for t in clean_doc), dtype=np.int32)
                offsets = np.empty(len(clean_doc) + 1, dtype=np.int64)
                offsets[0] = 0
                np.cumsum(lens + 1, out=offsets[1:])  # includes trailing space
                offsets[-1] = len(clean_doc_str)
            else:
                offsets = np.array([0], dtype=np.int64)

            list_candidates = data["long_answer_candidates"]
            list_new_candidates = []
            max_tok = len(doc_text_split)
            for cand in list_candidates:
                cand_start = int(cand["start_token"])
                cand_stop = int(cand["end_token"])

                cand_start = max(0, min(cand_start, max_tok))
                cand_stop = max(0, min(cand_stop, max_tok))
                if cand_stop < cand_start:
                    cand_stop = cand_start

                num_tag_bef_start = int(tag_prefix[cand_start])
                num_tag_bef_stop = int(tag_prefix[cand_stop])
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
                "document": clean_doc_str,
                "raw_document": doc_text_raw,
                "candidates": list_new_candidates,
            }
            list_sample.append(sample)

            len_ques = len(question.split())
            part_len = INSTANCE_WORDS_LEN - len_ques

            num_ins = (len(clean_doc) - part_len) // STRIDE
            for part_id in range(num_ins + 1):
                part_start = part_id * STRIDE
                part_end = min(len(clean_doc), part_id * STRIDE + part_len)

                if part_start >= part_end:
                    part = ""
                else:
                    cs = int(offsets[part_start])
                    ce = int(offsets[part_end] - 1)  # remove extra trailing space
                    if part_end == len(clean_doc):
                        ce = len(clean_doc_str)
                    part = clean_doc_str[cs:ce]

                instance = {
                    "question": question,
                    "context": part,
                    "example_id": str(example_id),
                    "part_id": part_id,
                    "target": "NO_ANSWER",
                    "start": 0,
                    "stop": 0,
                }
                list_instances.append(instance)

    return list_instances, list_sample


def mergeInstanceResult(test_res, list_test_ins):
    test_res = np.asarray(test_res)
    start_idx = test_res[:, 0, :].argmax(axis=1)
    stop_idx = test_res[:, 1, :].argmax(axis=1)
    target_idx = test_res[:, 2, :].argmax(axis=1)

    rows = np.arange(test_res.shape[0])
    start_score = test_res[rows, 0, start_idx]
    stop_score = test_res[rows, 1, stop_idx]
    target_score = test_res[rows, 2, target_idx]

    start_CLS = test_res[:, 0, 0]
    stop_CLS = test_res[:, 1, 0]

    for i, ins in enumerate(list_test_ins):
        ins["start"] = int(start_idx[i])
        ins["stop"] = int(stop_idx[i])
        ins["target"] = int(target_idx[i])
        ins["start_score"] = float(start_score[i])
        ins["stop_score"] = float(stop_score[i])
        ins["target_score"] = float(target_score[i])
        ins["start_CLS"] = float(start_CLS[i])
        ins["stop_CLS"] = float(stop_CLS[i])
    return list_test_ins


def mergeDocumentResult(doc_df, ins_df):
    doc_eids = doc_df["example_id"].to_numpy()
    doc_idx = {eid: i for i, eid in enumerate(doc_eids)}
    list_doc_id = doc_df["example_id"].unique()

    ins_example = ins_df["example_id"].to_numpy()
    ins_part = ins_df["part_id"].to_numpy(np.int32)
    ins_start = ins_df["start"].to_numpy(np.int32)
    ins_stop = ins_df["stop"].to_numpy(np.int32)
    ins_target = ins_df["target"].to_numpy(np.int32)
    ins_start_score = ins_df["start_score"].to_numpy(np.float32)
    ins_start_CLS = ins_df["start_CLS"].to_numpy(np.float32)

    order = np.argsort(ins_example, kind="mergesort")
    ex_sorted = ins_example[order]
    change = np.flatnonzero(ex_sorted[1:] != ex_sorted[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(ex_sorted)]))

    doc_question = doc_df["question"].to_numpy()
    doc_raw = doc_df["raw_document"].to_numpy()
    doc_cands = doc_df["candidates"].to_list()

    list_doc_result_lan = []

    unique_sorted = ex_sorted[bounds[:-1]]

    for doc_id in list_doc_id:
        pos = np.searchsorted(unique_sorted, doc_id)
        if pos >= unique_sorted.shape[0] or unique_sorted[pos] != doc_id:
            continue
        s = int(bounds[pos])
        e = int(bounds[pos + 1])
        idxs = order[s:e]

        nz_mask = (
            (ins_start[idxs] != 0) | (ins_stop[idxs] != 0) | (ins_target[idxs] != 0)
        )
        idxs_nz = idxs[nz_mask]
        if idxs_nz.size == 0:
            real_start = 0
            real_stop = 0
            real_target = AnswerTypeRev[0]
        else:
            valid = ins_stop[idxs_nz] > ins_start[idxs_nz]
            idxs_v = idxs_nz[valid]
            real_start = 0
            real_stop = 0
            real_target = AnswerTypeRev[0]
            max_vote = float(LAN_VOTE_THRESHOLD)
            if idxs_v.size > 0:
                votes = ins_start_score[idxs_v] - ins_start_CLS[idxs_v]
                best = int(np.argmax(votes))
                if float(votes[best]) > max_vote:
                    bi = int(idxs_v[best])
                    part_start = int(ins_part[bi]) * 128
                    real_start = int(ins_start[bi]) + part_start
                    real_stop = int(ins_stop[bi]) + part_start + 1
                    real_target = AnswerTypeRev[int(ins_target[bi])]

        di = doc_idx[doc_id]
        lan_start_raw = -1
        lan_stop_raw = -1

        if real_start != 0:
            list_cands = doc_cands[di]
            rs = int(real_start)
            re = int(real_stop)

            best_score = -1
            best_ratio = -1.0
            best_match_id = 1

            for cand_id, cand in enumerate(list_cands):
                lan_start = int(cand[0]["start_token"])
                lan_stop = int(cand[0]["end_token"])
                inter = min(re, lan_stop) - max(rs, lan_start) + 1
                if inter < 0:
                    inter = 0
                lan_len = lan_stop - lan_start + 1
                ratio = float(inter) / lan_len if lan_len > 0 else 0.0

                if inter > best_score:
                    best_score = inter
                    best_match_id = cand_id
                    best_ratio = ratio
                elif inter > 0 and inter == best_score:
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match_id = cand_id

            lan_raw_res = list_cands[best_match_id][1]
            lan_start_raw = lan_raw_res["start_token"]
            lan_stop_raw = lan_raw_res["end_token"]

        doc_result_lan = {
            "example_id": doc_id,
            "question": doc_question[di],
            "raw_document": doc_raw[di],
            "start_token": lan_start_raw,
            "stop_token": lan_stop_raw,
            "target": real_target,
        }
        list_doc_result_lan.append(doc_result_lan)
    return list_doc_result_lan


def preprocess_data(data, tokenizer):
    questions = [d["question"] for d in data]
    contexts = [d["context"] for d in data]
    enc = tokenizer(
        questions,
        contexts,
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    x1 = np.asarray(enc["input_ids"], dtype=np.int32)
    x2 = np.asarray(enc["token_type_ids"], dtype=np.int32)
    x3 = np.asarray(enc["attention_mask"], dtype=np.int32)
    y = np.asarray(
        [[d["start"], d["stop"], AnswerType[d["target"]]] for d in data], dtype=np.int32
    )
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
    except Exception:
        print("No TPU detected")
        strategy = tf.distribute.get_strategy()
    return strategy


def resolve_local_model_dir():
    candidates = [
        "../input/tensorflow2-question-answering/tensorflow-question-answer-fine-data",
        "../input/tensorflow2-question-answering/tensorflow2-question-answering/tensorflow-question-answer-fine-data",
        "../input/tensorflow-question-answer-fine-data",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return "bert-base-uncased"


_MODEL_NAME = None
_TOKENIZER_BASE = None
_TOKENIZER_WITH_TAG = None


def get_model_name_cached():
    global _MODEL_NAME
    if _MODEL_NAME is None:
        _MODEL_NAME = resolve_local_model_dir()
    return _MODEL_NAME


def get_tokenizer_base():
    global _TOKENIZER_BASE
    if _TOKENIZER_BASE is None:
        model_name = get_model_name_cached()
        _TOKENIZER_BASE = AutoTokenizer.from_pretrained(
            model_name, local_files_only=True if os.path.isdir(model_name) else False
        )
    return _TOKENIZER_BASE


def get_tokenizer_with_tag():
    global _TOKENIZER_WITH_TAG
    if _TOKENIZER_WITH_TAG is None:
        model_name = get_model_name_cached()
        _TOKENIZER_WITH_TAG = AutoTokenizer.from_pretrained(
            model_name, local_files_only=True if os.path.isdir(model_name) else False
        )
        _TOKENIZER_WITH_TAG.add_special_tokens({"unk_token": "<tag>"})
    return _TOKENIZER_WITH_TAG


def build_model(model_name):
    BertModel = TFBertModel.from_pretrained(model_name)
    NUM_TARGET = 5

    class MyQAModel(tf.keras.Model):
        def __init__(self, *inputs, **kwargs):
            super().__init__(*inputs, **kwargs)
            self.bert = BertModel
            self.dropout1 = tf.keras.layers.Dropout(0.2)
            self.dropout2 = tf.keras.layers.Dropout(0.2)
            self.start_logits = tf.keras.layers.Dense(1)
            self.stop_logits = tf.keras.layers.Dense(1)
            self.target = tf.keras.layers.Dense(NUM_TARGET)

        def call(self, inputs, **kwargs):
            bert_res = self.bert(
                inputs[0], token_type_ids=inputs[1], attention_mask=inputs[2]
            )
            dropout_res0 = self.dropout1(bert_res[0])
            start_logits = tf.squeeze(self.start_logits(dropout_res0), -1)
            stop_logits = tf.squeeze(self.stop_logits(dropout_res0), -1)
            dropout_res1 = self.dropout1(bert_res[1])
            targets = self.target(dropout_res1)
            paddings = tf.constant([[0, 0], [0, 512 - NUM_TARGET]])
            targets = tf.pad(targets, paddings)
            res = tf.stack([start_logits, stop_logits, targets], axis=1)
            return res

    return MyQAModel()


_STRATEGY = None
_MODEL1 = None
_MODEL2 = None


def _get_strategy_cached():
    global _STRATEGY
    if _STRATEGY is None:
        _STRATEGY = get_strategy()
    return _STRATEGY


def _resolve_weight_path(preferred_filename: str):
    direct_candidates = [
        f"../input/model1/{preferred_filename}",
        f"../input/tensorflow2-question-answering/{preferred_filename}",
        f"../input/tensorflow2-question-answering/tensorflow2-question-answering/{preferred_filename}",
        f"../input/tensorflow2-question-answering/model1/{preferred_filename}",
        f"../input/tensorflow2-question-answering/tensorflow2-question-answering/model1/{preferred_filename}",
    ]
    for p in direct_candidates:
        if os.path.isfile(p):
            return p

    search_roots = [
        "../input/tensorflow2-question-answering",
        "../input",
    ]
    all_h5 = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(".h5"):
                    all_h5.append(os.path.join(dirpath, fn))

    base = os.path.basename(preferred_filename)
    for p in all_h5:
        if os.path.basename(p) == base:
            return p

    stem = os.path.splitext(base)[0]
    scored = []
    for p in all_h5:
        bn = os.path.basename(p)
        score = 0
        if stem in bn:
            score += 10
        if "weights" in bn:
            score += 2
        if "0.555" in bn and "0.555" in base:
            score += 5
        if "0.701" in bn and "0.701" in base:
            score += 5
        scored.append((score, p))
    scored.sort(reverse=True, key=lambda x: x[0])
    if scored and scored[0][0] > 0:
        return scored[0][1]

    raise FileNotFoundError(
        f"Could not find weights file {preferred_filename}. "
        f"Tried direct paths: {direct_candidates}. Also searched for .h5 under {search_roots} but found none matching."
    )


def _compile_and_load(weights_path):
    strategy = _get_strategy_cached()
    model_name = get_model_name_cached()
    with strategy.scope():
        m = build_model(model_name)
        optAdam = tf.keras.optimizers.Adam(learning_rate=0.00005)
        lossSCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metricSCA = tf.keras.metrics.SparseCategoricalAccuracy()
        m.compile(optimizer=optAdam, loss=lossSCE, metrics=[metricSCA])

        dummy_ids = tf.zeros((1, 512), dtype=tf.int32)
        dummy_tt = tf.zeros((1, 512), dtype=tf.int32)
        dummy_am = tf.ones((1, 512), dtype=tf.int32)
        _ = m([dummy_ids, dummy_tt, dummy_am], training=False)

        m.load_weights(weights_path)
    return m


def get_model1():
    global _MODEL1
    if _MODEL1 is None:
        w1 = _resolve_weight_path("weights-02-0.555.h5")
        _MODEL1 = _compile_and_load(w1)
    return _MODEL1


def get_model2():
    global _MODEL2
    if _MODEL2 is None:
        w2 = _resolve_weight_path("weights-02-0.701.h5")
        _MODEL2 = _compile_and_load(w2)
    return _MODEL2


def getRawInstanceResults(list_test, verbose=True, debug=False):
    if verbose:
        print("Getting raw result for all the instances generated from test file")

    tokenizer = get_tokenizer_base()
    testModel = get_model1()
    if verbose:
        print("Finish loading pretrained weights for the model")

    batch = 512
    chunk_size = 4096
    res_chunks = []

    n = len(list_test)
    for s in range(0, n, chunk_size):
        chunk = list_test[s : s + chunk_size]
        x_test1, x_test2, x_test3, _ = preprocess_data(chunk, tokenizer)
        if verbose and s == 0:
            print("Finish tokenizing ", n, " data for the first model (chunked)")

        ds = tf.data.Dataset.from_tensor_slices((x_test1, x_test2, x_test3))
        opts = tf.data.Options()
        opts.experimental_deterministic = True
        ds = (
            ds.with_options(opts)
            .batch(batch, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )
        res = testModel.predict(ds, verbose=0)
        res_chunks.append(np.asarray(res, dtype=np.float32))

    res_all = (
        np.concatenate(res_chunks, axis=0)
        if res_chunks
        else np.zeros((0, 3, 512), dtype=np.float32)
    )
    if verbose:
        print("Finish calculating raw result, get an array of size: ", res_all.shape)
    return res_all


def getListShortSample(doc_lan_df):
    INSTANCE_WORDS_LEN = 500
    STRIDE = 384
    list_instances = []
    for row in doc_lan_df.itertuples(index=False):
        question = row.question
        long_start = row.start_token
        long_stop = row.stop_token
        raw_doc = row.raw_document
        example_id = row.example_id
        if long_start != -1 and long_stop != -1:
            lan_split = raw_doc.split()[long_start:long_stop]
            len_ques = len(question.split())
            len_lan = len(lan_split)
            part_len = INSTANCE_WORDS_LEN - len_ques

            if len_lan > part_len:
                num_ins = (len_lan - part_len) // STRIDE + 2
            else:
                num_ins = 1

            for part_id in range(num_ins):
                part_start = part_id * STRIDE
                part_end = min(len(lan_split), part_id * STRIDE + part_len)
                part = " ".join(lan_split[part_start:part_end])

                instance = {
                    "question": question,
                    "context": part,
                    "example_id": example_id,
                    "part_id": part_id,
                }
                list_instances.append(instance)
    return list_instances


def preprocess_san(list_san_ins):
    tokenizer = get_tokenizer_with_tag()
    questions = [d["question"] for d in list_san_ins]
    contexts = [d["context"] for d in list_san_ins]
    enc = tokenizer(
        questions,
        contexts,
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    x1 = np.asarray(enc["input_ids"], dtype=np.int32)
    x2 = np.asarray(enc["token_type_ids"], dtype=np.int32)
    x3 = np.asarray(enc["attention_mask"], dtype=np.int32)
    return x1, x2, x3


def getSanInstanceResults(list_san_ins, verbose=True):
    if verbose:
        print(
            "Calculating raw result for ", len(list_san_ins), " short answer instances"
        )
    if len(list_san_ins) == 0:
        return np.zeros((0, 3, 512), dtype=np.float32)

    testModel = get_model2()
    if verbose:
        print("Finish loading the weights for the second model")

    batch = 512
    chunk_size = 4096
    res_chunks = []
    n = len(list_san_ins)
    for s in range(0, n, chunk_size):
        chunk = list_san_ins[s : s + chunk_size]
        x_san1, x_san2, x_san3 = preprocess_san(chunk)
        if verbose and s == 0:
            print("Finish tokenizing test instance for the second model (chunked)")

        ds = tf.data.Dataset.from_tensor_slices((x_san1, x_san2, x_san3))
        opts = tf.data.Options()
        opts.experimental_deterministic = True
        ds = (
            ds.with_options(opts)
            .batch(batch, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )
        san_res = testModel.predict(ds, verbose=0)
        res_chunks.append(np.asarray(san_res, dtype=np.float32))

    san_all = (
        np.concatenate(res_chunks, axis=0)
        if res_chunks
        else np.zeros((0, 3, 512), dtype=np.float32)
    )
    if verbose:
        print("Finish calculating, get result of size: ", san_all.shape)
    return san_all


def mergeSanInstanceResult(san_res, list_san_ins, debug=True):
    san_res = san_res.numpy() if hasattr(san_res, "numpy") else np.asarray(san_res)
    start_idx = san_res[:, 0, :].argmax(axis=1)
    stop_idx = san_res[:, 1, :].argmax(axis=1)
    target_idx = san_res[:, 2, :].argmax(axis=1)

    rows = np.arange(san_res.shape[0])
    start_score = san_res[rows, 0, start_idx]
    stop_score = san_res[rows, 1, stop_idx]
    target_score = san_res[rows, 2, target_idx]
    start_CLS = san_res[:, 0, 0]
    stop_CLS = san_res[:, 1, 0]

    for i, ins in enumerate(list_san_ins):
        ins["start"] = int(start_idx[i])
        ins["stop"] = int(stop_idx[i])
        ins["target"] = AnswerTypeRev[int(target_idx[i])]
        ins["start_score"] = float(start_score[i])
        ins["stop_score"] = float(stop_score[i])
        ins["target_score"] = float(target_score[i])
        ins["start_CLS"] = float(start_CLS[i])
        ins["stop_CLS"] = float(stop_CLS[i])
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

    san_example = (
        san_ins_res_df["example_id"].to_numpy()
        if san_ins_res_df.shape[0]
        else np.array([], dtype=object)
    )
    san_groups = {}
    for i, eid in enumerate(san_example):
        san_groups.setdefault(eid, []).append(i)

    if san_ins_res_df.shape[0]:
        san_part = san_ins_res_df["part_id"].to_numpy(np.int32)
        san_start = san_ins_res_df["start"].to_numpy(np.int32)
        san_stop = san_ins_res_df["stop"].to_numpy(np.int32)
        san_target = san_ins_res_df["target"].to_numpy()
        san_start_score = san_ins_res_df["start_score"].to_numpy(np.float32)
        san_start_CLS = san_ins_res_df["start_CLS"].to_numpy(np.float32)

    for row in doc_lan_df.itertuples(index=False):
        doc_id = row.example_id
        lan_start = row.start_token
        lan_stop = row.stop_token
        lan_target = row.target

        out_san_start = -1
        out_san_stop = -1
        final_target = AnswerTypeRev[0]

        idxs = san_groups.get(doc_id, [])
        max_score = float(SAN_VOTE_THRESHOLD)
        for j in idxs:
            part_id = int(san_part[j])
            part_start = part_id * SAN_STRIDE
            start_in_part = int(san_start[j])
            stop_in_part = int(san_stop[j])

            start_in_lan = start_in_part + part_start
            stop_in_lan = stop_in_part + part_start

            if start_in_part > 0 and stop_in_part > start_in_part:
                vote = float(san_start_score[j] - san_start_CLS[j])
                if vote > max_score:
                    out_san_start = start_in_lan + int(lan_start)
                    out_san_stop = stop_in_lan + int(lan_start)
                    max_score = vote
            final_target = str(san_target[j]) if idxs else final_target

        if lan_target != "NO_ANSWER" and lan_target != "SHORT" and lan_target != "LONG":
            if final_target == "NO_ANSWER" or final_target == "LONG":
                final_target = lan_target

        san_lan_res_doc = {
            "example_id": doc_id,
            "question": row.question,
            "raw_document": row.raw_document,
            "lan_start": lan_start,
            "lan_stop": lan_stop,
            "san_start": out_san_start,
            "san_stop": out_san_stop,
            "target": final_target,
        }
        list_san_lan_res_doc.append(san_lan_res_doc)
    return list_san_lan_res_doc


def getFinalResult(f_test):
    list_all_ins, list_doc = parseTestAll(f_test)

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
    lines = {"example_id": [], "PredictionString": []}
    for doc in san_lan_doc:
        doc_id = doc["example_id"]

        long_text = doc_id + "_long"
        lines["example_id"].append(long_text)

        short_text = doc_id + "_short"
        lines["example_id"].append(short_text)

        if doc["lan_start"] != -1 and doc["lan_stop"] != -1:
            long_res = str(doc["lan_start"]) + ":" + str(doc["lan_stop"])
            lines["PredictionString"].append(long_res)
        else:
            lines["PredictionString"].append("")

        if doc["san_start"] != -1 and doc["san_stop"] != -1:
            short_res = str(doc["san_start"]) + ":" + str(doc["san_stop"])
            lines["PredictionString"].append(short_res)
        else:
            lines["PredictionString"].append("")
    return lines


def getSubmission():
    sample_path = "../input/tensorflow2-question-answering/sample_submission.csv"
    if not os.path.isfile(sample_path):
        sample_path = "../input/sample_submission.csv"
    sample_df = pd.read_csv(sample_path)
    sample_df["example_id"] = sample_df["example_id"].astype(str)

    try:
        san_lan_doc = getFinalResult(
            "../input/tensorflow2-question-answering/simplified-nq-test.jsonl"
        )
        pred_df = pd.DataFrame(getLines(san_lan_doc))
        pred_df["example_id"] = pred_df["example_id"].astype(str)
        pred_df["PredictionString"] = pred_df["PredictionString"].fillna("").astype(str)

        merged = sample_df[["example_id"]].merge(
            pred_df, on="example_id", how="left", sort=False
        )
        merged["PredictionString"] = merged["PredictionString"].fillna("").astype(str)
    except Exception as e:
        print("Inference failed; writing blank submission. Error was:", repr(e))
        merged = sample_df.copy()
        merged["PredictionString"] = ""

    merged.to_csv(
        "./submission.csv", index=False, columns=["example_id", "PredictionString"]
    )




## === cell 1
getSubmission()
print("Wrote submission.csv; head:")
print(pd.read_csv("./submission.csv").head())
print("Rows:", pd.read_csv("./submission.csv").shape[0])
print("Columns:", pd.read_csv("./submission.csv").columns.tolist())
