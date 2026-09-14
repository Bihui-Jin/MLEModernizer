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

0.3778966131907308

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_KERAS", "1")

import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import re
import json

import tensorflow as tf

tf.keras.backend.clear_session()

try:
    from transformers import AutoTokenizer, TFBertModel
except Exception as e:
    raise ImportError(
        "transformers is required but not available in this environment. "
        "Please ensure the Kaggle image provides transformers."
    ) from e

SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass


debug = False

BASE1 = "../input/tensorflow2-question-answering"
BASE2 = "../input"


def _first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


f_train = _first_existing(
    os.path.join(BASE1, "simplified-nq-train.jsonl"),
    os.path.join(BASE2, "simplified-nq-train.jsonl"),
)
f_test = _first_existing(
    os.path.join(BASE1, "simplified-nq-test.jsonl"),
    os.path.join(BASE2, "simplified-nq-test.jsonl"),
)

num_train_samples = 44943
num_test_samples = 346

AnswerType = {"NO_ANSWER": 0, "YES": 1, "NO": 2, "SHORT": 3, "LONG": 4}
AnswerTypeRev = {0: "NO_ANSWER", 1: "YES", 2: "NO", 3: "SHORT", 4: "LONG"}

cleanr = re.compile("<.*?>")


def clean_html(raw_html):
    cleantext = re.sub(cleanr, "<tag>", raw_html)
    return cleantext


def _tag_prefix(doc_text_split):
    is_tag = np.fromiter(
        (1 if t == "<tag>" else 0 for t in doc_text_split), dtype=np.int32
    )
    return np.concatenate(([0], np.cumsum(is_tag, dtype=np.int64)))


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

            doc_text_raw = data["document_text"]
            doc_text_raw = clean_html(doc_text_raw)
            doc_text_split = doc_text_raw.split()

            prefix_tag = _tag_prefix(doc_text_split)

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

                        num_tag_doc_before_sans = int(prefix_tag[san_start])
                        num_tag_san = int(prefix_tag[san_stop] - prefix_tag[san_start])

                        new_san_start = san_start - num_tag_doc_before_sans
                        new_san_stop = (
                            new_san_start + (san_stop - san_start) - num_tag_san
                        )
                    else:
                        is_san = False

                    num_tag_doc_before_lan = int(prefix_tag[lan_start])
                    num_tag_lan = int(prefix_tag[lan_stop] - prefix_tag[lan_start])

                    new_lan_start = lan_start - num_tag_doc_before_lan
                    new_lan_stop = new_lan_start + (lan_stop - lan_start) - num_tag_lan

            clean_doc = [t for t in doc_text_split if t != "<tag>"]

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

                                if debug:
                                    print(
                                        "SHORT ANSWER HERE: ",
                                        part[san_start_ins:san_stop_ins],
                                    )
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
                                    if debug:
                                        print(
                                            "LONG ANSWER HERE: ",
                                            part[lan_start_ins:lan_stop_ins],
                                        )

                                    if is_yes_no:
                                        target_ans_ins = yes_no
                                        count_yes_no_ans_ins += 1
                                    else:
                                        count_long_ans_ins += 1
                            else:
                                is_keep = False

                part = " ".join(part)
                instance = {}
                instance["question"] = question
                instance["context"] = part
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
                        if ins_rand > drop_null_instances_rate:
                            count_no_ans_ins += 1
                            if to_second_list:
                                second_list_instances.append(instance)
                            else:
                                list_instances.append(instance)
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


def getRawAndCleanTextDocs(test_file):
    if debug:
        n_sam = 2
    else:
        n_sam = 346
    list_sample = []

    with open(test_file) as f:
        progress = tqdm(f)
        for sam_count, line in enumerate(progress):
            data = json.loads(line)
            example_id = data["example_id"]
            question = data["question_text"]

            doc_text_raw = data["document_text"]
            doc_text_raw = clean_html(doc_text_raw)
            doc_text_split = doc_text_raw.split()
            clean_doc = [t for t in doc_text_split if t != "<tag>"]
            list_candidates = data["long_answer_candidates"]
            list_new_candidates = []
            prefix_tag = _tag_prefix(doc_text_split)
            for cand in list_candidates:
                cand_start = cand["start_token"]
                cand_stop = cand["end_token"]

                num_tag_bef_start = int(prefix_tag[cand_start])
                num_tag_bef_stop = int(prefix_tag[cand_stop])
                new_start = cand_start - num_tag_bef_start
                new_stop = cand_stop - num_tag_bef_stop

                new_cand = {}
                new_cand["end_token"] = new_stop
                new_cand["start_token"] = new_start
                new_cand["top_level"] = cand["top_level"]

                list_new_candidates.append([new_cand, cand])
            sample = {}
            sample["example_id"] = str(example_id)
            sample["question"] = question
            sample["document"] = " ".join(clean_doc)
            sample["raw_document"] = doc_text_raw
            sample["candidates"] = list_new_candidates

            list_sample.append(sample)
    return list_sample


def mergeInstanceResult(test_res, list_test_ins):
    start_idx = np.argmax(test_res[:, 0, :], axis=-1)
    stop_idx = np.argmax(test_res[:, 1, :], axis=-1)
    target_idx = np.argmax(test_res[:, 2, :], axis=-1)

    start_score = test_res[np.arange(test_res.shape[0]), 0, start_idx]
    stop_score = test_res[np.arange(test_res.shape[0]), 1, stop_idx]
    target_score = test_res[np.arange(test_res.shape[0]), 2, target_idx]

    start_cls = test_res[:, 0, 0]
    stop_cls = test_res[:, 1, 0]

    for i in range(len(list_test_ins)):
        list_test_ins[i]["start"] = int(start_idx[i])
        list_test_ins[i]["stop"] = int(stop_idx[i])
        list_test_ins[i]["target"] = int(target_idx[i])
        list_test_ins[i]["start_score"] = float(start_score[i])
        list_test_ins[i]["stop_score"] = float(stop_score[i])
        list_test_ins[i]["target_score"] = float(target_score[i])
        list_test_ins[i]["start_CLS"] = float(start_cls[i])
        list_test_ins[i]["stop_CLS"] = float(stop_cls[i])
    return list_test_ins


def mergeDocumentResult(doc_df, ins_df):
    ins_by_doc = {k: v for k, v in ins_df.groupby("example_id", sort=False)}
    doc_by_id = {row.example_id: row for row in doc_df.itertuples(index=False)}

    list_doc_result_lan = []
    for doc_id, samrow in doc_by_id.items():
        ins_of_doc = ins_by_doc.get(doc_id)
        if ins_of_doc is None or ins_of_doc.shape[0] == 0:
            doc_result_lan = {
                "example_id": doc_id,
                "question": samrow.question,
                "raw_document": samrow.raw_document,
                "start_token": -1,
                "stop_token": -1,
                "target": AnswerTypeRev[0],
            }
            list_doc_result_lan.append(doc_result_lan)
            continue

        mask = (
            (ins_of_doc["start"].to_numpy() != 0)
            | (ins_of_doc["stop"].to_numpy() != 0)
            | (ins_of_doc["target"].to_numpy() != 0)
        )
        cand_rows = ins_of_doc.loc[mask]

        real_start = 0
        real_stop = 0
        real_target = AnswerTypeRev[0]
        max_vote = 0.005

        lan_start_raw = -1
        lan_stop_raw = -1

        part_ids = cand_rows["part_id"].to_numpy(dtype=np.int64)
        starts = cand_rows["start"].to_numpy(dtype=np.int64)
        stops = cand_rows["stop"].to_numpy(dtype=np.int64)
        votes = (
            cand_rows["start_score"].to_numpy() - cand_rows["start_CLS"].to_numpy()
        ) + (cand_rows["stop_score"].to_numpy() - cand_rows["stop_CLS"].to_numpy())
        targets = cand_rows["target"].to_numpy(dtype=np.int64)

        valid = stops > starts
        if np.any(valid):
            v_votes = votes[valid]
            v_idx = np.argmax(v_votes)
            if v_votes[v_idx] > max_vote:
                valid_idx = np.flatnonzero(valid)[v_idx]
                part_start = int(part_ids[valid_idx]) * 128
                real_start = int(starts[valid_idx]) + part_start
                real_stop = int(stops[valid_idx]) + part_start + 1
                real_target = AnswerTypeRev[int(targets[valid_idx])]
                max_vote = float(v_votes[v_idx])

        if real_start != 0:
            list_cands = samrow.candidates
            best_inter = 0
            best_ratio = 0.0
            best_match_id = 0
            rs = real_start
            re_ = real_stop
            for cand_id, cand in enumerate(list_cands):
                lan_start = int(cand[0]["start_token"])
                lan_stop = int(cand[0]["end_token"])
                inter = min(re_, lan_stop) - max(rs, lan_start) + 1
                if inter < 0:
                    inter = 0
                if inter > 0:
                    lan_len = max(1, (lan_stop - lan_start + 1))
                    ratio = float(inter) / lan_len
                    if inter > best_inter or (
                        inter == best_inter and ratio > best_ratio
                    ):
                        best_inter = inter
                        best_match_id = cand_id
                        best_ratio = ratio
            lan_raw_res = list_cands[best_match_id][1]
            lan_start_raw = lan_raw_res["start_token"]
            lan_stop_raw = lan_raw_res["end_token"]

        doc_result_lan = {
            "example_id": doc_id,
            "question": samrow.question,
            "raw_document": samrow.raw_document,
            "start_token": lan_start_raw,
            "stop_token": lan_stop_raw,
            "target": real_target,
        }
        list_doc_result_lan.append(doc_result_lan)

    return list_doc_result_lan


_MODEL_NAME_CACHE = None
_TOKENIZER_CACHE = {}


def _resolve_local_model_dir():
    candidates = [
        "../input/tensorflow-question-answer-fine-data",
        "../input/tensorflow2-question-answering/tensorflow-question-answer-fine-data",
        "../input/tensorflow2-question-answering/tensorflow2-question-answering/tensorflow-question-answer-fine-data",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return "bert-base-uncased"


def _get_model_name():
    global _MODEL_NAME_CACHE
    if _MODEL_NAME_CACHE is None:
        _MODEL_NAME_CACHE = _resolve_local_model_dir()
    return _MODEL_NAME_CACHE


def _get_tokenizer():
    model_name = _get_model_name()
    tok = _TOKENIZER_CACHE.get(model_name)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(
            model_name, local_files_only=(model_name != "bert-base-uncased")
        )
        _TOKENIZER_CACHE[model_name] = tok
    return tok


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
        return_token_type_ids=True,
        return_attention_mask=True,
    )
    x1 = np.asarray(enc["input_ids"], dtype=np.int32)
    x2 = np.asarray(enc.get("token_type_ids", [[0] * 512] * len(data)), dtype=np.int32)
    x3 = np.asarray(enc["attention_mask"], dtype=np.int32)
    y = np.asarray(
        [[d["start"], d["stop"], AnswerType[d["target"]]] for d in data], dtype=np.int32
    )
    return x1, x2, x3, y


_STRATEGY_CACHE = None


def get_strategy():
    global _STRATEGY_CACHE
    if _STRATEGY_CACHE is not None:
        return _STRATEGY_CACHE
    try:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        print(
            "Running on TPU ", tpu_cluster_resolver.cluster_spec().as_dict()["worker"]
        )
        tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
        strategy = tf.distribute.experimental.TPUStrategy(tpu_cluster_resolver)
    except Exception as e:
        print(str(e))
        print("No TPU detected")
        strategy = tf.distribute.get_strategy()
    _STRATEGY_CACHE = strategy
    return strategy


_BERT_BACKBONE_CACHE = {}


def build_model(model_name):
    bert = _BERT_BACKBONE_CACHE.get(model_name)
    if bert is None:
        bert = TFBertModel.from_pretrained(model_name)
        _BERT_BACKBONE_CACHE[model_name] = bert

    NUM_TARGET = 5

    class MyQAModel(tf.keras.Model):
        def __init__(self, *inputs, **kwargs):
            super().__init__(*inputs, **kwargs)
            self.bert = bert
            self.dropout1 = tf.keras.layers.Dropout(0.2)
            self.dropout2 = tf.keras.layers.Dropout(0.2)

            self.start_logits = tf.keras.layers.Dense(1)
            self.stop_logits = tf.keras.layers.Dense(1)

            self.target = tf.keras.layers.Dense(NUM_TARGET)

        def call(self, inputs, **kwargs):
            bert_res = self.bert(
                inputs[0],
                token_type_ids=inputs[1],
                attention_mask=inputs[2],
                training=kwargs.get("training", False),
            )

            dropout_res0 = self.dropout1(
                bert_res[0], training=kwargs.get("training", False)
            )
            start_logits = tf.squeeze(self.start_logits(dropout_res0), -1)
            stop_logits = tf.squeeze(self.stop_logits(dropout_res0), -1)

            dropout_res1 = self.dropout1(
                bert_res[1], training=kwargs.get("training", False)
            )
            targets = self.target(dropout_res1)
            paddings = tf.constant([[[0, 0]], [[0, 512 - NUM_TARGET]]])
            targets = tf.pad(targets, paddings)

            res = tf.stack([start_logits, stop_logits, targets], axis=1)
            return res

    myQAModel = MyQAModel()
    return myQAModel


_MODEL_CACHE = {}


def _get_compiled_model(weights_path):
    model = _MODEL_CACHE.get(weights_path)
    if model is not None:
        return model
    model_name = _get_model_name()
    strategy = get_strategy()
    with strategy.scope():
        model = build_model(model_name)
        optAdam = tf.keras.optimizers.Adam(learning_rate=0.00005)
        lossSCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metricSCA = tf.keras.metrics.SparseCategoricalAccuracy()
        model.compile(optimizer=optAdam, loss=lossSCE, metrics=[metricSCA])
        model.load_weights(weights_path)
    _MODEL_CACHE[weights_path] = model
    return model


def getRawInstanceResults(list_test, verbose=True, debug=False):
    if verbose:
        print("Getting raw result for all the instances generated from test file")

    tokenizer = _get_tokenizer()
    x_test1, x_test2, x_test3, y_test = preprocess_data(list_test, tokenizer)
    if verbose:
        print("Finish tokenizing ", len(list_test), " data for the first model")

    x_test1 = tf.constant(x_test1, dtype=tf.int32)
    x_test2 = tf.constant(x_test2, dtype=tf.int32)
    x_test3 = tf.constant(x_test3, dtype=tf.int32)

    w1 = _first_existing(
        "../input/model1/weights-02-0.555.h5",
        "../input/tensorflow2-question-answering/model1/weights-02-0.555.h5",
    )
    testModel = _get_compiled_model(w1)

    if verbose:
        print("Finish loading pretrained weights for the model")

    res = testModel.predict([x_test1, x_test2, x_test3], batch_size=256, verbose=0)
    res = tf.nn.softmax(res, axis=-1).numpy()

    if verbose:
        print("Finish calculating raw result, get an array of size: ", res.shape)
    return res


def getListShortSample(doc_lan_df):
    INSTANCE_WORDS_LEN = 500
    STRIDE = 384
    list_instances = []
    for row in doc_lan_df.itertuples(index=False):
        question = row.question
        long_start = int(row.start_token)
        long_stop = int(row.stop_token)
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

                part = lan_split[part_start:part_end]
                part = " ".join(part)

                instance = {}
                instance["question"] = question
                instance["context"] = part
                instance["example_id"] = example_id
                instance["part_id"] = part_id

                list_instances.append(instance)
    return list_instances


def preprocess_san(list_san_ins):
    tokenizer = _get_tokenizer()
    questions = [d["question"] for d in list_san_ins]
    contexts = [d["context"] for d in list_san_ins]
    enc = tokenizer(
        questions,
        contexts,
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        return_token_type_ids=True,
        return_attention_mask=True,
    )
    x1 = np.asarray(enc["input_ids"], dtype=np.int32)
    x2 = np.asarray(
        enc.get("token_type_ids", [[0] * 512] * len(list_san_ins)), dtype=np.int32
    )
    x3 = np.asarray(enc["attention_mask"], dtype=np.int32)
    return x1, x2, x3


def getSanInstanceResults(list_san_ins, verbose=True):
    if verbose:
        print(
            "Calculating raw result for ", len(list_san_ins), " short answer instances"
        )

    x_san1, x_san2, x_san3 = preprocess_san(list_san_ins)
    if verbose:
        print("Finish tokenizing test instance for the second model")

    x_san1 = tf.constant(x_san1, dtype=tf.int32)
    x_san2 = tf.constant(x_san2, dtype=tf.int32)
    x_san3 = tf.constant(x_san3, dtype=tf.int32)

    w2 = _first_existing(
        "../input/model1/weights-02-0.701.h5",
        "../input/tensorflow2-question-answering/model1/weights-02-0.701.h5",
    )
    testModel = _get_compiled_model(w2)

    if verbose:
        print("Finish loading the weights for the second model")

    san_res = testModel.predict([x_san1, x_san2, x_san3], batch_size=256, verbose=0)
    san_res = tf.nn.softmax(san_res, axis=-1)

    if verbose:
        print("Finish calculating, get result of size: ", san_res.shape)

    return san_res


def mergeSanInstanceResult(san_res, list_san_ins, debug=True):
    san_np = san_res.numpy() if hasattr(san_res, "numpy") else np.asarray(san_res)
    start_idx = np.argmax(san_np[:, 0, :], axis=-1)
    stop_idx = np.argmax(san_np[:, 1, :], axis=-1)
    target_idx = np.argmax(san_np[:, 2, :], axis=-1)

    start_score = san_np[np.arange(san_np.shape[0]), 0, start_idx]
    stop_score = san_np[np.arange(san_np.shape[0]), 1, stop_idx]
    target_score = san_np[np.arange(san_np.shape[0]), 2, target_idx]

    start_cls = san_np[:, 0, 0]
    stop_cls = san_np[:, 1, 0]

    for i in range(len(list_san_ins)):
        start = int(start_idx[i])
        stop = int(stop_idx[i])
        target = int(target_idx[i])

        list_san_ins[i]["start"] = start
        list_san_ins[i]["stop"] = stop
        list_san_ins[i]["target"] = AnswerTypeRev[target]

        list_san_ins[i]["start_score"] = float(start_score[i])
        list_san_ins[i]["stop_score"] = float(stop_score[i])
        list_san_ins[i]["target_score"] = float(target_score[i])

        list_san_ins[i]["start_CLS"] = float(start_cls[i])
        list_san_ins[i]["stop_CLS"] = float(stop_cls[i])

        if debug:
            question = list_san_ins[i]["question"]
            context = list_san_ins[i]["context"]
            context_split = context.split()

            if start > 0 and stop > start:
                print(
                    AnswerTypeRev[target],
                    ": ",
                    question,
                    ": ",
                    " ".join(context_split[start : stop + 1]),
                )
                print(
                    float(start_score[i] - start_cls[i]),
                    float(start_score[i] - start_cls[i] + stop_score[i] - stop_cls[i]),
                )
            if target > 0 and target != 3:
                print(AnswerTypeRev[target], ": ", question, ": ", context)
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

    san_by_doc = {k: v for k, v in san_ins_res_df.groupby("example_id", sort=False)}

    for row in doc_lan_df.itertuples(index=False):
        doc_id = row.example_id

        lan_start = int(row.start_token)
        lan_stop = int(row.stop_token)
        lan_target = row.target

        san_start = -1
        san_stop = -1
        final_target = AnswerTypeRev[0]

        san_list = san_by_doc.get(doc_id)
        max_score = 0.00001
        if san_list is not None and san_list.shape[0] > 0:
            part_id = san_list["part_id"].to_numpy(dtype=np.int64)
            part_start = part_id * SAN_STRIDE
            start_in_part = san_list["start"].to_numpy(dtype=np.int64)
            stop_in_part = san_list["stop"].to_numpy(dtype=np.int64)
            valid = (start_in_part > 0) & (stop_in_part > start_in_part)
            if np.any(valid):
                score = (
                    san_list["start_score"].to_numpy()
                    - san_list["start_CLS"].to_numpy()
                ) + (
                    san_list["stop_score"].to_numpy() - san_list["stop_CLS"].to_numpy()
                )
                v_score = score[valid]
                best = np.argmax(v_score)
                if float(v_score[best]) > max_score:
                    idx = np.flatnonzero(valid)[best]
                    start_in_lan = int(start_in_part[idx] + part_start[idx])
                    stop_in_lan = int(stop_in_part[idx] + part_start[idx])
                    san_start = start_in_lan + lan_start
                    san_stop = stop_in_lan + lan_start
                    max_score = float(v_score[best])
            final_target = san_list["target"].iloc[-1]

        if lan_target != "NO_ANSWER" and lan_target not in ("SHORT", "LONG"):
            if final_target in ("NO_ANSWER", "LONG"):
                final_target = lan_target

        san_lan_res_doc = {}
        san_lan_res_doc["example_id"] = doc_id
        san_lan_res_doc["question"] = row.question
        san_lan_res_doc["raw_document"] = row.raw_document
        san_lan_res_doc["lan_start"] = lan_start
        san_lan_res_doc["lan_stop"] = lan_stop
        san_lan_res_doc["san_start"] = san_start
        san_lan_res_doc["san_stop"] = san_stop
        san_lan_res_doc["target"] = final_target

        list_san_lan_res_doc.append(san_lan_res_doc)
    return list_san_lan_res_doc


def _parse_test_docs_and_instances(filename):
    INSTANCE_WORDS_LEN = 500
    STRIDE = 128
    list_instances = []
    list_sample = []

    with open(filename) as f:
        progress = tqdm(f)
        for sam_count, line in enumerate(progress):
            data = json.loads(line)
            example_id = str(data["example_id"])
            question = data["question_text"]

            doc_text_raw = clean_html(data["document_text"])
            doc_text_split = doc_text_raw.split()
            clean_doc = [t for t in doc_text_split if t != "<tag>"]

            list_candidates = data["long_answer_candidates"]
            list_new_candidates = []
            prefix_tag = _tag_prefix(doc_text_split)
            for cand in list_candidates:
                cand_start = cand["start_token"]
                cand_stop = cand["end_token"]

                num_tag_bef_start = int(prefix_tag[cand_start])
                num_tag_bef_stop = int(prefix_tag[cand_stop])
                new_start = cand_start - num_tag_bef_start
                new_stop = cand_stop - num_tag_bef_stop

                new_cand = {
                    "end_token": new_stop,
                    "start_token": new_start,
                    "top_level": cand["top_level"],
                }
                list_new_candidates.append([new_cand, cand])

            sample = {
                "example_id": example_id,
                "question": question,
                "document": " ".join(clean_doc),
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
                part = " ".join(clean_doc[part_start:part_end])
                instance = {
                    "question": question,
                    "context": part,
                    "example_id": example_id,
                    "part_id": part_id,
                    "target": "NO_ANSWER",
                    "start": 0,
                    "stop": 0,
                }
                list_instances.append(instance)

    return list_sample, list_instances


def getFinalResult(f_test):
    list_doc, list_all_ins = _parse_test_docs_and_instances(f_test)

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
        line1, line2 = {}, {}

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
    lines_df = pd.DataFrame(lines)

    lines_df["PredictionString"] = lines_df["PredictionString"].fillna("").astype(str)
    lines_df["example_id"] = lines_df["example_id"].astype(str)

    sample_path = _first_existing(
        "../input/sample_submission.csv",
        "../input/tensorflow2-question-answering/sample_submission.csv",
    )
    sample_df = pd.read_csv(sample_path)
    sample_df["example_id"] = sample_df["example_id"].astype(str)

    out_df = sample_df[["example_id"]].merge(
        lines_df[["example_id", "PredictionString"]], on="example_id", how="left"
    )
    out_df["PredictionString"] = out_df["PredictionString"].fillna("").astype(str)

    out_df.to_csv(
        "./submission.csv", index=False, columns=["example_id", "PredictionString"]
    )
    print("Wrote submission.csv with shape:", out_df.shape)
    return out_df




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
getSubmission()

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/1001573765.py in <cell line: 0>()
----> 1 getSubmission()

/tmp/ipykernel_11/2403633801.py in getSubmission()
    976 
    977 def getSubmission():
--> 978     san_lan_doc = getFinalResult(f_test)
    979     lines = getLines(san_lan_doc)
    980     lines_df = pd.DataFrame(lines)

/tmp/ipykernel_11/2403633801.py in getFinalResult(f_test)
    923 
    924 def getFinalResult(f_test):
--> 925     list_doc, list_all_ins = _parse_test_docs_and_instances(f_test)
    926 
    927     all_ins_res = getRawInstanceResults(list_all_ins)

/tmp/ipykernel_11/2403633801.py in _parse_test_docs_and_instances(filename)
    880 
    881                 num_tag_bef_start = int(prefix_tag[cand_start])
--> 882                 num_tag_bef_stop = int(prefix_tag[cand_stop])
    883                 new_start = cand_start - num_tag_bef_start
    884                 new_stop = cand_stop - num_tag_bef_stop

IndexError: index 10502 is out of bounds for axis 0 with size 10350
