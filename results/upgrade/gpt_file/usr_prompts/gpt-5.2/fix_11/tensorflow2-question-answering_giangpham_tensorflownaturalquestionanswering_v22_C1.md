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
import os, sys, json, re, random, shutil, string

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf

from transformers import AutoTokenizer, TFBertModel

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

debug = False

f_train = "/kaggle/input/simplified-nq-train.jsonl"
f_test = "/kaggle/input/simplified-nq-test.jsonl"

num_train_samples = 44943
num_test_samples = 346

AnswerType = {"NO_ANSWER": 0, "YES": 1, "NO": 2, "SHORT": 3, "LONG": 4}
AnswerTypeRev = {0: "NO_ANSWER", 1: "YES", 2: "NO", 3: "SHORT", 4: "LONG"}

cleanr = re.compile("<.*?>")


def clean_html(raw_html):
    cleantext = re.sub(cleanr, "<tag>", raw_html)
    return cleantext


def resolve_local_model_dir():
    base_candidates = [
        "/kaggle/input/tensorflow2-question-answering/tensorflow2-question-answering",
        "/kaggle/input/tensorflow2-question-answering",
        "/kaggle/input",
    ]

    def is_valid_hf_dir(p: str) -> bool:
        if not os.path.isdir(p):
            return False
        cfg = os.path.join(p, "config.json")
        if not os.path.isfile(cfg):
            return False
        return True

    for p in base_candidates:
        if is_valid_hf_dir(p):
            return p

    search_root = "/kaggle/input"
    max_depth = 6
    for root, dirs, files in os.walk(search_root):
        depth = root[len(search_root) :].count(os.sep)
        if depth > max_depth:
            dirs[:] = []
            continue
        if "config.json" in files and is_valid_hf_dir(root):
            return root

    return "bert-base-uncased"


def resolve_weights():
    """
    Bugfix: original notebook assumed attached .h5 weights, but they may be absent.
    We return (None, None) when not found, and later run with base checkpoint only.
    """
    preferred = [
        "/kaggle/input/model1/weights-02-0.555.h5",
        "/kaggle/input/model1/weights-02-0.701.h5",
    ]
    if all(os.path.isfile(p) for p in preferred):
        return preferred[0], preferred[1]

    search_root = "/kaggle/input"
    max_depth = 6
    found = []
    for root, dirs, files in os.walk(search_root):
        depth = root[len(search_root) :].count(os.sep)
        if depth > max_depth:
            dirs[:] = []
            continue
        for fn in files:
            fn_l = fn.lower()
            if fn_l.endswith(".h5") and ("weights" in fn_l or "weight" in fn_l):
                found.append(os.path.join(root, fn))

    if not found:
        return None, None

    def rank_key(p):
        name = os.path.basename(p)
        s = 0
        if "0.701" in name:
            s += 100
        if "0.555" in name:
            s += 90
        if "weights-02" in name:
            s += 50
        if "weights" in name.lower():
            s += 10
        return (-s, len(p))

    found_sorted = sorted(found, key=rank_key)
    if len(found_sorted) == 1:
        return found_sorted[0], found_sorted[0]
    return found_sorted[0], found_sorted[1]


MODEL_DIR = resolve_local_model_dir()
MODEL1_WEIGHTS, MODEL2_WEIGHTS = resolve_weights()

print("Resolved MODEL_DIR:", MODEL_DIR)
print("Resolved MODEL1_WEIGHTS:", MODEL1_WEIGHTS)
print("Resolved MODEL2_WEIGHTS:", MODEL2_WEIGHTS)

_LOCAL_ONLY = os.path.isdir(MODEL_DIR)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=_LOCAL_ONLY)

try:
    TOKENIZER.add_special_tokens({"unk_token": "<tag>"})
except Exception:
    pass


def _prefix_tag_counts(tokens, tag_token="<tag>"):
    pref = np.empty(len(tokens) + 1, dtype=np.int32)
    pref[0] = 0
    c = 0
    for i, t in enumerate(tokens):
        if t == tag_token:
            c += 1
        pref[i + 1] = c
    return pref


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

            tag_pref = _prefix_tag_counts(doc_text_split, "<tag>")

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

                        num_tag_doc_before_sans = int(tag_pref[san_start])
                        num_tag_san = int(tag_pref[san_stop] - tag_pref[san_start])

                        new_san_start = san_start - num_tag_doc_before_sans
                        new_san_stop = (
                            new_san_start + san_stop - san_start - num_tag_san
                        )
                    else:
                        is_san = False

                    num_tag_doc_before_lan = int(tag_pref[lan_start])
                    num_tag_lan = int(tag_pref[lan_stop] - tag_pref[lan_start])

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

                        if debug:
                            print(target_ans_ins)
                            print(an_start_ins, an_stop_ins)
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
            if debug:
                print("\n")

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


def parse_test_once(test_file):
    INSTANCE_WORDS_LEN = 500
    STRIDE = 128

    list_instances = []
    list_docs = []

    with open(test_file) as f:
        for line in tqdm(f):
            data = json.loads(line)
            example_id = str(data["example_id"])
            question = data["question_text"]

            doc_text_raw = clean_html(data["document_text"])
            doc_text_split = doc_text_raw.split()

            tag_pref = _prefix_tag_counts(doc_text_split, "<tag>")
            doc_len = len(doc_text_split)

            clean_doc = list(filter(("<tag>").__ne__, doc_text_split))

            list_candidates = data["long_answer_candidates"]
            list_new_candidates = []
            for cand in list_candidates:
                cand_start = int(cand["start_token"])
                cand_stop = int(cand["end_token"])

                cand_start = max(0, min(cand_start, doc_len))
                cand_stop = max(0, min(cand_stop, doc_len))

                num_tag_bef_start = int(tag_pref[cand_start])
                num_tag_bef_stop = int(tag_pref[cand_stop])

                new_start = cand_start - num_tag_bef_start
                new_stop = cand_stop - num_tag_bef_stop

                new_cand = {"end_token": new_stop, "start_token": new_start}
                list_new_candidates.append([new_cand, cand])

            sample = {
                "example_id": example_id,
                "question": question,
                "document": " ".join(clean_doc),
                "raw_document": doc_text_raw,
                "candidates": list_new_candidates,
            }
            list_docs.append(sample)

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

    return list_instances, list_docs


def mergeInstanceResult(test_res, list_test_ins):
    test_res = np.asarray(test_res)
    for i in range(len(list_test_ins)):
        ins_res = test_res[i]
        start = int(np.argmax(ins_res[0]))
        stop = int(np.argmax(ins_res[1]))
        target = int(np.argmax(ins_res[2]))

        start_score = float(ins_res[0][start])
        stop_score = float(ins_res[1][stop])
        target_score = float(ins_res[2][target])

        start_CLS = float(ins_res[0][0])
        stop_CLS = float(ins_res[1][0])

        list_test_ins[i]["start"] = start
        list_test_ins[i]["stop"] = stop
        list_test_ins[i]["target"] = target

        list_test_ins[i]["start_score"] = start_score
        list_test_ins[i]["stop_score"] = stop_score
        list_test_ins[i]["target_score"] = target_score

        list_test_ins[i]["start_CLS"] = start_CLS
        list_test_ins[i]["stop_CLS"] = stop_CLS
    return list_test_ins


def mergeDocumentResult(doc_df, ins_df, debug_local=False):
    grouped = {}
    for row in ins_df.itertuples(index=False):
        d = getattr(row, "example_id")
        grouped.setdefault(d, []).append(row)

    list_doc_result_lan = []
    for doc in doc_df.itertuples(index=False):
        doc_id = getattr(doc, "example_id")
        question = getattr(doc, "question")
        raw_document = getattr(doc, "raw_document")
        candidates = getattr(doc, "candidates")

        if debug_local:
            print(doc_id, " - QUESTION: ", question)

        ins_rows = grouped.get(doc_id, [])

        real_start = 0
        real_stop = 0
        real_target = AnswerTypeRev[0]
        max_vote = 0.05

        lan_start_raw = -1
        lan_stop_raw = -1

        for r in ins_rows:
            if (
                getattr(r, "start") == 0
                and getattr(r, "stop") == 0
                and getattr(r, "target") == 0
            ):
                continue

            part_id = getattr(r, "part_id")
            part_start = part_id * 128
            start = getattr(r, "start")
            stop = getattr(r, "stop")

            vote = (getattr(r, "start_score") - getattr(r, "start_CLS")) + (
                getattr(r, "stop_score") - getattr(r, "stop_CLS")
            )
            if vote > max_vote:
                real_start = start + part_start
                real_stop = stop + part_start + 1
                real_target = AnswerTypeRev[getattr(r, "target")]
                max_vote = vote

        if debug_local:
            doc_tokens = getattr(doc, "document").split()
            print(
                max_vote, real_target, ": ", " ".join(doc_tokens[real_start:real_stop])
            )

        if real_start != 0 and real_stop != 0:
            best_score = 0.5
            shortest = 10**18
            best_match_id = 1

            for cand_id, cand in enumerate(candidates):
                lan_start = cand[0]["start_token"]
                lan_stop = cand[0]["end_token"]

                inter = max(0, min(real_stop, lan_stop) - max(real_start, lan_start))

                if inter > best_score:
                    best_score = float(inter)
                    best_match_id = cand_id
                    shortest = lan_stop - lan_start
                elif inter == best_score:
                    ln = lan_stop - lan_start
                    if shortest > ln:
                        shortest = ln
                        best_match_id = cand_id

            lan_raw_res = candidates[best_match_id][1]
            lan_start_raw = lan_raw_res["start_token"]
            lan_stop_raw = lan_raw_res["end_token"]

            if debug_local:
                print(" ".join(raw_document.split()[lan_start_raw:lan_stop_raw]))

        doc_result_lan = {
            "example_id": doc_id,
            "question": question,
            "raw_document": raw_document,
            "start_token": lan_start_raw,
            "stop_token": lan_stop_raw,
            "target": real_target,
        }
        list_doc_result_lan.append(doc_result_lan)

        if debug_local:
            print("\n")

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
        return_tensors="np",
    )
    x1 = enc["input_ids"].astype(np.int32, copy=False)
    x2 = enc["token_type_ids"].astype(np.int32, copy=False)
    x3 = enc["attention_mask"].astype(np.int32, copy=False)
    y = np.asarray(
        [[d["start"], d["stop"], AnswerType[d["target"]]] for d in data], dtype=np.int32
    )
    return x1, x2, x3, y


_STRATEGY = None


def get_strategy():
    global _STRATEGY
    if _STRATEGY is not None:
        return _STRATEGY
    try:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        print(
            "Running on TPU ", tpu_cluster_resolver.cluster_spec().as_dict()["worker"]
        )
        tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
        _STRATEGY = tf.distribute.experimental.TPUStrategy(tpu_cluster_resolver)
    except Exception as e:
        print(e)
        print("No TPU detected")
        _STRATEGY = tf.distribute.get_strategy()
    return _STRATEGY


def build_model(model_name, local_only):
    BertModelLayer = TFBertModel.from_pretrained(
        model_name, local_files_only=local_only
    )

    NUM_TARGET = 5

    class MyQAModel(tf.keras.Model):
        def __init__(self, *inputs, **kwargs):
            super().__init__(*inputs, **kwargs)
            self.bert = BertModelLayer
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

    myQAModel = MyQAModel()
    return myQAModel


@tf.function(reduce_retracing=True)
def _infer_probs(model, b1, b2, b3):
    logits = model([b1, b2, b3], training=False)
    return tf.nn.softmax(logits, axis=-1)


def _safe_load_weights(model, weights_path, verbose=True):
    if weights_path is None:
        if verbose:
            print(
                "No .h5 weights provided; using base transformer checkpoint with randomly initialized QA heads."
            )
        return False
    if not os.path.isfile(weights_path):
        if verbose:
            print(
                "Weights path not found:",
                weights_path,
                "-> using base transformer checkpoint with randomly initialized QA heads.",
            )
        return False
    model.load_weights(weights_path)
    return True


def getRawInstanceResults(list_test, verbose=True, debug_local=False):
    if verbose:
        print("Getting raw result for all the instances generated from test file")

    x_test1, x_test2, x_test3, y_test = preprocess_data(list_test, TOKENIZER)
    if verbose:
        print("Finish tokenizing ", len(list_test), " data for the first model")

    strategy = get_strategy()
    with strategy.scope():
        testModel = build_model(MODEL_DIR, local_only=_LOCAL_ONLY)
        _ = testModel(
            [
                tf.convert_to_tensor(x_test1[:1]),
                tf.convert_to_tensor(x_test2[:1]),
                tf.convert_to_tensor(x_test3[:1]),
            ],
            training=False,
        )
        _safe_load_weights(testModel, MODEL1_WEIGHTS, verbose=verbose)

    if verbose:
        print("Finish preparing model for inference")

    bs = 256
    ds = tf.data.Dataset.from_tensor_slices((x_test1, x_test2, x_test3)).batch(
        bs, drop_remainder=False
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)

    n = x_test1.shape[0]
    out = np.empty((n, 3, 512), dtype=np.float32)
    offset = 0
    for b1, b2, b3 in ds:
        probs = _infer_probs(testModel, b1, b2, b3)
        bs_eff = int(probs.shape[0])
        out[offset : offset + bs_eff] = probs.numpy()
        offset += bs_eff
    test_res = out

    if verbose:
        print("Finish calculating raw result, get an array of size: ", test_res.shape)
    return test_res


def getListShortSample(doc_lan_df, debug_local=False):
    INSTANCE_WORDS_LEN = 500
    STRIDE = 384
    list_instances = []
    for index, row in doc_lan_df.iterrows():
        question = row["question"]
        long_start = row["start_token"]
        long_stop = row["stop_token"]
        raw_doc = row["raw_document"]
        example_id = row["example_id"]
        target = row["target"]
        if long_start != -1 and long_stop != -1 and target == "SHORT":
            if debug_local:
                print(" ".join(raw_doc.split()[long_start:long_stop]))
            lan_split = raw_doc.split()[long_start:long_stop]
            len_ques = len(question.split())
            len_lan = len(lan_split)
            part_len = INSTANCE_WORDS_LEN - len_ques

            if len_lan > part_len:
                num_ins = (len_lan - part_len) // STRIDE + 2
            else:
                num_ins = 1
            if debug_local:
                print("Num instance: ", num_ins)

            for part_id in range(num_ins):
                part_start = part_id * STRIDE
                part_end = min(len(lan_split), part_id * STRIDE + part_len)

                part = lan_split[part_start:part_end]
                part = " ".join(part)

                if debug_local:
                    print(part)
                instance = {}
                instance["question"] = question
                instance["context"] = part
                instance["example_id"] = example_id
                instance["part_id"] = part_id

                list_instances.append(instance)
                if debug_local:
                    print("Part ID: ", part_id)
                    print("\n")
    return list_instances


def preprocess_san(list_san_ins):
    if len(list_san_ins) == 0:
        return (
            np.zeros((0, 512), dtype=np.int32),
            np.zeros((0, 512), dtype=np.int32),
            np.zeros((0, 512), dtype=np.int32),
        )

    questions = [d["question"] for d in list_san_ins]
    contexts = [d["context"] for d in list_san_ins]
    enc = TOKENIZER(
        questions,
        contexts,
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="np",
    )
    x1 = enc["input_ids"].astype(np.int32, copy=False)
    x2 = enc["token_type_ids"].astype(np.int32, copy=False)
    x3 = enc["attention_mask"].astype(np.int32, copy=False)
    return x1, x2, x3


def getSanInstanceResults(list_san_ins, verbose=True):
    if len(list_san_ins) == 0:
        if verbose:
            print("No short answer instances to score; skipping short answer model.")
        return tf.zeros((0, 3, 512), dtype=tf.float32)

    if verbose:
        print(
            "Calculating raw result for ", len(list_san_ins), " short answer instances"
        )

    x_san1, x_san2, x_san3 = preprocess_san(list_san_ins)
    if verbose:
        print("Finish tokenizing test instance for the second model")

    strategy = get_strategy()
    with strategy.scope():
        testModel = build_model(MODEL_DIR, local_only=_LOCAL_ONLY)
        _ = testModel(
            [
                tf.convert_to_tensor(x_san1[:1]),
                tf.convert_to_tensor(x_san2[:1]),
                tf.convert_to_tensor(x_san3[:1]),
            ],
            training=False,
        )
        _safe_load_weights(testModel, MODEL2_WEIGHTS, verbose=verbose)

    if verbose:
        print("Finish preparing second model for inference")

    bs = 256
    ds = (
        tf.data.Dataset.from_tensor_slices((x_san1, x_san2, x_san3))
        .batch(bs, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    n = x_san1.shape[0]
    out = np.empty((n, 3, 512), dtype=np.float32)
    offset = 0
    for b1, b2, b3 in ds:
        probs = _infer_probs(testModel, b1, b2, b3)
        bs_eff = int(probs.shape[0])
        out[offset : offset + bs_eff] = probs.numpy()
        offset += bs_eff
    san_res = out

    if verbose:
        print("Finish calculating, get result of size: ", san_res.shape)

    return san_res


def mergeSanInstanceResult(san_res, list_san_ins, debug_local=True):
    san_res = np.asarray(san_res)
    for i in range(len(list_san_ins)):
        ins_res = san_res[i]
        start = int(np.argmax(ins_res[0]))
        stop = int(np.argmax(ins_res[1]))
        target = int(np.argmax(ins_res[2]))

        start_score = float(ins_res[0][start])
        stop_score = float(ins_res[1][stop])
        target_score = float(ins_res[2][target])

        start_CLS = float(ins_res[0][0])
        stop_CLS = float(ins_res[1][0])

        list_san_ins[i]["start"] = start
        list_san_ins[i]["stop"] = stop
        list_san_ins[i]["target"] = AnswerTypeRev[int(target)]

        list_san_ins[i]["start_score"] = start_score
        list_san_ins[i]["stop_score"] = stop_score
        list_san_ins[i]["target_score"] = target_score

        list_san_ins[i]["start_CLS"] = start_CLS
        list_san_ins[i]["stop_CLS"] = stop_CLS

        if debug_local:
            question = list_san_ins[i]["question"]
            context = list_san_ins[i]["context"]
            context_split = context.split()

            if start > 0 and stop > start:
                print(
                    AnswerTypeRev[int(target)],
                    ": ",
                    question,
                    ": ",
                    " ".join(context_split[start : stop + 1]),
                )
                print(
                    start_score - start_CLS,
                    start_score - start_CLS + stop_score - stop_CLS,
                )
            if target > 0 and target != 3:
                print(AnswerTypeRev[int(target)], ": ", question, ": ", context)
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

    grouped = {}
    for r in san_ins_res_df.itertuples(index=False):
        grouped.setdefault(getattr(r, "example_id"), []).append(r)

    for row in doc_lan_df.itertuples(index=False):
        doc_id = getattr(row, "example_id")

        lan_start = getattr(row, "start_token")
        lan_stop = getattr(row, "stop_token")
        lan_target = getattr(row, "target")

        san_start = -1
        san_stop = -1
        final_target = lan_target

        san_list = grouped.get(doc_id, [])

        max_score = 0.00001
        for san in san_list:
            part_id = getattr(san, "part_id")
            part_start = part_id * SAN_STRIDE
            start_in_part = getattr(san, "start")
            stop_in_part = getattr(san, "stop")

            start_in_lan = start_in_part + part_start
            stop_in_lan = stop_in_part + part_start

            if start_in_part > 0 and stop_in_part > start_in_part:
                score = (getattr(san, "start_score") - getattr(san, "start_CLS")) + (
                    getattr(san, "stop_score") - getattr(san, "stop_CLS")
                )
                if score > max_score:
                    san_start = start_in_lan + lan_start
                    san_stop = stop_in_lan + lan_start
                    max_score = score
            final_target = (
                getattr(san, "target") if hasattr(san, "target") else final_target
            )

        if verbose:
            print(doc_id, " QUESTION: ", getattr(row, "question"))
            print(
                final_target,
                " ".join(getattr(row, "raw_document").split()[lan_start:lan_stop]),
            )
            print(
                "SHORT: ",
                " ".join(getattr(row, "raw_document").split()[san_start:san_stop]),
            )
            print("\n")
        san_lan_res_doc = {}
        san_lan_res_doc["example_id"] = doc_id
        san_lan_res_doc["question"] = getattr(row, "question")
        san_lan_res_doc["raw_document"] = getattr(row, "raw_document")
        san_lan_res_doc["lan_start"] = lan_start
        san_lan_res_doc["lan_stop"] = lan_stop
        san_lan_res_doc["san_start"] = san_start
        san_lan_res_doc["san_stop"] = san_stop
        san_lan_res_doc["target"] = final_target

        list_san_lan_res_doc.append(san_lan_res_doc)
    return list_san_lan_res_doc


def getFinalResult(f_test_path):
    list_all_ins, list_doc = parse_test_once(f_test_path)

    all_ins_res = getRawInstanceResults(list_all_ins)
    list_fine_res_all_ins = mergeInstanceResult(all_ins_res, list_all_ins)
    fine_res_all_ins_df = pd.DataFrame(list_fine_res_all_ins)

    doc_df = pd.DataFrame(list_doc)

    doc_res_lan = mergeDocumentResult(doc_df, fine_res_all_ins_df)
    doc_res_lan_df = pd.DataFrame(doc_res_lan)

    list_san_ins = getListShortSample(doc_res_lan_df)
    raw_san_ins_res = getSanInstanceResults(list_san_ins)
    list_san_res = mergeSanInstanceResult(
        raw_san_ins_res, list_san_ins, debug_local=False
    )
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

    sample_path = "/kaggle/input/sample_submission.csv"
    if not os.path.isfile(sample_path):
        sample_path = (
            "/kaggle/input/tensorflow2-question-answering/sample_submission.csv"
        )
    sample_df = pd.read_csv(sample_path)

    merged = sample_df[["example_id"]].merge(pred_df, on="example_id", how="left")
    merged["PredictionString"] = merged["PredictionString"].fillna("")
    merged.to_csv(
        "./submission.csv", index=False, columns=["example_id", "PredictionString"]
    )
    return merged




## === cell 1
sub_df = getSubmission()
print(sub_df.head())
print("Wrote ./submission.csv with shape:", sub_df.shape)
print("Exists:", os.path.exists("./submission.csv"))
print(
    "File size bytes:",
    os.path.getsize("./submission.csv") if os.path.exists("./submission.csv") else None,
)
print("MODEL_DIR used:", MODEL_DIR)
print(
    "MODEL1_WEIGHTS:",
    MODEL1_WEIGHTS,
    "exists:",
    (os.path.exists(MODEL1_WEIGHTS) if MODEL1_WEIGHTS else False),
)
print(
    "MODEL2_WEIGHTS:",
    MODEL2_WEIGHTS,
    "exists:",
    (os.path.exists(MODEL2_WEIGHTS) if MODEL2_WEIGHTS else False),
)
print("Tokenizer vocab size:", len(TOKENIZER))
