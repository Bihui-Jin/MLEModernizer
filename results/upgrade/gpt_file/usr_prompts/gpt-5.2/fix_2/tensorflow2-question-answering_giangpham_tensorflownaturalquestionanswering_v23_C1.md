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

3.10

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
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers"])



## === cell 1
import numpy as np
import pandas as pd
import sys
import random
from tqdm import tqdm
import re
import string
import shutil
import json

import tensorflow as tf
from transformers import AutoTokenizer, BertConfig, TFBertModel



## === cell 2
f_test = "../input/simplified-nq-test.jsonl"
f_train = "../input/simplified-nq-train.jsonl"

num_train_samples = 307372
num_test_samples = 346

if not os.path.exists(f_test):
    alt = "../input/tensorflow2-question-answering/simplified-nq-test.jsonl"
    if os.path.exists(alt):
        f_test = alt

if not os.path.exists(f_train):
    alt = "../input/tensorflow2-question-answering/simplified-nq-train.jsonl"
    if os.path.exists(alt):
        f_train = alt

print("Using f_test:", f_test)
print("Using f_train:", f_train)




## === cell 3
def get_id_df(filename=f_test):
    list_id = []
    with open(filename) as f:
        progress = tqdm(f)
        for sam_count, line in enumerate(progress):
            data = json.loads(line)
            example_id = str(data["example_id"])
            list_id.append({"example_id": example_id})
    list_id_df = pd.DataFrame(list_id)
    return list_id_df




## === cell 4
AnswerType = {
    "NO_ANSWER": 0,
    "YES": 1,
    "NO": 2,
    "SHORT": 3,
    "LONG": 4,
}

AnswerTypeRev = {
    0: "NO_ANSWER",
    1: "YES",
    2: "NO",
    3: "SHORT",
    4: "LONG",
}




## === cell 5
def preprocess_data(data, tokenizer, debug=False):
    progress = tqdm(data, total=len(data))
    x1, x2, x3, y = [], [], [], []
    for sam in progress:
        tokenized_sam = tokenizer.encode_plus(
            sam["question"],
            sam["context"],
            padding="max_length",
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )

        input_ids = tokenized_sam["input_ids"]
        attention_mask = tokenized_sam["attention_mask"]
        token_type_ids = tokenized_sam.get("token_type_ids", [0] * len(input_ids))

        x1.append(tf.cast(input_ids, tf.int32))
        x2.append(tf.cast(token_type_ids, tf.int32))
        x3.append(tf.cast(attention_mask, tf.int32))

        y.append([sam["start"], sam["stop"], AnswerType[sam["target"]]])

    x1 = tf.convert_to_tensor(x1)
    x2 = tf.convert_to_tensor(x2)
    x3 = tf.convert_to_tensor(x3)
    y = tf.convert_to_tensor(y)
    return x1, x2, x3, y




## === cell 6
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




## === cell 7
def mergeInstanceResult(test_res, list_test_ins):
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




## === cell 8
def mergeDocumentRes(ins_df, val_id_df, threshold=0.0001, stride=128, debug=False):
    list_doc_lan = []
    for idx, doc in val_id_df.iterrows():
        doc_id = doc["example_id"]
        ins_of_doc = ins_df.loc[ins_df["example_id"] == doc_id]

        start_ins = ins_of_doc.loc[ins_of_doc["start"] != 0]
        stop_ins = ins_of_doc.loc[ins_of_doc["stop"] != 0]
        all_non_zero = pd.concat([start_ins, stop_ins]).drop_duplicates()

        best_start = -1
        best_stop = -1
        best_target = 0
        best_score = threshold

        for idx_ins, ins in all_non_zero.iterrows():
            ins_start = int(ins["start"])
            ins_stop = int(ins["stop"])
            ins_target = int(ins["target"])

            part_start = int(ins["part_start"])

            real_start = int(ins_start + part_start)
            real_stop = int(ins_stop + part_start)

            s_start = float(ins["start_score"])
            s_stop = float(ins["stop_score"])

            cls_start = float(ins["start_CLS"])
            cls_stop = float(ins["stop_CLS"])

            if real_stop > real_start:
                score = (s_start - cls_start) + (s_stop - cls_stop)
                if score > best_score:
                    best_score = score
                    best_start = real_start
                    best_stop = real_stop
                    best_target = ins_target

        doc_lan = {
            "example_id": doc_id,
            "start": best_start,
            "stop": best_stop,
            "target": best_target,
            "score": best_score,
        }

        if debug and idx == 101:
            print(doc_lan)

        list_doc_lan.append(doc_lan)

    list_doc_lan_df = pd.DataFrame(list_doc_lan)
    return list_doc_lan_df




## === cell 9
cleanr = re.compile("<.*?>")


def clean_html(raw_html):
    cleantext = re.sub(cleanr, "<tag>", raw_html)
    return cleantext


def parseDataClean(
    filename=f_test,
    is_val=True,
    drop_noanswer_rate=0.95,
    drop_null_instances_rate=0.98,
    debug=False,
):
    INSTANCE_WORDS_LEN = 500
    STRIDE = 128
    list_instances = []

    with open(filename) as f:
        progress = tqdm(f)
        for sam_count, line in enumerate(progress):
            data = json.loads(line)
            example_id = str(data["example_id"])

            doc_text_raw = data["document_text"]
            doc_text_tag = clean_html(doc_text_raw)
            doc_tag_split = doc_text_tag.split()

            clean_doc = list(filter(("<tag>").__ne__, doc_tag_split))
            question = data["question_text"]

            len_ques = len(question.split())
            part_len = INSTANCE_WORDS_LEN - len_ques
            if part_len < 16:
                part_len = 16

            num_ins = (len(clean_doc) - part_len) // STRIDE + 1
            if num_ins < 0:
                num_ins = 0

            for part_id in range(num_ins + 1):
                part_start = part_id * STRIDE
                part_stop = min(len(clean_doc), part_id * STRIDE + part_len)

                part_split = clean_doc[part_start:part_stop]
                part = " ".join(part_split)

                instance = {
                    "example_id": example_id,
                    "part_start": part_start,
                    "part_stop": part_stop,
                    "question": question,
                    "context": part,
                    "start": 0,
                    "stop": 0,
                    "target": "NO_ANSWER",
                }
                list_instances.append(instance)
    return list_instances




## === cell 10
def getMapping(set_id, filename=f_test):
    list_cand_maps = []
    with open(filename) as f:
        progress = tqdm(f)
        for sam_count, line in enumerate(progress):
            data = json.loads(line)
            example_id = str(data["example_id"])

            if example_id in set_id:
                doc_text_raw = data["document_text"]
                doc_text_raw = clean_html(doc_text_raw)
                doc_text_split = doc_text_raw.split()

                clean_doc = list(filter(("<tag>").__ne__, doc_text_split))

                list_candidates = data["long_answer_candidates"]
                list_new_candidates = []
                for cand in list_candidates:
                    cand_start = cand["start_token"]
                    cand_stop = cand["end_token"]

                    num_tag_bef_start = doc_text_split[0:cand_start].count("<tag>")
                    num_tag_bef_stop = doc_text_split[0:cand_stop].count("<tag>")

                    new_start = cand_start - num_tag_bef_start
                    new_stop = cand_stop - num_tag_bef_stop

                    list_new_candidates.append(
                        {"end_token": new_stop, "start_token": new_start}
                    )

                list_cand_maps.append(
                    {
                        "example_id": str(example_id),
                        "new_candidates": list_new_candidates,
                        "old_candidates": list_candidates,
                    }
                )
    return list_cand_maps




## === cell 11
def _resolve_local_or_fallback(model_path, fallback="bert-base-uncased"):
    if model_path and os.path.isdir(model_path):
        return model_path, True
    return fallback, False


def build_model(model_name, debug=False):
    resolved_name, is_local = _resolve_local_or_fallback(model_name)
    encoder = TFBertModel.from_pretrained(resolved_name, local_files_only=is_local)

    tokenizer = AutoTokenizer.from_pretrained(resolved_name, local_files_only=is_local)

    tags = ["``", "''", "--"]
    special_tokens_dict = {"additional_special_tokens": tags}
    tokenizer.add_special_tokens(special_tokens_dict)

    encoder.resize_token_embeddings(len(tokenizer))

    NUM_TARGET = 5

    class MyQAModel(tf.keras.Model):
        def __init__(self, *inputs, **kwargs):
            super().__init__(*inputs, **kwargs)
            self.bert = encoder
            self.start_logits = tf.keras.layers.Dense(1)
            self.stop_logits = tf.keras.layers.Dense(1)
            self.target = tf.keras.layers.Dense(NUM_TARGET)

        def call(self, inputs, **kwargs):
            bert_res = self.bert(
                inputs[0],
                token_type_ids=inputs[1],
                attention_mask=inputs[2],
            )

            seq_out = bert_res[0]
            pooled_out = bert_res[1]

            start_logits = tf.squeeze(self.start_logits(seq_out), -1)
            stop_logits = tf.squeeze(self.stop_logits(seq_out), -1)

            targets = self.target(pooled_out)
            paddings = tf.constant([[0, 0], [0, 512 - NUM_TARGET]])
            targets = tf.pad(targets, paddings)

            res = tf.stack([start_logits, stop_logits, targets], axis=1)
            return res

    model = MyQAModel()
    return model




## === cell 12
def getRawInstanceResults(list_test, verbose=True, debug=False):
    if verbose:
        print("Getting raw result for all the instances generated from test file")

    model_name = "../input/tensorflow-question-answer-fine-data"
    resolved_name, is_local = _resolve_local_or_fallback(model_name)

    tokenizer = AutoTokenizer.from_pretrained(resolved_name, local_files_only=is_local)

    tags = ["``", "''", "--"]
    special_tokens_dict = {"additional_special_tokens": tags}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    if verbose:
        print("Added toks:", num_added_toks, "Tokenizer size:", len(tokenizer))

    x_test1, x_test2, x_test3, y_test = preprocess_data(list_test, tokenizer)
    if verbose:
        print("Finish tokenizing", len(list_test), "data for the first model")
        print(x_test1.shape)

    if verbose:
        print("Preparing model")

    strategy = get_strategy()
    with strategy.scope():
        testModel = build_model(resolved_name)
        x = np.ones([1, 512], dtype=int)
        testModel.predict([x, x, x], verbose=0)

        weights_path = "../input/model1/weights-01.h5"
        if os.path.exists(weights_path):
            testModel.load_weights(weights_path)
        else:
            print("WARNING: weights not found:", weights_path)

        optAdam = tf.keras.optimizers.Adam(learning_rate=0.00005)
        lossSCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metricSCA = tf.keras.metrics.SparseCategoricalAccuracy()
        testModel.compile(optimizer=optAdam, loss=lossSCE, metrics=[metricSCA])

    if verbose:
        print("Finish preparing model/weights for the model")

    test_res = testModel.predict([x_test1, x_test2, x_test3], verbose=1)
    if verbose:
        print("Finish calculating raw result, get an array of size:", test_res.shape)
    return test_res




## === cell 13
def getSubmissionLan(doc_res_df, doc_cand_df, threshold=0.0001, debug=False):
    doc_res_df.example_id = doc_res_df.example_id.astype(str)
    doc_cand_df.example_id = doc_cand_df.example_id.astype(str)

    combine_df = pd.merge(doc_res_df, doc_cand_df, on="example_id", how="left")

    lines = []
    for rid, doc in combine_df.iterrows():
        example_id = doc["example_id"]
        long_id = str(example_id) + "_long"

        line_long = {"example_id": long_id}

        an_start = int(doc["start"])
        an_stop = int(doc["stop"])
        an_target = int(doc["target"])
        lan_start, lan_stop = -1, -1

        if (
            an_start > 0
            and an_stop > 0
            and isinstance(doc.get("new_candidates", None), list)
        ):
            candidates = doc["new_candidates"]
            an_range = range(an_start, an_stop + 1)

            best_inter = 0.5
            shortest = 10**18
            best_id = 0
            for cidx, cand in enumerate(candidates):
                c_start = int(cand["start_token"])
                c_stop = int(cand["end_token"])
                c_range = range(c_start, c_stop + 1)

                inter = len(set(an_range).intersection(c_range))
                if float(inter) > best_inter:
                    best_id = cidx
                    best_inter = inter
                    shortest = len(c_range)
                elif inter == best_inter:
                    if shortest > len(c_range):
                        best_id = cidx
                        shortest = len(c_range)

            real_candidates = doc.get("old_candidates", None)
            if isinstance(real_candidates, list) and len(real_candidates) > best_id:
                lan_start = real_candidates[best_id]["start_token"]
                lan_stop = real_candidates[best_id]["end_token"]

        if lan_start > 0 and lan_stop > 0 and an_target != 0:
            long_string = str(lan_start) + ":" + str(lan_stop)
        else:
            long_string = ""

        line_long["PredictionString"] = long_string
        lines.append(line_long)

    lines_df = pd.DataFrame(lines)
    return lines_df.sort_values("example_id")




## === cell 14
def getSanCandidate(sub, filename=f_test, debug=False):
    INSTANCE_WORDS_LEN = 500
    STRIDE = 256

    list_doc_lan_res = []
    for rowid, row in sub.iterrows():
        example_id = str(row["example_id"]).replace("_long", "")
        lan_start, lan_stop = -1, -1

        if str(row["PredictionString"]) != "":
            tokens = str(row["PredictionString"]).split(":")
            lan_start = int(tokens[0])
            lan_stop = int(tokens[1])

        list_doc_lan_res.append(
            {"example_id": example_id, "lan_start": lan_start, "lan_stop": lan_stop}
        )

    list_doc_lan_res_df = pd.DataFrame(list_doc_lan_res)
    set_id = set(list_doc_lan_res_df["example_id"].values.tolist())

    list_san_ins = []
    with open(filename) as f:
        progress = tqdm(f)
        for sam_count, line in enumerate(progress):
            data = json.loads(line)
            example_id = str(data["example_id"])
            if example_id in set_id:
                ans = list_doc_lan_res_df.loc[
                    list_doc_lan_res_df["example_id"] == example_id
                ]
                lan_start, lan_stop = -1, -1
                for _, r in ans.iterrows():
                    lan_start = int(r["lan_start"])
                    lan_stop = int(r["lan_stop"])

                doc_text = data["document_text"]
                doc_text_split = doc_text.split()
                question = data["question_text"]

                if lan_start > -1 and lan_stop > -1:
                    if lan_stop - lan_start <= INSTANCE_WORDS_LEN:
                        offset = (INSTANCE_WORDS_LEN - (lan_stop - lan_start)) // 2
                        part_start = max(0, lan_start - offset)
                        part_stop = min(lan_stop + offset, len(doc_text_split))
                        context = " ".join(doc_text_split[part_start:part_stop])
                        list_san_ins.append(
                            {
                                "example_id": example_id,
                                "part_start": part_start,
                                "part_stop": part_stop,
                                "question": question,
                                "context": context,
                                "start": 0,
                                "stop": 0,
                                "target": "NO_ANSWER",
                            }
                        )
                    else:
                        part_length = INSTANCE_WORDS_LEN
                        num_parts = (
                            lan_stop - lan_start - INSTANCE_WORDS_LEN
                        ) // STRIDE + 1
                        for part_id in range(num_parts + 1):
                            part_start = lan_start + part_id * STRIDE
                            part_stop = min(
                                len(doc_text_split),
                                lan_start + part_id * STRIDE + part_length,
                            )
                            context = " ".join(doc_text_split[part_start:part_stop])
                            list_san_ins.append(
                                {
                                    "example_id": example_id,
                                    "part_start": part_start,
                                    "part_stop": part_stop,
                                    "question": question,
                                    "context": context,
                                    "start": 0,
                                    "stop": 0,
                                    "target": "NO_ANSWER",
                                }
                            )
    return list_san_ins




## === cell 15
def create_model_san(tokenizer_san, model_name_san, debug=False):
    resolved_name, is_local = _resolve_local_or_fallback(model_name_san)
    encoder = TFBertModel.from_pretrained(resolved_name, local_files_only=is_local)
    encoder.resize_token_embeddings(len(tokenizer_san))

    NUM_TARGET = 5

    class MyQAModel(tf.keras.Model):
        def __init__(self, *inputs, **kwargs):
            super().__init__(*inputs, **kwargs)
            self.bert = encoder
            self.start_logits = tf.keras.layers.Dense(1)
            self.stop_logits = tf.keras.layers.Dense(1)
            self.target = tf.keras.layers.Dense(NUM_TARGET)

        def call(self, inputs, **kwargs):
            bert_res = self.bert(
                inputs[0],
                token_type_ids=inputs[1],
                attention_mask=inputs[2],
            )
            seq_out = bert_res[0]
            pooled_out = bert_res[1]

            start_logits = tf.squeeze(self.start_logits(seq_out), -1)
            stop_logits = tf.squeeze(self.stop_logits(seq_out), -1)

            targets = self.target(pooled_out)
            paddings = tf.constant([[0, 0], [0, 512 - NUM_TARGET]])
            targets = tf.pad(targets, paddings)

            res = tf.stack([start_logits, stop_logits, targets], axis=1)
            return res

    model = MyQAModel()
    return model




## === cell 16
def getSanRawRes(list_san_ins, verbose=1):
    print(
        "Getting raw result for short answer instance generated from found long answers"
    )

    model_name_san = "../input/tensorflow-question-answer-fine-data"
    resolved_name, is_local = _resolve_local_or_fallback(model_name_san)

    tokenizer_san = AutoTokenizer.from_pretrained(
        resolved_name, local_files_only=is_local
    )

    tags_san = [
        "<Dd>",
        "<Dl>",
        "<Dt>",
        "<H1>",
        "<H2>",
        "<H3>",
        "<Li>",
        "<Ol>",
        "<P>",
        "<Table>",
        "<Td>",
        "<Th>",
        "<Tr>",
        "<Ul>",
        "</Dd>",
        "</Dl>",
        "</Dt>",
        "</H1>",
        "</H2>",
        "</H3>",
        "</Li>",
        "</Ol>",
        "</P>",
        "</Table>",
        "</Td>",
        "</Th>",
        "</Tr>",
        "</Ul>",
        "<Th_colspan=",
        "</Th_colspan=",
        "``",
        "''",
        "--",
    ]
    special_tokens_dict_san = {"additional_special_tokens": tags_san}
    tokenizer_san.add_special_tokens(special_tokens_dict_san)
    print("Short answer vocab size:", len(tokenizer_san))

    x_san1, x_san2, x_san3, y_san = preprocess_data(list_san_ins, tokenizer_san)
    print(
        "Finish tokenizing", len(list_san_ins), "instances for short answer candidates"
    )
    print(x_san1.shape)

    strategy_san = get_strategy()
    with strategy_san.scope():
        sanModel = create_model_san(tokenizer_san, resolved_name)
        x = np.ones([1, 512], dtype=int)
        sanModel.predict([x, x, x], verbose=0)

        weights_path = "../input/model1/weights-14.h5"
        if os.path.exists(weights_path):
            sanModel.load_weights(weights_path)
        else:
            print("WARNING: weights not found:", weights_path)

        optAdam = tf.keras.optimizers.Adam(learning_rate=0.00005)
        lossSCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metricSCA = tf.keras.metrics.SparseCategoricalAccuracy()
        sanModel.compile(optimizer=optAdam, loss=lossSCE, metrics=[metricSCA])

    if verbose:
        print("Finish loading pretrained weights for the model for short answer")

    test_res = sanModel.predict([x_san1, x_san2, x_san3], verbose=1)
    if verbose:
        print("Finish calculating raw result, get an array of size:", test_res.shape)
    return test_res




## === cell 17
def getSanSubmission(doc_res_df, threshold=0.0001, debug=False):
    doc_res_df.example_id = doc_res_df.example_id.astype(str)
    lines = []
    for rid, doc in doc_res_df.iterrows():
        example_id = doc["example_id"]
        short_id = str(example_id) + "_short"

        an_start = int(doc["start"])
        an_stop = int(doc["stop"])
        an_target = int(doc["target"])

        if an_start > 0 and an_stop > 0 and an_target != 4:
            short_string = str(an_start) + ":" + str(an_stop)
        else:
            short_string = ""

        if an_target == 1 or an_target == 2:
            short_string = AnswerTypeRev[an_target]

        lines.append({"example_id": short_id, "PredictionString": short_string})

    lines_df = pd.DataFrame(lines)
    return lines_df.sort_values("example_id")




## === cell 18
list_id_df = get_id_df()
set_id = set(list_id_df["example_id"].values.tolist())

lan_map = getMapping(set_id)
list_mappings_df = pd.DataFrame(lan_map)

print("Docs in test:", len(list_id_df), "Mappings:", len(list_mappings_df))



## === cell 19
list_all_ins = parseDataClean(f_test)
all_ins_res = getRawInstanceResults(list_all_ins)

list_fine_res_all_ins = mergeInstanceResult(all_ins_res, list_all_ins)
fine_res_all_ins_df = pd.DataFrame(list_fine_res_all_ins)

docAnsDf = mergeDocumentRes(fine_res_all_ins_df, list_id_df)
subLan = getSubmissionLan(docAnsDf, list_mappings_df)

print("Long submission rows:", len(subLan))
subLan.head()



## === cell 20
list_san_ins = getSanCandidate(subLan, debug=False)
print("Short instances:", len(list_san_ins))

if len(list_san_ins) == 0:
    subSan = pd.DataFrame(
        {
            "example_id": [
                str(eid) + "_short"
                for eid in list_id_df["example_id"].astype(str).tolist()
            ],
            "PredictionString": [""] * len(list_id_df),
        }
    ).sort_values("example_id")
else:
    sanRawRes = getSanRawRes(list_san_ins)
    list_fine_res_san_ins = mergeInstanceResult(sanRawRes, list_san_ins)
    fine_res_san_ins_df = pd.DataFrame(list_fine_res_san_ins)

    docSanAnsDf = mergeDocumentRes(fine_res_san_ins_df, list_id_df)
    subSan = getSanSubmission(docSanAnsDf, threshold=0.2)

print("Short submission rows:", len(subSan))
subSan.head()



## === cell 21
sub = pd.concat([subLan, subSan], ignore_index=True)
sub_sorted = sub.sort_values("example_id")

sub_sorted["example_id"] = sub_sorted["example_id"].astype(str)
sub_sorted["PredictionString"] = sub_sorted["PredictionString"].fillna("").astype(str)

expected = 2 * len(list_id_df)
print("Combined rows:", len(sub_sorted), "Expected:", expected)

sub_sorted.to_csv(
    "./submission.csv", index=False, columns=["example_id", "PredictionString"]
)
print("Wrote ./submission.csv")
print(sub_sorted.head(10))
