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

# 5. Target score

0.4909733124018838

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
pass



## === cell 1
import numpy as np
import pandas as pd
import sys
import random
from tqdm import tqdm
import re
import string
import os
import shutil
import json
from transformers import (
    AutoTokenizer,
    TFBertMainLayer,
    TFBertForPreTraining,
    BertConfig,
    TFBertModel,
)
import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy as sce



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
f_test = "../input/tensorflow2-question-answering/simplified-nq-test.jsonl"
f_train = "../input/tensorflow2-question-answering/simplified-nq-train.jsonl"
num_train_samples = 307372
num_test_samples = 346




## === cell 3
def get_id_df(filename=f_test):
    list_id = []
    with open(filename) as f:
        progress = tqdm(f, disable=True)
        for sam_count, line in enumerate(progress):
            data = json.loads(line)
            example_id = str(data["example_id"])
            doc = {"example_id": example_id}
            list_id.append(doc)
    list_id_df = pd.DataFrame(list_id)
    return list_id_df




## === cell 4
AnswerType = {"NO_ANSWER": 0, "YES": 1, "NO": 2, "SHORT": 3, "LONG": 4}

AnswerTypeRev = {0: "NO_ANSWER", 1: "YES", 2: "NO", 3: "SHORT", 4: "LONG"}




## === cell 5
def preprocess_data(data, tokenizer, debug=False):
    progress = tqdm(data, total=len(data), disable=True)
    x1 = []
    x2 = []
    x3 = []
    y = []
    for sam in progress:
        tokenized_sam = tokenizer.encode_plus(
            sam["question"],
            sam["context"],
            padding="max_length",
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )

        x1.append(tf.cast(tokenized_sam["input_ids"], tf.int32))
        x2.append(tf.cast(tokenized_sam["token_type_ids"], tf.int32))
        x3.append(tf.cast(tokenized_sam["attention_mask"], tf.int32))

        y.append([sam["start"], sam["stop"], AnswerType[sam["target"]]])

    x1 = tf.convert_to_tensor(x1)
    x2 = tf.convert_to_tensor(x2)
    x3 = tf.convert_to_tensor(x3)

    y = tf.convert_to_tensor(y)
    return x1, x2, x3, y




## === cell 6
def get_strategy():
    try:
        tpu_cluster_resolver = (
            tf.distribute.cluster_resolver.TPUClusterResolver()
        )  # TPU detection
        print(
            "Running on TPU ", tpu_cluster_resolver.cluster_spec().as_dict()["worker"]
        )
        tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
        strategy = tf.distribute.experimental.TPUStrategy(tpu_cluster_resolver)
    except ValueError as e:
        print(e)
        print("No TPU detected")
        strategy = tf.distribute.get_strategy()
    return strategy




## === cell 7
def mergeInstanceResult(test_res, list_test_ins):
    for i in range(len(list_test_ins)):
        ins_res = test_res[i]
        start = np.argmax(ins_res[0])
        stop = np.argmax(ins_res[1])
        target = np.argmax(ins_res[2])

        start_score = ins_res[0][start]
        stop_score = ins_res[1][stop]
        target_score = ins_res[2][target]

        start_CLS = ins_res[0][0]
        stop_CLS = ins_res[1][0]

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
    STRIDE = stride
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

            part_start = ins["part_start"]

            real_start = int(ins_start + part_start)
            real_stop = int(ins_stop + part_start)

            s_start = ins["start_score"]
            s_stop = ins["stop_score"]

            cls_start = ins["start_CLS"]
            cls_stop = ins["stop_CLS"]

            if real_stop > real_start:
                if s_start - cls_start + s_stop - cls_stop > best_score:
                    best_score = s_start - cls_start + s_stop - cls_stop
                    best_start = real_start
                    best_stop = real_stop
                    best_target = ins_target

        doc_lan = {}
        doc_lan["example_id"] = doc_id
        doc_lan["start"] = best_start
        doc_lan["stop"] = best_stop
        doc_lan["target"] = best_target
        doc_lan["score"] = best_score

        if debug:
            if idx == 101:
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
    num, count_drop, count_yes_no, count_long, count_short, count_no_answer = (
        0,
        0,
        0,
        0,
        0,
        0,
    )
    list_instances = []

    with open(filename) as f:
        progress = tqdm(f, disable=True)
        for sam_count, line in enumerate(progress):
            data = json.loads(line)
            example_id = str(data["example_id"])

            doc_text_raw = data["document_text"]
            doc_text_tag = clean_html(
                doc_text_raw
            )  # change all html tags to the form <tag>
            doc_tag_split = doc_text_tag.split()

            lan_start, lan_stop, san_start, san_stop = -1, -1, -1, -1

            clean_doc = list(filter(("<tag>").__ne__, doc_tag_split))

            question = data["question_text"]  # question

            len_ques = len(question.split())
            part_len = INSTANCE_WORDS_LEN - len_ques

            num_ins = (len(clean_doc) - part_len) // STRIDE + 1

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
        progress = tqdm(f, disable=True)
        for sam_count, line in enumerate(progress):

            data = json.loads(line)
            example_id = str(data["example_id"])

            if example_id in set_id:
                doc_text_raw = data["document_text"]
                doc_text_raw = clean_html(
                    doc_text_raw
                )  # change all html tags to the form <tag>
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

                    new_cand = {}
                    new_cand["end_token"] = new_stop
                    new_cand["start_token"] = new_start

                    list_new_candidates.append(new_cand)
                sample = {}
                sample["example_id"] = str(example_id)
                sample["new_candidates"] = list_new_candidates
                sample["old_candidates"] = list_candidates

                list_cand_maps.append(sample)
    return list_cand_maps




## === cell 11
def build_model(model_name, debug=False):
    encoder = TFBertModel.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tags = ["``", "''", "--"]

    special_tokens_dict = {"additional_special_tokens": tags}

    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

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
                inputs[0], token_type_ids=inputs[1], attention_mask=inputs[2]
            )
            dropout_res1 = bert_res[0]

            start_logits = tf.squeeze(self.start_logits(dropout_res1), -1)
            dropout_res2 = bert_res[0]

            stop_logits = tf.squeeze(self.stop_logits(dropout_res2), -1)
            dropout_res3 = bert_res[1]

            targets = self.target(dropout_res3)

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
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tags = ["``", "''", "--"]

    special_tokens_dict = {"additional_special_tokens": tags}

    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(num_added_toks)
    print(len(tokenizer))

    x_test1, x_test2, x_test3, y_test = preprocess_data(list_test, tokenizer)
    if verbose:
        print("Finish tokenizing ", len(list_test), " data for the first model")
        print(x_test1.shape)

    if verbose:
        print("Preparing model")

    strategy = get_strategy()
    with strategy.scope():
        testModel = build_model(model_name)
        x = np.ones([1, 512], dtype=int)
        testModel.predict([x, x, x])
        testModel.load_weights("../input/model1/weights-02.h5")
        optAdam = tf.keras.optimizers.Adam(learning_rate=0.00005)
        lossSCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metricSCA = tf.keras.metrics.SparseCategoricalAccuracy()
        testModel.compile(optimizer=optAdam, loss=lossSCE, metrics=[metricSCA])

    if verbose:
        print("Finish loading pretrained weights for the model")

    test_res = testModel.predict([x_test1, x_test2, x_test3], verbose=1)

    if verbose:
        print("Finish calculating raw result, get an array of size: ", test_res.shape)
    return test_res




## === cell 13
def getSubmissionLan(doc_res_df, doc_cand_df, threshold=0.0001, debug=False):
    doc_res_df.example_id = doc_res_df.example_id.astype(str)
    doc_cand_df.example_id = doc_cand_df.example_id.astype(str)
    if debug:
        print(doc_res_df.dtypes)
        print(doc_cand_df.dtypes)

    combine_df = pd.merge(doc_res_df, doc_cand_df, on="example_id")
    lines = []
    for id, doc in combine_df.iterrows():

        example_id = doc["example_id"]
        long_id = str(example_id) + "_long"
        short_id = str(example_id) + "_short"

        line_long = {}
        line_long["example_id"] = long_id

        an_start = int(doc["start"])
        an_stop = int(doc["stop"])
        an_target = doc["target"]
        an_score = doc["score"]
        lan_start, lan_stop = -1, -1

        if an_start > 0 and an_stop > 0:
            candidates = doc["new_candidates"]
            an_range = [*range(an_start, an_stop + 1, 1)]

            best_inter = 0.5
            shortest = 10000000000000
            best_id = 0
            for cidx, cand in enumerate(candidates):
                c_start = int(cand["start_token"])
                c_stop = int(cand["end_token"])

                c_range = [*range(c_start, c_stop + 1, 1)]
                inter = len(list(set(an_range) & set(c_range)))

                if float(inter) > best_inter:
                    best_id = cidx
                    best_inter = inter
                    shortest = len(c_range)
                elif inter == best_inter:
                    if shortest > len(c_range):
                        best_id = cidx
                        shortest = len(c_range)

            real_candidates = doc["old_candidates"]
            lan_start = real_candidates[best_id]["start_token"]
            lan_stop = real_candidates[best_id]["end_token"]

            if debug:
                if id == 101:
                    print(lan_start, lan_stop)

        if lan_start > 0 and lan_stop > 0 and an_target != 0:
            long_string = str(lan_start) + ":" + str(lan_stop)
        else:
            long_string = ""

        line_long["PredictionString"] = long_string
        lines.append(line_long)

    lines_df = pd.DataFrame(lines)
    sorted_df = lines_df.sort_values("example_id")
    return sorted_df




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

        sam = {"example_id": example_id, "lan_start": lan_start, "lan_stop": lan_stop}
        list_doc_lan_res.append(sam)

    list_doc_lan_res_df = pd.DataFrame(list_doc_lan_res)

    set_id = set(list_doc_lan_res_df["example_id"].values.tolist())

    list_san_ins = []

    with open(filename) as f:
        progress = tqdm(f, disable=True)
        for sam_count, line in enumerate(progress):
            data = json.loads(line)
            example_id = str(data["example_id"])
            if example_id in set_id:
                ans = list_doc_lan_res_df.loc[
                    list_doc_lan_res_df["example_id"] == example_id
                ]
                lan_start, lan_stop = -1, -1
                for rowid, row in ans.iterrows():
                    lan_start = row["lan_start"]
                    lan_stop = row["lan_stop"]
                if debug:
                    print(example_id, lan_start, lan_stop)
                doc_text = data["document_text"]
                doc_text_split = doc_text.split()
                question = data["question_text"]

                if lan_start > -1 and lan_stop > -1:
                    if lan_stop - lan_start <= INSTANCE_WORDS_LEN:
                        offset = (INSTANCE_WORDS_LEN - (lan_stop - lan_start)) // 2
                        part_start = max(0, lan_start - offset)
                        part_stop = min(lan_stop + offset, len(doc_text_split))
                        part_split = doc_text_split[part_start:part_stop]
                        context = " ".join(part_split)
                        ins = {
                            "example_id": example_id,
                            "part_start": part_start,
                            "part_stop": part_stop,
                            "question": question,
                            "context": context,
                            "start": 0,
                            "stop": 0,
                            "target": "NO_ANSWER",
                        }
                        list_san_ins.append(ins)
                        if debug:
                            print(ins)
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
                            part_split = doc_text_split[part_start:part_stop]

                            context = " ".join(part_split)
                            ins = {
                                "example_id": example_id,
                                "part_start": part_start,
                                "part_stop": part_stop,
                                "question": question,
                                "context": context,
                                "start": 0,
                                "stop": 0,
                                "target": "NO_ANSWER",
                            }
                            list_san_ins.append(ins)
                            if debug:
                                print(ins)
    return list_san_ins




## === cell 15
def create_model_san(tokenizer_san, model_name_san, debug=False):
    config = BertConfig()
    if debug:
        print(config)
    encoder = TFBertModel.from_pretrained(model_name_san)
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
                inputs[0], token_type_ids=inputs[1], attention_mask=inputs[2]
            )

            dropout_res1 = bert_res[0]

            start_logits = tf.squeeze(self.start_logits(dropout_res1), -1)

            dropout_res2 = bert_res[0]

            stop_logits = tf.squeeze(self.stop_logits(dropout_res2), -1)

            dropout_res3 = bert_res[1]

            targets = self.target(dropout_res3)

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

    tokenizer_san = AutoTokenizer.from_pretrained(model_name_san)

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

    num_added_toks_san = tokenizer_san.add_special_tokens(special_tokens_dict_san)
    print("Short answer vocab size: ", len(tokenizer_san))

    x_san1, x_san2, x_san3, y_san = preprocess_data(list_san_ins, tokenizer_san)
    print(
        "Finish tokenizing ",
        len(list_san_ins),
        " instances for short answer candidates",
    )
    print(x_san1.shape)

    strategy_san = get_strategy()
    with strategy_san.scope():
        sanModel = create_model_san(tokenizer_san, model_name_san)
        x = np.ones([1, 512], dtype=int)
        sanModel.predict([x, x, x])
        sanModel.load_weights("../input/model1/weights-14.h5")
        optAdam = tf.keras.optimizers.Adam(learning_rate=0.00005)
        lossSCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metricSCA = tf.keras.metrics.SparseCategoricalAccuracy()
        sanModel.compile(optimizer=optAdam, loss=lossSCE, metrics=[metricSCA])

    if verbose:
        print("Finish loading pretrained weights for the model for short answer")

    test_res = sanModel.predict([x_san1, x_san2, x_san3], verbose=1)

    if verbose:
        print("Finish calculating raw result, get an array of size: ", test_res.shape)
    return test_res




## === cell 17
def getSanSubmission(doc_res_df, threshold=0.0001, debug=False):
    doc_res_df.example_id = doc_res_df.example_id.astype(str)
    lines = []
    for id, doc in doc_res_df.iterrows():
        example_id = doc["example_id"]
        short_id = str(example_id) + "_short"

        line_short = {}
        line_short["example_id"] = short_id

        an_start = int(doc["start"])
        an_stop = int(doc["stop"])
        an_target = int(doc["target"])
        an_score = float(doc["score"])

        if an_start > 0 and an_stop > 0 and an_target != 4 and an_stop - an_start < 30:
            short_string = str(an_start) + ":" + str(an_stop)
        else:
            short_string = ""

        if an_target == 1 or an_target == 2:
            short_string = AnswerTypeRev[an_target]

        line_short["PredictionString"] = short_string
        lines.append(line_short)

    lines_df = pd.DataFrame(lines)
    sorted_df = lines_df.sort_values("example_id")
    return sorted_df




## === cell 18
def refineLan(sub, list_mapping_df, debug=False):
    newsub = []
    for rowid, row in sub.iterrows():
        if "long" in str(row["example_id"]):
            example_id = str(row["example_id"]).replace("_long", "")

            longid = str(row["example_id"])
            longStr = str(row["PredictionString"])

            lan_start, lan_stop = -1, -1

            if str(row["PredictionString"]) != "":
                tokens = str(row["PredictionString"]).split(":")
                lan_start = int(tokens[0])
                lan_stop = int(tokens[1])

            san_start, san_stop = -1, -1

            sanid = str(example_id) + "_short"
            san = sub.loc[sub["example_id"] == sanid].iloc[0]
            sanStr = str(san["PredictionString"])

            if sanStr != "" and sanStr != "YES" and sanStr != "NO":
                tokensans = sanStr.split(":")
                san_start = int(tokensans[0])
                san_stop = int(tokensans[1])

                if san_start < lan_start or san_stop > lan_stop:  # san is not in lan
                    cands = list_mapping_df.loc[
                        list_mapping_df["example_id"] == example_id
                    ].iloc[0]["old_candidates"]

                    an_range = [*range(san_start, san_stop + 1, 1)]
                    best_inter = 0.5
                    shortest = 10000000000000
                    best_id = 0
                    for cidx, cand in enumerate(cands):
                        c_start = int(cand["start_token"])
                        c_stop = int(cand["end_token"])

                        c_range = [*range(c_start, c_stop + 1, 1)]
                        inter = len(list(set(an_range) & set(c_range)))

                        if float(inter) > best_inter:
                            best_id = cidx
                            best_inter = inter
                            shortest = len(c_range)
                        elif inter == best_inter:
                            if shortest > len(c_range):
                                best_id = cidx
                                shortest = len(c_range)

                    lan_start = cands[best_id]["start_token"]
                    lan_stop = cands[best_id]["end_token"]
                    longStr = str(lan_start) + ":" + str(lan_stop)

            longline = {"example_id": longid, "PredictionString": longStr}
            shortline = {"example_id": sanid, "PredictionString": sanStr}
            newsub.append(longline)
            newsub.append(shortline)
    newsubdf = pd.DataFrame(newsub)
    newsubsorted = newsubdf.sort_values("example_id")
    return newsubsorted




## === cell 19
list_id_df = get_id_df()



## === cell 20
set_id = set(list_id_df["example_id"].values.tolist())
lan_map = getMapping(set_id)



## === cell 21
list_mappings_df = pd.DataFrame(lan_map)
list_mappings_df.head()



## === cell 22
list_all_ins = parseDataClean(f_test)
all_ins_res = getRawInstanceResults(list_all_ins)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/tensorflow-question-answer-fine-data'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_11/443213008.py in <cell line: 0>()
      1 list_all_ins = parseDataClean(f_test)
----> 2 all_ins_res = getRawInstanceResults(list_all_ins)
      3 

/tmp/ipykernel_11/662773816.py in getRawInstanceResults(list_test, verbose, debug)
      4 
      5     model_name = "../input/tensorflow-question-answer-fine-data"
----> 6     tokenizer = AutoTokenizer.from_pretrained(model_name)
      7 
      8     tags = ["``", "''", "--"]

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
    981 
    982         # Next, let's try to use the tokenizer_config file to get the tokenizer class.
--> 983         tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
    984         if "_commit_hash" in tokenizer_config:
    985             kwargs["_commit_hash"] = tokenizer_config["_commit_hash"]

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in get_tokenizer_config(pretrained_model_name_or_path, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, **kwargs)
    813 
    814     commit_hash = kwargs.get("_commit_hash", None)
--> 815     resolved_config_file = cached_file(
    816         pretrained_model_name_or_path,
    817         TOKENIZER_CONFIG_FILE,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    520 
    521         # Now we try to recover if we can find all files correctly in the cache
--> 522         resolved_files = [
    523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in <listcomp>(.0)
    521         # Now we try to recover if we can find all files correctly in the cache
    522         resolved_files = [
--> 523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]
    525         if all(file is not None for file in resolved_files):

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
    138 ):
    139     # We try to see if we have a cached version (not up to date):
--> 140     resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
    141     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
    142         return resolved_file

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    104         ):
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 
    108             elif arg_name == "token" and arg_value is not None:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    152 
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"
    156             f" '{repo_id}'. Use `repo_type` argument if needed."

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/tensorflow-question-answer-fine-data'. Use `repo_type` argument if needed.

## === cell 23
list_fine_res_all_ins = mergeInstanceResult(all_ins_res, list_all_ins)
fine_res_all_ins_df = pd.DataFrame(list_fine_res_all_ins)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3458661183.py in <cell line: 0>()
----> 1 list_fine_res_all_ins = mergeInstanceResult(all_ins_res, list_all_ins)
      2 fine_res_all_ins_df = pd.DataFrame(list_fine_res_all_ins)
      3 

NameError: name 'all_ins_res' is not defined

## === cell 24
fine_res_all_ins_df.head()



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/17008897.py in <cell line: 0>()
----> 1 fine_res_all_ins_df.head()
      2 

NameError: name 'fine_res_all_ins_df' is not defined

## === cell 25
docAnsDf = mergeDocumentRes(fine_res_all_ins_df, list_id_df)



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3432278440.py in <cell line: 0>()
----> 1 docAnsDf = mergeDocumentRes(fine_res_all_ins_df, list_id_df)
      2 

NameError: name 'fine_res_all_ins_df' is not defined

## === cell 26
docAnsDf.head()



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3863313500.py in <cell line: 0>()
----> 1 docAnsDf.head()
      2 

NameError: name 'docAnsDf' is not defined

## === cell 27
subLan = getSubmissionLan(docAnsDf, list_mappings_df)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4209187204.py in <cell line: 0>()
----> 1 subLan = getSubmissionLan(docAnsDf, list_mappings_df)
      2 

NameError: name 'docAnsDf' is not defined

## === cell 28
subLan.head(20)



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2908301988.py in <cell line: 0>()
----> 1 subLan.head(20)
      2 

NameError: name 'subLan' is not defined

## === cell 29
list_san_ins = getSanCandidate(subLan, debug=False)



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2902297250.py in <cell line: 0>()
----> 1 list_san_ins = getSanCandidate(subLan, debug=False)
      2 

NameError: name 'subLan' is not defined

## === cell 30
sanRawRes = getSanRawRes(list_san_ins)



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/294003646.py in <cell line: 0>()
----> 1 sanRawRes = getSanRawRes(list_san_ins)
      2 

NameError: name 'list_san_ins' is not defined

## === cell 31
list_fine_res_san_ins = mergeInstanceResult(sanRawRes, list_san_ins)
fine_res_san_ins_df = pd.DataFrame(list_fine_res_san_ins)



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4163522254.py in <cell line: 0>()
----> 1 list_fine_res_san_ins = mergeInstanceResult(sanRawRes, list_san_ins)
      2 fine_res_san_ins_df = pd.DataFrame(list_fine_res_san_ins)
      3 

NameError: name 'sanRawRes' is not defined

## === cell 32
fine_res_san_ins_df.head()



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2274669907.py in <cell line: 0>()
----> 1 fine_res_san_ins_df.head()
      2 

NameError: name 'fine_res_san_ins_df' is not defined

## === cell 33
docSanAnsDf = mergeDocumentRes(fine_res_san_ins_df, list_id_df)



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3618507780.py in <cell line: 0>()
----> 1 docSanAnsDf = mergeDocumentRes(fine_res_san_ins_df, list_id_df)
      2 

NameError: name 'fine_res_san_ins_df' is not defined

## === cell 34
docSanAnsDf.head()



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2199362185.py in <cell line: 0>()
----> 1 docSanAnsDf.head()
      2 

NameError: name 'docSanAnsDf' is not defined

## === cell 35
subSan = getSanSubmission(docSanAnsDf, threshold=0.2)



## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3479299088.py in <cell line: 0>()
----> 1 subSan = getSanSubmission(docSanAnsDf, threshold=0.2)
      2 

NameError: name 'docSanAnsDf' is not defined

## === cell 36
subSan.head(20)



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2797309472.py in <cell line: 0>()
----> 1 subSan.head(20)
      2 

NameError: name 'subSan' is not defined

## === cell 37
sub = pd.concat([subLan, subSan])
sub_sorted = sub.sort_values("example_id")



## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4152968997.py in <cell line: 0>()
----> 1 sub = pd.concat([subLan, subSan])
      2 sub_sorted = sub.sort_values("example_id")
      3 

NameError: name 'subLan' is not defined

## === cell 38
sub_sorted.head(20)



## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3290350948.py in <cell line: 0>()
----> 1 sub_sorted.head(20)
      2 

NameError: name 'sub_sorted' is not defined

## === cell 39
try:
    refineSub = refineLan(sub, list_mappings_df, debug=True)
    final_sub = refineSub
except Exception as e:
    print("Refinement error:", e)
    final_sub = sub_sorted



## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2253775507.py in <cell line: 0>()
      1 try:
----> 2     refineSub = refineLan(sub, list_mappings_df, debug=True)
      3     final_sub = refineSub

NameError: name 'sub' is not defined

During handling of the above exception, another exception occurred:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2253775507.py in <cell line: 0>()
      4 except Exception as e:
      5     print("Refinement error:", e)
----> 6     final_sub = sub_sorted
      7 

NameError: name 'sub_sorted' is not defined

## === cell 40
output_path = "./submission.csv"
try:
    final_sub.to_csv(
        output_path, index=False, columns=["example_id", "PredictionString"]
    )
    print(f"Submission written to {output_path} with {final_sub.shape[0]} rows.")
except Exception as e:
    print("Failed to write refined submission, attempting plain concat:", e)
    sub.to_csv(output_path, index=False, columns=["example_id", "PredictionString"])
    print(f"Fallback submission written to {output_path} with {sub.shape[0]} rows.")

## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2546401026.py in <cell line: 0>()
      2 try:
----> 3     final_sub.to_csv(
      4         output_path, index=False, columns=["example_id", "PredictionString"]

NameError: name 'final_sub' is not defined

During handling of the above exception, another exception occurred:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2546401026.py in <cell line: 0>()
      7 except Exception as e:
      8     print("Failed to write refined submission, attempting plain concat:", e)
----> 9     sub.to_csv(output_path, index=False, columns=["example_id", "PredictionString"])
     10     print(f"Fallback submission written to {output_path} with {sub.shape[0]} rows.")

NameError: name 'sub' is not defined
