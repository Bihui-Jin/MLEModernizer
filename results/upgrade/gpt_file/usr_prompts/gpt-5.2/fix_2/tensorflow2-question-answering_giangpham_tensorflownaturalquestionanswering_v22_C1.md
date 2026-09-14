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

0.3731778425655977

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, json, re, random, shutil, string
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from transformers import AutoTokenizer, TFBertModel



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
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




## === cell 2
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


def getRawAndCleanTextDocs(test_file):
    debug_local = False
    if debug_local:
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

        list_test_ins[i]["start"] = int(start)
        list_test_ins[i]["stop"] = int(stop)
        list_test_ins[i]["target"] = int(target)

        list_test_ins[i]["start_score"] = float(start_score)
        list_test_ins[i]["stop_score"] = float(stop_score)
        list_test_ins[i]["target_score"] = float(target_score)

        list_test_ins[i]["start_CLS"] = float(start_CLS)
        list_test_ins[i]["stop_CLS"] = float(stop_CLS)
    return list_test_ins


def mergeDocumentResult(doc_df, ins_df, debug_local=False):
    list_doc_result_lan = []
    for idx, doc in doc_df.iterrows():
        doc_id = doc["example_id"]
        if debug_local:
            print(doc_id, " - QUESTION: ", doc["question"])

        ins_of_doc = ins_df.loc[ins_df["example_id"] == doc_id]

        start_ins = ins_of_doc.loc[ins_of_doc["start"] != 0]
        stop_ins = ins_of_doc.loc[ins_of_doc["stop"] != 0]
        target_ins = ins_of_doc.loc[ins_of_doc["target"] != 0]

        all_non_zero = pd.concat([start_ins, stop_ins, target_ins]).drop_duplicates()

        real_start = 0
        real_stop = 0
        real_target = AnswerTypeRev[0]

        max_vote = 0.05

        lan_start_raw = -1
        lan_stop_raw = -1

        for index, row in all_non_zero.iterrows():
            part_id = row["part_id"]
            part_start = part_id * 128
            stop = row["stop"]
            start = row["start"]
            if (
                row["start_score"]
                - row["start_CLS"]
                + row["stop_score"]
                - row["stop_CLS"]
                > max_vote
            ):
                real_start = start + part_start
                real_stop = stop + part_start + 1
                real_target = AnswerTypeRev[row["target"]]
                max_vote = (
                    row["start_score"]
                    - row["start_CLS"]
                    + row["stop_score"]
                    - row["stop_CLS"]
                )
        if debug_local:
            print(
                max_vote,
                real_target,
                ": ",
                " ".join(doc["document"].split()[real_start:real_stop]),
            )

        if real_start != 0 and real_stop != 0:
            list_cands = doc["candidates"]
            real_range = [*range(real_start, real_stop, 1)]

            best_score = 0.5
            shortest = 100000000000
            best_match_id = 1

            for cand_id, cand in enumerate(list_cands):
                lan_start = cand[0]["start_token"]
                lan_stop = cand[0]["end_token"]
                lan_range = [*range(lan_start, lan_stop, 1)]

                inter = float(len(list(set(real_range) & set(lan_range))))

                if inter > best_score:
                    best_score = inter
                    best_match_id = cand_id
                    shortest = len(lan_range)

                elif inter == best_score:
                    if shortest > len(lan_range):
                        shortest = len(lan_range)
                        best_match_id = cand_id

            lan_raw_res = list_cands[best_match_id][1]

            lan_start_raw = lan_raw_res["start_token"]
            lan_stop_raw = lan_raw_res["end_token"]

            if debug_local:
                print(" ".join(doc["raw_document"].split()[lan_start_raw:lan_stop_raw]))
        doc_result_lan = {}
        doc_result_lan["example_id"] = doc_id
        doc_result_lan["question"] = doc["question"]
        doc_result_lan["raw_document"] = doc["raw_document"]
        doc_result_lan["start_token"] = lan_start_raw
        doc_result_lan["stop_token"] = lan_stop_raw
        doc_result_lan["target"] = real_target
        list_doc_result_lan.append(doc_result_lan)
        if debug_local:
            print("\n")
    return list_doc_result_lan


def preprocess_data(data, tokenizer):
    progress = tqdm(data, total=len(data))
    x1 = []
    x2 = []
    x3 = []
    y = []
    for one_sam in progress:
        tokenized_sam = tokenizer.encode_plus(
            one_sam["question"],
            one_sam["context"],
            padding="max_length",
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )
        x1.append(tf.cast(tokenized_sam["input_ids"], tf.int32))
        x2.append(tf.cast(tokenized_sam["token_type_ids"], tf.int32))
        x3.append(tf.cast(tokenized_sam["attention_mask"], tf.int32))
        y.append([one_sam["start"], one_sam["stop"], AnswerType[one_sam["target"]]])
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
        print(e)
        print("No TPU detected")
        strategy = tf.distribute.get_strategy()
    return strategy


def build_model(model_name):
    BertModelLayer = TFBertModel.from_pretrained(model_name, local_files_only=True)

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


def getRawInstanceResults(list_test, verbose=True, debug_local=False):
    if verbose:
        print("Getting raw result for all the instances generated from test file")

    model_name = "/kaggle/input/tensorflow-question-answer-fine-data"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    x_test1, x_test2, x_test3, y_test = preprocess_data(list_test, tokenizer)
    if verbose:
        print("Finish tokenizing ", len(list_test), " data for the first model")

    x_test1 = tf.convert_to_tensor(x_test1)
    x_test2 = tf.convert_to_tensor(x_test2)
    x_test3 = tf.convert_to_tensor(x_test3)
    y_test = tf.convert_to_tensor(y_test)

    if verbose:
        print("Preparing model")

    strategy = get_strategy()
    with strategy.scope():
        testModel = build_model(model_name)
        optAdam = tf.keras.optimizers.Adam(learning_rate=0.00005)
        lossSCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metricSCA = tf.keras.metrics.SparseCategoricalAccuracy()
        testModel.compile(optimizer=optAdam, loss=lossSCE, metrics=[metricSCA])

    testModel.evaluate(
        x=[x_test1[0:16], x_test2[0:16], x_test3[0:16]], y=y_test[0:16], verbose=0
    )

    with strategy.scope():
        testModel.load_weights("/kaggle/input/model1/weights-02-0.555.h5")

    if verbose:
        print("Finish loading pretrained weights for the model")

    test_res = []
    num_iter = len(x_test1) // 256 + 1
    for i in range(num_iter):
        start = i * 256
        stop = min(len(x_test1), (i + 1) * 256)
        if start >= stop:
            continue
        testSam = [x_test1[start:stop], x_test2[start:stop], x_test3[start:stop]]
        res = testModel.predict(testSam, verbose=0)
        res = tf.nn.softmax(res)
        test_res.extend(res)

    test_res = np.array(test_res)

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
    debug_local = True
    model_name = "/kaggle/input/tensorflow-question-answer-fine-data"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.add_special_tokens({"unk_token": "<tag>"})
    if debug_local:
        print(tokenizer.unk_token)
    progress = tqdm(list_san_ins, total=len(list_san_ins))
    x1 = []
    x2 = []
    x3 = []
    for one_sam in progress:
        tokenized_sam = tokenizer.encode_plus(
            one_sam["question"],
            one_sam["context"],
            padding="max_length",
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )
        x1.append(tf.cast(tokenized_sam["input_ids"], tf.int32))
        x2.append(tf.cast(tokenized_sam["token_type_ids"], tf.int32))
        x3.append(tf.cast(tokenized_sam["attention_mask"], tf.int32))
    return x1, x2, x3


def getSanInstanceResults(list_san_ins, verbose=True):
    if verbose:
        print(
            "Calculating raw result for ", len(list_san_ins), " short answer instances"
        )

    x_san1, x_san2, x_san3 = preprocess_san(list_san_ins)
    if verbose:
        print("Finish tokenizing test instance for the second model")

    x_san1 = tf.convert_to_tensor(x_san1)
    x_san2 = tf.convert_to_tensor(x_san2)
    x_san3 = tf.convert_to_tensor(x_san3)

    strategy = get_strategy()
    model_name = "/kaggle/input/tensorflow-question-answer-fine-data"
    with strategy.scope():
        testModel = build_model(model_name)
        optAdam = tf.keras.optimizers.Adam(learning_rate=0.00005)
        lossSCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metricSCA = tf.keras.metrics.SparseCategoricalAccuracy()
        testModel.compile(optimizer=optAdam, loss=lossSCE, metrics=[metricSCA])

    tem_y = tf.convert_to_tensor([[0, 0, 0]])
    testModel.evaluate(x=[x_san1[0:1], x_san2[0:1], x_san3[0:1]], y=tem_y, verbose=0)

    with strategy.scope():
        testModel.load_weights("/kaggle/input/model1/weights-02-0.701.h5")

    if verbose:
        print("Finish loading the weights for the second model")

    testSam = [x_san1, x_san2, x_san3]
    san_res = testModel.predict(testSam, verbose=0)
    san_res = tf.nn.softmax(san_res)

    if verbose:
        print("Finish calculating, get result of size: ", san_res.shape)

    return san_res


def mergeSanInstanceResult(san_res, list_san_ins, debug_local=True):
    for i in range(len(list_san_ins)):
        ins_res = san_res[i].numpy()
        start = np.argmax(ins_res[0])
        stop = np.argmax(ins_res[1])
        target = np.argmax(ins_res[2])

        start_score = ins_res[0][start]
        stop_score = ins_res[1][stop]
        target_score = ins_res[2][target]

        start_CLS = ins_res[0][0]
        stop_CLS = ins_res[1][0]

        list_san_ins[i]["start"] = int(start)
        list_san_ins[i]["stop"] = int(stop)
        list_san_ins[i]["target"] = AnswerTypeRev[int(target)]

        list_san_ins[i]["start_score"] = float(start_score)
        list_san_ins[i]["stop_score"] = float(stop_score)
        list_san_ins[i]["target_score"] = float(target_score)

        list_san_ins[i]["start_CLS"] = float(start_CLS)
        list_san_ins[i]["stop_CLS"] = float(stop_CLS)

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
    for index, row in doc_lan_df.iterrows():
        doc_id = row["example_id"]

        lan_start = row["start_token"]
        lan_stop = row["stop_token"]
        lan_target = row["target"]

        san_start = -1
        san_stop = -1
        final_target = lan_target

        san_list = san_ins_res_df.loc[san_ins_res_df["example_id"] == doc_id]

        max_score = 0.00001
        for san_idx, san in san_list.iterrows():
            part_id = san["part_id"]
            part_start = part_id * SAN_STRIDE
            start_in_part = san["start"]
            stop_in_part = san["stop"]

            start_in_lan = start_in_part + part_start
            stop_in_lan = stop_in_part + part_start

            if start_in_part > 0 and stop_in_part > start_in_part:
                if (
                    san["start_score"]
                    - san["start_CLS"]
                    + san["stop_score"]
                    - san["stop_CLS"]
                    > max_score
                ):
                    san_start = start_in_lan + lan_start
                    san_stop = stop_in_lan + lan_start
                    max_score = (
                        san["start_score"]
                        - san["start_CLS"]
                        + san["stop_score"]
                        - san["stop_CLS"]
                    )
            final_target = san["target"]

        if verbose:
            print(doc_id, " QUESTION: ", row["question"])
            print(
                final_target, " ".join(row["raw_document"].split()[lan_start:lan_stop])
            )
            print("SHORT: ", " ".join(row["raw_document"].split()[san_start:san_stop]))
            print("\n")
        san_lan_res_doc = {}
        san_lan_res_doc["example_id"] = doc_id
        san_lan_res_doc["question"] = row["question"]
        san_lan_res_doc["raw_document"] = row["raw_document"]
        san_lan_res_doc["lan_start"] = lan_start
        san_lan_res_doc["lan_stop"] = lan_stop
        san_lan_res_doc["san_start"] = san_start
        san_lan_res_doc["san_stop"] = san_stop
        san_lan_res_doc["target"] = final_target

        list_san_lan_res_doc.append(san_lan_res_doc)
    return list_san_lan_res_doc


def getFinalResult(f_test_path):
    list_all_ins, _ = parseData(f_test_path)
    all_ins_res = getRawInstanceResults(list_all_ins)
    list_fine_res_all_ins = mergeInstanceResult(all_ins_res, list_all_ins)
    fine_res_all_ins_df = pd.DataFrame(list_fine_res_all_ins)

    list_doc = getRawAndCleanTextDocs(f_test_path)
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
    lines_df = pd.DataFrame(lines)
    sorted_df = lines_df.sort_values("example_id")
    sorted_df.to_csv(
        "./submission.csv", index=False, columns=["example_id", "PredictionString"]
    )
    return sorted_df




## === cell 3
sub_df = getSubmission()
print(sub_df.head())
print("Wrote ./submission.csv with shape:", sub_df.shape)
print("Exists:", os.path.exists("./submission.csv"))
print(
    "File size bytes:",
    os.path.getsize("./submission.csv") if os.path.exists("./submission.csv") else None,
)

## --- ERROR in cell 3, traceback:
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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/tensorflow-question-answer-fine-data'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_11/1297450789.py in <cell line: 0>()
      1 # Generate submission.csv
----> 2 sub_df = getSubmission()
      3 print(sub_df.head())
      4 print("Wrote ./submission.csv with shape:", sub_df.shape)
      5 print("Exists:", os.path.exists("./submission.csv"))

/tmp/ipykernel_11/2993902416.py in getSubmission()
    813 
    814 def getSubmission():
--> 815     san_lan_doc = getFinalResult(f_test)
    816     lines = getLines(san_lan_doc)
    817     lines_df = pd.DataFrame(lines)

/tmp/ipykernel_11/2993902416.py in getFinalResult(f_test_path)
    757 def getFinalResult(f_test_path):
    758     list_all_ins, _ = parseData(f_test_path)
--> 759     all_ins_res = getRawInstanceResults(list_all_ins)
    760     list_fine_res_all_ins = mergeInstanceResult(all_ins_res, list_all_ins)
    761     fine_res_all_ins_df = pd.DataFrame(list_fine_res_all_ins)

/tmp/ipykernel_11/2993902416.py in getRawInstanceResults(list_test, verbose, debug_local)
    467     # The previous "../input/..." relative path triggers HFValidationError in some transformers versions.
    468     model_name = "/kaggle/input/tensorflow-question-answer-fine-data"
--> 469     tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    470 
    471     x_test1, x_test2, x_test3, y_test = preprocess_data(list_test, tokenizer)

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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/tensorflow-question-answer-fine-data'. Use `repo_type` argument if needed.
