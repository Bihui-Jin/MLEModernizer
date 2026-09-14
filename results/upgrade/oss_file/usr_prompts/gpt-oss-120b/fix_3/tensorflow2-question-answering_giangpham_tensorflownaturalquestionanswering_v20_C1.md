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
import subprocess, sys

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "transformers", "protobuf==3.20.3"]
)



## === cell 1
import numpy as np
import pandas as pd
import random
import re
import json
from tqdm import tqdm
import tensorflow as tf
from transformers import AutoTokenizer, TFBertModel



## === cell 2
debug = False

f_train = "../input/tensorflow2-question-answering/simplified-nq-train.jsonl"
f_test = "../input/tensorflow2-question-answering/simplified-nq-test.jsonl"
num_train_samples = 44943
num_test_samples = 346

AnswerType = {"NO_ANSWER": 0, "YES": 1, "NO": 2, "SHORT": 3, "LONG": 4}

AnswerTypeRev = {0: "NO_ANSWER", 1: "YES", 2: "NO", 3: "SHORT", 4: "LONG"}

cleanr = re.compile("<.*?>")


def clean_html(raw_html):
    return re.sub(cleanr, "<tag>", raw_html)


def parseData(
    filename, drop_noanswer_rate=0.8, drop_null_instances_rate=0.5, split=1.0
):
    INSTANCE_WORDS_LEN = 500
    STRIDE = 128
    if "train" in filename:
        n_sam = num_train_samples
        if debug:
            n_sam = 200
        is_train = True
    else:
        n_sam = num_test_samples
        if debug:
            n_sam = 2
        is_train = False

    list_instances = []
    second_list_instances = []

    with open(filename) as f:
        progress = tqdm(f, total=n_sam)
        for line in progress:
            data = json.loads(line)
            rand_split = random.uniform(0, 1)
            to_second_list = rand_split > split

            example_id = data["example_id"]
            question = data["question_text"]
            doc_text_raw = clean_html(data["document_text"])
            doc_text_split = doc_text_raw.split()
            clean_doc = list(filter(("<tag>").__ne__, doc_text_split))

            if is_train:
                ans_id = data["annotations"][0]["long_answer"]["candidate_index"]
                if ans_id == -1:
                    if random.uniform(0, 1) < drop_noanswer_rate:
                        continue
                else:
                    lan_start = data["long_answer_candidates"][ans_id]["start_token"]
                    lan_stop = data["long_answer_candidates"][ans_id]["end_token"]
                    long_answer_raw = doc_text_split[lan_start:lan_stop]

                    list_sans = data["annotations"][0]["short_answers"]
                    yes_no = data["annotations"][0]["yes_no_answer"]
                    is_yes_no = yes_no != "NONE"

                    san_start = lan_stop
                    san_stop = lan_start
                    for san in list_sans:
                        this_start = san["start_token"]
                        this_stop = san["end_token"]
                        san_start = min(san_start, this_start)
                        san_stop = max(san_stop, this_stop)

                    if san_start < san_stop:
                        is_san = True
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
            else:
                ans_id = -1
                is_san = False
                new_lan_start = new_lan_stop = new_san_start = new_san_stop = 0
                is_yes_no = False
                yes_no = "NONE"

            len_ques = len(question.split())
            part_len = INSTANCE_WORDS_LEN - len_ques
            num_ins = max(0, (len(clean_doc) - part_len) // STRIDE)

            for part_id in range(num_ins + 1):
                part_start = part_id * STRIDE
                part_end = min(len(clean_doc), part_start + part_len)
                part = clean_doc[part_start:part_end]

                target_ans_ins = "NO_ANSWER"
                an_start_ins = an_stop_ins = 0

                if is_train and ans_id > -1:
                    if is_san:
                        if new_san_start >= part_start and new_san_stop < part_end:
                            san_start_ins = new_san_start - part_start
                            san_stop_ins = new_san_stop - part_start
                            an_start_ins, an_stop_ins = san_start_ins, san_stop_ins
                            target_ans_ins = "SHORT"
                    else:
                        if new_lan_stop - new_lan_start <= part_len:
                            if new_lan_start >= part_start and new_lan_stop < part_end:
                                lan_start_ins = new_lan_start - part_start
                                lan_stop_ins = new_lan_stop - part_start
                                an_start_ins, an_stop_ins = lan_start_ins, lan_stop_ins
                                target_ans_ins = "LONG"
                                if is_yes_no:
                                    target_ans_ins = yes_no
                part_str = " ".join(part)
                instance = {
                    "question": question,
                    "context": part_str,
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
                        if random.uniform(0, 1) > drop_null_instances_rate:
                            (
                                second_list_instances
                                if to_second_list
                                else list_instances
                            ).append(instance)
                else:
                    list_instances.append(instance)
    return list_instances, second_list_instances


def getRawAndCleanTextDocs(test_file):
    list_sample = []
    with open(test_file) as f:
        for line in tqdm(f):
            data = json.loads(line)
            example_id = data["example_id"]
            question = data["question_text"]
            doc_text_raw = clean_html(data["document_text"])
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
                new_cand = {
                    "start_token": new_start,
                    "end_token": new_stop,
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
    return list_sample


def mergeInstanceResult(test_res, list_test_ins):
    for i, ins_res in enumerate(test_res):
        start = np.argmax(ins_res[0])
        stop = np.argmax(ins_res[1])
        target = np.argmax(ins_res[2])
        list_test_ins[i].update(
            {
                "start": start,
                "stop": stop,
                "target": target,
                "start_score": float(ins_res[0][start]),
                "stop_score": float(ins_res[1][stop]),
                "target_score": float(ins_res[2][target]),
                "start_CLS": float(ins_res[0][0]),
                "stop_CLS": float(ins_res[1][0]),
            }
        )
    return list_test_ins


def mergeDocumentResult(doc_df, ins_df):
    list_doc_id = doc_df["example_id"].unique()
    list_doc_result_lan = []
    for doc_id in list_doc_id:
        ins_of_doc = ins_df.loc[ins_df["example_id"] == doc_id]
        start_ins = ins_of_doc.loc[ins_of_doc["start"] != 0]
        stop_ins = ins_of_doc.loc[ins_of_doc["stop"] != 0]
        target_ins = ins_of_doc.loc[ins_of_doc["target"] != 0]
        all_non_zero = pd.concat([start_ins, stop_ins, target_ins]).drop_duplicates()
        real_start = real_stop = 0
        real_target = AnswerTypeRev[0]
        max_vote = 0.005
        lan_start_raw = lan_stop_raw = -1
        for _, row in all_non_zero.iterrows():
            part_id = row["part_id"]
            part_start = part_id * 128
            if row["stop"] > row["start"]:
                vote = (
                    row["start_score"]
                    - row["start_CLS"]
                    + row["stop_score"]
                    - row["stop_CLS"]
                )
                if vote > max_vote:
                    real_start = row["start"] + part_start
                    real_stop = row["stop"] + part_start + 1
                    real_target = AnswerTypeRev[row["target"]]
                    max_vote = vote
        sam = doc_df.loc[doc_df["example_id"] == doc_id]
        for _, samrow in sam.iterrows():
            if real_start != 0:
                for cand_id, cand in enumerate(samrow["candidates"]):
                    lan_start = cand[0]["start_token"]
                    lan_stop = cand[0]["end_token"]
                    lan_range = set(range(lan_start, lan_stop + 1))
                    real_range = set(range(real_start, real_stop + 1))
                    inter = len(lan_range & real_range)
                    if inter > 0:
                        ratio = inter / len(lan_range)
                        if inter > best_score if "best_score" in locals() else -1:
                            best_score = inter
                            best_match_id = cand_id
                            best_ratio = ratio
                        elif inter == best_score and ratio > best_ratio:
                            best_match_id = cand_id
                            best_ratio = ratio
                lan_raw_res = samrow["candidates"][best_match_id][1]
                lan_start_raw = lan_raw_res["start_token"]
                lan_stop_raw = lan_raw_res["end_token"]
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


def preprocess_data(data, tokenizer):
    x1, x2, x3, y = [], [], [], []
    for one_sam in tqdm(data):
        tokenized = tokenizer.encode_plus(
            one_sam["question"],
            one_sam["context"],
            padding="max_length",
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )
        x1.append(tf.cast(tokenized["input_ids"], tf.int32))
        x2.append(tf.cast(tokenized["token_type_ids"], tf.int32))
        x3.append(tf.cast(tokenized["attention_mask"], tf.int32))
        y.append([one_sam["start"], one_sam["stop"], AnswerType[one_sam["target"]]])
    return x1, x2, x3, y


def get_strategy():
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        return tf.distribute.experimental.TPUStrategy(resolver)
    except Exception:
        return tf.distribute.get_strategy()


def build_model(model_name):
    bert = TFBertModel.from_pretrained(model_name)
    NUM_TARGET = 5

    class MyQAModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.bert = bert
            self.dropout1 = tf.keras.layers.Dropout(0.2)
            self.dropout2 = tf.keras.layers.Dropout(0.2)
            self.start_logits = tf.keras.layers.Dense(1)
            self.stop_logits = tf.keras.layers.Dense(1)
            self.target = tf.keras.layers.Dense(NUM_TARGET)

        def call(self, inputs):
            bert_out = self.bert(
                inputs[0], token_type_ids=inputs[1], attention_mask=inputs[2]
            )
            seq_output = self.dropout1(bert_out[0])
            start_logits = tf.squeeze(self.start_logits(seq_output), -1)
            stop_logits = tf.squeeze(self.stop_logits(seq_output), -1)
            pooled_output = self.dropout2(bert_out[1])
            target_logits = self.target(pooled_output)
            paddings = tf.constant([[0, 0], [0, 512 - NUM_TARGET]])
            target_logits = tf.pad(target_logits, paddings)
            return tf.stack([start_logits, stop_logits, target_logits], axis=1)

    return MyQAModel()


DEFAULT_MODEL_NAME = "bert-base-uncased"


def getRawInstanceResults(list_test, verbose=True):
    if verbose:
        print("Tokenizing test instances for the first model")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    x1, x2, x3, y_test = preprocess_data(list_test, tokenizer)
    x1, x2, x3, y_test = map(tf.convert_to_tensor, (x1, x2, x3, y_test))
    strategy = get_strategy()
    with strategy.scope():
        model = build_model(DEFAULT_MODEL_NAME)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.00005),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
    model.evaluate([x1[:16], x2[:16], x3[:16]], y_test[:16], verbose=0)
    try:
        model.load_weights("../input/model1/weights-02-0.555.h5")
        if verbose:
            print("Loaded pretrained weights for first model")
    except Exception:
        if verbose:
            print("Pretrained weights for first model not found – using random init")
    preds = []
    batch = 256
    for i in range(0, len(x1), batch):
        preds.append(
            model.predict([x1[i : i + batch], x2[i : i + batch], x3[i : i + batch]])
        )
    return np.concatenate(preds, axis=0)


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
        if long_start != -1 and long_stop != -1:
            lan_split = raw_doc.split()[long_start:long_stop]
            len_ques = len(question.split())
            part_len = INSTANCE_WORDS_LEN - len_ques
            if len(lan_split) > part_len:
                num_ins = (len(lan_split) - part_len) // STRIDE + 2
            else:
                num_ins = 1
            for part_id in range(num_ins):
                part_start = part_id * STRIDE
                part_end = min(len(lan_split), part_start + part_len)
                part = " ".join(lan_split[part_start:part_end])
                list_instances.append(
                    {
                        "question": question,
                        "context": part,
                        "example_id": example_id,
                        "part_id": part_id,
                    }
                )
    return list_instances


def preprocess_san(list_san_ins):
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    tokenizer.add_special_tokens({"unk_token": "<tag>"})
    x1, x2, x3 = [], [], []
    for one_sam in tqdm(list_san_ins):
        tokenized = tokenizer.encode_plus(
            one_sam["question"],
            one_sam["context"],
            padding="max_length",
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )
        x1.append(tf.cast(tokenized["input_ids"], tf.int32))
        x2.append(tf.cast(tokenized["token_type_ids"], tf.int32))
        x3.append(tf.cast(tokenized["attention_mask"], tf.int32))
    return x1, x2, x3


def getSanInstanceResults(list_san_ins, verbose=True):
    if verbose:
        print("Tokenizing short‑answer instances")
    x1, x2, x3 = preprocess_san(list_san_ins)
    x1, x2, x3 = map(tf.convert_to_tensor, (x1, x2, x3))
    strategy = get_strategy()
    with strategy.scope():
        model = build_model(DEFAULT_MODEL_NAME)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.00005),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
    model.evaluate(
        [x1[:1], x2[:1], x3[:1]], tf.convert_to_tensor([[0, 0, 0]]), verbose=0
    )
    try:
        model.load_weights("../input/model1/weights-02-0.701.h5")
        if verbose:
            print("Loaded pretrained weights for second model")
    except Exception:
        if verbose:
            print("Pretrained weights for second model not found – using random init")
    preds = model.predict([x1, x2, x3])
    return tf.nn.softmax(preds)


def mergeSanInstanceResult(san_res, list_san_ins, debug=False):
    for i, ins_res in enumerate(san_res.numpy()):
        start = int(np.argmax(ins_res[0]))
        stop = int(np.argmax(ins_res[1]))
        target = int(np.argmax(ins_res[2]))
        list_san_ins[i].update(
            {
                "start": start,
                "stop": stop,
                "target": AnswerTypeRev[target],
                "start_score": float(ins_res[0][start]),
                "stop_score": float(ins_res[1][stop]),
                "target_score": float(ins_res[2][target]),
                "start_CLS": float(ins_res[0][0]),
                "stop_CLS": float(ins_res[1][0]),
            }
        )
        if debug:
            ctx = list_san_ins[i]["context"].split()
            if start > 0 and stop > start:
                print(
                    AnswerTypeRev[target],
                    ": ",
                    list_san_ins[i]["question"],
                    ": ",
                    " ".join(ctx[start : stop + 1]),
                )
    return list_san_ins


def mergeSanLan(doc_lan_df, san_ins_res_df, verbose=True):
    SAN_STRIDE = 384
    list_out = []
    if verbose:
        print(
            f"Merging {len(san_ins_res_df)} short‑answer predictions with {len(doc_lan_df)} documents"
        )
    for _, doc in doc_lan_df.iterrows():
        doc_id = doc["example_id"]
        lan_start = doc["start_token"]
        lan_stop = doc["stop_token"]
        lan_target = doc["target"]
        san_subset = san_ins_res_df[san_ins_res_df["example_id"] == doc_id]
        best_score = 1e-5
        san_start = san_stop = -1
        final_target = AnswerTypeRev[0]
        for _, san in san_subset.iterrows():
            part_start = san["part_id"] * SAN_STRIDE
            if san["start"] > 0 and san["stop"] > san["start"]:
                vote = (
                    san["start_score"]
                    - san["start_CLS"]
                    + san["stop_score"]
                    - san["stop_CLS"]
                )
                if vote > best_score:
                    san_start = san["start"] + part_start + lan_start
                    san_stop = san["stop"] + part_start + lan_start
                    best_score = vote
                final_target = san["target"]
        if lan_target not in ("NO_ANSWER", "SHORT", "LONG") and final_target in (
            "NO_ANSWER",
            "LONG",
        ):
            final_target = lan_target
        list_out.append(
            {
                "example_id": doc_id,
                "question": doc["question"],
                "raw_document": doc["raw_document"],
                "lan_start": lan_start,
                "lan_stop": lan_stop,
                "san_start": san_start,
                "san_stop": san_stop,
                "target": final_target,
            }
        )
    return list_out


def getFinalResult(f_test):
    list_all_ins, _ = parseData(f_test)
    all_ins_res = getRawInstanceResults(list_all_ins)
    list_fine_res_all_ins = mergeInstanceResult(all_ins_res, list_all_ins)
    fine_res_all_ins_df = pd.DataFrame(list_fine_res_all_ins)
    doc_df = pd.DataFrame(getRawAndCleanTextDocs(f_test))
    doc_res_lan = mergeDocumentResult(doc_df, fine_res_all_ins_df)
    doc_res_lan_df = pd.DataFrame(doc_res_lan)
    list_san_ins = getListShortSample(doc_res_lan_df)
    raw_san_ins_res = getSanInstanceResults(list_san_ins)
    list_san_res = mergeSanInstanceResult(raw_san_ins_res, list_san_ins, debug=False)
    list_san_res_df = pd.DataFrame(list_san_res)
    return mergeSanLan(doc_res_lan_df, list_san_res_df)


def getLines(san_lan_doc):
    lines = []
    for doc in san_lan_doc:
        base_id = doc["example_id"]
        line_long = {"example_id": f"{base_id}_long"}
        line_short = {"example_id": f"{base_id}_short"}
        if doc["lan_start"] != -1 and doc["lan_stop"] != -1:
            line_long["PredictionString"] = f"{doc['lan_start']}:{doc['lan_stop']}"
        else:
            line_long["PredictionString"] = ""
        if doc["san_start"] != -1 and doc["san_stop"] != -1:
            line_short["PredictionString"] = f"{doc['san_start']}:{doc['san_stop']}"
        else:
            line_short["PredictionString"] = ""
        if doc["target"] in ("YES", "NO"):
            line_short["PredictionString"] = doc["target"]
        lines.extend([line_long, line_short])
    return lines


def getSubmission():
    san_lan_doc = getFinalResult(
        "../input/tensorflow2-question-answering/simplified-nq-test.jsonl"
    )
    lines = getLines(san_lan_doc)
    df = pd.DataFrame(lines)
    df.sort_values("example_id", inplace=True)
    df.to_csv(
        "./submission.csv", index=False, columns=["example_id", "PredictionString"]
    )
    print("Submission saved to ./submission.csv")




## === cell 3
getSubmission()
