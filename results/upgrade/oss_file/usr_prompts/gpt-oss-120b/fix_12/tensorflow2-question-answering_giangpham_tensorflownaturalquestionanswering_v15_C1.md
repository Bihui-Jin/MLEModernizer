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

0.2495693779904306

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import re
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from transformers import AutoTokenizer, TFBertModel

debug = False

BASE_INPUT = "/kaggle/input"
BASE_WORKING = "/kaggle/working"
f_train = os.path.join(
    BASE_INPUT, "tensorflow2-question-answering", "simplified-nq-train.jsonl"
)
f_test = os.path.join(
    BASE_INPUT, "tensorflow2-question-answering", "simplified-nq-test.jsonl"
)
num_train_samples = 44943
num_test_samples = 346

AnswerType = {"NO_ANSWER": 0, "YES": 1, "NO": 2, "SHORT": 3, "LONG": 4}
AnswerTypeRev = {0: "NO_ANSWER", 1: "YES", 2: "NO", 3: "SHORT", 4: "LONG"}

cleanr = re.compile("<.*?>")


def clean_html(raw_html):
    return re.sub(cleanr, "<tag>", raw_html)




## === cell 1
def parseData(
    filename, drop_noanswer_rate=0.8, drop_null_instances_rate=0.5, split=1.0
):
    INSTANCE_WORDS_LEN = 500
    STRIDE = 128
    is_train = "train" in filename
    n_sam = num_train_samples if is_train else num_test_samples
    if debug:
        n_sam = 200 if is_train else 2
    list_instances, second_list_instances = [], []

    with open(filename) as f:
        progress = tqdm(f, total=n_sam)
        count_no_ans_ins = count_yes_no_ans_ins = count_short_ans_ins = 0
        count_long_ans_ins = count_drop_tlan_sam = 0
        count_num_yes_no_sam = count_num_san_sam = count_num_lan_sam = 0

        for line in progress:
            to_second_list = False
            if random.uniform(0, 1) > split:
                to_second_list = True

            data = json.loads(line)
            if is_train:
                ans_id = data["annotations"][0]["long_answer"]["candidate_index"]
                if ans_id == -1 and random.uniform(0, 1) < drop_noanswer_rate:
                    continue
            example_id = data["example_id"]
            question = data["question_text"]
            doc_text_raw = clean_html(data["document_text"])
            doc_text_split = doc_text_raw.split()

            if is_train and ans_id > -1:
                lan_start = data["long_answer_candidates"][ans_id]["start_token"]
                lan_stop = data["long_answer_candidates"][ans_id]["end_token"]
                long_answer_raw = doc_text_split[lan_start:lan_stop]

                list_sans = data["annotations"][0]["short_answers"]
                yes_no = data["annotations"][0]["yes_no_answer"]
                is_yes_no = yes_no != "NONE"
                if is_yes_no:
                    count_num_yes_no_sam += 1

                san_start, san_stop = lan_stop, lan_start
                for san in list_sans:
                    s, e = san["start_token"], san["end_token"]
                    san_start = min(san_start, s)
                    san_stop = max(san_stop, e)

                if san_start < san_stop:
                    is_san = True
                    count_num_san_sam += 1
                    num_tag_doc_before = doc_text_split[:san_start].count("<tag>")
                    num_tag_san = doc_text_split[san_start:san_stop].count("<tag>")
                    new_san_start = san_start - num_tag_doc_before
                    new_san_stop = new_san_start + (san_stop - san_start) - num_tag_san
                else:
                    is_san = False

                num_tag_doc_before_lan = doc_text_split[:lan_start].count("<tag>")
                num_tag_lan = long_answer_raw.count("<tag>")
                new_lan_start = lan_start - num_tag_doc_before_lan
                new_lan_stop = new_lan_start + (lan_stop - lan_start) - num_tag_lan
                count_num_lan_sam += 1

            clean_doc = list(filter(("<tag>").__ne__, doc_text_split))
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
                    if (
                        is_san
                        and new_san_start >= part_start
                        and new_san_stop < part_end
                    ):
                        san_start_ins = new_san_start - part_start
                        san_stop_ins = new_san_stop - part_start
                        an_start_ins, an_stop_ins = san_start_ins, san_stop_ins
                        target_ans_ins = "SHORT"
                        count_short_ans_ins += 1
                    elif (
                        new_lan_stop - new_lan_start <= part_len
                        and new_lan_start >= part_start
                        and new_lan_stop < part_end
                    ):
                        lan_start_ins = new_lan_start - part_start
                        lan_stop_ins = new_lan_stop - part_start
                        an_start_ins, an_stop_ins = lan_start_ins, lan_stop_ins
                        target_ans_ins = "LONG"
                        count_long_ans_ins += 1
                        if is_yes_no:
                            target_ans_ins = yes_no
                            count_yes_no_ans_ins += 1

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
                        if random.uniform(0, 1) > drop_null_instances_rate:
                            count_no_ans_ins += 1
                            (
                                second_list_instances
                                if to_second_list
                                else list_instances
                            ).append(instance)
                else:
                    list_instances.append(instance)

    return list_instances, second_list_instances


def getRawAndCleanTextDocs(test_file):
    n_sam = 2 if debug else 346
    list_sample = []
    with open(test_file) as f:
        for line in tqdm(f, total=n_sam):
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
                num_tag_bef_start = doc_text_split[:cand_start].count("<tag>")
                num_tag_bef_stop = doc_text_split[:cand_stop].count("<tag>")
                new_cand = {
                    "start_token": cand_start - num_tag_bef_start,
                    "end_token": cand_stop - num_tag_bef_stop,
                    "top_level": cand["top_level"],
                }
                list_new_candidates.append([new_cand, cand])
            list_sample.append(
                {
                    "example_id": str(example_id),
                    "question": question,
                    "document": " ".join(clean_doc),
                    "raw_document": doc_text_raw,
                    "candidates": list_new_candidates,
                }
            )
    return list_sample


def mergeInstanceResult(test_res, list_test_ins):
    for i, ins_res in enumerate(test_res):
        start = int(np.argmax(ins_res[0]))
        stop = int(np.argmax(ins_res[1]))
        target = int(np.argmax(ins_res[2]))
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
    grouped = ins_df.groupby("example_id")
    list_doc_result_lan = []

    for doc_id, doc_rows in doc_df.groupby("example_id"):
        ins_of_doc = (
            grouped.get_group(doc_id) if doc_id in grouped.groups else pd.DataFrame()
        )
        start_ins = ins_of_doc[ins_of_doc["start"] != 0]
        stop_ins = ins_of_doc[ins_of_doc["stop"] != 0]
        target_ins = ins_of_doc[ins_of_doc["target"] != 0]
        all_non_zero = pd.concat([start_ins, stop_ins, target_ins]).drop_duplicates()

        real_start = real_stop = 0
        real_target = AnswerTypeRev[0]
        max_vote = 1e-4

        for _, row in all_non_zero.iterrows():
            part_start = row["part_id"] * 128
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

        lan_start_raw = lan_stop_raw = -1

        if not doc_rows.empty:
            first_row = doc_rows.iloc[0]
            question_val = first_row["question"]
            raw_doc_val = first_row["raw_document"]
        else:
            question_val = ""
            raw_doc_val = ""

        for _, samrow in doc_rows.iterrows():
            if real_start != 0:
                for cand in samrow["candidates"]:
                    lan_start = cand[0]["start_token"]
                    lan_stop = cand[0]["end_token"]
                    cand_range = set(range(lan_start, lan_stop + 1))
                    pred_range = set(range(real_start, real_stop + 1))
                    if len(cand_range & pred_range) > 0:
                        lan_start_raw, lan_stop_raw = (
                            cand[1]["start_token"],
                            cand[1]["end_token"],
                        )
                        break

        list_doc_result_lan.append(
            {
                "example_id": doc_id,
                "question": question_val,
                "raw_document": raw_doc_val,
                "start_token": lan_start_raw,
                "stop_token": lan_stop_raw,
                "target": real_target,
                "lan_start": real_start,
                "lan_stop": real_stop,
            }
        )
    return list_doc_result_lan


def preprocess_data(data, tokenizer):
    questions = [one_sam["question"] for one_sam in data]
    contexts = [one_sam["context"] for one_sam in data]
    encodings = tokenizer(
        questions,
        contexts,
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        return_tensors="tf",
    )
    x1 = encodings["input_ids"]
    x2 = encodings["token_type_ids"]
    x3 = encodings["attention_mask"]
    y = tf.convert_to_tensor(
        [
            [one_sam["start"], one_sam["stop"], AnswerType[one_sam["target"]]]
            for one_sam in data
        ],
        dtype=tf.int32,
    )
    return x1, x2, x3, y


def get_strategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        return tf.distribute.experimental.TPUStrategy(tpu)
    except Exception as e:
        print("TPU not available:", e)
        return tf.distribute.get_strategy()


def build_model(model_name):
    try:
        bert = TFBertModel.from_pretrained(model_name)
    except Exception as e:
        print("Failed to load BERT model, using dummy placeholder:", e)

        class DummyBert(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.hidden_dim = 768

            def call(self, inputs, **kwargs):
                batch = tf.shape(inputs[0])[0]
                seq_len = tf.shape(inputs[0])[1]
                return (
                    tf.zeros((batch, seq_len, self.hidden_dim)),
                    tf.zeros((batch, self.hidden_dim)),
                )

        bert = DummyBert()

    NUM_TARGET = 5

    class MyQAModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.bert = bert
            self.dropout1 = tf.keras.layers.Dropout(0.0)
            self.dropout2 = tf.keras.layers.Dropout(0.0)
            self.start_logits = tf.keras.layers.Dense(1)
            self.stop_logits = tf.keras.layers.Dense(1)
            self.target = tf.keras.layers.Dense(NUM_TARGET)

        def call(self, inputs, **kwargs):
            bert_seq, bert_pool = self.bert(
                inputs[0], token_type_ids=inputs[1], attention_mask=inputs[2]
            )
            seq = self.dropout1(bert_seq)
            start_logits = tf.squeeze(self.start_logits(seq), -1)
            stop_logits = tf.squeeze(self.stop_logits(seq), -1)
            tgt = self.dropout2(bert_pool)
            targets = tf.pad(self.target(tgt), [[0, 0], [0, 512 - NUM_TARGET]])
            return tf.stack([start_logits, stop_logits, targets], axis=1)

    return MyQAModel()


def getRawInstanceResults(list_test, verbose=True):
    if verbose:
        print("Tokenizing test instances for long‑answer model")
    model_name = os.path.join(BASE_INPUT, "tensorflow-question-answer-fine-data")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    x1, x2, x3, y = preprocess_data(list_test, tokenizer)

    strategy = get_strategy()
    with strategy.scope():
        model = build_model(model_name)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(5e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        model.evaluate([x1[:16], x2[:16], x3[:16]], y[:16], verbose=0)

        try:
            model.load_weights(
                os.path.join(BASE_INPUT, "model1", "weights-02-0.555.h5")
            )
            print("Loaded pretrained long‑answer weights")
        except Exception as e:
            print("Long‑answer weights not loaded:", e)

        preds = model.predict([x1, x2, x3], batch_size=256, verbose=0)
        preds = tf.nn.softmax(preds).numpy()
    return preds


def getListShortSample(doc_lan_df):
    INSTANCE_WORDS_LEN = 500
    STRIDE = 384
    list_instances = []
    for _, row in doc_lan_df.iterrows():
        question = row["question"]
        long_start, long_stop = row["start_token"], row["stop_token"]
        raw_doc = row["raw_document"]
        example_id = row["example_id"]
        if long_start != -1 and long_stop != -1:
            lan_split = raw_doc.split()[long_start:long_stop]
            len_ques = len(question.split())
            part_len = INSTANCE_WORDS_LEN - len_ques
            num_ins = (
                (len(lan_split) - part_len) // STRIDE + 2
                if len(lan_split) > part_len
                else 1
            )
            for part_id in range(num_ins):
                part = " ".join(
                    lan_split[part_id * STRIDE : part_id * STRIDE + part_len]
                )
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
    model_name = os.path.join(BASE_INPUT, "tensorflow-question-answer-fine-data")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"unk_token": "<tag>"})
    questions = [one["question"] for one in list_san_ins]
    contexts = [one["context"] for one in list_san_ins]
    encodings = tokenizer(
        questions,
        contexts,
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        return_tensors="tf",
    )
    x1 = encodings["input_ids"]
    x2 = encodings["token_type_ids"]
    x3 = encodings["attention_mask"]
    return x1, x2, x3


def getSanInstanceResults(list_san_ins, verbose=True):
    if verbose:
        print("Tokenizing short‑answer instances")
    x1, x2, x3 = preprocess_san(list_san_ins)

    strategy = get_strategy()
    model_name = os.path.join(BASE_INPUT, "tensorflow-question-answer-fine-data")
    with strategy.scope():
        model = build_model(model_name)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(5e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        model.evaluate(
            [x1[:1], x2[:1], x3[:1]],
            tf.convert_to_tensor([[0, 0, 0]]),
            verbose=0,
        )

        try:
            model.load_weights(
                os.path.join(BASE_INPUT, "model1", "weights-02-0.701.h5")
            )
            print("Loaded pretrained short‑answer weights")
        except Exception as e:
            print("Short‑answer weights not loaded:", e)

        preds = model.predict([x1, x2, x3], batch_size=256, verbose=0)
        preds = tf.nn.softmax(preds).numpy()
    return preds


def mergeSanInstanceResult(san_res, list_san_ins, debug=False):
    for i, ins in enumerate(list_san_ins):
        start = int(np.argmax(san_res[i][0]))
        stop = int(np.argmax(san_res[i][1]))
        target = int(np.argmax(san_res[i][2]))
        ins.update(
            {
                "start": start,
                "stop": stop,
                "target": AnswerTypeRev[target],
                "start_score": float(san_res[i][0][start]),
                "stop_score": float(san_res[i][1][stop]),
                "target_score": float(san_res[i][2][target]),
                "start_CLS": float(san_res[i][0][0]),
                "stop_CLS": float(san_res[i][1][0]),
            }
        )
        if debug:
            ctx = ins["context"].split()
            if start > 0 and stop > start:
                print(
                    AnswerTypeRev[target],
                    ":",
                    ins["question"],
                    ":",
                    " ".join(ctx[start : stop + 1]),
                )
    return list_san_ins


def mergeSanLan(doc_lan_df, san_ins_res_df, verbose=True):
    SAN_STRIDE = 384
    results = []
    if verbose:
        print("Merging short‑answer results back to documents")
    san_group = san_ins_res_df.groupby("example_id")
    for _, row in doc_lan_df.iterrows():
        doc_id = row["example_id"]
        lan_start, lan_stop = row["start_token"], row["stop_token"]
        lan_target = row["target"]
        if doc_id not in san_group.groups:
            san_subset = pd.DataFrame()
        else:
            san_subset = san_group.get_group(doc_id)

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

        if lan_target not in ("NO_ANSWER", "SHORT", "LONG"):
            if final_target in ("NO_ANSWER", "LONG"):
                final_target = lan_target

        results.append(
            {
                "example_id": doc_id,
                "question": row["question"],
                "raw_document": row["raw_document"],
                "lan_start": lan_start,
                "lan_stop": lan_stop,
                "san_start": san_start,
                "san_stop": san_stop,
                "target": final_target,
            }
        )
    return results


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
    san_res_df = pd.DataFrame(list_san_res)

    return mergeSanLan(doc_res_lan_df, san_res_df)


def getLines(san_lan_doc):
    """
    Build submission rows.
    Long answer spans are always written.
    Short answers are now written when predicted:
        - YES / NO are written directly.
        - Span predictions are written as start:stop.
        - Otherwise the field is left empty.
    Additionally, when a long answer is predicted but no short span is,
    the long span is reused for the short field to give a reasonable guess.
    """
    lines = []
    for doc in san_lan_doc:
        base_id = doc["example_id"]
        lines.append(
            {
                "example_id": f"{base_id}_long",
                "PredictionString": (
                    ""
                    if doc["lan_start"] == -1
                    else f"{doc['lan_start']}:{doc['lan_stop']}"
                ),
            }
        )
        short_pred = ""
        if doc["target"] == "YES" or doc["target"] == "NO":
            short_pred = doc["target"]
        elif (
            doc["target"] == "SHORT"
            and doc["san_start"] != -1
            and doc["san_stop"] != -1
        ):
            short_pred = f"{doc['san_start']}:{doc['san_stop']}"
        elif (
            doc["target"] == "LONG" and doc["lan_start"] != -1 and doc["lan_stop"] != -1
        ):
            short_pred = f"{doc['lan_start']}:{doc['lan_stop']}"
        lines.append({"example_id": f"{base_id}_short", "PredictionString": short_pred})
    return lines


def getSubmission():
    san_lan_doc = getFinalResult(
        os.path.join(
            BASE_INPUT, "tensorflow2-question-answering", "simplified-nq-test.jsonl"
        )
    )
    lines = getLines(san_lan_doc)
    df = pd.DataFrame(lines).sort_values("example_id")
    submission_path = os.path.join(BASE_WORKING, "submission.csv")
    df.to_csv(submission_path, index=False, columns=["example_id", "PredictionString"])
    print(f"Submission written to {submission_path}")




## === cell 2
getSubmission()
