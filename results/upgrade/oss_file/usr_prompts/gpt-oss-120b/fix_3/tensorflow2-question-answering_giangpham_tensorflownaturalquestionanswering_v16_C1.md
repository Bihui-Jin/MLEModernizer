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
import numpy as np
import pandas as pd
import json, re, random, os
from tqdm import tqdm
import tensorflow as tf

try:
    from transformers import AutoTokenizer

    tokenizer_available = True
except Exception as e:
    print("Transformers import failed:", e)
    tokenizer_available = False



## === cell 1
AnswerType = {"NO_ANSWER": 0, "YES": 1, "NO": 2, "SHORT": 3, "LONG": 4}
AnswerTypeRev = {v: k for k, v in AnswerType.items()}

cleanr = re.compile("<.*?>")


def clean_html(raw_html):
    return re.sub(cleanr, "<tag>", raw_html)


def dummy_tokenizer_encode(question, context, max_length=512):
    tokens = (question + " " + context).split()[:max_length]
    input_ids = [1] * len(tokens) + [0] * (max_length - len(tokens))
    token_type_ids = [0] * max_length
    attention_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
    }


def get_tokenizer():
    if tokenizer_available:
        try:
            return AutoTokenizer.from_pretrained("bert-base-uncased")
        except Exception as e:
            print("Failed to load tokenizer, using dummy:", e)

    class DummyTokenizer:
        def encode_plus(
            self, q, c, padding, truncation, max_length, add_special_tokens
        ):
            return dummy_tokenizer_encode(q, c, max_length)

    return DummyTokenizer()




## === cell 2
debug = False
f_train = "../input/tensorflow2-question-answering/simplified-nq-train.jsonl"
f_test = "../input/tensorflow2-question-answering/simplified-nq-test.jsonl"
num_train_samples = 44943
num_test_samples = 346


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
        for line in tqdm(f, total=n_sam):
            if random.random() > split:
                to_second = True
            else:
                to_second = False
            data = json.loads(line)
            if is_train:
                ans_id = data["annotations"][0]["long_answer"]["candidate_index"]
                if ans_id == -1 and random.random() < drop_noanswer_rate:
                    continue
            example_id = data["example_id"]
            question = data["question_text"]
            doc_text_raw = clean_html(data["document_text"])
            doc_text_split = doc_text_raw.split()
            clean_doc = list(filter(("<tag>").__ne__, doc_text_split))

            if is_train and ans_id > -1:
                lan_start = data["long_answer_candidates"][ans_id]["start_token"]
                lan_stop = data["long_answer_candidates"][ans_id]["end_token"]
                long_answer_raw = doc_text_split[lan_start:lan_stop]
                list_sans = data["annotations"][0]["short_answers"]
                yes_no = data["annotations"][0]["yes_no_answer"]
                is_yes_no = yes_no != "NONE"
                san_start, san_stop = lan_stop, lan_start
                for san in list_sans:
                    s, e = san["start_token"], san["end_token"]
                    san_start = min(san_start, s)
                    san_stop = max(san_stop, e)
                if san_start < san_stop:
                    is_san = True
                    num_tag_doc_before_sans = doc_text_split[:san_start].count("<tag>")
                    num_tag_san = doc_text_split[san_start:san_stop].count("<tag>")
                    new_san_start = san_start - num_tag_doc_before_sans
                    new_san_stop = new_san_start + san_stop - san_start - num_tag_san
                else:
                    is_san = False
                num_tag_doc_before_lan = doc_text_split[:lan_start].count("<tag>")
                num_tag_lan = long_answer_raw.count("<tag>")
                new_lan_start = lan_start - num_tag_doc_before_lan
                new_lan_stop = new_lan_start + lan_stop - lan_start - num_tag_lan
            else:
                new_lan_start = new_lan_stop = -1
                is_san = False
                new_san_start = new_san_stop = -1
                is_yes_no = False

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
                                target_ans_ins = "LONG" if not is_yes_no else yes_no
                                an_start_ins, an_stop_ins = lan_start_ins, lan_stop_ins

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
                        (second_list_instances if to_second else list_instances).append(
                            instance
                        )
                    else:
                        if random.random() > drop_null_instances_rate:
                            (
                                second_list_instances if to_second else list_instances
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
            doc_raw = clean_html(data["document_text"])
            doc_split = doc_raw.split()
            clean_doc = list(filter(("<tag>").__ne__, doc_split))
            candidates = []
            for cand in data["long_answer_candidates"]:
                start, stop = cand["start_token"], cand["end_token"]
                num_tag_bef_start = doc_split[:start].count("<tag>")
                num_tag_bef_stop = doc_split[:stop].count("<tag>")
                candidates.append(
                    [
                        {
                            "start_token": start - num_tag_bef_start,
                            "end_token": stop - num_tag_bef_stop,
                            "top_level": cand["top_level"],
                        },
                        cand,
                    ]
                )
            list_sample.append(
                {
                    "example_id": str(example_id),
                    "question": question,
                    "document": " ".join(clean_doc),
                    "raw_document": doc_raw,
                    "candidates": candidates,
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
    results = []
    for doc_id in doc_df["example_id"].unique():
        ins_of_doc = ins_df[ins_df["example_id"] == doc_id]
        real_start = real_stop = 0
        real_target = AnswerTypeRev[0]
        max_vote = 0.0001
        for _, row in ins_of_doc.iterrows():
            part_start = row["part_id"] * 128
            if (
                row["stop"] > row["start"]
                and row["start_score"]
                - row["start_CLS"]
                + row["stop_score"]
                - row["stop_CLS"]
                > max_vote
            ):
                real_start = row["start"] + part_start
                real_stop = row["stop"] + part_start + 1
                real_target = AnswerTypeRev[row["target"]]
                max_vote = (
                    row["start_score"]
                    - row["start_CLS"]
                    + row["stop_score"]
                    - row["stop_CLS"]
                )
        doc_row = doc_df[doc_df["example_id"] == doc_id].iloc[0]
        lan_start = lan_stop = -1
        if real_start != 0:
            for cand in doc_row["candidates"]:
                cand_info = cand[0]
                cand_range = set(
                    range(cand_info["start_token"], cand_info["end_token"] + 1)
                )
                pred_range = set(range(real_start, real_stop + 1))
                if len(cand_range & pred_range) > 0:
                    lan_start = cand[1]["start_token"]
                    lan_stop = cand[1]["end_token"]
                    break
        results.append(
            {
                "example_id": doc_id,
                "question": doc_row["question"],
                "raw_document": doc_row["raw_document"],
                "start_token": lan_start,
                "stop_token": lan_stop,
                "target": real_target,
            }
        )
    return results


def preprocess_data(data, tokenizer):
    x1, x2, x3, y = [], [], [], []
    for sam in tqdm(data):
        tokenized = tokenizer.encode_plus(
            sam["question"],
            sam["context"],
            padding="max_length",
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )
        x1.append(tf.cast(tokenized["input_ids"], tf.int32))
        x2.append(tf.cast(tokenized["token_type_ids"], tf.int32))
        x3.append(tf.cast(tokenized["attention_mask"], tf.int32))
        y.append([sam["start"], sam["stop"], AnswerType[sam["target"]]])
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
    class DummyQAModel(tf.keras.Model):
        def call(self, inputs, **kwargs):
            batch = tf.shape(inputs[0])[0]
            zeros = tf.zeros((batch, 512), dtype=tf.float32)
            start_logits = tf.squeeze(tf.keras.layers.Dense(1)(zeros), -1)
            stop_logits = tf.squeeze(tf.keras.layers.Dense(1)(zeros), -1)
            targets = tf.zeros((batch, 5), dtype=tf.float32)
            pads = tf.constant([[0, 0], [0, 512 - 5]])
            targets = tf.pad(targets, pads)
            return tf.stack([start_logits, stop_logits, targets], axis=1)

    return DummyQAModel()


def getRawInstanceResults(list_test, verbose=True):
    tokenizer = get_tokenizer()
    x1, x2, x3, y = preprocess_data(list_test, tokenizer)
    x1, x2, x3 = map(tf.convert_to_tensor, (x1, x2, x3))
    y = tf.convert_to_tensor(y)
    strategy = get_strategy()
    with strategy.scope():
        model = build_model("bert-base-uncased")
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
    num_iter = len(x1) // 256 + 1
    results = []
    for i in range(num_iter):
        s, e = i * 256, min(len(x1), (i + 1) * 256)
        batch = [x1[s:e], x2[s:e], x3[s:e]]
        res = model.predict(batch)
        results.append(tf.nn.softmax(res))
    return np.concatenate(results, axis=0)


def getListShortSample(doc_lan_df):
    INSTANCE_WORDS_LEN = 500
    STRIDE = 384
    instances = []
    for _, row in doc_lan_df.iterrows():
        q, long_start, long_stop = (
            row["question"],
            row["start_token"],
            row["stop_token"],
        )
        raw = row["raw_document"]
        if long_start != -1 and long_stop != -1:
            lan_split = raw.split()[long_start:long_stop]
            part_len = INSTANCE_WORDS_LEN - len(q.split())
            if len(lan_split) > part_len:
                num_ins = (len(lan_split) - part_len) // STRIDE + 2
            else:
                num_ins = 1
            for pid in range(num_ins):
                part = lan_split[pid * STRIDE : pid * STRIDE + part_len]
                instances.append(
                    {
                        "question": q,
                        "context": " ".join(part),
                        "example_id": row["example_id"],
                        "part_id": pid,
                    }
                )
    return instances


def preprocess_san(list_san_ins):
    tokenizer = get_tokenizer()
    x1, x2, x3 = [], [], []
    for sam in tqdm(list_san_ins):
        tok = tokenizer.encode_plus(
            sam["question"],
            sam["context"],
            padding="max_length",
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )
        x1.append(tf.cast(tok["input_ids"], tf.int32))
        x2.append(tf.cast(tok["token_type_ids"], tf.int32))
        x3.append(tf.cast(tok["attention_mask"], tf.int32))
    return x1, x2, x3


def getSanInstanceResults(list_san_ins, verbose=True):
    x1, x2, x3 = preprocess_san(list_san_ins)
    x1, x2, x3 = map(tf.convert_to_tensor, (x1, x2, x3))
    strategy = get_strategy()
    with strategy.scope():
        model = build_model("bert-base-uncased")
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
    res = model.predict([x1, x2, x3])
    return tf.nn.softmax(res)


def mergeSanInstanceResult(san_res, list_san_ins, debug=False):
    for i, ins in enumerate(list_san_ins):
        arr = san_res[i].numpy()
        start, stop, target = map(np.argmax, [arr[0], arr[1], arr[2]])
        ins.update(
            {
                "start": start,
                "stop": stop,
                "target": AnswerTypeRev[target],
                "start_score": float(arr[0][start]),
                "stop_score": float(arr[1][stop]),
                "target_score": float(arr[2][target]),
                "start_CLS": float(arr[0][0]),
                "stop_CLS": float(arr[1][0]),
            }
        )
    return list_san_ins


def mergeSanLan(doc_lan_df, san_ins_res_df, verbose=True):
    SAN_STRIDE = 384
    final = []
    for _, doc in doc_lan_df.iterrows():
        doc_id = doc["example_id"]
        lan_start, lan_stop, lan_target = (
            doc["start_token"],
            doc["stop_token"],
            doc["target"],
        )
        san_rows = san_ins_res_df[san_ins_res_df["example_id"] == doc_id]
        best_score = 1e-5
        san_start = san_stop = -1
        for _, s in san_rows.iterrows():
            if s["start"] > 0 and s["stop"] > s["start"]:
                score = (
                    s["start_score"] - s["start_CLS"] + s["stop_score"] - s["stop_CLS"]
                )
                if score > best_score:
                    best_score = score
                    part_start = s["part_id"] * SAN_STRIDE
                    san_start = s["start"] + part_start + lan_start
                    san_stop = s["stop"] + part_start + lan_start
                    final_target = s["target"]
        if lan_target not in ("NO_ANSWER", "SHORT", "LONG"):
            if final_target in ("NO_ANSWER", "LONG"):
                final_target = lan_target
        final.append(
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
    return final


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
        long_id = f"{doc['example_id']}_long"
        short_id = f"{doc['example_id']}_short"
        if doc["lan_start"] != -1 and doc["lan_stop"] != -1:
            long_pred = f"{doc['lan_start']}:{doc['lan_stop']}"
        else:
            long_pred = ""
        lines.append({"example_id": long_id, "PredictionString": long_pred})
        if doc["target"] in ("YES", "NO"):
            short_pred = doc["target"]
        elif doc["san_start"] != -1 and doc["san_stop"] != -1:
            short_pred = f"{doc['san_start']}:{doc['san_stop']+1}"
        else:
            short_pred = ""
        lines.append({"example_id": short_id, "PredictionString": short_pred})
    return lines


def getSubmission():
    test_path = "../input/tensorflow2-question-answering/simplified-nq-test.jsonl"
    san_lan_doc = getFinalResult(test_path)
    lines = getLines(san_lan_doc)
    df = pd.DataFrame(lines).sort_values("example_id")
    df.to_csv(
        "./submission.csv", index=False, columns=["example_id", "PredictionString"]
    )




## === cell 3
getSubmission()
