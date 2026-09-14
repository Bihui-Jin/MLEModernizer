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

0.3131359851988899

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf

AnswerType = {
    "NO_ANSWER": 0,
    "LONG": 1,
    "SHORT": 2,
    "YES": 3,
    "NO": 4,
}
AnswerTypeRev = {v: k for k, v in AnswerType.items()}


def get_tokenizer():
    """Return a very simple tokenizer compatible with the code expectations.

    The dummy tokenizer implements `batch_encode_plus` and returns zero‑filled
    NumPy arrays of the correct shape (batch, 512). This is sufficient because
    the dummy model does not rely on actual token IDs.
    """

    class DummyTokenizer:
        def __init__(self, max_len=512):
            self.max_len = max_len

        def batch_encode_plus(
            self,
            pairs,
            padding,
            truncation,
            max_length,
            add_special_tokens,
            return_tensors,
        ):
            batch_size = len(pairs)
            input_ids = np.zeros((batch_size, max_length), dtype=np.int32)
            token_type_ids = np.zeros((batch_size, max_length), dtype=np.int32)
            attention_mask = np.zeros((batch_size, max_length), dtype=np.int32)
            return {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
            }

    return DummyTokenizer()


debug = False
f_train = "../input/tensorflow2-question-answering/simplified-nq-train.jsonl"
f_test = "../input/tensorflow2-question-answering/simplified-nq-test.jsonl"
num_train_samples = 44943
num_test_samples = 346


def parseData(
    filename,
    drop_noanswer_rate=0.8,
    drop_null_instances_rate=0.5,
    split=1.0,
    max_samples=None,
):
    """Parse training or test data and generate sliding‑window instances."""
    INSTANCE_WORDS_LEN = 500
    STRIDE = 128
    is_train = "train" in filename
    n_sam = num_train_samples if is_train else num_test_samples
    if debug:
        n_sam = 200 if is_train else 2
    if max_samples is None:
        max_samples = n_sam
    list_instances = []
    second_list_instances = [] if is_train else None
    with open(filename) as f:
        for line_i, line in enumerate(f):
            if line_i >= max_samples:
                break
            to_second = random.random() > split
            data = json.loads(line)
            example_id = data["example_id"]
            question = data["question_text"]
            doc_raw = clean_html(data["document_text"])
            doc_split = doc_raw.split()
            clean_doc = [tok for tok in doc_split if tok != "<tag>"]
            tag_prefix = np.cumsum([1 if t == "<tag>" else 0 for t in doc_split])
            if is_train:
                ans_id = data["annotations"][0]["long_answer"]["candidate_index"]
                if ans_id == -1 and random.random() < drop_noanswer_rate:
                    continue
            else:
                ans_id = -1  # dummy for uniform handling

            if is_train and ans_id > -1:
                lan = data["long_answer_candidates"][ans_id]
                lan_start, lan_stop = lan["start_token"], lan["end_token"]
                tags_before_lan = tag_prefix[lan_start - 1] if lan_start > 0 else 0
                tags_in_lan = (
                    tag_prefix[lan_stop - 1] - tags_before_lan if lan_stop > 0 else 0
                )
                new_lan_start = lan_start - tags_before_lan
                new_lan_stop = lan_stop - tags_before_lan - tags_in_lan

                list_sans = data["annotations"][0]["short_answers"]
                yes_no = data["annotations"][0]["yes_no_answer"]
                is_yes_no = yes_no != "NONE"
                san_start, san_stop = lan_stop, lan_start
                for san in list_sans:
                    s, e = san["start_token"], san["end_token"]
                    san_start = min(san_start, s)
                    san_stop = max(san_stop, e)
                if san_start < san_stop:
                    tags_before_san = tag_prefix[san_start - 1] if san_start > 0 else 0
                    tags_in_san = (
                        tag_prefix[san_stop - 1] - tags_before_san
                        if san_stop > 0
                        else 0
                    )
                    new_san_start = san_start - tags_before_san
                    new_san_stop = new_san_start + (san_stop - san_start) - tags_in_san
                    is_san = True
                else:
                    is_san = False
                    new_san_start = new_san_stop = -1
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

                instance = {
                    "question": question,
                    "context": " ".join(part),
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


def getRawAndCleanTextDocs(test_file, max_samples=None):
    """Load test documents and pre‑compute candidate token offsets."""
    list_sample = []
    with open(test_file) as f:
        for line_i, line in enumerate(f):
            if max_samples is not None and line_i >= max_samples:
                break
            data = json.loads(line)
            example_id = data["example_id"]
            question = data["question_text"]
            doc_raw = clean_html(data["document_text"])
            doc_split = doc_raw.split()
            clean_doc = [tok for tok in doc_split if tok != "<tag>"]
            tag_prefix = np.cumsum([1 if t == "<tag>" else 0 for t in doc_split])
            candidates = []
            for cand in data["long_answer_candidates"]:
                start, stop = cand["start_token"], cand["end_token"]
                tags_before_start = tag_prefix[start - 1] if start > 0 else 0
                tags_before_stop = tag_prefix[stop - 1] if stop > 0 else 0
                candidates.append(
                    [
                        {
                            "start_token": start - tags_before_start,
                            "end_token": stop - tags_before_stop,
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
        target_idx = int(np.argmax(ins_res[2]))
        if target_idx >= len(AnswerType):
            target_idx = AnswerType["NO_ANSWER"]
        list_test_ins[i].update(
            {
                "start": start,
                "stop": stop,
                "target": target_idx,
                "start_score": float(ins_res[0][start]),
                "stop_score": float(ins_res[1][stop]),
                "target_score": float(ins_res[2][target_idx]),
                "start_CLS": float(ins_res[0][0]),
                "stop_CLS": float(ins_res[1][0]),
            }
        )
    return list_test_ins


def mergeDocumentResult(doc_df, ins_df):
    """Vectorised majority‑vote style merging."""
    ins_df = ins_df.copy()
    ins_df["global_start"] = ins_df["start"] + ins_df["part_id"] * 128
    ins_df["global_stop"] = ins_df["stop"] + ins_df["part_id"] * 128 + 1
    ins_df["vote"] = (
        ins_df["start_score"]
        - ins_df["start_CLS"]
        + ins_df["stop_score"]
        - ins_df["stop_CLS"]
    )
    mask = (ins_df["stop"] > ins_df["start"]) & (ins_df["vote"] > 0)
    ins_df = ins_df[mask]

    best = (
        ins_df.groupby("example_id")
        .apply(lambda g: g.loc[g["vote"].idxmax()] if not g.empty else None)
        .reset_index(drop=True)
    )
    merged = pd.merge(doc_df, best, on="example_id", how="left", suffixes=("", "_best"))
    results = []
    for _, row in merged.iterrows():
        real_start = row["global_start"] if pd.notnull(row["global_start"]) else 0
        real_stop = row["global_stop"] if pd.notnull(row["global_stop"]) else 0
        real_target = (
            AnswerTypeRev[int(row["target"])]
            if pd.notnull(row["target"])
            else AnswerTypeRev[0]
        )
        lan_start = lan_stop = -1
        if real_start != 0:
            for cand in row["candidates"]:
                cand_info = cand[0]
                cand_range = set(
                    range(cand_info["start_token"], cand_info["end_token"] + 1)
                )
                pred_range = set(range(int(real_start), int(real_stop) + 1))
                if cand_range & pred_range:
                    lan_start = cand[1]["start_token"]
                    lan_stop = cand[1]["end_token"]
                    break
        results.append(
            {
                "example_id": row["example_id"],
                "question": row["question"],
                "raw_document": row["raw_document"],
                "start_token": lan_start,
                "stop_token": lan_stop,
                "target": real_target,
            }
        )
    return results


def preprocess_data(data, tokenizer):
    """Tokenize all instances in a single batch call."""
    pairs = [(sam["question"], sam["context"]) for sam in data]

    encodings = tokenizer.batch_encode_plus(
        pairs,
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        return_tensors="np",
    )
    input_ids = tf.convert_to_tensor(encodings["input_ids"], dtype=tf.int32)
    token_type_ids = tf.convert_to_tensor(encodings["token_type_ids"], dtype=tf.int32)
    attention_masks = tf.convert_to_tensor(encodings["attention_mask"], dtype=tf.int32)

    targets = [[sam["start"], sam["stop"], AnswerType[sam["target"]]] for sam in data]
    y = tf.convert_to_tensor(targets, dtype=tf.int32)

    return input_ids, token_type_ids, attention_masks, y


def get_strategy():
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        return tf.distribute.experimental.TPUStrategy(resolver)
    except Exception:
        return tf.distribute.get_strategy()


def build_model(model_name):
    """Return a dummy QA model that mimics BERT outputs without heavy computation."""

    class DummyQAModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.start_dense = tf.keras.layers.Dense(512)
            self.stop_dense = tf.keras.layers.Dense(512)
            self.target_dense = tf.keras.layers.Dense(5)  # only 5 answer types

        def call(self, inputs, **kwargs):
            batch = tf.shape(inputs[0])[0]
            rand_feat = tf.random.uniform((batch, 5), dtype=tf.float32)
            start_logits = self.start_dense(rand_feat)  # (batch, 512)
            stop_logits = self.stop_dense(rand_feat)  # (batch, 512)
            target_logits = self.target_dense(rand_feat)  # (batch, 5)
            target_padded = tf.pad(target_logits, [[0, 0], [0, 512 - 5]])
            return tf.stack(
                [start_logits, stop_logits, target_padded], axis=1
            )  # (batch, 3, 512)

    return DummyQAModel()


def getRawInstanceResults(list_test, verbose=True):
    tokenizer = get_tokenizer()
    x1, x2, x3, y = preprocess_data(list_test, tokenizer)
    strategy = get_strategy()
    with strategy.scope():
        model = build_model("bert-base-uncased")
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
    res = model.predict([x1, x2, x3], verbose=0)
    return tf.nn.softmax(res)


def getListShortSample(doc_lan_df):
    """Create short‑answer windows using vectorised operations where possible."""
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
                        "lan_start": long_start,
                        "lan_stop": long_stop,
                    }
                )
    return instances


def preprocess_san(list_san_ins):
    """Batch‑encode short‑answer windows."""
    tokenizer = get_tokenizer()
    pairs = [(sam["question"], sam["context"]) for sam in list_san_ins]

    encodings = tokenizer.batch_encode_plus(
        pairs,
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        return_tensors="np",
    )
    input_ids = tf.convert_to_tensor(encodings["input_ids"], dtype=tf.int32)
    token_type_ids = tf.convert_to_tensor(encodings["token_type_ids"], dtype=tf.int32)
    attention_masks = tf.convert_to_tensor(encodings["attention_mask"], dtype=tf.int32)

    return input_ids, token_type_ids, attention_masks


def getSanInstanceResults(list_san_ins, verbose=True):
    x1, x2, x3 = preprocess_san(list_san_ins)
    strategy = get_strategy()
    with strategy.scope():
        model = build_model("bert-base-uncased")
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
    res = model.predict([x1, x2, x3], verbose=0)
    return tf.nn.softmax(res)


def mergeSanInstanceResult(san_res, list_san_ins, debug=False):
    for i, ins in enumerate(list_san_ins):
        arr = san_res[i].numpy()
        start, stop, target_idx = map(np.argmax, [arr[0], arr[1], arr[2]])
        if target_idx >= len(AnswerType):
            target_idx = AnswerType["NO_ANSWER"]
        ins.update(
            {
                "start": start,
                "stop": stop,
                "target": AnswerTypeRev[target_idx],
                "start_score": float(arr[0][start]),
                "stop_score": float(arr[1][stop]),
                "target_score": float(arr[2][target_idx]),
                "start_CLS": float(arr[0][0]),
                "stop_CLS": float(arr[1][0]),
            }
        )
    return list_san_ins


def mergeSanLan(doc_lan_df, san_ins_res_df, verbose=True):
    """Vectorised selection of the best short‑answer candidate."""
    SAN_STRIDE = 384
    san_ins_res_df["global_start"] = (
        san_ins_res_df["start"]
        + san_ins_res_df["part_id"] * SAN_STRIDE
        + san_ins_res_df["lan_start"]
    )
    san_ins_res_df["global_stop"] = (
        san_ins_res_df["stop"]
        + san_ins_res_df["part_id"] * SAN_STRIDE
        + san_ins_res_df["lan_start"]
    )
    san_ins_res_df["score"] = (
        san_ins_res_df["start_score"]
        - san_ins_res_df["start_CLS"]
        + san_ins_res_df["stop_score"]
        - san_ins_res_df["stop_CLS"]
    )
    mask = (san_ins_res_df["stop"] > 0) & (
        san_ins_res_df["stop"] > san_ins_res_df["start"]
    )
    san_ins_res_df = san_ins_res_df[mask]

    best = (
        san_ins_res_df.groupby("example_id")
        .apply(lambda g: g.loc[g["score"].idxmax()] if not g.empty else None)
        .reset_index(drop=True)
    )
    merged = pd.merge(
        doc_lan_df, best, on="example_id", how="left", suffixes=("", "_san")
    )
    final = []
    for _, row in merged.iterrows():
        lan_target = row["target"]
        final_target = lan_target
        if pd.notnull(row["target_san"]):
            if row["target_san"] not in ("NO_ANSWER", "LONG"):
                final_target = row["target_san"]
        final.append(
            {
                "example_id": row["example_id"],
                "question": row["question"],
                "raw_document": row["raw_document"],
                "lan_start": row["start_token"],
                "lan_stop": row["stop_token"],
                "san_start": (
                    int(row["global_start"]) if pd.notnull(row["global_start"]) else -1
                ),
                "san_stop": (
                    int(row["global_stop"]) if pd.notnull(row["global_stop"]) else -1
                ),
                "target": final_target,
            }
        )
    return final


def getFinalResult(f_test):
    list_all_ins, _ = parseData(f_test, max_samples=num_test_samples)
    all_ins_res = getRawInstanceResults(list_all_ins)
    list_fine_res_all_ins = mergeInstanceResult(all_ins_res, list_all_ins)
    fine_res_all_ins_df = pd.DataFrame(list_fine_res_all_ins)
    doc_df = pd.DataFrame(getRawAndCleanTextDocs(f_test, max_samples=num_test_samples))
    doc_res_lan = mergeDocumentResult(doc_df, fine_res_all_ins_df)
    doc_res_lan_df = pd.DataFrame(doc_res_lan)
    list_san_ins = getListShortSample(doc_res_lan_df)
    if len(list_san_ins) == 0:
        return doc_res_lan
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
    print("Submission file written to ./submission.csv")


def clean_html(text):
    """Placeholder HTML cleaner – returns text unchanged for this dummy pipeline."""
    return text




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
getSubmission()

## --- ERROR in outputing the csv:
Invalid submission: Submission length 692 != 2 * answers length 30738
