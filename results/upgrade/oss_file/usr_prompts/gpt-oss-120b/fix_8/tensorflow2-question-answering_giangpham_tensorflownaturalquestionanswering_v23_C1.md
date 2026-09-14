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

0.4635627530364373

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
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

try:
    import tensorflow as tf
except Exception:  # pragma: no cover

    class _DummyTensor:
        @staticmethod
        def cast(x, dtype):
            return x

        @staticmethod
        def zeros(shape, dtype):
            return np.zeros(shape, dtype=dtype)

    class _DummyRandom:
        @staticmethod
        def set_seed(seed):
            pass

    class _DummyKeras:
        class losses:
            @staticmethod
            def sparse_categorical_crossentropy(*args, **kwargs):
                pass

    class _DummyDistribute:
        @staticmethod
        def get_strategy():
            return None

    class _DummyTF:
        random = _DummyRandom()
        cast = _DummyTensor.cast
        zeros = _DummyTensor.zeros
        keras = _DummyKeras()
        distribute = _DummyDistribute()

        @staticmethod
        def config():
            class _Config:
                @staticmethod
                def experimental_connect_to_cluster(*args, **kwargs):
                    pass

            return _Config()

    tf = _DummyTF()

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
f_test = "../input/tensorflow2-question-answering/simplified-nq-test.jsonl"
num_test_samples = 346  # limit to a manageable subset for the timeout

test_data = []
with open(f_test, "r") as f:
    for i, line in enumerate(tqdm(f, desc="Loading test JSONL")):
        if i >= num_test_samples:
            break
        test_data.append(json.loads(line))




## === cell 2
def get_id_df(data_list=test_data):
    list_id = [{"example_id": str(item["example_id"])} for item in data_list]
    return pd.DataFrame(list_id)




## === cell 3
AnswerType = {"NO_ANSWER": 0, "YES": 1, "NO": 2, "SHORT": 3, "LONG": 4}
AnswerTypeRev = {0: "NO_ANSWER", 1: "YES", 2: "NO", 3: "SHORT", 4: "LONG"}




## === cell 4
def preprocess_data_batch(data, tokenizer, debug=False):
    """
    Batch‑encode questions and contexts. Returns TensorFlow tensors directly
    (removing the extra NumPy conversion that caused unnecessary overhead).
    """
    questions = [sam["question"] for sam in data]
    contexts = [sam["context"] for sam in data]

    encodings = tokenizer.batch_encode_plus(
        list(zip(questions, contexts)),
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        return_tensors="tf",  # return TF tensors directly
    )
    x1 = tf.cast(encodings["input_ids"], tf.int32)
    x2 = tf.cast(encodings["token_type_ids"], tf.int32)
    x3 = tf.cast(encodings["attention_mask"], tf.int32)

    y = tf.zeros([len(data), 3], dtype=tf.int32)

    return x1, x2, x3, y




## === cell 5
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




## === cell 6
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




## === cell 7
def mergeDocumentRes(ins_df, val_id_df, threshold=0.0001, stride=128, debug=False):
    """
    Vectorized version: group instances by document, then compute the best start/stop
    using pandas operations to avoid Python‑level loops over rows.
    """
    STRIDE = stride
    ins_df = ins_df.copy()
    ins_df["real_start"] = ins_df["start"] + ins_df["part_start"]
    ins_df["real_stop"] = ins_df["stop"] + ins_df["part_start"]

    cond = (ins_df["start"] != 0) | (ins_df["stop"] != 0)
    ins_df = ins_df[cond]

    results = []
    for doc_id, group in ins_df.groupby("example_id"):
        best_start = -1
        best_stop = -1
        best_target = 0
        best_score = threshold

        scores = (
            group["start_score"]
            - group["start_CLS"]
            + group["stop_score"]
            - group["stop_CLS"]
        )
        for idx, row in group.iterrows():
            if row["real_stop"] > row["real_start"]:
                if scores.loc[idx] > best_score:
                    best_score = scores.loc[idx]
                    best_start = int(row["real_start"])
                    best_stop = int(row["real_stop"])
                    best_target = int(row["target"])

        results.append(
            {
                "example_id": doc_id,
                "start": best_start,
                "stop": best_stop,
                "target": best_target,
                "score": best_score,
            }
        )
    return pd.DataFrame(results)




## === cell 8
cleanr = re.compile("<.*?>")


def clean_html(raw_html):
    return re.sub(cleanr, "<tag>", raw_html)


def parseDataClean(
    data_list=test_data,
    is_val=True,
    drop_noanswer_rate=0.95,
    drop_null_instances_rate=0.98,
    debug=False,
):
    INSTANCE_WORDS_LEN = 500
    STRIDE = 128
    list_instances = []

    for data in tqdm(data_list, desc="Creating instances"):
        example_id = str(data["example_id"])
        doc_text_raw = data["document_text"]
        doc_text_tag = clean_html(doc_text_raw)
        doc_tag_split = doc_text_tag.split()
        clean_doc = list(filter(("<tag>").__ne__, doc_tag_split))

        question = data["question_text"]
        len_ques = len(question.split())
        part_len = INSTANCE_WORDS_LEN - len_ques

        num_ins = (len(clean_doc) - part_len) // STRIDE + 1
        for part_id in range(num_ins + 1):
            part_start = part_id * STRIDE
            part_stop = min(len(clean_doc), part_id * STRIDE + part_len)
            part = " ".join(clean_doc[part_start:part_stop])
            list_instances.append(
                {
                    "example_id": example_id,
                    "part_start": part_start,
                    "part_stop": part_stop,
                    "question": question,
                    "context": part,
                    "start": 0,
                    "stop": 0,
                    "target": "NO_ANSWER",
                }
            )
    return list_instances




## === cell 9
def getMapping(set_id, data_list=test_data):
    list_cand_maps = []
    for data in tqdm(data_list, desc="Mapping candidates"):
        example_id = str(data["example_id"])
        if example_id in set_id:
            doc_text_raw = clean_html(data["document_text"])
            doc_text_split = doc_text_raw.split()
            clean_doc = list(filter(("<tag>").__ne__, doc_text_split))

            list_new_candidates = []
            for cand in data["long_answer_candidates"]:
                cand_start = cand["start_token"]
                cand_stop = cand["end_token"]

                num_tag_bef_start = doc_text_split[0:cand_start].count("<tag>")
                num_tag_bef_stop = doc_text_split[0:cand_stop].count("<tag>")

                new_start = cand_start - num_tag_bef_start
                new_stop = cand_stop - num_tag_bef_stop

                list_new_candidates.append(
                    {"start_token": new_start, "end_token": new_stop}
                )
            list_cand_maps.append(
                {
                    "example_id": example_id,
                    "new_candidates": list_new_candidates,
                    "old_candidates": data["long_answer_candidates"],
                }
            )
    return list_cand_maps




## === cell 10
def getRawInstanceResults(list_test, verbose=True, debug=False):
    """
    Return dummy zero predictions of shape (len(list_test), 3, 512) so that the
    downstream pipeline can run without actual model weights.
    """
    if verbose:
        print("Generating dummy raw results for", len(list_test), "instances")
    dummy_res = np.zeros((len(list_test), 3, 512), dtype=np.float32)
    return dummy_res




## === cell 11
def getSubmissionLan(doc_res_df, doc_cand_df, threshold=0.0001, debug=False):
    if doc_res_df.empty:
        lines = []
        for _, doc in doc_cand_df.iterrows():
            long_id = f"{doc['example_id']}_long"
            line_long = {"example_id": long_id}
            if doc["old_candidates"]:
                cand = doc["old_candidates"][0]
                lan_start = cand["start_token"]
                lan_stop = cand["end_token"]
                line_long["PredictionString"] = f"{lan_start}:{lan_stop}"
            else:
                line_long["PredictionString"] = ""
            lines.append(line_long)
        return pd.DataFrame(lines).sort_values("example_id")

    doc_res_df = doc_res_df.astype({"example_id": str})
    doc_cand_df = doc_cand_df.astype({"example_id": str})

    combine_df = pd.merge(doc_res_df, doc_cand_df, on="example_id")
    lines = []
    for _, doc in combine_df.iterrows():
        long_id = f"{doc['example_id']}_long"
        line_long = {"example_id": long_id}
        an_start, an_stop, an_target = (
            int(doc["start"]),
            int(doc["stop"]),
            doc["target"],
        )
        lan_start, lan_stop = -1, -1
        if an_start > 0 and an_stop > 0:
            candidates = doc["new_candidates"]
            an_range = set(range(an_start, an_stop + 1))
            best_inter, shortest, best_id = 0.5, float("inf"), 0
            for cidx, cand in enumerate(candidates):
                c_range = set(range(cand["start_token"], cand["end_token"] + 1))
                inter = len(an_range & c_range)
                if inter > best_inter or (
                    inter == best_inter and len(c_range) < shortest
                ):
                    best_inter, shortest, best_id = inter, len(c_range), cidx
            real_candidates = doc["old_candidates"]
            lan_start = real_candidates[best_id]["start_token"]
            lan_stop = real_candidates[best_id]["end_token"]
        line_long["PredictionString"] = (
            f"{lan_start}:{lan_stop}"
            if lan_start > 0 and lan_stop > 0 and an_target != 0
            else ""
        )
        lines.append(line_long)
    return pd.DataFrame(lines).sort_values("example_id")




## === cell 12
def getSanCandidate(sub, data_list=test_data, debug=False):
    INSTANCE_WORDS_LEN = 500
    STRIDE = 256

    df_lan = pd.DataFrame(
        [
            {
                "example_id": str(row["example_id"]).replace("_long", ""),
                "lan_start": (
                    int(row["PredictionString"].split(":")[0])
                    if row["PredictionString"]
                    else -1
                ),
                "lan_stop": (
                    int(row["PredictionString"].split(":")[1])
                    if row["PredictionString"]
                    else -1
                ),
            }
            for _, row in sub.iterrows()
        ]
    )
    lan_lookup = df_lan.set_index("example_id").to_dict("index")

    list_san_ins = []
    for data in tqdm(data_list, desc="Generating short‑answer instances"):
        example_id = str(data["example_id"])
        if example_id not in lan_lookup:
            continue
        lan_info = lan_lookup[example_id]
        lan_start, lan_stop = lan_info["lan_start"], lan_info["lan_stop"]
        doc_text_split = data["document_text"].split()
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
                num_parts = (lan_stop - lan_start - INSTANCE_WORDS_LEN) // STRIDE + 1
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




## === cell 13
def getSanRawRes(list_san_ins, verbose=1, debug=False):
    """
    Return dummy zero predictions for short‑answer instances.
    """
    if verbose:
        print(
            "Generating dummy short‑answer raw results for",
            len(list_san_ins),
            "instances",
        )
    dummy_res = np.zeros((len(list_san_ins), 3, 512), dtype=np.float32)
    return dummy_res




## === cell 14
def getSanSubmission(doc_res_df, threshold=0.0001, debug=False):
    if doc_res_df.empty:
        return pd.DataFrame(columns=["example_id", "PredictionString"])

    doc_res_df = doc_res_df.astype({"example_id": str})
    lines = []
    for _, doc in doc_res_df.iterrows():
        short_id = f"{doc['example_id']}_short"
        line_short = {"example_id": short_id}
        an_start, an_stop, an_target = (
            int(doc["start"]),
            int(doc["stop"]),
            int(doc["target"]),
        )
        if an_start > 0 and an_stop > 0 and an_target != 4:
            short_string = f"{an_start}:{an_stop}"
        elif an_target in (1, 2):
            short_string = AnswerTypeRev[an_target]
        else:
            short_string = ""
        line_short["PredictionString"] = short_string
        lines.append(line_short)
    return pd.DataFrame(lines).sort_values("example_id")




## === cell 15
list_id_df = get_id_df()



## === cell 16
set_id = set(list_id_df["example_id"].values.tolist())
lan_map = getMapping(set_id)



## === cell 17
list_mappings_df = pd.DataFrame(lan_map)



## === cell 18
list_all_ins = parseDataClean()
all_ins_res = getRawInstanceResults(list_all_ins)



## === cell 19
list_fine_res_all_ins = mergeInstanceResult(all_ins_res, list_all_ins)
fine_res_all_ins_df = pd.DataFrame(list_fine_res_all_ins)



## === cell 20
docAnsDf = mergeDocumentRes(fine_res_all_ins_df, list_id_df)



## === cell 21
subLan = getSubmissionLan(docAnsDf, list_mappings_df)



## === cell 22
list_san_ins = getSanCandidate(subLan)



## === cell 23
sanRawRes = getSanRawRes(list_san_ins)



## === cell 24
list_fine_res_san_ins = mergeInstanceResult(sanRawRes, list_san_ins)
fine_res_san_ins_df = pd.DataFrame(list_fine_res_san_ins)



## === cell 25
docSanAnsDf = mergeDocumentRes(fine_res_san_ins_df, list_id_df)



## === cell 26
subSan = getSanSubmission(docSanAnsDf)



## === cell 27
sub = pd.concat([subLan, subSan])
sub_sorted = sub.sort_values("example_id")



## === cell 28
if sub_sorted.empty:
    base_ids = list_id_df["example_id"].unique()
    rows = []
    for eid in base_ids:
        rows.append({"example_id": f"{eid}_long", "PredictionString": ""})
        rows.append({"example_id": f"{eid}_short", "PredictionString": ""})
    sub_sorted = pd.DataFrame(rows)

sub_sorted.to_csv(
    "./submission.csv", index=False, columns=["example_id", "PredictionString"]
)

## --- ERROR in outputing the csv:
Invalid submission: Submission length 346 != 2 * answers length 30738
