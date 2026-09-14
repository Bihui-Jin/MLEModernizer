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

0.3773075191355245

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
import json
import random
import re
import os

try:
    import tensorflow as tf
except Exception:  # pragma: no cover

    class _TFStub:
        @staticmethod
        def zeros(shape, dtype=None):
            return np.zeros(shape, dtype=dtype)

        @staticmethod
        def constant(value):
            return np.array(value)

        @staticmethod
        def shape(tensor):
            return tensor.shape

        @staticmethod
        def stack(arrays, axis=0):
            return np.stack(arrays, axis=axis)

    tf = _TFStub()


class SimpleTokenizer:
    def __init__(self, max_length=512):
        self.max_length = max_length

    def batch_encode_plus(
        self,
        pairs,
        padding="max_length",
        truncation=True,
        max_length=None,
        add_special_tokens=True,
        return_tensors=None,
    ):
        batch = len(pairs)
        max_len = max_length or self.max_length
        zeros = tf.zeros([batch, max_len], dtype=np.int32)
        return {
            "input_ids": zeros,
            "token_type_ids": zeros,
            "attention_mask": zeros,
        }

    def add_special_tokens(self, *args, **kwargs):
        pass


tokenizer = SimpleTokenizer(max_length=512)
tokenizer.add_special_tokens({"unk_token": "<tag>"})

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


_test_json_cache = None  # caches parsed JSON lines from the test file
_model_cache = {}  # not used – kept for compatibility


def _load_test_jsons():
    global _test_json_cache
    if _test_json_cache is not None:
        return _test_json_cache
    n_sam = num_test_samples if not debug else 2
    samples = []
    with open(f_test) as f:
        for i, line in enumerate(f):
            if i >= n_sam:
                break
            samples.append(json.loads(line))
    _test_json_cache = samples
    return samples


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

    processed = 0
    if not is_train:
        json_lines = _load_test_jsons()
    with open(filename) as f:
        for line in f:
            if processed >= n_sam:
                break
            processed += 1

            to_second_list = False
            if random.uniform(0, 1) > split:
                to_second_list = True

            if not is_train:
                data = json_lines[processed - 1]
            else:
                data = json.loads(line)

            if is_train:
                ans_id = data["annotations"][0]["long_answer"]["candidate_index"]
                if ans_id == -1 and random.uniform(0, 1) < drop_noanswer_rate:
                    continue

            example_id = data["example_id"]
            question = data["question_text"]
            doc_text_raw = clean_html(data["document_text"])
            doc_text_split = doc_text_raw.split()

            if is_train:
                if ans_id > -1:
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
                        if san_start > this_start:
                            san_start = this_start
                        if san_stop < this_stop:
                            san_stop = this_stop

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

            clean_doc = list(filter(("<tag>").__ne__, doc_text_split))
            len_ques = len(question.split())
            part_len = INSTANCE_WORDS_LEN - len_ques
            num_ins = (len(clean_doc) - part_len) // STRIDE
            for part_id in range(num_ins + 1):
                part_start = part_id * STRIDE
                part_end = min(len(clean_doc), part_id * STRIDE + part_len)
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
                                target_ans_ins = "LONG"
                                an_start_ins, an_stop_ins = lan_start_ins, lan_stop_ins
                                if is_yes_no:
                                    target_ans_ins = yes_no
                        else:
                            continue

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
                            (
                                second_list_instances
                                if to_second_list
                                else list_instances
                            ).append(instance)
                else:
                    list_instances.append(instance)

    return list_instances, second_list_instances


def getRawAndCleanTextDocs(test_file):
    debug_local = False
    n_sam = 346 if not debug_local else 2
    list_sample = []

    json_lines = _load_test_jsons()
    processed = 0
    for data in json_lines:
        if processed >= n_sam:
            break
        processed += 1
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
        start = int(np.argmax(ins_res[0]))
        stop = int(np.argmax(ins_res[1]))
        target = int(np.argmax(ins_res[2]))
        list_test_ins[i]["start"] = start
        list_test_ins[i]["stop"] = stop
        list_test_ins[i]["target"] = target
        list_test_ins[i]["start_score"] = float(ins_res[0][start])
        list_test_ins[i]["stop_score"] = float(ins_res[1][stop])
        list_test_ins[i]["target_score"] = float(ins_res[2][target])
        list_test_ins[i]["start_CLS"] = float(ins_res[0][0])
        list_test_ins[i]["stop_CLS"] = float(ins_res[1][0])
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
        max_vote = 1e-4

        for _, row in all_non_zero.iterrows():
            part_id = row["part_id"]
            part_start = part_id * 128
            start = row["start"]
            stop = row["stop"]
            if stop > start:
                if row["start_score"] - row["start_CLS"] > max_vote:
                    real_start = start + part_start
                    real_stop = stop + part_start + 1
                    real_target = AnswerTypeRev[row["target"]]
                    max_vote = row["start_score"] - row["start_CLS"]

        sam = doc_df.loc[doc_df["example_id"] == doc_id]
        for _, samrow in sam.iterrows():
            if real_start != 0:
                list_cands = samrow["candidates"]
                real_range = set(range(real_start, real_stop + 1))
                best_score = best_ratio = best_match_id = 0
                for cand_id, cand in enumerate(list_cands):
                    lan_start = cand[0]["start_token"]
                    lan_stop = cand[0]["end_token"]
                    lan_range = set(range(lan_start, lan_stop + 1))
                    inter = len(real_range & lan_range)
                    ratio = inter / len(lan_range) if lan_range else 0
                    if inter > best_score or (
                        inter == best_score and ratio > best_ratio
                    ):
                        best_score, best_ratio, best_match_id = inter, ratio, cand_id
                lan_raw_res = list_cands[best_match_id][1]
                lan_start_raw = lan_raw_res["start_token"]
                lan_stop_raw = lan_raw_res["end_token"]
            else:
                lan_start_raw = -1
                lan_stop_raw = -1

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


def _dummy_predict(num_instances):
    """Zero logits matching the shape expected downstream."""
    return np.zeros((num_instances, 3, 512), dtype=np.float32)


def getRawInstanceResults(list_test, verbose=True):
    if verbose:
        print("Generating dummy predictions for long‑answer instances")
    num_instances = len(list_test)
    return _dummy_predict(num_instances)


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


def getSanInstanceResults(list_san_ins, verbose=True):
    if verbose:
        print("Generating dummy predictions for short‑answer instances")
    num_instances = len(list_san_ins)
    return _dummy_predict(num_instances)


def mergeSanInstanceResult(san_res, list_san_ins, debug=False):
    for i, ins_res in enumerate(san_res):
        start = int(np.argmax(ins_res[0]))
        stop = int(np.argmax(ins_res[1]))
        target = int(np.argmax(ins_res[2]))
        list_san_ins[i]["start"] = start
        list_san_ins[i]["stop"] = stop
        list_san_ins[i]["target"] = AnswerTypeRev[target]
        list_san_ins[i]["start_score"] = float(ins_res[0][start])
        list_san_ins[i]["stop_score"] = float(ins_res[1][stop])
        list_san_ins[i]["target_score"] = float(ins_res[2][target])
        list_san_ins[i]["start_CLS"] = float(ins_res[0][0])
        list_san_ins[i]["stop_CLS"] = float(ins_res[1][0])
    return list_san_ins


def mergeSanLan(doc_lan_df, san_ins_res_df, verbose=True):
    SAN_STRIDE = 384
    list_san_lan_res_doc = []
    if verbose:
        print("Merging short‑answer results with document results")
    has_example_id = "example_id" in san_ins_res_df.columns
    for _, row in doc_lan_df.iterrows():
        doc_id = row["example_id"]
        lan_start = row["start_token"]
        lan_stop = row["stop_token"]
        lan_target = row["target"]
        san_start = san_stop = -1
        final_target = AnswerTypeRev[0]
        if has_example_id:
            san_list = san_ins_res_df[san_ins_res_df["example_id"] == doc_id]
        else:
            san_list = pd.DataFrame(columns=[])
        max_score = 1e-5
        for _, san in san_list.iterrows():
            part_id = san["part_id"]
            part_start = part_id * SAN_STRIDE
            if san["start"] > 0 and san["stop"] > san["start"]:
                score_diff = san["start_score"] - san["start_CLS"]
                if score_diff > max_score:
                    san_start = san["start"] + part_start + lan_start
                    san_stop = san["stop"] + part_start + lan_start
                    max_score = score_diff
                    final_target = san["target"]
        if lan_target not in ("NO_ANSWER", "SHORT", "LONG"):
            if final_target in ("NO_ANSWER", "LONG"):
                final_target = lan_target
        list_san_lan_res_doc.append(
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
    return list_san_lan_res_doc


def getFinalResult(f_test):
    list_all_ins, _ = parseData(f_test)
    all_ins_res = getRawInstanceResults(list_all_ins)
    list_fine_res_all_ins = mergeInstanceResult(all_ins_res, list_all_ins)
    fine_res_all_ins_df = pd.DataFrame(list_fine_res_all_ins)

    list_doc = getRawAndCleanTextDocs(f_test)
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
    lines = {"example_id": [], "PredictionString": []}
    for doc in san_lan_doc:
        doc_id = doc["example_id"]
        long_id = f"{doc_id}_long"
        short_id = f"{doc_id}_short"
        lines["example_id"].extend([long_id, short_id])
        if doc["lan_start"] != -1 and doc["lan_stop"] != -1:
            lines["PredictionString"].append(f"{doc['lan_start']}:{doc['lan_stop']}")
        else:
            lines["PredictionString"].append("")
        if doc["san_start"] != -1 and doc["san_stop"] != -1:
            lines["PredictionString"].append(f"{doc['san_start']}:{doc['san_stop']}")
        else:
            lines["PredictionString"].append("")
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
    print("Submission written to ./submission.csv")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
getSubmission()

## --- ERROR in outputing the csv:
Invalid submission: Submission length 692 != 2 * answers length 30738
