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

0.3778966131907308

# 6. Current score

0.3197

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.3197) has done: 'The fix adds missing imports, provides a lightweight fallback implementation for `getFinalResult` that directly selects the first long‑answer candidate (avoiding the absent models and tokenizer), and keeps the rest of the pipeline unchanged. This resolves the NameErrors, ensures a valid CSV submission is written, and yields a non‑zero baseline score that moves toward the target.'
- What this solution (achieved 0.3197) has done: 'The fix avoids the TensorFlow import error by lazily importing TensorFlow inside a try/except block and providing a harmless fallback when it fails. It also improves the baseline prediction by selecting the longest long‑answer candidate instead of always the first one, which should move the score closer to the target while keeping the core logic unchanged. The script now reliably writes a correctly formatted `submission.csv`.'

# 9. Code solution

## === cell 0
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

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


def get_strategy():
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        return tf.distribute.experimental.TPUStrategy(resolver)
    except Exception:
        return tf.distribute.get_strategy()


def parseData(
    filename, drop_noanswer_rate=0.8, drop_null_instances_rate=0.5, split=1.0
):
    pass


def parse_test_file(filename):
    INSTANCE_WORDS_LEN = 500
    STRIDE = 128

    list_instances = []
    list_docs = []

    def safe_tag_prefix(idx, tag_prefix):
        if idx < len(tag_prefix):
            return tag_prefix[idx]
        return tag_prefix[-1]

    with open(filename) as f:
        for line in tqdm(f, total=num_test_samples):
            data = json.loads(line)

            example_id = data["example_id"]
            question = data["question_text"]
            doc_text_raw = clean_html(data["document_text"])
            doc_text_split = doc_text_raw.split()
            clean_doc = list(filter(("<tag>").__ne__, doc_text_split))

            tag_prefix = [0]
            for token in doc_text_split:
                tag_prefix.append(tag_prefix[-1] + (1 if token == "<tag>" else 0))

            list_candidates = data["long_answer_candidates"]
            list_new_candidates = []
            for cand in list_candidates:
                cand_start = cand["start_token"]
                cand_stop = cand["end_token"]
                new_start = cand_start - safe_tag_prefix(cand_start, tag_prefix)
                new_stop = cand_stop - safe_tag_prefix(cand_stop, tag_prefix)
                new_cand = {
                    "start_token": new_start,
                    "end_token": new_stop,
                    "top_level": cand["top_level"],
                }
                list_new_candidates.append([new_cand, cand])
            list_docs.append(
                {
                    "example_id": str(example_id),
                    "question": question,
                    "document": " ".join(clean_doc),
                    "raw_document": doc_text_raw,
                    "candidates": list_new_candidates,
                }
            )

            len_ques = len(question.split())
            part_len = INSTANCE_WORDS_LEN - len_ques
            num_ins = max(0, (len(clean_doc) - part_len) // STRIDE)
            for part_id in range(num_ins + 1):
                part_start = part_id * STRIDE
                part_end = min(len(clean_doc), part_start + part_len)
                part = clean_doc[part_start:part_end]
                part_str = " ".join(part)
                instance = {
                    "question": question,
                    "context": part_str,
                    "example_id": str(example_id),
                    "part_id": part_id,
                    "target": "NO_ANSWER",
                    "start": 0,
                    "stop": 0,
                }
                list_instances.append(instance)
    return list_instances, list_docs


def getRawAndCleanTextDocs(test_file):
    return []


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
    stride = 128
    max_vote = 0.005
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
        lan_start_raw = lan_stop_raw = -1

        for _, row in all_non_zero.iterrows():
            part_id = row["part_id"]
            part_start = part_id * stride
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
                best_score = -1
                best_match_id = None
                best_ratio = 0.0
                for cand_id, cand in enumerate(samrow["candidates"]):
                    lan_start = cand[0]["start_token"]
                    lan_stop = cand[0]["end_token"]
                    lan_len = lan_stop - lan_start + 1
                    overlap_start = max(lan_start, real_start)
                    overlap_end = min(lan_stop, real_stop)
                    inter = max(0, overlap_end - overlap_start + 1)
                    if inter > 0:
                        ratio = inter / lan_len
                        if inter > best_score:
                            best_score = inter
                            best_match_id = cand_id
                            best_ratio = ratio
                        elif inter == best_score and ratio > best_ratio:
                            best_match_id = cand_id
                            best_ratio = ratio
                if best_match_id is not None:
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


def preprocess_data(data):
    """Tokenize using the global tokenizer in batch mode and return TensorFlow tensors (including dummy y)."""
    questions = [one_sam["question"] for one_sam in data]
    contexts = [one_sam["context"] for one_sam in data]
    pairs = list(zip(questions, contexts))
    tokenized = _GLOBAL_TOKENIZER.batch_encode_plus(
        pairs,
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        return_tensors="np",
    )
    x1 = tokenized["input_ids"]
    x2 = tokenized["token_type_ids"]
    x3 = tokenized["attention_mask"]
    y = np.array(
        [
            [one_sam["start"], one_sam["stop"], AnswerType[one_sam["target"]]]
            for one_sam in data
        ],
        dtype=np.int32,
    )
    return (
        tf.constant(x1, dtype=tf.int32),
        tf.constant(x2, dtype=tf.int32),
        tf.constant(x3, dtype=tf.int32),
        tf.constant(y, dtype=tf.int32),
    )


def preprocess_data_no_y(data):
    """Tokenize test instances without constructing the unused target label array."""
    questions = [one_sam["question"] for one_sam in data]
    contexts = [one_sam["context"] for one_sam in data]
    pairs = list(zip(questions, contexts))
    tokenized = _GLOBAL_TOKENIZER.batch_encode_plus(
        pairs,
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        return_tensors="np",
    )
    x1 = tokenized["input_ids"]
    x2 = tokenized["token_type_ids"]
    x3 = tokenized["attention_mask"]
    return (
        tf.constant(x1, dtype=tf.int32),
        tf.constant(x2, dtype=tf.int32),
        tf.constant(x3, dtype=tf.int32),
    )


def preprocess_san(list_san_ins):
    """Tokenize short‑answer instances using batch mode."""
    questions = [one_sam["question"] for one_sam in list_san_ins]
    contexts = [one_sam["context"] for one_sam in list_san_ins]
    pairs = list(zip(questions, contexts))
    tokenized = _GLOBAL_TOKENIZER.batch_encode_plus(
        pairs,
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        return_tensors="np",
    )
    x1 = tokenized["input_ids"]
    x2 = tokenized["token_type_ids"]
    x3 = tokenized["attention_mask"]
    return (
        tf.constant(x1, dtype=tf.int32),
        tf.constant(x2, dtype=tf.int32),
        tf.constant(x3, dtype=tf.int32),
    )


def getRawInstanceResults(list_test, verbose=True):
    if verbose:
        print("Tokenizing test instances for the first model")
    x1, x2, x3 = preprocess_data_no_y(list_test)

    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(4)

    preds = model_raw.predict([x1, x2, x3], batch_size=256, verbose=0)
    return preds


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
        print("Tokenizing short‑answer instances and running second model")
    x1, x2, x3 = preprocess_san(list_san_ins)

    preds = model_san.predict([x1, x2, x3], batch_size=256, verbose=0)
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
    """
    Lightweight fallback: parse the test file and select the longest long‑answer candidate
    for each document. This avoids missing model/tokenizer dependencies while giving a
    slightly better baseline than always picking the first candidate.
    """
    _, doc_records = parse_test_file(f_test)

    results = []
    for doc in doc_records:
        lan_start = lan_stop = -1
        target = "NO_ANSWER"
        if doc["candidates"]:
            best_cand = max(
                doc["candidates"], key=lambda c: c[0]["end_token"] - c[0]["start_token"]
            )
            cand = best_cand[0]
            lan_start = cand["start_token"]
            lan_stop = cand["end_token"]
            target = "LONG"
        results.append(
            {
                "example_id": doc["example_id"],
                "question": doc["question"],
                "raw_document": doc["raw_document"],
                "lan_start": lan_start,
                "lan_stop": lan_stop,
                "san_start": -1,
                "san_stop": -1,
                "target": target,
            }
        )
    return results


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


try:
    import tensorflow as tf
except Exception:
    tf = None

    class _DummyModel:
        def predict(self, *args, **kwargs):
            raise RuntimeError("TensorFlow model not available in this environment.")

    model_raw = model_san = _DummyModel()
    _GLOBAL_TOKENIZER = None




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
getSubmission()
