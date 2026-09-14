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

0.03143

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.57117) has done: 'The script now avoids importing TensorFlow (which caused the protobuf error), skips loading the huge JSONL test file, and builds the required ID list directly from the provided sample submission. It then creates a submission file containing all expected rows with empty predictions, ensuring the submission length matches the competition’s requirement.'
- What this solution (achieved 0.35197) has done: 'The changes add the missing imports, define the global `test_data` list, and fix the workflow so a valid submission CSV is produced.  
We keep the original placeholder model functions unchanged (they are never called) and replace the final cell with logic that builds long‑answer predictions from the first candidate (if any) and leaves short answers blank, then writes the required `submission.csv`. This resolves the NameError issues and creates a correctly‑formatted submission file, moving the solution toward the target score.'
- What this solution (achieved 0.03143) has done: 'We fix the error by returning a DataFrame from `getMapping` instead of a list, so `getSubmissionLan` can iterate over it. In the empty‑result branch we also choose the longest candidate (most tokens) rather than always the first one, giving a slightly better baseline without altering the core logic. The rest of the pipeline remains unchanged, and a valid `submission.csv` is written.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import json
import re
from tqdm import tqdm

sample_sub_path = "../input/tensorflow2-question-answering/sample_submission.csv"
sub_template = pd.read_csv(sample_sub_path, dtype=str)

test_data = []




## === cell 1
def get_id_df():
    """
    Create a DataFrame of unique base example IDs by stripping the
    '_long' / '_short' suffixes from the sample submission IDs.
    """
    ids = (
        sub_template["example_id"]
        .str.replace("_long", "")
        .str.replace("_short", "")
        .unique()
    )
    return pd.DataFrame({"example_id": ids})




## === cell 2
AnswerType = {"NO_ANSWER": 0, "YES": 1, "NO": 2, "SHORT": 3, "LONG": 4}
AnswerTypeRev = {0: "NO_ANSWER", 1: "YES", 2: "NO", 3: "SHORT", 4: "LONG"}




## === cell 3
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




## === cell 4
def get_strategy():
    """
    Simplified strategy getter – for environments without TPUs or TensorFlow,
    just return the default strategy (which may be None).
    """
    try:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        print(
            "Running on TPU ",
            tpu_cluster_resolver.cluster_spec().as_dict()["worker"],
        )
        tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
        strategy = tf.distribute.experimental.TPUStrategy(tpu_cluster_resolver)
    except Exception as e:
        print(e)
        print("No TPU detected or TensorFlow unavailable")
        strategy = tf.distribute.get_strategy()
    return strategy




## === cell 5
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




## === cell 6
cleanr = re.compile("<.*?>")


def clean_html(raw_html):
    return re.sub(cleanr, "<tag>", raw_html)




## === cell 7
def getMapping(set_id, test_path="../input/simplified-nq-test.jsonl"):
    """
    Load long‑answer candidate mappings only for the IDs needed for submission.
    This avoids loading the entire 15 GB training file.
    Returns a DataFrame with columns:
        example_id, new_candidates, old_candidates
    """
    list_cand_maps = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Mapping candidates"):
            data = json.loads(line)
            example_id = str(data["example_id"])
            if example_id in set_id:
                doc_text_raw = clean_html(data["document_text"])
                doc_text_split = doc_text_raw.split()
                clean_doc = [tok for tok in doc_text_split if tok != "<tag>"]
                list_new_candidates = []
                for cand in data.get("long_answer_candidates", []):
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
                        "old_candidates": data.get("long_answer_candidates", []),
                    }
                )
    return pd.DataFrame(list_cand_maps)




## === cell 8
def getRawInstanceResults(list_test, verbose=True, debug=False):
    """
    Return dummy zero predictions of shape (len(list_test), 3, 512) so that the
    downstream pipeline can run without actual model weights.
    """
    if verbose:
        print("Generating dummy raw results for", len(list_test), "instances")
    dummy_res = np.zeros((len(list_test), 3, 512), dtype=np.float32)
    return dummy_res




## === cell 9
def getSubmissionLan(doc_res_df, doc_cand_df, threshold=0.0001, debug=False):
    if doc_res_df.empty:
        lines = []
        for _, doc in doc_cand_df.iterrows():
            long_id = f"{doc['example_id']}_long"
            line_long = {"example_id": long_id}
            if doc["old_candidates"]:
                longest = max(
                    doc["old_candidates"],
                    key=lambda c: c["end_token"] - c["start_token"],
                )
                lan_start = longest["start_token"]
                lan_stop = longest["end_token"]
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




## === cell 10
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




## === cell 11
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




## === cell 12
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




## === cell 13
list_id_df = get_id_df()



## === cell 14
set_id = set(list_id_df["example_id"].values.tolist())
lan_map = getMapping(set_id)  # returns a DataFrame now




## === cell 15
def parseDataClean():
    return []


list_all_ins = parseDataClean()
all_ins_res = getRawInstanceResults(list_all_ins)



## === cell 16
list_fine_res_all_ins = mergeInstanceResult(all_ins_res, list_all_ins)
fine_res_all_ins_df = pd.DataFrame(list_fine_res_all_ins)



## === cell 17
empty_res_df = pd.DataFrame(columns=["example_id", "start", "stop", "target"])
long_sub = getSubmissionLan(empty_res_df, lan_map)

short_sub = long_sub.copy()
short_sub["example_id"] = short_sub["example_id"].str.replace("_long", "_short")

submission_df = pd.concat([long_sub, short_sub], ignore_index=True)
submission_df = submission_df[["example_id", "PredictionString"]].sort_values(
    "example_id"
)

submission_path = "./submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with {len(submission_df)} rows.")



## === cell 18
pass



## === cell 19
pass



## === cell 20
pass



## === cell 21
pass



## === cell 22
pass



## === cell 23
pass



## === cell 24
pass



## === cell 25
pass
