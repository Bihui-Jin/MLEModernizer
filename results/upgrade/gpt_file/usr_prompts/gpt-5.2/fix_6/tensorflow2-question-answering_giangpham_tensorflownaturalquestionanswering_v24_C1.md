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

0.4644976468180888

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, subprocess, textwrap, json, re, random, shutil, string
import numpy as np
import pandas as pd

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "transformers==4.41.2",
        "protobuf==3.20.3",
    ]
)



## === cell 1
import tensorflow as tf
from tqdm import tqdm
from transformers import AutoTokenizer, TFBertModel



## === cell 2
BASE_INPUT = "/kaggle/input/tensorflow2-question-answering"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/input"

f_test = os.path.join(BASE_INPUT, "simplified-nq-test.jsonl")
f_train = os.path.join(BASE_INPUT, "simplified-nq-train.jsonl")

num_train_samples = 307372
num_test_samples = 346

assert os.path.exists(f_test), f"Missing test file at {f_test}"
assert os.path.exists(
    os.path.join(BASE_INPUT, "sample_submission.csv")
), "Missing sample_submission.csv"

tf.random.set_seed(1234)
np.random.seed(1234)
random.seed(1234)
try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass



## === cell 3
cleanr = re.compile("<.*?>")


def clean_html(raw_html):
    return re.sub(cleanr, "<tag>", raw_html)


def _safe_prefix_count(prefix_cumsum: np.ndarray, idx_exclusive: int) -> int:
    """
    Return number of tags before position idx_exclusive in the original token sequence,
    using a cumsum array over positions. Robust to idx_exclusive outside bounds.
    """
    if idx_exclusive <= 0 or prefix_cumsum.size == 0:
        return 0
    j = idx_exclusive - 1
    if j < 0:
        return 0
    if j >= prefix_cumsum.size:
        j = prefix_cumsum.size - 1
    return int(prefix_cumsum[j])


def load_test_cache(filename=f_test, verbose=True):
    """
    Returns:
      id_df: DataFrame with example_id
      clean_doc_tokens_by_id: dict[example_id] -> list[str] (tags removed)
      raw_doc_tokens_by_id: dict[example_id] -> list[str] (raw split including tags)
      question_by_id: dict[example_id] -> str
      cand_map_rows: list of dict rows as produced by getMapping()
    """
    ids = []
    clean_doc_tokens_by_id = {}
    raw_doc_tokens_by_id = {}
    question_by_id = {}
    cand_map_rows = []

    with open(filename) as f:
        it = tqdm(f, desc="Reading test jsonl") if verbose else f
        for line in it:
            data = json.loads(line)
            example_id = str(data["example_id"])
            ids.append({"example_id": example_id})

            question = data["question_text"]
            question_by_id[example_id] = question

            doc_text_raw = data["document_text"]
            doc_text_tag = clean_html(doc_text_raw)
            doc_tag_split = doc_text_tag.split()
            raw_doc_tokens_by_id[example_id] = doc_tag_split
            clean_doc = [tok for tok in doc_tag_split if tok != "<tag>"]
            clean_doc_tokens_by_id[example_id] = clean_doc

            list_candidates = data.get("long_answer_candidates", [])
            list_new_candidates = []
            tag_prefix = np.fromiter(
                (1 if t == "<tag>" else 0 for t in doc_tag_split), dtype=np.int32
            )
            tag_prefix = np.cumsum(tag_prefix)

            n_tokens = len(doc_tag_split)
            for cand in list_candidates:
                cand_start = int(cand["start_token"])
                cand_stop = int(cand["end_token"])

                if n_tokens == 0:
                    continue
                if cand_start < 0:
                    cand_start = 0
                if cand_stop < 0:
                    cand_stop = 0
                if cand_start > n_tokens and cand_stop > n_tokens:
                    continue
                cand_start_clamped = min(cand_start, n_tokens)
                cand_stop_clamped = min(cand_stop, n_tokens)

                num_tag_bef_start = _safe_prefix_count(tag_prefix, cand_start_clamped)
                num_tag_bef_stop = _safe_prefix_count(tag_prefix, cand_stop_clamped)

                new_start = cand_start_clamped - num_tag_bef_start
                new_stop = cand_stop_clamped - num_tag_bef_stop

                if new_stop < new_start:
                    continue

                list_new_candidates.append(
                    {"end_token": int(new_stop), "start_token": int(new_start)}
                )

            cand_map_rows.append(
                {
                    "example_id": example_id,
                    "new_candidates": list_new_candidates,
                    "old_candidates": list_candidates,
                }
            )

    id_df = pd.DataFrame(ids)
    return (
        id_df,
        clean_doc_tokens_by_id,
        raw_doc_tokens_by_id,
        question_by_id,
        cand_map_rows,
    )




## === cell 4
AnswerType = {
    "NO_ANSWER": 0,
    "YES": 1,
    "NO": 2,
    "SHORT": 3,
    "LONG": 4,
}

AnswerTypeRev = {
    0: "NO_ANSWER",
    1: "YES",
    2: "NO",
    3: "SHORT",
    4: "LONG",
}




## === cell 5
def preprocess_data(data, tokenizer, debug=False):
    questions = [sam["question"] for sam in data]
    contexts = [sam["context"] for sam in data]

    enc = tokenizer(
        questions,
        contexts,
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    x1 = tf.convert_to_tensor(np.asarray(enc["input_ids"], dtype=np.int32))
    x2 = tf.convert_to_tensor(
        np.asarray(
            enc.get("token_type_ids", np.zeros_like(enc["input_ids"])), dtype=np.int32
        )
    )
    x3 = tf.convert_to_tensor(np.asarray(enc["attention_mask"], dtype=np.int32))

    y = tf.convert_to_tensor(
        np.asarray(
            [[sam["start"], sam["stop"], AnswerType[sam["target"]]] for sam in data],
            dtype=np.int32,
        )
    )
    return x1, x2, x3, y




## === cell 6
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
        print(str(e))
        print("No TPU detected")
        strategy = tf.distribute.get_strategy()
    return strategy




## === cell 7
def mergeInstanceResult(test_res, list_test_ins):
    start_idx = np.argmax(test_res[:, 0, :], axis=1).astype(int)
    stop_idx = np.argmax(test_res[:, 1, :], axis=1).astype(int)
    target_idx = np.argmax(test_res[:, 2, :], axis=1).astype(int)

    row = np.arange(test_res.shape[0])
    start_score = test_res[row, 0, start_idx].astype(float)
    stop_score = test_res[row, 1, stop_idx].astype(float)
    target_score = test_res[row, 2, target_idx].astype(float)

    start_CLS = test_res[:, 0, 0].astype(float)
    stop_CLS = test_res[:, 1, 0].astype(float)

    for i in range(len(list_test_ins)):
        list_test_ins[i]["start"] = int(start_idx[i])
        list_test_ins[i]["stop"] = int(stop_idx[i])
        list_test_ins[i]["target"] = int(target_idx[i])

        list_test_ins[i]["start_score"] = float(start_score[i])
        list_test_ins[i]["stop_score"] = float(stop_score[i])
        list_test_ins[i]["target_score"] = float(target_score[i])

        list_test_ins[i]["start_CLS"] = float(start_CLS[i])
        list_test_ins[i]["stop_CLS"] = float(stop_CLS[i])
    return list_test_ins




## === cell 8
def mergeDocumentRes(ins_df, val_id_df, threshold=0.0001, stride=128, debug=False):
    grouped = {}
    for ex_id, g in ins_df.groupby("example_id", sort=False):
        grouped[str(ex_id)] = g

    list_doc_lan = []
    for idx, doc in val_id_df.iterrows():
        doc_id = str(doc["example_id"])
        g = grouped.get(doc_id, None)
        best_start = -1
        best_stop = -1
        best_target = 0
        best_score = threshold

        if g is not None and len(g) > 0:
            mask = (g["start"].to_numpy() != 0) | (g["stop"].to_numpy() != 0)
            gg = g.loc[mask]
            if len(gg) > 0:
                starts = gg["start"].to_numpy(dtype=np.int32)
                stops = gg["stop"].to_numpy(dtype=np.int32)
                targets = gg["target"].to_numpy(dtype=np.int32)
                part_starts = gg["part_start"].to_numpy(dtype=np.int32)

                real_starts = starts + part_starts
                real_stops = stops + part_starts

                s_start = gg["start_score"].to_numpy(dtype=np.float32)
                s_stop = gg["stop_score"].to_numpy(dtype=np.float32)
                cls_start = gg["start_CLS"].to_numpy(dtype=np.float32)
                cls_stop = gg["stop_CLS"].to_numpy(dtype=np.float32)

                valid = real_stops > real_starts
                if np.any(valid):
                    scores = (s_start - cls_start) + (s_stop - cls_stop)
                    scores = np.where(valid, scores, -np.inf)
                    best_i = int(np.argmax(scores))
                    if float(scores[best_i]) > best_score:
                        best_score = float(scores[best_i])
                        best_start = int(real_starts[best_i])
                        best_stop = int(real_stops[best_i])
                        best_target = int(targets[best_i])

        doc_lan = {
            "example_id": doc_id,
            "start": best_start,
            "stop": best_stop,
            "target": best_target,
            "score": best_score,
        }

        if debug and idx == 101:
            print(doc_lan)

        list_doc_lan.append(doc_lan)

    return pd.DataFrame(list_doc_lan)




## === cell 9
def get_id_df(filename=f_test):
    list_id = []
    with open(filename) as f:
        progress = tqdm(f)
        for sam_count, line in enumerate(progress):
            data = json.loads(line)
            example_id = str(data["example_id"])
            list_id.append({"example_id": example_id})
    return pd.DataFrame(list_id)




## === cell 10
def getMapping(set_id, filename=f_test):
    list_cand_maps = []
    with open(filename) as f:
        progress = tqdm(f)
        for sam_count, line in enumerate(progress):
            data = json.loads(line)
            example_id = str(data["example_id"])

            if example_id in set_id:
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

                    list_new_candidates.append(
                        {"end_token": new_stop, "start_token": new_start}
                    )

                list_cand_maps.append(
                    {
                        "example_id": str(example_id),
                        "new_candidates": list_new_candidates,
                        "old_candidates": list_candidates,
                    }
                )
    return list_cand_maps




## === cell 11
def build_model(model_name, tokenizer, debug=False):
    encoder = TFBertModel.from_pretrained(model_name, local_files_only=True)
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
                inputs[0],
                token_type_ids=inputs[1],
                attention_mask=inputs[2],
            )
            seq_out = bert_res[0]
            pooled_out = bert_res[1]

            start_logits = tf.squeeze(self.start_logits(seq_out), -1)
            stop_logits = tf.squeeze(self.stop_logits(seq_out), -1)

            targets = self.target(pooled_out)
            paddings = tf.constant([[0, 0], [0, 512 - NUM_TARGET]])
            targets = tf.pad(targets, paddings)

            res = tf.stack([start_logits, stop_logits, targets], axis=1)
            return res

    return MyQAModel()




## === cell 12
def _is_hf_model_dir(p: str) -> bool:
    if not p or not os.path.isdir(p):
        return False
    return os.path.exists(os.path.join(p, "config.json")) and (
        os.path.exists(os.path.join(p, "vocab.txt"))
        or os.path.exists(os.path.join(p, "tokenizer.json"))
        or os.path.exists(os.path.join(p, "tokenizer_config.json"))
    )


def _resolve_local_model_dir(path_like: str) -> str:
    if path_like and os.path.exists(path_like):
        return os.path.abspath(path_like)

    if path_like.startswith("../input/"):
        alt = "/kaggle/input/" + path_like[len("../input/") :]
        if os.path.exists(alt):
            return os.path.abspath(alt)

    base = os.path.basename(path_like.rstrip("/"))
    for root in [BASE_INPUT, "/kaggle/input"]:
        cand = os.path.join(root, base)
        if os.path.exists(cand):
            return os.path.abspath(cand)

    for root in [BASE_INPUT, "/kaggle/input"]:
        try:
            for d1 in os.listdir(root):
                p1 = os.path.join(root, d1)
                if not os.path.isdir(p1):
                    continue
                if os.path.basename(p1) == base:
                    return os.path.abspath(p1)
                try:
                    for d2 in os.listdir(p1):
                        p2 = os.path.join(p1, d2)
                        if os.path.isdir(p2) and os.path.basename(p2) == base:
                            return os.path.abspath(p2)
                except Exception:
                    pass
        except Exception:
            pass

    return os.path.abspath(path_like)


def _resolve_weights_path(path_like: str) -> str:
    if os.path.exists(path_like):
        return os.path.abspath(path_like)

    cands = []
    if path_like.startswith("../input/"):
        cands.append("/kaggle/input/" + path_like[len("../input/") :])

    fname = os.path.basename(path_like)
    for root in [BASE_INPUT, "/kaggle/input"]:
        cands.append(os.path.join(root, "model1", fname))
        cands.append(
            os.path.join(root, "tensorflow2-question-answering", "model1", fname)
        )
        cands.append(os.path.join(root, "tensorflow2-question-answering", fname))

    for c in cands:
        if os.path.exists(c):
            return os.path.abspath(c)

    for root in [BASE_INPUT, "/kaggle/input"]:
        for dirpath, dirnames, filenames in os.walk(root):
            if fname in filenames:
                return os.path.abspath(os.path.join(dirpath, fname))
            if dirpath.count(os.sep) - root.count(os.sep) >= 6:
                dirnames[:] = []
    return os.path.abspath(path_like)


def _pick_hf_model_dir() -> str:
    preferred = [
        BASE_INPUT,
        os.path.join(BASE_INPUT, "tensorflow2-question-answering"),
        os.path.join(
            BASE_INPUT,
            "tensorflow2-question-answering",
            "tensorflow2-question-answering",
        ),
        "/kaggle/input/tensorflow2-question-answering",
        "/kaggle/input/tensorflow2-question-answering/tensorflow2-question-answering",
    ]

    expanded = []
    for p in preferred:
        expanded.append(p)
        try:
            if os.path.isdir(p):
                for d in os.listdir(p):
                    expanded.append(os.path.join(p, d))
        except Exception:
            pass

    for p in expanded:
        if _is_hf_model_dir(p):
            return os.path.abspath(p)

    for root in [BASE_INPUT, "/kaggle/input"]:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if "config.json" in filenames and (
                "vocab.txt" in filenames
                or "tokenizer.json" in filenames
                or "tokenizer_config.json" in filenames
            ):
                return os.path.abspath(dirpath)
            if dirpath.count(os.sep) - root.count(os.sep) >= 6:
                dirnames[:] = []
    raise FileNotFoundError(
        "Could not find a local HuggingFace model directory with config/tokenizer files under /kaggle/input."
    )




## === cell 13
def getRawInstanceResults(list_test, verbose=True, debug=False):
    if verbose:
        print("Getting raw result for all the instances generated from test file")

    model_name = _pick_hf_model_dir()
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    tags = ["``", "''", "--"]
    special_tokens_dict = {"additional_special_tokens": tags}
    tokenizer.add_special_tokens(special_tokens_dict)

    x_test1, x_test2, x_test3, y_test = preprocess_data(list_test, tokenizer)
    if verbose:
        print("Finish tokenizing ", len(list_test), " data for the first model")
        print(x_test1.shape)

    if verbose:
        print("Preparing model")

    strategy = get_strategy()
    with strategy.scope():
        testModel = build_model(model_name, tokenizer)
        x = np.ones([1, 512], dtype=int)
        _ = testModel.predict([x, x, x], verbose=0)

        weights_path = _resolve_weights_path("../input/model1/weights-01.h5")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing weights file: {weights_path}")
        testModel.load_weights(weights_path)

        optAdam = tf.keras.optimizers.Adam(learning_rate=0.00005)
        lossSCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metricSCA = tf.keras.metrics.SparseCategoricalAccuracy()
        testModel.compile(optimizer=optAdam, loss=lossSCE, metrics=[metricSCA])

    if verbose:
        print("Finish loading pretrained weights for the model")

    test_res = testModel.predict([x_test1, x_test2, x_test3], verbose=1, batch_size=8)

    if verbose:
        print("Finish calculating raw result, get an array of size: ", test_res.shape)
    return test_res




## === cell 14
def getSubmissionLan(doc_res_df, doc_cand_df, threshold=0.0001, debug=False):
    doc_res_df = doc_res_df.copy()
    doc_res_df.example_id = doc_res_df.example_id.astype(str)

    cand_map = {}
    for _, r in doc_cand_df.iterrows():
        cand_map[str(r["example_id"])] = (
            r.get("new_candidates", None),
            r.get("old_candidates", None),
        )

    lines = []
    for _, doc in doc_res_df.iterrows():
        example_id = str(doc["example_id"])
        long_id = example_id + "_long"

        an_start = int(doc["start"])
        an_stop = int(doc["stop"])
        an_target = int(doc["target"])

        lan_start, lan_stop = -1, -1
        new_candidates, old_candidates = cand_map.get(example_id, (None, None))

        if an_start > 0 and an_stop > 0 and isinstance(new_candidates, list):
            best_inter = 0.5
            shortest = 10**18
            best_id = 0

            a0, a1 = an_start, an_stop
            for cidx, cand in enumerate(new_candidates):
                c_start = int(cand["start_token"])
                c_stop = int(cand["end_token"])
                if c_stop < c_start:
                    continue
                inter = max(0, min(a1, c_stop) - max(a0, c_start) + 1)
                if float(inter) > best_inter:
                    best_id = cidx
                    best_inter = float(inter)
                    shortest = c_stop - c_start + 1
                elif float(inter) == float(best_inter):
                    clen = c_stop - c_start + 1
                    if shortest > clen:
                        best_id = cidx
                        shortest = clen

            if isinstance(old_candidates, list) and len(old_candidates) > best_id:
                lan_start = int(old_candidates[best_id]["start_token"])
                lan_stop = int(old_candidates[best_id]["end_token"])

        if lan_start > 0 and lan_stop > 0 and an_target != 0:
            long_string = f"{lan_start}:{lan_stop}"
        else:
            long_string = ""

        lines.append({"example_id": long_id, "PredictionString": long_string})

    return pd.DataFrame(lines).sort_values("example_id")




## === cell 15
def parseDataClean_from_cache(
    id_df,
    clean_doc_tokens_by_id,
    question_by_id,
    debug=False,
):
    INSTANCE_WORDS_LEN = 500
    STRIDE = 128
    list_instances = []

    for example_id in id_df["example_id"].astype(str).values.tolist():
        clean_doc = clean_doc_tokens_by_id[example_id]
        question = question_by_id[example_id]
        len_ques = len(question.split())
        part_len = INSTANCE_WORDS_LEN - len_ques
        if part_len < 16:
            part_len = 16

        num_ins = (len(clean_doc) - part_len) // STRIDE + 1
        if num_ins < 0:
            num_ins = 0

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




## === cell 16
def create_model_san(tokenizer_san, model_name_san, debug=False):
    encoder = TFBertModel.from_pretrained(model_name_san, local_files_only=True)
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
                inputs[0],
                token_type_ids=inputs[1],
                attention_mask=inputs[2],
            )
            seq_out = bert_res[0]
            pooled_out = bert_res[1]

            start_logits = tf.squeeze(self.start_logits(seq_out), -1)
            stop_logits = tf.squeeze(self.stop_logits(seq_out), -1)

            targets = self.target(pooled_out)
            paddings = tf.constant([[0, 0], [0, 512 - NUM_TARGET]])
            targets = tf.pad(targets, paddings)

            res = tf.stack([start_logits, stop_logits, targets], axis=1)
            return res

    return MyQAModel()




## === cell 17
def getSanRawRes(list_san_ins, verbose=1):
    print(
        "Getting raw result for short answer instance generated from found long answers"
    )

    if list_san_ins is None or len(list_san_ins) == 0:
        print("No short-answer instances to score; skipping SAN model inference.")
        return np.zeros((0, 3, 512), dtype=np.float32)

    model_name_san = _pick_hf_model_dir()
    tokenizer_san = AutoTokenizer.from_pretrained(model_name_san, local_files_only=True)

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
    tokenizer_san.add_special_tokens({"additional_special_tokens": tags_san})
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
        _ = sanModel.predict([x, x, x], verbose=0)

        weights_path = _resolve_weights_path("../input/model1/weights-14.h5")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing weights file: {weights_path}")
        sanModel.load_weights(weights_path)

        optAdam = tf.keras.optimizers.Adam(learning_rate=0.00005)
        lossSCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metricSCA = tf.keras.metrics.SparseCategoricalAccuracy()
        sanModel.compile(optimizer=optAdam, loss=lossSCE, metrics=[metricSCA])

    if verbose:
        print("Finish loading pretrained weights for the model for short answer")

    test_res = sanModel.predict([x_san1, x_san2, x_san3], verbose=1, batch_size=8)
    if verbose:
        print("Finish calculating raw result, get an array of size: ", test_res.shape)
    return test_res




## === cell 18
def getSanSubmission(doc_res_df, threshold=0.0001, debug=False):
    doc_res_df.example_id = doc_res_df.example_id.astype(str)
    lines = []
    for _, doc in doc_res_df.iterrows():
        example_id = doc["example_id"]
        short_id = str(example_id) + "_short"

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

        lines.append({"example_id": short_id, "PredictionString": short_string})

    return pd.DataFrame(lines).sort_values("example_id")




## === cell 19
def getSanCandidate_from_cache(
    sub_long, raw_doc_tokens_by_id, question_by_id, debug=False
):
    INSTANCE_WORDS_LEN = 500
    STRIDE = 256

    long_pred = {}
    for _, row in sub_long.iterrows():
        ex = str(row["example_id"]).replace("_long", "")
        ps = str(row["PredictionString"])
        if ps != "":
            a, b = ps.split(":")
            long_pred[ex] = (int(a), int(b))
        else:
            long_pred[ex] = (-1, -1)

    list_san_ins = []
    for example_id, (lan_start, lan_stop) in long_pred.items():
        doc_text_split = raw_doc_tokens_by_id.get(example_id, None)
        if doc_text_split is None:
            continue
        question = question_by_id[example_id]

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




## === cell 20
def refineLan(sub, list_mapping_df, debug=False):
    sub_map = dict(
        zip(
            sub["example_id"].astype(str).tolist(),
            sub["PredictionString"].astype(str).tolist(),
        )
    )
    map_old_cands = {}
    for _, r in list_mapping_df.iterrows():
        map_old_cands[str(r["example_id"])] = r["old_candidates"]

    newsub = []
    for _, row in sub.iterrows():
        exid = str(row["example_id"])
        if "long" in exid:
            example_id = exid.replace("_long", "")
            longid = exid
            longStr = str(row["PredictionString"])

            lan_start, lan_stop = -1, -1
            if longStr != "":
                tokens = longStr.split(":")
                lan_start = int(tokens[0])
                lan_stop = int(tokens[1])

            sanid = example_id + "_short"
            sanStr = str(sub_map.get(sanid, ""))

            if sanStr != "" and sanStr not in ("YES", "NO") and longStr != "":
                tokensans = sanStr.split(":")
                san_start = int(tokensans[0])
                san_stop = int(tokensans[1])

                if san_start < lan_start or san_stop > lan_stop:
                    cands = map_old_cands.get(example_id, [])
                    a0, a1 = san_start, san_stop

                    best_inter = 0.5
                    shortest = 10**18
                    best_id = 0
                    for cidx, cand in enumerate(cands):
                        c_start = int(cand["start_token"])
                        c_stop = int(cand["end_token"])
                        if c_stop < c_start:
                            continue
                        inter = max(0, min(a1, c_stop) - max(a0, c_start) + 1)
                        if float(inter) > best_inter:
                            best_id = cidx
                            best_inter = float(inter)
                            shortest = c_stop - c_start + 1
                        elif float(inter) == float(best_inter):
                            clen = c_stop - c_start + 1
                            if shortest > clen:
                                best_id = cidx
                                shortest = clen

                    if len(cands) > 0:
                        lan_start = int(cands[best_id]["start_token"])
                        lan_stop = int(cands[best_id]["end_token"])
                        longStr = f"{lan_start}:{lan_stop}"

            newsub.append({"example_id": longid, "PredictionString": longStr})
            newsub.append({"example_id": sanid, "PredictionString": sanStr})

    return pd.DataFrame(newsub).sort_values("example_id")




## === cell 21
(
    list_id_df,
    clean_doc_tokens_by_id,
    raw_doc_tokens_by_id,
    question_by_id,
    cand_map_rows,
) = load_test_cache(f_test, verbose=True)
list_mappings_df = pd.DataFrame(cand_map_rows)

list_all_ins = parseDataClean_from_cache(
    list_id_df, clean_doc_tokens_by_id, question_by_id
)
all_ins_res = getRawInstanceResults(list_all_ins)

list_fine_res_all_ins = mergeInstanceResult(all_ins_res, list_all_ins)
fine_res_all_ins_df = pd.DataFrame(list_fine_res_all_ins)

docAnsDf = mergeDocumentRes(fine_res_all_ins_df, list_id_df)
subLan = getSubmissionLan(docAnsDf, list_mappings_df)

subLan_only = subLan  # already long-only
list_san_ins = getSanCandidate_from_cache(
    subLan_only, raw_doc_tokens_by_id, question_by_id, debug=False
)
sanRawRes = getSanRawRes(list_san_ins)

if sanRawRes.shape[0] == 0:
    subSan = pd.DataFrame(
        {
            "example_id": list_id_df["example_id"].astype(str) + "_short",
            "PredictionString": [""] * len(list_id_df),
        }
    ).sort_values("example_id")
else:
    list_fine_res_san_ins = mergeInstanceResult(sanRawRes, list_san_ins)
    fine_res_san_ins_df = pd.DataFrame(list_fine_res_san_ins)

    docSanAnsDf = mergeDocumentRes(fine_res_san_ins_df, list_id_df)
    subSan = getSanSubmission(docSanAnsDf, threshold=0.2)

sub = pd.concat([subLan, subSan], ignore_index=True)
sub_sorted = sub.sort_values("example_id")

refineSub = refineLan(sub_sorted, list_mappings_df, debug=False)

sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")
sample = pd.read_csv(sample_path)

final = sample[["example_id"]].merge(refineSub, on="example_id", how="left")
final["PredictionString"] = final["PredictionString"].fillna("")
final["example_id"] = final["example_id"].astype(str)

final = final[["example_id", "PredictionString"]]

final.to_csv("./submission.csv", index=False)
print("Wrote submission.csv with shape:", final.shape)
print(final.head())

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3229265867.py in <cell line: 0>()
     11     list_id_df, clean_doc_tokens_by_id, question_by_id
     12 )
---> 13 all_ins_res = getRawInstanceResults(list_all_ins)
     14 
     15 list_fine_res_all_ins = mergeInstanceResult(all_ins_res, list_all_ins)

/tmp/ipykernel_11/1230019249.py in getRawInstanceResults(list_test, verbose, debug)
      3         print("Getting raw result for all the instances generated from test file")
      4 
----> 5     model_name = _pick_hf_model_dir()
      6     tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
      7 

/tmp/ipykernel_11/2512587262.py in _pick_hf_model_dir()
    119             if dirpath.count(os.sep) - root.count(os.sep) >= 6:
    120                 dirnames[:] = []
--> 121     raise FileNotFoundError(
    122         "Could not find a local HuggingFace model directory with config/tokenizer files under /kaggle/input."
    123     )

FileNotFoundError: Could not find a local HuggingFace model directory with config/tokenizer files under /kaggle/input.
