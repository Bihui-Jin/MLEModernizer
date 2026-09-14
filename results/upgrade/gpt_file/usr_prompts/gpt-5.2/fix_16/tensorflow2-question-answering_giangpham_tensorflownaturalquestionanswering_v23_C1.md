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

0.57117

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.57117) has done: 'I fix the initial protobuf/transformers import crash by removing the forced pure-Python protobuf environment variables that are incompatible with the Kaggle runtime. Then I fix the TensorFlow dataset batching bug that feeds 1D tensors into BERT by building a proper `(x1, x2, x3)` dataset so each batch has shape `(batch, 512)`. Finally, I add small safety fallbacks so that if a stage fails to produce predictions, the code still writes a valid `submission.csv` with the exact required columns/row order from `sample_submission.csv`.'
- What this solution (achieved 0.57117) has done: 'I fix the immediate runtime crash coming from an incompatible `protobuf` version used by `transformers` by forcing the pure-Python protobuf implementation *before* importing TensorFlow/Transformers (this is the minimal change that restores execution in this environment). I also add a small, score-neutral fallback that uses the local Kaggle BERT bundle if the previously referenced model directory doesn’t exist, so the script can still load models without internet access. Because your current score (0.57117) is already above the target (0.46356) and within the ±10% band, I not change thresholds or prediction logic to avoid drifting the score further. The rest of the pipeline (instance building, model, merging, and submission formatting) is preserved.'
- What this solution (achieved 0.57117) has done: 'The protobuf crash is caused by forcing the pure-Python protobuf runtime, which is incompatible with the Kaggle TensorFlow/transformers stack; removing those environment overrides fixes the `MessageFactory.GetPrototype` error and lets imports succeed. I keep the model, preprocessing, instance building, and merging logic unchanged, only adding a small, safe import-time fallback that retries without transformers if needed and ensures determinism settings don’t crash older TF builds. Since your current score (0.57117) is already within ±10% of the target (0.46356), I won’t change thresholds or prediction logic to avoid drifting the score. The pipeline run end-to-end and always write `./submission.csv` with the exact row order/ids from `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

for k in (
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION",
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION",
):
    if k in os.environ:
        os.environ.pop(k, None)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import re
import json

import tensorflow as tf

try:
    from transformers import AutoTokenizer, TFBertModel
except Exception as e:
    raise RuntimeError(
        "Failed to import transformers. This notebook requires transformers to run."
    ) from e

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

print("TF version:", tf.__version__)
print("Python:", sys.version)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
f_test = "../input/simplified-nq-test.jsonl"
f_train = "../input/simplified-nq-train.jsonl"

num_train_samples = 307372
num_test_samples = 346

if not os.path.exists(f_test):
    alt = "../input/tensorflow2-question-answering/simplified-nq-test.jsonl"
    if os.path.exists(alt):
        f_test = alt

if not os.path.exists(f_train):
    alt = "../input/tensorflow2-question-answering/simplified-nq-train.jsonl"
    if os.path.exists(alt):
        f_train = alt

print("Using f_test:", f_test)
print("Using f_train:", f_train)



## === cell 2
_TAG_RE = re.compile(r"<[^>]*>")


def _split_and_strip_tags_to_tokens(text: str):
    return _TAG_RE.sub(" ", text).split()


def clean_html(raw_html):
    return _TAG_RE.sub("<tag>", raw_html)


def build_test_artifacts_single_pass(
    filename,
    tokenizer_for_len,
    build_instances=True,
    build_mapping=True,
):
    MODEL_MAX_LEN = 512
    STRIDE = 128
    MIN_CONTEXT_TOKENS = 16

    list_ids = []
    list_cand_maps = []

    inst_example_id = []
    inst_part_start = []
    inst_part_stop = []
    inst_question = []
    inst_context = []

    qlen_cache = {}
    doc_cache = {}  # example_id -> {"raw_split_tokens": [...], "question_text": ...}

    json_loads = json.loads
    tokenize_q = tokenizer_for_len.tokenize
    tag_sub_space = _TAG_RE.sub

    with open(filename) as f:
        for line in tqdm(f, desc="Single-pass parse test (ids/mapping/instances)"):
            data = json_loads(line)
            example_id = str(data["example_id"])
            list_ids.append(example_id)

            doc_text = data["document_text"]
            question = data["question_text"]

            raw_split_tokens = doc_text.split()
            doc_cache[example_id] = {
                "raw_split_tokens": raw_split_tokens,
                "question_text": question,
            }

            if build_instances:
                clean_tokens = tag_sub_space(" ", doc_text).split()

                q_len = qlen_cache.get(question)
                if q_len is None:
                    q_len = len(tokenize_q(question))
                    qlen_cache[question] = q_len

                part_len = MODEL_MAX_LEN - q_len - 3
                if part_len < MIN_CONTEXT_TOKENS:
                    part_len = MIN_CONTEXT_TOKENS

                num_ins = (len(clean_tokens) - part_len) // STRIDE + 1
                if num_ins < 0:
                    num_ins = 0

                join = " ".join
                part_text_cache = {}

                for part_id in range(num_ins + 1):
                    part_start = part_id * STRIDE
                    part_stop = min(len(clean_tokens), part_start + part_len)
                    key = (part_start, part_stop)
                    ctx = part_text_cache.get(key)
                    if ctx is None:
                        ctx = join(clean_tokens[part_start:part_stop])
                        part_text_cache[key] = ctx

                    inst_example_id.append(example_id)
                    inst_part_start.append(part_start)
                    inst_part_stop.append(part_stop)
                    inst_question.append(question)
                    inst_context.append(ctx)

            if build_mapping:
                doc_text_raw = tag_sub_space("<tag>", doc_text)
                doc_text_split = doc_text_raw.split()

                arr = np.asarray(doc_text_split, dtype=object)
                is_tag = (arr == "<tag>").astype(np.int32)
                prefix_tag = np.empty(len(is_tag) + 1, dtype=np.int32)
                prefix_tag[0] = 0
                np.cumsum(is_tag, out=prefix_tag[1:])

                max_idx = len(prefix_tag) - 1
                list_candidates = data["long_answer_candidates"]
                list_new_candidates = []
                for cand in list_candidates:
                    cand_start = int(cand["start_token"])
                    cand_stop = int(cand["end_token"])

                    cs = min(max(cand_start, 0), max_idx)
                    ce = min(max(cand_stop, 0), max_idx)

                    new_start = cand_start - int(prefix_tag[cs])
                    new_stop = cand_stop - int(prefix_tag[ce])

                    list_new_candidates.append(
                        {"end_token": int(new_stop), "start_token": int(new_start)}
                    )

                list_cand_maps.append(
                    {
                        "example_id": example_id,
                        "new_candidates": list_new_candidates,
                        "old_candidates": list_candidates,
                    }
                )

    list_id_df = pd.DataFrame({"example_id": list_ids})
    mappings_df = pd.DataFrame(list_cand_maps)

    if build_instances:
        n = len(inst_example_id)
        list_instances = [
            {
                "example_id": inst_example_id[i],
                "part_start": int(inst_part_start[i]),
                "part_stop": int(inst_part_stop[i]),
                "question": inst_question[i],
                "context": inst_context[i],
                "start": 0,
                "stop": 0,
                "target": "NO_ANSWER",
            }
            for i in range(n)
        ]
    else:
        list_instances = []

    return list_id_df, mappings_df, list_instances, doc_cache




## === cell 3
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




## === cell 4
def preprocess_data(data, tokenizer, debug=False, return_tf=True):
    if isinstance(data, dict) and "questions" in data and "contexts" in data:
        questions = data["questions"]
        contexts = data["contexts"]
        starts = data.get("starts", None)
        stops = data.get("stops", None)
        targets = data.get("targets", None)
    else:
        questions = [sam["question"] for sam in data]
        contexts = [sam["context"] for sam in data]
        starts = [sam["start"] for sam in data]
        stops = [sam["stop"] for sam in data]
        targets = [AnswerType[sam["target"]] for sam in data]

    enc = tokenizer(
        questions,
        contexts,
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors=("tf" if return_tf else "np"),
    )

    if return_tf:
        x1 = tf.cast(enc["input_ids"], tf.int32)
        x2 = tf.cast(
            enc.get("token_type_ids", tf.zeros_like(enc["input_ids"])), tf.int32
        )
        x3 = tf.cast(enc["attention_mask"], tf.int32)
    else:
        x1 = tf.convert_to_tensor(enc["input_ids"], dtype=tf.int32)
        x2 = tf.convert_to_tensor(
            enc.get("token_type_ids", np.zeros_like(enc["input_ids"])), dtype=tf.int32
        )
        x3 = tf.convert_to_tensor(enc["attention_mask"], dtype=tf.int32)

    if starts is None:
        y = tf.zeros((len(questions), 3), dtype=tf.int32)
    else:
        y_np = np.empty((len(questions), 3), dtype=np.int32)
        y_np[:, 0] = np.asarray(starts, dtype=np.int32)
        y_np[:, 1] = np.asarray(stops, dtype=np.int32)
        y_np[:, 2] = np.asarray(targets, dtype=np.int32)
        y = tf.convert_to_tensor(y_np, dtype=tf.int32)

    return x1, x2, x3, y




## === cell 5
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
        print("No TPU detected:", repr(e))
        strategy = tf.distribute.get_strategy()
    return strategy




## === cell 6
def mergeInstanceResult(test_res, list_test_ins):
    start_idx = np.argmax(test_res[:, 0, :], axis=1).astype(np.int32)
    stop_idx = np.argmax(test_res[:, 1, :], axis=1).astype(np.int32)
    target_idx = np.argmax(test_res[:, 2, :], axis=1).astype(np.int32)

    rows = np.arange(test_res.shape[0])
    start_score = test_res[rows, 0, start_idx].astype(np.float32)
    stop_score = test_res[rows, 1, stop_idx].astype(np.float32)
    target_score = test_res[rows, 2, target_idx].astype(np.float32)

    start_CLS = test_res[rows, 0, 0].astype(np.float32)
    stop_CLS = test_res[rows, 1, 0].astype(np.float32)

    for i in range(len(list_test_ins)):
        ins = list_test_ins[i]
        ins["start"] = int(start_idx[i])
        ins["stop"] = int(stop_idx[i])
        ins["target"] = int(target_idx[i])
        ins["start_score"] = float(start_score[i])
        ins["stop_score"] = float(stop_score[i])
        ins["target_score"] = float(target_score[i])
        ins["start_CLS"] = float(start_CLS[i])
        ins["stop_CLS"] = float(stop_CLS[i])
    return list_test_ins




## === cell 7
def mergeDocumentRes(ins_df, val_id_df, threshold=0.0001, stride=128, debug=False):
    if ins_df.empty:
        return pd.DataFrame(
            {
                "example_id": val_id_df["example_id"].astype(str),
                "start": -1,
                "stop": -1,
                "target": 0,
                "score": threshold,
            }
        )

    needed_cols = [
        "example_id",
        "start",
        "stop",
        "target",
        "part_start",
        "start_score",
        "stop_score",
        "start_CLS",
        "stop_CLS",
    ]
    ins = ins_df[needed_cols].copy()
    ins["example_id"] = ins["example_id"].astype(str)

    ex = ins["example_id"].to_numpy(dtype=object)
    s = ins["start"].to_numpy(dtype=np.int32)
    e = ins["stop"].to_numpy(dtype=np.int32)
    tgt = ins["target"].to_numpy(dtype=np.int32)
    ps = ins["part_start"].to_numpy(dtype=np.int32)
    ss = ins["start_score"].to_numpy(dtype=np.float32)
    es = ins["stop_score"].to_numpy(dtype=np.float32)
    sc = ins["start_CLS"].to_numpy(dtype=np.float32)
    ec = ins["stop_CLS"].to_numpy(dtype=np.float32)

    m = (s != 0) | (e != 0)
    if not np.any(m):
        return pd.DataFrame(
            {
                "example_id": val_id_df["example_id"].astype(str),
                "start": -1,
                "stop": -1,
                "target": 0,
                "score": threshold,
            }
        )

    ex = ex[m]
    s = s[m]
    e = e[m]
    tgt = tgt[m]
    ps = ps[m]
    ss = ss[m]
    es = es[m]
    sc = sc[m]
    ec = ec[m]

    real_s = s + ps
    real_e = e + ps
    valid = real_e > real_s
    score = (ss - sc) + (es - ec)
    score = np.where(valid, score, -np.inf)

    order = np.argsort(ex, kind="mergesort")
    ex_s = ex[order]
    score_s = score[order]
    real_s_s = real_s[order]
    real_e_s = real_e[order]
    tgt_s = tgt[order]

    uniq, start_idx, counts = np.unique(ex_s, return_index=True, return_counts=True)
    best_score = np.empty(len(uniq), dtype=np.float32)
    best_start = np.empty(len(uniq), dtype=np.int32)
    best_stop = np.empty(len(uniq), dtype=np.int32)
    best_target = np.empty(len(uniq), dtype=np.int32)

    for gi in range(len(uniq)):
        a = start_idx[gi]
        b = a + counts[gi]
        j = a + int(np.argmax(score_s[a:b]))
        best_score[gi] = float(score_s[j])
        best_start[gi] = int(real_s_s[j])
        best_stop[gi] = int(real_e_s[j])
        best_target[gi] = int(tgt_s[j])

    best_score = best_score.astype(np.float32)
    keep = best_score > float(threshold)

    best_map = {
        str(uniq[i]): (
            int(best_start[i]) if keep[i] else -1,
            int(best_stop[i]) if keep[i] else -1,
            int(best_target[i]) if keep[i] else 0,
            float(best_score[i]) if keep[i] else float(threshold),
        )
        for i in range(len(uniq))
    }

    out = []
    val_ids = val_id_df["example_id"].astype(str).tolist()
    for idx, doc_id in enumerate(val_ids):
        st, sp, tg, scv = best_map.get(doc_id, (-1, -1, 0, float(threshold)))
        doc_lan = {
            "example_id": doc_id,
            "start": st,
            "stop": sp,
            "target": tg,
            "score": scv,
        }
        if debug and idx == 101:
            print(doc_lan)
        out.append(doc_lan)

    return pd.DataFrame(out)




## === cell 8
def _resolve_local_or_fallback(model_path, fallback="bert-base-uncased"):
    if model_path and os.path.isdir(model_path):
        return model_path, True
    return fallback, False


def build_model(model_name, debug=False):
    resolved_name, is_local = _resolve_local_or_fallback(model_name)
    encoder = TFBertModel.from_pretrained(resolved_name, local_files_only=is_local)
    tokenizer = AutoTokenizer.from_pretrained(resolved_name, local_files_only=is_local)

    tags = ["``", "''", "--"]
    special_tokens_dict = {"additional_special_tokens": tags}
    tokenizer.add_special_tokens(special_tokens_dict)
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
                inputs[0], token_type_ids=inputs[1], attention_mask=inputs[2]
            )
            seq_out = bert_res[0]
            pooled_out = bert_res[1]

            start_logits = tf.squeeze(self.start_logits(seq_out), -1)
            stop_logits = tf.squeeze(self.stop_logits(seq_out), -1)

            targets = self.target(pooled_out)
            paddings = tf.constant([[0, 0], [0, 512 - NUM_TARGET]])
            targets = tf.pad(targets, paddings)

            return tf.stack([start_logits, stop_logits, targets], axis=1)

    return MyQAModel()




## === cell 9
_TOKENIZER_CACHE = {}
_MODEL_CACHE = {}


def _get_cached_tokenizer(resolved_name, is_local, additional_special_tokens):
    key = (resolved_name, bool(is_local), tuple(additional_special_tokens))
    tok = _TOKENIZER_CACHE.get(key)
    if tok is not None:
        return tok
    tok = AutoTokenizer.from_pretrained(resolved_name, local_files_only=is_local)
    tok.add_special_tokens(
        {"additional_special_tokens": list(additional_special_tokens)}
    )
    _TOKENIZER_CACHE[key] = tok
    return tok


def _get_cached_compiled_model(model_builder_key, builder_fn, weights_path):
    m = _MODEL_CACHE.get(model_builder_key)
    if m is not None:
        return m
    m = builder_fn()
    x = np.ones([1, 512], dtype=np.int32)
    m.predict([x, x, x], verbose=0)
    if os.path.exists(weights_path):
        m.load_weights(weights_path)
    else:
        print("WARNING: weights not found:", weights_path)
    optAdam = tf.keras.optimizers.Adam(learning_rate=0.00005)
    lossSCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metricSCA = tf.keras.metrics.SparseCategoricalAccuracy()
    m.compile(optimizer=optAdam, loss=lossSCE, metrics=[metricSCA])
    _MODEL_CACHE[model_builder_key] = m
    return m


def _predict_batched(model, x1, x2, x3, batch_size=16, verbose=0):
    ds = tf.data.Dataset.from_tensor_slices((x1, x2, x3))
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return model.predict(ds, verbose=verbose)


def getRawInstanceResults(list_test, verbose=True, debug=False):
    if verbose:
        print("Getting raw result for all the instances generated from test file")

    model_name = "../input/tensorflow-question-answer-fine-data"
    if not os.path.isdir(model_name):
        alt_local = "../input/tensorflow2-question-answering/tensorflow-question-answer-fine-data"
        if os.path.isdir(alt_local):
            model_name = alt_local

    resolved_name, is_local = _resolve_local_or_fallback(model_name)

    tags = ["``", "''", "--"]
    tokenizer = _get_cached_tokenizer(resolved_name, is_local, tags)
    if verbose:
        print("Tokenizer size:", len(tokenizer))

    strategy = get_strategy()
    with strategy.scope():
        weights_path = "../input/model1/weights-01.h5"
        model_key = ("lan", resolved_name, weights_path, len(tokenizer))

        def _builder():
            return build_model(resolved_name)

        testModel = _get_cached_compiled_model(model_key, _builder, weights_path)

    CHUNK = 4096
    out_chunks = []
    n = len(list_test)
    if verbose:
        print("Total instances:", n, "Tokenize/predict chunk size:", CHUNK)

    for i in range(0, n, CHUNK):
        chunk = list_test[i : i + CHUNK]
        qs = [s["question"] for s in chunk]
        cs = [s["context"] for s in chunk]
        x1, x2, x3, _ = preprocess_data(
            {"questions": qs, "contexts": cs}, tokenizer, return_tf=True
        )
        res = _predict_batched(testModel, x1, x2, x3, batch_size=32, verbose=0)
        out_chunks.append(res)

    test_res = (
        np.concatenate(out_chunks, axis=0)
        if out_chunks
        else np.zeros((0, 3, 512), dtype=np.float32)
    )
    if verbose:
        print("Finish calculating raw result, get an array of size:", test_res.shape)
    return test_res




## === cell 10
def getSubmissionLan(doc_res_df, doc_cand_df, threshold=0.0001, debug=False):
    doc_res_df = doc_res_df.copy()
    doc_cand_df = doc_cand_df.copy()
    doc_res_df["example_id"] = doc_res_df["example_id"].astype(str)
    doc_cand_df["example_id"] = doc_cand_df["example_id"].astype(str)

    combine_df = pd.merge(doc_res_df, doc_cand_df, on="example_id", how="left")

    ex_ids = combine_df["example_id"].astype(str).to_numpy()
    starts = combine_df["start"].to_numpy(dtype=np.int32)
    stops = combine_df["stop"].to_numpy(dtype=np.int32)
    targets = combine_df["target"].to_numpy(dtype=np.int32)
    new_cands = combine_df["new_candidates"].to_list()
    old_cands = combine_df["old_candidates"].to_list()

    lines = []
    for i in range(len(combine_df)):
        example_id = ex_ids[i]
        long_id = str(example_id) + "_long"

        an_start = int(starts[i])
        an_stop = int(stops[i])
        an_target = int(targets[i])
        lan_start, lan_stop = -1, -1

        cand_list = new_cands[i]
        if an_start > 0 and an_stop > 0 and isinstance(cand_list, list):
            best_inter = 0.5
            shortest = 10**18
            best_id = 0

            for cidx, cand in enumerate(cand_list):
                c_start = int(cand["start_token"])
                c_stop = int(cand["end_token"])

                left = max(an_start, c_start)
                right = min(an_stop, c_stop)
                inter = (right - left + 1) if right >= left else 0

                if float(inter) > best_inter:
                    best_id = cidx
                    best_inter = float(inter)
                    shortest = c_stop - c_start + 1
                elif float(inter) == best_inter:
                    clen = c_stop - c_start + 1
                    if shortest > clen:
                        best_id = cidx
                        shortest = clen

            real_candidates = old_cands[i]
            if isinstance(real_candidates, list) and len(real_candidates) > best_id:
                lan_start = int(real_candidates[best_id]["start_token"])
                lan_stop = int(real_candidates[best_id]["end_token"])

        long_string = (
            str(lan_start) + ":" + str(lan_stop)
            if (lan_start > 0 and lan_stop > 0 and an_target != 0)
            else ""
        )
        lines.append({"example_id": long_id, "PredictionString": long_string})

    return pd.DataFrame(lines).sort_values("example_id")




## === cell 11
def getSanCandidate(sub, filename=f_test, debug=False, doc_cache=None):
    INSTANCE_WORDS_LEN = 500
    STRIDE = 256

    list_doc_lan_res = []
    for _, row in sub.iterrows():
        example_id = str(row["example_id"]).replace("_long", "")
        lan_start, lan_stop = -1, -1
        if str(row["PredictionString"]) != "":
            tokens = str(row["PredictionString"]).split(":")
            lan_start = int(tokens[0])
            lan_stop = int(tokens[1])
        list_doc_lan_res.append(
            {"example_id": example_id, "lan_start": lan_start, "lan_stop": lan_stop}
        )

    list_doc_lan_res_df = pd.DataFrame(list_doc_lan_res)
    set_id = set(list_doc_lan_res_df["example_id"].values.tolist())

    lan_dict = {
        r["example_id"]: (int(r["lan_start"]), int(r["lan_stop"]))
        for r in list_doc_lan_res_df.to_dict("records")
    }

    list_san_ins = []

    if doc_cache is not None:
        join = " ".join
        get_cache = doc_cache.get
        get_lan = lan_dict.get

        for example_id in tqdm(set_id, desc="Building short-answer instances (cached)"):
            cached = get_cache(example_id)
            if cached is None:
                continue

            lan_start, lan_stop = get_lan(example_id, (-1, -1))
            doc_text_split = cached["raw_split_tokens"]
            question = cached["question_text"]

            join_cache = {}

            if lan_start > -1 and lan_stop > -1:
                if lan_stop - lan_start <= INSTANCE_WORDS_LEN:
                    offset = (INSTANCE_WORDS_LEN - (lan_stop - lan_start)) // 2
                    part_start = max(0, lan_start - offset)
                    part_stop = min(lan_stop + offset, len(doc_text_split))
                    key = (part_start, part_stop)
                    context = join_cache.get(key)
                    if context is None:
                        context = join(doc_text_split[part_start:part_stop])
                        join_cache[key] = context
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
                    num_parts = (
                        lan_stop - lan_start - INSTANCE_WORDS_LEN
                    ) // STRIDE + 1
                    for part_id in range(num_parts + 1):
                        part_start = lan_start + part_id * STRIDE
                        part_stop = min(
                            len(doc_text_split),
                            lan_start + part_id * STRIDE + part_length,
                        )
                        key = (part_start, part_stop)
                        context = join_cache.get(key)
                        if context is None:
                            context = join(doc_text_split[part_start:part_stop])
                            join_cache[key] = context
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

    with open(filename) as f:
        for line in tqdm(f, desc="Building short-answer instances"):
            data = json.loads(line)
            example_id = str(data["example_id"])
            if example_id not in set_id:
                continue

            lan_start, lan_stop = lan_dict.get(example_id, (-1, -1))
            doc_text = data["document_text"]
            doc_text_split = doc_text.split()
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
                    num_parts = (
                        lan_stop - lan_start - INSTANCE_WORDS_LEN
                    ) // STRIDE + 1
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




## === cell 12
def create_model_san(tokenizer_san, model_name_san, debug=False):
    resolved_name, is_local = _resolve_local_or_fallback(model_name_san)
    encoder = TFBertModel.from_pretrained(resolved_name, local_files_only=is_local)
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
                inputs[0], token_type_ids=inputs[1], attention_mask=inputs[2]
            )
            seq_out = bert_res[0]
            pooled_out = bert_res[1]

            start_logits = tf.squeeze(self.start_logits(seq_out), -1)
            stop_logits = tf.squeeze(self.stop_logits(seq_out), -1)

            targets = self.target(pooled_out)
            paddings = tf.constant([[0, 0], [0, 512 - NUM_TARGET]])
            targets = tf.pad(targets, paddings)

            return tf.stack([start_logits, stop_logits, targets], axis=1)

    return MyQAModel()




## === cell 13
def getSanRawRes(list_san_ins, verbose=1):
    print(
        "Getting raw result for short answer instance generated from found long answers"
    )

    model_name_san = "../input/tensorflow-question-answer-fine-data"
    if not os.path.isdir(model_name_san):
        alt_local = "../input/tensorflow2-question-answering/tensorflow-question-answer-fine-data"
        if os.path.isdir(alt_local):
            model_name_san = alt_local

    resolved_name, is_local = _resolve_local_or_fallback(model_name_san)

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
    tokenizer_san = _get_cached_tokenizer(resolved_name, is_local, tags_san)
    print("Short answer vocab size:", len(tokenizer_san))

    strategy_san = get_strategy()
    with strategy_san.scope():
        weights_path = "../input/model1/weights-14.h5"
        model_key = ("san", resolved_name, weights_path, len(tokenizer_san))

        def _builder():
            return create_model_san(tokenizer_san, resolved_name)

        sanModel = _get_cached_compiled_model(model_key, _builder, weights_path)

    if verbose:
        print("Finish loading pretrained weights for the model for short answer")

    CHUNK = 4096
    out_chunks = []
    n = len(list_san_ins)
    for i in range(0, n, CHUNK):
        chunk = list_san_ins[i : i + CHUNK]
        qs = [s["question"] for s in chunk]
        cs = [s["context"] for s in chunk]
        x1, x2, x3, _ = preprocess_data(
            {"questions": qs, "contexts": cs}, tokenizer_san, return_tf=True
        )
        res = _predict_batched(sanModel, x1, x2, x3, batch_size=32, verbose=0)
        out_chunks.append(res)

    test_res = (
        np.concatenate(out_chunks, axis=0)
        if out_chunks
        else np.zeros((0, 3, 512), dtype=np.float32)
    )
    if verbose:
        print("Finish calculating raw result, get an array of size:", test_res.shape)
    return test_res




## === cell 14
def getSanSubmission(doc_res_df, threshold=0.0001, debug=False):
    doc_res_df = doc_res_df.copy()
    doc_res_df["example_id"] = doc_res_df["example_id"].astype(str)
    lines = []
    for _, doc in doc_res_df.iterrows():
        example_id = doc["example_id"]
        short_id = str(example_id) + "_short"

        an_start = int(doc["start"])
        an_stop = int(doc["stop"])
        an_target = int(doc["target"])

        if an_start > 0 and an_stop > 0 and an_target != 4:
            short_string = str(an_start) + ":" + str(an_stop)
        else:
            short_string = ""

        if an_target == 1 or an_target == 2:
            short_string = AnswerTypeRev[an_target]

        lines.append({"example_id": short_id, "PredictionString": short_string})

    return pd.DataFrame(lines).sort_values("example_id")




## === cell 15
model_name_len = "../input/tensorflow-question-answer-fine-data"
if not os.path.isdir(model_name_len):
    alt_local = (
        "../input/tensorflow2-question-answering/tensorflow-question-answer-fine-data"
    )
    if os.path.isdir(alt_local):
        model_name_len = alt_local

resolved_name_len, is_local_len = _resolve_local_or_fallback(model_name_len)
_tokenizer_for_len = AutoTokenizer.from_pretrained(
    resolved_name_len, local_files_only=is_local_len
)

list_id_df, list_mappings_df, list_all_ins, _DOC_CACHE = (
    build_test_artifacts_single_pass(
        f_test,
        tokenizer_for_len=_tokenizer_for_len,
        build_instances=True,
        build_mapping=True,
    )
)

print("Docs in test:", len(list_id_df), "Mappings:", len(list_mappings_df))
print("Long-answer instances:", len(list_all_ins))



## === cell 16
subLan = None
try:
    all_ins_res = getRawInstanceResults(list_all_ins)
    list_fine_res_all_ins = mergeInstanceResult(all_ins_res, list_all_ins)
    fine_res_all_ins_df = pd.DataFrame(list_fine_res_all_ins)

    docAnsDf = mergeDocumentRes(fine_res_all_ins_df, list_id_df)
    subLan = getSubmissionLan(docAnsDf, list_mappings_df)

    print("Long submission rows:", len(subLan))
    print(subLan.head())
except Exception as e:
    print("ERROR during long-answer stage, falling back to blank longs:", repr(e))
    subLan = pd.DataFrame(
        {
            "example_id": [
                str(eid) + "_long"
                for eid in list_id_df["example_id"].astype(str).tolist()
            ],
            "PredictionString": [""] * len(list_id_df),
        }
    ).sort_values("example_id")



## === cell 17
subSan = None

list_san_ins = getSanCandidate(subLan, debug=False, doc_cache=_DOC_CACHE)
print("Short instances:", len(list_san_ins))

if len(list_san_ins) == 0:
    subSan = pd.DataFrame(
        {
            "example_id": [
                str(eid) + "_short"
                for eid in list_id_df["example_id"].astype(str).tolist()
            ],
            "PredictionString": [""] * len(list_id_df),
        }
    ).sort_values("example_id")
else:
    sanRawRes = getSanRawRes(list_san_ins)
    list_fine_res_san_ins = mergeInstanceResult(sanRawRes, list_san_ins)
    fine_res_san_ins_df = pd.DataFrame(list_fine_res_san_ins)

    docSanAnsDf = mergeDocumentRes(fine_res_san_ins_df, list_id_df)
    subSan = getSanSubmission(docSanAnsDf, threshold=0.2)

print("Short submission rows:", len(subSan))
print(subSan.head())



## === cell 18
sample_path = "../input/sample_submission.csv"
if not os.path.exists(sample_path):
    alt = "../input/tensorflow2-question-answering/sample_submission.csv"
    if os.path.exists(alt):
        sample_path = alt

sample_sub = pd.read_csv(sample_path)
expected_ids = sample_sub["example_id"].astype(str).tolist()

if subLan is None:
    subLan = pd.DataFrame({"example_id": [], "PredictionString": []})
if subSan is None:
    subSan = pd.DataFrame({"example_id": [], "PredictionString": []})

sub = pd.concat([subLan, subSan], ignore_index=True)
sub["example_id"] = sub["example_id"].astype(str)
sub["PredictionString"] = sub["PredictionString"].fillna("").astype(str)

sub_map = dict(
    zip(sub["example_id"].values.tolist(), sub["PredictionString"].values.tolist())
)
final_pred = [sub_map.get(eid, "") for eid in expected_ids]
final_sub = pd.DataFrame({"example_id": expected_ids, "PredictionString": final_pred})

print("Combined rows:", len(final_sub), "Expected:", len(expected_ids))
final_sub.to_csv("./submission.csv", index=False)
print("Wrote ./submission.csv")
print(final_sub.head(10))
print("Submission non-empty predictions:", (final_sub["PredictionString"] != "").sum())
