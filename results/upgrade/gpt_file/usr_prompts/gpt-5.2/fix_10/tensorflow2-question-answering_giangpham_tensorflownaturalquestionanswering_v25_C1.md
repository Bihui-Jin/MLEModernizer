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

0.4909733124018838

# 6. Current score

0.57117

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.57117) has done: 'Your notebook currently can’t yield a Kaggle score because it is unlikely to run end-to-end as written: it references missing input paths (`../input/model1/...` and `../input/tensorflow-question-answer-fine-data`), has wrong test/train filenames for this dataset, and the `refineLan()` call passes the unsorted `sub` instead of `sub_sorted`. I make minimal changes to (1) use the correct provided dataset paths/filenames, (2) fall back safely to a valid “blank” submission if the pretrained models/weights aren’t available in your environment (so you always get a valid `submission.csv`), and (3) fix the `refineLan()` call bug and ensure the submission aligns exactly to the sample submission’s `example_id` order. These changes preserve your core modeling/postprocessing logic when weights exist, but guarantee a valid CSV otherwise.'
- What this solution (achieved 0.57117) has done: 'I fix two hard runtime blockers: the `transformers` import error (caused by an incompatible protobuf runtime) and the incorrect dataset paths/filenames that don’t exist in your environment. These fixes keep your modeling and post-processing unchanged; they only ensure the notebook can actually load test data and (when available) run inference. I also keep your submission alignment to `sample_submission.csv` order and ensure `PredictionString` is always a string, so Kaggle accepts the file. Because your current score (0.57117) is already above the target (0.49097), I not make any score-improving changes—only correctness/stability fixes.'
- What this solution (achieved 0.57117) has done: 'I fix the hard runtime blocker causing `transformers` to crash (`MessageFactory.GetPrototype` protobuf incompatibility) by disabling TF/Flax backends for Transformers and forcing the pure-Python protobuf implementation before any TF/transformers import. I also make the model-availability check robust to the actual Kaggle dataset path you have (it’s under `/kaggle/input/tensorflow2-question-answering/...`, not `../input/model1/...`), while keeping the exact same model/inference/post-processing logic when the weights exist. Since your current score (0.57117) is already above the target (0.49097), I won’t make any score-improving changes—only correctness/stability fixes and path resolution so it runs end-to-end and writes `submission.csv`. Finally, I ensure the submission aligns exactly to `sample_submission.csv` order and `PredictionString` is always a string.'
- What this solution (achieved 0.57117) has done: 'I fix the hard runtime blocker that prevents `transformers` from importing in this Kaggle environment (protobuf `MessageFactory.GetPrototype` issue) by forcing the pure-Python protobuf implementation *and* ensuring the env vars are set before any protobuf/TF/transformers import happens. I also correct the dataset paths to the actual provided locations under `/kaggle/input/` (the files are in a dataset subfolder), so the script can always find `simplified-nq-*.jsonl` and `sample_submission.csv`. Since your current score (0.57117) is already above the target (0.49097) and within the ±10% tolerance band, I avoid any score-changing modeling/postprocessing changes and only make stability/correctness fixes. Finally, I keep the existing “blank submission fallback” so a valid `submission.csv` is always produced even if models/weights are unavailable.'
- What this solution (achieved 0.57117) has done: 'I fix the runtime crash in the very first cell by ensuring the protobuf implementation and Transformers backend flags are set before any protobuf-related import, and by falling back cleanly if Transformers/TensorFlow still can’t be imported. I also fix a logic/robustness issue in `refineLan()` where it can throw on missing mappings or missing short rows by adding safe guards (this preserves the same refinement logic when data exists, but prevents pipeline aborts). Since your current score (0.57117) is already above the target (0.49097) and within the ±10% band, I not make any score-improving changes—only correctness/stability changes so it runs end-to-end and always writes a valid `submission.csv`. Finally, I keep submission ordering aligned exactly to `sample_submission.csv` and ensure `PredictionString` is always a string.'
- What this solution (achieved 0.57117) has done: 'I fix the runtime crash happening before your try/except by forcing a compatible protobuf implementation *and* importing `google.protobuf` only inside the guarded block, so the notebook can always proceed. I also make the Transformers/TensorFlow availability detection more robust (without changing any model logic) and keep your existing blank-submission fallback so a valid `submission.csv` is always written. Since your current score (0.57117) is already above the target (0.49097) and within the ±10% band, I not make any score-improving changes—only stability/correctness fixes. Finally, I ensure the output column type is always a string and that the submission aligns exactly to `sample_submission.csv` order.'
- What this solution (achieved 0.57117) has done: 'I fix the protobuf/Transformers import crash by ensuring no protobuf-dependent libraries (including TensorFlow) are imported at module import time, and by forcing the pure-Python protobuf implementation before any such import happens. Then I make the Transformers/TensorFlow availability probe fully guarded so the notebook can always continue to the “blank submission” fallback instead of crashing in cell 1. These changes are score-neutral (they only affect runtime stability and whether your existing inference code can run), and since your current score is already above the target band, I not change any modeling/post-processing thresholds or logic. Finally, I keep submission ordering aligned to `sample_submission.csv` and ensure `PredictionString` is always written as a string.'
- What this solution (achieved 0.57117) has done: 'I fix the immediate runtime crash caused by importing `google.protobuf` before your try/except can protect the pipeline; this crash prevents any submission from being written. The minimal fix is to remove the unconditional `google.protobuf` import probe and instead rely on guarded TensorFlow/Transformers imports after setting the environment variables. I also keep your existing “blank submission fallback” unchanged so the notebook always produces a valid `submission.csv` even when TF/Transformers can’t be used. Since your current score is already above the target band (and higher-is-better), I won’t change any modeling/postprocessing logic or thresholds—only stability/correctness.'
- What this solution (achieved 0.57117) has done: 'I fix the immediate crash in the first cell by ensuring protobuf-dependent imports (TensorFlow/transformers) are fully contained inside a guarded block and by setting the required environment variables before any such import happens. This keeps your core model/inference logic identical when the pretrained assets are available, but allows the notebook to run end-to-end without dying at import time. Because your current score is already above the target (and within the ±10% band), I not change any thresholds or scoring-related logic—only stability/correctness so you always get a valid `submission.csv`. I also keep the existing robust path resolution and blank-submission fallback unchanged.'

# 9. Code solution

## === cell 0
import os, json, re, random, sys, shutil, string

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from tqdm import tqdm

random.seed(1234)
np.random.seed(1234)

TRANSFORMERS_OK = False
tf = None
AutoTokenizer = None
TFBertModel = None

try:
    import tensorflow as tf  # noqa: F401
    from transformers import AutoTokenizer, TFBertModel  # noqa: F401

    TRANSFORMERS_OK = True
except Exception as e:
    print(
        "Transformers/TensorFlow import failed; will generate a blank submission. Error:",
        repr(e),
    )
    TRANSFORMERS_OK = False




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _resolve_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


f_test = _resolve_first_existing(
    [
        "/kaggle/input/simplified-nq-test.jsonl",
        "/kaggle/input/simplified-nq-kaggle-test.jsonl",
        "/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl",
        "/kaggle/input/tensorflow2-question-answering/simplified-nq-kaggle-test.jsonl",
    ]
)
f_train = _resolve_first_existing(
    [
        "/kaggle/input/simplified-nq-train.jsonl",
        "/kaggle/input/tensorflow2-question-answering/simplified-nq-train.jsonl",
    ]
)
sample_sub_path = _resolve_first_existing(
    [
        "/kaggle/input/sample_submission.csv",
        "/kaggle/input/tensorflow2-question-answering/sample_submission.csv",
    ]
)

assert (
    sample_sub_path is not None
), "Missing sample_submission.csv in expected locations."
assert f_test is not None, "Missing simplified-nq test jsonl in expected locations."

if f_train is None:
    print("Warning: train file not found (not required for submission generation).")

print("Resolved paths:")
print(" sample_sub_path:", sample_sub_path)
print(" f_test:", f_test)
print(" f_train:", f_train)



## === cell 2
AnswerType = {"NO_ANSWER": 0, "YES": 1, "NO": 2, "SHORT": 3, "LONG": 4}
AnswerTypeRev = {v: k for k, v in AnswerType.items()}




## === cell 3
def get_id_df(filename=f_test):
    list_id = []
    with open(filename) as f:
        progress = tqdm(f)
        for _, line in enumerate(progress):
            data = json.loads(line)
            example_id = str(data["example_id"])
            list_id.append({"example_id": example_id})
    return pd.DataFrame(list_id)




## === cell 4
cleanr = re.compile("<.*?>")


def clean_html(raw_html):
    return re.sub(cleanr, "<tag>", raw_html)


def parseDataClean(
    filename=f_test,
    is_val=True,
    drop_noanswer_rate=0.95,
    drop_null_instances_rate=0.98,
    debug=False,
):
    INSTANCE_WORDS_LEN = 500
    STRIDE = 128
    list_instances = []

    with open(filename) as f:
        progress = tqdm(f)
        for _, line in enumerate(progress):
            data = json.loads(line)
            example_id = str(data["example_id"])

            doc_text_raw = data["document_text"]
            doc_text_tag = clean_html(doc_text_raw)
            doc_tag_split = doc_text_tag.split()

            clean_doc = list(filter(("<tag>").__ne__, doc_tag_split))
            question = data["question_text"]

            len_ques = len(question.split())
            part_len = INSTANCE_WORDS_LEN - len_ques
            if part_len <= 0:
                part_len = 16

            num_ins = (len(clean_doc) - part_len) // STRIDE + 1
            if num_ins < 0:
                num_ins = 0

            for part_id in range(num_ins + 1):
                part_start = part_id * STRIDE
                part_stop = min(len(clean_doc), part_id * STRIDE + part_len)
                part_split = clean_doc[part_start:part_stop]
                part = " ".join(part_split)

                instance = {
                    "example_id": example_id,
                    "part_start": part_start,
                    "part_stop": part_stop,
                    "question": question,
                    "context": part,
                    "start": 0,
                    "stop": 0,
                    "target": "NO_ANSWER",
                }
                list_instances.append(instance)

    return list_instances




## === cell 5
def getMapping(set_id, filename=f_test):
    list_cand_maps = []
    with open(filename) as f:
        progress = tqdm(f)
        for _, line in enumerate(progress):
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




## === cell 6
if TRANSFORMERS_OK:
    import tensorflow as tf

    def preprocess_data(data, tokenizer, debug=False):
        progress = tqdm(data, total=len(data))
        x1, x2, x3, y = [], [], [], []
        for sam in progress:
            tokenized_sam = tokenizer.encode_plus(
                sam["question"],
                sam["context"],
                padding="max_length",
                truncation=True,
                max_length=512,
                add_special_tokens=True,
            )
            x1.append(tf.cast(tokenized_sam["input_ids"], tf.int32))
            x2.append(tf.cast(tokenized_sam["token_type_ids"], tf.int32))
            x3.append(tf.cast(tokenized_sam["attention_mask"], tf.int32))
            y.append([sam["start"], sam["stop"], AnswerType[sam["target"]]])

        return (
            tf.convert_to_tensor(x1),
            tf.convert_to_tensor(x2),
            tf.convert_to_tensor(x3),
            tf.convert_to_tensor(y),
        )

    def get_strategy():
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
            print("No TPU detected:", repr(e))
            strategy = tf.distribute.get_strategy()
        return strategy

    def mergeInstanceResult(test_res, list_test_ins):
        for i in range(len(list_test_ins)):
            ins_res = test_res[i]
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

    def mergeDocumentRes(ins_df, val_id_df, threshold=0.0001, stride=128, debug=False):
        list_doc_lan = []
        for idx, doc in val_id_df.iterrows():
            doc_id = doc["example_id"]
            ins_of_doc = ins_df.loc[ins_df["example_id"] == doc_id]

            start_ins = ins_of_doc.loc[ins_of_doc["start"] != 0]
            stop_ins = ins_of_doc.loc[ins_of_doc["stop"] != 0]
            all_non_zero = pd.concat([start_ins, stop_ins]).drop_duplicates()

            best_start, best_stop, best_target = -1, -1, 0
            best_score = threshold

            for _, ins in all_non_zero.iterrows():
                ins_start = int(ins["start"])
                ins_stop = int(ins["stop"])
                ins_target = int(ins["target"])
                part_start = int(ins["part_start"])

                real_start = ins_start + part_start
                real_stop = ins_stop + part_start

                s_start = float(ins["start_score"])
                s_stop = float(ins["stop_score"])
                cls_start = float(ins["start_CLS"])
                cls_stop = float(ins["stop_CLS"])

                if real_stop > real_start:
                    score = (s_start - cls_start) + (s_stop - cls_stop)
                    if score > best_score:
                        best_score = score
                        best_start = real_start
                        best_stop = real_stop
                        best_target = ins_target

            list_doc_lan.append(
                {
                    "example_id": doc_id,
                    "start": best_start,
                    "stop": best_stop,
                    "target": best_target,
                    "score": best_score,
                }
            )

            if debug and idx == 101:
                print(list_doc_lan[-1])

        return pd.DataFrame(list_doc_lan)

    def build_model(model_name, debug=False):
        encoder = TFBertModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    def getRawInstanceResults(list_test, verbose=True, debug=False):
        if verbose:
            print("Getting raw result for all the instances generated from test file")

        model_name_candidates = [
            "../input/tensorflow-question-answer-fine-data",
            "/kaggle/input/tensorflow2-question-answering/tensorflow-question-answer-fine-data",
            "/kaggle/input/tensorflow2-question-answering/tensorflow2-question-answering/tensorflow-question-answer-fine-data",
        ]
        weights_candidates = [
            "../input/model1/weights-02.h5",
            "/kaggle/input/tensorflow2-question-answering/model1/weights-02.h5",
            "/kaggle/input/tensorflow2-question-answering/tensorflow2-question-answering/model1/weights-02.h5",
        ]

        model_name = _resolve_first_existing(model_name_candidates)
        weights_path = _resolve_first_existing(weights_candidates)

        if (model_name is None) or (weights_path is None):
            raise FileNotFoundError(
                f"Missing model folder or weights. Tried model_name={model_name_candidates} weights={weights_candidates}"
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tags = ["``", "''", "--"]
        tokenizer.add_special_tokens({"additional_special_tokens": tags})

        x_test1, x_test2, x_test3, _ = preprocess_data(list_test, tokenizer)

        strategy = get_strategy()
        with strategy.scope():
            testModel = build_model(model_name)
            x = np.ones([1, 512], dtype=int)
            testModel.predict([x, x, x], verbose=0)
            testModel.load_weights(weights_path)
            optAdam = tf.keras.optimizers.Adam(learning_rate=0.00005)
            lossSCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metricSCA = tf.keras.metrics.SparseCategoricalAccuracy()
            testModel.compile(optimizer=optAdam, loss=lossSCE, metrics=[metricSCA])

        return testModel.predict([x_test1, x_test2, x_test3], verbose=1)

    def getSubmissionLan(doc_res_df, doc_cand_df, threshold=0.0001, debug=False):
        doc_res_df = doc_res_df.copy()
        doc_cand_df = doc_cand_df.copy()
        doc_res_df.example_id = doc_res_df.example_id.astype(str)
        doc_cand_df.example_id = doc_cand_df.example_id.astype(str)

        combine_df = pd.merge(doc_res_df, doc_cand_df, on="example_id", how="left")
        lines = []
        for _, doc in combine_df.iterrows():
            example_id = doc["example_id"]
            long_id = str(example_id) + "_long"

            an_start = int(doc["start"])
            an_stop = int(doc["stop"])
            an_target = int(doc["target"])

            lan_start, lan_stop = -1, -1

            if (
                an_start > 0
                and an_stop > 0
                and isinstance(doc.get("new_candidates", None), list)
                and isinstance(doc.get("old_candidates", None), list)
            ):
                candidates = doc["new_candidates"]
                an_range = range(an_start, an_stop + 1)

                best_inter = 0.5
                shortest = 10**18
                best_id = 0
                for cidx, cand in enumerate(candidates):
                    c_start = int(cand["start_token"])
                    c_stop = int(cand["end_token"])
                    c_range = range(c_start, c_stop + 1)
                    inter = len(set(an_range).intersection(c_range))
                    if float(inter) > best_inter or (
                        inter == best_inter and len(c_range) < shortest
                    ):
                        best_id = cidx
                        best_inter = float(inter)
                        shortest = len(c_range)

                real_candidates = doc["old_candidates"]
                if len(real_candidates) > 0:
                    best_id = min(best_id, len(real_candidates) - 1)
                    lan_start = int(real_candidates[best_id]["start_token"])
                    lan_stop = int(real_candidates[best_id]["end_token"])

            if lan_start > 0 and lan_stop > 0 and an_target != 0:
                long_string = f"{lan_start}:{lan_stop}"
            else:
                long_string = ""

            lines.append({"example_id": long_id, "PredictionString": long_string})

        return pd.DataFrame(lines).sort_values("example_id")

    def getSanCandidate(sub, filename=f_test, debug=False):
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
        set_id = set(list_doc_lan_res_df["example_id"].tolist())

        list_san_ins = []
        with open(filename) as f:
            progress = tqdm(f)
            for _, line in enumerate(progress):
                data = json.loads(line)
                example_id = str(data["example_id"])
                if example_id in set_id:
                    ans = list_doc_lan_res_df.loc[
                        list_doc_lan_res_df["example_id"] == example_id
                    ].iloc[0]
                    lan_start, lan_stop = int(ans["lan_start"]), int(ans["lan_stop"])

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

    def create_model_san(tokenizer_san, model_name_san, debug=False):
        encoder = TFBertModel.from_pretrained(model_name_san)
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

    def getSanRawRes(list_san_ins, verbose=1):
        print(
            "Getting raw result for short answer instance generated from found long answers"
        )

        model_name_san_candidates = [
            "../input/tensorflow-question-answer-fine-data",
            "/kaggle/input/tensorflow2-question-answering/tensorflow-question-answer-fine-data",
            "/kaggle/input/tensorflow2-question-answering/tensorflow2-question-answering/tensorflow-question-answer-fine-data",
        ]
        weights_candidates = [
            "../input/model1/weights-14.h5",
            "/kaggle/input/tensorflow2-question-answering/model1/weights-14.h5",
            "/kaggle/input/tensorflow2-question-answering/tensorflow2-question-answering/model1/weights-14.h5",
        ]

        model_name_san = _resolve_first_existing(model_name_san_candidates)
        weights_path = _resolve_first_existing(weights_candidates)

        if (model_name_san is None) or (weights_path is None):
            raise FileNotFoundError(
                f"Missing model folder or weights. Tried model_name={model_name_san_candidates} weights={weights_candidates}"
            )

        tokenizer_san = AutoTokenizer.from_pretrained(model_name_san)
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

        x_san1, x_san2, x_san3, _ = preprocess_data(list_san_ins, tokenizer_san)

        strategy_san = get_strategy()
        with strategy_san.scope():
            sanModel = create_model_san(tokenizer_san, model_name_san)
            x = np.ones([1, 512], dtype=int)
            sanModel.predict([x, x, x], verbose=0)
            sanModel.load_weights(weights_path)
            optAdam = tf.keras.optimizers.Adam(learning_rate=0.00005)
            lossSCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metricSCA = tf.keras.metrics.SparseCategoricalAccuracy()
            sanModel.compile(optimizer=optAdam, loss=lossSCE, metrics=[metricSCA])

        return sanModel.predict([x_san1, x_san2, x_san3], verbose=1)

    def getSanSubmission(doc_res_df, threshold=0.0001, debug=False):
        doc_res_df = doc_res_df.copy()
        doc_res_df.example_id = doc_res_df.example_id.astype(str)

        lines = []
        for _, doc in doc_res_df.iterrows():
            example_id = doc["example_id"]
            short_id = str(example_id) + "_short"

            an_start = int(doc["start"])
            an_stop = int(doc["stop"])
            an_target = int(doc["target"])

            if (
                an_start > 0
                and an_stop > 0
                and an_target != 4
                and an_stop - an_start < 30
            ):
                short_string = f"{an_start}:{an_stop}"
            else:
                short_string = ""

            if an_target == 1 or an_target == 2:
                short_string = AnswerTypeRev[an_target]

            lines.append({"example_id": short_id, "PredictionString": short_string})

        return pd.DataFrame(lines).sort_values("example_id")

    def refineLan(sub, list_mapping_df, debug=False):
        newsub = []
        sub = sub.copy()
        sub["example_id"] = sub["example_id"].astype(str)
        list_mapping_df = list_mapping_df.copy()
        list_mapping_df["example_id"] = list_mapping_df["example_id"].astype(str)

        for _, row in sub.iterrows():
            if "long" in str(row["example_id"]):
                example_id = str(row["example_id"]).replace("_long", "")

                longid = str(row["example_id"])
                longStr = str(row["PredictionString"])

                lan_start, lan_stop = -1, -1
                if str(row["PredictionString"]) != "":
                    tokens = str(row["PredictionString"]).split(":")
                    lan_start, lan_stop = int(tokens[0]), int(tokens[1])

                sanid = str(example_id) + "_short"
                san_rows = sub.loc[sub["example_id"] == sanid]
                sanStr = ""
                if len(san_rows) > 0:
                    sanStr = str(san_rows.iloc[0]["PredictionString"])

                if sanStr != "" and sanStr != "YES" and sanStr != "NO":
                    tokensans = sanStr.split(":")
                    san_start, san_stop = int(tokensans[0]), int(tokensans[1])

                    if (lan_start >= 0 and lan_stop >= 0) and (
                        san_start < lan_start or san_stop > lan_stop
                    ):
                        map_rows = list_mapping_df.loc[
                            list_mapping_df["example_id"] == example_id
                        ]
                        if len(map_rows) > 0:
                            cands = map_rows.iloc[0]["old_candidates"]
                            an_range = range(san_start, san_stop + 1)

                            best_inter = 0.5
                            shortest = 10**18
                            best_id = 0
                            for cidx, cand in enumerate(cands):
                                c_start = int(cand["start_token"])
                                c_stop = int(cand["end_token"])
                                c_range = range(c_start, c_stop + 1)
                                inter = len(set(an_range).intersection(c_range))
                                if float(inter) > best_inter or (
                                    inter == best_inter and len(c_range) < shortest
                                ):
                                    best_id = cidx
                                    best_inter = float(inter)
                                    shortest = len(c_range)

                            if len(cands) > 0:
                                best_id = min(best_id, len(cands) - 1)
                                lan_start = int(cands[best_id]["start_token"])
                                lan_stop = int(cands[best_id]["end_token"])
                                longStr = f"{lan_start}:{lan_stop}"

                newsub.append({"example_id": longid, "PredictionString": longStr})
                newsub.append({"example_id": sanid, "PredictionString": sanStr})

        return pd.DataFrame(newsub).sort_values("example_id")




## === cell 7
sample_sub = pd.read_csv(sample_sub_path)
sample_sub["example_id"] = sample_sub["example_id"].astype(str)


def _exists_any(paths):
    return any(os.path.exists(p) for p in paths)


model_dir_candidates = [
    "../input/tensorflow-question-answer-fine-data",
    "/kaggle/input/tensorflow2-question-answering/tensorflow-question-answer-fine-data",
    "/kaggle/input/tensorflow2-question-answering/tensorflow2-question-answering/tensorflow-question-answer-fine-data",
]
w02_candidates = [
    "../input/model1/weights-02.h5",
    "/kaggle/input/tensorflow2-question-answering/model1/weights-02.h5",
    "/kaggle/input/tensorflow2-question-answering/tensorflow2-question-answering/model1/weights-02.h5",
]
w14_candidates = [
    "../input/model1/weights-14.h5",
    "/kaggle/input/tensorflow2-question-answering/model1/weights-14.h5",
    "/kaggle/input/tensorflow2-question-answering/tensorflow2-question-answering/model1/weights-14.h5",
]

can_run_models = (
    TRANSFORMERS_OK
    and _exists_any(model_dir_candidates)
    and _exists_any(w02_candidates)
    and _exists_any(w14_candidates)
)

if not can_run_models:
    print(
        "Pretrained model/weights not found or transformers/tf unavailable; writing blank submission.csv."
    )
    submission = sample_sub.copy()
    submission["PredictionString"] = ""
    submission["PredictionString"] = submission["PredictionString"].astype(str)
    submission.to_csv(
        "./submission.csv", index=False, columns=["example_id", "PredictionString"]
    )
else:
    list_id_df = get_id_df(f_test)
    set_id = set(list_id_df["example_id"].values.tolist())

    lan_map = getMapping(set_id, f_test)
    list_mappings_df = pd.DataFrame(lan_map)

    list_all_ins = parseDataClean(f_test)
    all_ins_res = getRawInstanceResults(list_all_ins)

    list_fine_res_all_ins = mergeInstanceResult(all_ins_res, list_all_ins)
    fine_res_all_ins_df = pd.DataFrame(list_fine_res_all_ins)

    docAnsDf = mergeDocumentRes(fine_res_all_ins_df, list_id_df)
    subLan = getSubmissionLan(docAnsDf, list_mappings_df)

    list_san_ins = getSanCandidate(subLan, debug=False)
    sanRawRes = getSanRawRes(list_san_ins)

    list_fine_res_san_ins = mergeInstanceResult(sanRawRes, list_san_ins)
    fine_res_san_ins_df = pd.DataFrame(list_fine_res_san_ins)

    docSanAnsDf = mergeDocumentRes(fine_res_san_ins_df, list_id_df)
    subSan = getSanSubmission(docSanAnsDf, threshold=0.2)

    sub = pd.concat([subLan, subSan], ignore_index=True)
    sub_sorted = sub.sort_values("example_id")

    refineSub = refineLan(sub_sorted, list_mappings_df, debug=False)

    submission = sample_sub[["example_id"]].merge(
        refineSub, on="example_id", how="left"
    )
    submission["PredictionString"] = (
        submission["PredictionString"].fillna("").astype(str)
    )

    submission.to_csv(
        "./submission.csv", index=False, columns=["example_id", "PredictionString"]
    )

print("Wrote ./submission.csv with shape:", pd.read_csv("./submission.csv").shape)
print(pd.read_csv("./submission.csv").head())
